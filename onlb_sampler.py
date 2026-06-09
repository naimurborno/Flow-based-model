"""
onlb_sampler.py
---------------
Orthogonal Null-Space Langevin Backtracking (ONLB) Sampler for SD3 Flow Matching.

Key fix over v1:
  The forward pass now uses scheduler.step() exactly as the standard pipeline
  does, so the trajectory is correct.  The actual sigma-based dt is extracted
  from consecutive scheduler timesteps and used in the backward pass so the
  arithmetic stays consistent.

Algorithm:
  Phase 1 — Forward pass  : scheduler.step() ODE, cache (x_k, v_k, dt_k)
  Phase 2 — Backward pass : zero model calls, pure linear algebra on cache
  Phase 3 — Shell project : re-normalise x̃_0 onto √D Gaussian shell
  Phase 4 — Forward decode: scheduler.step() ODE from diverse seed x̃_0

Total model calls: 2N.
"""

import torch
from typing import Dict, Any, Optional


class ONLBSampler:

    def __init__(
        self,
        unet,
        scheduler,
        cfg          : dict,
        device       : str   = "cuda",
        lam          : float = 0.05,   # repulsion weight — keep small to start
        eta          : float = 0.01,   # Langevin temperature — keep small to start
        max_drift    : float = 3.0,    # hard clamp: max ‖x̃ − x_anchor‖ / √D per step
        eps          : float = 1e-8,
    ):
        self.unet      = unet
        self.scheduler = scheduler
        self.cfg       = cfg
        self.device    = device
        self.lam       = lam
        self.eta       = eta
        self.max_drift = max_drift
        self.eps       = eps

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      20)
        self.guidance_scale = f_cfg.get("guidance_scale", 7.5)
        self.do_cfg         = self.guidance_scale > 1.0

    # ================================================================== #
    #  PUBLIC ENTRY                                                       #
    # ================================================================== #

    def run(
        self,
        latents           : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        x0_original = latents.clone()

        # Phase 1
        print("\n[ONLB] ── Phase 1: Forward pass & trajectory caching ──")
        cached_x, cached_v, cached_dt, x_N = self._forward_pass(
            latents, text_embeddings, pooled_embeddings, tag="Cache"
        )

        # sanity: x_N should look like an image latent, not noise
        print(f"[ONLB]   x_N  mean={x_N.mean():.4f}  std={x_N.std():.4f}  "
              f"norm={x_N.norm():.4f}")

        # Phase 2
        print("\n[ONLB] ── Phase 2: Orthogonal Langevin backward pass ──")
        x0_tilde = self._backward_pass(cached_x, cached_v, cached_dt)

        # Phase 3
        print("\n[ONLB] ── Phase 3: Hypersphere projection ──")
        x0_diverse = self._project_to_shell(x0_tilde, x0_original)

        escape = (x0_diverse - x0_original).norm().item()
        print(f"[ONLB]   escape distance ‖x̃₀ − x₀‖₂ = {escape:.4f}")

        # Phase 4
        print("\n[ONLB] ── Phase 4: Forward decode from diverse seed ──")
        _, _, _, x_N_diverse = self._forward_pass(
            x0_diverse, text_embeddings, pooled_embeddings, tag="Decode"
        )

        return {
            "original_latents" : x_N,
            "diverse_latents"  : x_N_diverse,
            "x0_original"      : x0_original,
            "x0_diverse"       : x0_diverse,
            "escape_distance"  : escape,
        }

    # ================================================================== #
    #  PHASE 1 — FORWARD PASS                                            #
    # ================================================================== #

    def _forward_pass(
        self,
        x0                : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor],
        tag               : str = "Cache",
    ):
        """
        Use scheduler.step() — the same path as custom_flow_loop.py — so the
        trajectory is numerically identical to what the pipeline produces.

        Also record the actual dt at each step (sigma difference) so the
        backward pass uses the same scale.

        Returns:
          cached_x  [N+1]  — latent at each step boundary
          cached_v  [N]    — velocity at each step
          cached_dt [N]    — actual |dt| used at each step (sigma-based)
          x_N              — final image latent
        """
        # fresh scheduler state
        self.scheduler.set_timesteps(self.num_steps)
        timesteps = self.scheduler.timesteps   # e.g. [999, 966, ..., 0]
        N         = len(timesteps)

        cached_x  = [None] * (N + 1)
        cached_v  = [None] * N
        cached_dt = [None] * N

        x           = x0.clone()
        cached_x[0] = x.clone()

        for k, t in enumerate(timesteps):
            t_val = t.item() if hasattr(t, "item") else float(t)

            v = self._velocity_forward(x, t, text_embeddings, pooled_embeddings)
            cached_v[k] = v.detach().clone()

            # ── actual dt from scheduler sigma values ──────────────── #
            # sigma_t  is the noise level at step k
            # sigma_next is the noise level at step k+1  (or 0 at end)
            # dt = sigma_next - sigma_t  (negative: noise decreasing)
            # We store |dt| for backward pass scaling.
            if hasattr(self.scheduler, "sigmas"):
                sigma_t    = self.scheduler.sigmas[k].item()
                sigma_next = self.scheduler.sigmas[k + 1].item()
                dt         = abs(sigma_next - sigma_t)
            else:
                # fallback: uniform spacing
                dt = 1.0 / N
            cached_dt[k] = dt

            # ── step with scheduler (correct) ──────────────────────── #
            x = self.scheduler.step(v, t, x).prev_sample
            cached_x[k + 1] = x.detach().clone()

            if (k + 1) % 5 == 0 or k == 0:
                print(f"  [{tag}] step {k+1:>3}/{N} | t={t_val:>6.1f} | "
                      f"dt={dt:.4f} | mean={x.mean():.4f} | std={x.std():.4f}")

        return cached_x, cached_v, cached_dt, x

    # ================================================================== #
    #  PHASE 2 — ORTHOGONAL LANGEVIN BACKWARD PASS                       #
    # ================================================================== #

    def _backward_pass(
        self,
        cached_x  : list,
        cached_v  : list,
        cached_dt : list,
    ) -> torch.Tensor:
        """
        Walk backwards from x_N → x̃_0.
        All intermediate math is done in fp32 to prevent fp16 NaN/Inf.
        Result is cast back to the original dtype at the end.
        """
        N          = len(cached_v)
        orig_dtype = cached_x[N].dtype
        x_tilde    = cached_x[N].clone().float()   # fp32 from here on
        D          = x_tilde.numel()

        for k in range(N - 1, -1, -1):
            dt        = cached_dt[k]
            x_anchor  = cached_x[k].float()
            v_forward = cached_v[k].float()

            # ── raw displacement from anchor ────────────────────────── #
            delta      = x_tilde - x_anchor
            delta_norm = delta.norm() + self.eps
            max_norm   = self.max_drift * (D ** 0.5)
            if delta_norm > max_norm:
                delta = delta * (max_norm / delta_norm)

            u = self.lam * delta

            # ── Gram-Schmidt: project u onto null-space of v_forward ── #
            v_fwd_flat  = v_forward.reshape(-1)
            u_flat      = u.reshape(-1)
            v_fwd_sq    = v_fwd_flat.dot(v_fwd_flat) + self.eps
            proj_scalar = torch.dot(u_flat, v_fwd_flat) / v_fwd_sq
            v_ortho     = u - proj_scalar * v_forward

            # magnitude-match v_ortho to v_forward
            v_fwd_norm   = v_forward.norm() + self.eps
            v_ortho_norm = v_ortho.norm()   + self.eps
            v_ortho      = v_ortho * (v_fwd_norm / v_ortho_norm)

            # ── Langevin noise ──────────────────────────────────────── #
            xi          = torch.randn_like(x_tilde)          # fp32
            noise_scale = self.eta * v_fwd_norm.item()

            # ── backward update ─────────────────────────────────────── #
            x_tilde = (
                x_tilde
                - dt * v_forward
                + dt * v_ortho
                + noise_scale * xi
            )

            # NaN guard per step
            if not torch.isfinite(x_tilde).all():
                print(f"  [Backward] WARNING: NaN/Inf at step k={k}, "
                      f"reverting to anchor x_anchor")
                x_tilde = x_anchor.clone()

            if (N - k) % 5 == 0 or k == N - 1:
                print(f"  [Backward] step {N-k:>3}/{N} | k={k} | "
                      f"|delta|={delta.norm():.3f} | |v_ortho|={v_ortho.norm():.3f} | "
                      f"noise_s={noise_scale:.4f} | "
                      f"mean={x_tilde.mean():.4f} | std={x_tilde.std():.4f}")

        return x_tilde.to(orig_dtype)

    # ================================================================== #
    #  PHASE 3 — HYPERSPHERE PROJECTION                                  #
    # ================================================================== #

    def _project_to_shell(
        self,
        x          : torch.Tensor,
        x0_original: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project x onto the √D Gaussian typical shell.

        Also prints how far x̃_0 already is from x_0 before projection,
        and after — useful for debugging whether Phase 2 is doing anything.
        """
        D      = x.numel()
        target = D ** 0.5

        norm_before = x.norm().item()
        x_proj      = target * x / (x.norm() + self.eps)
        norm_after  = x_proj.norm().item()

        print(f"[ONLB]   D={D} | target shell radius = {target:.2f}")
        print(f"[ONLB]   ‖x̃_0‖  before={norm_before:.4f}  after={norm_after:.4f}")
        print(f"[ONLB]   ‖x̃_0 − x_0‖ before projection = "
              f"{(x - x0_original).norm().item():.4f}")

        return x_proj

    # ================================================================== #
    #  VELOCITY FORWARD                                                  #
    # ================================================================== #

    def _velocity_forward(
        self,
        x                 : torch.Tensor,
        t                 : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        v_θ(x, t) with CFG.  Mirrors _velocity_forward in custom_flow_loop.py.

        SD3 MMDiT CFG layout:
          text_embeddings  shape: [2, seq, D]  — [uncond, cond]
          latent_input     shape: [2, C, H, W] — [x_dup, x_dup]
          Both batch dims must match; MMDiT cross-attends position-wise.
        """
        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype

        if self.do_cfg:
            # text_embeddings is already [2, seq, D] from encode_prompt
            latent_input = torch.cat([x, x], dim=0).to(device=device, dtype=dtype)
        else:
            latent_input = x.to(device=device, dtype=dtype)

        enc_hs = text_embeddings.to(device=device, dtype=dtype)   # [B, seq, D]

        t_val   = t.item() if hasattr(t, "item") else float(t)
        t_batch = torch.tensor(
            [t_val] * latent_input.shape[0],
            device=device, dtype=dtype
        )

        kwargs = dict(
            hidden_states         = latent_input,
            timestep              = t_batch,
            encoder_hidden_states = enc_hs,
        )
        if pooled_embeddings is not None:
            kwargs["pooled_projections"] = pooled_embeddings.to(device=device, dtype=dtype)

        with torch.no_grad():
            output = self.unet(**kwargs).sample

        if self.do_cfg:
            v_uncond, v_cond = output.chunk(2)
            output = v_uncond + self.guidance_scale * (v_cond - v_uncond)

        return output


# ══════════════════════════════════════════════════════════════════════ #
#  DROP-IN RUNNER                                                         #
# ══════════════════════════════════════════════════════════════════════ #

def run_sd3_onlb(opts: dict):
    """
    Drop-in runner for inference.py MODEL_REGISTRY.
    Set model_name: "sd3_onlb" in config.yaml to activate.
    """
    from pathlib import Path
    from pipeline_wrapper import SD3PipelineWrapper

    cfg    = opts.get("_cfg", {})
    device = opts["device"]

    wrapper = SD3PipelineWrapper(cfg, device=device)
    wrapper.load()

    prompt_embeds, pooled_embeds = wrapper.encode_prompt(
        opts["prompt"], opts["negative_prompt"]
    )
    latents = wrapper.get_initial_latents(seed=opts["seed"])

    onlb_cfg      = cfg.get("onlb", {})
    lam           = onlb_cfg.get("lam",           0.05)
    eta           = onlb_cfg.get("eta",           0.01)
    max_drift     = onlb_cfg.get("max_drift",     3.0)
    save_original = onlb_cfg.get("save_original", True)

    print(f"\n[ONLB] λ={lam}  η={eta}  max_drift={max_drift}  steps={opts['num_steps']}")

    sampler = ONLBSampler(
        unet      = wrapper.transformer,
        scheduler = wrapper.scheduler,
        cfg       = cfg,
        device    = device,
        lam       = lam,
        eta       = eta,
        max_drift = max_drift,
    )

    result = sampler.run(latents, prompt_embeds, pooled_embeds)

    # save diverse
    out_path = Path(opts["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.decode_latents(result["diverse_latents"]).save(out_path)
    print(f"\n[ONLB] Diverse image  → {out_path}")

    # save original for side-by-side comparison
    if save_original:
        orig_path = out_path.with_stem(out_path.stem + "_original")
        wrapper.decode_latents(result["original_latents"]).save(orig_path)
        print(f"[ONLB] Original image → {orig_path}")

    print(f"[ONLB] Escape distance = {result['escape_distance']:.4f}")