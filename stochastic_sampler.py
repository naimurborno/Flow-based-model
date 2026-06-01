"""
stochastic_sampler.py
---------------------
Stochastic Velocity Branching Sampler for SD3 Flow Matching.

Replaces the standard single-path ODE loop with a K-branch
explore-then-commit strategy at each denoising step.

Pipeline per step t:
  1. One forward pass  → vbase = vθ(xt, t)
  2. Branch K candidates → ut(k) = vbase + σt · ϵk
  3. Euler look-ahead   → xt+Δt(k) = xt + ut(k) · Δt
  4. R_entropy          → centroid distance (O(K), no extra forward pass)
  5. R_fidelity         → one batched forward pass at look-ahead coords
  6. Gibbs policy       → P(k) = softmax(α · R_total(k))
  7. Hard sample k*     → commit xt+Δt = xt+Δt(k*)

Usage:
    sampler = StochasticVelocitySampler(
        unet      = wrapper.transformer,
        scheduler = wrapper.scheduler,
        cfg       = cfg,
        device    = device,
    )
    result = sampler.run(latents, text_embeddings, pooled_embeddings)
    final_latents = result["latents"]
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional


class StochasticVelocitySampler:

    def __init__(
        self,
        unet,
        scheduler,
        cfg          : dict,
        device       : str  = "cuda",
        K            : int  = 5,       # number of velocity branches
        sigma_max    : float = 1.0,    # max noise scale at t=0
        lam          : float = 0.5,    # λ: weight of entropy reward
        alpha        : float = 1.0,    # α: Gibbs temperature
    ):
        self.unet      = unet
        self.scheduler = scheduler
        self.cfg       = cfg
        self.device    = device
        self.K         = K
        self.sigma_max = sigma_max
        self.lam       = lam
        self.alpha     = alpha

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      50)
        self.guidance_scale = f_cfg.get("guidance_scale", 7.5)
        self.do_cfg         = self.guidance_scale > 1.0

        self.scheduler.set_timesteps(self.num_steps)
        self.timesteps = self.scheduler.timesteps   # descending: 1000 → 0

    # ================================================================== #
    #  MAIN ENTRY                                                         #
    # ================================================================== #

    def run(
        self,
        latents           : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        self.scheduler.set_timesteps(self.num_steps)
        self.timesteps = self.scheduler.timesteps

        trajectory   = []
        chosen_log   = []   # track which k* was chosen each step

        for i, t in enumerate(self.timesteps):

            t_val = t.item() if hasattr(t, "item") else float(t)

            # ── Step 2: baseline velocity (one forward pass) ──────────── #
            t_batch = t.reshape(1).expand(
                (2 if self.do_cfg else 1)
            ).to(self.device)

            latent_input = (
                torch.cat([latents] * 2) if self.do_cfg else latents
            )

            with torch.no_grad():
                raw_output = self._forward(
                    latent_input, t_batch, text_embeddings, pooled_embeddings
                )

            vbase = self._apply_cfg(raw_output) if self.do_cfg else raw_output
            # vbase shape: [1, C, H, W]

            # ── Step 3: branch K velocity candidates ─────────────────── #
            # σt high at t=0 (start of generation), zero at t=1000 (clean)
            sigma_t = self.sigma_max * (1.0 - t_val / 1000.0)

            noise   = torch.randn(
                self.K, *vbase.shape[1:],
                device=self.device, dtype=vbase.dtype
            )                                              # [K, C, H, W]
            vbase_k = vbase.expand(self.K, -1, -1, -1)   # [K, C, H, W]
            ut_k    = vbase_k + sigma_t * noise           # [K, C, H, W]

            # ── Step 4: Euler look-ahead ──────────────────────────────── #
            if i < len(self.timesteps) - 1:
                delta_t = (self.timesteps[i + 1] - t).item()
            else:
                delta_t = -t_val   # last step: go to 0

            xt_k = latents.expand(self.K, -1, -1, -1) + ut_k * delta_t
            # xt_k shape: [K, C, H, W]

            # ── Step 5: R_entropy via centroid trick (O(K)) ───────────── #
            centroid_x  = xt_k.mean(dim=0, keepdim=True)          # [1,C,H,W]
            dist_x      = ((xt_k - centroid_x) ** 2).sum(
                dim=(1, 2, 3)
            )                                                       # [K]
            r_entropy   = self.K * dist_x                          # [K]

            # ── Step 6: R_fidelity (one batched forward pass) ─────────── #
            if i < len(self.timesteps) - 1:
                t_next     = self.timesteps[i + 1]
                t_next_val = t_next.item()
            else:
                t_next_val = 0.0
                t_next     = torch.tensor(0.0, device=self.device)

            r_fidelity = self._compute_fidelity(
                xt_k, ut_k, t_next, t_next_val, text_embeddings, pooled_embeddings
            )                                                       # [K]

            # ── Step 7: total reward + Gibbs policy ───────────────────── #
            r_total = r_fidelity + self.lam * r_entropy            # [K]

            # normalize before softmax to prevent collapse
            r_total = (r_total - r_total.mean()) / (r_total.std() + 1e-8)

            logits  = self.alpha * r_total
            probs   = torch.softmax(logits, dim=0)                 # [K]

            # ── Step 8: hard sample one winner ───────────────────────── #
            k_star  = torch.multinomial(probs, num_samples=1).item()
            latents = xt_k[k_star].unsqueeze(0)                    # [1,C,H,W]

            chosen_log.append({"step": i, "t": t_val, "k_star": k_star,
                                "prob": probs[k_star].item()})

            if self.cfg.get("save_trajectory", False):
                trajectory.append(latents.clone().cpu())

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  [Stochastic] step {i+1:>3}/{self.num_steps} | "
                    f"t={t_val:.1f} | σ={sigma_t:.4f} | "
                    f"k*={k_star} (p={probs[k_star].item():.3f}) | "
                    f"latent_mean={latents.mean():.4f}"
                )

        return {
            "latents"    : latents,
            "trajectory" : trajectory,
            "chosen_log" : chosen_log,
        }

    # ================================================================== #
    #  FIDELITY: batched forward at look-ahead coords                     #
    # ================================================================== #

    def _compute_fidelity(
        self,
        xt_k              : torch.Tensor,   # [K, C, H, W]
        ut_k              : torch.Tensor,   # [K, C, H, W]
        t_next            : torch.Tensor,
        t_next_val        : float,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Run one batched forward pass over all K look-ahead coordinates.
        R_fidelity(k) = -‖vmanifold(k) - ut(k)‖²
        """
        K = self.K

        # expand text embeddings to match K (and 2K if CFG)
        if self.do_cfg:
            # text_embeddings is [2, seq, D] (uncond + cond)
            # replicate for K candidates → [2K, seq, D]
            te_k = text_embeddings.repeat_interleave(K, dim=0)   # [2K, seq, D]
            pe_k = (pooled_embeddings.repeat_interleave(K, dim=0)
                    if pooled_embeddings is not None else None)
            latent_input = torch.cat([xt_k, xt_k], dim=0)        # [2K, C, H, W]
        else:
            te_k         = text_embeddings.expand(K, -1, -1)
            pe_k         = (pooled_embeddings.expand(K, -1)
                            if pooled_embeddings is not None else None)
            latent_input = xt_k                                   # [K, C, H, W]

        t_batch = t_next.reshape(1).expand(latent_input.shape[0]).to(self.device)

        with torch.no_grad():
            raw = self._forward(latent_input, t_batch, te_k, pe_k)

        if self.do_cfg:
            # raw shape [2K, C, H, W]; split uncond/cond for each candidate
            v_uncond, v_cond = raw[:K], raw[K:]
            v_manifold = v_uncond + self.guidance_scale * (v_cond - v_uncond)
        else:
            v_manifold = raw   # [K, C, H, W]

        # per-candidate L2 mismatch → scalar reward
        diff       = (v_manifold - ut_k) ** 2              # [K, C, H, W]
        r_fidelity = -diff.sum(dim=(1, 2, 3))              # [K]
        return r_fidelity

    # ================================================================== #
    #  FORWARD PASS                                                       #
    # ================================================================== #

    def _forward(
        self,
        latent_input      : torch.Tensor,
        t_batch           : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype

        latent_input    = latent_input.to(device=device, dtype=dtype)
        text_embeddings = text_embeddings.to(device=device, dtype=dtype)
        t_batch         = t_batch.to(device=device, dtype=dtype)

        kwargs = dict(
            hidden_states         = latent_input,
            timestep              = t_batch,
            encoder_hidden_states = text_embeddings,
        )
        if pooled_embeddings is not None:
            kwargs["pooled_projections"] = pooled_embeddings.to(
                device=device, dtype=dtype
            )

        return self.unet(**kwargs).sample

    # ================================================================== #
    #  CFG                                                                #
    # ================================================================== #

    def _apply_cfg(self, velocity: torch.Tensor) -> torch.Tensor:
        v_uncond, v_cond = velocity.chunk(2)
        return v_uncond + self.guidance_scale * (v_cond - v_uncond)