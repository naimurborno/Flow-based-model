"""
stochastic_sampler.py
---------------------
Stochastic Velocity Branching Sampler for SD3 Flow Matching.

Two-phase sampling strategy:
──────────────────────────────────────────────────────────────────────────────
Phase 1  (steps 0 … branch_steps-1)  — "Rotational Diversity Branching"
  At each step the current latent is rotated into K angularly-spaced candidates
  before the random perturbation is applied.

  Per-step pipeline:
    1a. Generate K rotated latents via Givens rotation at angles
            θk = 1° + k * (360°/K),  k = 0…K-1
        (offset by 1° so candidate-0 is never identical to the input)
    1b. FFT-HF injection: for every rotated candidate replace its high-frequency
        components with those of z_orig (the very first latent, kept frozen).
        Low frequencies stay from the rotated version → structural/color diversity.
        High frequencies from z_orig → shared fine-texture prior.
    2.  One baseline velocity forward pass → vbase = vθ(xt, t)
    3.  Branch K candidates: ut(k) = vbase + σt · ϵk  (random perturbation on top)
    4.  Euler look-ahead:    xt+Δt(k) = rotated_latent(k) + ut(k) · Δt
    5.  R_entropy  → centroid distance (O(K), no extra forward pass)
    6.  R_fidelity → one batched forward pass at look-ahead coords
    7.  Gibbs policy → P(k) = softmax(α · R_total(k))
    8.  Hard sample k* → commit xt+Δt = xt+Δt(k*)

Phase 2  (steps branch_steps … end)  — standard stochastic branching
  Identical to original pipeline: random perturbation only, no rotation.

Hyperparameters (all exposed in config.yaml under stochastic_sampler):
    K            — number of branches / rotated candidates
    sigma_max    — peak noise magnitude (decays with sigma_curr)
    lam          — λ: entropy reward weight
    alpha        — α: Gibbs temperature
    branch_steps — how many leading steps use Phase-1 rotational branching
    lf_cutoff    — low-frequency radius fraction for FFT HF-swap  (default 0.1)
──────────────────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List


# ══════════════════════════════════════════════════════════════════════════════ #
#  HybridManifoldSampler                                                         #
#  ──────────────────────────────────────────────────────────────────────────── #
#  Dual-Stream Spatial Tokenization + Hybrid Manifold Perturbation              #
#                                                                                #
#  At t=0 only:                                                                  #
#    1. Run cond + uncond passes and grab hidden states from a target layer.     #
#    2. Patchify → concatenate → SVD → merged basis V, singular values S.        #
#    3. Sample pink noise, project onto V, scale by S, unproject → d.            #
#    4. Inject: h_cond ← h_cond + α * d  via a one-shot residual hook.          #
#                                                                                #
#  All subsequent steps run unmodified (hook is removed after step 0).           #
# ══════════════════════════════════════════════════════════════════════════════ #

class HybridManifoldSampler:
    """
    Implements the dual-stream spatial tokenization and hybrid manifold
    perturbation described in the design doc.

    Args:
        transformer  : the SD3 MMDiT transformer
        target_layer : which transformer block index to hook (default 12)
        patch_size   : P — spatial patch side length in token space (default 2)
        K_svd        : number of SVD components to keep (default 64)
        alpha        : injection scale α (default 0.05)
        pink_beta    : 1/f^beta noise exponent (default 1.0 = pink noise)
        device       : "cuda" | "cpu"
    """

    def __init__(
        self,
        transformer,
        target_layer : int   = 12,
        patch_size   : int   = 2,
        K_svd        : int   = 64,
        alpha        : float = 0.05,
        pink_beta    : float = 1.0,
        device       : str   = "cuda",
    ):
        self.transformer  = transformer
        self.target_layer = target_layer
        self.P            = patch_size
        self.K_svd        = K_svd
        self.alpha        = alpha
        self.pink_beta    = pink_beta
        self.device       = device

        self._hook_handle       = None
        self._captured_states   : List[torch.Tensor] = []   # filled by hook
        self._inject_delta      : Optional[torch.Tensor] = None  # [1, N, D]

    # ──────────────────────────────────────────────────────────────────── #
    #  PUBLIC: build perturbation at step 0, return inject fn for step 1+  #
    # ──────────────────────────────────────────────────────────────────── #

    def build_perturbation(
        self,
        h_cond   : torch.Tensor,   # [1, N, D]  — cond hidden states at target layer
        h_uncond : torch.Tensor,   # [1, N, D]  — uncond hidden states at target layer
    ) -> torch.Tensor:
        """
        Given cond and uncond hidden states captured at t=0, compute the
        hybrid manifold perturbation vector d ∈ R^{1×N×D}.

        Steps:
          1. Patchify both streams → [M, P²D]
          2. Concatenate → H_merged [2M, P²D]
          3. Center + truncated SVD → V [P²D, K], S [K]
          4. Sample pink noise ε_patches [M, P²D]
          5. c = ε V  → c_scaled = c ⊙ S  → d_patches = c_scaled Vᵀ
          6. Unpatchify → d [1, N, D]
        """
        N, D = h_cond.shape[1], h_cond.shape[2]
        P    = self.P

        # ── 1. Patchify ──────────────────────────────────────────────── #
        h_cond_p   = self._patchify(h_cond.float(),   P)   # [M, P²D]
        h_uncond_p = self._patchify(h_uncond.float(), P)   # [M, P²D]
        M          = h_cond_p.shape[0]

        # ── 2. Merge ─────────────────────────────────────────────────── #
        H_merged = torch.cat([h_cond_p, h_uncond_p], dim=0)   # [2M, P²D]

        # ── 3. Center + SVD ──────────────────────────────────────────── #
        mu        = H_merged.mean(dim=0, keepdim=True)          # [1, P²D]
        H_c       = H_merged - mu                               # [2M, P²D]

        K_svd = min(self.K_svd, H_c.shape[0] - 1, H_c.shape[1] - 1)

        # Sanitize before decomposition
        H_c = torch.nan_to_num(H_c, nan=0.0, posinf=0.0, neginf=0.0)
        std = H_c.std()
        if std < 1e-6:
            # Degenerate matrix — return zero perturbation
            patch_dim = h_cond_p.shape[1]
            M2        = h_cond_p.shape[0]
            d_patches = torch.zeros(M2, patch_dim, device=self.device)
            d = self._unpatchify(d_patches, P, N, D)
            return d.to(dtype=h_cond.dtype)
        H_c = H_c / std   # unit-scale before randomized SVD

        # Randomized SVD — O(N·K), stable on well-conditioned matrices.
        # torch.svd_lowrank is more numerically stable than pca_lowrank.
        U, S, V = torch.svd_lowrank(H_c, q=K_svd, niter=2)
        # V : [P*D, K], S : [K]  — re-scale S to match original magnitude
        S = S * std

        # ── 4. Pink noise ────────────────────────────────────────────── #
        patch_dim   = h_cond_p.shape[1]               # = P * D  (1-D windows along seq)
        eps_patches = self._pink_noise(M, patch_dim)  # [M, P*D]

        # ── 5. Project → scale → unproject ──────────────────────────── #
        c           = eps_patches @ V                    # [M, K]
        # Normalise S to [0,1] range — raw singular values can be huge,
        # blowing up d and producing NaN after fp16 cast.
        S_norm      = S / (S.max().clamp(min=1e-6))
        c_scaled    = c * S_norm.unsqueeze(0)            # [M, K]
        d_patches   = c_scaled @ V.T                     # [M, P*D]

        # ── 6. Unpatchify ────────────────────────────────────────────── #
        d = self._unpatchify(d_patches, P, N, D)         # [1, N, D]

        # Safety: clamp to unit-norm per token to prevent fp16 overflow
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d_rms = d.norm() / (d.numel() ** 0.5 + 1e-6)
        if d_rms > 1.0:
            d = d / d_rms

        return d.to(dtype=h_cond.dtype)

    # ──────────────────────────────────────────────────────────────────── #
    #  HOOK REGISTRATION                                                    #
    # ──────────────────────────────────────────────────────────────────── #

    def register_capture_hook(self):
        """
        Register a forward hook on transformer_blocks[target_layer].
        Captures the block's output hidden states for both passes.
        Automatically deregistered after first activation.
        """
        self._captured_states = []
        block = self.transformer.transformer_blocks[self.target_layer]

        def _hook(module, input, output):
            # SD3 block output: tuple (hidden_states, ...) or just tensor
            hs = output[0] if isinstance(output, tuple) else output
            self._captured_states.append(hs.detach().clone())

        self._hook_handle = block.register_forward_hook(_hook)

    def deregister_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def register_inject_hook(self, d: torch.Tensor):
        """
        Register a one-shot hook that adds α*d to the cond stream output
        at target_layer during the NEXT forward pass, then self-removes.
        d : [1, N, D]
        """
        block   = self.transformer.transformer_blocks[self.target_layer]
        alpha   = self.alpha
        handle_box = [None]

        def _inject_hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            hs       = output[0] if is_tuple else output
            # In a CFG pass the cond slice is the second half
            B = hs.shape[0]
            if B > 1:
                # cond = second half
                hs = hs.clone()
                hs[B // 2 :] = hs[B // 2 :] + alpha * d.to(hs.dtype).to(hs.device)
            else:
                hs = hs + alpha * d.to(hs.dtype).to(hs.device)
            # self-remove
            handle_box[0].remove()
            return (hs,) + output[1:] if is_tuple else hs

        handle_box[0] = block.register_forward_hook(_inject_hook)

    # ──────────────────────────────────────────────────────────────────── #
    #  SPATIAL PATCH HELPERS                                                #
    # ──────────────────────────────────────────────────────────────────── #

    @staticmethod
    def _patchify(h: torch.Tensor, P: int) -> torch.Tensor:
        """
        h : [1, N, D]
        Reshape N tokens into M non-overlapping windows of size P (along seq dim).
        M = N // P  (tail tokens discarded if N % P != 0)
        Returns [M, P*D]
        """
        _, N, D = h.shape
        M       = N // P
        h_trim  = h[0, : M * P, :]     # [M*P, D]
        patches = h_trim.reshape(M, P * D)
        return patches

    @staticmethod
    def _unpatchify(patches: torch.Tensor, P: int, N: int, D: int) -> torch.Tensor:
        """
        patches : [M, P*D]  →  [1, N, D]
        Tail positions (if N % P != 0) are zero-padded.
        """
        M    = patches.shape[0]
        seq  = patches.reshape(M * P, D)    # [M*P, D]
        # Pad back to N if needed
        if M * P < N:
            pad  = torch.zeros(N - M * P, D, device=patches.device, dtype=patches.dtype)
            seq  = torch.cat([seq, pad], dim=0)
        return seq.unsqueeze(0)             # [1, N, D]

    # ──────────────────────────────────────────────────────────────────── #
    #  PINK NOISE                                                           #
    # ──────────────────────────────────────────────────────────────────── #

    def _pink_noise(self, M: int, dim: int) -> torch.Tensor:
        """
        Generate [M, dim] spatially correlated 1/f^beta noise.
        We treat each of the M patches as a 1-D signal of length dim,
        colour it in frequency domain, and return real-valued samples.
        """
        white = torch.randn(M, dim, device=self.device)
        freqs = torch.fft.rfftfreq(dim, device=self.device)
        freqs[0] = 1.0                              # avoid divide-by-zero at DC
        power    = freqs ** (-self.pink_beta / 2.0)
        spectrum = torch.fft.rfft(white) * power.unsqueeze(0)
        pink     = torch.fft.irfft(spectrum, n=dim)
        # unit-variance normalise per patch
        std  = pink.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return pink / std


class StochasticVelocitySampler:

    def __init__(
        self,
        unet,
        scheduler,
        cfg          : dict,
        device       : str   = "cuda",
        K            : int   = 5,      # number of velocity branches
        sigma_max    : float = 1.0,    # max noise scale at t=0
        lam          : float = 0.5,    # λ: weight of entropy reward
        alpha        : float = 1.0,    # α: Gibbs temperature
        branch_steps : int   = 10,     # steps that use Phase-1 rotational branching
        lf_cutoff    : float = 0.1,    # FFT low-frequency radius fraction
    ):
        self.unet         = unet
        self.scheduler    = scheduler
        self.cfg          = cfg
        self.device       = device
        self.K            = K
        self.sigma_max    = sigma_max
        self.lam          = lam
        self.alpha        = alpha
        self.branch_steps = branch_steps
        self.lf_cutoff    = lf_cutoff

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      50)
        self.guidance_scale = f_cfg.get("guidance_scale", 7.5)
        self.do_cfg         = self.guidance_scale > 1.0

        self.scheduler.set_timesteps(self.num_steps)
        self.timesteps = self.scheduler.timesteps   # descending: 1000 → 0

        # z_orig is set at the start of run() and kept frozen for HF injection
        self._z_orig: Optional[torch.Tensor] = None

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

        # ── Freeze z_orig for HF injection throughout Phase 1 ────────── #
        self._z_orig = latents.clone().detach()   # [1, C, H, W]

        trajectory = []
        chosen_log = []

        for i, t_discrete in enumerate(self.scheduler.timesteps):
            t_discrete_val = t_discrete.item()

            sigma_curr = self.scheduler.sigmas[i].item()
            sigma_next = self.scheduler.sigmas[i + 1].item()
            delta_t    = sigma_next - sigma_curr   # small negative fraction

            # ── Baseline velocity (single forward pass) ──────────────── #
            t_batch      = t_discrete.reshape(1).expand(
                2 if self.do_cfg else 1
            ).to(self.device)
            latent_input = torch.cat([latents] * 2) if self.do_cfg else latents

            with torch.no_grad():
                raw_output = self._forward(
                    latent_input, t_batch, text_embeddings, pooled_embeddings
                )
            vbase = self._apply_cfg(raw_output) if self.do_cfg else raw_output

            # ── Random perturbation (both phases) ────────────────────── #
            noise_scale = self.sigma_max * sigma_curr
            noise       = torch.randn(
                self.K, *vbase.shape[1:], device=self.device, dtype=vbase.dtype
            )
            vbase_k = vbase.expand(self.K, -1, -1, -1)
            ut_k    = vbase_k + noise_scale * noise   # [K, C, H, W]

            # ── Build K starting latents ──────────────────────────────── #
            if i < self.branch_steps:
                # Phase 1: rotated + HF-injected candidates
                base_k = self._make_rotated_candidates(latents)   # [K, C, H, W]
                phase  = "rot"
            else:
                # Phase 2: all K branches start from the same current latent
                base_k = latents.expand(self.K, -1, -1, -1)
                phase  = "std"

            # ── Euler look-ahead ─────────────────────────────────────── #
            xt_k = base_k + ut_k * delta_t   # [K, C, H, W]

            # ── R_entropy (centroid trick) ────────────────────────────── #
            centroid_x = xt_k.mean(dim=0, keepdim=True)
            dist_x     = ((xt_k.float() - centroid_x.float()) ** 2).sum(dim=(1, 2, 3))
            r_entropy  = self.K * dist_x

            # ── R_fidelity ───────────────────────────────────────────── #
            if i < len(self.scheduler.timesteps) - 1:
                t_next_discrete = self.scheduler.timesteps[i + 1]
            else:
                t_next_discrete = torch.tensor(0.0, device=self.device)

            r_fidelity = self._compute_fidelity(
                xt_k=xt_k,
                ut_k=ut_k,
                t_next=t_next_discrete,
                t_next_val=t_next_discrete.item(),
                text_embeddings=text_embeddings,
                pooled_embeddings=pooled_embeddings,
            )

            # ── Gibbs policy ─────────────────────────────────────────── #
            r_total = r_fidelity + self.lam * r_entropy
            r_std   = r_total.std()
            r_std   = r_std if r_std > 1e-4 else torch.tensor(1.0, device=self.device)
            r_total = (r_total - r_total.mean()) / r_std
            probs   = torch.softmax(self.alpha * r_total, dim=0)

            # ── Selection ────────────────────────────────────────────── #
            k_star  = torch.multinomial(probs, num_samples=1).item()
            latents = xt_k[k_star].unsqueeze(0)

            chosen_log.append({
                "step"  : i,
                "phase" : phase,
                "t"     : t_discrete_val,
                "k_star": k_star,
                "prob"  : probs[k_star].item(),
            })

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  [Stochastic/{phase}] step {i+1:>3}/{len(self.scheduler.timesteps)} | "
                    f"t_disc={t_discrete_val:.1f} | dt={delta_t:.4f} | σ={noise_scale:.4f} | "
                    f"k*={k_star} (p={probs[k_star].item():.3f}) | "
                    f"latent_mean={latents.mean():.4f}"
                )

        return {
            "latents"    : latents,
            "trajectory" : trajectory,
            "chosen_log" : chosen_log,
        }

    # ================================================================== #
    #  PHASE-1 HELPERS                                                    #
    # ================================================================== #

    def _make_rotated_candidates(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Return K rotated + HF-injected variants of `latents`.

        Angles:  θk = 1° + k * (360° / K),  k = 0 … K-1
          • k=0 → 1°  (not 0°, so it never equals the input exactly)
          • K=3 → [1°, 121°, 241°]
          • K=4 → [1°,  91°, 181°, 271°]

        Each rotated latent then gets its HF components replaced by those
        of z_orig (frozen initial latent), via FFT.
        """
        K       = self.K
        z_flat  = latents.reshape(-1).float()          # [D]
        D       = z_flat.shape[0]
        orig_shape = latents.shape                      # [1, C, H, W]

        candidates = []
        for k in range(1, K + 1):                          # k = 1 … K  (skip k=0 → 0°)
            deg   = k * (360.0 / (K + 1))                  # evenly spaced, 0° excluded
            rad   = math.radians(deg)

            z_rot = self._givens_rotate(z_flat, rad)        # [D]
            z_rot = z_rot.reshape(orig_shape).to(latents.dtype)

            z_hyb = self._fft_hf_inject(z_rot, self._z_orig)
            candidates.append(z_hyb)

        return torch.cat(candidates, dim=0)  

        return torch.cat(candidates, dim=0)   # [K, C, H, W]

    def _givens_rotate(self, z_flat: torch.Tensor, theta: float) -> torch.Tensor:
        """
        Apply a Givens rotation by `theta` radians to dimensions (0, 1) of z_flat.

        Givens rotation matrix G (acting on the 2D subspace spanned by e0, e1):
            [ cos θ  -sin θ ]
            [ sin θ   cos θ ]

        The rest of the vector is unchanged.  Result has identical ‖·‖₂.
        """
        c, s   = math.cos(theta), math.sin(theta)
        z_out  = z_flat.clone()
        x0, x1 = z_flat[0].item(), z_flat[1].item()
        z_out[0] = c * x0 - s * x1
        z_out[1] = s * x0 + c * x1
        return z_out

    def _fft_hf_inject(
        self,
        z_rot  : torch.Tensor,   # [1, C, H, W]  — rotated candidate (provides HF)
        z_orig : torch.Tensor,   # [1, C, H, W]  — frozen original   (provides LF)
    ) -> torch.Tensor:
        """
        FFT-based frequency swap:
            Z_out = LF(z_orig) + HF(z_rot)

        A circular mask of radius `lf_cutoff * max(H_f, W_f)` separates
        low from high frequencies (DC-centred in the rfft2 layout).

        Both inputs are temporarily cast to float32 for FFT precision;
        the result is cast back to the input dtype before returning.
        """
        H, W   = z_rot.shape[-2], z_rot.shape[-1]

        Z_rot  = torch.fft.rfft2(z_rot.float(),  norm="ortho")
        Z_orig = torch.fft.rfft2(z_orig.float(), norm="ortho")

        H_f, W_f = Z_rot.shape[-2], Z_rot.shape[-1]

        # Circular LF mask (DC at corner in rfft2 layout)
        yy      = torch.arange(H_f, device=self.device).float()
        xx      = torch.arange(W_f, device=self.device).float()
        dist    = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        radius  = max(H_f, W_f) * self.lf_cutoff
        lf_mask = dist <= radius                             # [H_f, W_f]  bool

        # Start from z_rot's spectrum (keeps all HF from rotated candidate)
        Z_out  = Z_rot.clone()
        # Overwrite LF region with z_orig's spectrum (injects original LF)
        Z_out[:, :, lf_mask] = Z_orig[:, :, lf_mask]

        z_out  = torch.fft.irfft2(Z_out, s=(H, W), norm="ortho")
        return z_out.to(z_rot.dtype)
    # ================================================================== #
    #  FIDELITY: batched forward at look-ahead coords                     #
    # ================================================================== #

    def _compute_fidelity(
        self,
        xt_k              : torch.Tensor,
        ut_k              : torch.Tensor,
        t_next            : torch.Tensor,
        t_next_val        : float,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        R_fidelity(k) = -‖v_manifold(k) - ut(k)‖²
        One batched forward pass over all K look-ahead coordinates.
        """
        K = self.K

        if self.do_cfg:
            uncond_te, cond_te = text_embeddings[0:1], text_embeddings[1:2]
            te_k = torch.cat([uncond_te.expand(K, -1, -1),
                               cond_te.expand(K, -1, -1)], dim=0)   # [2K, seq, D]
            if pooled_embeddings is not None:
                uncond_pe, cond_pe = pooled_embeddings[0:1], pooled_embeddings[1:2]
                pe_k = torch.cat([uncond_pe.expand(K, -1),
                                   cond_pe.expand(K, -1)], dim=0)   # [2K, D]
            else:
                pe_k = None
            latent_input = torch.cat([xt_k, xt_k], dim=0)           # [2K, C, H, W]
        else:
            te_k         = text_embeddings.expand(K, -1, -1)
            pe_k         = (pooled_embeddings.expand(K, -1)
                            if pooled_embeddings is not None else None)
            latent_input = xt_k                                      # [K, C, H, W]

        t_batch = t_next.reshape(1).expand(latent_input.shape[0]).to(self.device)

        with torch.no_grad():
            raw = self._forward(latent_input, t_batch, te_k, pe_k)

        if self.do_cfg:
            v_uncond, v_cond = raw[:K], raw[K:]
            v_manifold = v_uncond + self.guidance_scale * (v_cond - v_uncond)
        else:
            v_manifold = raw

        v_manifold = v_manifold.float()
        ut_k       = ut_k.float()
        diff       = (v_manifold - ut_k) ** 2
        r_fidelity = -diff.sum(dim=(1, 2, 3))
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