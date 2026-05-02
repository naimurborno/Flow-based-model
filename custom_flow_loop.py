"""
custom_flow_loop.py
-------------------
Flow Matching ODE integration loop (SD3 / FLUX style).

Key conceptual shift from DDPM → Flow Matching:
  ┌─────────────────────────────────────────────────────────┐
  │  DDPM:  network predicts NOISE  ε                       │
  │         x_{t-1} = f(x_t, ε_θ, t)   [stochastic]        │
  │                                                         │
  │  FLOW:  network predicts VELOCITY  v                    │
  │         dx/dt = v_θ(x_t, t)        [deterministic ODE] │
  │         x(t) = (1-t)·x_data + t·x_noise                │
  │         v*   = x_data − x_noise    [straight path]      │
  └─────────────────────────────────────────────────────────┘

Solvers available:
  - Euler   (1st-order): x_{t+dt} = x_t + dt * v_θ(x_t, t)
  - Heun    (2nd-order): predictor-corrector, same NFE×2 but higher quality
"""

import torch
from typing import Dict, Any


class FlowMatchingLoop:

    def __init__(
        self,
        unet,                       # velocity-predicting transformer (DiT / MMDiT)
        scheduler,                  # FlowMatchEulerDiscreteScheduler
        cfg    : dict,
        device : str = "cuda",
    ):
        self.unet      = unet
        self.scheduler = scheduler
        self.cfg       = cfg
        self.device    = device

        f_cfg                = cfg.get("flow", {})
        self.num_steps       = f_cfg.get("num_steps",      50)
        self.guidance_scale  = f_cfg.get("guidance_scale", 7.5)
        self.solver          = f_cfg.get("solver",         "euler")   # "euler" | "heun"
        self.do_cfg          = self.guidance_scale > 1.0

        # Build continuous timesteps: t ∈ [1.0 → 0.0]
        # t=1 → pure noise,  t=0 → clean data
        self.scheduler.set_timesteps(self.num_steps)
        self.timesteps = self.scheduler.timesteps   # shape [num_steps]

    # ================================================================== #
    #  MAIN ENTRY                                                         #
    # ================================================================== #

    def run(
        self,
        latents         : torch.Tensor,   # [1, C, H, W]  — pure noise x(t=1)
        text_embeddings : torch.Tensor,   # [2, 77, D]    — [uncond; cond]
        pooled_embeddings : torch.Tensor = None,  # SD3/FLUX pooled text (optional)
    ) -> Dict[str, Any]:

        trajectory = []

        for i, t in enumerate(self.timesteps):

            # ---------------------------------------------------------- #
            # STEP 1 — prepare inputs for CFG                            #
            # Duplicate latents along batch for uncond + cond forward    #
            # ---------------------------------------------------------- #
            latent_input = (
                torch.cat([latents] * 2) if self.do_cfg else latents
            )

            # Flow schedulers do NOT scale the input (unlike DDPM's
            # scheduler.scale_model_input). Latents enter the network as-is.

            # ---------------------------------------------------------- #
            # STEP 2 — network forward: predict VELOCITY v_θ(x_t, t)    #
            # The network output is now a direction (not noise).          #
            # v* = x_data − x_noise  for straight-line flow paths        #
            # ---------------------------------------------------------- #
            # t_batch = t.expand(latent_input.shape[0])   # broadcast scalar t
            t_batch = t.reshape(1).expand(latent_input.shape[0]).to(self.device)
            with torch.no_grad():
                model_output = self._velocity_forward(
                    latent_input, t_batch, text_embeddings, pooled_embeddings
                )

            # ---------------------------------------------------------- #
            # STEP 3 — Classifier-Free Guidance on velocity field        #
            # Same formula, but applied to v instead of ε               #
            # ---------------------------------------------------------- #
            if self.do_cfg:
                model_output = self._apply_cfg(model_output)

            # ---------------------------------------------------------- #
            # STEP 4 — ODE step: integrate velocity to get x_{t+dt}     #
            #                                                             #
            # Euler:  x_{t+dt} = x_t  +  dt * v_θ                       #
            # Heun:   predictor-corrector (see _heun_step)               #
            # ---------------------------------------------------------- #
            if self.solver == "heun" and i < len(self.timesteps) - 1:
                latents = self._heun_step(
                    latents, model_output, t, self.timesteps[i + 1],
                    text_embeddings, pooled_embeddings
                )
            else:
                # Delegate to scheduler for Euler (handles dt internally)
                latents = self.scheduler.step(
                    model_output, t, latents
                ).prev_sample

            # ---------------------------------------------------------- #
            # STEP 5 — (Optional) mid-loop interventions                 #
            # Same hook pattern as the DDPM version                      #
            # ---------------------------------------------------------- #
            latents = self._step_callback(latents, t, i, model_output)

            if self.cfg.get("save_trajectory", False):
                trajectory.append(latents.clone().cpu())

            if (i + 1) % 10 == 0 or i == 0:
                t_val = t.item() if hasattr(t, "item") else float(t)
                print(f"  [Flow ODE] step {i+1:>3}/{self.num_steps} | "
                      f"t={t_val:.3f} | "
                      f"latent_mean={latents.mean():.4f} | "
                      f"latent_std={latents.std():.4f}")

        return {
            "latents"    : latents,
            "trajectory" : trajectory,
        }

    # ================================================================== #
    #  VELOCITY FORWARD PASS                                              #
    # ================================================================== #

    def _velocity_forward(
    self,
    latent_input      : torch.Tensor,
    t_batch           : torch.Tensor,
    text_embeddings   : torch.Tensor,
    pooled_embeddings : torch.Tensor = None,
    ) -> torch.Tensor:

        # Normalise timestep: SD3 scheduler gives t in [0,1000] → need [0,1]
        t_norm = t_batch.float()        # [0, 1000] → [0.0, 1.0]

        # Move everything to the same device + dtype as the model
        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype

        latent_input    = latent_input.to(device=device, dtype=dtype)
        text_embeddings = text_embeddings.to(device=device, dtype=dtype)
        t_norm          = t_norm.to(device=device, dtype=dtype)

        kwargs = dict(
            hidden_states         = latent_input,
            timestep              = t_norm,
            encoder_hidden_states = text_embeddings,
        )

        if pooled_embeddings is not None:
            kwargs["pooled_projections"] = pooled_embeddings.to(device=device, dtype=dtype)

        return self.unet(**kwargs).sample
    # ================================================================== #
    #  CFG ON VELOCITY                                                    #
    # ================================================================== #

    def _apply_cfg(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        CFG applied to the predicted velocity field.

        Standard:   v_guided = v_uncond + w * (v_cond − v_uncond)

        Experiments to try:
          - Time-varying guidance: w(t) = w * t   (stronger early, weaker late)
          - Rescaled CFG (Lin et al. 2023) — prevent over-saturation
          - Perp-neg guidance (Kirchhoff et al.)
        """
        v_uncond, v_cond = velocity.chunk(2)

        # -- Standard CFG --------------------------------------------- #
        guided = v_uncond + self.guidance_scale * (v_cond - v_uncond)

        # -- Time-varying guidance (uncomment to try) ----------------- #
        # t_frac = ...   # pass t into this method if needed
        # guided = v_uncond + self.guidance_scale * t_frac * (v_cond - v_uncond)

        # -- Rescaled CFG (Lin et al. 2023) --------------------------- #
        # phi = 0.7
        # std_pos = v_cond.std()
        # std_cfg = guided.std()
        # guided  = phi * (std_pos / std_cfg) * guided + (1 - phi) * guided

        return guided

    # ================================================================== #
    #  HEUN SOLVER (2nd-order predictor-corrector)                        #
    # ================================================================== #

    def _heun_step(
        self,
        x_t               : torch.Tensor,
        v_t               : torch.Tensor,   # velocity at t (already CFG-guided)
        t                 : torch.Tensor,
        t_next            : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Heun's method (trapezoidal rule):
          x_pred  = x_t + dt * v(x_t, t)              ← Euler predictor
          v_next  = network(x_pred, t_next)            ← evaluate at endpoint
          x_{t+1} = x_t + dt/2 * (v(x_t,t) + v_next) ← corrector

        Doubles NFE vs Euler but gives O(dt²) accuracy (vs O(dt) for Euler).
        """
        dt = t_next - t   # negative (t decreases toward 0)

        # -- Predictor (Euler) ---------------------------------------- #
        x_pred = x_t + dt * v_t

        # -- Evaluate velocity at predicted point --------------------- #
        t_next_batch = t_next.expand(x_pred.shape[0])
        latent_input = torch.cat([x_pred] * 2) if self.do_cfg else x_pred

        with torch.no_grad():
            v_next = self._velocity_forward(
                latent_input, t_next_batch, text_embeddings, pooled_embeddings
            )
        if self.do_cfg:
            v_next = self._apply_cfg(v_next)

        # -- Corrector (trapezoidal average) -------------------------- #
        x_next = x_t + (dt / 2) * (v_t + v_next)
        return x_next

    # ================================================================== #
    #  STEP CALLBACK                                                      #
    # ================================================================== #

    def _step_callback(
        self,
        latents    : torch.Tensor,
        t          : torch.Tensor,
        step_idx   : int,
        velocity   : torch.Tensor,
    ) -> torch.Tensor:
        """
        Hook called after every ODE step. Return (possibly modified) latents.

        Example flow-specific interventions:
          - At t > 0.5 (early, noisy): inject style / structure guidance
          - At t < 0.3 (late, sharp):  sharpen / denoise further
          - Log ||v_θ||  to monitor training signal quality
          - SDEdit-style blending: mix with reference latents at t_start
        """

        # ---- Example: log velocity norm ----------------------------- #
        # print(f"  [callback] step={step_idx} | ||v||={velocity.norm():.3f}")

        # ---- Example: early-step noise injection -------------------- #
        # t_val = t.item() if hasattr(t, "item") else float(t)
        # if t_val > 0.8:
        #     latents = latents + 0.005 * torch.randn_like(latents)

        return latents   # return unchanged by default