"""
Md. Naimur Asif Borno
pipeline_wrapper.py  (Flow Matching edition)
--------------------------------------------
Loads SD3 / FLUX via Diffusers and exposes each component individually.

Architecture shift from SD 1.5 → SD3:
  ┌─────────────────────────┬────────────────────────────────────┐
  │       SD 1.5 (DDPM)     │       SD3 / FLUX (Flow Matching)   │
  ├─────────────────────────┼────────────────────────────────────┤
  │ UNet (conv-heavy)       │ MMDiT / DiT (pure transformer)     │
  │ CLIPTokenizer (77 tok)  │ CLIP-L + CLIP-G + T5-XXL (3 enc.) │
  │ DDIMScheduler           │ FlowMatchEulerDiscreteScheduler    │
  │ ε-prediction            │ v-prediction (velocity field)      │
  │ t ∈ {0..1000}  discrete │ t ∈ [0.0, 1.0]  continuous        │
  └─────────────────────────┴────────────────────────────────────┘

Design rule:
    Load once → patch freely → weights stay frozen unless you explicitly unfreeze.
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image


class SD3PipelineWrapper:

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = device

        # Components (populated in load())
        self.pipe              = None
        self.tokenizer         = None   # CLIP-L
        self.tokenizer_2       = None   # CLIP-G
        self.tokenizer_3       = None   # T5-XXL
        self.text_encoder      = None   # CLIP-L encoder
        self.text_encoder_2    = None   # CLIP-G encoder
        self.text_encoder_3    = None   # T5-XXL encoder
        self.transformer       = None   # MMDiT (replaces UNet)
        self.vae               = None
        self.scheduler         = None   # FlowMatchEulerDiscreteScheduler

    # ================================================================== #
    #  LOAD                                                               #
    # ================================================================== #

    def load(self):
        model_id = self.cfg.get("model_id", "stabilityai/stable-diffusion-3-medium-diffusers")

        print(f"[Pipeline] Loading: {model_id}")
        print(f"[Pipeline] Scheduler: FlowMatchEulerDiscrete (built-in for SD3)")

        # Load full SD3 pipeline
        # SD3 requires HF token if gated — set HF_TOKEN env var or pass token=
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype = torch.float16,
            text_encoder_3=None,
          tokenizer_3=None,
        ).to(self.device)
        self.pipe.enable_model_cpu_offload()

        # Extract components individually
        self.tokenizer      = self.pipe.tokenizer       # CLIP-L
        self.tokenizer_2    = self.pipe.tokenizer_2     # CLIP-G
        self.tokenizer_3    = None    # T5-XXL
        self.text_encoder   = self.pipe.text_encoder    # CLIP-L
        self.text_encoder_2 = self.pipe.text_encoder_2  # CLIP-G
        self.text_encoder_3 = None # T5-XXL
        self.transformer    = self.pipe.transformer     # MMDiT (velocity predictor)
        self.vae            = self.pipe.vae

        # Flow scheduler — continuous t ∈ [0, 1]
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Freeze everything — zero-shot
        self._freeze_all()

        print("[Pipeline] Loaded & frozen ✔")
        self._print_memory()

    # ================================================================== #
    #  PATCH  — modify architecture after loading weights                 #
    # ================================================================== #

    def patch(self):
        """
        Called after load(). Modify transformer / vae / text_encoders here.

        Flow-specific experiments to try:
          - Register attention hooks on MMDiT joint-attention blocks
          - Replace or extend positional embeddings
          - Inject LoRA / adapter layers into transformer blocks
          - Modify the time-step embedding (sinusoidal → learned)
        """

        # ---- Experiment A: Register attention hooks -----------------
        # self._register_attention_hooks()

        # ---- Experiment B: Inject adapters into DiT blocks ----------
        # self._inject_adapters()

        print("[Pipeline] patch() called — add your patches here")

    # ================================================================== #
    #  PROMPT ENCODING  (3 encoders: CLIP-L, CLIP-G, T5-XXL)            #
    # ================================================================== #

    def encode_prompt(
        self,
        prompt          : str,
        negative_prompt : str = "",
    ):
        """
        Returns:
            prompt_embeds       : [2, seq_len, D]   — [uncond; cond] for CFG
            pooled_prompt_embeds: [2, D_pooled]     — pooled CLIP output

        SD3 uses THREE text encoders:
          - CLIP-L  → token embeddings  (seq)
          - CLIP-G  → token embeddings  (seq) + pooled
          - T5-XXL  → token embeddings  (seq, longer context)
        All three are concatenated along the sequence dimension → MMDiT input.
        """
        do_cfg = self.cfg.get("flow", {}).get("guidance_scale", 7.5) > 1.0

        with torch.no_grad():
            cond_emb, cond_pooled = self._encode_prompt_single(prompt)

            if do_cfg:
                uncond_emb, uncond_pooled = self._encode_prompt_single(negative_prompt)
                # CFG convention: [uncond; cond]
                prompt_embeds        = torch.cat([uncond_emb,    cond_emb])
                pooled_prompt_embeds = torch.cat([uncond_pooled, cond_pooled])
            else:
                prompt_embeds        = cond_emb
                pooled_prompt_embeds = cond_pooled

        return prompt_embeds, pooled_prompt_embeds

    def _encode_prompt_single(self, text: str):
        max_len_clip = self.tokenizer.model_max_length    # 77

        def _tok(tokenizer, text, max_length):
            return tokenizer(
                text,
                padding        = "max_length",
                max_length     = max_length,
                truncation     = True,
                return_tensors = "pt",
            ).input_ids.to(self.device)

        # --- CLIP-L -------------------------------------------------------
        ids_l    = _tok(self.tokenizer, text, max_len_clip)
        out_l    = self.text_encoder(ids_l, output_hidden_states=True)
        emb_l    = out_l.hidden_states[-2]               # [1, 77, 768]
        pooled_l = out_l.hidden_states[-1][:, -1, :]     # [1, 768]

        # --- CLIP-G -------------------------------------------------------
        ids_g    = _tok(self.tokenizer_2, text, max_len_clip)
        out_g    = self.text_encoder_2(ids_g, output_hidden_states=True)
        emb_g    = out_g.hidden_states[-2]               # [1, 77, 1280]
        pooled_g = out_g.text_embeds                     # [1, 1280]

        # --- T5 is None — fill its slot with zeros -----------------------
        D_joint      = 4096
        emb_l_pad    = torch.nn.functional.pad(emb_l, (0, D_joint - emb_l.shape[-1]))
        emb_g_pad    = torch.nn.functional.pad(emb_g, (0, D_joint - emb_g.shape[-1]))
        emb_t5_zero  = torch.zeros(1, 256, D_joint, dtype=emb_l.dtype, device=self.device)

        emb_all = torch.cat([emb_l_pad, emb_g_pad, emb_t5_zero], dim=1)  # [1, 410, 4096]
        pooled  = torch.cat([pooled_l, pooled_g], dim=-1)                 # [1, 2048]

        return emb_all, pooled
    # ================================================================== #
    #  LATENT HELPERS                                                     #
    # ================================================================== #

    def get_initial_latents(self, seed: int = 42) -> torch.Tensor:
        """
        Pure noise in latent space. x(t=1) = N(0, I).

        SD3 latent shape: [1, 16, H/8, W/8]
        Note: SD3 VAE has 16 channels (vs SD1.5's 4).
        Flow matching starts at t=1 (pure noise) and integrates to t=0.
        """
        gen_cfg = self.cfg.get("generation", {})
        H       = gen_cfg.get("height", 512)
        W       = gen_cfg.get("width",  512)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents   = torch.randn(
            (1, self.transformer.config.in_channels, H // 8, W // 8),
            generator = generator,
            device    = self.device,
            dtype     = torch.float16,
        )
        # Flow matching: no init_noise_sigma scaling needed (unlike DDPM)
        # The scheduler handles time-dependent scaling internally
        return latents

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode latents → RGB PIL Image via VAE. Same as SD 1.5."""
        latents = latents.to(dtype=torch.float16)

        with torch.no_grad():
            # SD3 VAE uses a different scaling factor than SD 1.5
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor
            ).sample

        # [-1, 1] → [0, 255]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype("uint8")
        return Image.fromarray(image)

    # ================================================================== #
    #  INTERNALS                                                          #
    # ================================================================== #

    def _freeze_all(self):
        models = [
            self.text_encoder, self.text_encoder_2,
            self.text_encoder_3, self.transformer, self.vae
        ]
        for model in models:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = False

    def _print_memory(self):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            print(f"[Pipeline] GPU memory used: {alloc:.2f} GB")

    # ---- Patch helpers ---------------------------------------------- #

    def _register_attention_hooks(self):
        """Hook into MMDiT joint-attention blocks."""
        for name, module in self.transformer.named_modules():
            if hasattr(module, "to_q") and hasattr(module, "to_k"):
                module.register_forward_hook(
                    lambda m, inp, out, n=name: print(
                        f"[hook] {n} | out_shape={out[0].shape if isinstance(out, tuple) else out.shape}"
                    )
                )

    def _inject_adapters(self):
        """Insert lightweight adapter after each DiT transformer block."""
        # Implement LoRA or bottleneck adapters here
        pass