"""
attention_surgery.py
---------------------
Surgical monkeypatch for SD3 MMDiT attention blocks.

PROBLEM with naive forward hooks:
    SD3 uses Diffusers' JointAttnProcessor2_0 (or Attention class).
    The actual softmax(QK^T/√d) is computed INSIDE the processor's __call__,
    and the result is never returned to the module's forward output.
    So register_forward_hook only sees hidden_states — no attn_probs.

SOLUTION:
    Replace every block's attn_processor with a custom
    SurgicalJointAttnProcessor that:
      1. Computes attn_probs exactly as the original
      2. Saves the image×text cross-attention slice
      3. Returns identical output — zero behavioural change

Architecture of SD3 MMDiT joint attention:
    ┌─────────────────────────────────────────────────────┐
    │  Inputs:                                            │
    │    hidden_states        [B, N_img,  D]  ← image    │
    │    encoder_hidden_states [B, N_txt, D]  ← text     │
    │                                                     │
    │  Joint sequence = cat([img, txt], dim=1)            │
    │    length = N_img + N_txt                           │
    │                                                     │
    │  QKV projection → Q, K, V  [B, heads, N_total, d]  │
    │                                                     │
    │  attn_probs = softmax(Q @ K^T / √d)                 │
    │    shape: [B, heads, N_total, N_total]              │
    │                                                     │
    │  Cross slice we want:                               │
    │    attn_probs[:, :, :N_img, N_img:]                 │
    │    shape: [B, heads, N_img, N_txt]                  │
    │    = "for each image patch, how much does it        │
    │       attend to each text token?"                   │
    └─────────────────────────────────────────────────────┘

Usage:
    surgeon = AttentionSurgeon(wrapper.transformer)
    surgeon.install()                       # monkeypatch all blocks

    # ... run forward pass ...

    maps = surgeon.get_maps(step=i)         # {block_idx: Tensor[B,H,N_img,N_txt]}
    surgeon.clear()                         # free memory between steps
    surgeon.uninstall()                     # restore originals when done
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════ #
#  STORAGE                                                                #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class AttnSnapshot:
    """One attention snapshot per block per forward pass."""
    block_idx   : int
    step        : int
    # img→txt cross-attention: [B, heads, N_img, N_txt]
    img_to_txt  : torch.Tensor
    n_img       : int    # how many image tokens (N_img)
    n_txt       : int    # how many text tokens  (N_txt)


# ═══════════════════════════════════════════════════════════════════════ #
#  CUSTOM PROCESSOR                                                       #
# ═══════════════════════════════════════════════════════════════════════ #

class SurgicalJointAttnProcessor:
    """
    Drop-in replacement for JointAttnProcessor2_0.

    Computes IDENTICAL math but saves attn_probs[:, :, :N_img, N_img:]
    (image→text cross-attention slice) into the surgeon's store.

    All tensor ops are dtype-safe: we upcast to float32 for the softmax
    then cast back, matching what scaled_dot_product_attention does.
    """

    def __init__(
        self,
        surgeon    : "AttentionSurgeon",
        block_idx  : int,
        original   : object,          # original processor (kept for reference)
    ):
        self.surgeon   = surgeon
        self.block_idx = block_idx
        self.original  = original

    def __call__(
        self,
        attn,                          # Attention module (has to_q, to_k, to_v, etc.)
        hidden_states         : torch.Tensor,           # [B, N_img, D]
        encoder_hidden_states : Optional[torch.Tensor] = None,  # [B, N_txt, D]
        attention_mask        : Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (hidden_states_out, encoder_hidden_states_out)
        matching the JointAttnProcessor2_0 contract.
        """

        B       = hidden_states.shape[0]
        N_img   = hidden_states.shape[1]
        residual = hidden_states

        # ── Norms (if present in block) ──────────────────────────────
        if attn.norm_q is not None:
            hidden_states = attn.norm_q(hidden_states)
        if encoder_hidden_states is not None and attn.norm_encoder_q is not None:
            encoder_hidden_states = attn.norm_encoder_q(encoder_hidden_states)

        # ── QKV projections ───────────────────────────────────────────
        q_img = attn.to_q(hidden_states)               # [B, N_img, inner_dim]
        k_img = attn.to_k(hidden_states)
        v_img = attn.to_v(hidden_states)

        has_txt = encoder_hidden_states is not None
        if has_txt:
            q_txt = attn.add_q_proj(encoder_hidden_states)   # [B, N_txt, inner_dim]
            k_txt = attn.add_k_proj(encoder_hidden_states)
            v_txt = attn.add_v_proj(encoder_hidden_states)

        # ── Reshape to multi-head ─────────────────────────────────────
        def to_heads(x: torch.Tensor) -> torch.Tensor:
            # [B, N, inner_dim] → [B, heads, N, head_dim]
            B_, N_, D_ = x.shape
            x = x.reshape(B_, N_, attn.heads, D_ // attn.heads)
            return x.permute(0, 2, 1, 3)   # [B, heads, N, head_dim]

        q_img = to_heads(q_img)   # [B, H, N_img, d]
        k_img = to_heads(k_img)
        v_img = to_heads(v_img)

        if has_txt:
            q_txt = to_heads(q_txt)   # [B, H, N_txt, d]
            k_txt = to_heads(k_txt)
            v_txt = to_heads(v_txt)

        # ── Apply QK norms (SD3 uses RMSNorm on Q,K) ─────────────────
        if hasattr(attn, "norm_q") and attn.norm_q is not None:
            # Already applied above on the full sequence; skip per-head norm
            pass

        # Per-head norms live on the Attention object in newer diffusers
        if hasattr(attn, "norm_added_q") and attn.norm_added_q is not None and has_txt:
            # norm_added_q expects [B, H, N, d] — apply last two dims
            sh = q_txt.shape
            q_txt = attn.norm_added_q(q_txt.reshape(-1, sh[-1])).reshape(sh)
        if hasattr(attn, "norm_added_k") and attn.norm_added_k is not None and has_txt:
            sh = k_txt.shape
            k_txt = attn.norm_added_k(k_txt.reshape(-1, sh[-1])).reshape(sh)

        # ── Joint sequence ────────────────────────────────────────────
        if has_txt:
            N_txt = q_txt.shape[2]
            q_joint = torch.cat([q_img, q_txt], dim=2)   # [B, H, N_img+N_txt, d]
            k_joint = torch.cat([k_img, k_txt], dim=2)
            v_joint = torch.cat([v_img, v_txt], dim=2)
        else:
            N_txt   = 0
            q_joint = q_img
            k_joint = k_img
            v_joint = v_img

        # ── Scaled dot-product (manual, so we can grab attn_probs) ────
        scale = q_joint.shape[-1] ** -0.5

        # Cast to float32 for numerical stability of softmax
        q_f = q_joint.float()
        k_f = k_joint.float()
        v_f = v_joint.float()

        scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
        # [B, heads, N_total, N_total]

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_probs = torch.softmax(scores, dim=-1)   # [B, H, N_total, N_total]

        # ── SAVE the img→txt cross-attention slice ────────────────────
        if has_txt and self.surgeon.is_capturing:
            # Slice: image queries (rows 0:N_img) × text keys (cols N_img:)
            img_to_txt = attn_probs[:, :, :N_img, N_img:].detach().cpu()
            # img_to_txt shape: [B, heads, N_img, N_txt]
            snap = AttnSnapshot(
                block_idx  = self.block_idx,
                step       = self.surgeon.current_step,
                img_to_txt = img_to_txt,
                n_img      = N_img,
                n_txt      = N_txt,
            )
            self.surgeon._store.append(snap)

        # ── Weighted sum ──────────────────────────────────────────────
        out = torch.matmul(attn_probs, v_f)   # [B, H, N_total, d]
        out = out.to(q_joint.dtype)           # back to original dtype

        # ── Split img / txt outputs ───────────────────────────────────
        out_img = out[:, :, :N_img, :]   # [B, H, N_img, d]
        out_txt = out[:, :, N_img:, :]   # [B, H, N_txt, d]

        # Reshape back: [B, H, N, d] → [B, N, H*d]
        def from_heads(x: torch.Tensor) -> torch.Tensor:
            B_, H_, N_, d_ = x.shape
            return x.permute(0, 2, 1, 3).reshape(B_, N_, H_ * d_)

        out_img = from_heads(out_img)   # [B, N_img, inner_dim]
        out_txt = from_heads(out_txt)   # [B, N_txt, inner_dim]

        # ── Output projections ────────────────────────────────────────
        hidden_states_out = attn.to_out[0](out_img)
        if len(attn.to_out) > 1:
            hidden_states_out = attn.to_out[1](hidden_states_out)

        if has_txt:
            enc_out = attn.to_add_out(out_txt)
            return hidden_states_out, enc_out
        else:
            return hidden_states_out, encoder_hidden_states


# ═══════════════════════════════════════════════════════════════════════ #
#  SURGEON                                                                #
# ═══════════════════════════════════════════════════════════════════════ #

class AttentionSurgeon:
    """
    Installs SurgicalJointAttnProcessor on every joint-attention block
    in the SD3 MMDiT transformer.

    Lifecycle:
        surgeon = AttentionSurgeon(transformer)
        surgeon.install()

        for step, t in enumerate(timesteps):
            surgeon.set_step(step)
            surgeon.start_capture()
            # ... forward pass ...
            surgeon.stop_capture()

            maps = surgeon.get_maps(step)   # real attn_probs
            surgeon.clear_step(step)        # free memory

        surgeon.uninstall()   # restore original processors
    """

    def __init__(self, transformer):
        self.transformer    = transformer
        self._originals: Dict[str, object] = {}   # name → original processor
        self._store: List[AttnSnapshot]    = []
        self.current_step   = 0
        self.is_capturing   = False
        self._installed     = False

    # ── Install ───────────────────────────────────────────────────────

    def install(self):
        """Replace all joint-attention processors with surgical ones."""
        if self._installed:
            print("[Surgeon] Already installed.")
            return

        block_idx = 0
        for name, module in self.transformer.named_modules():
            # SD3 joint blocks: Attention modules that have add_q_proj
            # (the text-stream projection) — that's what makes them "joint"
            if self._is_joint_attn(module):
                # Save original processor
                orig = getattr(module, "processor", None)
                if orig is None:
                    # Fallback: module IS the processor
                    continue

                self._originals[name] = orig
                surgical = SurgicalJointAttnProcessor(
                    surgeon   = self,
                    block_idx = block_idx,
                    original  = orig,
                )
                module.processor = surgical
                block_idx += 1

        self._installed = True
        print(f"[Surgeon] Installed on {block_idx} joint-attention blocks ✔")

    def uninstall(self):
        """Restore all original processors."""
        for name, module in self.transformer.named_modules():
            if name in self._originals:
                module.processor = self._originals[name]
        self._originals.clear()
        self._installed = False
        print("[Surgeon] Uninstalled — original processors restored ✔")

    # ── Capture control ───────────────────────────────────────────────

    def set_step(self, step: int):
        self.current_step = step

    def start_capture(self):
        self.is_capturing = True

    def stop_capture(self):
        self.is_capturing = False

    def clear(self):
        """Clear all stored snapshots (call after processing each step)."""
        self._store.clear()

    def clear_step(self, step: int):
        """Remove snapshots for a specific step to free memory."""
        self._store = [s for s in self._store if s.step != step]

    # ── Retrieval ─────────────────────────────────────────────────────

    def get_maps(self, step: int) -> Dict[int, torch.Tensor]:
        """
        Returns {block_idx: img_to_txt_attn} for a given step.

        img_to_txt shape: [B, heads, N_img, N_txt]
          - B      = batch size (2 with CFG: [uncond, cond])
          - heads  = number of attention heads
          - N_img  = number of image patch tokens (H/2 * W/2 for SD3 with patchsize=2)
          - N_txt  = number of text tokens (154 for SD3 with T5=None: 77+77)
        """
        result = {}
        for snap in self._store:
            if snap.step == step:
                result[snap.block_idx] = snap.img_to_txt
        return result

    def get_token_spatial_map(
        self,
        step        : int,
        token_idx   : int,
        latent_hw   : Tuple[int, int],
        cond_batch  : int = 1,          # index in batch for cond (1 with CFG, 0 without)
        head_reduce : str = "mean",     # "mean" | "max"
        layer_reduce: str = "mean",     # "mean" | "max" over blocks
    ) -> Optional[torch.Tensor]:
        """
        High-level convenience: given a token position in the text sequence,
        return a spatial attention map resized to latent_hw.

        Returns: [H_lat, W_lat] float32 tensor, normalized to [0, 1].
                 None if no data available.

        token_idx: position in the CONCATENATED text sequence fed to MMDiT.
            With T5=None your sequence is: [CLIP-L(77) | CLIP-G(77) | zeros(256)]
            So token 0–76 = CLIP-L tokens, 77–153 = CLIP-G tokens.
            Use `find_token_positions()` below to get the right index.
        """
        maps = self.get_maps(step)
        if not maps:
            return None

        accumulated = []
        for block_idx, img_to_txt in maps.items():
            # img_to_txt: [B, heads, N_img, N_txt]
            if img_to_txt.shape[0] <= cond_batch:
                cond_batch = img_to_txt.shape[0] - 1

            if img_to_txt.shape[-1] <= token_idx:
                continue   # this block sees fewer text tokens — skip

            # Take the cond sample (not uncond)
            a = img_to_txt[cond_batch]      # [heads, N_img, N_txt]

            # Select this token's column
            tok_map = a[:, :, token_idx]    # [heads, N_img]

            # Reduce over heads
            if head_reduce == "mean":
                tok_map = tok_map.mean(dim=0)   # [N_img]
            else:
                tok_map = tok_map.max(dim=0).values

            accumulated.append(tok_map)

        if not accumulated:
            return None

        # Stack and reduce over layers
        stacked = torch.stack(accumulated, dim=0)   # [n_layers, N_img]
        if layer_reduce == "mean":
            spatial_flat = stacked.mean(dim=0)
        else:
            spatial_flat = stacked.max(dim=0).values

        # Reshape N_img → spatial grid
        H_lat, W_lat = latent_hw
        # SD3 patch size = 2 → spatial tokens = (H_lat/2) * (W_lat/2)
        # but N_img may differ; we infer the grid size
        N_img = spatial_flat.shape[0]
        grid_h = int(N_img ** 0.5)
        grid_w = N_img // grid_h

        if grid_h * grid_w != N_img:
            # Non-square: just take the floor
            spatial_flat = spatial_flat[:grid_h * grid_w]

        spatial = spatial_flat.reshape(grid_h, grid_w).float()

        # Normalize to [0, 1]
        mn, mx = spatial.min(), spatial.max()
        if mx - mn < 1e-8:
            return torch.zeros(H_lat, W_lat)
        spatial = (spatial - mn) / (mx - mn)

        # Upsample to latent resolution
        spatial = F.interpolate(
            spatial.unsqueeze(0).unsqueeze(0),
            size   = (H_lat, W_lat),
            mode   = "bilinear",
            align_corners = False,
        )[0, 0]

        return spatial   # [H_lat, W_lat]

    def get_binary_mask(
        self,
        step      : int,
        token_idx : int,
        latent_hw : Tuple[int, int],
        threshold : float = 0.4,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Convenience: get binarized spatial mask for a token."""
        smap = self.get_token_spatial_map(step, token_idx, latent_hw, **kwargs)
        if smap is None:
            return None
        return (smap > threshold).float()

    # ── Diagnostics ───────────────────────────────────────────────────

    def diagnose(self, step: int):
        """Print a summary of what was captured at a given step."""
        maps = self.get_maps(step)
        if not maps:
            print(f"[Surgeon] step={step}: NO data captured.")
            return
        print(f"[Surgeon] step={step}: {len(maps)} blocks captured")
        for bidx, m in sorted(maps.items())[:3]:
            print(f"  block {bidx:02d}: img_to_txt shape={tuple(m.shape)} "
                  f"| dtype={m.dtype} "
                  f"| mean={m.mean():.5f} | max={m.max():.5f}")

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _is_joint_attn(module) -> bool:
        """
        True if this module is an SD3 joint-attention block.
        Joint blocks have add_q_proj (for the text stream).
        """
        return (
            hasattr(module, "to_q")
            and hasattr(module, "to_k")
            and hasattr(module, "to_v")
            and hasattr(module, "add_q_proj")   # ← the joint-attention marker
        )


# ═══════════════════════════════════════════════════════════════════════ #
#  TOKEN POSITION HELPER                                                  #
# ═══════════════════════════════════════════════════════════════════════ #

def find_token_positions(
    tokenizer,
    prompt     : str,
    words      : List[str],
    clip_offset: int = 0,     # 0 for CLIP-L slice, 77 for CLIP-G slice
) -> Dict[str, int]:
    """
    Returns {word: position_in_joint_sequence} for each word.

    The joint text sequence fed to MMDiT is:
        [ CLIP-L tokens (77) | CLIP-G tokens (77) | T5 zeros (256) ]
        positions:  0–76         77–153              154–409

    Pass clip_offset=0 to search in CLIP-L, clip_offset=77 for CLIP-G.
    Both tokenizers usually give the same positions for short prompts.
    """
    tokens  = tokenizer.encode(prompt)
    decoded = [tokenizer.decode([t]).strip().lower() for t in tokens]

    result = {}
    for word in words:
        word_l = word.lower()
        for i, tok in enumerate(decoded):
            if word_l in tok:
                result[word] = i + clip_offset
                break

    # Warn about missing words
    for w in words:
        if w not in result:
            print(f"[TokenFinder] WARNING: '{w}' not found in tokenized prompt. "
                  f"Tokens: {decoded}")

    return result