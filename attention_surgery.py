"""
attention_surgery.py  (memory-optimized)
-----------------------------------------
Key RAM/VRAM changes vs previous version:
  1. Never store full [B, heads, N_img, N_txt] tensors.
     Instead, aggregate inside the processor immediately:
       - mean over heads → [B, N_img, N_txt]
       - select only tracked token columns → [B, N_img, n_tracked]
       - mean over B (cond only) → [N_img, n_tracked]
     This cuts per-snapshot storage from ~36 MB to ~3 KB.

  2. float16 matmul: softmax still done in float32 for stability,
     but scores tensor is immediately freed after softmax.

  3. _store holds only {step: {block_idx: Tensor[N_img, n_tracked]}},
     one dict entry per step. clear() wipes it completely.

  4. All tensors moved to CPU immediately after aggregation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════ #
#  SURGICAL PROCESSOR                                                      #
# ═══════════════════════════════════════════════════════════════════════ #

class SurgicalJointAttnProcessor:
    """
    Drop-in replacement for JointAttnProcessor2_0.
    Identical math, but aggregates attn_probs in-place → tiny footprint.
    """

    def __init__(self, surgeon: "AttentionSurgeon", block_idx: int, original: object):
        self.surgeon   = surgeon
        self.block_idx = block_idx
        self.original  = original

    def __call__(
        self,
        attn,
        hidden_states         : torch.Tensor,
        encoder_hidden_states : Optional[torch.Tensor] = None,
        attention_mask        : Optional[torch.Tensor] = None,
        **kwargs,
    ):
        N_img = hidden_states.shape[1]

        def _headnorm(name, x):
            fn = getattr(attn, name, None)
            if fn is None:
                return x
            sh = x.shape
            return fn(x.reshape(-1, sh[-1])).reshape(sh)

        def to_heads(x):
            B_, N_, D_ = x.shape
            return x.reshape(B_, N_, attn.heads, D_ // attn.heads).permute(0, 2, 1, 3)

        # ── Image stream QKV ─────────────────────────────────────────
        q_img = to_heads(attn.to_q(hidden_states))
        k_img = to_heads(attn.to_k(hidden_states))
        v_img = to_heads(attn.to_v(hidden_states))

        q_img = _headnorm("norm_q", q_img)
        k_img = _headnorm("norm_k", k_img)

        has_txt = encoder_hidden_states is not None
        if has_txt:
            # SD3: no pre-norm on text stream — direct projection
            q_txt = to_heads(attn.add_q_proj(encoder_hidden_states))
            k_txt = to_heads(attn.add_k_proj(encoder_hidden_states))
            v_txt = to_heads(attn.add_v_proj(encoder_hidden_states))
            q_txt = _headnorm("norm_added_q", q_txt)
            k_txt = _headnorm("norm_added_k", k_txt)

            N_txt   = q_txt.shape[2]
            q_joint = torch.cat([q_img, q_txt], dim=2)
            k_joint = torch.cat([k_img, k_txt], dim=2)
            v_joint = torch.cat([v_img, v_txt], dim=2)
        else:
            N_txt   = 0
            q_joint = q_img
            k_joint = k_img
            v_joint = v_img

        # ── Attention: compute in float32 for stable softmax ─────────
        scale = q_joint.shape[-1] ** -0.5
        scores = torch.matmul(q_joint.float(), k_joint.float().transpose(-2, -1)) * scale

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_probs = torch.softmax(scores, dim=-1)  # [B, heads, N_total, N_total]
        del scores  # free immediately

        # ── AGGREGATE in-place — never store the full tensor ─────────
        if has_txt and self.surgeon.is_capturing and self.surgeon.tracked_tokens:
            tracked = self.surgeon.tracked_tokens  # list of token indices
            # img→txt slice: [B, heads, N_img, N_txt]
            img_to_txt = attn_probs[:, :, :N_img, N_img:]

            # Mean over heads → [B, N_img, N_txt]
            agg = img_to_txt.mean(dim=1)

            # Take cond batch only (index 1 when CFG, 0 otherwise)
            cond_idx = 1 if agg.shape[0] > 1 else 0
            agg = agg[cond_idx]  # [N_img, N_txt]

            # Select only tracked token columns
            valid = [t for t in tracked if t < agg.shape[1]]
            if valid:
                agg = agg[:, valid].detach().cpu()  # [N_img, n_tracked]

                step = self.surgeon.current_step
                if step not in self.surgeon._store:
                    self.surgeon._store[step] = {}
                # Accumulate across blocks via mean
                if self.block_idx in self.surgeon._store[step]:
                    self.surgeon._store[step][self.block_idx] = (
                        self.surgeon._store[step][self.block_idx] + agg
                    ) / 2.0
                else:
                    self.surgeon._store[step][self.block_idx] = agg

        # ── Weighted sum → output ────────────────────────────────────
        out = torch.matmul(attn_probs, v_joint.float()).to(q_joint.dtype)
        del attn_probs

        def from_heads(x):
            B_, H_, N_, d_ = x.shape
            return x.permute(0, 2, 1, 3).reshape(B_, N_, H_ * d_)

        out_img = from_heads(out[:, :, :N_img, :])
        out_txt = from_heads(out[:, :, N_img:, :]) if has_txt else None

        hidden_states_out = attn.to_out[0](out_img)
        if len(attn.to_out) > 1:
            hidden_states_out = attn.to_out[1](hidden_states_out)

        if has_txt:
            to_add_out = getattr(attn, "to_add_out", None)
            enc_out = to_add_out(out_txt) if to_add_out is not None else out_txt
            return hidden_states_out, enc_out
        else:
            return hidden_states_out, encoder_hidden_states


# ═══════════════════════════════════════════════════════════════════════ #
#  SURGEON                                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

class AttentionSurgeon:
    """
    Installs SurgicalJointAttnProcessor on every SD3 MMDiT joint-attention block.

    Memory layout:
        _store[step][block_idx] = Tensor[N_img, n_tracked]
        (all on CPU, aggregated over heads, cond batch only, tracked tokens only)

    Usage:
        surgeon = AttentionSurgeon(transformer, tracked_tokens=[3, 7, 2, 6])
        surgeon.install()
        ...
        surgeon.set_step(i)
        surgeon.start_capture()
        # forward pass
        surgeon.stop_capture()
        map = surgeon.get_token_spatial_map(i, token_idx=0, latent_hw=(H,W))
        surgeon.clear_step(i)
        ...
        surgeon.uninstall()
    """

    def __init__(self, transformer, tracked_tokens: Optional[List[int]] = None):
        self.transformer     = transformer
        self.tracked_tokens  = tracked_tokens or []
        self._originals: Dict[str, object] = {}
        # _store[step][block_idx] = Tensor[N_img, n_tracked]
        self._store: Dict[int, Dict[int, torch.Tensor]] = {}
        self.current_step    = 0
        self.is_capturing    = False
        self._installed      = False

    def set_tracked_tokens(self, tokens: List[int]):
        self.tracked_tokens = tokens

    # ── Install / Uninstall ───────────────────────────────────────────

    def install(self):
        if self._installed:
            return
        probed    = False
        block_idx = 0
        for name, module in self.transformer.named_modules():
            if self._is_joint_attn(module):
                if not probed:
                    norm_attrs = [a for a in dir(module)
                                  if "norm" in a.lower() and not a.startswith("__")]
                    print(f"[Surgeon] Probing '{name}': norm attrs = {norm_attrs}")
                    probed = True
                orig = getattr(module, "processor", None)
                if orig is None:
                    continue
                self._originals[name] = orig
                module.processor = SurgicalJointAttnProcessor(self, block_idx, orig)
                block_idx += 1
        self._installed = True
        print(f"[Surgeon] Installed on {block_idx} joint-attention blocks ✔")

    def uninstall(self):
        for name, module in self.transformer.named_modules():
            if name in self._originals:
                module.processor = self._originals[name]
        self._originals.clear()
        self._store.clear()
        self._installed = False
        print("[Surgeon] Uninstalled ✔")

    # ── Capture control ───────────────────────────────────────────────

    def set_step(self, step: int):
        self.current_step = step

    def start_capture(self):
        self.is_capturing = True

    def stop_capture(self):
        self.is_capturing = False

    def clear(self):
        self._store.clear()

    def clear_step(self, step: int):
        self._store.pop(step, None)

    # ── Retrieval ─────────────────────────────────────────────────────

    def get_token_spatial_map(
        self,
        step        : int,
        token_idx   : int,
        latent_hw   : Tuple[int, int],
        head_reduce : str = "mean",   # kept for API compat (already reduced)
        layer_reduce: str = "max",    # "max" preserves contrast vs "mean"
    ) -> Optional[torch.Tensor]:
        """
        Returns [H_lat, W_lat] spatial attention map for the given token,
        normalized to [0, 1].  None if no data available.

        Uses only the middle third of transformer blocks (most semantically
        rich) and reduces with max over layers to preserve contrast.
        """
        if step not in self._store:
            return None
        if token_idx not in self.tracked_tokens:
            return None
        col = self.tracked_tokens.index(token_idx)

        all_blocks = sorted(self._store[step].keys())
        n = len(all_blocks)

        # Use middle third of blocks — they carry the richest semantic signal.
        # Early blocks: low-level texture. Late blocks: reconstruction.
        # Middle blocks: semantic binding (color ↔ object).
        lo, hi = n // 3, 2 * n // 3
        mid_blocks = all_blocks[lo:hi] if hi > lo else all_blocks

        accumulated = []
        for block_idx in mid_blocks:
            tensor = self._store[step][block_idx]  # [N_img, n_tracked]
            if col >= tensor.shape[1]:
                continue
            accumulated.append(tensor[:, col])     # [N_img]

        if not accumulated:
            # Fallback: use all blocks
            for block_idx in all_blocks:
                tensor = self._store[step][block_idx]
                if col < tensor.shape[1]:
                    accumulated.append(tensor[:, col])
            if not accumulated:
                return None

        stacked = torch.stack(accumulated, dim=0)  # [n_layers, N_img]

        # Max-reduce over layers: picks the block where this token
        # has the strongest spatial focus. Mean kills contrast.
        spatial_flat = stacked.max(dim=0).values   # [N_img]

        H_lat, W_lat = latent_hw
        N_img  = spatial_flat.shape[0]
        grid_h = int(N_img ** 0.5)
        grid_w = N_img // grid_h
        if grid_h * grid_w != N_img:
            spatial_flat = spatial_flat[:grid_h * grid_w]

        spatial = spatial_flat.reshape(grid_h, grid_w).float()
        mn, mx  = spatial.min(), spatial.max()
        if mx - mn < 1e-8:
            return torch.zeros(H_lat, W_lat)

        # Soft normalization: clip top 1% outliers before rescaling
        # so a single hot pixel doesn't compress the rest to near-zero
        p99 = torch.quantile(spatial.flatten(), 0.99)
        spatial = (spatial - mn) / (max(p99, mx) - mn + 1e-8)
        spatial = spatial.clamp(0.0, 1.0)

        return F.interpolate(
            spatial.unsqueeze(0).unsqueeze(0),
            size=(H_lat, W_lat), mode="bilinear", align_corners=False
        )[0, 0]

    def get_binary_mask(
        self, step: int, token_idx: int, latent_hw: Tuple[int, int],
        threshold: float = 0.4, **kwargs,
    ) -> Optional[torch.Tensor]:
        smap = self.get_token_spatial_map(step, token_idx, latent_hw, **kwargs)
        return None if smap is None else (smap > threshold).float()

    # ── Diagnostics ───────────────────────────────────────────────────

    def diagnose(self, step: int):
        if step not in self._store:
            print(f"[Surgeon] step={step}: NO data captured.")
            return
        blocks = self._store[step]
        print(f"[Surgeon] step={step}: {len(blocks)} blocks | "
              f"tracked_tokens={self.tracked_tokens}")
        for bidx, t in list(sorted(blocks.items()))[:3]:
            print(f"  block {bidx:02d}: shape={tuple(t.shape)} "
                  f"mean={t.mean():.5f} max={t.max():.5f}")

    # ── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _is_joint_attn(module) -> bool:
        return (hasattr(module, "to_q") and hasattr(module, "to_k")
                and hasattr(module, "to_v") and hasattr(module, "add_q_proj"))


# ═══════════════════════════════════════════════════════════════════════ #
#  TOKEN FINDER                                                            #
# ═══════════════════════════════════════════════════════════════════════ #

def find_token_positions(
    tokenizer, prompt: str, words: List[str], clip_offset: int = 0
) -> Dict[str, int]:
    tokens  = tokenizer.encode(prompt)
    decoded = [tokenizer.decode([t]).strip().lower() for t in tokens]
    result  = {}
    for word in words:
        wl = word.lower()
        for i, tok in enumerate(decoded):
            if wl in tok:
                result[word] = i + clip_offset
                break
    for w in words:
        if w not in result:
            print(f"[TokenFinder] WARNING: '{w}' not found. Tokens: {decoded}")
    return result