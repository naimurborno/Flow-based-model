"""
experiment3_spatial_leakage.py
-------------------------------
Experiment 3 — Spatial Leakage Ratio

QUESTION BEING ANSWERED:
    Does the geometric concentration of color perturbations (proved in Phase 0)
    actually cause color to SPREAD INTO THE WRONG SPATIAL REGIONS?

    i.e. when you add "red" to a prompt about a car next to a bicycle,
    does the "red" perturbation bleed into the bicycle's token region?
    And is this leakage STRONGER for color than for shape or texture?

WHAT THIS DOES:
    For each two-object conflict prompt ("red car next to blue bicycle"):
        1. Run two forward passes:
              Pass A: "red car next to blue bicycle"
              Pass B: "car next to bicycle"           ← SAME objects, NO attributes
           Both passes: identical seed, identical latents, CFG disabled
        2. Compute ΔH = H(with_attrs) - H(without_attrs)
        3. Compute perturbation map M = ||ΔH||_2 per token → [grid, grid]
        4. Get object masks from noun token attention maps:
              mask_car      = attention("car" token)     > threshold
              mask_bicycle  = attention("bicycle" token) > threshold
        5. Compute leakage ratio per attribute type:
              Leak = energy of M in NON-TARGET region
                     ──────────────────────────────────
                     energy of M in TARGET region
           Leak > 1 → more energy outside object than inside → leakage confirmed

DECISION RULE (written BEFORE running):
    PROCEED if:
        color Leak_ratio > 1.0                    (color actually leaks)
        color Leak_ratio > shape Leak_ratio        (color leaks MORE than shape)
        color Leak_ratio > texture Leak_ratio      (color leaks MORE than texture)

USAGE:
    python experiment3_spatial_leakage.py
    python experiment3_spatial_leakage.py --smoke_test --device cpu --steps 5
    python experiment3_spatial_leakage.py --steps 10 --output_dir results/exp3/

OUTPUTS:
    results/exp3/
        fig1_leakage_ratio_bars.png     — main result: leak ratio per attr type
        fig2_perturbation_maps.png      — M maps with object masks overlaid
        fig3_per_block_leakage.png      — which blocks drive leakage
        fig4_attention_masks.png        — sanity check: do masks look right?
        exp3_table.txt                  — numeric leakage ratios
        exp3_verdict.txt                — PROCEED or STOP with reason

COMPATIBLE WITH:
    pipeline_wrapper.py   (SD3PipelineWrapper)
    custom_flow_loop.py   (FlowMatchingLoop)
    config.yaml
    Phase0_Viability.py   (same hook pattern)
"""

# ==================================================================== #
#  IMPORTS                                                              #
# ==================================================================== #

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pipeline_wrapper import SD3PipelineWrapper
from utils             import load_config, set_seed


# ==================================================================== #
#  CONFLICT PROMPTS                                                     #
#                                                                       #
#  Each entry: (with_attrs_prompt, without_attrs_prompt,               #
#               target_object, non_target_object,                      #
#               attr_type)                                              #
#                                                                       #
#  CRITICAL DESIGN RULE:                                                #
#   without_attrs_prompt keeps BOTH objects — only removes attributes  #
#   This ensures object structure cancels in ΔH, only attr signal left #
# ==================================================================== #

CONFLICT_PROMPTS = {

    "color": [
        # (with_attrs,                          without_attrs,             target,    non_target)
        ("red car next to blue bicycle",         "car next to bicycle",     "car",     "bicycle"),
        ("yellow banana beside green apple",     "banana beside apple",     "banana",  "apple"),
        ("red rose next to white lily",          "rose next to lily",       "rose",    "lily"),
        ("black cat next to orange cat",         "cat next to cat",         "cat",     "cat"),
        ("red house beside blue barn",           "house beside barn",       "house",   "barn"),
        ("purple bag next to yellow bag",        "bag next to bag",         "bag",     "bag"),
        ("pink flower beside orange flower",     "flower beside flower",    "flower",  "flower"),
        ("green bottle next to red bottle",      "bottle next to bottle",   "bottle",  "bottle"),
    ],

    "shape": [
        ("round table next to square chair",     "table next to chair",     "table",   "chair"),
        ("oval mirror beside rectangular door",  "mirror beside door",      "mirror",  "door"),
        ("triangular sign next to circular clock","sign next to clock",     "sign",    "clock"),
        ("round ball next to square box",        "ball next to box",        "ball",    "box"),
        ("curved sofa beside angular desk",      "sofa beside desk",        "sofa",    "desk"),
        ("flat plate next to tall glass",        "plate next to glass",     "plate",   "glass"),
    ],

    "texture": [
        ("metallic car next to wooden bicycle",  "car next to bicycle",     "car",     "bicycle"),
        ("glossy table beside rusty chair",      "table beside chair",      "table",   "chair"),
        ("smooth stone next to rough brick",     "stone next to brick",     "stone",   "brick"),
        ("silky curtain next to rough carpet",   "curtain next to carpet",  "curtain", "carpet"),
        ("shiny vase beside matte pot",          "vase beside pot",         "vase",    "pot"),
        ("wooden door next to metallic window",  "door next to window",     "door",    "window"),
    ],
}

# Blocks to probe — based on DAVE finding that blocks 12-14 matter most
BLOCKS_TO_WATCH = [0, 12, 23]

# Attention mask threshold — noun token attention map binarization
MASK_THRESHOLD = 0.3   # top 30% of attention mass = object region

# Decision thresholds — written BEFORE running
PROCEED_COLOR_LEAK_GT_1        = True   # color Leak > 1.0
PROCEED_COLOR_GT_SHAPE         = True   # color Leak > shape Leak
PROCEED_COLOR_GT_TEXTURE       = True   # color Leak > texture Leak


# ==================================================================== #
#  HIDDEN STATE + ATTENTION EXTRACTOR                                   #
# ==================================================================== #

class FeatureExtractor:
    """
    Captures BOTH hidden states and attention maps from MMDiT blocks.

    Hidden states: output of each JointTransformerBlock → image tokens [B, D, C]
    Attention maps: output of the attention submodule → [B, heads, D_total, D_total]

    SD3 MMDiT specifics:
        - JointTransformerBlock returns (encoder_hidden_states, hidden_states)
        - Index 0 = text tokens (encoder_hidden_states) — can be None on last block
        - Index 1 = image tokens (hidden_states) — what we want
        - Last block has context_pre_only=True → encoder_hidden_states = None
    """

    def __init__(self, blocks_to_watch: List[int]):
        self.blocks_to_watch = blocks_to_watch
        self.hidden_states:  Dict[int, torch.Tensor] = {}
        self.attention_maps: Dict[int, torch.Tensor] = {}
        self._active = False

    def register(self, transformer) -> list:
        if not hasattr(transformer, "transformer_blocks"):
            raise AttributeError(
                "[Extractor] transformer.transformer_blocks not found."
            )

        blocks = transformer.transformer_blocks
        print(f"[Extractor] Model has {len(blocks)} transformer blocks.")
        print(f"[Extractor] Watching blocks: {self.blocks_to_watch}")

        handles = []

        for idx, block in enumerate(blocks):
            if idx not in self.blocks_to_watch:
                continue

            # ---- Hook 1: capture IMAGE hidden states after each block ----
            def make_hidden_hook(block_idx):
                def hook(module, inputs, outputs):
                    if not self._active:
                        return
                    # outputs = (encoder_hidden_states, hidden_states)
                    # encoder_hidden_states can be None on last block
                    if isinstance(outputs, tuple):
                        h = outputs[1]   # image tokens — index 1
                    else:
                        h = outputs
                    if h is None:
                        return
                    self.hidden_states[block_idx] = h.detach().cpu().float()
                return hook

            # ---- Hook 2: capture attention weights from attn submodule ----
            def make_attn_hook(block_idx):
                def hook(module, inputs, outputs):
                    if not self._active:
                        return
                    # Attention output is typically a tuple (attn_output, attn_weights)
                    # or just attn_output depending on diffusers version
                    # We try to capture the weights if available
                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        attn_weights = outputs[1]
                        if attn_weights is not None:
                            self.attention_maps[block_idx] = (
                                attn_weights.detach().cpu().float()
                            )
                return hook

            handles.append(
                block.register_forward_hook(make_hidden_hook(idx))
            )

            # Try to hook attention submodule — may vary by diffusers version
            if hasattr(block, "attn"):
                handles.append(
                    block.attn.register_forward_hook(make_attn_hook(idx))
                )

        self._active = True
        print(f"[Extractor] Registered {len(handles)} hooks.")
        return handles

    def clear(self):
        self.hidden_states.clear()
        self.attention_maps.clear()

    def remove(self, handles: list):
        for h in handles:
            h.remove()
        self._active = False


# ==================================================================== #
#  SINGLE FORWARD PASS                                                  #
# ==================================================================== #

def single_forward_pass(
    wrapper   : SD3PipelineWrapper,
    extractor : FeatureExtractor,
    latents   : torch.Tensor,
    prompt    : str,
    step_idx  : int,
    timesteps : torch.Tensor,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Runs a SINGLE transformer forward pass at timestep step_idx.
    Returns:
        hidden_states  {block_idx: [1, D, C]}
        attention_maps {block_idx: [1, heads, D_total, D_total]}  (may be empty)

    CFG disabled (guidance_scale=1.0) → batch size = 1, clean signal.
    Same latents and timestep must be used for both paired prompts.
    """
    extractor.clear()

    # Encode prompt — CFG disabled means no doubling
    prompt_embeds, pooled_embeds = wrapper.encode_prompt(
        prompt          = prompt,
        negative_prompt = "",
    )

    t = timesteps[step_idx].reshape(1).to(wrapper.device)

    device = wrapper.device
    dtype  = next(wrapper.transformer.parameters()).dtype

    lat  = latents.to(device=device, dtype=dtype)
    emb  = prompt_embeds.to(device=device, dtype=dtype)
    pool = pooled_embeds.to(device=device, dtype=dtype)
    t    = t.to(device=device, dtype=dtype)

    with torch.no_grad():
        _ = wrapper.transformer(
            hidden_states         = lat,
            timestep              = t,
            encoder_hidden_states = emb,
            pooled_projections    = pool,
        )

    return (
        dict(extractor.hidden_states),
        dict(extractor.attention_maps),
    )


# ==================================================================== #
#  PERTURBATION MAP                                                     #
# ==================================================================== #

def compute_perturbation_map(
    h_with    : torch.Tensor,   # [1, D, C]
    h_without : torch.Tensor,   # [1, D, C]
) -> Tuple[np.ndarray, int]:
    """
    Computes M = ||ΔH||_2 per token, reshaped to 2D spatial grid.

    Returns:
        M         : np.ndarray [grid_size, grid_size]
        grid_size : int
    """
    delta = h_with[0] - h_without[0]           # [D, C]
    norms = torch.norm(delta, dim=-1)           # [D]

    D = norms.shape[0]
    grid_size = int(D ** 0.5)

    if grid_size * grid_size != D:
        raise ValueError(
            f"Token count D={D} is not a perfect square. "
            f"Cannot reshape to 2D grid. Check image resolution."
        )

    M = norms.numpy().reshape(grid_size, grid_size)
    return M, grid_size


# ==================================================================== #
#  OBJECT MASK FROM NOUN ATTENTION                                      #
# ==================================================================== #

def get_noun_token_index(
    wrapper : SD3PipelineWrapper,
    prompt  : str,
    noun    : str,
) -> Optional[int]:
    """
    Finds the token index of a noun in the prompt using CLIP-L tokenizer.
    Returns None if noun not found.

    Note: This is an approximation — CLIP uses BPE tokenization so
    multi-character words may be split. We find the first matching token.
    """
    tokenizer = wrapper.tokenizer
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    noun_ids   = tokenizer(noun,   return_tensors="pt").input_ids[0]

    # noun_ids typically: [BOS, token(s), EOS] — we want just the token(s)
    # Remove BOS (index 0) and EOS (last index)
    noun_token_ids = noun_ids[1:-1]

    prompt_tokens = prompt_ids.tolist()
    noun_tokens   = noun_token_ids.tolist()

    # Find first occurrence of noun token sequence in prompt
    for i in range(len(prompt_tokens) - len(noun_tokens) + 1):
        if prompt_tokens[i:i + len(noun_tokens)] == noun_tokens:
            return i   # return index of first noun token

    return None


def build_object_mask_from_attention(
    attention_map : Optional[torch.Tensor],   # [1, heads, D_total, D_total]
    noun_token_idx: Optional[int],
    grid_size     : int,
    threshold     : float = MASK_THRESHOLD,
) -> Optional[np.ndarray]:
    """
    Builds a binary spatial mask for an object by thresholding the
    attention map of its noun token.

    D_total = D_text + D_image tokens (joint attention in MMDiT)
    We want only the image→image or text→image portion.

    Returns:
        mask: np.ndarray [grid_size, grid_size] bool, or None if unavailable
    """
    if attention_map is None or noun_token_idx is None:
        return None

    # attention_map: [1, heads, D_total, D_total]
    # Average across heads
    attn = attention_map[0].mean(dim=0)   # [D_total, D_total]

    D_total = attn.shape[0]
    D_image = grid_size * grid_size

    # Text tokens occupy the first (D_total - D_image) positions
    D_text = D_total - D_image

    if noun_token_idx >= D_text:
        return None   # index out of text range

    # Get attention from noun token to ALL image tokens
    # attn[noun_token_idx, D_text:] = attention from noun to image patches
    attn_to_image = attn[noun_token_idx, D_text:]   # [D_image]

    if attn_to_image.shape[0] != D_image:
        return None

    # Normalize to [0, 1]
    attn_map = attn_to_image.numpy()
    if attn_map.max() > 0:
        attn_map = attn_map / attn_map.max()

    # Reshape to spatial grid
    attn_spatial = attn_map.reshape(grid_size, grid_size)

    # Binarize — top threshold% of attention = object region
    mask = attn_spatial >= threshold
    return mask


def build_spatial_half_mask(
    grid_size  : int,
    side       : str,   # "left" or "right"
) -> np.ndarray:
    """
    Fallback mask when attention maps are unavailable.
    For "X next to Y" prompts, left half ≈ object 1, right half ≈ object 2.
    Very rough but useful as sanity check.
    """
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mid  = grid_size // 2
    if side == "left":
        mask[:, :mid] = True
    else:
        mask[:, mid:] = True
    return mask


# ==================================================================== #
#  LEAKAGE RATIO COMPUTATION                                            #
# ==================================================================== #

def compute_leakage_ratio(
    M              : np.ndarray,   # [grid_size, grid_size]
    target_mask    : np.ndarray,   # [grid_size, grid_size] bool
    non_target_mask: np.ndarray,   # [grid_size, grid_size] bool
) -> Dict[str, float]:
    """
    Computes leakage ratio:

        Leak = mean energy in NON-TARGET region
               ──────────────────────────────────
               mean energy in TARGET region

    Leak > 1 → attribute bleeds more into wrong object than right object
    Leak < 1 → attribute stays within its target object

    Returns dict with ratio and component energies for diagnostics.
    """
    M_sq = M ** 2   # energy = squared magnitude

    target_energy     = M_sq[target_mask].mean()     if target_mask.any()     else 0.0
    non_target_energy = M_sq[non_target_mask].mean() if non_target_mask.any() else 0.0

    if target_energy == 0:
        ratio = float("nan")
    else:
        ratio = float(non_target_energy / target_energy)

    return {
        "leak_ratio"       : ratio,
        "target_energy"    : float(target_energy),
        "non_target_energy": float(non_target_energy),
    }


# ==================================================================== #
#  PLOTTING                                                             #
# ==================================================================== #

def plot_leakage_ratio_bars(
    leak_records : Dict[str, List[float]],
    out_dir      : Path,
) -> Dict[str, float]:
    """
    Figure 1 — Bar chart of mean leakage ratio per attribute type.
    This is the PRIMARY result figure.
    """
    attrs   = ["color", "shape", "texture"]
    colors  = ["#E74C3C", "#3498DB", "#2ECC71"]
    means   = []
    stds    = []

    for attr in attrs:
        vals = [v for v in leak_records.get(attr, []) if not np.isnan(v)]
        means.append(np.mean(vals) if vals else 0.0)
        stds.append(np.std(vals)   if vals else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(attrs))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, width=0.5)

    # Reference line at 1.0 — above = leakage confirmed
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5,
               label="Leak ratio = 1.0 (threshold)")

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in attrs], fontsize=13)
    ax.set_ylabel("Leakage Ratio (non-target / target)", fontsize=12)
    ax.set_title(
        "Experiment 3: Spatial Leakage Ratio per Attribute Type\n"
        "Ratio > 1.0 means attribute bleeds more into wrong object",
        fontsize=12
    )
    ax.legend(fontsize=10)

    # Annotate bars with values
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()
    path = out_dir / "fig1_leakage_ratio_bars.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

    return dict(zip(attrs, means))


def plot_perturbation_maps_with_masks(
    sample_data : Dict[str, dict],   # attr_type → {M, target_mask, non_target_mask}
    out_dir     : Path,
):
    """
    Figure 2 — Perturbation maps with object masks overlaid.
    Visual sanity check that masks are reasonable and M differs across attr types.
    """
    attrs = [a for a in ["color", "shape", "texture"] if a in sample_data]
    if not attrs:
        return

    fig, axes = plt.subplots(len(attrs), 3,
                             figsize=(12, 4 * len(attrs)))
    if len(attrs) == 1:
        axes = axes[np.newaxis, :]

    for row, attr in enumerate(attrs):
        d = sample_data[attr]
        M              = d["M"]
        target_mask    = d.get("target_mask")
        non_target_mask= d.get("non_target_mask")

        # Col 0: raw perturbation map
        ax = axes[row, 0]
        im = ax.imshow(M, cmap="hot", interpolation="nearest")
        ax.set_title(f"{attr.capitalize()}\nPerturbation Map M", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")

        # Col 1: target mask
        ax = axes[row, 1]
        if target_mask is not None:
            ax.imshow(target_mask.astype(float), cmap="Blues",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(f"Target Mask\n({d.get('target', 'object 1')})", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No attention map\nFallback mask used",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Target Mask (fallback)", fontsize=10)
        ax.axis("off")

        # Col 2: M with mask overlay
        ax = axes[row, 2]
        ax.imshow(M, cmap="hot", interpolation="nearest")
        if target_mask is not None:
            # Overlay target as green, non-target as blue
            overlay = np.zeros((*M.shape, 4))
            if target_mask is not None:
                overlay[target_mask, 1] = 0.5   # green = target
                overlay[target_mask, 3] = 0.3
            if non_target_mask is not None:
                overlay[non_target_mask, 2] = 0.5  # blue = non-target
                overlay[non_target_mask, 3] = 0.3
            ax.imshow(overlay, interpolation="nearest")
        ax.set_title("M + Masks\n(green=target, blue=non-target)", fontsize=10)
        ax.axis("off")

    plt.suptitle("Experiment 3: Perturbation Maps with Object Masks",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "fig2_perturbation_maps.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_per_block_leakage(
    block_leak_records : Dict[int, Dict[str, List[float]]],
    out_dir            : Path,
):
    """
    Figure 3 — Per-block leakage ratio.
    Shows which transformer blocks drive leakage — should peak at blocks 12-14.
    """
    blocks = sorted(block_leak_records.keys())
    attrs  = ["color", "shape", "texture"]
    colors = {"color": "#E74C3C", "shape": "#3498DB", "texture": "#2ECC71"}

    fig, ax = plt.subplots(figsize=(9, 5))

    for attr in attrs:
        means = []
        for b in blocks:
            vals = [v for v in block_leak_records[b].get(attr, [])
                    if not np.isnan(v)]
            means.append(np.mean(vals) if vals else 0.0)
        ax.plot(blocks, means, marker="o", linewidth=2,
                color=colors[attr], label=attr.capitalize())

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2,
               alpha=0.7, label="Leak = 1.0")
    ax.set_xlabel("Transformer Block Index", fontsize=12)
    ax.set_ylabel("Mean Leakage Ratio", fontsize=12)
    ax.set_title("Experiment 3: Leakage Ratio per Block\n"
                 "(Expected peak at blocks 12-14)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xticks(blocks)
    plt.tight_layout()

    path = out_dir / "fig3_per_block_leakage.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_attention_masks_sanity(
    mask_samples : Dict[str, dict],
    out_dir      : Path,
):
    """
    Figure 4 — Sanity check: visualize raw attention maps and derived masks.
    """
    if not mask_samples:
        return

    n = len(mask_samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (key, d) in enumerate(mask_samples.items()):
        raw_attn = d.get("raw_attn")
        mask     = d.get("mask")

        ax = axes[row, 0]
        if raw_attn is not None:
            ax.imshow(raw_attn, cmap="viridis", interpolation="nearest")
            ax.set_title(f"Raw Attention: '{key}'", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No attention captured",
                    ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

        ax = axes[row, 1]
        if mask is not None:
            ax.imshow(mask.astype(float), cmap="Blues",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(f"Derived Mask: '{key}'", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No mask",
                    ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    plt.suptitle("Experiment 3: Attention Mask Sanity Check", fontsize=12)
    plt.tight_layout()
    path = out_dir / "fig4_attention_masks.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ==================================================================== #
#  VERDICT                                                              #
# ==================================================================== #

def write_verdict(
    mean_leakage : Dict[str, float],
    out_dir      : Path,
) -> bool:
    """
    Applies decision rule and writes verdict file.
    Decision rule written BEFORE running — no post-hoc threshold tuning.
    """
    color_leak   = mean_leakage.get("color",   0.0)
    shape_leak   = mean_leakage.get("shape",   0.0)
    texture_leak = mean_leakage.get("texture", 0.0)

    c1 = color_leak > 1.0
    c2 = color_leak > shape_leak
    c3 = color_leak > texture_leak

    proceed = c1 and c2 and c3

    lines = [
        "=" * 60,
        "EXPERIMENT 3 VERDICT",
        "=" * 60,
        "",
        "DECISION RULE (set before running):",
        f"  C1: color Leak > 1.0          → {color_leak:.4f}  {'✅' if c1 else '❌'}",
        f"  C2: color Leak > shape Leak   → {color_leak:.4f} vs {shape_leak:.4f}  {'✅' if c2 else '❌'}",
        f"  C3: color Leak > texture Leak → {color_leak:.4f} vs {texture_leak:.4f}  {'✅' if c3 else '❌'}",
        "",
    ]

    if proceed:
        lines += [
            "VERDICT: ✅ PROCEED TO EXPERIMENT 4",
            "",
            "Color attribute perturbations leak MORE into the wrong",
            "spatial object region than shape or texture perturbations.",
            "The geometric concentration (Phase 0) is confirmed to cause",
            "disproportionate spatial leakage.",
            "",
            "NEXT STEP:",
            "  Run experiment4_hidden_output_correlation.py",
            "  to show that hidden LL leakage predicts visible image failure.",
        ]
    else:
        lines += ["VERDICT: ❌ STOP — check which condition failed", ""]

        if not c1:
            lines += [
                "FAILED: C1 — color Leak ratio ≤ 1.0",
                "  Meaning: color perturbations do NOT bleed more into",
                "  wrong object region than into correct region.",
                "  Action: Check attention masks — are they correctly",
                "          identifying object regions?",
                "          Try lower MASK_THRESHOLD (currently 0.3).",
                "          Try different blocks (currently [0, 12, 23]).",
            ]

        if not c2:
            lines += [
                "FAILED: C2 — color Leak ≤ shape Leak",
                "  Meaning: shape leaks as much or more than color.",
                "  Action: Color is not uniquely prone to leakage.",
                "          The geometric concentration finding from Phase 0",
                "          may not directly map to spatial misbinding.",
                "          Revise hypothesis.",
            ]

        if not c3:
            lines += [
                "FAILED: C3 — color Leak ≤ texture Leak",
                "  Meaning: texture leaks as much or more than color.",
                "  Action: Same as C2 failure — check if texture",
                "          prompts are creating similar global perturbations.",
            ]

    verdict_text = "\n".join(lines)
    print(f"\n{verdict_text}\n")
    (out_dir / "exp3_verdict.txt").write_text(verdict_text)

    return proceed


# ==================================================================== #
#  MAIN                                                                 #
# ==================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3 — Spatial Leakage Ratio"
    )
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps",      type=int, default=10,
                        help="ODE steps (10 is enough for feature extraction)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/exp3/")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 2 prompts per attr only — fast sanity check")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 1. Load config — override CFG to disable it                       #
    # ---------------------------------------------------------------- #
    cfg = load_config(args.config)
    cfg["flow"]["guidance_scale"] = 1.0    # CRITICAL: disable CFG
    cfg["flow"]["num_steps"]      = args.steps

    print(f"\n[Exp3] Device   : {args.device}")
    print(f"[Exp3] Steps    : {args.steps}")
    print(f"[Exp3] Seed     : {args.seed}")
    print(f"[Exp3] Output   : {out_dir}")
    print(f"[Exp3] CFG      : DISABLED (guidance_scale=1.0)")

    # ---------------------------------------------------------------- #
    # 2. Load pipeline                                                   #
    # ---------------------------------------------------------------- #
    wrapper = SD3PipelineWrapper(cfg, device=args.device)
    wrapper.load()
    block = wrapper.transformer.transformer_blocks[12]
    print(type(block))
    print([name for name, _ in block.named_children()])
    print([name for name, _ in block.named_modules()])
    # ---------------------------------------------------------------- #
    # 3. Setup extractor and timesteps                                   #
    # ---------------------------------------------------------------- #
    extractor = FeatureExtractor(blocks_to_watch=BLOCKS_TO_WATCH)
    handles   = extractor.register(wrapper.transformer)

    wrapper.scheduler.set_timesteps(args.steps)
    timesteps = wrapper.scheduler.timesteps

    n = args.steps
    step_indices = {
        "early": 0,
        "mid"  : n // 2,
        "late" : max(0, n - 2),
    }
    print(f"\n[Timesteps] {step_indices}")

    # ---------------------------------------------------------------- #
    # 4. Shared latents — SAME for all pairs                            #
    # ---------------------------------------------------------------- #
    shared_latents = wrapper.get_initial_latents(seed=args.seed)
    print(f"[Latents] Shape: {shared_latents.shape}  seed={args.seed}\n")

    # ---------------------------------------------------------------- #
    # 5. Storage                                                         #
    # ---------------------------------------------------------------- #
    # leak_records[attr_type] = list of leak ratios (one per prompt×block×step)
    leak_records: Dict[str, List[float]] = {
        attr: [] for attr in CONFLICT_PROMPTS
    }

    # Per-block storage for Figure 3
    block_leak_records: Dict[int, Dict[str, List[float]]] = {
        b: {attr: [] for attr in CONFLICT_PROMPTS}
        for b in BLOCKS_TO_WATCH
    }

    # Sample data for Figure 2 (one representative per attr type)
    sample_maps:  Dict[str, dict] = {}

    # Mask sanity check samples for Figure 4
    mask_samples: Dict[str, dict] = {}

    # ---------------------------------------------------------------- #
    # 6. Select prompts                                                  #
    # ---------------------------------------------------------------- #
    prompts_to_run = CONFLICT_PROMPTS
    if args.smoke_test:
        print("[Smoke test] Running 2 prompts per attribute only.\n")
        prompts_to_run = {
            attr: pairs[:2] for attr, pairs in CONFLICT_PROMPTS.items()
        }

    total = sum(len(v) for v in prompts_to_run.values())
    count = 0

    # ---------------------------------------------------------------- #
    # 7. Main loop                                                       #
    # ---------------------------------------------------------------- #
    for attr_type, prompt_list in prompts_to_run.items():
        print(f"\n{'─'*60}")
        print(f"  Attribute type: {attr_type.upper()}  ({len(prompt_list)} prompts)")
        print(f"{'─'*60}")

        for with_prompt, without_prompt, target_noun, non_target_noun in prompt_list:
            count += 1
            print(f"\n  [{count}/{total}]")
            print(f"    WITH   : '{with_prompt}'")
            print(f"    WITHOUT: '{without_prompt}'")
            print(f"    Target : '{target_noun}'  |  Non-target: '{non_target_noun}'")

            # Get token indices for mask building
            target_token_idx = get_noun_token_index(
                wrapper, without_prompt, target_noun
            )
            non_target_token_idx = get_noun_token_index(
                wrapper, without_prompt, non_target_noun
            )
            print(f"    Token indices — target: {target_token_idx}  "
                  f"non-target: {non_target_token_idx}")

            for step_name, step_idx in step_indices.items():

                # ---------------------------------------------------- #
                # Pass A: WITH attributes                               #
                # ---------------------------------------------------- #
                h_with, attn_with = single_forward_pass(
                    wrapper   = wrapper,
                    extractor = extractor,
                    latents   = shared_latents,
                    prompt    = with_prompt,
                    step_idx  = step_idx,
                    timesteps = timesteps,
                )

                # ---------------------------------------------------- #
                # Pass B: WITHOUT attributes — SAME objects, no attrs  #
                # ---------------------------------------------------- #
                h_without, attn_without = single_forward_pass(
                    wrapper   = wrapper,
                    extractor = extractor,
                    latents   = shared_latents,
                    prompt    = without_prompt,
                    step_idx  = step_idx,
                    timesteps = timesteps,
                )

                # ---------------------------------------------------- #
                # Compute ΔH → M and leakage ratio per block           #
                # ---------------------------------------------------- #
                for block_idx in BLOCKS_TO_WATCH:
                    if block_idx not in h_with or block_idx not in h_without:
                        print(f"    [WARN] Block {block_idx} not captured.")
                        continue

                    try:
                        M, grid_size = compute_perturbation_map(
                            h_with[block_idx],
                            h_without[block_idx],
                        )
                    except ValueError as e:
                        print(f"    [WARN] {e}")
                        continue

                    # ---- Build object masks -------------------------
                    # Try attention-based masks first
                    attn_map = attn_without.get(block_idx)  # use without-attr pass

                    target_mask = build_object_mask_from_attention(
                        attn_map, target_token_idx, grid_size
                    )
                    non_target_mask = build_object_mask_from_attention(
                        attn_map, non_target_token_idx, grid_size
                    )

                    # Fall back to spatial halves if attention unavailable
                    using_fallback = False
                    if target_mask is None or not target_mask.any():
                        target_mask  = build_spatial_half_mask(grid_size, "left")
                        using_fallback = True
                    if non_target_mask is None or not non_target_mask.any():
                        non_target_mask = build_spatial_half_mask(grid_size, "right")
                        using_fallback = True

                    if using_fallback:
                        print(f"    [INFO] Block {block_idx} step {step_name}: "
                              f"using spatial fallback masks")

                    # ---- Compute leakage ratio ----------------------
                    result = compute_leakage_ratio(M, target_mask, non_target_mask)
                    ratio  = result["leak_ratio"]

                    if not np.isnan(ratio):
                        leak_records[attr_type].append(ratio)
                        block_leak_records[block_idx][attr_type].append(ratio)

                    # ---- Save sample for visualization --------------
                    if (attr_type not in sample_maps
                            and block_idx == 12
                            and step_name == "mid"):
                        sample_maps[attr_type] = {
                            "M"              : M.copy(),
                            "target_mask"    : target_mask.copy(),
                            "non_target_mask": non_target_mask.copy(),
                            "target"         : target_noun,
                            "non_target"     : non_target_noun,
                            "prompt"         : with_prompt,
                        }

                    # ---- Save mask sanity sample --------------------
                    mask_key = f"{attr_type}_{target_noun}_block{block_idx}"
                    if (mask_key not in mask_samples
                            and attn_map is not None
                            and target_token_idx is not None):
                        # Build raw attention map for visualization
                        attn_avg = attn_map[0].mean(dim=0)   # [D_total, D_total]
                        D_image  = grid_size * grid_size
                        D_text   = attn_avg.shape[0] - D_image
                        if (target_token_idx < D_text
                                and attn_avg.shape[1] > D_text):
                            raw = attn_avg[target_token_idx, D_text:].numpy()
                            raw = raw.reshape(grid_size, grid_size)
                            if raw.max() > 0:
                                raw = raw / raw.max()
                            mask_samples[mask_key] = {
                                "raw_attn": raw,
                                "mask"    : target_mask,
                            }

            print(f"    ✓ done  "
                  f"(running mean leak — "
                  f"color: {np.nanmean(leak_records['color']):.3f}  "
                  f"shape: {np.nanmean(leak_records['shape']):.3f}  "
                  f"texture: {np.nanmean(leak_records['texture']):.3f})")

    # ---------------------------------------------------------------- #
    # 8. Aggregate                                                       #
    # ---------------------------------------------------------------- #
    print("\n[Aggregating results...]")
    mean_leakage = {}
    for attr in CONFLICT_PROMPTS:
        vals = [v for v in leak_records[attr] if not np.isnan(v)]
        mean_leakage[attr] = np.mean(vals) if vals else 0.0

    # ---------------------------------------------------------------- #
    # 9. Save numeric table                                             #
    # ---------------------------------------------------------------- #
    table_lines = [
        "Experiment 3 — Spatial Leakage Ratio Table",
        "=" * 60,
        "",
        f"{'Attr':<12} {'Mean Leak':>12} {'Std':>8} {'N samples':>10}",
        "-" * 46,
    ]
    for attr in ["color", "shape", "texture"]:
        vals = [v for v in leak_records[attr] if not np.isnan(v)]
        m    = np.mean(vals) if vals else 0.0
        s    = np.std(vals)  if vals else 0.0
        table_lines.append(
            f"{attr:<12} {m:>12.4f} {s:>8.4f} {len(vals):>10d}"
        )

    table_lines += [
        "",
        "Per-block mean leakage ratio:",
        f"{'Block':<8} {'color':>10} {'shape':>10} {'texture':>12}",
        "-" * 42,
    ]
    for b in sorted(block_leak_records.keys()):
        def bmean(attr):
            vals = [v for v in block_leak_records[b][attr] if not np.isnan(v)]
            return np.mean(vals) if vals else 0.0
        table_lines.append(
            f"{b:<8} {bmean('color'):>10.4f} "
            f"{bmean('shape'):>10.4f} {bmean('texture'):>12.4f}"
        )

    table_text = "\n".join(table_lines)
    print(f"\n{table_text}\n")
    (out_dir / "exp3_table.txt").write_text(table_text)

    # ---------------------------------------------------------------- #
    # 10. Plots                                                          #
    # ---------------------------------------------------------------- #
    print("[Plotting...]")
    plot_leakage_ratio_bars(leak_records, out_dir)
    plot_perturbation_maps_with_masks(sample_maps, out_dir)
    plot_per_block_leakage(block_leak_records, out_dir)
    plot_attention_masks_sanity(mask_samples, out_dir)

    # ---------------------------------------------------------------- #
    # 11. Verdict                                                        #
    # ---------------------------------------------------------------- #
    proceed = write_verdict(mean_leakage, out_dir)

    # ---------------------------------------------------------------- #
    # 12. Cleanup                                                        #
    # ---------------------------------------------------------------- #
    extractor.remove(handles)

    print(f"\n{'='*60}")
    print(f"  Experiment 3 complete.")
    print(f"  Results → {out_dir}")
    if proceed:
        print("  Status  → ✅ PROCEED to Experiment 4")
    else:
        print("  Status  → ❌ STOP — read exp3_verdict.txt for guidance")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()