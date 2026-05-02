"""
phase0_viability.py
-------------------
Phase 0 — Viability Screening

ONE QUESTION:
    Does a separable spectral pattern exist between color, shape,
    and texture attribute perturbations in SD3 hidden token representations?

WHAT THIS DOES:
    For each prompt pair ("red car" vs "car"):
        1. Two forward passes — identical seed, identical latents, CFG disabled
        2. Subtract hidden states → ΔH (pure attribute signal)
        3. Compute spatial perturbation map M = ||ΔH||_2
        4. Apply 2D Haar DWT → {LL, LH, HL, HH}
        5. Compute normalised subband energy

DECISION RULE (written before running):
    PROCEED if:  color e_LL > 0.40
                 AND color e_LL > shape e_LL
                 AND color e_LL > texture e_LL
    STOP if:     any condition fails

USAGE:
    python phase0_viability.py
    python phase0_viability.py --device cpu --steps 5   # smoke test
    python phase0_viability.py --output_dir results/phase0/

OUTPUTS:
    results/phase0/
        fig1_subband_energy_bars.png   — main result bar chart
        fig2_perturbation_maps.png     — visual sanity check of M maps
        fig3_per_block_breakdown.png   — which blocks show separation
        phase0_table.txt               — numeric energy table
        phase0_verdict.txt             — PROCEED or STOP with reason

COMPATIBLE WITH:
    pipeline_wrapper.py  (SD3PipelineWrapper)
    custom_flow_loop.py  (FlowMatchingLoop)
    config.yaml
"""

# ================================================================== #
#  IMPORTS                                                            #
# ================================================================== #

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pywt
except ImportError:
    print("[ERROR] Install PyWavelets:  pip install PyWavelets")
    sys.exit(1)

from pipeline_wrapper import SD3PipelineWrapper
from utils             import load_config, set_seed


# ================================================================== #
#  PROMPT PAIRS                                                       #
# All use "car" as the fixed object so object identity cancels in ΔH #
# ================================================================== #

PAIRS = {
    "color": [
        ("red car",      "car"),
        ("blue car",     "car"),
        ("green car",    "car"),
        ("yellow car",   "car"),
        ("black car",    "car"),
        ("white car",    "car"),
        ("orange car",   "car"),
        ("purple car",   "car"),
        ("pink car",     "car"),
        ("brown car",    "car"),
    ],
    "shape": [
        ("round car",       "car"),
        ("square car",      "car"),
        ("oval car",        "car"),
        ("rectangular car", "car"),
        ("curved car",      "car"),
        ("angular car",     "car"),
        ("boxy car",        "car"),
        ("sleek car",       "car"),
        ("flat car",        "car"),
        ("tall car",        "car"),
    ],
    "texture": [
        ("metallic car",  "car"),
        ("wooden car",    "car"),
        ("glossy car",    "car"),
        ("rusty car",     "car"),
        ("matte car",     "car"),
        ("shiny car",     "car"),
        ("scratched car", "car"),
        ("painted car",   "car"),
        ("carbon car",    "car"),
        ("chrome car",    "car"),
    ],
}

# Blocks to probe — early / middle / late
BLOCKS_TO_WATCH = [0, 12, 23]

# Decision thresholds — written BEFORE running
PROCEED_THRESHOLD_LL          = 0.40   # color e_LL must exceed this
PROCEED_COLOR_ABOVE_SHAPE     = True   # color e_LL must exceed shape e_LL
PROCEED_COLOR_ABOVE_TEXTURE   = True   # color e_LL must exceed texture e_LL


# ================================================================== #
#  HIDDEN STATE EXTRACTOR                                             #
# Hooks into SD3 transformer_blocks to capture output hidden states  #
# ================================================================== #

class HiddenStateExtractor:
    """
    Registers forward hooks on selected MMDiT transformer blocks.
    Captures the hidden state OUTPUT of each block after each forward pass.

    Storage key: block_index (int)
    Storage value: tensor [B, D, C] on CPU float32
    """

    def __init__(self, blocks_to_watch: List[int]):
        self.blocks_to_watch = blocks_to_watch
        self.captured: Dict[int, torch.Tensor] = {}
        self._active = False

    def register(self, transformer) -> list:
        """Attach hooks. Returns handle list for later removal."""
        if not hasattr(transformer, "transformer_blocks"):
            raise AttributeError(
                "[Extractor] transformer.transformer_blocks not found.\n"
                "Run: print([n for n,_ in wrapper.transformer.named_children()])\n"
                "to find the correct attribute name."
            )

        blocks = transformer.transformer_blocks
        print(f"[Extractor] Model has {len(blocks)} transformer blocks.")
        print(f"[Extractor] Watching: {self.blocks_to_watch}")

        handles = []
        for idx, block in enumerate(blocks):
            if idx not in self.blocks_to_watch:
                continue

            def make_hook(block_idx):
                def hook(module, inputs, outputs):
                    if not self._active:
                        return
                    # outputs can be tuple (hidden, encoder_hidden) or tensor
                    h = outputs[0] if isinstance(outputs, tuple) else outputs
                    # Store as CPU float32 — safe for subtraction
                    self.captured[block_idx] = h.detach().cpu().float()
                return hook

            handles.append(block.register_forward_hook(make_hook(idx)))

        self._active = True
        print(f"[Extractor] Registered {len(handles)} hooks.")
        return handles

    def clear(self):
        self.captured.clear()

    def remove(self, handles: list):
        for h in handles:
            h.remove()
        self._active = False


# ================================================================== #
#  SINGLE FORWARD PASS                                                #
# Runs ONE ODE step at a specific timestep index and captures hidden  #
# states via the extractor hooks.                                     #
# ================================================================== #

def single_forward_pass(
    wrapper    : SD3PipelineWrapper,
    extractor  : HiddenStateExtractor,
    latents    : torch.Tensor,
    prompt     : str,
    step_idx   : int,
    timesteps  : torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """
    Runs a SINGLE transformer forward pass at timestep `step_idx`.
    Returns captured hidden states {block_idx: tensor [1, D, C]}.

    CFG is disabled — guidance_scale=1.0 in config — so batch size = 1.
    The extractor captures states automatically via hooks.
    """
    extractor.clear()

    # Encode prompt (CFG disabled → returns single embedding, not doubled)
    prompt_embeds, pooled_embeds = wrapper.encode_prompt(
        prompt          = prompt,
        negative_prompt = "",
    )

    t = timesteps[step_idx].reshape(1).to(wrapper.device)

    # Get model device and dtype
    device = next(wrapper.transformer.parameters()).device
    dtype  = next(wrapper.transformer.parameters()).dtype

    latents_in    = latents.to(device=device, dtype=dtype)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_embeds = pooled_embeds.to(device=device, dtype=dtype)
    t             = t.to(device=device, dtype=dtype)

    with torch.no_grad():
        _ = wrapper.transformer(
            hidden_states         = latents_in,
            timestep              = t,
            encoder_hidden_states = prompt_embeds,
            pooled_projections    = pooled_embeds,
        )

    # Return copy of captured states
    return {k: v.clone() for k, v in extractor.captured.items()}


# ================================================================== #
#  WAVELET DECOMPOSITION                                              #
# ================================================================== #

def compute_perturbation_map(
    h_with    : torch.Tensor,   # [1, D, C] — hidden states WITH attribute
    h_without : torch.Tensor,   # [1, D, C] — hidden states WITHOUT attribute
) -> Tuple[np.ndarray, int]:
    """
    Compute spatial perturbation map M from counterfactual subtraction.

    Returns:
        M         : [grid, grid] numpy array
        grid_size : int
    """
    # ΔH = H(attr+obj) − H(obj)
    delta_h = h_with - h_without        # [1, D, C]
    delta_h = delta_h.squeeze(0)        # [D, C]

    D, C = delta_h.shape
    grid_size = int(round(D ** 0.5))

    if grid_size * grid_size != D:
        raise ValueError(
            f"[DWT] Token count D={D} is not a perfect square.\n"
            f"Check image resolution and patch size.\n"
            f"For SD3 at 512×512 with patch_size=2: D should be 1024 (32×32)."
        )

    # L2 norm over channel dimension → scalar per token
    M_flat = delta_h.norm(dim=-1).numpy()    # [D]
    M      = M_flat.reshape(grid_size, grid_size)   # [grid, grid]

    return M, grid_size


def wavelet_subband_energy(M: np.ndarray, wavelet: str = "haar") -> Dict[str, float]:
    """
    Apply 2D single-level Haar DWT to spatial map M.
    Return normalised energy per subband.

    LL = low×low   → global/smooth → where color signal should dominate
    LH = low×high  → horizontal edges → shape signal
    HL = high×low  → vertical edges   → shape signal
    HH = high×high → fine texture     → texture signal
    """
    LL, (LH, HL, HH) = pywt.dwt2(M, wavelet=wavelet)

    E_LL = float(np.mean(LL ** 2))
    E_LH = float(np.mean(LH ** 2))
    E_HL = float(np.mean(HL ** 2))
    E_HH = float(np.mean(HH ** 2))

    total = E_LL + E_LH + E_HL + E_HH + 1e-8

    return {
        "LL": E_LL / total,
        "LH": E_LH / total,
        "HL": E_HL / total,
        "HH": E_HH / total,
        # Also store raw subbands for visualisation
        "_LL_raw": LL,
        "_LH_raw": LH,
        "_HL_raw": HL,
        "_HH_raw": HH,
    }


# ================================================================== #
#  PLOTTING                                                           #
# ================================================================== #

def plot_energy_bars(
    results : Dict[str, Dict[str, List[float]]],
    # results[attr_type][subband] = list of energy values across all pairs/blocks/steps
    output_dir : Path,
):
    """
    Figure 1: Grouped bar chart — mean subband energy per attribute type.
    This is the PRIMARY output of Phase 0.
    """
    attr_types = ["color", "shape", "texture"]
    subbands   = ["LL", "LH", "HL", "HH"]
    colors     = {"LL": "#e74c3c", "LH": "#3498db", "HL": "#2ecc71", "HH": "#f39c12"}

    # Compute means and stds
    means = {a: {s: np.mean(results[a][s]) if results[a][s] else 0.0
                 for s in subbands} for a in attr_types}
    stds  = {a: {s: np.std(results[a][s])  if results[a][s] else 0.0
                 for s in subbands} for a in attr_types}

    x     = np.arange(len(attr_types))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, sb in enumerate(subbands):
        vals = [means[a][sb] for a in attr_types]
        errs = [stds[a][sb]  for a in attr_types]
        bars = ax.bar(
            x + offsets[i] * width, vals, width,
            label=sb, color=colors[sb],
            yerr=errs, capsize=4,
            alpha=0.85, edgecolor="black", linewidth=0.6
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v*100:.0f}%",
                ha="center", va="bottom", fontsize=7
            )

    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1,
               label="Uniform baseline (25%)", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(attr_types, fontsize=13)
    ax.set_ylabel("Normalised Subband Energy", fontsize=12)
    ax.set_title(
        "Phase 0 — Subband Energy per Attribute Type\n"
        "Hypothesis: Color → LL dominant | Shape → LH/HL | Texture → HH",
        fontsize=11
    )
    ax.legend(title="Subband", fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "fig1_subband_energy_bars.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    return means


def plot_perturbation_maps(
    sample_maps : Dict[str, np.ndarray],
    # sample_maps[attr_type] = one representative M map [grid, grid]
    output_dir  : Path,
):
    """
    Figure 2: Visual sanity check.
    Shows what the perturbation map M looks like per attribute type.
    Color should look smooth, shape edgy, texture noisy.
    If all look like random noise → subtraction is too noisy → need more seeds.
    """
    attr_types = [k for k in ["color", "shape", "texture"] if k in sample_maps]
    n = len(attr_types)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    cmaps = {"color": "Reds", "shape": "Blues", "texture": "Greens"}

    for ax, attr in zip(axes, attr_types):
        M = sample_maps[attr]
        im = ax.imshow(M, cmap=cmaps.get(attr, "viridis"), interpolation="nearest")
        ax.set_title(
            f"{attr.upper()} perturbation map\n"
            f"(smooth=low-freq, edgy=mid-freq, noisy=high-freq)",
            fontsize=9
        )
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "Spatial Perturbation Maps  M = ||ΔH||₂\n"
        "Sanity check: do maps look frequency-appropriate?",
        fontsize=11
    )
    plt.tight_layout()
    path = output_dir / "fig2_perturbation_maps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_per_block_breakdown(
    block_results : Dict[int, Dict[str, Dict[str, float]]],
    # block_results[block_idx][attr_type][subband] = mean energy
    output_dir    : Path,
):
    """
    Figure 3: Per-block subband energy for LL only.
    Shows which blocks have strongest color vs shape LL separation.
    """
    blocks     = sorted(block_results.keys())
    attr_types = ["color", "shape", "texture"]
    colors_map = {"color": "#e74c3c", "shape": "#3498db", "texture": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(8, 5))

    x      = np.arange(len(blocks))
    width  = 0.25
    offset = [-1, 0, 1]

    for i, attr in enumerate(attr_types):
        vals = [block_results[b].get(attr, {}).get("LL", 0.0) for b in blocks]
        ax.bar(x + offset[i] * width, vals, width,
               label=attr, color=colors_map[attr],
               alpha=0.85, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Block {b}" for b in blocks], fontsize=11)
    ax.set_ylabel("Mean LL Subband Energy", fontsize=12)
    ax.set_title(
        "LL Energy per Block per Attribute Type\n"
        "Higher color bar = stronger frequency separation at that block",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="Uniform")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "fig3_per_block_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ================================================================== #
#  VERDICT                                                            #
# ================================================================== #

def write_verdict(
    means      : Dict[str, Dict[str, float]],
    output_dir : Path,
):
    """Apply decision rule and write verdict file."""

    color_ll   = means.get("color",   {}).get("LL", 0.0)
    shape_ll   = means.get("shape",   {}).get("LL", 0.0)
    texture_ll = means.get("texture", {}).get("LL", 0.0)

    c1 = color_ll   > PROCEED_THRESHOLD_LL
    c2 = color_ll   > shape_ll
    c3 = color_ll   > texture_ll

    proceed = c1 and c2 and c3

    lines = [
        "=" * 60,
        "PHASE 0 VERDICT",
        "=" * 60,
        "",
        "DECISION RULE (set before running):",
        f"  C1: color e_LL > {PROCEED_THRESHOLD_LL}   → {color_ll:.3f} {'✅' if c1 else '❌'}",
        f"  C2: color e_LL > shape e_LL        → {color_ll:.3f} vs {shape_ll:.3f} {'✅' if c2 else '❌'}",
        f"  C3: color e_LL > texture e_LL      → {color_ll:.3f} vs {texture_ll:.3f} {'✅' if c3 else '❌'}",
        "",
    ]

    if proceed:
        lines += [
            "VERDICT: ✅ PROCEED TO EXPERIMENT 1+2",
            "",
            "The spectral separation signal exists.",
            "Color attribute perturbations show disproportionate LL energy.",
            "The hypothesis is alive and worth full investigation.",
            "",
            "NEXT STEP:",
            "  Run experiment_1_2.py with:",
            "    20 pairs per attribute type",
            "    3 seeds",
            "    5 timesteps",
            "    blocks [0, 6, 12, 14, 23]",
        ]
    else:
        lines += [
            "VERDICT: ❌ STOP — REVISE CLAIM BEFORE PROCEEDING",
            "",
        ]
        if not c1:
            lines += [
                f"REASON: color e_LL = {color_ll:.3f} is below threshold {PROCEED_THRESHOLD_LL}",
                "Signal is too weak. Possible fixes:",
                "  → Average ΔH across 3-5 seeds instead of 1",
                "  → Focus exclusively on blocks 12-14",
                "  → Try 2-level DWT (deeper frequency decomposition)",
            ]
        if not c2:
            lines += [
                f"REASON: color e_LL ({color_ll:.3f}) ≤ shape e_LL ({shape_ll:.3f})",
                "No color-shape spectral separation found. Possible fixes:",
                "  → Try channel-space PCA instead of spatial DWT",
                "  → Check if shape prompts are too semantically similar to color",
                "  → Revise claim: LL dominance may not be color-specific",
            ]
        if not c3:
            lines += [
                f"REASON: color e_LL ({color_ll:.3f}) ≤ texture e_LL ({texture_ll:.3f})",
                "No color-texture spectral separation found.",
                "  → Same fixes as above",
            ]

    lines += [
        "",
        "=" * 60,
        "RAW MEAN ENERGIES",
        "=" * 60,
        "",
        f"{'Attr':<10} {'e_LL':>8} {'e_LH':>8} {'e_HL':>8} {'e_HH':>8} {'Dominant':>10}",
        "-" * 56,
    ]

    for attr in ["color", "shape", "texture"]:
        e = means.get(attr, {})
        ll  = e.get("LL", 0)
        lh  = e.get("LH", 0)
        hl  = e.get("HL", 0)
        hh  = e.get("HH", 0)
        dom = max({"LL": ll, "LH": lh, "HL": hl, "HH": hh}, key=lambda k: {"LL":ll,"LH":lh,"HL":hl,"HH":hh}[k])
        lines.append(f"{attr:<10} {ll:>8.3f} {lh:>8.3f} {hl:>8.3f} {hh:>8.3f} {dom:>10}")

    text = "\n".join(lines)

    # Print to console
    print(f"\n{text}\n")

    # Save to file
    path = output_dir / "phase0_verdict.txt"
    path.write_text(text)
    print(f"  Verdict saved → {path}")

    return proceed


# ================================================================== #
#  MAIN                                                               #
# ================================================================== #

def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 — Viability Screening")
    p.add_argument("--config",     type=str, default="config.yaml")
    p.add_argument("--output_dir", type=str, default="results/phase0/")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--steps",      type=int, default=10,
                   help="ODE steps (10 is enough; 5 for smoke test)")
    p.add_argument("--smoke_test", action="store_true",
                   help="Run only 2 pairs per attr type for quick debug")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  PHASE 0 — Viability Screening")
    print(f"  Device    : {args.device}")
    print(f"  ODE steps : {args.steps}")
    print(f"  Seed      : {args.seed}")
    print(f"  Blocks    : {BLOCKS_TO_WATCH}")
    print(f"  Output    : {out_dir}")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------------- #
    # 1. Load config — override for Phase 0                            #
    # ---------------------------------------------------------------- #
    cfg = load_config(args.config)

    # CRITICAL: disable CFG — clean single-pass extraction
    cfg["flow"]["guidance_scale"] = 1.0
    cfg["flow"]["num_steps"]      = args.steps
    cfg["flow"]["solver"]         = "euler"

    # ---------------------------------------------------------------- #
    # 2. Load pipeline                                                  #
    # ---------------------------------------------------------------- #
    wrapper = SD3PipelineWrapper(cfg, device=args.device)
    wrapper.load()

    # Probe token grid shape
    gen_cfg   = cfg.get("generation", {})
    H, W      = gen_cfg.get("height", 512), gen_cfg.get("width", 512)
    try:
        patch_size = wrapper.transformer.config.patch_size
    except AttributeError:
        patch_size = 2
        print(f"  [WARN] patch_size not found in config, assuming {patch_size}")

    grid_size = (H // 8) // patch_size
    print(f"\n[Grid] {H}×{W} image → {H//8}×{W//8} latent "
          f"→ patch_size={patch_size} → {grid_size}×{grid_size} token grid "
          f"= {grid_size**2} tokens\n")

    # ---------------------------------------------------------------- #
    # 3. Setup extractor and timesteps                                  #
    # ---------------------------------------------------------------- #
    extractor = HiddenStateExtractor(blocks_to_watch=BLOCKS_TO_WATCH)
    handles   = extractor.register(wrapper.transformer)

    wrapper.scheduler.set_timesteps(args.steps)
    timesteps = wrapper.scheduler.timesteps  # [num_steps] on CPU

    # Select 3 representative timestep indices
    n = args.steps
    step_indices = {
        "early" : 0,                  # t ≈ 1.0 (pure noise)
        "mid"   : n // 2,             # t ≈ 0.5
        "late"  : max(0, n - 2),      # t ≈ 0.1 (near clean)
    }
    print(f"[Timesteps] Using indices: {step_indices}")
    for name, idx in step_indices.items():
        t_val = timesteps[idx].item()
        print(f"  {name:5s}: step {idx:2d}  t={t_val:.3f}")

    # ---------------------------------------------------------------- #
    # 4. Generate shared latents — SAME for all pairs                   #
    # ---------------------------------------------------------------- #
    shared_latents = wrapper.get_initial_latents(seed=args.seed)
    print(f"\n[Latents] Shape: {shared_latents.shape}  seed={args.seed}")

    # ---------------------------------------------------------------- #
    # 5. Main loop                                                      #
    # ---------------------------------------------------------------- #
    # Storage
    # energy_records[attr_type][subband] = list of float values
    energy_records = {
        attr: {"LL": [], "LH": [], "HL": [], "HH": []}
        for attr in PAIRS
    }

    # Per-block storage for Figure 3
    # block_energy[block_idx][attr_type][subband] = list of float
    block_energy = {
        b: {attr: {"LL": [], "LH": [], "HL": [], "HH": []}
            for attr in PAIRS}
        for b in BLOCKS_TO_WATCH
    }

    # One representative M map per attribute type for Figure 2
    sample_maps: Dict[str, np.ndarray] = {}

    pairs_to_run = PAIRS
    if args.smoke_test:
        print("\n[Smoke test] Running 2 pairs per attribute only.\n")
        pairs_to_run = {attr: pairs[:2] for attr, pairs in PAIRS.items()}

    total_pairs = sum(len(v) for v in pairs_to_run.values())
    pair_count  = 0

    for attr_type, pairs in pairs_to_run.items():
        print(f"\n{'─'*50}")
        print(f"  Attribute type: {attr_type.upper()}  ({len(pairs)} pairs)")
        print(f"{'─'*50}")

        for with_prompt, without_prompt in pairs:
            pair_count += 1
            print(f"  [{pair_count}/{total_pairs}]  "
                  f"'{with_prompt}'  vs  '{without_prompt}'")

            for step_name, step_idx in step_indices.items():

                # ---------------------------------------------------- #
                # Pass 1: WITH attribute                                 #
                # ---------------------------------------------------- #
                h_with = single_forward_pass(
                    wrapper    = wrapper,
                    extractor  = extractor,
                    latents    = shared_latents,
                    prompt     = with_prompt,
                    step_idx   = step_idx,
                    timesteps  = timesteps,
                )

                # ---------------------------------------------------- #
                # Pass 2: WITHOUT attribute (same everything)            #
                # ---------------------------------------------------- #
                h_without = single_forward_pass(
                    wrapper    = wrapper,
                    extractor  = extractor,
                    latents    = shared_latents,
                    prompt     = without_prompt,
                    step_idx   = step_idx,
                    timesteps  = timesteps,
                )

                # ---------------------------------------------------- #
                # Compute ΔH → M → DWT per captured block               #
                # ---------------------------------------------------- #
                for block_idx in BLOCKS_TO_WATCH:
                    if block_idx not in h_with or block_idx not in h_without:
                        print(f"    [WARN] Block {block_idx} not captured. "
                              f"Check hook registration.")
                        continue

                    try:
                        M, gs = compute_perturbation_map(
                            h_with[block_idx],
                            h_without[block_idx],
                        )
                    except ValueError as e:
                        print(f"    [WARN] {e}")
                        continue

                    energies = wavelet_subband_energy(M, wavelet="haar")

                    # Record energies
                    for sb in ["LL", "LH", "HL", "HH"]:
                        energy_records[attr_type][sb].append(energies[sb])
                        block_energy[block_idx][attr_type][sb].append(energies[sb])

                    # Save one representative map per attr type
                    if (attr_type not in sample_maps
                            and block_idx == 12
                            and step_name == "mid"):
                        sample_maps[attr_type] = M.copy()

            print(f"    ✓ done")

    # ---------------------------------------------------------------- #
    # 6. Aggregate results                                              #
    # ---------------------------------------------------------------- #
    print("\n[Aggregating results...]")

    # Mean energy per attr type per subband
    mean_energies = {
        attr: {
            sb: float(np.mean(vals)) if vals else 0.0
            for sb, vals in sbs.items()
        }
        for attr, sbs in energy_records.items()
    }

    # Mean per block
    block_means = {
        b: {
            attr: {
                sb: float(np.mean(vals)) if vals else 0.0
                for sb, vals in sbs.items()
            }
            for attr, sbs in attr_dict.items()
        }
        for b, attr_dict in block_energy.items()
    }

    # ---------------------------------------------------------------- #
    # 7. Save numeric table                                             #
    # ---------------------------------------------------------------- #
    table_lines = [
        "Phase 0 — Numeric Energy Table",
        "=" * 56,
        f"{'Attr':<10} {'e_LL':>8} {'e_LH':>8} {'e_HL':>8} {'e_HH':>8}",
        "-" * 46,
    ]
    for attr in ["color", "shape", "texture"]:
        e = mean_energies[attr]
        table_lines.append(
            f"{attr:<10} {e['LL']:>8.4f} {e['LH']:>8.4f} "
            f"{e['HL']:>8.4f} {e['HH']:>8.4f}"
        )
    table_lines.append("")
    table_lines.append("Per-block LL energy:")
    table_lines.append(f"{'Block':<8} {'color_LL':>10} {'shape_LL':>10} {'texture_LL':>12}")
    table_lines.append("-" * 42)
    for b in sorted(block_means.keys()):
        c_ll = block_means[b]["color"]["LL"]
        s_ll = block_means[b]["shape"]["LL"]
        t_ll = block_means[b]["texture"]["LL"]
        table_lines.append(f"{b:<8} {c_ll:>10.4f} {s_ll:>10.4f} {t_ll:>12.4f}")

    table_text = "\n".join(table_lines)
    print(f"\n{table_text}\n")
    (out_dir / "phase0_table.txt").write_text(table_text)

    # ---------------------------------------------------------------- #
    # 8. Plots                                                          #
    # ---------------------------------------------------------------- #
    print("[Plotting...]")
    means_for_verdict = plot_energy_bars(energy_records, out_dir)
    plot_perturbation_maps(sample_maps, out_dir)
    plot_per_block_breakdown(block_means, out_dir)

    # ---------------------------------------------------------------- #
    # 9. Verdict                                                        #
    # ---------------------------------------------------------------- #
    proceed = write_verdict(means_for_verdict, out_dir)

    # ---------------------------------------------------------------- #
    # 10. Cleanup                                                       #
    # ---------------------------------------------------------------- #
    extractor.remove(handles)

    print(f"\n{'='*60}")
    print(f"  Phase 0 complete.")
    print(f"  Results → {out_dir}")
    if proceed:
        print("  Status  → ✅ PROCEED to Experiment 1+2")
    else:
        print("  Status  → ❌ STOP — read phase0_verdict.txt for guidance")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()