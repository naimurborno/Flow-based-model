"""
color_leakage_experiment.py
-----------------------------
Tests the hypothesis:
  t_latent_contamination < t_attention_corruption < t_final_leakage

Works on top of the existing SD3 pipeline wrapper + flow loop.

Two measurement streams at every ODE timestep:
  Hook A — Latent Chromatic Contamination (CC)
  Hook B — Attention Misbinding (AC)

Then an optional causal intervention at t_latent to verify upstream causality.

Usage:
    python color_leakage_experiment.py \
        --config config.yaml \
        --output_dir results/leakage/ \
        --seeds 5 \
        --causal_repair
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image

from pipeline_wrapper import SD3PipelineWrapper
from custom_flow_loop import FlowMatchingLoop
from utils import load_config, set_seed, save_results


# ═══════════════════════════════════════════════════════════════════════ #
# EXPERIMENT CONFIG                                                       #
# ═══════════════════════════════════════════════════════════════════════ #

COLOR_PROMPTS = [
    {
        "prompt":   "a red cube and a blue sphere on a white table",
        "neg":      "blurry, low quality",
        "objects":  ["cube", "sphere"],
        "colors":   ["red", "blue"],
        # Target color vectors in SD3 VAE latent space (c3, c4 channels)
        # These are approximate — calibrated in Phase 0
        "target_colors": {
            "red":  torch.tensor([0.45, -0.30]),   # [c3, c4] direction for red
            "blue": torch.tensor([-0.35,  0.40]),  # [c3, c4] direction for blue
        },
    },
    {
        "prompt":   "a green apple and a yellow banana on a wooden table",
        "neg":      "blurry, low quality",
        "objects":  ["apple", "banana"],
        "colors":   ["green", "yellow"],
        "target_colors": {
            "green":  torch.tensor([0.30,  0.25]),
            "yellow": torch.tensor([0.40, -0.10]),
        },
    },
]

# SD3 has 16 latent channels; chromatic channels are indices 8–11 (empirical)
# For SD 1.5 (4ch), c3=idx2, c4=idx3. For SD3 (16ch) we use 8–11 as chromatic.
CHROMA_CHANNELS = [8, 9, 10, 11]   # indices into the 16-channel SD3 latent
CHROMA_PAIR     = [8, 9]           # the two channels used for 2D color signature


# ═══════════════════════════════════════════════════════════════════════ #
# ATTENTION HOOK MANAGER                                                  #
# ═══════════════════════════════════════════════════════════════════════ #

class AttentionHookManager:
    """
    Registers forward hooks on every MMDiT joint-attention block that has
    both to_q and to_k projections. Stores cross-attention maps keyed by
    layer name + timestep.
    """

    def __init__(self, transformer):
        self.transformer = transformer
        self.hooks: List = []
        self.attention_maps: Dict[str, torch.Tensor] = {}   # name → [heads, seq, seq]
        self._current_step = 0

    def register(self):
        """Attach hooks to all attention layers."""
        self.remove()   # clear old hooks first
        for name, module in self.transformer.named_modules():
            # SD3 MMDiT attention modules expose attn_map via forward output
            if hasattr(module, "to_q") and hasattr(module, "to_k"):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _make_hook(self, name: str):
        def hook_fn(module, inputs, output):
            # output is usually (hidden_states,) or (hidden_states, attn_weights)
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]   # [batch, heads, seq, seq]
                if attn_weights is not None:
                    key = f"step{self._current_step:03d}_{name}"
                    self.attention_maps[key] = attn_weights.detach().cpu()
        return hook_fn

    def set_step(self, step: int):
        self._current_step = step

    def get_step_maps(self, step: int) -> Dict[str, torch.Tensor]:
        prefix = f"step{step:03d}_"
        return {k[len(prefix):]: v for k, v in self.attention_maps.items()
                if k.startswith(prefix)}


# ═══════════════════════════════════════════════════════════════════════ #
# LATENT CHROMATIC CONTAMINATION MEASUREMENT                              #
# ═══════════════════════════════════════════════════════════════════════ #

def get_region_mask(
    attn_maps: Dict[str, torch.Tensor],
    token_indices: List[int],
    latent_hw: Tuple[int, int],
    threshold: float = 0.5,
) -> Optional[torch.Tensor]:
    """
    Aggregates cross-attention maps for given token indices across all layers.
    Returns a binary spatial mask [H_latent, W_latent].

    token_indices: positions of noun tokens in the text sequence (e.g. "cube" = idx 3)
    """
    if not attn_maps:
        return None

    accumulated = None
    count = 0
    for name, attn in attn_maps.items():
        # attn shape: [batch, heads, seq_q (latent), seq_k (text)]
        if attn.dim() == 4:
            # average over batch + heads; take cross-attn (latent→text) cols
            a = attn.mean(dim=(0, 1))   # [seq_q, seq_k]
            # sum over the token indices belonging to this noun
            if a.shape[-1] > max(token_indices):
                noun_map = a[:, token_indices].sum(dim=-1)   # [seq_q]
                if accumulated is None:
                    accumulated = noun_map
                else:
                    accumulated = accumulated + noun_map
                count += 1

    if accumulated is None or count == 0:
        return None

    accumulated = accumulated / count
    H, W = latent_hw

    # Reshape: seq_q → spatial grid
    spatial_size = H * W
    if accumulated.shape[0] >= spatial_size:
        spatial = accumulated[:spatial_size].reshape(H, W)
    else:
        # Some layers use different resolutions — skip
        return None

    # Normalize and threshold
    mn, mx = spatial.min(), spatial.max()
    if mx - mn < 1e-6:
        return None
    spatial = (spatial - mn) / (mx - mn)
    return (spatial > threshold).float()   # [H, W]


def measure_chromatic_contamination(
    latents: torch.Tensor,      # [1, C, H, W]
    mask_A:  torch.Tensor,      # [H, W]  — region for object A
    mask_B:  torch.Tensor,      # [H, W]  — region for object B
    color_A: torch.Tensor,      # [2]     — expected chroma for A
    color_B: torch.Tensor,      # [2]     — expected chroma for B
) -> Tuple[float, float, float]:
    """
    Returns:
        cc_A_in_B : how much of color_A leaked into region B
        cc_B_in_A : how much of color_B leaked into region A
        total_cc  : combined contamination score
    """
    # Extract chromatic channels [1, 2, H, W]
    c = latents[0, CHROMA_PAIR, :, :]   # [2, H, W]

    H, W = c.shape[1], c.shape[2]
    device = c.device
    mA = mask_A.to(device)
    mB = mask_B.to(device)
    cA = color_A.to(device)
    cB = color_B.to(device)

    def region_mean_chroma(mask: torch.Tensor) -> Optional[torch.Tensor]:
        if mask.sum() < 1:
            return None
        # Resize mask to match latent spatial dims if needed
        if mask.shape != (H, W):
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (H, W), mode="nearest")[0, 0]
        weighted = (c * mask.unsqueeze(0)).sum(dim=(1, 2)) / (mask.sum() + 1e-8)
        return weighted   # [2]

    sig_A = region_mean_chroma(mA)
    sig_B = region_mean_chroma(mB)

    if sig_A is None or sig_B is None:
        return 0.0, 0.0, 0.0

    # Cosine similarity of region B's chroma with expected color_A (contamination)
    def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    cc_A_in_B = max(0.0, cos_sim(sig_B, cA))   # color_A bleeding into region_B
    cc_B_in_A = max(0.0, cos_sim(sig_A, cB))   # color_B bleeding into region_A
    total_cc  = (cc_A_in_B + cc_B_in_A) / 2.0

    return cc_A_in_B, cc_B_in_A, total_cc


# ═══════════════════════════════════════════════════════════════════════ #
# ATTENTION CORRUPTION MEASUREMENT                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def measure_attention_corruption(
    attn_maps:     Dict[str, torch.Tensor],
    color_tok_A:   int,    # token index of color word for object A (e.g. "red")
    color_tok_B:   int,    # token index of color word for object B (e.g. "blue")
    noun_tok_A:    int,    # token index of noun A (e.g. "cube")
    noun_tok_B:    int,    # token index of noun B (e.g. "sphere")
    latent_hw:     Tuple[int, int],
) -> float:
    """
    Measures cross-attention misbinding:
    How much does 'red' attend to the sphere region, and 'blue' to the cube region?

    Returns an IoU-like overlap score (higher = more misbinding).
    """
    if not attn_maps:
        return 0.0

    # Get spatial attention maps for each token
    def token_spatial_map(token_idx: int) -> Optional[torch.Tensor]:
        accum = None
        n = 0
        for name, attn in attn_maps.items():
            if attn.dim() != 4:
                continue
            a = attn.mean(dim=(0, 1))   # [seq_q, seq_k]
            if a.shape[-1] <= token_idx:
                continue
            sq = a[:, token_idx]        # [seq_q]
            H, W = latent_hw
            if sq.shape[0] < H * W:
                continue
            spatial = sq[:H * W].reshape(H, W)
            spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8)
            if accum is None:
                accum = spatial
            else:
                accum = accum + spatial
            n += 1
        if accum is None or n == 0:
            return None
        return (accum / n > 0.5).float()

    noun_map_A = token_spatial_map(noun_tok_A)   # where object A is
    noun_map_B = token_spatial_map(noun_tok_B)   # where object B is
    color_map_A = token_spatial_map(color_tok_A) # where color A attends
    color_map_B = token_spatial_map(color_tok_B) # where color B attends

    if any(m is None for m in [noun_map_A, noun_map_B, color_map_A, color_map_B]):
        return 0.0

    def iou(a: torch.Tensor, b: torch.Tensor) -> float:
        inter = (a * b).sum()
        union = ((a + b) > 0).float().sum()
        return float(inter / (union + 1e-8))

    # Misbinding: color_A attending to object_B's region, and vice versa
    misbind_A = iou(color_map_A, noun_map_B)
    misbind_B = iou(color_map_B, noun_map_A)

    return (misbind_A + misbind_B) / 2.0


# ═══════════════════════════════════════════════════════════════════════ #
# TOKEN INDEX FINDER                                                      #
# ═══════════════════════════════════════════════════════════════════════ #

def find_token_indices(tokenizer, prompt: str, words: List[str]) -> Dict[str, int]:
    """
    Returns the position of each word in the tokenized prompt.
    Uses CLIP-L tokenizer (same as the cond embedding sequence).
    """
    tokens = tokenizer.encode(prompt)
    decoded = [tokenizer.decode([t]) for t in tokens]
    result = {}
    for word in words:
        for i, tok in enumerate(decoded):
            if word.lower() in tok.lower().strip():
                result[word] = i
                break
    return result


# ═══════════════════════════════════════════════════════════════════════ #
# CAUSAL INTERVENTION                                                     #
# ═══════════════════════════════════════════════════════════════════════ #

def repair_chromatic_contamination(
    latents:  torch.Tensor,    # [1, C, H, W]
    mask_A:   torch.Tensor,    # [H, W]
    mask_B:   torch.Tensor,    # [H, W]
    color_A:  torch.Tensor,    # [2] expected chroma for A
    color_B:  torch.Tensor,    # [2] expected chroma for B
    strength: float = 0.7,
) -> torch.Tensor:
    """
    Corrects chromatic contamination by nudging pixels in each region
    back toward their expected color direction in c3,c4 space.
    Does NOT touch attention weights.
    """
    repaired = latents.clone()
    device   = latents.device
    H, W     = latents.shape[2], latents.shape[3]

    for mask, target_color in [(mask_A, color_A), (mask_B, color_B)]:
        if mask.sum() < 1:
            continue
        msk = mask.to(device)
        if msk.shape != (H, W):
            msk = F.interpolate(msk.unsqueeze(0).unsqueeze(0), (H, W), mode="nearest")[0, 0]

        for i, ch_idx in enumerate(CHROMA_PAIR):
            ch_plane = repaired[0, ch_idx, :, :]          # [H, W]
            # Region mean
            region_mean = (ch_plane * msk).sum() / (msk.sum() + 1e-8)
            # Target value proportional to expected color direction
            target_val  = target_color[i].to(device) * ch_plane.abs().mean()
            # Blend toward target
            correction  = target_val - region_mean
            repaired[0, ch_idx, :, :] = ch_plane + msk * correction * strength

    return repaired


# ═══════════════════════════════════════════════════════════════════════ #
# INSTRUMENTED FLOW LOOP                                                  #
# ═══════════════════════════════════════════════════════════════════════ #

class InstrumentedFlowLoop(FlowMatchingLoop):
    """
    Extends FlowMatchingLoop with per-step CC and AC measurement.
    Optionally applies chromatic repair at t_repair.
    """

    def __init__(self, *args, hook_manager: AttentionHookManager,
                 token_config: dict,
                 prompt_config: dict,
                 latent_hw: Tuple[int, int],
                 do_causal_repair: bool = False,
                 repair_at_step: Optional[int] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_manager     = hook_manager
        self.token_config     = token_config
        self.prompt_config    = prompt_config
        self.latent_hw        = latent_hw
        self.do_causal_repair = do_causal_repair
        self.repair_at_step   = repair_at_step

        # Measurement logs
        self.cc_log: List[float] = []    # contamination at each step
        self.ac_log: List[float] = []    # attention corruption at each step
        self.t_log:  List[float] = []    # timestep value

    def _step_callback(
        self,
        latents:  torch.Tensor,
        t:        torch.Tensor,
        step_idx: int,
        velocity: torch.Tensor,
    ) -> torch.Tensor:

        t_val = t.item() if hasattr(t, "item") else float(t)
        self.hook_manager.set_step(step_idx)

        # ── Get attention maps from this step ─────────────────────────
        attn_maps = self.hook_manager.get_step_maps(step_idx)

        tc = self.token_config
        pc = self.prompt_config

        # ── Build region masks from noun attention ─────────────────────
        mask_A = get_region_mask(
            attn_maps,
            token_indices=[tc.get(pc["objects"][0], 5)],
            latent_hw=self.latent_hw,
        )
        mask_B = get_region_mask(
            attn_maps,
            token_indices=[tc.get(pc["objects"][1], 7)],
            latent_hw=self.latent_hw,
        )

        # ── Fallback: split image in half if masks unavailable ─────────
        H, W = self.latent_hw
        if mask_A is None:
            mask_A = torch.zeros(H, W)
            mask_A[:, : W // 2] = 1.0
        if mask_B is None:
            mask_B = torch.zeros(H, W)
            mask_B[:, W // 2:] = 1.0

        # ── Hook A: Latent Chromatic Contamination ─────────────────────
        color_A = pc["target_colors"][pc["colors"][0]]
        color_B = pc["target_colors"][pc["colors"][1]]

        _, _, total_cc = measure_chromatic_contamination(
            latents, mask_A, mask_B, color_A, color_B
        )
        self.cc_log.append(total_cc)

        # ── Hook B: Attention Corruption ──────────────────────────────
        total_ac = measure_attention_corruption(
            attn_maps,
            color_tok_A  = tc.get(pc["colors"][0], 2),
            color_tok_B  = tc.get(pc["colors"][1], 4),
            noun_tok_A   = tc.get(pc["objects"][0], 5),
            noun_tok_B   = tc.get(pc["objects"][1], 7),
            latent_hw    = self.latent_hw,
        )
        self.ac_log.append(total_ac)
        self.t_log.append(t_val)

        # ── Optional: causal repair at specified step ─────────────────
        if self.do_causal_repair and self.repair_at_step == step_idx:
            print(f"  [REPAIR] Applying chromatic correction at step {step_idx} (t={t_val:.3f})")
            latents = repair_chromatic_contamination(latents, mask_A, mask_B, color_A, color_B)

        return latents


# ═══════════════════════════════════════════════════════════════════════ #
# ONSET DETECTION                                                         #
# ═══════════════════════════════════════════════════════════════════════ #

def detect_onset(values: List[float], window: int = 5, sigma_thresh: float = 2.0) -> Optional[int]:
    """
    Returns the first index where values exceed baseline mean + 2σ.
    Baseline is computed over the first `window` steps.
    """
    if len(values) <= window:
        return None
    baseline = values[:window]
    mu  = np.mean(baseline)
    sig = np.std(baseline) + 1e-8
    for i in range(window, len(values)):
        if values[i] > mu + sigma_thresh * sig:
            return i
    return None


# ═══════════════════════════════════════════════════════════════════════ #
# FINAL LEAKAGE SCORE (pixel-level)                                       #
# ═══════════════════════════════════════════════════════════════════════ #

def compute_final_leakage(image: Image.Image, prompt_cfg: dict) -> float:
    """
    Measures color leakage in the final decoded image.
    Splits image in half, checks if wrong colors appear in each half.
    Returns a normalized leakage score [0, 1].
    """
    img = np.array(image).astype(float) / 255.0   # [H, W, 3]
    H, W = img.shape[:2]

    left  = img[:, :W//2, :]
    right = img[:, W//2:, :]

    colors = prompt_cfg["colors"]

    # Define simple RGB anchors
    color_rgb = {
        "red":    np.array([1.0, 0.0, 0.0]),
        "blue":   np.array([0.0, 0.0, 1.0]),
        "green":  np.array([0.0, 0.8, 0.0]),
        "yellow": np.array([1.0, 1.0, 0.0]),
    }

    def region_color_affinity(region: np.ndarray, target: np.ndarray) -> float:
        mean_px = region.reshape(-1, 3).mean(axis=0)
        dot = np.dot(mean_px / (np.linalg.norm(mean_px) + 1e-8),
                     target / (np.linalg.norm(target) + 1e-8))
        return float(max(0.0, dot))

    # Color A should be left, Color B right (heuristic)
    if colors[0] in color_rgb and colors[1] in color_rgb:
        cA = color_rgb[colors[0]]
        cB = color_rgb[colors[1]]
        # Leakage = wrong color appearing in wrong region
        leak_A_in_right = region_color_affinity(right, cA)
        leak_B_in_left  = region_color_affinity(left,  cB)
        return (leak_A_in_right + leak_B_in_left) / 2.0

    return 0.0


# ═══════════════════════════════════════════════════════════════════════ #
# PLOTTING                                                                #
# ═══════════════════════════════════════════════════════════════════════ #

def plot_results(
    results:    List[dict],
    output_dir: Path,
    prompt_label: str,
):
    """
    Plots:
      1. CC(t) and AC(t) curves per seed, with mean ± std band
      2. Onset timestep distributions
      3. Intervention bar chart (if causal repair was run)
    """
    fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    text_color  = "#e6edf3"
    cc_color    = "#f97316"   # orange
    ac_color    = "#3b82f6"   # blue
    grid_color  = "#21262d"

    def style_ax(ax):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=text_color, labelsize=9)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.grid(color=grid_color, linestyle="--", linewidth=0.5, alpha=0.6)

    # ── Panel 1: CC + AC mean curves ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1)

    all_cc = np.array([r["cc_log"] for r in results if "cc_log" in r])
    all_ac = np.array([r["ac_log"] for r in results if "ac_log" in r])
    t_vals = results[0]["t_log"] if results else []

    if len(all_cc) > 0 and len(t_vals) > 0:
        steps = range(len(t_vals))
        cc_mean, cc_std = all_cc.mean(0), all_cc.std(0)
        ac_mean, ac_std = all_ac.mean(0), all_ac.std(0)

        ax1.plot(steps, cc_mean, color=cc_color, lw=2, label="CC(t) — Latent Chromatic Contamination")
        ax1.fill_between(steps, cc_mean - cc_std, cc_mean + cc_std, alpha=0.2, color=cc_color)

        ax1.plot(steps, ac_mean, color=ac_color, lw=2, label="AC(t) — Attention Corruption")
        ax1.fill_between(steps, ac_mean - ac_std, ac_mean + ac_std, alpha=0.2, color=ac_color)

        # Mark mean onsets
        for r in results:
            cc_onset = r.get("cc_onset")
            ac_onset = r.get("ac_onset")
            if cc_onset is not None:
                ax1.axvline(cc_onset, color=cc_color, linestyle=":", alpha=0.3, lw=1)
            if ac_onset is not None:
                ax1.axvline(ac_onset, color=ac_color, linestyle=":", alpha=0.3, lw=1)

        # Mean onset lines
        cc_onsets = [r["cc_onset"] for r in results if r.get("cc_onset") is not None]
        ac_onsets = [r["ac_onset"] for r in results if r.get("ac_onset") is not None]
        if cc_onsets:
            ax1.axvline(np.mean(cc_onsets), color=cc_color, lw=2.5,
                        label=f"Mean CC onset: step {np.mean(cc_onsets):.1f}")
        if ac_onsets:
            ax1.axvline(np.mean(ac_onsets), color=ac_color, lw=2.5,
                        label=f"Mean AC onset: step {np.mean(ac_onsets):.1f}")

    ax1.set_title(f"CC(t) vs AC(t)  —  {prompt_label}", fontsize=11, pad=10)
    ax1.set_xlabel("Denoising Step (t=1.0 → t=0.0)")
    ax1.set_ylabel("Score")
    ax1.legend(fontsize=8, facecolor="#161b22", labelcolor=text_color, framealpha=0.8)

    # ── Panel 2: Onset histogram ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2)

    cc_onsets = [r["cc_onset"] for r in results if r.get("cc_onset") is not None]
    ac_onsets = [r["ac_onset"] for r in results if r.get("ac_onset") is not None]

    if cc_onsets:
        ax2.hist(cc_onsets, bins=min(10, len(cc_onsets)),
                 color=cc_color, alpha=0.7, label="CC onset", edgecolor="#0d1117")
    if ac_onsets:
        ax2.hist(ac_onsets, bins=min(10, len(ac_onsets)),
                 color=ac_color, alpha=0.7, label="AC onset", edgecolor="#0d1117")

    ax2.set_title("Onset Step Distribution", fontsize=11, pad=10)
    ax2.set_xlabel("Step index at onset")
    ax2.set_ylabel("Count (seeds)")
    ax2.legend(fontsize=8, facecolor="#161b22", labelcolor=text_color, framealpha=0.8)

    # ── Panel 3: Per-seed onset comparison ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3)

    seeds = list(range(len(results)))
    cc_o  = [r.get("cc_onset") or 0 for r in results]
    ac_o  = [r.get("ac_onset") or 0 for r in results]

    x = np.arange(len(seeds))
    width = 0.35
    ax3.bar(x - width/2, cc_o, width, label="CC onset", color=cc_color, alpha=0.8)
    ax3.bar(x + width/2, ac_o, width, label="AC onset", color=ac_color, alpha=0.8)
    ax3.set_title("Per-Seed Onset Steps", fontsize=11, pad=10)
    ax3.set_xlabel("Seed index")
    ax3.set_ylabel("Step index")
    ax3.set_xticks(x)
    ax3.legend(fontsize=8, facecolor="#161b22", labelcolor=text_color, framealpha=0.8)

    # ── Panel 4: Final leakage comparison ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)

    baseline_leakage   = [r["final_leakage"] for r in results if not r.get("repaired")]
    repaired_leakage   = [r["final_leakage"] for r in results if r.get("repaired")]

    categories = []
    values     = []
    colors_bar = []
    if baseline_leakage:
        categories.append("Baseline")
        values.append(np.mean(baseline_leakage))
        colors_bar.append("#ef4444")
    if repaired_leakage:
        categories.append("After\nLatent Repair")
        values.append(np.mean(repaired_leakage))
        colors_bar.append("#22c55e")

    if categories:
        bars = ax4.bar(categories, values, color=colors_bar, alpha=0.85, edgecolor="#0d1117")
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", color=text_color, fontsize=9)

    ax4.set_title("Final Leakage: Baseline vs Repaired", fontsize=11, pad=10)
    ax4.set_ylabel("Leakage Score")
    ax4.set_ylim(0, max(values or [0.1]) * 1.3 + 0.01)

    # ── Panel 5: Causal chain summary ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5)
    ax5.axis("off")

    # Determine hypothesis verdict
    cc_mean_onset = np.mean(cc_onsets) if cc_onsets else None
    ac_mean_onset = np.mean(ac_onsets) if ac_onsets else None

    if cc_mean_onset is not None and ac_mean_onset is not None:
        if cc_mean_onset < ac_mean_onset:
            verdict     = "✓ SUPPORTED"
            verdict_col = "#22c55e"
            chain       = f"t_latent ({cc_mean_onset:.0f}) < t_attn ({ac_mean_onset:.0f})"
        elif cc_mean_onset > ac_mean_onset:
            verdict     = "✗ REFUTED"
            verdict_col = "#ef4444"
            chain       = f"t_attn ({ac_mean_onset:.0f}) < t_latent ({cc_mean_onset:.0f})"
        else:
            verdict     = "~ INCONCLUSIVE"
            verdict_col = "#f59e0b"
            chain       = "CC onset ≈ AC onset"
    else:
        verdict     = "~ INSUFFICIENT DATA"
        verdict_col = "#f59e0b"
        chain       = "Could not detect reliable onsets"

    summary = (
        f"HYPOTHESIS:\n"
        f"t_latent < t_attn < t_final\n\n"
        f"RESULT:\n"
        f"{chain}\n\n"
        f"VERDICT:\n"
        f"{verdict}\n\n"
    )
    if baseline_leakage and repaired_leakage:
        reduction = (np.mean(baseline_leakage) - np.mean(repaired_leakage)) / (np.mean(baseline_leakage) + 1e-8)
        summary += f"Causal repair reduced\nfinal leakage by {reduction*100:.1f}%"

    ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
             fontsize=9.5, verticalalignment="top",
             color=text_color, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#1c2128", edgecolor=verdict_col, lw=2))
    ax5.text(0.5, 0.12, verdict, transform=ax5.transAxes,
             fontsize=14, ha="center", color=verdict_col, fontweight="bold")

    plt.suptitle("Color Leakage: Latent Contamination vs Attention Corruption",
                 fontsize=13, color=text_color, y=0.98)

    out_path = output_dir / "color_leakage_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"[Plot] Saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════ #
# MAIN                                                                    #
# ═══════════════════════════════════════════════════════════════════════ #

def run_single_seed(
    wrapper:       SD3PipelineWrapper,
    prompt_cfg:    dict,
    seed:          int,
    output_dir:    Path,
    cfg:           dict,
    do_repair:     bool = False,
    repair_step:   Optional[int] = None,
) -> dict:

    set_seed(seed)
    device = wrapper.device

    # Encode prompt
    prompt_embeds, pooled_embeds = wrapper.encode_prompt(
        prompt          = prompt_cfg["prompt"],
        negative_prompt = prompt_cfg.get("neg", ""),
    )

    # Initial latents
    latents = wrapper.get_initial_latents(seed=seed)
    H_lat   = latents.shape[2]
    W_lat   = latents.shape[3]

    # Find token indices
    token_idx = find_token_indices(
        wrapper.tokenizer,
        prompt_cfg["prompt"],
        prompt_cfg["objects"] + prompt_cfg["colors"],
    )
    print(f"  [Tokens] {token_idx}")

    # Build attention hook manager
    hook_mgr = AttentionHookManager(wrapper.transformer)
    hook_mgr.register()

    # Build instrumented loop
    loop = InstrumentedFlowLoop(
        unet              = wrapper.transformer,
        scheduler         = wrapper.scheduler,
        cfg               = cfg,
        device            = device,
        hook_manager      = hook_mgr,
        token_config      = token_idx,
        prompt_config     = prompt_cfg,
        latent_hw         = (H_lat, W_lat),
        do_causal_repair  = do_repair,
        repair_at_step    = repair_step,
    )

    # Run
    result = loop.run(
        latents           = latents,
        text_embeddings   = prompt_embeds,
        pooled_embeddings = pooled_embeds,
    )

    hook_mgr.remove()

    # Decode image
    image = wrapper.decode_latents(result["latents"])
    suffix = "_repaired" if do_repair else ""
    img_path = output_dir / f"seed{seed:03d}{suffix}.png"
    image.save(img_path)

    # Detect onsets
    cc_onset = detect_onset(loop.cc_log)
    ac_onset = detect_onset(loop.ac_log)

    # Final leakage
    final_leakage = compute_final_leakage(image, prompt_cfg)

    print(f"  [Seed {seed}] CC onset={cc_onset} | AC onset={ac_onset} | "
          f"final_leakage={final_leakage:.4f}")

    if cc_onset is not None and ac_onset is not None:
        if cc_onset < ac_onset:
            print(f"  → ✓ Latent contamination precedes attention corruption "
                  f"(Δ={ac_onset - cc_onset} steps)")
        else:
            print(f"  → ✗ Attention corruption does NOT follow latent onset")

    return {
        "seed":          seed,
        "cc_log":        loop.cc_log,
        "ac_log":        loop.ac_log,
        "t_log":         loop.t_log,
        "cc_onset":      cc_onset,
        "ac_onset":      ac_onset,
        "final_leakage": final_leakage,
        "repaired":      do_repair,
        "image_path":    str(img_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str,  default="config.yaml")
    parser.add_argument("--output_dir",   type=str,  default="results/leakage/")
    parser.add_argument("--seeds",        type=int,  default=3,
                        help="Number of seeds per prompt (5 recommended, 3 for fast test)")
    parser.add_argument("--prompt_idx",   type=int,  default=0,
                        help="Which prompt from COLOR_PROMPTS to use (0 or 1)")
    parser.add_argument("--causal_repair",action="store_true",
                        help="Also run with chromatic repair to test causality")
    parser.add_argument("--repair_step",  type=int,  default=None,
                        help="Step at which to apply latent repair (default: auto=CC onset)")
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_cfg = COLOR_PROMPTS[args.prompt_idx]
    print(f"\n{'='*60}")
    print(f"Experiment: Color Leakage Temporal Analysis")
    print(f"Prompt    : {prompt_cfg['prompt']}")
    print(f"Seeds     : {args.seeds}")
    print(f"Causal repair: {args.causal_repair}")
    print(f"{'='*60}\n")

    # Load pipeline once
    wrapper = SD3PipelineWrapper(cfg, device=args.device)
    wrapper.load()

    # ── Phase 1: Baseline runs ────────────────────────────────────────
    print("\n[Phase 1] Baseline runs (no intervention)")
    baseline_results = []
    for s in range(args.seeds):
        print(f"\n  Seed {s}")
        r = run_single_seed(wrapper, prompt_cfg, seed=s,
                            output_dir=output_dir, cfg=cfg, do_repair=False)
        baseline_results.append(r)

    # Determine mean CC onset for repair step
    cc_onsets = [r["cc_onset"] for r in baseline_results if r["cc_onset"] is not None]
    mean_cc_onset = int(np.mean(cc_onsets)) if cc_onsets else 10
    repair_step   = args.repair_step if args.repair_step is not None else mean_cc_onset
    print(f"\n[Info] Mean CC onset = {mean_cc_onset}  →  repair_step = {repair_step}")

    all_results = baseline_results.copy()

    # ── Phase 2 (optional): Causal repair runs ────────────────────────
    if args.causal_repair:
        print(f"\n[Phase 2] Causal repair runs (repair at step {repair_step})")
        for s in range(args.seeds):
            print(f"\n  Seed {s} (repaired)")
            r = run_single_seed(wrapper, prompt_cfg, seed=s,
                                output_dir=output_dir, cfg=cfg,
                                do_repair=True, repair_step=repair_step)
            all_results.append(r)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    cc_o = [r["cc_onset"] for r in baseline_results if r["cc_onset"] is not None]
    ac_o = [r["ac_onset"] for r in baseline_results if r["ac_onset"] is not None]
    fl   = [r["final_leakage"] for r in baseline_results]

    if cc_o and ac_o:
        print(f"Mean CC onset step   : {np.mean(cc_o):.1f} ± {np.std(cc_o):.1f}")
        print(f"Mean AC onset step   : {np.mean(ac_o):.1f} ± {np.std(ac_o):.1f}")
        verdict = "✓ HYPOTHESIS SUPPORTED" if np.mean(cc_o) < np.mean(ac_o) else "✗ HYPOTHESIS REFUTED"
        print(f"Temporal ordering    : {verdict}")
    print(f"Mean final leakage   : {np.mean(fl):.4f}")

    if args.causal_repair:
        fl_rep = [r["final_leakage"] for r in all_results if r.get("repaired")]
        if fl_rep:
            reduction = (np.mean(fl) - np.mean(fl_rep)) / (np.mean(fl) + 1e-8)
            print(f"Leakage after repair : {np.mean(fl_rep):.4f}  "
                  f"(reduction: {reduction*100:.1f}%)")
            if reduction > 0.1:
                print("✓ CAUSAL INTERVENTION EFFECTIVE — latent is upstream cause")
            else:
                print("✗ Repair did not reduce leakage — latent may not be causal")

    # ── Save results ──────────────────────────────────────────────────
    plot_path = plot_results(all_results, output_dir, prompt_cfg["prompt"][:40])

    results_json = output_dir / "results.json"
    serializable = []
    for r in all_results:
        serializable.append({k: v for k, v in r.items()
                              if isinstance(v, (str, int, float, bool, list, type(None)))})
    with open(results_json, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n[Done] Results → {output_dir}")
    print(f"       Plot    → {plot_path}")
    print(f"       JSON    → {results_json}")


if __name__ == "__main__":
    main()