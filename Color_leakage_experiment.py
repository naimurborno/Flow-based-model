"""
color_leakage_experiment.py  (v2 — surgical attention extraction)
------------------------------------------------------------------
Tests: t_latent_contamination < t_attention_corruption < t_final_leakage

Key change from v1:
    AttentionSurgeon monkeypatches every MMDiT joint-attention processor
    and computes attn_probs = softmax(Q @ K^T / √d) explicitly,
    saving the image→text cross-attention slice [B, heads, N_img, N_txt].
    This gives REAL spatial maps → REAL masks → REAL AC scores.

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

from pipeline_wrapper  import SD3PipelineWrapper
from custom_flow_loop  import FlowMatchingLoop
from utils             import load_config, set_seed, save_results
from attention_surgery import AttentionSurgeon, find_token_positions


# ═══════════════════════════════════════════════════════════════════════ #
#  EXPERIMENT CONFIG                                                      #
# ═══════════════════════════════════════════════════════════════════════ #

COLOR_PROMPTS = [
    {
        "prompt":   "a red cube and a blue sphere on a white table",
        "neg":      "blurry, low quality",
        "objects":  ["cube", "sphere"],
        "colors":   ["red", "blue"],
        "target_colors": {
            "red":  torch.tensor([ 0.45, -0.30]),
            "blue": torch.tensor([-0.35,  0.40]),
        },
    },
    {
        "prompt":   "a green apple and a yellow banana on a wooden table",
        "neg":      "blurry, low quality",
        "objects":  ["apple", "banana"],
        "colors":   ["green", "yellow"],
        "target_colors": {
            "green":  torch.tensor([ 0.30,  0.25]),
            "yellow": torch.tensor([ 0.40, -0.10]),
        },
    },
]

# SD3 VAE: 16 latent channels. Channels 8-9 carry chromatic information.
CHROMA_PAIR = [8, 9]


# ═══════════════════════════════════════════════════════════════════════ #
#  LATENT CHROMATIC CONTAMINATION  (Hook A)                              #
# ═══════════════════════════════════════════════════════════════════════ #

def measure_chromatic_contamination(
    latents  : torch.Tensor,
    mask_A   : torch.Tensor,
    mask_B   : torch.Tensor,
    color_A  : torch.Tensor,
    color_B  : torch.Tensor,
) -> Tuple[float, float, float]:
    device = latents.device
    c      = latents[0, CHROMA_PAIR, :, :].float()
    H, W   = c.shape[1], c.shape[2]

    def region_chroma(mask):
        m = mask.to(device)
        if m.shape != (H, W):
            m = F.interpolate(m.unsqueeze(0).unsqueeze(0), (H, W), mode="nearest")[0, 0]
        if m.sum() < 1:
            return None
        return (c * m.unsqueeze(0)).sum(dim=(1, 2)) / (m.sum() + 1e-8)

    sig_A = region_chroma(mask_A)
    sig_B = region_chroma(mask_B)
    if sig_A is None or sig_B is None:
        return 0.0, 0.0, 0.0

    def cos(a, b):
        return float(F.cosine_similarity(
            a.unsqueeze(0).to(device),
            b.unsqueeze(0).to(device),
        ).item())

    lAB = max(0.0, cos(sig_B, color_A.to(device)))
    lBA = max(0.0, cos(sig_A, color_B.to(device)))
    return lAB, lBA, (lAB + lBA) / 2.0


# ═══════════════════════════════════════════════════════════════════════ #
#  ATTENTION CORRUPTION  (Hook B)                                         #
# ═══════════════════════════════════════════════════════════════════════ #

def measure_attention_corruption(
    surgeon     : AttentionSurgeon,
    step        : int,
    tok_color_A : int,
    tok_color_B : int,
    tok_noun_A  : int,
    tok_noun_B  : int,
    latent_hw   : Tuple[int, int],
    threshold   : float = 0.35,
) -> float:
    def binary(tok):
        m = surgeon.get_token_spatial_map(step, tok, latent_hw)
        return None if m is None else (m > threshold).float()

    nA = binary(tok_noun_A)
    nB = binary(tok_noun_B)
    cA = binary(tok_color_A)
    cB = binary(tok_color_B)

    if any(m is None for m in [nA, nB, cA, cB]):
        return 0.0

    def iou(a, b):
        return float((a * b).sum() / (((a + b) > 0).float().sum() + 1e-8))

    return (iou(cA, nB) + iou(cB, nA)) / 2.0


# ═══════════════════════════════════════════════════════════════════════ #
#  CAUSAL REPAIR                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

def repair_chromatic_contamination(
    latents, mask_A, mask_B, color_A, color_B, strength=0.7
):
    repaired = latents.clone()
    device   = latents.device
    H, W     = latents.shape[2], latents.shape[3]

    for mask, target in [(mask_A, color_A), (mask_B, color_B)]:
        m = mask.to(device)
        if m.shape != (H, W):
            m = F.interpolate(m.unsqueeze(0).unsqueeze(0), (H, W), mode="nearest")[0, 0]
        if m.sum() < 1:
            continue
        for i, ch in enumerate(CHROMA_PAIR):
            plane      = repaired[0, ch, :, :].float()
            r_mean     = (plane * m).sum() / (m.sum() + 1e-8)
            target_val = target[i].to(device) * plane.abs().mean()
            repaired[0, ch, :, :] = (
                plane + m * (target_val - r_mean) * strength
            ).to(latents.dtype)

    return repaired


# ═══════════════════════════════════════════════════════════════════════ #
#  ONSET DETECTION                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def detect_onset(values, window=5, sigma=2.0):
    if len(values) <= window:
        return None
    mu, std = np.mean(values[:window]), np.std(values[:window]) + 1e-8
    for i in range(window, len(values)):
        if values[i] > mu + sigma * std:
            return i
    return None


# ═══════════════════════════════════════════════════════════════════════ #
#  INSTRUMENTED FLOW LOOP                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

class InstrumentedFlowLoop(FlowMatchingLoop):

    def __init__(self, *args, surgeon, token_positions, prompt_cfg,
                 latent_hw, do_repair=False, repair_at_step=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.surgeon         = surgeon
        self.token_positions = token_positions
        self.prompt_cfg      = prompt_cfg
        self.latent_hw       = latent_hw
        self.do_repair       = do_repair
        self.repair_at_step  = repair_at_step
        self.cc_log: List[float] = []
        self.ac_log: List[float] = []
        self.t_log:  List[float] = []

    # Wrap velocity forward to bracket with capture on/off
    def _velocity_forward(self, latent_input, t_batch, text_emb, pooled_emb=None):
        self.surgeon.start_capture()
        out = super()._velocity_forward(latent_input, t_batch, text_emb, pooled_emb)
        self.surgeon.stop_capture()
        return out

    def _step_callback(self, latents, t, step_idx, velocity):
        t_val = float(t.item()) if hasattr(t, "item") else float(t)
        self.t_log.append(t_val)
        self.surgeon.set_step(step_idx)

        pc, tp, hw = self.prompt_cfg, self.token_positions, self.latent_hw

        # Get region masks from real attention maps
        tok_nA = tp.get(pc["objects"][0])
        tok_nB = tp.get(pc["objects"][1])
        mask_A = surgeon_mask(self.surgeon, step_idx, tok_nA, hw)
        mask_B = surgeon_mask(self.surgeon, step_idx, tok_nB, hw)

        # Fallback: spatial split
        H, W = hw
        if mask_A is None:
            mask_A = torch.zeros(H, W); mask_A[:, :W//2] = 1.0
        if mask_B is None:
            mask_B = torch.zeros(H, W); mask_B[:, W//2:] = 1.0

        color_A = pc["target_colors"][pc["colors"][0]]
        color_B = pc["target_colors"][pc["colors"][1]]

        # Hook A
        _, _, cc = measure_chromatic_contamination(latents, mask_A, mask_B, color_A, color_B)
        self.cc_log.append(cc)

        # Hook B
        tok_cA = tp.get(pc["colors"][0])
        tok_cB = tp.get(pc["colors"][1])
        if all(v is not None for v in [tok_cA, tok_cB, tok_nA, tok_nB]):
            ac = measure_attention_corruption(
                self.surgeon, step_idx,
                tok_cA, tok_cB, tok_nA, tok_nB, hw
            )
        else:
            ac = 0.0
        self.ac_log.append(ac)

        if step_idx % 10 == 0:
            self.surgeon.diagnose(step_idx)
            print(f"  [step {step_idx:3d} t={t_val:.3f}] CC={cc:.4f}  AC={ac:.4f}")

        if self.do_repair and self.repair_at_step == step_idx:
            print(f"  [REPAIR] step={step_idx}")
            latents = repair_chromatic_contamination(latents, mask_A, mask_B, color_A, color_B)

        self.surgeon.clear_step(step_idx)
        return latents


def surgeon_mask(surgeon, step, tok_idx, hw, threshold=0.35):
    if tok_idx is None:
        return None
    m = surgeon.get_token_spatial_map(step, tok_idx, hw)
    return None if m is None else (m > threshold).float()


# ═══════════════════════════════════════════════════════════════════════ #
#  FINAL LEAKAGE                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

def compute_final_leakage(image, prompt_cfg):
    img = np.array(image).astype(float) / 255.0
    H, W = img.shape[:2]
    left, right = img[:, :W//2], img[:, W//2:]
    cmap = {"red": [1,0,0], "blue": [0,0,1], "green": [0,.8,0], "yellow": [1,1,0]}

    def aff(region, target):
        m = region.reshape(-1, 3).mean(0)
        t = np.array(target, dtype=float)
        return float(max(0, np.dot(m/(np.linalg.norm(m)+1e-8), t/(np.linalg.norm(t)+1e-8))))

    c = prompt_cfg["colors"]
    if c[0] in cmap and c[1] in cmap:
        return (aff(right, cmap[c[0]]) + aff(left, cmap[c[1]])) / 2.0
    return 0.0


# ═══════════════════════════════════════════════════════════════════════ #
#  PLOTTING                                                               #
# ═══════════════════════════════════════════════════════════════════════ #

def plot_results(results, output_dir, label):
    fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)
    txt, orc, blu, grd = "#e6edf3", "#f97316", "#3b82f6", "#21262d"

    def sax(ax):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=txt, labelsize=9)
        for l in [ax.xaxis.label, ax.yaxis.label, ax.title]: l.set_color(txt)
        for s in ax.spines.values(): s.set_color(grd)
        ax.grid(color=grd, ls="--", lw=0.5, alpha=0.6)

    base = [r for r in results if not r.get("repaired")]
    rep  = [r for r in results if r.get("repaired")]

    ax1 = fig.add_subplot(gs[0, :2]); sax(ax1)
    if base:
        s   = range(len(base[0]["cc_log"]))
        cc  = np.array([r["cc_log"] for r in base])
        ac  = np.array([r["ac_log"] for r in base])
        ccm, ccs = cc.mean(0), cc.std(0)
        acm, acs = ac.mean(0), ac.std(0)
        ax1.plot(s, ccm, color=orc, lw=2, label="CC(t) — Latent Contamination")
        ax1.fill_between(s, ccm-ccs, ccm+ccs, alpha=0.18, color=orc)
        ax1.plot(s, acm, color=blu, lw=2, label="AC(t) — Attention Corruption")
        ax1.fill_between(s, acm-acs, acm+acs, alpha=0.18, color=blu)
        cco = [r["cc_onset"] for r in base if r["cc_onset"] is not None]
        aco = [r["ac_onset"] for r in base if r["ac_onset"] is not None]
        if cco: ax1.axvline(np.mean(cco), color=orc, lw=2.5, label=f"CC onset ≈{np.mean(cco):.0f}")
        if aco: ax1.axvline(np.mean(aco), color=blu, lw=2.5, label=f"AC onset ≈{np.mean(aco):.0f}")
    ax1.set_title(f"CC(t) vs AC(t) — {label}", fontsize=11, pad=10)
    ax1.set_xlabel("Denoising Step"); ax1.set_ylabel("Score")
    ax1.legend(fontsize=8, facecolor="#161b22", labelcolor=txt, framealpha=0.8)

    ax2 = fig.add_subplot(gs[0, 2]); sax(ax2)
    cco = [r["cc_onset"] for r in base if r["cc_onset"] is not None]
    aco = [r["ac_onset"] for r in base if r["ac_onset"] is not None]
    if cco: ax2.hist(cco, bins=min(8, len(cco)), color=orc, alpha=0.7, label="CC", ec="#0d1117")
    if aco: ax2.hist(aco, bins=min(8, len(aco)), color=blu, alpha=0.7, label="AC", ec="#0d1117")
    ax2.set_title("Onset Distribution", fontsize=11, pad=10)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Count")
    ax2.legend(fontsize=8, facecolor="#161b22", labelcolor=txt)

    ax3 = fig.add_subplot(gs[1, 0]); sax(ax3)
    x, w = np.arange(len(base)), 0.35
    ax3.bar(x-w/2, [r.get("cc_onset") or 0 for r in base], w, color=orc, alpha=0.8, label="CC")
    ax3.bar(x+w/2, [r.get("ac_onset") or 0 for r in base], w, color=blu, alpha=0.8, label="AC")
    ax3.set_title("Per-Seed Onset Steps", fontsize=11, pad=10)
    ax3.set_xlabel("Seed"); ax3.set_ylabel("Step")
    ax3.legend(fontsize=8, facecolor="#161b22", labelcolor=txt)

    ax4 = fig.add_subplot(gs[1, 1]); sax(ax4)
    cats, vals, bcols = [], [], []
    if base:
        cats.append("Baseline"); vals.append(np.mean([r["final_leakage"] for r in base])); bcols.append("#ef4444")
    if rep:
        cats.append("After\nRepair"); vals.append(np.mean([r["final_leakage"] for r in rep])); bcols.append("#22c55e")
    if cats:
        bars = ax4.bar(cats, vals, color=bcols, alpha=0.85, ec="#0d1117")
        for b, v in zip(bars, vals):
            ax4.text(b.get_x()+b.get_width()/2, v+0.003, f"{v:.3f}", ha="center", color=txt, fontsize=9)
    ax4.set_title("Final Leakage Score", fontsize=11, pad=10)
    ax4.set_ylabel("Leakage")
    ax4.set_ylim(0, max(vals or [0.1])*1.35+0.01)

    ax5 = fig.add_subplot(gs[1, 2]); sax(ax5); ax5.axis("off")
    cco = [r["cc_onset"] for r in base if r["cc_onset"] is not None]
    aco = [r["ac_onset"] for r in base if r["ac_onset"] is not None]
    if cco and aco:
        cm, am = np.mean(cco), np.mean(aco)
        if cm < am:   verd, vc, chain = "✓ SUPPORTED", "#22c55e", f"t_latent({cm:.0f}) < t_attn({am:.0f})"
        elif cm > am: verd, vc, chain = "✗ REFUTED",   "#ef4444", f"t_attn({am:.0f}) < t_latent({cm:.0f})"
        else:         verd, vc, chain = "~ INCONCLUSIVE","#f59e0b","onset tied"
    else:
        verd, vc, chain = "~ NO DATA", "#f59e0b", "—"

    causal = ""
    if base and rep:
        bl, rp = np.mean([r["final_leakage"] for r in base]), np.mean([r["final_leakage"] for r in rep])
        causal = f"\nRepair reduced leakage\nby {(bl-rp)/(bl+1e-8)*100:.1f}%"

    ax5.text(0.05, 0.95,
             f"HYPOTHESIS\nt_latent < t_attn < t_final\n\nCHAIN:\n{chain}\n\nVERDICT:\n{verd}{causal}",
             transform=ax5.transAxes, fontsize=9.5, va="top", color=txt, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#1c2128", ec=vc, lw=2))

    plt.suptitle("Color Leakage: Latent Contamination vs Attention Corruption",
                 fontsize=13, color=txt, y=0.98)
    out = output_dir / "color_leakage_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[Plot] → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════ #
#  SINGLE-SEED RUN                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def run_single_seed(wrapper, prompt_cfg, seed, output_dir, cfg,
                    do_repair=False, repair_step=None):
    set_seed(seed)
    device = wrapper.device

    prompt_embeds, pooled_embeds = wrapper.encode_prompt(
        prompt=prompt_cfg["prompt"], negative_prompt=prompt_cfg.get("neg", "")
    )
    latents      = wrapper.get_initial_latents(seed=seed)
    H_lat, W_lat = latents.shape[2], latents.shape[3]

    all_words = prompt_cfg["objects"] + prompt_cfg["colors"]
    tok_pos   = find_token_positions(wrapper.tokenizer, prompt_cfg["prompt"], all_words)
    print(f"  [Tokens] {tok_pos}")

    surgeon = AttentionSurgeon(wrapper.transformer)
    surgeon.install()

    loop = InstrumentedFlowLoop(
        unet            = wrapper.transformer,
        scheduler       = wrapper.scheduler,
        cfg             = cfg,
        device          = device,
        surgeon         = surgeon,
        token_positions = tok_pos,
        prompt_cfg      = prompt_cfg,
        latent_hw       = (H_lat, W_lat),
        do_repair       = do_repair,
        repair_at_step  = repair_step,
    )

    result = loop.run(
        latents=latents, text_embeddings=prompt_embeds, pooled_embeddings=pooled_embeds
    )
    surgeon.uninstall()

    image    = wrapper.decode_latents(result["latents"])
    suffix   = "_repaired" if do_repair else ""
    img_path = output_dir / f"seed{seed:03d}{suffix}.png"
    image.save(img_path)

    cc_onset = detect_onset(loop.cc_log)
    ac_onset = detect_onset(loop.ac_log)
    final_lk = compute_final_leakage(image, prompt_cfg)

    print(f"  [Seed {seed}{'R' if do_repair else ' '}] "
          f"CC_onset={cc_onset} | AC_onset={ac_onset} | leak={final_lk:.4f}")
    if cc_onset is not None and ac_onset is not None:
        d = ac_onset - cc_onset
        print(f"    Δ={d:+d}  → {'✓ latent first' if d > 0 else '✗ attn first'}")

    return dict(seed=seed, cc_log=loop.cc_log, ac_log=loop.ac_log, t_log=loop.t_log,
                cc_onset=cc_onset, ac_onset=ac_onset, final_leakage=final_lk,
                repaired=do_repair, image_path=str(img_path))


# ═══════════════════════════════════════════════════════════════════════ #
#  MAIN                                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        default="config.yaml")
    p.add_argument("--output_dir",    default="results/leakage/")
    p.add_argument("--seeds",         type=int, default=3)
    p.add_argument("--prompt_idx",    type=int, default=0)
    p.add_argument("--causal_repair", action="store_true")
    p.add_argument("--repair_step",   type=int, default=None)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg        = load_config(args.config)
    out        = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    prompt_cfg = COLOR_PROMPTS[args.prompt_idx]

    print(f"\n{'='*60}\nPrompt: {prompt_cfg['prompt']}\nSeeds: {args.seeds}\n{'='*60}\n")

    wrapper = SD3PipelineWrapper(cfg, device=args.device)
    wrapper.load()

    # Phase 1: baseline
    print("[Phase 1] Baseline runs")
    baseline = [run_single_seed(wrapper, prompt_cfg, s, out, cfg) for s in range(args.seeds)]

    cco        = [r["cc_onset"] for r in baseline if r["cc_onset"] is not None]
    rep_step   = args.repair_step or (int(np.mean(cco)) if cco else 10)
    all_results = baseline.copy()

    # Phase 2: causal repair
    if args.causal_repair:
        print(f"\n[Phase 2] Repair at step {rep_step}")
        for s in range(args.seeds):
            all_results.append(
                run_single_seed(wrapper, prompt_cfg, s, out, cfg,
                                do_repair=True, repair_step=rep_step)
            )

    # Summary
    print(f"\n{'='*60}\nSUMMARY")
    cco = [r["cc_onset"] for r in baseline if r["cc_onset"] is not None]
    aco = [r["ac_onset"] for r in baseline if r["ac_onset"] is not None]
    fl  = [r["final_leakage"] for r in baseline]
    if cco and aco:
        print(f"CC onset: {np.mean(cco):.1f} ± {np.std(cco):.1f}")
        print(f"AC onset: {np.mean(aco):.1f} ± {np.std(aco):.1f}")
        print("VERDICT :", "✓ SUPPORTED" if np.mean(cco) < np.mean(aco) else "✗ REFUTED")
    print(f"Leakage : {np.mean(fl):.4f}")
    if args.causal_repair:
        fl_r = [r["final_leakage"] for r in all_results if r.get("repaired")]
        if fl_r:
            print(f"Post-repair: {np.mean(fl_r):.4f}  "
                  f"(↓{(np.mean(fl)-np.mean(fl_r))/(np.mean(fl)+1e-8)*100:.1f}%)")

    plot_results(all_results, out, prompt_cfg["prompt"][:45])
    with open(out / "results.json", "w") as f:
        json.dump([{k: v for k, v in r.items()
                    if isinstance(v, (str, int, float, bool, list, type(None)))}
                   for r in all_results], f, indent=2)
    print(f"\n[Done] → {out}")


if __name__ == "__main__":
    main()