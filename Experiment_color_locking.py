"""
Experiment 4 — Premature Color-Locking
Claim: Color subspace concentrates (locks) earlier in ODE than shape/texture,
       BEFORE spatial layout has resolved (high spatial entropy).
Two measurements per timestep:
  1. Subspace concentration: effective rank of ΔH (low rank = locked)
  2. Spatial entropy: entropy of ||h||_2 map (high entropy = no boundaries yet)
If color rank drops while spatial entropy is still high → claim confirmed.
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_wrapper import SD3PipelineWrapper
from utils import load_config, set_seed

# ── Prompts ──────────────────────────────────────────────────────────────────
PROMPTS = {
    "color": [
        ("red car next to blue bicycle",        "car next to bicycle",        "car", "bicycle"),
        ("yellow banana beside green apple",     "banana beside apple",        "banana", "apple"),
        ("red house beside blue barn",           "house beside barn",          "house", "barn"),
        ("orange cat next to purple dog",        "cat next to dog",            "cat", "dog"),
        ("pink shirt beside black jacket",       "shirt beside jacket",        "shirt", "jacket"),
        ("blue boat next to red lighthouse",     "boat next to lighthouse",    "boat", "lighthouse"),
        ("green frog beside yellow duck",        "frog beside duck",           "frog", "duck"),
        ("red apple beside blue cup",            "apple beside cup",           "apple", "cup"),
    ],
    "shape": [
        ("round table next to square chair",     "table next to chair",        "table", "chair"),
        ("oval mirror beside rectangular door",  "mirror beside door",         "mirror", "door"),
        ("circular clock next to triangular sign","clock next to sign",        "clock", "sign"),
        ("curved sofa beside angular desk",      "sofa beside desk",           "sofa", "desk"),
        ("round bowl next to square box",        "bowl next to box",           "bowl", "box"),
        ("spherical lamp beside flat screen",    "lamp beside screen",         "lamp", "screen"),
        ("cylindrical vase next to flat book",   "vase next to book",          "vase", "book"),
        ("round wheel beside square window",     "wheel beside window",        "wheel", "window"),
    ],
    "texture": [
        ("metallic car next to wooden bicycle",  "car next to bicycle",        "car", "bicycle"),
        ("glossy table beside rusty chair",      "table beside chair",         "table", "chair"),
        ("smooth stone next to rough brick",     "stone next to brick",        "stone", "brick"),
        ("silky curtain beside rough carpet",    "curtain beside carpet",      "curtain", "carpet"),
        ("shiny metal bowl beside clay pot",     "bowl beside pot",            "bowl", "pot"),
        ("velvet sofa beside wooden desk",       "sofa beside desk",           "sofa", "desk"),
        ("granite counter beside plastic shelf", "counter beside shelf",       "counter", "shelf"),
        ("woven basket beside glass vase",       "basket beside vase",         "basket", "vase"),
    ],
}

WATCH_BLOCKS   = [8, 10, 12, 14, 16]
WATCH_STEPS    = list(range(0, 50, 5))   # every 5th step → 10 timesteps
SMOKE_STEPS    = [0, 10, 25, 40, 49]
SMOKE_PROMPTS  = 2


# ── Geometry helpers ─────────────────────────────────────────────────────────

def effective_rank(matrix: torch.Tensor) -> float:
    """Effective rank via singular value entropy. Low = concentrated/locked."""
    flat = matrix.reshape(-1, matrix.shape[-1]).float()
    if flat.shape[0] < 2:
        return 1.0
    _, S, _ = torch.linalg.svd(flat, full_matrices=False)
    S = S[S > 1e-8]
    if len(S) == 0:
        return 1.0
    p = S / S.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())


def spatial_entropy(hidden: torch.Tensor) -> float:
    """Entropy of per-token norm map. High = uniform = no boundaries."""
    norms = hidden.float().norm(dim=-1).squeeze()   # [D]
    norms = norms / (norms.sum() + 1e-8)
    return float(-(norms * torch.log(norms + 1e-12)).sum().item())


# ── Hook infrastructure ───────────────────────────────────────────────────────

class HiddenStateCapture:
    def __init__(self, transformer, watch_blocks):
        self.captures = {}   # block_idx → tensor
        self._hooks   = []
        for idx in watch_blocks:
            block = transformer.transformer_blocks[idx]
            h = block.register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)

    def _make_hook(self, idx):
        def hook(module, inp, out):
            t = out[1] if (isinstance(out, tuple) and len(out) > 1 and out[1] is not None) else \
                out[0] if isinstance(out, tuple) else out
            self.captures[idx] = t.detach().cpu()
        return hook

    def clear(self):
        self.captures.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()


# ── Single forward pass ───────────────────────────────────────────────────────

def forward_pass(wrapper, prompt, t_val, latents, seed=42):
    set_seed(seed)
    emb, pooled = wrapper.encode_prompt(prompt, "")

    scheduler = wrapper.scheduler
    scheduler.set_timesteps(50)
    timesteps = scheduler.timesteps

    # find closest timestep
    diffs = [(abs(tt.item() - t_val), i) for i, tt in enumerate(timesteps)]
    step_idx = min(diffs)[1]
    t = timesteps[step_idx]

    t_batch = t.reshape(1).to(wrapper.device)
    lat = latents.to(wrapper.device)
    emb = emb[1:2]        # cond only (no CFG)
    pooled = pooled[1:2]

    device = next(wrapper.transformer.parameters()).device
    dtype  = next(wrapper.transformer.parameters()).dtype

    with torch.no_grad():
        wrapper.transformer(
            hidden_states         = lat.to(device=device, dtype=dtype),
            timestep              = t_batch.to(device=device, dtype=dtype),
            encoder_hidden_states = emb.to(device=device, dtype=dtype),
            pooled_projections    = pooled.to(device=device, dtype=dtype),
        )
    return t.item()


# ── Main experiment ───────────────────────────────────────────────────────────

def run(args):
    cfg = load_config(args.config)
    cfg["flow"]["guidance_scale"] = 1.0
    wrapper = SD3PipelineWrapper(cfg, device=args.device)
    wrapper.load()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scheduler = wrapper.scheduler
    scheduler.set_timesteps(50)
    timesteps_vals = [t.item() for t in scheduler.timesteps]

    watch_steps = SMOKE_STEPS if args.smoke_test else WATCH_STEPS

    # results[attr][step_idx] = {rank: [], spatial_entropy: []}
    results = {
        attr: {s: {"rank": [], "spatial_entropy": []} for s in watch_steps}
        for attr in PROMPTS
    }

    capture = HiddenStateCapture(wrapper.transformer, WATCH_BLOCKS)

    for attr, prompt_list in PROMPTS.items():
        n = SMOKE_PROMPTS if args.smoke_test else len(prompt_list)
        print(f"\n── {attr.upper()} ({n} prompts) ──")

        for pidx, (with_p, without_p, _, _) in enumerate(prompt_list[:n]):
            print(f"  [{pidx+1}/{n}] {with_p}")

            latents = wrapper.get_initial_latents(seed=42 + pidx)

            for step_idx in watch_steps:
                t_val = timesteps_vals[step_idx]

                # pass WITH attributes
                capture.clear()
                forward_pass(wrapper, with_p, t_val, latents, seed=42)
                h_with = {k: v.clone() for k, v in capture.captures.items()}

                # pass WITHOUT attributes
                capture.clear()
                forward_pass(wrapper, without_p, t_val, latents, seed=42)
                h_without = {k: v.clone() for k, v in capture.captures.items()}

                if not h_with or not h_without:
                    continue

                # aggregate over watched blocks
                rank_vals = []
                entropy_vals = []
                for b in WATCH_BLOCKS:
                    if b not in h_with or b not in h_without:
                        continue
                    dH = h_with[b] - h_without[b]   # [1, D, C]
                    rank_vals.append(effective_rank(dH.squeeze(0)))
                    entropy_vals.append(spatial_entropy(h_with[b].squeeze(0)))

                if rank_vals:
                    results[attr][step_idx]["rank"].append(np.mean(rank_vals))
                    results[attr][step_idx]["spatial_entropy"].append(np.mean(entropy_vals))

    capture.remove()

    # ── Aggregate ────────────────────────────────────────────────────────────
    summary = {}
    for attr in PROMPTS:
        summary[attr] = {}
        for s in watch_steps:
            r = results[attr][s]["rank"]
            e = results[attr][s]["spatial_entropy"]
            summary[attr][s] = {
                "mean_rank":    float(np.mean(r)) if r else None,
                "mean_entropy": float(np.mean(e)) if e else None,
                "t_val":        timesteps_vals[s],
            }

    with open(out / "exp4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_map = {"color": "red", "shape": "blue", "texture": "green"}

    for attr in PROMPTS:
        steps_sorted = sorted(watch_steps)
        t_vals   = [summary[attr][s]["t_val"]    for s in steps_sorted if summary[attr][s]["mean_rank"] is not None]
        ranks    = [summary[attr][s]["mean_rank"] for s in steps_sorted if summary[attr][s]["mean_rank"] is not None]
        entropies= [summary[attr][s]["mean_entropy"] for s in steps_sorted if summary[attr][s]["mean_entropy"] is not None]

        axes[0].plot(t_vals, ranks,     marker="o", label=attr, color=colors_map[attr])
        axes[1].plot(t_vals, entropies, marker="o", label=attr, color=colors_map[attr])

    axes[0].set_title("Effective Rank of ΔH over ODE trajectory\n(Low = color-locked)")
    axes[0].set_xlabel("Timestep t (1=noise → 0=image)")
    axes[0].set_ylabel("Effective Rank")
    axes[0].legend(); axes[0].invert_xaxis()

    axes[1].set_title("Spatial Entropy of Hidden States\n(High = no boundaries yet)")
    axes[1].set_xlabel("Timestep t")
    axes[1].set_ylabel("Spatial Entropy")
    axes[1].legend(); axes[1].invert_xaxis()

    plt.tight_layout()
    plt.savefig(out / "exp4_locking_vs_layout.png", dpi=150)
    plt.close()

    # ── Verdict ───────────────────────────────────────────────────────────────
    _print_verdict(summary, watch_steps, timesteps_vals, out)


def _print_verdict(summary, watch_steps, timesteps_vals, out):
    print("\n" + "="*60)
    print("EXPERIMENT 4 VERDICT")
    print("="*60)
    print("Claim: Color locks early (rank drops) while spatial entropy still high")
    print()

    steps_sorted = sorted(watch_steps)

    # Find when color rank first drops below shape/texture rank
    color_locks_early = False
    color_lock_step   = None
    for s in steps_sorted:
        c_r = summary["color"][s]["mean_rank"]
        s_r = summary["shape"][s]["mean_rank"]
        t_r = summary["texture"][s]["mean_rank"]
        if c_r is None:
            continue
        if c_r < s_r and c_r < t_r:
            color_locks_early = True
            color_lock_step = s
            break

    # Check if spatial entropy is still high at that step
    high_entropy_at_lock = False
    if color_lock_step is not None:
        e = summary["color"][color_lock_step]["mean_entropy"]
        max_e = max(
            summary["color"][s]["mean_entropy"] or 0
            for s in steps_sorted
            if summary["color"][s]["mean_entropy"] is not None
        )
        high_entropy_at_lock = (e > 0.7 * max_e) if max_e > 0 else False

    print(f"C1: Color rank drops below shape/texture early → {'✅' if color_locks_early else '❌'}")
    if color_lock_step is not None:
        t_v = timesteps_vals[color_lock_step]
        print(f"    First observed at step {color_lock_step} (t={t_v:.3f})")
    print(f"C2: Spatial entropy still high at color-lock step → {'✅' if high_entropy_at_lock else '❌'}")
    print()

    if color_locks_early and high_entropy_at_lock:
        verdict = "✅ PROCEED — Premature color-locking confirmed. Color locks before object boundaries resolve."
    elif color_locks_early:
        verdict = "⚠️  PARTIAL — Color locks early but spatial layout may already be resolving."
    else:
        verdict = "❌ STOP — Color does not lock earlier than shape/texture. Recheck hypothesis."

    print(f"VERDICT: {verdict}")
    print()

    # Table
    print(f"{'Step':>5} {'t':>6} | {'color_rank':>10} {'shape_rank':>10} {'tex_rank':>10} | {'entropy':>8}")
    print("-"*65)
    for s in steps_sorted:
        t_v  = timesteps_vals[s]
        c_r  = summary["color"][s]["mean_rank"]
        s_r  = summary["shape"][s]["mean_rank"]
        t_r  = summary["texture"][s]["mean_rank"]
        e    = summary["color"][s]["mean_entropy"]
        cr   = f"{c_r:.3f}" if c_r else "N/A"
        sr   = f"{s_r:.3f}" if s_r else "N/A"
        tr   = f"{t_r:.3f}" if t_r else "N/A"
        ev   = f"{e:.3f}"   if e   else "N/A"
        print(f"{s:>5} {t_v:>6.1f} | {cr:>10} {sr:>10} {tr:>10} | {ev:>8}")

    with open(out / "exp4_verdict.txt", "w") as f:
        f.write(verdict + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--output_dir", default="results/exp4/")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    run(args)