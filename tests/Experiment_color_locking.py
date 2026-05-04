import torch
import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_wrapper import SD3PipelineWrapper
from utils import load_config, set_seed

PROMPTS = {
    "color": [
        ("red car next to blue bicycle",         "car next to bicycle",        "car", "bicycle"),
        ("yellow banana beside green apple",      "banana beside apple",        "banana", "apple"),
        ("red house beside blue barn",            "house beside barn",          "house", "barn"),
        ("orange cat next to purple dog",         "cat next to dog",            "cat", "dog"),
        ("pink shirt beside black jacket",        "shirt beside jacket",        "shirt", "jacket"),
        ("blue boat next to red lighthouse",      "boat next to lighthouse",    "boat", "lighthouse"),
        ("green frog beside yellow duck",         "frog beside duck",           "frog", "duck"),
        ("red apple beside blue cup",             "apple beside cup",           "apple", "cup"),
    ],
    "shape": [
        ("round table next to square chair",      "table next to chair",        "table", "chair"),
        ("oval mirror beside rectangular door",   "mirror beside door",         "mirror", "door"),
        ("circular clock next to triangular sign","clock next to sign",         "clock", "sign"),
        ("curved sofa beside angular desk",       "sofa beside desk",           "sofa", "desk"),
        ("round bowl next to square box",         "bowl next to box",           "bowl", "box"),
        ("spherical lamp beside flat screen",     "lamp beside screen",         "lamp", "screen"),
        ("cylindrical vase next to flat book",    "vase next to book",          "vase", "book"),
        ("round wheel beside square window",      "wheel beside window",        "wheel", "window"),
    ],
    "texture": [
        ("metallic car next to wooden bicycle",   "car next to bicycle",        "car", "bicycle"),
        ("glossy table beside rusty chair",       "table beside chair",         "table", "chair"),
        ("smooth stone next to rough brick",      "stone next to brick",        "stone", "brick"),
        ("silky curtain beside rough carpet",     "curtain beside carpet",      "curtain", "carpet"),
        ("shiny metal bowl beside clay pot",      "bowl beside pot",            "bowl", "pot"),
        ("velvet sofa beside wooden desk",        "sofa beside desk",           "sofa", "desk"),
        ("granite counter beside plastic shelf",  "counter beside shelf",       "counter", "shelf"),
        ("woven basket beside glass vase",        "basket beside vase",         "basket", "vase"),
    ],
}

WATCH_BLOCKS   = [8, 10, 12, 14, 16]
WATCH_STEPS    = list(range(0, 50, 5))
SMOKE_STEPS    = [0, 10, 25, 40, 49]
SMOKE_PROMPTS  = 2


def effective_rank(dH: torch.Tensor) -> float:
    # dH: [1024, C]
    h = dH.float()
    if h.shape[0] < 2:
        return 1.0
    _, S, _ = torch.linalg.svd(h, full_matrices=False)
    S = S[S > 1e-8]
    if len(S) == 0:
        return 1.0
    p = S / S.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())


def spatial_cv(h: torch.Tensor) -> float:
    # h: [1024, C] — coefficient of variation of token norms
    # Low CV = uniform tokens = no spatial boundaries yet
    norms = h.float().norm(dim=-1)
    mean = norms.mean()
    if mean < 1e-8:
        return 0.0
    return float((norms.std() / mean).item())


class HiddenStateCapture:
    def __init__(self, transformer, watch_blocks):
        self.captures = {}
        self._hooks = []
        for idx in watch_blocks:
            block = transformer.transformer_blocks[idx]
            self._hooks.append(block.register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, idx):
        def hook(module, inp, out):
            # SD3 returns (encoder_hidden_states, hidden_states)
            # out[1] = image tokens [1, 1024, C]
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                self.captures[idx] = out[1].detach().cpu()
            elif isinstance(out, tuple):
                self.captures[idx] = out[0].detach().cpu()
            else:
                self.captures[idx] = out.detach().cpu()
        return hook

    def clear(self):
        self.captures.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()


def forward_pass(wrapper, prompt, t_val, latents, seed=42):
    set_seed(seed)
    emb, pooled = wrapper._encode_prompt_single(prompt)

    wrapper.scheduler.set_timesteps(50)
    timesteps = wrapper.scheduler.timesteps
    step_idx = min(range(len(timesteps)), key=lambda i: abs(timesteps[i].item() - t_val))
    t = timesteps[step_idx]

    dev = next(wrapper.transformer.parameters()).device
    with torch.no_grad():
        wrapper.transformer(
            hidden_states         = latents.to(dev, torch.float16),
            timestep              = t.reshape(1).to(dev, torch.float16),
            encoder_hidden_states = emb.to(dev, torch.float16),
            pooled_projections    = pooled.to(dev, torch.float16),
        )
    return t.item()


def run(args):
    cfg = load_config("/content/Flow-based-model/config.yaml")
    cfg["flow"]["guidance_scale"] = 1.0
    wrapper = SD3PipelineWrapper(cfg, device=args.device)
    wrapper.load()
    # wrapper.pipe.disable_model_cpu_offload()
    wrapper.transformer = wrapper.pipe.transformer.to(args.device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    wrapper.scheduler.set_timesteps(50)
    timesteps_vals = [t.item() for t in wrapper.scheduler.timesteps]

    watch_steps = SMOKE_STEPS if args.smoke_test else WATCH_STEPS

    results = {
        attr: {s: {"rank": [], "cv": []} for s in watch_steps}
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

                capture.clear()
                forward_pass(wrapper, with_p, t_val, latents, seed=42)
                h_with = {k: v.clone() for k, v in capture.captures.items()}

                capture.clear()
                forward_pass(wrapper, without_p, t_val, latents, seed=42)
                h_without = {k: v.clone() for k, v in capture.captures.items()}

                if not h_with or not h_without:
                    continue

                rank_vals, cv_vals = [], []
                for b in WATCH_BLOCKS:
                    if b not in h_with or b not in h_without:
                        continue
                    dH    = (h_with[b] - h_without[b]).squeeze(0)   # [1024, C]
                    h_img = h_with[b].squeeze(0)                      # [1024, C]
                    rank_vals.append(effective_rank(dH))
                    cv_vals.append(spatial_cv(h_img))

                if rank_vals:
                    results[attr][step_idx]["rank"].append(float(np.mean(rank_vals)))
                    results[attr][step_idx]["cv"].append(float(np.mean(cv_vals)))

    capture.remove()

    summary = {}
    for attr in PROMPTS:
        summary[attr] = {}
        for s in watch_steps:
            r = results[attr][s]["rank"]
            c = results[attr][s]["cv"]
            summary[attr][s] = {
                "mean_rank": float(np.mean(r)) if r else None,
                "mean_cv":   float(np.mean(c)) if c else None,
                "t_val":     timesteps_vals[s],
            }

    with open(out / "exp4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _make_plots(summary, watch_steps, out)
    _print_verdict(summary, watch_steps, timesteps_vals, out)


def _make_plots(summary, watch_steps, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = {"color": "red", "shape": "blue", "texture": "green"}
    steps_sorted = sorted(watch_steps)

    for attr in PROMPTS:
        t_vals = [summary[attr][s]["t_val"]    for s in steps_sorted if summary[attr][s]["mean_rank"] is not None]
        ranks  = [summary[attr][s]["mean_rank"] for s in steps_sorted if summary[attr][s]["mean_rank"] is not None]
        cvs    = [summary[attr][s]["mean_cv"]   for s in steps_sorted if summary[attr][s]["mean_cv"]   is not None]
        axes[0].plot(t_vals, ranks, marker="o", label=attr, color=cmap[attr])
        axes[1].plot(t_vals, cvs,   marker="o", label=attr, color=cmap[attr])

    axes[0].set_title("Effective Rank of ΔH  (Low = locked)")
    axes[0].set_xlabel("Timestep t  (1000=noise → 0=image)")
    axes[0].set_ylabel("Effective Rank")
    axes[0].legend()
    axes[0].invert_xaxis()

    axes[1].set_title("Spatial CV of Token Norms  (Low = no boundaries)")
    axes[1].set_xlabel("Timestep t")
    axes[1].set_ylabel("Coefficient of Variation")
    axes[1].legend()
    axes[1].invert_xaxis()

    plt.tight_layout()
    plt.savefig(out / "exp4_locking_vs_layout.png", dpi=150)
    plt.close()


def _print_verdict(summary, watch_steps, timesteps_vals, out):
    print("\n" + "="*60)
    print("EXPERIMENT 4 VERDICT")
    print("="*60)
    print("Claim: Color rank drops early while spatial CV is still low (no boundaries)")
    print()

    steps_sorted = sorted(watch_steps)

    color_locks_early = False
    color_lock_step = None
    for s in steps_sorted:
        c_r = summary["color"][s]["mean_rank"]
        s_r = summary["shape"][s]["mean_rank"]
        t_r = summary["texture"][s]["mean_rank"]
        if c_r is None or s_r is None or t_r is None:
            continue
        if c_r < s_r and c_r < t_r:
            color_locks_early = True
            color_lock_step = s
            break

    low_cv_at_lock = False
    if color_lock_step is not None:
        cv_at_lock = summary["color"][color_lock_step]["mean_cv"]
        max_cv = max(
            (summary["color"][s]["mean_cv"] or 0)
            for s in steps_sorted
            if summary["color"][s]["mean_cv"] is not None
        )
        if cv_at_lock is not None and max_cv > 0:
            low_cv_at_lock = cv_at_lock < 0.5 * max_cv

    print(f"C1: Color rank drops below shape/texture early → {'✅' if color_locks_early else '❌'}")
    if color_lock_step is not None:
        print(f"    First at step {color_lock_step} (t={timesteps_vals[color_lock_step]:.1f})")
    print(f"C2: Spatial CV still low at color-lock step   → {'✅' if low_cv_at_lock else '❌'}")
    print()

    if color_locks_early and low_cv_at_lock:
        verdict = "✅ PROCEED — Premature color-locking confirmed."
    elif color_locks_early:
        verdict = "⚠️  PARTIAL — Color locks early but boundaries may already be resolving."
    else:
        verdict = "❌ STOP — Color does not lock earlier than shape/texture."

    print(f"VERDICT: {verdict}")
    print()

    fmt = lambda x: f"{x:.3f}" if x is not None else " N/A "
    print(f"{'Step':>5} {'t':>7} | {'color_rank':>10} {'shape_rank':>10} {'tex_rank':>10} | {'color_cv':>9} {'shape_cv':>9} {'tex_cv':>9}")
    print("-"*82)
    for s in steps_sorted:
        tv = timesteps_vals[s]
        print(
            f"{s:>5} {tv:>7.1f} | "
            f"{fmt(summary['color'][s]['mean_rank']):>10} "
            f"{fmt(summary['shape'][s]['mean_rank']):>10} "
            f"{fmt(summary['texture'][s]['mean_rank']):>10} | "
            f"{fmt(summary['color'][s]['mean_cv']):>9} "
            f"{fmt(summary['shape'][s]['mean_cv']):>9} "
            f"{fmt(summary['texture'][s]['mean_cv']):>9}"
        )

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