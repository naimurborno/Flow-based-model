"""
q1_entropy_analysis.py
----------------------
Correct Q1 experiment design:

  STEP 1 — For each prompt, run N_SEEDS different seeds through full denoising.
            At every (step, layer) for TARGET_LAYERS [12, 14], collect:
              - DC vector:  mean of image hidden states over patch dim  [dim]
              - Entropy:    text→image attention entropy (scalar)

  STEP 2 — After all seeds for a prompt, compute DC lock-in score per (step, layer):
              lock-in sim = mean pairwise cosine similarity of DC vectors across seeds
              High sim (≥ LOCKIN_SIM_THRESHOLD) = DC is locked (same across seeds)
              This matches DAVE's definition (they measured 0.998 cosine similarity)

  STEP 3 — After all prompts, compare:
              entropy at lock-in moments  (early steps + high sim)
              vs entropy at non-lock-in moments
               

  WHY LAYERS 12 & 14:
    DAVE's block-wise analysis (slide 5) showed these two blocks have the
    strongest DC dominance for color and texture attributes respectively.

  WHY EARLY STEPS (t > EARLY_T_THRESHOLD):
    DAVE showed lock-in happens at early denoising steps (high t = high noise).
    SD3 scheduler runs t in [0, 1000]; early approx t > 700.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import mannwhitneyu

# ---- Constants ---------------------------------------------------------- #
TARGET_LAYERS        = [12, 14]
LOCKIN_SIM_THRESHOLD = 0.998
EARLY_T_THRESHOLD    = 700
# ------------------------------------------------------------------------- #


class Q1EntropyAnalyzer:

    def __init__(self, transformer, n_seeds=5):
        self.transformer = transformer
        self.n_seeds     = n_seeds
        self.hooks       = []

        self._q_txt = {}
        self._k_img = {}

        self._raw           = {}
        self.lockin_results = []

        self.current_prompt = 0
        self.current_seed   = 0
        self.current_step   = 0
        self.current_t      = 1000.0

        cfg            = transformer.config
        self.num_heads = cfg.num_attention_heads
        self.head_dim  = cfg.attention_head_dim

    # ================================================================== #
    #  HOOK REGISTRATION                                                  #
    # ================================================================== #

    def register_hooks(self):
        for layer_idx in TARGET_LAYERS:
            block = self.transformer.transformer_blocks[layer_idx]
            attn  = block.attn

            h = attn.to_k.register_forward_hook(
                lambda m, inp, out, l=layer_idx:
                    self._k_img.update({l: out.detach().float()})
            )
            self.hooks.append(h)

            if hasattr(attn, "add_q_proj"):
                h = attn.add_q_proj.register_forward_hook(
                    lambda m, inp, out, l=layer_idx:
                        self._q_txt.update({l: out.detach().float()})
                )
                self.hooks.append(h)
            else:
                print(f"[Q1] Warning: layer {layer_idx} missing add_q_proj")

            h = block.register_forward_hook(self._make_block_hook(layer_idx))
            self.hooks.append(h)

        print(f"[Q1] Hooks registered on layers {TARGET_LAYERS}")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        print("[Q1] Hooks removed.")

    # ================================================================== #
    #  PER-BLOCK HOOK                                                     #
    # ================================================================== #

    def _make_block_hook(self, layer_idx):
        def hook(module, input, output):
            img_hidden = None
            if isinstance(output, tuple):
                for candidate in reversed(output):
                    if candidate is not None:
                        img_hidden = candidate
                        break
            elif output is not None:
                img_hidden = output

            if img_hidden is None:
                return

            batch    = img_hidden.shape[0]
            cond_img = img_hidden[batch // 2:].float()

            # DC vector: spatial mean over patch tokens
            dc_vec = cond_img.mean(dim=1).mean(dim=0).cpu()   # [dim]

            entropy = self._compute_entropy(layer_idx)

            key = (self.current_prompt, self.current_seed,
                   self.current_step, layer_idx)
            self._raw[key] = {
                "dc"      : dc_vec,
                "entropy" : entropy,
                "t"       : self.current_t,
            }

        return hook

    # ================================================================== #
    #  ENTROPY COMPUTATION                                                #
    # ================================================================== #

    def _compute_entropy(self, layer_idx):
        if layer_idx not in self._q_txt or layer_idx not in self._k_img:
            return None

        q_txt = self._q_txt[layer_idx]
        k_img = self._k_img[layer_idx]

        batch   = q_txt.shape[0]
        txt_seq = q_txt.shape[1]
        img_seq = k_img.shape[1]

        q_txt = q_txt[batch // 2:]
        k_img = k_img[batch // 2:]

        q_txt = q_txt.view(-1, txt_seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_img = k_img.view(-1, img_seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale        = self.head_dim ** -0.5
        attn_scores  = torch.matmul(q_txt, k_img.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        eps     = 1e-9
        entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
        return entropy.mean().item()

    # ================================================================== #
    #  STEP UPDATE                                                        #
    # ================================================================== #

    def update_step(self, step_idx, t):
        self.current_step = step_idx
        self.current_t    = t.item() if hasattr(t, "item") else float(t)

    # ================================================================== #
    #  LOCK-IN COMPUTATION — call after all seeds for a prompt           #
    # ================================================================== #

    def compute_lockin(self, prompt_idx, n_steps):
        for step_idx in range(n_steps):
            for layer_idx in TARGET_LAYERS:
                dc_vecs   = []
                entropies = []
                t_val     = None

                for seed_idx in range(self.n_seeds):
                    key = (prompt_idx, seed_idx, step_idx, layer_idx)
                    if key not in self._raw:
                        continue
                    entry = self._raw[key]
                    if entry["entropy"] is None:
                        continue
                    dc_vecs.append(entry["dc"])
                    entropies.append(entry["entropy"])
                    if t_val is None:
                        t_val = entry["t"]

                if len(dc_vecs) < 2 or t_val is None:
                    continue

                dc_stack   = torch.stack(dc_vecs)
                dc_norm    = F.normalize(dc_stack, dim=-1)
                sim_mat    = dc_norm @ dc_norm.T
                n          = sim_mat.shape[0]
                mask       = torch.triu(torch.ones(n, n), diagonal=1).bool()
                lockin_sim = sim_mat[mask].mean().item()

                self.lockin_results.append({
                    "prompt_idx"   : prompt_idx,
                    "step_idx"     : step_idx,
                    "layer"        : layer_idx,
                    "t"            : t_val,
                    "lockin_sim"   : lockin_sim,
                    "mean_entropy" : float(np.mean(entropies)),
                    "is_lockin"    : lockin_sim >= LOCKIN_SIM_THRESHOLD,
                })

        # Free memory for this prompt
        to_del = [k for k in self._raw if k[0] == prompt_idx]
        for k in to_del:
            del self._raw[k]

        n_locked = sum(
            1 for r in self.lockin_results
            if r["prompt_idx"] == prompt_idx and r["is_lockin"]
        )
        print(f"  [Q1] Prompt {prompt_idx}: "
              f"lock-in events = {n_locked} / {n_steps * len(TARGET_LAYERS)}")

    # ================================================================== #
    #  FINAL ANALYSIS                                                     #
    # ================================================================== #

    def final_analysis(self, output_dir="results/"):
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.lockin_results).dropna()

        if df.empty:
            print("[Q1] No data — check hooks.")
            return

        early      = df[df["t"] > EARLY_T_THRESHOLD].copy()
        locked     = early[early["is_lockin"]]
        not_locked = early[~early["is_lockin"]]

        print("\n" + "=" * 62)
        print("  Q1 RESULT: Entropy at DC Lock-in vs Non-Lock-in")
        print("=" * 62)
        print(f"  Layers analysed       : {TARGET_LAYERS}")
        print(f"  Early steps (t>{EARLY_T_THRESHOLD}): {len(early)} observations")
        print(f"  Lock-in (sim≥{LOCKIN_SIM_THRESHOLD})  : {len(locked)}")
        print(f"  Non-lock-in           : {len(not_locked)}")

        if len(locked) > 0 and len(not_locked) > 0:
            stat, p = mannwhitneyu(
                locked["mean_entropy"].values,
                not_locked["mean_entropy"].values,
                alternative="two-sided",
            )
            e_lock   = locked["mean_entropy"].mean()
            e_nolock = not_locked["mean_entropy"].mean()

            print(f"\n  Mean entropy at lock-in     : {e_lock:.4f}")
            print(f"  Mean entropy at non-lock-in : {e_nolock:.4f}")
            print(f"  Mann-Whitney U p-value      : {p:.2e}")

            if p < 0.05:
                direction = "LOW" if e_lock < e_nolock else "HIGH"
                mechanism = (
                    "peaked attention collapses patches to same mean"
                    if direction == "LOW"
                    else "uniform attention gives every patch identical signal"
                )
                print(f"\n  → {direction} entropy at lock-in moments")
                print(f"    ({mechanism})")
            else:
                print("\n  → No significant entropy difference")
                print("    Q1 alone may not explain DC lock-in → check Q2/Q3/Q4")

        print("=" * 62)
        self._plot(df, early, locked, not_locked, output_dir)

    # ================================================================== #
    #  PLOTTING                                                           #
    # ================================================================== #

    def _plot(self, df, early, locked, not_locked, output_dir):
        colors = {12: "steelblue", 14: "darkorange"}
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: DC lock-in similarity over steps
        for layer_idx in TARGET_LAYERS:
            ldf = df[df["layer"] == layer_idx].groupby("step_idx")["lockin_sim"].mean()
            axes[0, 0].plot(ldf.index, ldf.values,
                            label=f"Layer {layer_idx}", color=colors[layer_idx])
        axes[0, 0].axhline(LOCKIN_SIM_THRESHOLD, color="red",
                           linestyle="--", label=f"Threshold ({LOCKIN_SIM_THRESHOLD})")
        axes[0, 0].set_title("DC Cosine Similarity Across Seeds\n(high = DC locked)")
        axes[0, 0].set_xlabel("Denoising Step")
        axes[0, 0].set_ylabel("Mean Pairwise Cosine Sim")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_ylim(0, 1.05)

        # Plot 2: Entropy over steps
        for layer_idx in TARGET_LAYERS:
            ldf = df[df["layer"] == layer_idx].groupby("step_idx")["mean_entropy"].mean()
            axes[0, 1].plot(ldf.index, ldf.values,
                            label=f"Layer {layer_idx}", color=colors[layer_idx])
        axes[0, 1].set_title("Text→Image Attention Entropy Over Steps")
        axes[0, 1].set_xlabel("Denoising Step")
        axes[0, 1].set_ylabel("Mean Entropy")
        axes[0, 1].legend(fontsize=8)

        # Plot 3: Violin — entropy at lock-in vs not
        if len(locked) > 0 and len(not_locked) > 0:
            parts = axes[1, 0].violinplot(
                [locked["mean_entropy"].values, not_locked["mean_entropy"].values],
                positions=[0, 1], showmedians=True,
            )
            parts["bodies"][0].set_facecolor("tomato")
            parts["bodies"][1].set_facecolor("steelblue")
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_xticklabels(["DC Locked", "Not Locked"])
        axes[1, 0].set_title(
            f"Entropy Distribution (early steps t>{EARLY_T_THRESHOLD})\n"
            f"Layers {TARGET_LAYERS}"
        )
        axes[1, 0].set_ylabel("Mean Entropy")

        # Plot 4: Scatter lock-in sim vs entropy
        for layer_idx in TARGET_LAYERS:
            sub = early[early["layer"] == layer_idx]
            axes[1, 1].scatter(sub["lockin_sim"], sub["mean_entropy"],
                               alpha=0.4, s=12, label=f"Layer {layer_idx}",
                               color=colors[layer_idx])
        axes[1, 1].axvline(LOCKIN_SIM_THRESHOLD, color="red", linestyle="--")
        axes[1, 1].set_xlabel("DC Lock-in Similarity (across seeds)")
        axes[1, 1].set_ylabel("Text→Image Entropy")
        axes[1, 1].set_title("Lock-in Score vs Entropy (early steps)")
        axes[1, 1].legend(fontsize=8)

        plt.suptitle(
            "Q1: Does Attention Entropy Drive DC Lock-in? (Layers 12 & 14)",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        out_path = os.path.join(output_dir, "q1_entropy_vs_lockin.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Q1] Plot saved → {out_path}")

        self._plot_scatter(early, output_dir)

    # ================================================================== #
    #  SCATTER: Locked vs Not-Locked entropy data points                 #
    # ================================================================== #

    def _plot_scatter(self, early: pd.DataFrame, output_dir: str):
        """
        Scatter plot: X = mean_entropy, Y = lockin_sim
        Each point is one (prompt, step, layer) observation in early steps.
        Color encodes lock-in status. Split into two subplots by layer.
        Saved separately as q1_scatter_lockin_vs_entropy.png
        """
        if early.empty:
            print("[Q1] Scatter: no early-step data to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        layer_info = {
            12: {"ax": axes[0], "color_locked": "#d62728", "color_free": "#1f77b4"},
            14: {"ax": axes[1], "color_locked": "#e377c2", "color_free": "#2ca02c"},
        }

        for layer_idx, info in layer_info.items():
            ax      = info["ax"]
            sub     = early[early["layer"] == layer_idx]
            locked  = sub[sub["is_lockin"]]
            free    = sub[~sub["is_lockin"]]

            ax.scatter(
                free["mean_entropy"], free["lockin_sim"],
                c=info["color_free"], alpha=0.45, s=18,
                label=f"Not locked  (n={len(free)})", zorder=2,
            )
            ax.scatter(
                locked["mean_entropy"], locked["lockin_sim"],
                c=info["color_locked"], alpha=0.65, s=22,
                label=f"DC Locked   (n={len(locked)})", zorder=3,
            )

            ax.axhline(
                LOCKIN_SIM_THRESHOLD, color="black",
                linestyle="--", linewidth=1.2,
                label=f"Threshold ({LOCKIN_SIM_THRESHOLD})",
            )

            # Annotate mean entropy per group
            if len(locked) > 0:
                ax.axvline(
                    locked["mean_entropy"].mean(),
                    color=info["color_locked"], linestyle=":",
                    linewidth=1.2, alpha=0.8,
                    label=f"Locked mean entropy = {locked['mean_entropy'].mean():.3f}",
                )
            if len(free) > 0:
                ax.axvline(
                    free["mean_entropy"].mean(),
                    color=info["color_free"], linestyle=":",
                    linewidth=1.2, alpha=0.8,
                    label=f"Not-locked mean entropy = {free['mean_entropy'].mean():.3f}",
                )

            ax.set_xlabel("Text→Image Attention Entropy", fontsize=11)
            ax.set_ylabel("DC Lock-in Cosine Similarity (across seeds)", fontsize=11)
            ax.set_title(f"Layer {layer_idx}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=7.5, loc="lower right")
            ax.set_ylim(0.97, 1.005)

        fig.suptitle(
            f"Q1 Scatter: DC Lock-in vs Entropy (early steps t > {EARLY_T_THRESHOLD})\n"
            "Red/Pink = DC Locked  |  Blue/Green = Not Locked",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        scatter_path = os.path.join(output_dir, "q1_scatter_lockin_vs_entropy.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"[Q1] Scatter plot saved → {scatter_path}")