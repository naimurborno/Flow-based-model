"""
run_experiment.py  (Flow Matching + Q1 Entropy Analysis)
---------------------------------------------------------
Entry point. After all prompts are generated, calls Q1 analyzer
to compute Pearson r(entropy, DC) and save plots.

Q1 change: two lines added at the end of main().
"""

import torch
import argparse
from pathlib import Path

from pipeline_wrapper   import SD3PipelineWrapper
from custom_flow_loop   import FlowMatchingLoop
from utils              import load_config, set_seed, save_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--prompt",     type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)

    if args.prompt:
        cfg["prompts"] = [args.prompt]

    print(f"[INFO] Device  : {args.device}")
    print(f"[INFO] Prompts : {cfg['prompts']}")
    print(f"[INFO] Solver  : {cfg.get('flow', {}).get('solver', 'euler')}")

    # ------------------------------------------------------------------ #
    # 1. Load + patch (patch() now registers Q1 hooks)                   #
    # ------------------------------------------------------------------ #
    wrapper = SD3PipelineWrapper(cfg, device=args.device)
    wrapper.load()
    wrapper.patch()   # ← registers Q1EntropyAnalyzer hooks

    # ------------------------------------------------------------------ #
    # 2. Build flow loop — pass analyzer so it updates current_step      #
    # ------------------------------------------------------------------ #
    flow_loop = FlowMatchingLoop(
        unet       = wrapper.transformer,
        scheduler  = wrapper.scheduler,
        cfg        = cfg,
        device     = args.device,
        q1_analyzer = wrapper.q1_analyzer,   # ← Q1 addition
    )

    # ------------------------------------------------------------------ #
    # 3. Generate all prompts                                             #
    # ------------------------------------------------------------------ #
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, prompt in enumerate(cfg["prompts"]):
        print(f"\n[Prompt {i+1}/{len(cfg['prompts'])}] {prompt}")

        prompt_embeds, pooled_embeds = wrapper.encode_prompt(
            prompt          = prompt,
            negative_prompt = cfg.get("negative_prompt", ""),
        )
        latents = wrapper.get_initial_latents(seed=args.seed + i)

        result = flow_loop.run(
            latents           = latents,
            text_embeddings   = prompt_embeds,
            pooled_embeddings = pooled_embeds,
        )

        image    = wrapper.decode_latents(result["latents"])
        out_path = Path(args.output_dir) / f"output_{i:03d}.png"
        image.save(out_path)
        print(f"  Saved → {out_path}")

        all_results.append({"prompt": prompt, "latents": result["latents"]})

    save_results(all_results, args.output_dir)

    # ------------------------------------------------------------------ #
    # 4. Q1 Analysis — runs after all prompts are done                   #
    # ------------------------------------------------------------------ #
    print("\n[Q1] Running entropy vs DC correlation analysis...")
    wrapper.q1_analyzer.plot(output_dir=args.output_dir)   # ← Q1 addition
    wrapper.q1_analyzer.remove_hooks()                     # ← Q1 addition

    print(f"\n[Done] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()