"""
run_experiment.py  (Flow Matching edition)
------------------------------------------
Entry point for zero-shot SD3 flow matching text-to-image experiments.
Loads pretrained weights via Diffusers, patches the pipeline,
and runs the custom flow ODE integration loop.

Usage:
    python run_experiment.py \
        --prompt "a photo of an astronaut riding a horse" \
        --output_dir results/

Key differences vs DDPM version:
  - wrapper returns (prompt_embeds, pooled_embeds) — two tensors for SD3
  - denoiser is FlowMatchingLoop, not CustomDenoisingLoop
  - t ∈ [1.0 → 0.0] continuous, not discrete {999..0}
"""

import torch
import argparse
from pathlib import Path

from pipeline_wrapper   import SD3PipelineWrapper
from custom_flow_loop   import FlowMatchingLoop
from utils              import load_config, set_seed, save_results


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-Shot SD3 Flow Matching Experiment")
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--prompt",     type=str, default=None,
                        help="Single prompt (overrides config)")
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
    # 1. Load SD3 pipeline + extract components                           #
    # ------------------------------------------------------------------ #
    wrapper = SD3PipelineWrapper(cfg, device=args.device)
    wrapper.load()    # downloads / loads from cache
    wrapper.patch()   # your architectural patches go here

    # ------------------------------------------------------------------ #
    # 2. Build flow ODE loop                                              #
    #    Pass transformer (velocity predictor) instead of unet            #
    # ------------------------------------------------------------------ #
    flow_loop = FlowMatchingLoop(
        unet      = wrapper.transformer,   # MMDiT velocity predictor
        scheduler = wrapper.scheduler,     # FlowMatchEulerDiscreteScheduler
        cfg       = cfg,
        device    = args.device,
    )

    # ------------------------------------------------------------------ #
    # 3. Run over all prompts                                             #
    # ------------------------------------------------------------------ #
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, prompt in enumerate(cfg["prompts"]):
        print(f"\n[Prompt {i+1}/{len(cfg['prompts'])}] {prompt}")

        # Encode prompt → (sequence embeddings, pooled embeddings)
        # SD3 returns TWO tensors (unlike SD1.5's single tensor)
        prompt_embeds, pooled_embeds = wrapper.encode_prompt(
            prompt          = prompt,
            negative_prompt = cfg.get("negative_prompt", ""),
        )

        # Build initial latents: pure noise at t=1
        latents = wrapper.get_initial_latents(seed=args.seed + i)

        # Run flow ODE integration: t=1 → t=0
        result = flow_loop.run(
            latents           = latents,
            text_embeddings   = prompt_embeds,
            pooled_embeddings = pooled_embeds,
        )

        # Decode latents → PIL image
        image = wrapper.decode_latents(result["latents"])
        out_path = Path(args.output_dir) / f"output_{i:03d}.png"
        image.save(out_path)
        print(f"  Saved → {out_path}")

        all_results.append({
            "prompt"  : prompt,
            "latents" : result["latents"],
        })

    save_results(all_results, args.output_dir)
    print(f"\n[Done] All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()