# SD3 Flow Matching Experiment

Small research scaffold for zero-shot Stable Diffusion 3 / flow-matching text-to-image experiments.

The project loads an SD3-style Diffusers pipeline, exposes the text encoders, MMDiT transformer, VAE, and flow scheduler separately, then runs a custom flow ODE loop over latent noise.

## Files

- `run_experiment.py` - command-line entry point.
- `pipeline_wrapper.py` - loads SD3 and exposes individual components.
- `custom_flow_loop.py` - custom Euler / Heun flow-matching integration loop.
- `config.yaml` - model, prompt, generation, and flow solver settings.
- `utils.py` - config loading, seeding, and result saving.

## Install

```bash
pip install -r requirements.txt
```

SD3 Medium is gated on Hugging Face, so you may need to authenticate first:

```bash
huggingface-cli login
```

## Run

```bash
python run_experiment.py --prompt "a photo of an astronaut riding a horse on mars" --output_dir results/
```

Or edit `config.yaml` and run:

```bash
python run_experiment.py
```

Outputs are written to `results/` by default.

## Note

This is a flow-matching / rectified-flow-style generation repo, not a classical normalizing-flow model such as RealNVP or Glow.

