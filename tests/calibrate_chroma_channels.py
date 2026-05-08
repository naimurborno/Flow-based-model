"""
calibrate_chroma_channels.py
----------------------------
Rank SD3 VAE latent channels by how strongly they encode chroma.

This does not generate images. It creates simple RGB color cards, encodes them
with the VAE, and ranks candidate channels / channel pairs by hue separation
after penalizing brightness sensitivity.

Usage:
    python calibrate_chroma_channels.py --config config.yaml
    python calibrate_chroma_channels.py --channels 0-15 --top_k 20
    python calibrate_chroma_channels.py --channels 0,2,5,8,9,13
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from diffusers import AutoencoderKL

from utils import load_config


BASE_COLORS: Dict[str, Tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 0.85, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
}

BRIGHTNESS_LEVELS = (0.55, 0.75, 0.95)


def parse_channels(text: str) -> List[int]:
    channels = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            channels.extend(range(int(lo), int(hi) + 1))
        else:
            channels.append(int(part))
    return sorted(set(channels))


def make_color_batch(size: int, device: str, dtype: torch.dtype):
    images = []
    labels = []
    brightness = []

    for name, rgb in BASE_COLORS.items():
        base = np.array(rgb, dtype=np.float32)
        for level in BRIGHTNESS_LEVELS:
            color = np.clip(base * level, 0.0, 1.0)
            img = torch.tensor(color, dtype=dtype).view(3, 1, 1)
            img = img.expand(3, size, size).clone()
            images.append(img)
            labels.append(name)
            brightness.append(level)

    batch = torch.stack(images, dim=0).to(device)
    batch = batch * 2.0 - 1.0
    return batch, labels, np.array(brightness, dtype=np.float32)


def encode_color_cards(
    vae: AutoencoderKL,
    image_size: int,
    device: str,
    dtype: torch.dtype,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    batch, labels, brightness = make_color_batch(image_size, device, dtype)

    with torch.no_grad():
        encoded = vae.encode(batch)
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.mean
        else:
            latents = encoded[0]
        latents = latents * vae.config.scaling_factor

    # One vector per color card: [n_cards, n_channels]
    features = latents.float().mean(dim=(2, 3)).cpu().numpy()
    return features, labels, brightness


def residualize_brightness(features: np.ndarray, brightness: np.ndarray) -> np.ndarray:
    """Remove the best linear brightness fit from each channel."""
    x = np.stack([np.ones_like(brightness), brightness], axis=1)
    beta = np.linalg.pinv(x) @ features
    return features - x @ beta


def hue_score(values: np.ndarray, labels: Sequence[str]) -> float:
    """Between-hue variance divided by within-hue variance."""
    labels_arr = np.array(labels)
    overall = values.mean(axis=0, keepdims=True)
    between = 0.0
    within = 0.0

    for label in sorted(set(labels)):
        group = values[labels_arr == label]
        center = group.mean(axis=0, keepdims=True)
        between += len(group) * float(((center - overall) ** 2).sum())
        within += float(((group - center) ** 2).sum())

    return between / (within + 1e-8)


def brightness_score(values: np.ndarray, brightness: np.ndarray) -> float:
    """Mean absolute correlation with brightness. Lower is better."""
    vals = values if values.ndim == 2 else values[:, None]
    scores = []
    b = brightness - brightness.mean()
    b_norm = np.linalg.norm(b) + 1e-8
    for i in range(vals.shape[1]):
        v = vals[:, i] - vals[:, i].mean()
        scores.append(abs(float(np.dot(v, b) / ((np.linalg.norm(v) + 1e-8) * b_norm))))
    return float(np.mean(scores))


def rank_channels(
    features: np.ndarray,
    residual: np.ndarray,
    labels: Sequence[str],
    brightness: np.ndarray,
    channels: Sequence[int],
):
    rows = []
    for ch in channels:
        raw = features[:, ch]
        res = residual[:, ch]
        hue = hue_score(res[:, None], labels)
        bright = brightness_score(raw, brightness)
        score = hue / (1.0 + bright)
        rows.append({
            "channel": ch,
            "score": score,
            "hue_score": hue,
            "brightness_corr": bright,
        })
    return sorted(rows, key=lambda x: x["score"], reverse=True)


def rank_pairs(
    features: np.ndarray,
    residual: np.ndarray,
    labels: Sequence[str],
    brightness: np.ndarray,
    channels: Sequence[int],
):
    rows = []
    labels_arr = np.array(labels)

    for ch_a, ch_b in itertools.combinations(channels, 2):
        raw = features[:, [ch_a, ch_b]]
        res = residual[:, [ch_a, ch_b]]
        hue = hue_score(res, labels)
        bright = brightness_score(raw, brightness)
        score = hue / (1.0 + bright)

        targets = {}
        for label in sorted(set(labels)):
            center = raw[labels_arr == label].mean(axis=0)
            norm = np.linalg.norm(center) + 1e-8
            targets[label] = (center / norm).round(4).tolist()

        rows.append({
            "pair": [ch_a, ch_b],
            "score": score,
            "hue_score": hue,
            "brightness_corr": bright,
            "normalized_targets": targets,
        })

    return sorted(rows, key=lambda x: x["score"], reverse=True)


def print_table(title: str, rows: list, top_k: int):
    print(f"\n{title}")
    print("-" * len(title))
    for i, row in enumerate(rows[:top_k], 1):
        if "pair" in row:
            ident = f"{row['pair']}"
        else:
            ident = f"ch {row['channel']:02d}"
        print(
            f"{i:>2}. {ident:<9} "
            f"score={row['score']:.4f} "
            f"hue={row['hue_score']:.4f} "
            f"brightness_corr={row['brightness_corr']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--channels", default="0-15")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", default="results/chroma_channel_ranking.json")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = args.model_id or cfg.get(
        "model_id", "stabilityai/stable-diffusion-3-medium-diffusers"
    )
    channels = parse_channels(args.channels)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    print(f"[VAE] Loading {model_id} / vae")
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(args.device)
    vae.eval()

    features, labels, brightness = encode_color_cards(
        vae, args.image_size, args.device, dtype
    )

    max_channel = features.shape[1] - 1
    bad = [ch for ch in channels if ch < 0 or ch > max_channel]
    if bad:
        raise ValueError(f"Invalid channels {bad}; VAE has channels 0-{max_channel}")

    residual = residualize_brightness(features, brightness)
    channel_rows = rank_channels(features, residual, labels, brightness, channels)
    pair_rows = rank_pairs(features, residual, labels, brightness, channels)

    print_table("Top Single Channels", channel_rows, args.top_k)
    print_table("Top Channel Pairs", pair_rows, args.top_k)

    best = pair_rows[0]
    print("\nRecommended experiment edit:")
    print(f"CHROMA_PAIR = {best['pair']}")
    print("target_colors = {")
    for color in ("red", "blue", "green", "yellow"):
        if color in best["normalized_targets"]:
            print(f'    "{color}": torch.tensor({best["normalized_targets"][color]}),')
    print("}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "channels": channels,
        "labels": labels,
        "brightness_levels": list(BRIGHTNESS_LEVELS),
        "single_channels": channel_rows,
        "channel_pairs": pair_rows,
        "recommended_pair": best["pair"],
        "recommended_targets": best["normalized_targets"],
    }
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Done] Wrote {output}")


if __name__ == "__main__":
    main()
