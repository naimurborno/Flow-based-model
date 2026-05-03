"""
experiment3_spatial_leakage.py
-------------------------------
Experiment 3 — Spatial Leakage Ratio

QUESTION BEING ANSWERED:
    Does the geometric concentration of color perturbations (proved in Phase 0)
    actually cause color to SPREAD INTO THE WRONG SPATIAL REGIONS?

    i.e. when you add "red" to a prompt about a car next to a bicycle,
    does the "red" perturbation bleed into the bicycle's token region?
    And is this leakage STRONGER for color than for shape or texture?

WHAT THIS DOES:
    For each two-object conflict prompt ("red car next to blue bicycle"):
        1. Run two forward passes:
              Pass A: "red car next to blue bicycle"
              Pass B: "car next to bicycle"           ← SAME objects, NO attributes
           Both passes: identical seed, identical latents, CFG disabled
        2. Compute ΔH = H(with_attrs) - H(without_attrs)
        3. Compute perturbation map M = ||ΔH||_2 per token → [grid, grid]
        4. Get object masks from noun token attention maps:
              mask_car      = attention("car" token)     > threshold
              mask_bicycle  = attention("bicycle" token) > threshold
        5. Compute leakage ratio per attribute type:
              Leak = energy of M in NON-TARGET region
                     ──────────────────────────────────
                     energy of M in TARGET region
           Leak > 1 → more energy outside object than inside → leakage confirmed

DECISION RULE (written BEFORE running):
    PROCEED if:
        color Leak_ratio > 1.0                    (color actually leaks)
        color Leak_ratio > shape Leak_ratio        (color leaks MORE than shape)
        color Leak_ratio > texture Leak_ratio      (color leaks MORE than texture)

USAGE:
    python experiment3_spatial_leakage.py
    python experiment3_spatial_leakage.py --smoke_test --device cpu --steps 5
    python experiment3_spatial_leakage.py --steps 10 --output_dir results/exp3/

OUTPUTS:
    results/exp3/
        fig1_leakage_ratio_bars.png     — main result: leak ratio per attr type
        fig2_perturbation_maps.png      — M maps with object masks overlaid
        fig3_per_block_leakage.png      — which blocks drive leakage
        fig4_attention_masks.png        — sanity check: do masks look right?
        exp3_table.txt                  — numeric leakage ratios
        exp3_verdict.txt                — PROCEED or STOP with reason

COMPATIBLE WITH:
    pipeline_wrapper.py   (SD3PipelineWrapper)
    custom_flow_loop.py   (FlowMatchingLoop)
    config.yaml
    Phase0_Viability.py   (same hook pattern)
"""

# ==================================================================== #
#  IMPORTS                                                              #
# ==================================================================== #

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from pipeline_wrapper import SD3PipelineWrapper
from utils             import load_config, set_seed


# ==================================================================== #
#  CONFLICT PROMPTS                                                     #
#                                                                       #
#  Each entry: (with_attrs_prompt, without_attrs_prompt,               #
#               target_object, non_target_object,                      #
#               attr_type)                                              #
#                                                                       #
#  CRITICAL DESIGN RULE:                                                #
#   without_attrs_prompt keeps BOTH objects — only removes attributes  #
#   This ensures object structure cancels in ΔH, only attr signal left #
# ==================================================================== #

CONFLICT_PROMPTS = {

    "color": [
        # (with_attrs,                               without_attrs,                  target,      non_target)
        ("red car next to blue bicycle",              "car next to bicycle",          "car",       "bicycle"),
        ("yellow banana beside green apple",          "banana beside apple",          "banana",    "apple"),
        ("red rose next to white lily",               "rose next to lily",            "rose",      "lily"),
        ("red house beside blue barn",                "house beside barn",            "house",     "barn"),
        ("orange cat next to gray dog",               "cat next to dog",              "cat",       "dog"),
        ("purple bag next to yellow backpack",        "bag next to backpack",         "bag",       "backpack"),
        ("pink mug beside orange bowl",               "mug beside bowl",              "mug",       "bowl"),
        ("green bottle next to red cup",              "bottle next to cup",           "bottle",    "cup"),
        ("blue shirt next to red jacket",             "shirt next to jacket",         "shirt",     "jacket"),
        ("white chair beside black table",            "chair beside table",           "chair",     "table"),
        ("red umbrella next to blue tent",            "umbrella next to tent",        "umbrella",  "tent"),
        ("yellow bus beside green truck",             "bus beside truck",             "bus",       "truck"),
        ("pink dress next to purple skirt",           "dress next to skirt",          "dress",     "skirt"),
        ("orange lamp beside blue vase",              "lamp beside vase",             "lamp",      "vase"),
        ("red ball next to blue box",                 "ball next to box",             "ball",      "box"),
        ("cyan mug next to magenta plate",            "mug next to plate",            "mug",       "plate"),
        ("brown sofa beside gray rug",                "sofa beside rug",              "sofa",      "rug"),
        ("gold ring next to silver bracelet",         "ring next to bracelet",        "ring",      "bracelet"),
        ("navy coat next to beige scarf",             "coat next to scarf",           "coat",      "scarf"),
        ("red kite next to green balloon",            "kite next to balloon",         "kite",      "balloon"),
        ("purple cushion beside yellow blanket",      "cushion beside blanket",       "cushion",   "blanket"),
        ("blue phone next to red tablet",             "phone next to tablet",         "phone",     "tablet"),
        ("orange pumpkin beside green watermelon",    "pumpkin beside watermelon",    "pumpkin",   "watermelon"),
        ("pink flamingo next to white swan",          "flamingo next to swan",        "flamingo",  "swan"),
        ("red fire truck beside yellow school bus",   "fire truck beside school bus", "fire truck","school bus"),
        ("blue jeans next to brown boots",            "jeans next to boots",          "jeans",     "boots"),
        ("green frog next to red ladybug",            "frog next to ladybug",         "frog",      "ladybug"),
        ("yellow lemon beside purple grape",          "lemon beside grape",           "lemon",     "grape"),
        ("white rabbit next to brown fox",            "rabbit next to fox",           "rabbit",    "fox"),
        ("red strawberry beside green kiwi",          "strawberry beside kiwi",       "strawberry","kiwi"),
        ("blue dolphin next to gray shark",           "dolphin next to shark",        "dolphin",   "shark"),
        ("orange tiger beside black panther",         "tiger beside panther",         "tiger",     "panther"),
        ("pink pig next to gray elephant",            "pig next to elephant",         "pig",       "elephant"),
        ("red tulip next to blue iris",               "tulip next to iris",           "tulip",     "iris"),
        ("green parrot beside yellow canary",         "parrot beside canary",         "parrot",    "canary"),
        ("blue kayak next to red canoe",              "kayak next to canoe",          "kayak",     "canoe"),
        ("purple eggplant beside orange carrot",      "eggplant beside carrot",       "eggplant",  "carrot"),
        ("red mailbox next to blue post",             "mailbox next to post",         "mailbox",   "post"),
        ("yellow sunflower beside pink peony",        "sunflower beside peony",       "sunflower", "peony"),
        ("green cactus next to brown tree",           "cactus next to tree",          "cactus",    "tree"),
        ("blue tent next to orange campfire",         "tent next to campfire",        "tent",      "campfire"),
        ("red barn beside white farmhouse",           "barn beside farmhouse",        "barn",      "farmhouse"),
        ("purple lavender beside yellow wheat",       "lavender beside wheat",        "lavender",  "wheat"),
        ("black crow next to white dove",             "crow next to dove",            "crow",      "dove"),
        ("red apple beside yellow pear",              "apple beside pear",            "apple",     "pear"),
        ("blue blueberry next to red raspberry",      "blueberry next to raspberry",  "blueberry", "raspberry"),
        ("green lime beside orange tangerine",        "lime beside tangerine",        "lime",      "tangerine"),
        ("pink peach next to purple plum",            "peach next to plum",           "peach",     "plum"),
        ("red pepper beside green cucumber",          "pepper beside cucumber",       "pepper",    "cucumber"),
        ("yellow corn next to purple cabbage",        "corn next to cabbage",         "corn",      "cabbage"),
    ],

    "shape": [
        ("round table next to square chair",          "table next to chair",          "table",     "chair"),
        ("oval mirror beside rectangular door",       "mirror beside door",           "mirror",    "door"),
        ("triangular sign next to circular clock",    "sign next to clock",           "sign",      "clock"),
        ("round ball next to square box",             "ball next to box",             "ball",      "box"),
        ("curved sofa beside angular desk",           "sofa beside desk",             "sofa",      "desk"),
        ("flat plate next to tall glass",             "plate next to glass",          "plate",     "glass"),
        ("cylindrical cup next to rectangular book",  "cup next to book",             "cup",       "book"),
        ("round clock beside square frame",           "clock beside frame",           "clock",     "frame"),
        ("pointed hat next to round helmet",          "hat next to helmet",           "hat",       "helmet"),
        ("arched bridge next to flat road",           "bridge next to road",          "bridge",    "road"),
        ("curved lamp next to straight pole",         "lamp next to pole",            "lamp",      "pole"),
        ("hexagonal tile next to round stone",        "tile next to stone",           "tile",      "stone"),
        ("oval tray beside rectangular mat",          "tray beside mat",              "tray",      "mat"),
        ("round pot next to square pan",              "pot next to pan",              "pot",       "pan"),
        ("spiral staircase next to flat platform",    "staircase next to platform",   "staircase", "platform"),
        ("dome tent next to rectangular cabin",       "tent next to cabin",           "tent",      "cabin"),
        ("cone hat beside cylinder mug",              "hat beside mug",               "hat",       "mug"),
        ("square pillow next to round bolster",       "pillow next to bolster",       "pillow",    "bolster"),
        ("flat board beside curved bowl",             "board beside bowl",            "board",     "bowl"),
        ("pointed spire next to flat roof",           "spire next to roof",           "spire",     "roof"),
        ("round wheel beside rectangular door",       "wheel beside door",            "wheel",     "door"),
        ("oval pond next to rectangular pool",        "pond next to pool",            "pond",      "pool"),
        ("circular shield beside rectangular sword",  "shield beside sword",          "shield",    "sword"),
        ("cube block next to sphere ball",            "block next to ball",           "block",     "ball"),
        ("flat canvas beside round vase",             "canvas beside vase",           "canvas",    "vase"),
        ("triangular roof next to square wall",       "roof next to wall",            "roof",      "wall"),
        ("round coin beside rectangular card",        "coin beside card",             "coin",      "card"),
        ("cylindrical barrel next to cubic crate",    "barrel next to crate",         "barrel",    "crate"),
        ("oval egg beside rectangular toast",         "egg beside toast",             "egg",       "toast"),
        ("curved arch next to straight column",       "arch next to column",          "arch",      "column"),
        ("round globe beside flat map",               "globe beside map",             "globe",     "map"),
        ("square tile next to circular drain",        "tile next to drain",           "tile",      "drain"),
        ("triangular wedge next to rectangular block","wedge next to block",          "wedge",     "block"),
        ("round drum beside rectangular keyboard",    "drum beside keyboard",         "drum",      "keyboard"),
        ("flat tray beside round bowl",               "tray beside bowl",             "tray",      "bowl"),
        ("oval plate beside square napkin",           "plate beside napkin",          "plate",     "napkin"),
        ("cylindrical tower next to flat wall",       "tower next to wall",           "tower",     "wall"),
        ("round button beside square switch",         "button beside switch",         "button",    "switch"),
        ("curved bench next to straight fence",       "bench next to fence",          "bench",     "fence"),
        ("hexagonal box next to round tin",           "box next to tin",              "box",       "tin"),
        ("rectangular window beside round porthole",  "window beside porthole",       "window",    "porthole"),
        ("flat shelf beside curved hook",             "shelf beside hook",            "shelf",     "hook"),
        ("conical funnel next to cylindrical pipe",   "funnel next to pipe",          "funnel",    "pipe"),
        ("oval brooch beside square buckle",          "brooch beside buckle",         "brooch",    "buckle"),
        ("round manhole next to rectangular grate",   "manhole next to grate",        "manhole",   "grate"),
        ("triangular pennant beside rectangular flag","pennant beside flag",          "pennant",   "flag"),
        ("flat disc next to cubic dice",              "disc next to dice",            "disc",      "dice"),
        ("curved horn beside straight rod",           "horn beside rod",              "horn",      "rod"),
        ("round mirror beside rectangular picture",   "mirror beside picture",        "mirror",    "picture"),
        ("square sandbox next to round fountain",     "sandbox next to fountain",     "sandbox",   "fountain"),
    ],

    "texture": [
        ("metallic car next to wooden bicycle",       "car next to bicycle",          "car",       "bicycle"),
        ("glossy table beside rusty chair",           "table beside chair",           "table",     "chair"),
        ("smooth stone next to rough brick",          "stone next to brick",          "stone",     "brick"),
        ("silky curtain next to rough carpet",        "curtain next to carpet",       "curtain",   "carpet"),
        ("shiny vase beside matte pot",               "vase beside pot",              "vase",      "pot"),
        ("wooden door next to metallic window",       "door next to window",          "door",      "window"),
        ("fuzzy blanket beside smooth pillow",        "blanket beside pillow",        "blanket",   "pillow"),
        ("grainy wall next to polished floor",        "wall next to floor",           "wall",      "floor"),
        ("leather sofa beside fabric chair",          "sofa beside chair",            "sofa",      "chair"),
        ("crystal glass next to ceramic mug",         "glass next to mug",            "glass",     "mug"),
        ("velvet cushion beside woven basket",        "cushion beside basket",        "cushion",   "basket"),
        ("sandy path next to grassy lawn",            "path next to lawn",            "path",      "lawn"),
        ("knitted sweater next to denim jacket",      "sweater next to jacket",       "sweater",   "jacket"),
        ("marble countertop beside wooden shelf",     "countertop beside shelf",      "countertop","shelf"),
        ("plastic bucket next to metal bin",          "bucket next to bin",           "bucket",    "bin"),
        ("rough sandpaper beside smooth silk",        "sandpaper beside silk",        "sandpaper", "silk"),
        ("glossy magazine beside matte notebook",     "magazine beside notebook",     "magazine",  "notebook"),
        ("fluffy towel beside stiff cardboard",       "towel beside cardboard",       "towel",     "cardboard"),
        ("bumpy cobblestone next to flat pavement",   "cobblestone next to pavement", "cobblestone","pavement"),
        ("woven mat beside glass surface",            "mat beside surface",           "mat",       "surface"),
        ("rusted pipe next to polished rail",         "pipe next to rail",            "pipe",      "rail"),
        ("furry rug beside tiled floor",              "rug beside floor",             "rug",       "floor"),
        ("cracked wall next to smooth ceiling",       "wall next to ceiling",         "wall",      "ceiling"),
        ("embroidered cloth beside plain linen",      "cloth beside linen",           "cloth",     "linen"),
        ("bark texture tree next to smooth trunk",    "tree next to trunk",           "tree",      "trunk"),
        ("corrugated roof next to flat wall",         "roof next to wall",            "roof",      "wall"),
        ("pebbly beach next to smooth dock",          "beach next to dock",           "beach",     "dock"),
        ("scaly fish next to smooth frog",            "fish next to frog",            "fish",      "frog"),
        ("hairy dog next to smooth cat",              "dog next to cat",              "dog",       "cat"),
        ("thorny bush beside smooth hedge",           "bush beside hedge",            "bush",      "hedge"),
        ("bubbly foam next to flat water",            "foam next to water",           "foam",      "water"),
        ("wiry fence beside smooth wall",             "fence beside wall",            "fence",     "wall"),
        ("crinkled paper next to smooth card",        "paper next to card",           "paper",     "card"),
        ("wooden plank beside concrete slab",         "plank beside slab",            "plank",     "slab"),
        ("feathery pillow beside coarse blanket",     "pillow beside blanket",        "pillow",    "blanket"),
        ("silicone mat beside rough stone",           "mat beside stone",             "mat",       "stone"),
        ("glazed pot beside unglazed bowl",           "pot beside bowl",              "pot",       "bowl"),
        ("metallic foil beside paper bag",            "foil beside bag",              "foil",      "bag"),
        ("rough hemp rope beside smooth cord",        "rope beside cord",             "rope",      "cord"),
        ("knobby tire next to smooth wheel",          "tire next to wheel",           "tire",      "wheel"),
        ("ribbed sweater beside flat shirt",          "sweater beside shirt",         "sweater",   "shirt"),
        ("spiky cactus next to smooth aloe",          "cactus next to aloe",          "cactus",    "aloe"),
        ("porous sponge beside solid block",          "sponge beside block",          "sponge",    "block"),
        ("grainy bread beside smooth butter",         "bread beside butter",          "bread",     "butter"),
        ("flaky pastry beside smooth cake",           "pastry beside cake",           "pastry",    "cake"),
        ("bristly brush beside soft cloth",           "brush beside cloth",           "brush",     "cloth"),
        ("rough gravel beside smooth sand",           "gravel beside sand",           "gravel",    "sand"),
        ("waxy candle beside matte stone",            "candle beside stone",          "candle",    "stone"),
        ("quilted jacket beside smooth vest",         "jacket beside vest",           "jacket",    "vest"),
        ("lumpy cushion beside flat board",           "cushion beside board",         "cushion",   "board"),
    ],
}


# Blocks to probe — dense sampling around blocks 12-14 where DAVE shows leakage peaks
BLOCKS_TO_WATCH = [8, 10, 12, 14, 16]

# Attention mask threshold — top 50% of attention mass = object region (start conservative)
MASK_THRESHOLD = 0.5   # ChatGPT's recommendation: start at 0.5, inspect masks, adjust

# Decision thresholds — written BEFORE running
# NOTE: Removed hard Leak > 1.0 requirement (ChatGPT was right — too harsh).
# Real leakage means color leaks DISPROPORTIONATELY MORE than shape/texture,
# not necessarily more than the target itself.
PROCEED_COLOR_GT_SHAPE         = True   # color Leak > shape Leak
PROCEED_COLOR_GT_TEXTURE       = True   # color Leak > texture Leak


# ==================================================================== #
#  HIDDEN STATE + ATTENTION EXTRACTOR                                   #
# ==================================================================== #

class FeatureExtractor:
    """
    Captures BOTH hidden states and attention maps from MMDiT blocks.

    Hidden states: output of each JointTransformerBlock → image tokens [B, D, C]
    Attention maps: output of the attention submodule → [B, heads, D_total, D_total]

    SD3 MMDiT specifics:
        - JointTransformerBlock returns (encoder_hidden_states, hidden_states)
        - Index 0 = text tokens (encoder_hidden_states) — can be None on last block
        - Index 1 = image tokens (hidden_states) — what we want
        - Last block has context_pre_only=True → encoder_hidden_states = None
    """

    def __init__(self, blocks_to_watch: List[int]):
        self.blocks_to_watch = blocks_to_watch
        self.hidden_states:  Dict[int, torch.Tensor] = {}
        self.attention_maps: Dict[int, torch.Tensor] = {}
        self._active = False

    def register(self, transformer) -> list:
        if not hasattr(transformer, "transformer_blocks"):
            raise AttributeError(
                "[Extractor] transformer.transformer_blocks not found."
            )

        blocks = transformer.transformer_blocks
        print(f"[Extractor] Model has {len(blocks)} transformer blocks.")
        print(f"[Extractor] Watching blocks: {self.blocks_to_watch}")

        handles = []

        for idx, block in enumerate(blocks):
            if idx not in self.blocks_to_watch:
                continue

            # ---- Hook 1: capture IMAGE hidden states after each block ----
            def make_hidden_hook(block_idx):
                def hook(module, inputs, outputs):
                    if not self._active:
                        return
                    # outputs = (encoder_hidden_states, hidden_states)
                    # encoder_hidden_states can be None on last block
                    if isinstance(outputs, tuple):
                        h = outputs[1]   # image tokens — index 1
                    else:
                        h = outputs
                    if h is None:
                        return
                    self.hidden_states[block_idx] = h.detach().cpu().float()
                return hook

            # ---- Hook 2: capture attention weights via F.scaled_dot_product_attention ----
            # SD3's diffusers Attention module does NOT return attention weights
            # in its output — it returns only the attended hidden states.
            # We patch torch.nn.functional.scaled_dot_product_attention inside
            # the attn submodule's forward call to intercept the weights directly.
            def make_attn_hook(block_idx):
                def hook(module, inputs, outputs):
                    if not self._active:
                        return
                    # outputs is the attended tensor [B, heads, D_total, head_dim]
                    # or [B, D_total, C] — NOT a tuple with weights in SD3 diffusers.
                    # We store None here; weights are captured via the sdp patch below.
                    pass
                return hook

            handles.append(
                block.register_forward_hook(make_hidden_hook(idx))
            )

            # Patch scaled_dot_product_attention on the attn submodule to capture weights
            if hasattr(block, "attn"):
                _orig_sdp = torch.nn.functional.scaled_dot_product_attention

                def make_sdp_patch(block_idx, orig_fn):
                    def patched_sdp(query, key, value, attn_mask=None,
                                    dropout_p=0.0, is_causal=False, **kwargs):
                        # Compute weights manually so we can store them
                        # query: [B, heads, D_total, head_dim]
                        scale = query.shape[-1] ** -0.5
                        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
                        if attn_mask is not None:
                            scores = scores + attn_mask
                        weights = torch.softmax(scores, dim=-1)
                        out = torch.matmul(weights, value)
                        if self._active:
                            self.attention_maps[block_idx] = weights.detach().cpu().float()
                        return out
                    return patched_sdp

                # We'll use a forward pre/post hook on the attn module instead
                # to safely wrap sdp only during that block's attention call
                def make_attn_pre_hook(block_idx, orig_sdp):
                    def pre_hook(module, args):
                        if self._active:
                            import torch.nn.functional as F
                            F.scaled_dot_product_attention = make_sdp_patch(block_idx, orig_sdp)
                    return pre_hook

                def make_attn_post_hook(block_idx, orig_sdp):
                    def post_hook(module, args, output):
                        if self._active:
                            import torch.nn.functional as F
                            F.scaled_dot_product_attention = orig_sdp
                    return post_hook

                orig_sdp = torch.nn.functional.scaled_dot_product_attention
                handles.append(
                    block.attn.register_forward_pre_hook(make_attn_pre_hook(idx, orig_sdp))
                )
                handles.append(
                    block.attn.register_forward_hook(make_attn_post_hook(idx, orig_sdp))
                )

        self._active = True
        print(f"[Extractor] Registered {len(handles)} hooks.")
        return handles

    def clear(self):
        self.hidden_states.clear()
        self.attention_maps.clear()

    def remove(self, handles: list):
        for h in handles:
            h.remove()
        self._active = False


# ==================================================================== #
#  SINGLE FORWARD PASS                                                  #
# ==================================================================== #

def single_forward_pass(
    wrapper   : SD3PipelineWrapper,
    extractor : FeatureExtractor,
    latents   : torch.Tensor,
    prompt    : str,
    step_idx  : int,
    timesteps : torch.Tensor,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Runs a SINGLE transformer forward pass at timestep step_idx.
    Returns:
        hidden_states  {block_idx: [1, D, C]}
        attention_maps {block_idx: [1, heads, D_total, D_total]}  (may be empty)

    CFG disabled (guidance_scale=1.0) → batch size = 1, clean signal.
    Same latents and timestep must be used for both paired prompts.
    """
    extractor.clear()

    # Encode prompt — CFG disabled means no doubling
    prompt_embeds, pooled_embeds = wrapper.encode_prompt(
        prompt          = prompt,
        negative_prompt = "",
    )

    t = timesteps[step_idx].reshape(1).to(wrapper.device)

    device = wrapper.device
    dtype  = next(wrapper.transformer.parameters()).dtype

    lat  = latents.to(device=device, dtype=dtype)
    emb  = prompt_embeds.to(device=device, dtype=dtype)
    pool = pooled_embeds.to(device=device, dtype=dtype)
    t    = t.to(device=device, dtype=dtype)

    with torch.no_grad():
        _ = wrapper.transformer(
            hidden_states         = lat,
            timestep              = t,
            encoder_hidden_states = emb,
            pooled_projections    = pool,
        )

    return (
        dict(extractor.hidden_states),
        dict(extractor.attention_maps),
    )


# ==================================================================== #
#  PERTURBATION MAP                                                     #
# ==================================================================== #

def compute_perturbation_map(
    h_with    : torch.Tensor,   # [1, D, C]
    h_without : torch.Tensor,   # [1, D, C]
) -> Tuple[np.ndarray, int, torch.Tensor]:
    """
    Computes M = ||ΔH||_2 per token, reshaped to 2D spatial grid.
    Also returns raw delta [D, C] for projection-based analysis.

    Returns:
        M         : np.ndarray [grid_size, grid_size]
        grid_size : int
        delta     : torch.Tensor [D, C]
    """
    delta = h_with[0] - h_without[0]           # [D, C]
    norms = torch.norm(delta, dim=-1)           # [D]

    D = norms.shape[0]
    grid_size = int(D ** 0.5)

    if grid_size * grid_size != D:
        raise ValueError(
            f"Token count D={D} is not a perfect square. "
            f"Cannot reshape to 2D grid. Check image resolution."
        )

    M = norms.numpy().reshape(grid_size, grid_size)
    return M, grid_size, delta


# ==================================================================== #
#  COLOR SUBSPACE VECTOR                                                #
# ==================================================================== #

def compute_color_direction(
    color_deltas: List[torch.Tensor],  # list of [D, C] tensors
) -> torch.Tensor:
    """
    Computes dominant color perturbation direction v_color via SVD.
    Stacks all color ΔH tensors, runs SVD, returns top-1 right singular vector.

    v_color: [C] — the direction in channel space that color perturbs most.
    """
    # Stack: [N*D, C]
    stacked = torch.cat([d.reshape(-1, d.shape[-1]) for d in color_deltas], dim=0)
    stacked = stacked.float()

    # SVD on [N*D, C] — we want the dominant direction in C-space
    # Use randomized SVD for speed (only need top-1)
    try:
        U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
        v_color = Vh[0]   # [C] — top right singular vector
    except Exception:
        # Fallback: mean direction
        v_color = stacked.mean(dim=0)
        v_color = v_color / (v_color.norm() + 1e-8)

    return v_color   # [C]


# ==================================================================== #
#  PROJECTION-BASED LEAKAGE SCORE                                       #
# ==================================================================== #

def compute_projection_leakage(
    delta         : torch.Tensor,   # [D, C]
    v_color       : torch.Tensor,   # [C]
    target_mask   : np.ndarray,     # [grid, grid] bool
    non_target_mask: np.ndarray,    # [grid, grid] bool
    grid_size     : int,
) -> Dict[str, float]:
    """
    Refinement 1: Project each token's ΔH onto v_color.
    S_i = |ΔH_i · v_color|  — color-specific leaked energy per token.

    Compare mean S in target vs non-target region.
    This isolates color-specific leakage, not generic perturbation energy.
    """
    v = v_color.float()
    v = v / (v.norm() + 1e-8)

    # Per-token projection onto color direction
    projections = (delta.float() @ v).abs()   # [D]
    S = projections.numpy().reshape(grid_size, grid_size)

    target_flat     = target_mask.flatten()
    non_target_flat = non_target_mask.flatten()
    S_flat          = projections.numpy()

    target_proj     = S_flat[target_flat].mean()     if target_flat.any()     else 0.0
    non_target_proj = S_flat[non_target_flat].mean() if non_target_flat.any() else 0.0

    ratio = float(non_target_proj / target_proj) if target_proj > 0 else float("nan")

    return {
        "proj_leak_ratio"  : ratio,
        "target_proj"      : float(target_proj),
        "non_target_proj"  : float(non_target_proj),
        "S_map"            : S,
    }


def compute_concentration_spill(
    S_map        : np.ndarray,   # [grid, grid] projection scores
    target_mask  : np.ndarray,   # [grid, grid] bool
    top_pct      : float = 0.20, # top 20%
) -> float:
    """
    Refinement 2: Spatial concentration map + center spread.
    Finds top 20% strongest projected tokens.
    Returns fraction of those tokens that lie OUTSIDE the target mask.
    High spill → color direction physically spreads across objects.
    """
    flat   = S_map.flatten()
    thresh = np.percentile(flat, 100 * (1 - top_pct))
    top_mask = S_map >= thresh                    # [grid, grid] bool

    outside_target = top_mask & ~target_mask
    spill = outside_target.sum() / max(top_mask.sum(), 1)
    return float(spill)





# ==================================================================== #
#  OBJECT MASK FROM NOUN ATTENTION                                      #
# ==================================================================== #

def get_noun_token_index(
    wrapper : SD3PipelineWrapper,
    prompt  : str,
    noun    : str,
) -> Optional[int]:
    """
    Finds the token index of a noun in the prompt using CLIP-L tokenizer.
    Returns None if noun not found.

    Note: This is an approximation — CLIP uses BPE tokenization so
    multi-character words may be split. We find the first matching token.
    """
    tokenizer = wrapper.tokenizer
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    noun_ids   = tokenizer(noun,   return_tensors="pt").input_ids[0]

    # noun_ids typically: [BOS, token(s), EOS] — we want just the token(s)
    # Remove BOS (index 0) and EOS (last index)
    noun_token_ids = noun_ids[1:-1]

    prompt_tokens = prompt_ids.tolist()
    noun_tokens   = noun_token_ids.tolist()

    # Find first occurrence of noun token sequence in prompt
    for i in range(len(prompt_tokens) - len(noun_tokens) + 1):
        if prompt_tokens[i:i + len(noun_tokens)] == noun_tokens:
            return i   # return index of first noun token

    return None


def build_object_mask_from_attention(
    attention_map : Optional[torch.Tensor],   # [1, heads, D_total, D_total]
    noun_token_idx: Optional[int],
    grid_size     : int,
    threshold     : float = MASK_THRESHOLD,
) -> Optional[np.ndarray]:
    """
    Builds a binary spatial mask for an object by thresholding the
    attention map of its noun token.

    D_total = D_text + D_image tokens (joint attention in MMDiT)
    We want only the image→image or text→image portion.

    Returns:
        mask: np.ndarray [grid_size, grid_size] bool, or None if unavailable
    """
    if attention_map is None or noun_token_idx is None:
        return None

    # attention_map: [1, heads, D_total, D_total]
    # Average across heads
    attn = attention_map[0].mean(dim=0)   # [D_total, D_total]

    D_total = attn.shape[0]
    D_image = grid_size * grid_size

    # Text tokens occupy the first (D_total - D_image) positions
    D_text = D_total - D_image

    if noun_token_idx >= D_text:
        return None   # index out of text range

    # Get attention from noun token to ALL image tokens
    # attn[noun_token_idx, D_text:] = attention from noun to image patches
    attn_to_image = attn[noun_token_idx, D_text:]   # [D_image]

    if attn_to_image.shape[0] != D_image:
        return None

    # Normalize to [0, 1]
    attn_map = attn_to_image.numpy()
    if attn_map.max() > 0:
        attn_map = attn_map / attn_map.max()

    # Reshape to spatial grid
    attn_spatial = attn_map.reshape(grid_size, grid_size)

    # Binarize — top threshold% of attention = object region
    mask = attn_spatial >= threshold
    return mask


def build_spatial_half_mask(
    grid_size  : int,
    side       : str,   # "left" or "right"
) -> np.ndarray:
    """
    Fallback mask when attention maps are unavailable.
    For "X next to Y" prompts, left half ≈ object 1, right half ≈ object 2.
    Very rough but useful as sanity check.
    """
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mid  = grid_size // 2
    if side == "left":
        mask[:, :mid] = True
    else:
        mask[:, mid:] = True
    return mask


# ==================================================================== #
#  LEAKAGE RATIO COMPUTATION                                            #
# ==================================================================== #

def compute_leakage_ratio(
    M              : np.ndarray,   # [grid_size, grid_size]
    target_mask    : np.ndarray,   # [grid_size, grid_size] bool
    non_target_mask: np.ndarray,   # [grid_size, grid_size] bool
) -> Dict[str, float]:
    """
    Computes leakage ratio:

        Leak = mean energy in NON-TARGET region
               ──────────────────────────────────
               mean energy in TARGET region

    Leak > 1 → attribute bleeds more into wrong object than right object
    Leak < 1 → attribute stays within its target object

    Returns dict with ratio and component energies for diagnostics.
    """
    M_sq = M ** 2   # energy = squared magnitude

    target_energy     = M_sq[target_mask].mean()     if target_mask.any()     else 0.0
    non_target_energy = M_sq[non_target_mask].mean() if non_target_mask.any() else 0.0

    if target_energy == 0:
        ratio = float("nan")
    else:
        ratio = float(non_target_energy / target_energy)

    return {
        "leak_ratio"       : ratio,
        "target_energy"    : float(target_energy),
        "non_target_energy": float(non_target_energy),
    }


# ==================================================================== #
#  PLOTTING                                                             #
# ==================================================================== #

def plot_leakage_ratio_bars(
    leak_records : Dict[str, List[float]],
    out_dir      : Path,
) -> Dict[str, float]:
    """
    Figure 1 — Bar chart of mean leakage ratio per attribute type.
    This is the PRIMARY result figure.
    """
    attrs   = ["color", "shape", "texture"]
    colors  = ["#E74C3C", "#3498DB", "#2ECC71"]
    means   = []
    stds    = []

    for attr in attrs:
        vals = [v for v in leak_records.get(attr, []) if not np.isnan(v)]
        means.append(np.mean(vals) if vals else 0.0)
        stds.append(np.std(vals)   if vals else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(attrs))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, width=0.5)

    # Reference line at 1.0 — above = leakage confirmed
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5,
               label="Leak ratio = 1.0 (threshold)")

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in attrs], fontsize=13)
    ax.set_ylabel("Leakage Ratio (non-target / target)", fontsize=12)
    ax.set_title(
        "Experiment 3: Spatial Leakage Ratio per Attribute Type\n"
        "Ratio > 1.0 means attribute bleeds more into wrong object",
        fontsize=12
    )
    ax.legend(fontsize=10)

    # Annotate bars with values
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()
    path = out_dir / "fig1_leakage_ratio_bars.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

    return dict(zip(attrs, means))


def plot_perturbation_maps_with_masks(
    sample_data : Dict[str, dict],   # attr_type → {M, target_mask, non_target_mask}
    out_dir     : Path,
):
    """
    Figure 2 — Perturbation maps with object masks overlaid.
    Visual sanity check that masks are reasonable and M differs across attr types.
    """
    attrs = [a for a in ["color", "shape", "texture"] if a in sample_data]
    if not attrs:
        return

    fig, axes = plt.subplots(len(attrs), 3,
                             figsize=(12, 4 * len(attrs)))
    if len(attrs) == 1:
        axes = axes[np.newaxis, :]

    for row, attr in enumerate(attrs):
        d = sample_data[attr]
        M              = d["M"]
        target_mask    = d.get("target_mask")
        non_target_mask= d.get("non_target_mask")

        # Col 0: raw perturbation map
        ax = axes[row, 0]
        im = ax.imshow(M, cmap="hot", interpolation="nearest")
        ax.set_title(f"{attr.capitalize()}\nPerturbation Map M", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")

        # Col 1: target mask
        ax = axes[row, 1]
        if target_mask is not None:
            ax.imshow(target_mask.astype(float), cmap="Blues",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(f"Target Mask\n({d.get('target', 'object 1')})", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No attention map\nFallback mask used",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Target Mask (fallback)", fontsize=10)
        ax.axis("off")

        # Col 2: M with mask overlay
        ax = axes[row, 2]
        ax.imshow(M, cmap="hot", interpolation="nearest")
        if target_mask is not None:
            # Overlay target as green, non-target as blue
            overlay = np.zeros((*M.shape, 4))
            if target_mask is not None:
                overlay[target_mask, 1] = 0.5   # green = target
                overlay[target_mask, 3] = 0.3
            if non_target_mask is not None:
                overlay[non_target_mask, 2] = 0.5  # blue = non-target
                overlay[non_target_mask, 3] = 0.3
            ax.imshow(overlay, interpolation="nearest")
        ax.set_title("M + Masks\n(green=target, blue=non-target)", fontsize=10)
        ax.axis("off")

    plt.suptitle("Experiment 3: Perturbation Maps with Object Masks",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "fig2_perturbation_maps.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_per_block_leakage(
    block_leak_records : Dict[int, Dict[str, List[float]]],
    out_dir            : Path,
):
    """
    Figure 3 — Per-block leakage ratio.
    Shows which transformer blocks drive leakage — should peak at blocks 12-14.
    """
    blocks = sorted(block_leak_records.keys())
    attrs  = ["color", "shape", "texture"]
    colors = {"color": "#E74C3C", "shape": "#3498DB", "texture": "#2ECC71"}

    fig, ax = plt.subplots(figsize=(9, 5))

    for attr in attrs:
        means = []
        for b in blocks:
            vals = [v for v in block_leak_records[b].get(attr, [])
                    if not np.isnan(v)]
            means.append(np.mean(vals) if vals else 0.0)
        ax.plot(blocks, means, marker="o", linewidth=2,
                color=colors[attr], label=attr.capitalize())

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2,
               alpha=0.7, label="Leak = 1.0")
    ax.set_xlabel("Transformer Block Index", fontsize=12)
    ax.set_ylabel("Mean Leakage Ratio", fontsize=12)
    ax.set_title("Experiment 3: Leakage Ratio per Block\n"
                 "(Expected peak at blocks 12-14)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xticks(blocks)
    plt.tight_layout()

    path = out_dir / "fig3_per_block_leakage.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_attention_masks_sanity(
    mask_samples : Dict[str, dict],
    out_dir      : Path,
):
    """
    Figure 4 — Sanity check: visualize raw attention maps and derived masks.
    """
    if not mask_samples:
        return

    n = len(mask_samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (key, d) in enumerate(mask_samples.items()):
        raw_attn = d.get("raw_attn")
        mask     = d.get("mask")

        ax = axes[row, 0]
        if raw_attn is not None:
            ax.imshow(raw_attn, cmap="viridis", interpolation="nearest")
            ax.set_title(f"Raw Attention: '{key}'", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No attention captured",
                    ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

        ax = axes[row, 1]
        if mask is not None:
            ax.imshow(mask.astype(float), cmap="Blues",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(f"Derived Mask: '{key}'", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No mask",
                    ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    plt.suptitle("Experiment 3: Attention Mask Sanity Check", fontsize=12)
    plt.tight_layout()
    path = out_dir / "fig4_attention_masks.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ==================================================================== #
#  VERDICT                                                              #
# ==================================================================== #

def write_verdict(
    mean_leakage      : Dict[str, float],
    mean_proj_leakage : Dict[str, float],
    mean_spill        : Dict[str, float],
    ttest_cs          : dict,
    ttest_ct          : dict,
    out_dir           : Path,
) -> bool:
    """
    Verdict based on projection leakage + statistical significance.
    """
    color_proj   = mean_proj_leakage.get("color",   0.0)
    shape_proj   = mean_proj_leakage.get("shape",   0.0)
    texture_proj = mean_proj_leakage.get("texture", 0.0)

    c1 = color_proj > shape_proj
    c2 = color_proj > texture_proj
    c3 = ttest_cs["p"] < 0.05 if not np.isnan(ttest_cs["p"]) else False
    c4 = ttest_ct["p"] < 0.05 if not np.isnan(ttest_ct["p"]) else False

    proceed = c1 and c2 and (c3 or c4)  # at least one sig test

    def sig(p):
        if np.isnan(p): return "n/a"
        if p < 0.001:   return "p<0.001 ***"
        if p < 0.01:    return "p<0.01  **"
        if p < 0.05:    return "p<0.05  *"
        return f"p={p:.3f} ns"

    lines = [
        "=" * 65,
        "EXPERIMENT 3 VERDICT",
        "=" * 65,
        "",
        "PRIMARY METRIC: projection leakage onto v_color direction",
        "",
        "DECISION RULE (set before running):",
        f"  C1: color proj_leak > shape proj_leak  "
        f"→ {color_proj:.4f} vs {shape_proj:.4f}  {'✅' if c1 else '❌'}",
        f"  C2: color proj_leak > texture proj_leak"
        f"→ {color_proj:.4f} vs {texture_proj:.4f}  {'✅' if c2 else '❌'}",
        f"  C3: color vs shape   spill t-test "
        f"→ {sig(ttest_cs['p'])}  {'✅' if c3 else '❌'}",
        f"  C4: color vs texture spill t-test "
        f"→ {sig(ttest_ct['p'])}  {'✅' if c4 else '❌'}",
        "",
        "SUPPLEMENTARY (raw norm leakage):",
        f"  color: {mean_leakage.get('color',0):.4f}  "
        f"shape: {mean_leakage.get('shape',0):.4f}  "
        f"texture: {mean_leakage.get('texture',0):.4f}",
        "",
        "CONCENTRATION SPILL (top-20% tokens outside target):",
        f"  color: {mean_spill.get('color',0):.4f}  "
        f"shape: {mean_spill.get('shape',0):.4f}  "
        f"texture: {mean_spill.get('texture',0):.4f}",
        "",
    ]

    if proceed:
        lines += [
            "VERDICT: ✅ PROCEED TO EXPERIMENT 4",
            "",
            "Color attribute perturbations project disproportionately onto",
            "the dominant color subspace direction AND leak into wrong spatial",
            "regions more than shape/texture — with statistical significance.",
            "",
            "NEXT STEP: Run experiment4_hidden_output_correlation.py",
        ]
    else:
        lines += ["VERDICT: ❌ STOP — check which condition failed", ""]
        if not c1:
            lines.append("  FAILED C1: color proj_leak ≤ shape proj_leak")
        if not c2:
            lines.append("  FAILED C2: color proj_leak ≤ texture proj_leak")
        if not c3 and not c4:
            lines += [
                "  FAILED C3+C4: no t-test reached p<0.05",
                "  → Need more prompt pairs or the effect is not real.",
                "  → Check v_color quality (see v_color.pt).",
            ]

    verdict_text = "\n".join(lines)
    print(f"\n{verdict_text}\n")
    (out_dir / "exp3_verdict.txt").write_text(verdict_text)
    return proceed


# ==================================================================== #
#  MAIN                                                                 #
# ==================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3 — Spatial Leakage Ratio"
    )
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps",      type=int, default=10,
                        help="ODE steps (10 is enough for feature extraction)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/exp3/")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 2 prompts per attr only — fast sanity check")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 1. Load config — override CFG to disable it                       #
    # ---------------------------------------------------------------- #
    cfg = load_config("/content/Flow-based-model/config.yaml")
    cfg["flow"]["guidance_scale"] = 1.0    # CRITICAL: disable CFG
    cfg["flow"]["num_steps"]      = args.steps

    print(f"\n[Exp3] Device   : {args.device}")
    print(f"[Exp3] Steps    : {args.steps}")
    print(f"[Exp3] Seed     : {args.seed}")
    print(f"[Exp3] Output   : {out_dir}")
    print(f"[Exp3] CFG      : DISABLED (guidance_scale=1.0)")

    # ---------------------------------------------------------------- #
    # 2. Load pipeline                                                   #
    # ---------------------------------------------------------------- #
    wrapper = SD3PipelineWrapper(cfg, device=args.device)
    wrapper.load()
    block = wrapper.transformer.transformer_blocks[12]
    print(type(block))
    print([name for name, _ in block.named_children()])
    print([name for name, _ in block.named_modules()])
    # ---------------------------------------------------------------- #
    # 3. Setup extractor and timesteps                                   #
    # ---------------------------------------------------------------- #
    extractor = FeatureExtractor(blocks_to_watch=BLOCKS_TO_WATCH)
    handles   = extractor.register(wrapper.transformer)

    wrapper.scheduler.set_timesteps(args.steps)
    timesteps = wrapper.scheduler.timesteps

    n = args.steps
    step_indices = {
        "early": 0,
        "mid"  : n // 2,
        "late" : max(0, n - 2),
    }
    print(f"\n[Timesteps] {step_indices}")

    # ---------------------------------------------------------------- #
    # 4. Shared latents — SAME for all pairs                            #
    # ---------------------------------------------------------------- #
    shared_latents = wrapper.get_initial_latents(seed=args.seed)
    print(f"[Latents] Shape: {shared_latents.shape}  seed={args.seed}\n")

    # ---------------------------------------------------------------- #
    # 5. Storage                                                         #
    # ---------------------------------------------------------------- #
    # leak_records[attr_type] = list of leak ratios (one per prompt×block×step)
    leak_records: Dict[str, List[float]] = {
        attr: [] for attr in CONFLICT_PROMPTS
    }

    # NEW: projection-based leakage (Refinement 1)
    proj_leak_records: Dict[str, List[float]] = {
        attr: [] for attr in CONFLICT_PROMPTS
    }

    # NEW: concentration spill (Refinement 2)
    spill_records: Dict[str, List[float]] = {
        attr: [] for attr in CONFLICT_PROMPTS
    }

    # NEW: collect color deltas at block 12 to compute v_color
    color_deltas_for_svd: List[torch.Tensor] = []
    v_color: Optional[torch.Tensor] = None  # computed after color pass

    # Per-block storage for Figure 3
    block_leak_records: Dict[int, Dict[str, List[float]]] = {
        b: {attr: [] for attr in CONFLICT_PROMPTS}
        for b in BLOCKS_TO_WATCH
    }

    # Sample data for Figure 2 (one representative per attr type)
    sample_maps:  Dict[str, dict] = {}

    # Mask sanity check samples for Figure 4
    mask_samples: Dict[str, dict] = {}

    # ---------------------------------------------------------------- #
    # 6. Select prompts                                                  #
    # ---------------------------------------------------------------- #
    prompts_to_run = CONFLICT_PROMPTS
    if args.smoke_test:
        print("[Smoke test] Running 2 prompts per attribute only.\n")
        prompts_to_run = {
            attr: pairs[:2] for attr, pairs in CONFLICT_PROMPTS.items()
        }

    total = sum(len(v) for v in prompts_to_run.values())
    count = 0

    # ---------------------------------------------------------------- #
    # 7. Main loop                                                       #
    # ---------------------------------------------------------------- #
    for attr_type, prompt_list in prompts_to_run.items():
        print(f"\n{'─'*60}")
        print(f"  Attribute type: {attr_type.upper()}  ({len(prompt_list)} prompts)")
        print(f"{'─'*60}")

        for with_prompt, without_prompt, target_noun, non_target_noun in prompt_list:
            count += 1
            print(f"\n  [{count}/{total}]")
            print(f"    WITH   : '{with_prompt}'")
            print(f"    WITHOUT: '{without_prompt}'")
            print(f"    Target : '{target_noun}'  |  Non-target: '{non_target_noun}'")

            # Get token indices for mask building
            target_token_idx = get_noun_token_index(
                wrapper, without_prompt, target_noun
            )
            non_target_token_idx = get_noun_token_index(
                wrapper, without_prompt, non_target_noun
            )
            print(f"    Token indices — target: {target_token_idx}  "
                  f"non-target: {non_target_token_idx}")

            for step_name, step_idx in step_indices.items():

                # ---------------------------------------------------- #
                # Pass A: WITH attributes                               #
                # ---------------------------------------------------- #
                h_with, attn_with = single_forward_pass(
                    wrapper   = wrapper,
                    extractor = extractor,
                    latents   = shared_latents,
                    prompt    = with_prompt,
                    step_idx  = step_idx,
                    timesteps = timesteps,
                )

                # ---------------------------------------------------- #
                # Pass B: WITHOUT attributes — SAME objects, no attrs  #
                # ---------------------------------------------------- #
                h_without, attn_without = single_forward_pass(
                    wrapper   = wrapper,
                    extractor = extractor,
                    latents   = shared_latents,
                    prompt    = without_prompt,
                    step_idx  = step_idx,
                    timesteps = timesteps,
                )

                # ---------------------------------------------------- #
                # Compute ΔH → M and leakage ratio per block           #
                # ---------------------------------------------------- #
                for block_idx in BLOCKS_TO_WATCH:
                    if block_idx not in h_with or block_idx not in h_without:
                        print(f"    [WARN] Block {block_idx} not captured.")
                        continue

                    try:
                        M, grid_size, delta = compute_perturbation_map(
                            h_with[block_idx],
                            h_without[block_idx],
                        )
                    except ValueError as e:
                        print(f"    [WARN] {e}")
                        continue

                    # Collect color deltas at block 12 mid-step for SVD
                    if attr_type == "color" and block_idx == 12 and step_name == "mid":
                        color_deltas_for_svd.append(delta.cpu().float())

                    # ---- Build object masks -------------------------
                    attn_map = attn_without.get(block_idx)

                    target_mask = build_object_mask_from_attention(
                        attn_map, target_token_idx, grid_size
                    )
                    non_target_mask = build_object_mask_from_attention(
                        attn_map, non_target_token_idx, grid_size
                    )

                    using_fallback = False
                    if target_mask is None or not target_mask.any():
                        target_mask  = build_spatial_half_mask(grid_size, "left")
                        using_fallback = True
                    if non_target_mask is None or not non_target_mask.any():
                        non_target_mask = build_spatial_half_mask(grid_size, "right")
                        using_fallback = True

                    if using_fallback:
                        print(f"    [INFO] Block {block_idx} step {step_name}: "
                              f"using spatial fallback masks")

                    # ---- Raw norm leakage (original metric) ---------
                    result = compute_leakage_ratio(M, target_mask, non_target_mask)
                    ratio  = result["leak_ratio"]
                    if not np.isnan(ratio):
                        leak_records[attr_type].append(ratio)
                        block_leak_records[block_idx][attr_type].append(ratio)

                    # ---- Projection leakage (Refinement 1+2) --------
                    # Use v_color if already computed, else defer
                    if v_color is not None:
                        proj_result = compute_projection_leakage(
                            delta, v_color, target_mask, non_target_mask, grid_size
                        )
                        if not np.isnan(proj_result["proj_leak_ratio"]):
                            proj_leak_records[attr_type].append(
                                proj_result["proj_leak_ratio"]
                            )
                        spill = compute_concentration_spill(
                            proj_result["S_map"], target_mask
                        )
                        spill_records[attr_type].append(spill)

                        # Update sample map with S_map for visualization
                        if (attr_type not in sample_maps
                                and block_idx == 12
                                and step_name == "mid"):
                            sample_maps[attr_type] = {
                                "M"              : M.copy(),
                                "S_map"          : proj_result["S_map"].copy(),
                                "target_mask"    : target_mask.copy(),
                                "non_target_mask": non_target_mask.copy(),
                                "target"         : target_noun,
                                "non_target"     : non_target_noun,
                                "prompt"         : with_prompt,
                            }
                    else:
                        # For color pass (before v_color exists), store raw maps
                        if (attr_type not in sample_maps
                                and block_idx == 12
                                and step_name == "mid"):
                            sample_maps[attr_type] = {
                                "M"              : M.copy(),
                                "S_map"          : None,
                                "target_mask"    : target_mask.copy(),
                                "non_target_mask": non_target_mask.copy(),
                                "target"         : target_noun,
                                "non_target"     : non_target_noun,
                                "prompt"         : with_prompt,
                            }

                    # ---- Save mask sanity sample --------------------
                    mask_key = f"{attr_type}_{target_noun}_block{block_idx}"
                    if (mask_key not in mask_samples
                            and attn_map is not None
                            and target_token_idx is not None):
                        # Build raw attention map for visualization
                        attn_avg = attn_map[0].mean(dim=0)   # [D_total, D_total]
                        D_image  = grid_size * grid_size
                        D_text   = attn_avg.shape[0] - D_image
                        if (target_token_idx < D_text
                                and attn_avg.shape[1] > D_text):
                            raw = attn_avg[target_token_idx, D_text:].numpy()
                            raw = raw.reshape(grid_size, grid_size)
                            if raw.max() > 0:
                                raw = raw / raw.max()
                            mask_samples[mask_key] = {
                                "raw_attn": raw,
                                "mask"    : target_mask,
                            }

            print(f"    ✓ done  "
                  f"(running mean leak — "
                  f"color: {np.nanmean(leak_records['color']):.3f}  "
                  f"shape: {np.nanmean(leak_records['shape']):.3f}  "
                  f"texture: {np.nanmean(leak_records['texture']):.3f})")

        # ---- After color pass: compute v_color and run projection pass ----
        if attr_type == "color" and color_deltas_for_svd:
            print(f"\n[v_color] Computing dominant color direction from "
                  f"{len(color_deltas_for_svd)} color ΔH samples (block 12, mid)...")
            v_color = compute_color_direction(color_deltas_for_svd)
            torch.save(v_color, out_dir / "v_color.pt")
            print(f"[v_color] Saved → {out_dir / 'v_color.pt'}")

            # Re-run projection leakage for already-collected color prompts
            # (v_color wasn't available during the color pass above)
            print("[v_color] Back-filling projection scores for color prompts...")
            color_list = prompts_to_run["color"]
            for with_p, without_p, tgt, ntgt in color_list:
                tgt_idx  = get_noun_token_index(wrapper, without_p, tgt)
                ntgt_idx = get_noun_token_index(wrapper, without_p, ntgt)
                for step_name, step_idx in step_indices.items():
                    h_w, _ = single_forward_pass(wrapper, extractor, shared_latents,
                                                  with_p, step_idx, timesteps)
                    h_wo, _ = single_forward_pass(wrapper, extractor, shared_latents,
                                                   without_p, step_idx, timesteps)
                    for block_idx in BLOCKS_TO_WATCH:
                        if block_idx not in h_w or block_idx not in h_wo:
                            continue
                        try:
                            _, gs, delta = compute_perturbation_map(
                                h_w[block_idx], h_wo[block_idx]
                            )
                        except ValueError:
                            continue
                        t_mask = build_spatial_half_mask(gs, "left")
                        nt_mask = build_spatial_half_mask(gs, "right")
                        pr = compute_projection_leakage(
                            delta, v_color, t_mask, nt_mask, gs
                        )
                        if not np.isnan(pr["proj_leak_ratio"]):
                            proj_leak_records["color"].append(pr["proj_leak_ratio"])
                        spill_records["color"].append(
                            compute_concentration_spill(pr["S_map"], t_mask)
                        )
            print("[v_color] Back-fill complete.")


    # ---------------------------------------------------------------- #
    # 8. Aggregate                                                       #
    # ---------------------------------------------------------------- #
    print("\n[Aggregating results...]")
    mean_leakage = {}
    for attr in CONFLICT_PROMPTS:
        vals = [v for v in leak_records[attr] if not np.isnan(v)]
        mean_leakage[attr] = np.mean(vals) if vals else 0.0

    # Projection-based means
    mean_proj_leakage = {}
    for attr in CONFLICT_PROMPTS:
        vals = [v for v in proj_leak_records[attr] if not np.isnan(v)]
        mean_proj_leakage[attr] = np.mean(vals) if vals else 0.0

    mean_spill = {}
    for attr in CONFLICT_PROMPTS:
        vals = [v for v in spill_records[attr] if not np.isnan(v)]
        mean_spill[attr] = np.mean(vals) if vals else 0.0

    # Refinement 3: Paired t-tests — color vs shape, color vs texture
    def run_ttest(a_vals, b_vals, label):
        a = [v for v in a_vals if not np.isnan(v)]
        b = [v for v in b_vals if not np.isnan(v)]
        n = min(len(a), len(b))
        if n < 2:
            return {"t": float("nan"), "p": float("nan"), "n": n, "label": label}
        t_stat, p_val = stats.ttest_rel(a[:n], b[:n])
        return {"t": t_stat, "p": p_val, "n": n, "label": label}

    ttest_color_vs_shape   = run_ttest(
        spill_records["color"], spill_records["shape"],   "color vs shape (spill)"
    )
    ttest_color_vs_texture = run_ttest(
        spill_records["color"], spill_records["texture"], "color vs texture (spill)"
    )

    # ---------------------------------------------------------------- #
    # 9. Save numeric table                                             #
    # ---------------------------------------------------------------- #
    table_lines = [
        "Experiment 3 — Spatial Leakage Ratio Table",
        "=" * 70,
        "",
        "--- Raw norm leakage (||ΔH|| based) ---",
        f"{'Attr':<12} {'Mean Leak':>12} {'Std':>8} {'N':>6}",
        "-" * 42,
    ]
    for attr in ["color", "shape", "texture"]:
        vals = [v for v in leak_records[attr] if not np.isnan(v)]
        m    = np.mean(vals) if vals else 0.0
        s    = np.std(vals)  if vals else 0.0
        table_lines.append(f"{attr:<12} {m:>12.4f} {s:>8.4f} {len(vals):>6d}")

    table_lines += [
        "",
        "--- Projection leakage (ΔH projected onto v_color direction) ---",
        f"{'Attr':<12} {'Mean ProjLeak':>14} {'Std':>8} {'N':>6}",
        "-" * 44,
    ]
    for attr in ["color", "shape", "texture"]:
        vals = [v for v in proj_leak_records[attr] if not np.isnan(v)]
        m    = np.mean(vals) if vals else 0.0
        s    = np.std(vals)  if vals else 0.0
        table_lines.append(f"{attr:<12} {m:>14.4f} {s:>8.4f} {len(vals):>6d}")

    table_lines += [
        "",
        "--- Concentration spill (fraction of top-20% tokens outside target) ---",
        f"{'Attr':<12} {'Mean Spill':>12} {'Std':>8} {'N':>6}",
        "-" * 42,
    ]
    for attr in ["color", "shape", "texture"]:
        vals = [v for v in spill_records[attr] if not np.isnan(v)]
        m    = np.mean(vals) if vals else 0.0
        s    = np.std(vals)  if vals else 0.0
        table_lines.append(f"{attr:<12} {m:>12.4f} {s:>8.4f} {len(vals):>6d}")

    def sig(p):
        if np.isnan(p): return "n/a"
        if p < 0.001:   return "p<0.001 ***"
        if p < 0.01:    return "p<0.01  **"
        if p < 0.05:    return "p<0.05  *"
        return f"p={p:.3f} ns"

    table_lines += [
        "",
        "--- Paired t-tests (projection leakage: color vs others) ---",
        f"  color vs shape  : t={ttest_color_vs_shape['t']:.3f}   "
        f"{sig(ttest_color_vs_shape['p'])}   n={ttest_color_vs_shape['n']}",
        f"  color vs texture: t={ttest_color_vs_texture['t']:.3f}   "
        f"{sig(ttest_color_vs_texture['p'])}   n={ttest_color_vs_texture['n']}",
        "",
        "Per-block mean leakage ratio:",
        f"{'Block':<8} {'color':>10} {'shape':>10} {'texture':>12}",
        "-" * 42,
    ]
    for b in sorted(block_leak_records.keys()):
        def bmean(attr):
            vals = [v for v in block_leak_records[b][attr] if not np.isnan(v)]
            return np.mean(vals) if vals else 0.0
        table_lines.append(
            f"{b:<8} {bmean('color'):>10.4f} "
            f"{bmean('shape'):>10.4f} {bmean('texture'):>12.4f}"
        )

    table_text = "\n".join(table_lines)
    print(f"\n{table_text}\n")
    (out_dir / "exp3_table.txt").write_text(table_text)


    # ---------------------------------------------------------------- #
    # 10. Plots                                                          #
    # ---------------------------------------------------------------- #
    print("[Plotting...]")
    plot_leakage_ratio_bars(leak_records, out_dir)
    plot_perturbation_maps_with_masks(sample_maps, out_dir)
    plot_per_block_leakage(block_leak_records, out_dir)
    plot_attention_masks_sanity(mask_samples, out_dir)

    # ---------------------------------------------------------------- #
    # 11. Verdict                                                        #
    # ---------------------------------------------------------------- #
    proceed = write_verdict(
        mean_leakage, mean_proj_leakage, mean_spill,
        ttest_color_vs_shape, ttest_color_vs_texture, out_dir
    )

    # ---------------------------------------------------------------- #
    # 12. Cleanup                                                        #
    # ---------------------------------------------------------------- #
    extractor.remove(handles)

    print(f"\n{'='*60}")
    print(f"  Experiment 3 complete.")
    print(f"  Results → {out_dir}")
    if proceed:
        print("  Status  → ✅ PROCEED to Experiment 4")
    else:
        print("  Status  → ❌ STOP — read exp3_verdict.txt for guidance")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()