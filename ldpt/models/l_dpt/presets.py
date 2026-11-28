"""
Named L-DPT configurations: one class (L-DPT-v1m1), the released models as presets.

A preset is the full constructor kwargs of LatentDPT as a plain dict (every option
spelled out, so it does not move when a default changes). The names (and the weight
files, <name>.pth) read l-dpt_<decoder size>_<task>_<training data>:

    "l-dpt_large_depth_zeroshot"  L-DPT Large (zero-shot): trained on synthetic /
        multi-view data only (configs/hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py)
    "l-dpt_large_depth_mixed"     L-DPT Large (mixed): + the NYUv2 / KITTI / DIODE
        training splits (configs/hybrid/depth-l-dpt-v1m1-1-large-mixed.py,
        depth-l-dpt-v1m1-1a-large-mixed-anneal.py)
    "l-dpt_large_semseg_cocostuff"  L-DPT Large, COCO-Stuff semantic segmentation
        (171 classes, CLASS_NAMES)

Both depth models share one architecture (RELEASE_L) and differ only in their weights:
frozen DINOv3 ViT-7B/16 tapped at blocks [9, 19, 29, 39], latent decoder 1024 / 16
heads / (4, 4, 4, 4) blocks in bf16, exponential depth head with an fp32 readout, fp32
attention in the last decoder stage, an attention-logit bound of 1000, trained at short
side 768 with the L-DPT depth loss (docs/depth_loss.md). The segmentation model
(SEMSEG_L) taps the same encoder: latent decoder 2048 / 16 heads / (2, 2, 2, 2) blocks
with attention at width 1024, cosine attention with a learned temperature per head, a
BatchNorm on each tapped layer, 64 pixel features and a 1x1 classifier, decoder RoPE
coordinates normalised per axis, trained on 768 x 768 crops and evaluated on the whole
image at short side 768. Inference only (criterion=False): the training losses live in
the configs. In a config:
    model = dict(type="L-DPT-v1m1", **get_preset("l-dpt_large_depth_zeroshot"))

WEIGHTS pins the released decoder checkpoints (Hugging Face repo, revision, file,
sha256, and the variable naming a local copy; a slot of TBD values is not published
yet: the segmentation slot until its file is uploaded and pinned); DINOv3 weights are
not redistributed: weights.encoder_weights() takes a local file and checks the sha256
prefix in its official name. LEGACY_PRESETS maps the short
preset names of earlier release files ("l-zs", "l-mix") to the current names. This
module imports no torch.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
import difflib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ldpt.utils.seg_image import COCOSTUFF_CLASS_NAMES

__all__ = [
    "CLASS_NAMES",
    "DINOV3_SHA256_PREFIX",
    "IMAGE_MEAN",
    "IMAGE_STD",
    "INPUT_SIZE",
    "LEGACY_PRESETS",
    "PRESETS",
    "RELEASE_L",
    "SEMSEG_L",
    "WEIGHTS",
    "WeightSource",
    "canonical_preset",
    "get_preset",
]

TAPS_7B = [9, 19, 29, 39]

# the depth architecture, shared by both depth models (they differ only in their
# weights): the model of configs/hybrid/*.py (tests/test_configs.py checks it)
RELEASE_L: Dict[str, Any] = dict(
    encoder=dict(type="dinov3_vit7b"),
    encoder_inter_layers=TAPS_7B,
    freeze_encoder=True,
    encoder_train_mode=True,
    encoder_weight_dtype="bf16",  # frozen encoder stored in bf16 (bit-identical)
    task="relative_depth",
    out_dim=1,
    dec_dim=1024,
    dec_num_heads=16,
    dec_num_layers=(4, 4, 4, 4),
    dec_attn_downscale=1,
    dec_ffn_ratio=4.0,
    dec_pos_embed_rope_base=100,
    dec_pos_embed_rope_normalize_coords="min",
    dec_pos_embed_rope_shift_coords=None,
    dec_pos_embed_rope_jitter_coords=1.2,  # training only
    dec_pos_embed_rope_rescale_coords=8,  # training only
    dec_layer_scale_init=1.0e-05,
    dec_qkv_bias=False,
    dec_proj_bias=True,
    dec_ffn_bias=True,
    selection="stage",
    readout_kernel_size=3,
    precision="bf16",  # decoder under bf16 autocast; heads in fp32
    attn_dtype="bf16",
    head="exp",
    readout_fp32=True,
    attn_fp32="last_stage",
    block_fp32=None,
    dec_qk_bound=1000,
    dec_qk_norm=False,
    dec_conv_mixer=None,
    criterion=False,  # inference only: the training loss is in the configs
)

# the segmentation architecture: the COCO-Stuff model without its training loss
SEMSEG_L: Dict[str, Any] = dict(
    encoder=dict(type="dinov3_vit7b"),
    encoder_inter_layers=TAPS_7B,
    freeze_encoder=True,
    encoder_train_mode=True,
    encoder_weight_dtype="bf16",  # frozen encoder stored in bf16 (bit-identical)
    task="semseg",
    out_dim=64,  # pixel features before the 1x1 classifier (seg_head)
    num_classes=171,
    dec_dim=2048,
    dec_num_heads=16,
    dec_num_layers=(2, 2, 2, 2),
    dec_attn_downscale=2,  # attention at width 1024
    dec_ffn_ratio=4.0,
    dec_pos_embed_rope_base=100,
    dec_pos_embed_rope_normalize_coords="separate",  # per axis, as evaluated
    dec_pos_embed_rope_shift_coords=None,
    dec_pos_embed_rope_jitter_coords=None,
    dec_pos_embed_rope_rescale_coords=2,  # training only
    dec_layer_scale_init=1.0e-05,
    dec_qkv_bias=False,
    dec_proj_bias=True,
    dec_ffn_bias=True,
    selection="stage",
    readout_kernel_size=3,
    precision="bf16",  # decoder under bf16 autocast; classifier in fp32
    attn_dtype="fp32",
    readout_fp32=False,
    attn_fp32=None,
    block_fp32=None,
    dec_qk_bound=None,
    dec_qk_norm=True,  # cosine attention, learned temperature per head
    dec_qk_tau_max=200.0,
    dec_conv_mixer=None,
    tap_bn=True,  # a BatchNorm on each tapped encoder layer
    criterion=False,  # inference only: the training loss is in the configs
)

PRESETS: Dict[str, Dict[str, Any]] = {
    "l-dpt_large_depth_zeroshot": RELEASE_L,
    "l-dpt_large_depth_mixed": RELEASE_L,
    "l-dpt_large_semseg_cocostuff": SEMSEG_L,
}

# the short preset names of earlier release files (their "preset" field): read as
# the current names, never accepted as a preset
LEGACY_PRESETS: Dict[str, str] = {
    "l-zs": "l-dpt_large_depth_zeroshot",
    "l-mix": "l-dpt_large_depth_mixed",
}

# short side of the val pipeline (bicubic resize), per preset: the training resolution
INPUT_SIZE: Dict[str, int] = {
    "l-dpt_large_depth_zeroshot": 768,
    "l-dpt_large_depth_mixed": 768,
    "l-dpt_large_semseg_cocostuff": 768,
}
# the class of each output channel, per segmentation preset (dataset train-id order)
CLASS_NAMES: Dict[str, Tuple[str, ...]] = {
    "l-dpt_large_semseg_cocostuff": tuple(COCOSTUFF_CLASS_NAMES),
}
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# official DINOv3 file names end in the first 8 hex digits of their sha256
DINOV3_SHA256_PREFIX: Dict[str, str] = {
    "dinov3_vit7b": "a955f4ea",  # dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
    "dinov3_vith": "7c1da9a5",  # dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
    "dinov3_vitl": "8aa4cbdd",  # dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
}


TBD = "TBD"  # the value of a WeightSource field that is not pinned


@dataclass(frozen=True)
class WeightSource:
    """A decoder checkpoint on Hugging Face, pinned by revision and sha256. ``env``
    names a variable that may point at a local copy of the same file."""

    repo: str
    file: str
    revision: str
    sha256: str
    size: int  # bytes
    env: str
    format: str = "training"  # a training checkpoint, or "release": convert.py output

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/blob/{self.revision}/{self.file}"

    @property
    def published(self) -> bool:
        """False for a slot whose repo, revision or sha256 is not pinned yet (TBD)."""
        return TBD not in (self.repo, self.revision, self.sha256) and self.size > 0


# the released decoder checkpoints (convert.py files) on Hugging Face
RELEASE_REPO = "pointcept-bot/L-DPT"
RELEASE_REVISION = "0b212646e6bf7f2e0007c98ba5ed902408bfc27c"
WEIGHTS: Dict[str, WeightSource] = {
    "l-dpt_large_depth_zeroshot": WeightSource(
        repo=RELEASE_REPO,
        file="l-dpt_large_depth_zeroshot.pth",
        revision=RELEASE_REVISION,
        sha256="b4e7e95301fda7557760f787e7992f4ccca3c03cb70e46eadeb48e227b1b31ca",
        size=1_084_064_148,
        env="LDPT_CKPT_LARGE_DEPTH_ZEROSHOT",
        format="release",
    ),
    "l-dpt_large_depth_mixed": WeightSource(
        repo=RELEASE_REPO,
        file="l-dpt_large_depth_mixed.pth",
        revision=RELEASE_REVISION,
        sha256="4852b823c68fea30b3360a249e71daf72a9c6c2c74f747602a117b42f35d36fa",
        size=1_084_064_405,
        env="LDPT_CKPT_LARGE_DEPTH_MIXED",
        format="release",
    ),
    "l-dpt_large_semseg_cocostuff": WeightSource(
        repo=RELEASE_REPO,
        file="l-dpt_large_semseg_cocostuff.pth",
        revision="ef037204d77970fb4dbf65ee1f2a5a36b509143f",
        sha256="034397a1ef38e0aa190ebea1ff1c133d4f7bb7250d781d9019a240a65f2b1bb7",
        size=3_088_255_244,
        env="LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF",
        format="release",
    ),
}


def suggest(name: str, choices) -> str:
    """' (did you mean 'x'?)' for the closest choice, or ''."""
    close = difflib.get_close_matches(name, list(choices), n=1, cutoff=0.5)
    return f" (did you mean {close[0]!r}?)" if close else ""


def suggest_preset(name: str) -> str:
    """``suggest`` among the presets; for a name of LEGACY_PRESETS, its new name."""
    if name in LEGACY_PRESETS:
        return f" (renamed {LEGACY_PRESETS[name]!r})"
    return suggest(name, PRESETS)


def canonical_preset(name: Optional[str]) -> Optional[str]:
    """The current name of a preset name stored in a release file (LEGACY_PRESETS),
    else ``name`` unchanged (None stays None)."""
    return LEGACY_PRESETS.get(name, name) if name is not None else None


def get_preset(name: str, **overrides: Any) -> Dict[str, Any]:
    """A deep copy of preset ``name`` with ``overrides`` applied (unknown keys of the
    model raise when it is built)."""
    if name not in PRESETS:
        raise ValueError(
            f"unknown L-DPT preset {name!r}{suggest_preset(name)}; "
            f"available: {sorted(PRESETS)}"
        )
    kwargs = copy.deepcopy(PRESETS[name])
    kwargs.update(copy.deepcopy(overrides))
    return kwargs


def weight_source(name: str) -> Optional[WeightSource]:
    """The pinned weights of preset ``name``, None for a preset without released weights."""
    get_preset(name)
    return WEIGHTS.get(name)
