"""
Inference API: ``load`` a ready-to-run L-DPT, ``predict`` relative depth or semantic
segmentation of an image.

    from ldpt.models.l_dpt import api

    model = api.load(
        "l-dpt_large_depth_zeroshot",
        encoder_weights="dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    )
    depth = api.predict(model, image)   # HxWx3 uint8 (or float in [0, 1]) -> HxW float32
    depth = api.predict(model, image, size=None)  # at the image's native resolution
    depth = api.predict(model, image, size=None, encoder_tiles=None)  # one encoder pass

    seg = api.load("l-dpt_large_semseg_cocostuff.pth")  # a segmentation release file
    classes = api.predict(seg, image)   # HxW int64 class ids (seg.class_names)

``depth`` is positive relative depth: larger = farther, affine-invariant (compare it
with metric depth through a fit s * depth + t with s > 0 per image). The released depth models
("l-dpt_large_depth_zeroshot", "l-dpt_large_depth_mixed") have an exponential head:
values in [exp(-30), exp(30)]; a sigmoid-head model gives (0, 1). A segmentation model
("l-dpt_large_semseg_cocostuff": COCO-Stuff, 171 classes) gives one class id per pixel,
the argmax of its logits resized to the image, as its evaluation predicts.
``api.load("l-dpt_large_depth_zeroshot", ...)`` downloads the pinned weights file from
Hugging Face on first use; a local copy loads by path
(``api.load("l-dpt_large_depth_zeroshot.pth", ...)``), with ``weights=`` or with
``$LDPT_CKPT_LARGE_DEPTH_ZEROSHOT``.

``load`` builds the model on the meta device and assigns the memory-mapped checkpoint
tensors on the target device (the 7B encoder stored in bf16 by default), so no fp32
copy of the encoder is made. It builds the model as its release file or preset states
it (decoder precision too, unless ``precision`` is passed) and refuses weights of
another architecture (_ARCH_KEYS: the options that change the output). The decoder RoPE
of a depth model normalises token coordinates by the short side
(``dec_pos_embed_rope_normalize_coords="min"``), which keeps the token spacing of the
square training crops on any aspect ratio (``"separate"`` scales each axis to [-1, 1]
and squeezes the long side); a segmentation model keeps the value of its file or
preset (``"separate"``, as evaluated). Needs torch and numpy only (huggingface_hub to
download weights).

``predict`` uses the validation preprocessing (short side to the preset's INPUT_SIZE,
bicubic, ImageNet normalisation) and resizes the output back to the input size;
``size=None`` runs at the native resolution with NATIVE_MEMORY for the call (see
``predict``). A depth model's input with a side above 3072 px (192 tokens) runs the
frozen encoder on overlapping windows of 3072 x 3072 px and the decoder on the whole
latent grid in one pass (``encoder_tiles="auto"``, ENCODER_TILES; encoder_tiles.py); a
smaller one, and a segmentation model's at any size, is one forward pass. It picks the
attention order per GPU (``sdpa_backends="auto"``,
ldpt/utils/sdpa.py): cuDNN attention first on compute capability 10.x (B200, GB200,
GB300), where it is the fastest, else PyTorch's order (``sdpa_backends=None``, which
evaluation uses). Every option, and the time and memory of native-resolution
inference: docs/inference.md.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import inspect
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ldpt.utils.sdpa import SDPA_ORDER, auto_order, sdpa_context

from .convert import FORMATS, build_meta, check_decoder_state, load_release
from .encoder_tiles import EncoderTiles
from .l_dpt_v1m1_base import PRECISIONS, LatentDPT, _resolve_attn_dtype
from .modules import check_choice
from .presets import (
    CLASS_NAMES,
    IMAGE_MEAN,
    IMAGE_STD,
    INPUT_SIZE,
    PRESETS,
    get_preset,
    suggest_preset,
)
from .weights import (
    PathLike,
    decoder_state,
    encoder_state,
    encoder_weights as find_encoder_weights,
    fetch,
    read_checkpoint,
)

__all__ = [
    "CLASS_MAP_ELEMENTS",
    "ENCODER_TILES",
    "EncoderTiles",
    "NATIVE_MEMORY",
    "SDPA_ORDER",
    "auto_order",
    "class_map",
    "load",
    "predict",
]

log = logging.getLogger(__name__)

# the inference memory options of LatentDPT that predict(size=None) runs with unless the
# caller chose them (see predict): those of the demo and scripts/highres_infer.py,
# measured at 100 MP; same math as the defaults, in bf16 equal up to rounding
NATIVE_MEMORY: Dict[str, Any] = dict(selection_impl="stream", chunk_tokens=32768)

# the encoder tiling of predict(encoder_tiles="auto") for a depth model: an input with a
# side above 192 tokens (3072 px) runs the encoder on 192 x 192-token tiles that overlap
# by 64 tokens or more, with its RoPE coordinates times 2 inside a tile; a smaller input
# runs the plain forward, bit for bit (encoder_tiles.py). A segmentation model runs one
# encoder pass with "auto": in the tiles the classes of large regions fragment
ENCODER_TILES = EncoderTiles(tile=192, overlap=64, rope_scale=2.0)


def load(
    name_or_path: Union[str, PathLike],
    encoder_weights: Optional[PathLike] = None,
    device: Union[str, torch.device] = "auto",
    precision: Optional[str] = None,
    *,
    weights: Optional[PathLike] = None,
    run_config: Union[PathLike, None, bool] = None,
    revision: Optional[str] = None,
    verify: bool = True,
    share_encoder: Optional[nn.Module] = None,
    **overrides: Any,
) -> LatentDPT:
    """A LatentDPT with its weights, on ``device``, in eval mode.

    Args:
        name_or_path: a preset with released weights
            (``"l-dpt_large_depth_zeroshot"``, ``"l-dpt_large_depth_mixed"``,
            ``"l-dpt_large_semseg_cocostuff"``; their files are downloaded from
            Hugging Face on first use, or read from ``$LDPT_CKPT_LARGE_DEPTH_ZEROSHOT``
            / ``$LDPT_CKPT_LARGE_DEPTH_MIXED`` / ``$LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF``;
            a preset whose slot is not published yet needs ``weights=``), or a
            release file written by convert.py (it carries its config; a short preset
            name of LEGACY_PRESETS stored in it is read as the current name).
        encoder_weights: the official DINOv3 file (default ``$LDPT_DINOV3``); its sha256
            prefix is checked once (``verify``). Or a folder in the transformers format
            (a download of Meta's gated Hugging Face repo, e.g.
            ``facebook/dinov3-vit7b16-pretrain-lvd1689m``; its config.json is checked,
            the weights are read into the original layout: the same tensors). Not
            needed for a parameter-free encoder.
        device: ``"auto"`` (cuda, else mps, else cpu) or a device.
        precision: decoder precision. None (default): the release file's / preset's
            value, as trained and evaluated (``"bf16"`` for the release models);
            ``"fp32"`` or ``"bf16"`` to choose.
        weights: a local decoder checkpoint (training or release) instead of the preset's
            file. A release file must have the preset's architecture (_ARCH_KEYS, after
            the overrides), else it raises; a per-load option passed in ``overrides``
            (_LOAD_KEYS: attn_dtype, attn_fp32, block_fp32, readout_fp32,
            dec_pos_embed_rope_eval_scale) is the caller's choice and not compared. A
            training checkpoint records no model kwargs: it is checked against
            ``run_config``; without one, the preset (+ overrides) must be exactly the
            run's model (only the decoder keys and shapes are checked strictly: a head,
            bound or fp32 island of the run that the preset lacks would go unnoticed).
        run_config: with ``weights=<training checkpoint>``: the run's config file
            (``<run>/config.py``), whose ``model`` must agree with the preset (+
            overrides) as a release file must, else it raises. None (default):
            ``<run>/config.py`` when the checkpoint sits in ``<run>/model/`` (the
            trainer's layout), else no check; False: no check. Reading it needs the
            training config parser (addict, yapf): without it the default skips the
            check with a warning, a path raises ImportError.
        revision: another Hugging Face revision of the preset's file (not sha256-checked).
        verify: check the sha256 of the files against the pinned / official ones.
        share_encoder: the encoder of a model loaded before (``other.encoder``),
            used instead of reading ``encoder_weights``: the same module, no copy. The
            release models share the frozen DINOv3 ViT-7B, so a second one costs only
            its decoder. It must be this model's encoder (type, names and shapes,
            storage dtype, encoder config) on ``device``, else ValueError; which
            weights it holds is not checked.
        overrides: model kwargs replacing the preset's / file's; by default a frozen
            encoder is stored in bf16 (``encoder_weight_dtype="bf16"``, bit-identical
            outputs) and a depth model runs ``dec_pos_embed_rope_normalize_coords=
            "min"`` (see the module docstring; ``"separate"`` normalises each axis to
            [-1, 1]; a segmentation model keeps its file's / preset's value). The
            model is built inference-only (``criterion=False``: no training loss, so
            no loss code runs; a training forward raises) unless ``criterion`` is
            passed. ``selection_impl`` / ``chunk_tokens`` here also set what
            ``predict(size=None)`` runs with (see ``.native_memory``).

    Returns:
        LatentDPT in eval mode, with ``.preset`` (name or None), ``.class_names`` (a
        segmentation preset's classes, presets.CLASS_NAMES, in output-channel order;
        else None), ``.weights_info`` and ``.native_memory`` (the NATIVE_MEMORY options
        not in ``overrides``: what ``predict(size=None)`` sets for the call).
    """
    t0 = time.perf_counter()
    name = os.fspath(name_or_path)
    info: Dict[str, Any] = {}
    if precision is not None:
        check_choice("precision", precision, PRECISIONS)
    if name in PRESETS:
        preset = name
        config = get_preset(name)
        config.update(overrides)  # the architecture is compared after the overrides
        path = Path(weights).expanduser() if weights is not None else None
        if path is None:
            if run_config not in (None, False):
                raise ValueError("run_config= goes with weights=<training checkpoint>")
            path = fetch(name, revision=revision, verify=verify)
        ckpt = read_checkpoint(path)
        state = decoder_state(ckpt)
        if ckpt.get("format") in FORMATS and ckpt.get("config") is not None:
            if run_config not in (None, False):
                raise ValueError(
                    f"{path} is a release file: it carries its config, run_config= "
                    f"goes with a training checkpoint"
                )
            _check_same_architecture(
                ckpt["config"],
                config,
                f"{path} was converted",
                name,
                path,
                _chosen(overrides),
            )
        elif weights is not None:
            run_file = _run_config_file(path, run_config)
            run_model = None
            if run_file is not None:
                try:
                    run_model = _model_of(run_file)
                except ImportError as err:
                    if run_config is not None:  # asked for: the check must run
                        raise
                    log.warning(
                        "%s: not checked against %s (reading it needs the training "
                        "config parser: %s); the preset (+ overrides) must be the run's "
                        "model, pass run_config=False to silence this",
                        path,
                        run_file,
                        err,
                    )
            if run_model is not None:
                _check_same_architecture(
                    run_model,
                    config,
                    f"{path} was trained (per {run_file})",
                    name,
                    path,
                    _chosen(overrides),
                )
                info["run_config"] = str(run_file)
        info["decoder"] = str(path)
    else:
        path = Path(name).expanduser()
        if not path.is_file():
            raise ValueError(
                f"{name!r} is neither a preset{suggest_preset(name)} "
                f"({sorted(PRESETS)}) nor a file"
            )
        if weights is not None or run_config not in (None, False):
            raise ValueError(
                "weights= / run_config= go with a preset name, not with a file path"
            )
        config, state, release = load_release(path)
        preset = release["preset"]
        info.update(decoder=str(path), provenance=release["provenance"])
        config.update(overrides)
    if precision is not None:
        config["precision"] = precision
    if config.get("freeze_encoder", True) and "encoder_weight_dtype" not in overrides:
        config["encoder_weight_dtype"] = "bf16"
    if (
        "dec_pos_embed_rope_normalize_coords" not in overrides
        and config.get("task", "relative_depth") == "relative_depth"
    ):
        config["dec_pos_embed_rope_normalize_coords"] = "min"
    if "criterion" not in overrides:  # inference only: no training loss
        config["criterion"] = False
    dev = _device(device)
    model = build_meta(config)
    check_decoder_state(config, state, model)
    state = {k: v.to(dev) for k, v in state.items()}
    if share_encoder is not None:
        if encoder_weights is not None:
            raise ValueError("pass share_encoder= or encoder_weights=, not both")
        on = (
            next(iter(state.values())).device
            if state
            else torch.empty(0, device=dev).device
        )
        source = _check_shared_encoder(model.encoder, share_encoder, config, on)
        info["encoder"] = f"{source} (shared)" if source else "shared"
    elif any(k.startswith("encoder.") for k in model.state_dict()):
        enc_type = _encoder_type(config["encoder"])
        enc_path = find_encoder_weights(enc_type, encoder_weights, verify=verify)
        state.update(
            encoder_state(
                enc_path,
                dtypes=_storage_dtypes(model),
                device=dev,
                encoder_type=enc_type,
            )
        )
        info["encoder"] = str(enc_path)
    # a shared encoder goes in after the load: no load hook of this model (the bf16
    # cast) runs on a module another model owns
    loaded = model.load_state_dict(state, strict=share_encoder is None, assign=True)
    left = [k for k in loaded.missing_keys if not k.startswith("encoder.")]
    if left or loaded.unexpected_keys:  # (strict=False only skips the shared encoder)
        raise RuntimeError(
            f"keys missing {left[:8]} / unexpected "
            f"{list(loaded.unexpected_keys)[:8]}"
        )
    del state
    if share_encoder is not None:
        model.encoder = share_encoder
    elif "encoder" in info:  # what a later share_encoder= compares with
        model.encoder._ldpt_source = dict(
            config=_encoder_config(config["encoder"]), weights=info["encoder"]
        )
    meta = [n for n, t in _tensors(model) if t.is_meta]
    if meta:
        raise RuntimeError(
            f"tensors left on the meta device after loading: {meta[:8]}; the checkpoint "
            f"lacks them and the model derives no value for them"
        )
    model.eval()
    model.preset = preset
    names = CLASS_NAMES.get(preset or "")
    model.class_names = list(names) if names is not None else None
    model.weights_info = info
    model.native_memory = {k: v for k, v in NATIVE_MEMORY.items() if k not in overrides}
    log.info("load(%s): %.1f s on %s", name, time.perf_counter() - t0, dev)
    return model


@torch.inference_mode()
def predict(
    model: LatentDPT,
    image: Union[np.ndarray, Tensor],
    *,
    size: Union[int, str, None] = "auto",
    flip: bool = False,
    sdpa_backends: Union[Sequence[str], str, None] = "auto",
    encoder_tiles: Union[str, EncoderTiles, None, bool] = "auto",
) -> Union[np.ndarray, Tensor]:
    """Relative depth or the class map of one image, at the image's resolution.

    Relative depth (``model.task == "relative_depth"``): positive, larger = farther,
    affine-invariant (compared with metric depth through a fit s * depth + t, s > 0, per
    image); in [exp(-30), exp(30)] with the exp head (the released depth models), in
    (0, 1) with a sigmoid head.

    Semantic segmentation (``"semseg"``): the class id of every pixel (0 ..
    num_classes - 1, names in ``model.class_names``), as SemSegEvaluator predicts it:
    the logits (with ``flip``: the mean of the two softmaxes) resized bilinearly to the
    image, then the argmax (the lowest id on a tie). The resize runs over a few classes
    at a time when the image is large (CLASS_MAP_ELEMENTS values at most), with the
    same result. At ``size=None`` the logits are computed at the native resolution:
    num_classes x H x W x 4 bytes.

    Args:
        image: HxWx3 RGB, uint8 in [0, 255] or float in [0, 1] (numpy or torch).
        size: short side the image is resized to: ``"auto"`` (the preset's INPUT_SIZE,
            768 for the released models and for a file without a preset), an
            int, or None for the native resolution. None runs with the
            ``model.native_memory`` options (from ``load``: NATIVE_MEMORY unless the
            caller passed them) that are still at their LatentDPT default, and restores
            them after the call; a model built otherwise runs as it is.
        flip: also run the horizontally flipped image and average (the eval TTA):
            the depths, or the softmaxes of a segmentation model.
        sdpa_backends: the scaled-dot-product attention backends tried, in this order,
            while the model runs (names of ldpt.utils.sdpa.SDPA_BACKENDS: "cudnn",
            "flash", "efficient", "math"; the others are disabled for the call).
            Default ``"auto"`` (ldpt.utils.sdpa.auto_order on the model's device):
            SDPA_ORDER, cuDNN attention first, on a GPU of compute capability 10.x,
            where it ran native resolution from 32 MP up in less than half the time
            on a GB200 with one encoder pass; PyTorch's order elsewhere (cuDNN first
            is slower on an RTX PRO 6000 Blackwell, 12.0). A backend the device or the
            dtype cannot use is skipped (the fp32 attention of the release models' last
            decoder stage runs the memory-efficient kernel; on the CPU and MPS the order
            changes nothing). None (or ``[]``) keeps PyTorch's order (FlashAttention
            first in torch 2.8), as scripts/evaluate.py runs; the kernels differ at the
            bf16 rounding level. The flags are process-wide while the call runs and are
            restored after it.
        encoder_tiles: ``"auto"`` (default): for a depth model ENCODER_TILES, so an
            input (after the resize of ``size``) with a side above 192 tokens (3072
            px) runs the frozen encoder on overlapping 192 x 192-token tiles, each with
            its RoPE coordinates times 2, blends their selections onto the whole token
            grid and decodes that grid in one pass; a smaller input runs the plain
            forward, bit for bit (so the 768 px benchmarks are unchanged). For a
            segmentation model ``"auto"`` is None: in the tiles the classes of large
            regions fragment. An EncoderTiles (such as ENCODER_TILES) sets a tiling
            for either task; None or False runs the encoder once on the whole image,
            at any size.

    Shapes:
        image: [H, W, 3] -> depth [H, W] float32, or class ids [H, W] int64 (numpy
        for numpy input, else a tensor on the model's device)
    """
    segment = model.task == "semseg"
    device = _model_device(model)
    attention = sdpa_context(sdpa_backends, device)  # an unknown name raises here
    tiles = _encoder_tiles(encoder_tiles, segment)
    x = _to_image(image, device)  # [1, 3, H, W] in [0, 1]
    H, W = x.shape[-2:]
    if size == "auto":
        size = INPUT_SIZE.get(getattr(model, "preset", None) or "", 768)
    if size is not None:
        x = resize_short_side(x, int(size))
    mean = torch.tensor(IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    memory = _native_memory(model) if size is None else {}
    saved = {k: getattr(model, k) for k in memory}
    was_training = model.training
    model.eval()
    try:
        for k, v in memory.items():
            setattr(model, k, v)
        with attention:
            pred = model.infer(x, encoder_tiles=tiles)
            if flip:
                mirrored = model.infer(torch.flip(x, dims=[-1]), encoder_tiles=tiles)
                flipped = torch.flip(mirrored, dims=[-1])
                del mirrored
                if segment:  # SemSegEvaluator's TTA: the mean of the softmaxes
                    pred, flipped = pred.softmax(dim=1), flipped.softmax(dim=1)
                pred = torch.stack([pred, flipped], dim=1).mean(dim=1)
    finally:
        for k, v in saved.items():
            setattr(model, k, v)
        model.train(was_training)
    if segment:
        out = class_map(pred, (H, W))[0]
    else:
        if pred.shape[-2:] != (H, W):
            pred = F.interpolate(
                pred, size=(H, W), mode="bilinear", align_corners=False
            )
        out = pred[0, 0].float()
    return out.cpu().numpy() if isinstance(image, np.ndarray) else out


# the most resized class scores class_map holds at once (1 GiB in fp32)
CLASS_MAP_ELEMENTS = 1 << 28


def class_map(
    scores: Tensor, size: Tuple[int, int], max_elements: int = CLASS_MAP_ELEMENTS
) -> Tensor:
    """The argmax over classes of ``scores`` resized bilinearly to ``size``:
    SemSegEvaluator's ``F.interpolate(..., mode="bilinear", align_corners=False)`` and
    ``argmax(dim=1)`` (no resize at that size already), run on as many classes at a
    time as keep the resized block within ``max_elements`` values, with the same result
    (each class resizes on its own; on a tie the lowest class id, as argmax).

    Shapes:
        scores: [B, C, h, w] (logits or probabilities) -> [B, H, W] int64
    """
    if tuple(scores.shape[-2:]) == tuple(size):
        return scores.argmax(dim=1)
    B, C = scores.shape[:2]
    step = max(1, max_elements // max(1, B * size[0] * size[1]))
    if step >= C:
        resized = F.interpolate(scores, size=size, mode="bilinear", align_corners=False)
        return resized.argmax(dim=1)
    best = best_id = None
    for c0 in range(0, C, step):
        resized = F.interpolate(
            scores[:, c0 : c0 + step], size=size, mode="bilinear", align_corners=False
        )
        value, idx = resized.max(dim=1)
        idx += c0
        if best is None:
            best, best_id = value, idx
        else:  # strictly greater: a tie keeps the lower id
            better = value > best
            best = torch.where(better, value, best)
            best_id = torch.where(better, idx, best_id)
    return best_id


def resize_short_side(image: Tensor, size: int) -> Tensor:
    """torchvision ``Resize(size, bicubic)`` on a float tensor: short side to ``size``,
    long side ``int(size * long / short)``, antialiased bicubic, no clamping.

    Shapes:
        image: [B, C, H, W] -> [B, C, H', W']
    """
    h, w = image.shape[-2:]
    short, long = (w, h) if w <= h else (h, w)
    new_short, new_long = size, int(size * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    if (new_h, new_w) == (h, w):
        return image
    return F.interpolate(
        image, size=[new_h, new_w], mode="bicubic", align_corners=False, antialias=True
    )


# ---- helpers


def _encoder_tiles(
    value: Union[str, EncoderTiles, None, bool], segment: bool = False
) -> Optional[EncoderTiles]:
    """predict's encoder_tiles: "auto" -> ENCODER_TILES for a depth model, None (one
    encoder pass) for a segmentation model (``segment``); None / False -> None, an
    EncoderTiles as is; ValueError otherwise."""
    if isinstance(value, str) and value == "auto":
        return None if segment else ENCODER_TILES
    if value is None or value is False:
        return None
    if isinstance(value, EncoderTiles):
        return value
    raise ValueError(
        f'encoder_tiles must be "auto", an EncoderTiles, None or False, got {value!r}'
    )


def _native_memory(model: nn.Module) -> Dict[str, Any]:
    """The options of ``model.native_memory`` (set by load) whose attribute is still
    at its LatentDPT default: one set on the model since is the caller's choice."""
    options = getattr(model, "native_memory", None) or {}
    return {k: v for k, v in options.items() if getattr(model, k) == _DEFAULTS[k]}


def _to_image(image: Union[np.ndarray, Tensor], device: torch.device) -> Tensor:
    if isinstance(image, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(image))
    elif isinstance(image, Tensor):
        x = image
    else:
        raise TypeError(
            f"image must be a numpy array or a tensor, got {type(image).__name__}"
        )
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3 RGB, got shape {list(x.shape)}")
    if x.dtype == torch.uint8:
        x = x.to(device=device, dtype=torch.float32) / 255.0
    elif x.is_floating_point():
        x = x.to(device=device, dtype=torch.float32)
        lo, hi = x.min().item(), x.max().item()
        if lo < -1e-3 or hi > 1 + 1e-3:
            raise ValueError(
                f"a float image must be in [0, 1], got [{lo:.3g}, {hi:.3g}]: pass uint8 "
                f"for 0-255 values or divide by 255"
            )
    else:
        raise TypeError(f"image dtype must be uint8 or float, got {x.dtype}")
    return x.permute(2, 0, 1).unsqueeze(0).contiguous()


def _device(device: Union[str, torch.device]) -> torch.device:
    if str(device) != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_device(model: nn.Module) -> torch.device:
    for t in model.parameters():
        return t.device
    return torch.device("cpu")


def _tensors(model: nn.Module):
    yield from model.named_parameters()
    yield from model.named_buffers()


def _encoder_config(encoder: Any) -> Any:
    return dict(encoder) if isinstance(encoder, Mapping) else type(encoder).__name__


def _check_shared_encoder(
    built: nn.Module, shared: nn.Module, config: Mapping[str, Any], device
) -> Optional[str]:
    """``shared`` (load's share_encoder=) can stand for ``built`` (this config's
    encoder, on the meta device), else ValueError: same type, same parameter and buffer
    names and shapes, the Linear / Conv2d weights in the dtype this model stores them
    in, every tensor on ``device`` (where the decoder went), no meta tensor, and, when
    load() read its weights, the same encoder config. Returns that weight file or None.
    The weights themselves are not compared (another checkpoint of the same
    architecture would pass)."""
    if type(shared) is not type(built):
        raise ValueError(
            f"share_encoder=: a {type(shared).__name__}, this model's encoder is a "
            f"{type(built).__name__}"
        )
    want = {k: tuple(t.shape) for k, t in _tensors(built)}
    got = {k: tuple(t.shape) for k, t in _tensors(shared)}
    if want != got:
        diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
        raise ValueError(
            f"share_encoder=: not this model's encoder ({len(diff)} names or shapes "
            f"differ, e.g. {diff[:4]})"
        )
    meta = [k for k, t in _tensors(shared) if t.is_meta]
    if meta:
        raise ValueError(f"share_encoder=: tensors on the meta device: {meta[:4]}")
    stored = dict(_tensors(shared))
    linear_conv = {
        f"{name}.{p_name}" if name else p_name
        for name, m in built.named_modules()
        if isinstance(m, (nn.Linear, nn.Conv2d))
        for p_name, _ in m.named_parameters(recurse=False)
    }
    cast = {
        k: (stored[k].dtype, t.dtype)
        for k, t in _tensors(built)
        if k in linear_conv and stored[k].dtype != t.dtype
    }
    if cast:
        k, (got_dtype, want_dtype) = next(iter(cast.items()))
        raise ValueError(
            f"share_encoder=: {len(cast)} Linear / Conv2d tensors stored in another "
            f"dtype than this model's encoder_weight_dtype gives (e.g. {k}: "
            f"{got_dtype} vs {want_dtype})"
        )
    away = sorted({str(t.device) for _, t in _tensors(shared) if t.device != device})
    if away:
        raise ValueError(
            f"share_encoder=: the encoder is on {away}, this model's decoder on "
            f"{device}"
        )
    source = getattr(shared, "_ldpt_source", None)
    if source is None:
        return None
    if source["config"] != _encoder_config(config["encoder"]):
        raise ValueError(
            f"share_encoder=: that encoder was built from {source['config']}, this "
            f"model's config has {_encoder_config(config['encoder'])}"
        )
    return source["weights"]


def _encoder_type(encoder: Any) -> str:
    if isinstance(encoder, Mapping):
        return str(encoder.get("type"))
    return type(encoder).__name__


def _storage_dtypes(model: LatentDPT) -> Dict[str, torch.dtype]:
    """Encoder keys (without ``encoder.``) stored in ``encoder_weight_dtype``."""
    dtype = model.encoder_weight_dtype
    if dtype is None:
        return {}
    keys = {}
    for name, m in model.encoder.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            for p_name, p in m.named_parameters(recurse=False):
                keys[f"{name}.{p_name}" if name else p_name] = dtype
    return keys


# model kwargs that change the output (or the state-dict keys): a release file or the
# run's config.py and the preset (+ overrides) must agree on them (a missing key = the
# LatentDPT default). Not compared: the training-only RoPE augmentation (shift / jitter
# / rescale), init values, the criterion, and what the caller picks per load (precision,
# dec_pos_embed_rope_normalize_coords: api.load uses "min", selection_impl,
# chunk_tokens, encoder_weight_dtype: bit-identical under the frozen encoder's bf16
# autocast, and api.load stores a frozen encoder in bf16 unless overridden).
_ARCH_KEYS = (
    "encoder",
    "encoder_inter_layers",
    "freeze_encoder",
    "task",
    "out_dim",
    "num_classes",
    "selection",
    "readout_kernel_size",
    "tap_bn",
    "head",
    "readout_fp32",
    "attn_dtype",
    "attn_fp32",
    "block_fp32",
) + tuple(
    f"dec_{k}"
    for k in (
        "dim",
        "num_heads",
        "num_layers",
        "attn_downscale",
        "ffn_ratio",
        "pos_embed_rope_base",
        "pos_embed_rope_eval_scale",
        "qk_bound",
        "qk_norm",
        "qk_tau_max",
        "conv_mixer",
    )
)
# _ARCH_KEYS that need no retraining (compute precision of the attention / readout /
# blocks, the eval-only RoPE scale): compared when the preset states them, but a value
# the caller passes in overrides is the caller's choice, as precision is (and as with a
# load by path); the recipe keys must match the file even when passed
_LOAD_KEYS = (
    "attn_dtype",
    "attn_fp32",
    "block_fp32",
    "readout_fp32",
    "dec_pos_embed_rope_eval_scale",
)
_DEFAULTS = {
    k: p.default
    for k, p in inspect.signature(LatentDPT.__init__).parameters.items()
    if p.default is not inspect.Parameter.empty
}


def _arch(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The _ARCH_KEYS values ``config`` builds (defaults filled, legacy names and
    "None" strings resolved, tuples as lists)."""
    values = {k: _norm(config.get(k, _DEFAULTS.get(k))) for k in _ARCH_KEYS}
    values["attn_dtype"] = _resolve_attn_dtype(
        config.get("attn_dtype"), config.get("attn_impl")
    )
    if values["out_dim"] is None:
        values["out_dim"] = 1 if values["task"] == "relative_depth" else 32
    bound = values["dec_qk_bound"]
    if isinstance(bound, str) and bound.strip().lower() in ("", "none", "null"):
        values["dec_qk_bound"] = None
    return values


def _check_same_architecture(
    theirs: Mapping[str, Any],
    mine: Mapping[str, Any],
    what: str,
    name: str,
    path: Path,
    chosen: Iterable[str] = (),
) -> None:
    """ValueError unless ``theirs`` (a release file's / a run's model kwargs) and
    ``mine`` (preset + overrides) build the same model (_ARCH_KEYS), except on the
    _LOAD_KEYS in ``chosen`` (the per-load options the caller passed)."""
    a, b = _arch(theirs), _arch(mine)
    skip = set(chosen) & set(_LOAD_KEYS)
    diff = [k for k in _ARCH_KEYS if k not in skip and a[k] != b[k]]
    if diff:
        raise ValueError(
            f"{what} with another architecture than preset {name!r} (+ overrides): "
            + ", ".join(f"{k}: file {a[k]!r} vs preset {b[k]!r}" for k in diff)
            + f"; load a release file by path instead (api.load({str(path)!r})), or "
            f"pass the file's values as overrides"
        )


def _chosen(overrides: Mapping[str, Any]) -> set:
    """The _LOAD_KEYS the caller passed (the legacy attn_impl sets attn_dtype)."""
    chosen = {k for k in _LOAD_KEYS if k in overrides}
    if "attn_impl" in overrides:
        chosen.add("attn_dtype")
    return chosen


def _run_config_file(
    ckpt: Path, run_config: Union[PathLike, None, bool]
) -> Optional[Path]:
    """The config file a training checkpoint is checked against (see load)."""
    if run_config is False:
        return None
    if run_config is None:  # the trainer's layout: <run>/model/<checkpoint>
        beside = ckpt.parent.parent / "config.py"
        return beside if ckpt.parent.name == "model" and beside.is_file() else None
    if isinstance(run_config, bool):
        raise ValueError("run_config must be a path, None or False")
    path = Path(run_config).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"run_config {path} is not a file")
    return path


def _model_of(config_file: Path) -> Dict[str, Any]:
    """The ``model`` dict of a training config file (the training config parser)."""
    from ldpt.utils.config import Config  # addict / yapf: not needed otherwise

    return dict(Config.fromfile(str(config_file)).model)


def _norm(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    return value
