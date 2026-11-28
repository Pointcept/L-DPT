"""
Everything of the L-DPT Space except the Gradio UI (app.py), so it is testable without
gradio: the model registry (models.json + environment overrides), the DINOv3 weights,
resolution choices and the megapixel limit, the timing guide and the ZeroGPU slot of a
request (its duration and quota), colorizing, the 16-bit depth PNG, and
DepthService (the loaded models, sharing one DINOv3 encoder, with the high-res memory
options of api.load).

models.json:
    {"models": [{"id": "l-dpt_large_depth_zeroshot",  # [a-z0-9][a-z0-9_-]*
                 "label": "...", "description": "...",
                 "recommended": true,            # optional; the default is the first
                                                 # recommended entry, else the first
                 "source": {"repo": "user/name", "filename": "x.pth",
                            "revision": "main",  # or {"path": "weights/x.pth"},
                            "sha256": "..."},    # relative to models.json; sha256
                                                 # optional, checked after a download
                 "expect": {"head": "exp"}       # optional: release config values
                }, ...]}                         # checked at load (EXPECT_DEFAULTS)
    A source whose repo, filename or revision contains "TODO" is not published yet.
    $LDPT_CKPT_<NAME> names a local release file that replaces the source (not
    hashed): NAME is the id without "l-dpt_", upper-cased, other characters -> "_",
    the variable api.load reads for a preset (LDPT_CKPT_LARGE_DEPTH_ZEROSHOT);
    $LDPT_MODEL_<ID> (LDPT_MODEL_L_DPT_LARGE_DEPTH_ZEROSHOT) is a second name, read
    after it.

Release files come from `python -m ldpt.models.l_dpt.convert SRC DST --preset
l-dpt_large_depth_zeroshot` (or `--config <the run's config.py>`); they carry the model
kwargs. The DINOv3 ViT-7B weights are not redistributable; resolve_dinov3 takes, in
this order: $LDPT_DINOV3 (a local official .pth file or a local transformers-format
folder), $LDPT_DINOV3_REPO + $LDPT_DINOV3_FILE (a Hugging Face repo holding the .pth),
else Meta's gated transformers-format repo (DINOV3_HF_REPO at DINOV3_HF_REVISION;
$LDPT_DINOV3_HF_REPO / _REVISION override, "none" turns it off), downloaded with
$HF_TOKEN (an account granted access to it). api.load reads a transformers folder into
the original key layout (the same tensors).
"""

import dataclasses
import gc
import logging
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

# colour and the 16-bit depth PNG are the package's, so the demo and the scripts
# (scripts/demo_depth.py, scripts/highres_infer.py) write the same images
from ldpt.utils.depth_image import (  # noqa: F401  (the Space's names for them)
    COLORMAPS,
    DEPTH16_FORMAT,
    DISPLAY_CMAP,
    DISPLAY_PERCENTILES,
    PNG_TEXT_KEY,
    ToneCurve,
    add_png_text,
    colorize,
    colormap_lut,
    decode_depth16,
    depth16_png,
    depth_range,
    encode_depth16,
    png_text_chunk,
    png_texts,
    read_depth16,
    tone_curve,
    write_depth16,
)

HERE = Path(__file__).resolve().parent


log = logging.getLogger(__name__)

# ---- errors and megapixels


class InputError(ValueError):
    """A request the demo refuses; the message is shown to the user as is."""


class SetupError(RuntimeError):
    """The Space lacks a weight file or a setting; the message says which."""


def inference_hw(height: int, width: int, size: Optional[int]) -> Tuple[int, int]:
    """(H, W) the network runs at: api.resize_short_side's arithmetic."""
    if size is None:
        return int(height), int(width)
    h, w = int(height), int(width)
    short, long = (w, h) if w <= h else (h, w)
    new_short, new_long = int(size), int(size * long / short)
    return (new_long, new_short) if w <= h else (new_short, new_long)


def megapixels(height: int, width: int) -> float:
    return height * width / 1e6


def check_megapixels(height: int, width: int, max_mp: float) -> None:
    mp = megapixels(height, width)
    if mp > max_mp:
        raise InputError(
            f"The image is {width}x{height} ({mp:.1f} MP); this demo takes up to "
            f"{max_mp:g} MP. Downscale it and try again."
        )


def check_inference(
    height: int,
    width: int,
    size: Optional[int],
    max_infer_mp: float,
    *,
    flip: bool = False,
    max_flip_mp: Optional[float] = None,
    max_gpu_seconds: Optional[float] = None,
) -> None:
    """Refuse an inference above ``max_infer_mp``, a flip above ``max_flip_mp`` and a
    request whose estimated GPU time reaches ``max_gpu_seconds`` (None: no limit)."""
    h, w = inference_hw(height, width, size)
    mp = megapixels(h, w)
    if mp > max_infer_mp:
        raise InputError(
            f"Inference at {w}x{h} ({mp:.1f} MP) is above what this Space runs "
            f"({max_infer_mp:g} MP, GPU memory). Downscale the photo and try again."
        )
    if flip and max_flip_mp is not None and mp > max_flip_mp:
        raise InputError(
            f"The mirrored pass runs up to {max_flip_mp:g} MP here (inference at "
            f"{w}x{h}, {mp:.1f} MP). Turn it off or downscale the photo."
        )
    if max_gpu_seconds is not None:  # the duration on the slot it runs on
        slot = zerogpu_size(height, width, size)
        seconds = estimate_gpu_seconds(height, width, size, flip, slot)
        if seconds >= max_gpu_seconds:
            raise InputError(
                f"Inference at {w}x{h} ({mp:.1f} MP){' + flip' if flip else ''} needs "
                f"an estimated {seconds:.0f} GPU seconds, above this Space's limit of "
                f"{max_gpu_seconds:g} s. Downscale the photo and try again."
            )


# ---- the timing guide
#
# The measured time and peak GPU memory (torch.cuda.max_memory_allocated, GiB) of one
# photo with the release models: bf16, selection_impl="stream", chunk_tokens=32768,
# native resolution, no flip, api.predict's encoder tiling, the model loaded and warm.
# A photo with a side above ENCODER_TILE_PX runs the encoder on overlapping windows
# (api.ENCODER_TILES), whose count follows the photo's sides: its time follows a fit of
# its own (GpuTiming.tiled_fit), and a photo with both sides just above ENCODER_TILE_PX
# (where the windows overlap most) takes per_window seconds per window, more than the
# fit gives. Every time and memory figure the app shows comes
# from here (timing_rows, quoted_seconds), and so does the ZeroGPU
# reservation (GPU_SECONDS_*). README.md and docs/inference.md state the RTX PRO 6000's
# numbers, this Space's README.md the ZeroGPU ones; tests/test_hf_space.py
# (test_timing_guide_*) checks them and that each fit still matches its measurements,
# so a new profiling run is entered here first. Each GPU runs api.predict's attention
# order for it, PyTorch's on the RTX PRO 6000 and the half ZeroGPU slot (the full
# ZeroGPU GPU: measured with cuDNN attention first). A precomputed example names its
# GPU, not the seconds of its run.


# api.ENCODER_TILES: windows of 192 x 192 tokens of 16 px, overlapping by 64 tokens or
# more, when a side exceeds 192 tokens
ENCODER_TILE_TOKENS, ENCODER_TILE_OVERLAP, PATCH_PX = 192, 64, 16
ENCODER_TILE_PX = ENCODER_TILE_TOKENS * PATCH_PX  # 3072
# a timing row (megapixels only) counts as tiled from the size of a 4:3 photo whose long
# side is ENCODER_TILE_PX (7.08 MP)
TILED_MP = ENCODER_TILE_PX * ENCODER_TILE_PX * 3 / 4 / 1e6


def encoder_tiled(height: int, width: int) -> bool:
    """True when api.predict runs the encoder of an inference at height x width on
    overlapping windows (a side above ENCODER_TILE_PX)."""
    return max(int(height), int(width)) > ENCODER_TILE_PX


def encoder_windows(height: int, width: int) -> float:
    """The encoder windows of an inference at height x width (api.ENCODER_TILES), each
    counted as its share of a full 192 x 192-token window; 0 when it is not tiled. Per
    axis of n tokens: ceil((n - 64) / 128) full windows above 192 tokens, else one of
    n / 192."""
    if not encoder_tiled(height, width):
        return 0.0
    count = 1.0
    for px in (height, width):
        n = math.ceil(int(px) / PATCH_PX)
        if n <= ENCODER_TILE_TOKENS:
            count *= n / ENCODER_TILE_TOKENS
        else:
            count *= math.ceil(
                (n - ENCODER_TILE_OVERLAP)
                / (ENCODER_TILE_TOKENS - ENCODER_TILE_OVERLAP)
            )
    return count


@dataclasses.dataclass(frozen=True)
class GpuTiming:
    """The measured times of one GPU and the fits the UI estimates other sizes with."""

    name: str
    seconds: Mapping[float, float]  # photo megapixels -> seconds
    peak_gib: Mapping[float, float]  # photo megapixels -> peak GPU memory (GiB)
    fit: Tuple[float, float]  # seconds ~ fit[0] * MP + fit[1] * MP^2, one encoder pass
    # the same for a photo whose encoder runs tiled; None: fit
    tiled_fit: Optional[Tuple[float, float]] = None
    # seconds per full encoder window, the decoder included: what a photo with both sides
    # just above ENCODER_TILE_PX takes per window (3088 x 3088 to 4112 x 4112 px, 4
    # windows: within 7 % on the RTX PRO 6000), where the megapixel fits fall short. It
    # is above the photos with more windows (5136 x 5136 px, 9 windows: by 43 %) and
    # with one side a partial window (the 8 MP photos: by 22 % on the RTX PRO 6000, 16 %
    # on the half ZeroGPU slot), so expected_seconds applies it as a floor only when
    # both sides exceed ENCODER_TILE_PX, the reservation to every tiled photo
    # (encoder_windows); None: no floor
    per_window: Optional[float] = None

    def estimate(self, mp: float, tiled: Optional[bool] = None) -> float:
        """Seconds at ``mp`` megapixels: tiled_fit for a tiled encoder (``tiled``;
        None: from TILED_MP up), else fit."""
        if tiled is None:
            tiled = mp > TILED_MP
        a, b = self.tiled_fit if tiled and self.tiled_fit is not None else self.fit
        return a * mp + b * mp * mp


# The RTX PRO 6000's seconds per window: 4112 x 4112 px, 4 windows, 23.3 s (3088 x 3088
# px: 21.7 s, against 6.1 s in one pass; 5136 x 5136 px, 9 windows: 36.5 s). The half
# ZeroGPU slot takes ZEROGPU_PER_RTX times the RTX PRO 6000's time: 1.9, 1.9, 1.8 and
# 1.8 on the same photos at 2, 8, 32 and 102 MP, both with api.predict's encoder
# windows and PyTorch's attention order.
RTX_SECONDS_PER_WINDOW = 5.8
ZEROGPU_PER_RTX = 1.8

# The fits: relative least squares over the measurements, to two significant digits
# (a tiled_fit: the smallest largest relative error). This Space's default slot, half an
# RTX PRO 6000 Blackwell Server Edition, measured on 2026-11-28 on the live Space (the
# app's own GPU seconds and peaks; the release code: encoder windows above 3072 px,
# PyTorch's attention order): one photo area-resized to 1232 x 1632 (2.01 MP, one
# encoder pass), 2448 x 3264, 4896 x 6528 and 6708 x 8944 px (7.99, 31.96 and 60.0 MP)
# and at its native 8750 x 11664 px (102.06 MP), tiled from 8 MP up; the mean of two
# runs at 2, 8 and 32 MP, each time to two significant digits (the rough times the UI
# quotes). tiled_fit: within 3 % of each tiled measurement (8 MP: 14.4 s, the fit 14.0
# s). fit (one encoder pass: photos up to 3072 px, one measurement here) and
# per_window: the RTX PRO 6000's times ZEROGPU_PER_RTX (the fit at 2 MP: 1.1 s)
ZEROGPU = GpuTiming(
    name="ZeroGPU",
    seconds={2: 1.2, 8: 14, 32: 70, 60: 160, 102: 370},
    peak_gib={2: 14.8, 8: 18.6, 32: 21.0, 60: 25.2, 102: 33.1},
    fit=(0.43, 0.070),
    tiled_fit=(1.6, 0.019),
    per_window=ZEROGPU_PER_RTX * RTX_SECONDS_PER_WINDOW,
)
# a local GPU (app.py off a Space; 2.01, 7.99, 31.96, 60.22 and 102.06 MP photos, the
# encoder tiled from 8 MP up: 1 x 2, 2 x 3, 5 x 3 and 4 x 6 windows, width x height).
# fit: one encoder pass (2 MP; the 8 and 32 MP photos with encoder_tiles=None: 4.42 s
# and 48.1 s)
RTX_PRO_6000 = GpuTiming(
    name="RTX PRO 6000 Blackwell",
    seconds={2: 0.635, 8: 7.576, 32: 39.05, 60: 105.57, 102: 205.12},
    peak_gib={2: 14.8, 8: 18.6, 32: 21.0, 60: 24.5, 102: 33.1},
    fit=(0.24, 0.039),
    tiled_fit=(0.92, 0.012),
    per_window=RTX_SECONDS_PER_WINDOW,
)
# the full GPU a ZeroGPU request runs on when its estimated peak exceeds the half slot
# (zerogpu_size; from 121 MP, past the default input limit): one measurement on
# 2026-11-19, a 100.0 MP photo with the encoder in one pass and cuDNN attention first,
# so the half slot's one-pass fit of that run, (0.59, 0.060), scaled by one factor to
# it. The release code tiles such a photo: this one-pass fit stays above the half
# slot's tiled_fit from 79 MP up
ZEROGPU_XLARGE = GpuTiming(
    name="ZeroGPU (full GPU)",
    seconds={100: 379.9},
    peak_gib={100: 40.3},
    fit=(0.34, 0.035),
)
TIMING_ROWS_MP = (2, 8, 32, 60, 100)  # the photo sizes the UI quotes


def measured_mp(table: Mapping[float, float], mp: float) -> Optional[float]:
    """The megapixels of the entry measured within 5 % of ``mp`` (the docs' 100 MP rows
    are 102 MP photos), else None."""
    near = [k for k in table if abs(k - mp) <= 0.05 * mp]
    return min(near, key=lambda k: abs(k - mp)) if near else None


def measured_at(table: Mapping[float, float], mp: float) -> Optional[float]:
    """The entry measured within 5 % of ``mp`` megapixels, else None."""
    k = measured_mp(table, mp)
    return None if k is None else table[k]


def timing_rows(
    gpu: GpuTiming, mps: Tuple[float, ...] = TIMING_ROWS_MP
) -> List[Tuple[float, float, bool]]:
    """(megapixels, seconds, estimated) per size: the measurement at the megapixels it
    was measured at (205 s at 102 MP for 100 MP), else the fit at the size."""
    rows = []
    for mp in mps:
        k = measured_mp(gpu.seconds, mp)
        rows.append(
            (mp, gpu.estimate(mp), True) if k is None else (k, gpu.seconds[k], False)
        )
    return rows


def expected_seconds(height: int, width: int, gpu: GpuTiming = ZEROGPU) -> float:
    """GPU time of one request (the fit of its encoder path, at least per_window per
    encoder window when both sides exceed ENCODER_TILE_PX; no margin): what the UI tells
    a user to expect away from a measured size (quoted_seconds)."""
    seconds = gpu.estimate(megapixels(height, width), encoder_tiled(height, width))
    full = min(int(height), int(width)) > ENCODER_TILE_PX  # no partial window
    if gpu.per_window is not None and full:
        seconds = max(seconds, gpu.per_window * encoder_windows(height, width))
    return seconds


def quoted_seconds(
    height: int, width: int, gpu: GpuTiming = ZEROGPU
) -> Tuple[float, bool]:
    """The GPU time the UI quotes for one request at height x width, and whether it is
    an estimate: the measurement within 5 % of its megapixels on the same encoder path
    (a measurement counts as tiled from TILED_MP, as in GpuTiming.estimate), as the card
    and the GPU-time line quote it (370 s for a 102 MP photo), else expected_seconds."""
    k = measured_mp(gpu.seconds, megapixels(height, width))
    if k is not None and encoder_tiled(height, width) == (k > TILED_MP):
        return gpu.seconds[k], False
    return expected_seconds(height, width, gpu), True


def format_duration(seconds: float, estimate: bool = False) -> str:
    """In seconds only: below 10 s to two significant digits ("0.64 s", "8.0 s", as the
    docs state them), else to the second ("426 s"); an estimate from 100 s up to two
    significant digits ("750 s" for 748 s)."""
    if seconds < 10:
        return f"{seconds:#.2g}".rstrip(".") + " s"  # 8.0 s, 10 s (not "10.")
    if estimate and seconds >= 100:
        seconds = float(f"{seconds:.2g}")
    return f"{round(seconds)} s"


# The two ZeroGPU slots: spaces.GPU(duration=..., size="large" | "xlarge") in spaces
# 0.52.1, the version the Space's build installs (spaces/zero/decorator.py). Hugging
# Face's ZeroGPU docs: "large", the default, is half an NVIDIA RTX PRO 6000 Blackwell
# (48 GB) and counts once against the daily quota; "xlarge" is the whole GPU (96 GB) and
# counts twice. The client (spaces/zero/client.py) sends a large request's duration
# times the GPU's duration_factor, 1.5 for the RTX PRO 6000 Blackwell Server Edition
# (spaces/zero/configs.json), and an xlarge one's as is: ZEROGPU_QUOTA_FACTOR is what a
# request reserves of the quota per second of its duration (quota_seconds).
GPU_LARGE, GPU_XLARGE = "large", "xlarge"
ZEROGPU_QUOTA_FACTOR = {GPU_LARGE: 1.5, GPU_XLARGE: 2.0}

# The ZeroGPU reservation of a request (the spaces.GPU duration): BASE + PER_MP * MP +
# PER_MP2 * MP^2 at the inference megapixels (attention grows with the square of the
# tokens), at least BASE + MARGIN x ZEROGPU.per_window per encoder window (a photo just
# above ENCODER_TILE_PX), the MP terms twice with flip. The coefficients are MARGIN x the
# fit of the slot's timing and the photo's encoder path, to two significant digits: on
# the half slot ZEROGPU.fit for one encoder pass (GPU_SECONDS_PER_MP*), its tiled_fit
# for a photo with a side above ENCODER_TILE_PX (GPU_SECONDS_TILED_*), on the full GPU
# ZEROGPU_XLARGE.fit; BASE covers the first call of a process. ZeroGPU stops a task that
# outruns its duration, hence the margin. The duration is the estimate (at most MAX),
# and check_inference refuses an estimate at or above MAX on ZeroGPU.
GPU_SECONDS_BASE = 20.0
GPU_SECONDS_MARGIN = 1.3
GPU_SECONDS_PER_MP, GPU_SECONDS_PER_MP2 = (
    float(f"{GPU_SECONDS_MARGIN * c:.2g}") for c in ZEROGPU.fit
)
GPU_SECONDS_TILED_PER_MP, GPU_SECONDS_TILED_PER_MP2 = (
    float(f"{GPU_SECONDS_MARGIN * c:.2g}") for c in ZEROGPU.tiled_fit
)
GPU_SECONDS_XLARGE_PER_MP, GPU_SECONDS_XLARGE_PER_MP2 = (
    float(f"{GPU_SECONDS_MARGIN * c:.2g}") for c in ZEROGPU_XLARGE.fit
)
GPU_SECONDS_MAX = 2400  # ZeroGPU's longest task (Hugging Face PRO: 40 min)


def estimate_gpu_seconds(
    height: int,
    width: int,
    size: Optional[int],
    flip: bool,
    gpu_size: str = GPU_LARGE,
) -> float:
    """GPU seconds of one request on the ``gpu_size`` slot (the model above, not
    capped)."""
    h, w = inference_hw(height, width, size)
    mp = megapixels(h, w)
    if gpu_size == GPU_XLARGE:
        per_mp, per_mp2 = GPU_SECONDS_XLARGE_PER_MP, GPU_SECONDS_XLARGE_PER_MP2
    elif encoder_tiled(h, w):
        per_mp, per_mp2 = GPU_SECONDS_TILED_PER_MP, GPU_SECONDS_TILED_PER_MP2
    else:
        per_mp, per_mp2 = GPU_SECONDS_PER_MP, GPU_SECONDS_PER_MP2
    work = per_mp * mp + per_mp2 * mp * mp
    per_window = ZEROGPU.per_window or 0.0  # None: no floor
    windows = GPU_SECONDS_MARGIN * per_window * encoder_windows(h, w)
    return GPU_SECONDS_BASE + max(work, windows) * (2 if flip else 1)


def gpu_seconds(
    height: int,
    width: int,
    size: Optional[int],
    flip: bool,
    gpu_size: str = GPU_LARGE,
) -> int:
    """The spaces.GPU duration on the ``gpu_size`` slot: the estimate rounded up, at
    most GPU_SECONDS_MAX."""
    seconds = estimate_gpu_seconds(height, width, size, flip, gpu_size)
    return int(min(math.ceil(seconds), GPU_SECONDS_MAX))


# ---- the ZeroGPU slot of a request
#
# Peak GPU memory (GiB, torch.cuda.max_memory_allocated: the weights included) of one
# photo: the RTX PRO 6000's of the timing guide (the peak does not depend on the GPU).
# PEAK_GIB_FIT: GiB ~ a + b * MP + c * MP^2, fitted from 8 MP up: within 0.3 GiB of
# these peaks and above them at 8 and 102 MP (below 8 MP the fit overestimates).
PEAK_GIB = dict(RTX_PRO_6000.peak_gib)
PEAK_GIB_FIT = (18.3, 0.052, 0.00092)

# A request runs on the full GPU when its estimated peak exceeds ZEROGPU_LARGE_PEAK_GIB.
# The half slot has 47.5 GiB (half the 101973950464 B configs.json gives the whole
# GPU), the CUDA context included. A 100 MP photo with one encoder pass (41 GiB peak)
# failed there in the encoder with the allocator's NVML assert (is_out_of_memory),
# presumably because the caching allocator holds more than the peak it allocates (free
# blocks split between tensors of other sizes). spaces itself sends a Space whose weights exceed 42 GiB to
# the full GPU (xlarge_threshold, 45097156608 B: 5.5 GiB of the slot left for
# everything else); 38 GiB keeps 4 GiB (10 % of the peak) more for the allocator's
# split blocks. That is 80 % of the slot, and the fit reaches it at 121 MP, so every
# photo up to the default input limit (105 MP, 34 GiB) runs on the half slot.
ZEROGPU_LARGE_PEAK_GIB = 38.0


def estimate_peak_gib(mp: float) -> float:
    """Peak GPU memory (GiB) of an inference at ``mp`` megapixels (PEAK_GIB_FIT)."""
    a, b, c = PEAK_GIB_FIT
    return a + b * mp + c * mp * mp


def zerogpu_size_at(mp: float) -> str:
    """The ZeroGPU slot of an inference at ``mp`` megapixels: GPU_XLARGE when its
    estimated peak exceeds ZEROGPU_LARGE_PEAK_GIB, else GPU_LARGE."""
    return GPU_XLARGE if estimate_peak_gib(mp) > ZEROGPU_LARGE_PEAK_GIB else GPU_LARGE


def zerogpu_size(height: int, width: int, size: Optional[int] = None) -> str:
    """The ZeroGPU slot of a request, at its inference resolution (the mirrored pass
    runs after the first one: the same peak)."""
    h, w = inference_hw(height, width, size)
    return zerogpu_size_at(megapixels(h, w))


def zerogpu_timing(gpu_size: str) -> GpuTiming:
    """The timing guide of a ZeroGPU slot."""
    return ZEROGPU_XLARGE if gpu_size == GPU_XLARGE else ZEROGPU


def zerogpu_timing_rows(
    mps: Tuple[float, ...] = TIMING_ROWS_MP,
) -> List[Tuple[float, float, bool]]:
    """timing_rows of each size on the ZeroGPU slot it runs on, each at the size it
    stands for (the Space's rough times: the 102 MP measurement at 100 MP)."""
    rows = [timing_rows(zerogpu_timing(zerogpu_size_at(mp)), (mp,))[0] for mp in mps]
    return [(mp, seconds, est) for mp, (_, seconds, est) in zip(mps, rows)]


def quota_seconds(height: int, width: int, size: Optional[int], flip: bool) -> int:
    """The daily ZeroGPU quota a request reserves before it starts: its duration on its
    slot times ZEROGPU_QUOTA_FACTOR (what the spaces client asks for)."""
    slot = zerogpu_size(height, width, size)
    seconds = gpu_seconds(height, width, size, flip, slot)
    return round(seconds * ZEROGPU_QUOTA_FACTOR[slot])


# ---- images


def read_rgb(path: os.PathLike, max_mp: float) -> np.ndarray:
    """HxWx3 uint8 RGB with the EXIF orientation applied; the size is checked from the
    header before anything is decoded."""
    import cv2
    from PIL import Image, ImageOps, UnidentifiedImageError

    path = os.fspath(path)
    size = None
    try:
        with warnings.catch_warnings():  # the size is checked right below
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                size = im.size
    except Image.DecompressionBombError as e:
        raise InputError(f"The image is too large for this demo ({e}).") from None
    except (UnidentifiedImageError, OSError):
        pass  # not a format PIL knows: OpenCV may still read it
    if size is not None:
        check_megapixels(size[1], size[0], max_mp)
    image = cv2.imread(path, cv2.IMREAD_COLOR)  # applies the EXIF orientation
    if image is not None:
        check_megapixels(image.shape[0], image.shape[1], max_mp)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size is None:
        raise InputError("Cannot read this image (unsupported format).")
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def preview_hw(height: int, width: int, max_side: int) -> Tuple[int, int]:
    scale = min(1.0, max_side / max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def resize_area(image: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    import cv2

    if tuple(image.shape[:2]) == tuple(hw):
        return image
    return cv2.resize(image, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


# ---- model registry


@dataclasses.dataclass(frozen=True)
class ModelEntry:
    id: str
    label: str
    description: str
    recommended: bool = False
    path: Optional[str] = None
    repo: Optional[str] = None
    filename: Optional[str] = None
    revision: Optional[str] = None
    sha256: Optional[str] = None  # of the downloaded release file
    expect: Dict[str, Any] = dataclasses.field(default_factory=dict, hash=False)

    @property
    def env(self) -> str:
        """Environment variable with a local release file replacing the source."""
        return ckpt_env(self.id)

    @property
    def legacy_env(self) -> str:
        """A second variable for the same file (LDPT_MODEL_<ID>), read after ``env``."""
        return "LDPT_MODEL_" + re.sub(r"[^A-Z0-9]", "_", self.id.upper())

    @property
    def placeholder(self) -> bool:
        return self.path is None and any(
            "TODO" in (v or "") for v in (self.repo, self.filename, self.revision)
        )


def ckpt_env(model_id: str) -> str:
    """LDPT_CKPT_ + the id without its "l-dpt_" prefix, upper-cased, other characters
    -> "_": for a preset id, the variable of its local file in api.load
    (presets.WEIGHTS: LDPT_CKPT_LARGE_DEPTH_ZEROSHOT for l-dpt_large_depth_zeroshot)."""
    name = model_id[len("l-dpt_") :] if model_id.startswith("l-dpt_") else model_id
    return "LDPT_CKPT_" + re.sub(r"[^A-Z0-9]", "_", name.upper())


_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ENTRY_KEYS = {"id", "label", "description", "recommended", "source", "expect"}
_OPTIONAL_KEYS = {"recommended", "expect"}
# release-config keys an entry's "expect" may pin, with the LatentDPT defaults (what a
# release config without the key builds)
EXPECT_DEFAULTS: Dict[str, Any] = dict(
    head="sigmoid", dec_qk_norm=False, attn_fp32=None, dec_qk_bound=None
)


def _public(mapping: Mapping[str, Any]) -> set:
    return {k for k in mapping if not str(k).startswith("_")}


def parse_registry(data: Any) -> List[ModelEntry]:
    """Model entries of a models.json document, validated (ValueError on any problem;
    keys starting with "_" are comments)."""
    if not isinstance(data, Mapping) or _public(data) != {"models"}:
        raise ValueError('models.json: expected {"models": [...]}')
    models = data["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models.json: 'models' must be a non-empty list")
    entries, seen = [], {}
    for i, raw in enumerate(models):
        where = f"models.json: models[{i}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}: expected an object")
        unknown = _public(raw) - _ENTRY_KEYS
        missing = _ENTRY_KEYS - _OPTIONAL_KEYS - set(raw)
        if unknown or missing:
            raise ValueError(
                f"{where}: unknown keys {sorted(unknown)}, missing {sorted(missing)}"
            )
        mid = raw["id"]
        if not isinstance(mid, str) or not _ID.fullmatch(mid):
            raise ValueError(f"{where}: id {mid!r} must match {_ID.pattern}")
        for key in ("label", "description"):
            if not isinstance(raw[key], str) or not raw[key].strip():
                raise ValueError(f"{where} ({mid}): {key} must be a non-empty string")
        recommended = raw.get("recommended", False)
        if not isinstance(recommended, bool):
            raise ValueError(f"{where} ({mid}): recommended must be true or false")
        source = raw["source"]
        if not isinstance(source, Mapping):
            raise ValueError(f"{where} ({mid}): source must be an object")
        keys = _public(source)
        if keys == {"path"}:
            fields = dict(path=source["path"])
        elif {"repo", "filename"} <= keys <= {"repo", "filename", "revision", "sha256"}:
            fields = dict(
                repo=source["repo"],
                filename=source["filename"],
                revision=source.get("revision", "main"),
            )
            if "sha256" in keys:
                fields["sha256"] = source["sha256"]
        else:
            raise ValueError(
                f"{where} ({mid}): source is either {{path}} or {{repo, filename, "
                f"revision, sha256}}, got {sorted(keys)}"
            )
        if not all(isinstance(v, str) and v.strip() for v in fields.values()):
            raise ValueError(
                f"{where} ({mid}): source values must be non-empty strings"
            )
        if "sha256" in fields and not _SHA256.fullmatch(fields["sha256"]):
            raise ValueError(
                f"{where} ({mid}): source sha256 must be 64 lowercase hex digits"
            )
        expect = raw.get("expect", {})
        if not isinstance(expect, Mapping) or not set(expect) <= set(EXPECT_DEFAULTS):
            raise ValueError(
                f"{where} ({mid}): expect must be an object with keys from "
                f"{sorted(EXPECT_DEFAULTS)}"
            )
        if not all(
            v is None or isinstance(v, (bool, int, float, str)) for v in expect.values()
        ):
            raise ValueError(f"{where} ({mid}): expect values must be JSON scalars")
        entry = ModelEntry(
            id=mid,
            label=raw["label"].strip(),
            description=raw["description"].strip(),
            recommended=recommended,
            expect=dict(expect),
            **fields,
        )
        for key, value in (
            ("id", entry.id),
            ("label", entry.label),
            ("env", entry.env),
        ):
            if (key, value) in seen:
                raise ValueError(
                    f"{where}: {key} {value!r} already used by {seen[key, value]!r}"
                )
            seen[key, value] = mid
        entries.append(entry)
    return entries


@dataclasses.dataclass
class Registry:
    models: List[ModelEntry]
    base: Path  # folder of models.json: local paths are relative to it

    @classmethod
    def load(cls, path: os.PathLike = HERE / "models.json") -> "Registry":
        path = Path(path)
        return cls(parse_registry(json.loads(path.read_text())), path.parent)

    def get(self, model_id: str) -> ModelEntry:
        for entry in self.models:
            if entry.id == model_id:
                return entry
        raise InputError(
            f"unknown model {model_id!r}: one of {[m.id for m in self.models]}"
        )

    @property
    def default(self) -> ModelEntry:
        return next((m for m in self.models if m.recommended), self.models[0])

    def choices(self) -> List[Tuple[str, str]]:
        """(label, id) pairs for a dropdown."""
        return [
            (m.label + (" · recommended" if m.recommended else ""), m.id)
            for m in self.models
        ]


def _hf_download(**kwargs) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def _hf_snapshot(**kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _sha256(path: Path) -> str:
    """sha256 of a file, hashed once per (path, size, mtime) (ldpt's cache)."""
    from ldpt.models.l_dpt.weights import verified_sha256

    return verified_sha256(path)


def _local_file(value: str, what: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise SetupError(f"{what}: {path} is not a file")
    return path


def _fetch(what: str, hint: str, fetch: Callable[..., str], **kwargs) -> Path:
    """``fetch(**kwargs)`` (a Hugging Face download); any failure is a SetupError."""
    try:
        return Path(fetch(**kwargs))
    except Exception as e:  # missing repo / file, no access, offline, ...
        raise SetupError(
            f"{what}: download failed ({type(e).__name__}: {str(e)[:300]}). {hint}"
        ) from None


def resolve_release(
    entry: ModelEntry,
    base: Path,
    env: Mapping[str, str] = os.environ,
    download: Callable[..., str] = _hf_download,
    verify: bool = True,
    sha256: Callable[[Path], str] = _sha256,
) -> Path:
    """The local release file of a model: $LDPT_CKPT_<NAME> (entry.env), else
    $LDPT_MODEL_<ID> (entry.legacy_env), else its source (a download is checked
    against the source's sha256 when it has one and ``verify``)."""
    for var in (entry.env, entry.legacy_env):
        if env.get(var):
            return _local_file(env[var], var)
    if entry.path is not None:
        path = Path(entry.path).expanduser()
        return _local_file(str(path if path.is_absolute() else base / path), entry.id)
    if entry.placeholder:
        raise SetupError(
            f"{entry.label}: the weights are not published yet (TODO source in "
            f"models.json); set {entry.env} to its release file"
        )
    path = _fetch(
        f"{entry.label} ({entry.repo}/{entry.filename} at {entry.revision})",
        f"A private repo needs the HF_TOKEN secret of an account that can read it; "
        f"or set {entry.env} to a local release file.",
        download,
        repo_id=entry.repo,
        filename=entry.filename,
        revision=entry.revision,
        token=env.get("HF_TOKEN") or None,
    )
    if verify and entry.sha256 and sha256(path) != entry.sha256:
        raise SetupError(
            f"{entry.label}: {path} is not the release file models.json pins (sha256 "
            f"{entry.sha256[:12]}...): delete it from the Hugging Face cache and restart"
        )
    return path


def check_expect(entry: ModelEntry, config: Mapping[str, Any], release: Path) -> None:
    """SetupError unless the release ``config`` has the values ``entry.expect`` pins (a
    missing key: its EXPECT_DEFAULTS value)."""
    wrong = [
        f"{key}={config.get(key, EXPECT_DEFAULTS[key])!r} (models.json expects {value!r})"
        for key, value in entry.expect.items()
        if config.get(key, EXPECT_DEFAULTS[key]) != value
    ]
    if wrong:
        raise SetupError(
            f"{entry.label}: the release file {release} does not match its models.json "
            f"entry {entry.id!r}: " + ", ".join(wrong)
        )


def _release_config(path: Path) -> Mapping[str, Any]:
    from ldpt.models.l_dpt.weights import read_checkpoint  # memory-mapped torch.load

    return read_checkpoint(path).get("config") or {}


DINOV3_ENV = "LDPT_DINOV3"
# Meta's gated DINOv3 ViT-7B repo in the transformers format (config.json + 6
# safetensors shards, 27 GB fp32), pinned to a commit
DINOV3_HF_REPO = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
DINOV3_HF_REVISION = "b80367753773648a6793235ab9c65cdbb029506f"
DINOV3_HF_FILES = [
    "config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "model.safetensors",
]


def resolve_dinov3(
    env: Mapping[str, str] = os.environ,
    download: Callable[..., str] = _hf_download,
    snapshot: Callable[..., str] = _hf_snapshot,
) -> Path:
    """The DINOv3 ViT-7B weights: $LDPT_DINOV3 (a .pth file or a transformers folder),
    else $LDPT_DINOV3_REPO + $LDPT_DINOV3_FILE ($LDPT_DINOV3_REVISION optional), else a
    snapshot of $LDPT_DINOV3_HF_REPO (default DINOV3_HF_REPO at DINOV3_HF_REVISION;
    "none": off); downloads with $HF_TOKEN."""
    token = env.get("HF_TOKEN") or None
    if env.get(DINOV3_ENV):
        path = Path(env[DINOV3_ENV]).expanduser()
        return path if path.is_dir() else _local_file(str(path), DINOV3_ENV)
    repo, filename = env.get("LDPT_DINOV3_REPO"), env.get("LDPT_DINOV3_FILE")
    if repo and filename:
        return _fetch(
            f"DINOv3 ViT-7B ({repo}/{filename})",
            "Check LDPT_DINOV3_REPO / LDPT_DINOV3_FILE and the HF_TOKEN secret.",
            download,
            repo_id=repo,
            filename=filename,
            revision=env.get("LDPT_DINOV3_REVISION") or None,
            token=token,
        )
    if repo or filename:
        raise SetupError("set both LDPT_DINOV3_REPO and LDPT_DINOV3_FILE")
    hf_repo = env.get("LDPT_DINOV3_HF_REPO") or DINOV3_HF_REPO
    if hf_repo.strip().lower() not in ("none", "off", "0"):
        revision = env.get("LDPT_DINOV3_HF_REVISION") or (
            DINOV3_HF_REVISION if hf_repo == DINOV3_HF_REPO else None
        )
        return _fetch(
            f"DINOv3 ViT-7B ({hf_repo})",
            f"The repo is gated: request access at https://huggingface.co/{hf_repo} "
            f"with the account whose token is the HF_TOKEN secret"
            + ("" if token else " (no HF_TOKEN is set)")
            + "; or set LDPT_DINOV3 to a local copy.",
            snapshot,
            repo_id=hf_repo,
            revision=revision,
            allow_patterns=DINOV3_HF_FILES,
            token=token,
        )
    raise SetupError(
        "The DINOv3 ViT-7B weights are not redistributable: set LDPT_DINOV3 to a local "
        "copy (dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth or a transformers-format "
        f"folder), or let the Space download {DINOV3_HF_REPO} with the HF_TOKEN "
        "secret of an account granted access (unset LDPT_DINOV3_HF_REPO)"
    )


# ---- settings and the model service


def _env_flag(value: Optional[str], default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _env_optional_int(value: Optional[str], default: Optional[int]) -> Optional[int]:
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in ("0", "none", "off"):
        return None
    return int(value)


def _env_optional_float(
    value: Optional[str], default: Optional[float]
) -> Optional[float]:
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in ("none", "off"):
        return None
    return float(value)


def zero_gpu_active(env: Mapping[str, str] = os.environ) -> bool:
    """A ZeroGPU Space (spaces.config: SPACES_ZERO_GPU=1 / t / true)."""
    return (env.get("SPACES_ZERO_GPU") or "").strip().lower() in ("1", "t", "true")


ZERO_GPU_MAX_INFER_MP = 105.0  # ZeroGPU defaults: native inference up to the input
ZERO_GPU_MAX_FLIP_MP = 32.0  # limit (gpu_seconds stays under MAX), flip up to 32 MP


@dataclasses.dataclass
class Settings:
    max_mp: float = 105.0  # input images above it are refused
    max_infer_mp: float = 105.0  # inference resolutions above it are refused
    max_flip_mp: Optional[float] = None  # mirrored pass above it refused (None: any)
    max_gpu_seconds: Optional[float] = None  # longer estimates refused (ZeroGPU)
    preview_side: int = 2048  # long side of the images shown in the browser
    device: str = "auto"
    precision: Optional[str] = None  # None: each model's own (bf16 for the releases)
    selection_impl: str = "stream"  # high-res memory options of api.load
    chunk_tokens: Optional[int] = 32768
    verify: bool = True  # sha256 of the DINOv3 .pth / a downloaded release file
    # (hashed once), config.json of a DINOv3 transformers folder
    preload: bool = True  # load the default model at startup (app.py --lazy: no)
    output_dir: Path = Path(tempfile.gettempdir()) / "ldpt_space"
    keep_outputs_s: float = 3600.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        """Settings from LDPT_* variables; on ZeroGPU (zero_gpu_active) the defaults
        are inference up to 105 MP and flip up to 32 MP and the GPU-seconds limit."""
        d = cls()
        if zero_gpu_active(env):
            d = cls(
                max_infer_mp=ZERO_GPU_MAX_INFER_MP,
                max_flip_mp=ZERO_GPU_MAX_FLIP_MP,
                max_gpu_seconds=GPU_SECONDS_MAX,
                preload=True,
            )
        get = env.get
        return cls(
            max_mp=float(get("LDPT_MAX_MP") or d.max_mp),
            max_infer_mp=float(get("LDPT_MAX_INFER_MP") or d.max_infer_mp),
            max_flip_mp=_env_optional_float(get("LDPT_MAX_FLIP_MP"), d.max_flip_mp),
            max_gpu_seconds=d.max_gpu_seconds,
            preview_side=int(get("LDPT_PREVIEW_SIDE") or d.preview_side),
            device=get("LDPT_DEVICE") or d.device,
            precision=get("LDPT_PRECISION") or d.precision,
            selection_impl=get("LDPT_SELECTION_IMPL") or d.selection_impl,
            chunk_tokens=_env_optional_int(get("LDPT_CHUNK_TOKENS"), d.chunk_tokens),
            verify=_env_flag(get("LDPT_VERIFY"), d.verify),
            preload=_env_flag(get("LDPT_PRELOAD"), d.preload),
            output_dir=Path(get("LDPT_OUTPUT_DIR") or d.output_dir),
            keep_outputs_s=float(get("LDPT_KEEP_OUTPUTS") or d.keep_outputs_s),
        )


# A failed allocation of the CUDA caching allocator on ZeroGPU: its out-of-memory report
# queries NVML, which the container does not answer, so torch raises "NVML_SUCCESS == r
# INTERNAL ASSERT FAILED at .../CUDACachingAllocator.cpp" (a RuntimeError) instead of
# torch.cuda.OutOfMemoryError (a 100 MP photo on the half slot, in the encoder).
ALLOCATOR_ASSERTS = ("CUDACachingAllocator", "NVML_SUCCESS")


def is_out_of_memory(e: BaseException) -> bool:
    """``e`` is torch.cuda.OutOfMemoryError or the allocator's assert above."""
    if type(e).__name__ == "OutOfMemoryError":
        return True
    return isinstance(e, RuntimeError) and any(s in str(e) for s in ALLOCATOR_ASSERTS)


def user_error(e: BaseException) -> Optional[Exception]:
    """The InputError / SetupError behind ``e`` (the message to show), else None: ``e``
    itself, or the exception a ZeroGPU task raised, which spaces re-raises in the app's
    process as a ZeroGPUTaskException holding it (``exception``)."""
    for err in (e, getattr(e, "exception", None)):
        if isinstance(err, (InputError, SetupError)):
            return err
    return None


def _torch():
    """torch if it is imported (the stub-tested paths never import it)."""
    return sys.modules.get("torch")


def _cuda_ready() -> bool:
    torch = _torch()
    return torch is not None and torch.cuda.is_available()


class DepthService:
    """Keeps every model it loaded: the first one reads the DINOv3 weights, the next
    ones share its encoder module (api.load share_encoder=), so a second model costs
    only its decoder (~1 GB) and a switch loads nothing twice. On ZeroGPU every model
    must be loaded at startup (app.py): a load after it fails (CUDA outside
    spaces.GPU). ``loader(release_path, dinov3_path, encoder)`` (``encoder``: the
    module to share, None for the first load, the only one given a ``dinov3_path``)
    and ``predictor(model, rgb, size, flip)`` default to api.load (high-res memory
    options) / api.predict; ``config_reader(release_path)`` (the release file's
    config) feeds check_expect before a load; ``download`` / ``snapshot`` stand for
    hf_hub_download / snapshot_download. A prediction with a non-finite value is
    refused (InputError)."""

    def __init__(
        self,
        registry: Registry,
        settings: Optional[Settings] = None,
        *,
        env: Mapping[str, str] = os.environ,
        download: Callable[..., str] = _hf_download,
        snapshot: Callable[..., str] = _hf_snapshot,
        loader: Optional[Callable[[Path, Optional[Path], Any], Any]] = None,
        predictor: Optional[Callable[..., np.ndarray]] = None,
        config_reader: Optional[Callable[[Path], Mapping[str, Any]]] = None,
    ):
        self.registry = registry
        self.settings = settings or Settings.from_env(env)
        self.env, self.download, self.snapshot = env, download, snapshot
        self.loader = loader or self._api_load
        self.predictor = predictor or self._api_predict
        self.config_reader = config_reader or _release_config
        self.models: Dict[str, Any] = {}
        self.load_errors: Dict[str, str] = {}  # ensure_all(keep_going=True) skips
        self.frozen = False  # freeze(): no load any more (ZeroGPU after startup)
        self._lock = threading.RLock()

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "DepthService":
        return cls(Registry.load(), Settings.from_env(env), env=env)

    def _api_load(self, release: Path, dinov3: Optional[Path], encoder: Any) -> Any:
        from ldpt.models.l_dpt import api

        s = self.settings
        return api.load(
            release,
            encoder_weights=dinov3,
            share_encoder=encoder,
            device=s.device,
            precision=s.precision,
            verify=s.verify,
            selection_impl=s.selection_impl,
            chunk_tokens=s.chunk_tokens,
        )

    @staticmethod
    def _api_predict(model: Any, rgb: np.ndarray, size: Optional[int], flip: bool):
        from ldpt.models.l_dpt import api

        return api.predict(model, rgb, size=size, flip=flip)

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self.models

    def _shared_encoder(self) -> Any:
        """The encoder module of a loaded model, else None."""
        for model in self.models.values():
            encoder = getattr(model, "encoder", None)
            if encoder is not None:
                return encoder
        return None

    def ensure(self, model_id: str) -> float:
        """Load ``model_id`` unless it is loaded; seconds spent loading."""
        with self._lock:
            if self.is_loaded(model_id):
                return 0.0
            entry = self.registry.get(model_id)
            if self.frozen:
                raise SetupError(
                    f"{entry.label} is not available: it could not be loaded when "
                    f"this Space started (see the Space logs)."
                )
            release = resolve_release(
                entry,
                self.registry.base,
                self.env,
                self.download,
                verify=self.settings.verify,
            )
            check_expect(entry, self.config_reader(release), release)
            encoder = self._shared_encoder()
            dinov3 = (
                None
                if encoder is not None
                else resolve_dinov3(self.env, self.download, self.snapshot)
            )
            t0 = time.perf_counter()
            self.models[model_id] = self.loader(release, dinov3, encoder)
            return time.perf_counter() - t0

    def ensure_all(self, keep_going: bool = False) -> float:
        """Load every registry model (the default first); seconds spent loading. With
        ``keep_going`` a model other than the default that fails to load is skipped
        (logged, its error in ``load_errors``); the default's failure always raises."""
        default = self.registry.default.id
        seconds = self.ensure(default)
        for model_id in [m.id for m in self.registry.models if m.id != default]:
            try:
                seconds += self.ensure(model_id)
            except Exception as e:
                if not keep_going:
                    raise
                self.load_errors[model_id] = f"{type(e).__name__}: {e}"
                log.error("model %r not loaded: %s", model_id, e, exc_info=True)
        return seconds

    def freeze(self) -> None:
        """No load after this: ensure() of a model not loaded raises SetupError (on
        ZeroGPU a load after startup cannot work)."""
        self.frozen = True

    def release(self) -> None:
        """Drop every loaded model."""
        with self._lock:
            self.models.clear()
            gc.collect()
            if _cuda_ready():
                _torch().cuda.empty_cache()

    def predict(
        self, model_id: str, rgb: np.ndarray, size: Optional[int], flip: bool
    ) -> Tuple[np.ndarray, float, Optional[int]]:
        """(HxW float32 depth, seconds, peak GPU bytes or None)."""
        with self._lock:
            self.ensure(model_id)
            cuda = _cuda_ready()
            torch = _torch()
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            oom = False
            try:
                depth = self.predictor(self.models[model_id], rgb, size, flip)
                if cuda:
                    torch.cuda.synchronize()
            except Exception as e:
                if not is_out_of_memory(e):
                    raise
                oom = True  # leaving the clause drops e: its traceback holds the frames
            h, w = inference_hw(rgb.shape[0], rgb.shape[1], size)
            if oom:  # (and their GPU tensors), so they are freed before empty_cache
                gc.collect()
                if cuda:
                    try:
                        torch.cuda.empty_cache()
                    except Exception as e:  # the message below matters more
                        log.warning("empty_cache after running out of memory: %s", e)
                raise InputError(
                    f"Out of GPU memory at {w}x{h} ({megapixels(h, w):.1f} MP). "
                    f"Downscale the photo and try again."
                )
            seconds = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() if cuda else None
            depth = np.asarray(depth, dtype=np.float32)
            bad = depth.size - int(np.count_nonzero(np.isfinite(depth)))
            if bad:
                raise InputError(
                    f"The model returned {bad} non-finite depth values at {w}x{h}; "
                    f"nothing was written. Try a downscaled copy of the photo."
                )
            return depth, seconds, peak


# ---- outputs


@dataclasses.dataclass
class DepthResult:
    depth_preview: np.ndarray  # colorized, at most preview_side on the long side
    input_preview: np.ndarray  # the input at the same size
    files: List[str]  # 16-bit PNG, full-resolution colorized JPEG (PNG), sidecar JSON
    info: str
    meta: Dict[str, Any]


def safe_stem(path: Optional[os.PathLike]) -> str:
    stem = Path(os.fspath(path)).stem if path else "image"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:64] or "image"


def new_output_dir(root: Path, keep_s: float) -> Path:
    """A fresh folder under ``root``; folders older than ``keep_s`` are removed."""
    root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for old in root.iterdir():
        try:
            if old.is_dir() and now - old.stat().st_mtime > keep_s:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    return Path(tempfile.mkdtemp(prefix="run-", dir=root))


def format_bytes(n: Optional[int]) -> str:
    return "n/a (no GPU)" if n is None else f"{n / 2**30:.1f} GiB"


JPEG_MAX_SIDE = 65500  # libjpeg's largest side; a longer colorized depth is a PNG


def render_result(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    entry: ModelEntry,
    size: Optional[int],
    flip: bool,
    cmap: str,
    seconds: float,
    peak: Optional[int],
    settings: Settings,
    name: Optional[os.PathLike] = None,
    load_seconds: float = 0.0,
) -> DepthResult:
    """Previews, download files and the info line of one prediction. The colorized
    depth is a JPEG, or a PNG when a side exceeds JPEG_MAX_SIDE (meta color_file); it
    and the preview share the tone curve of the full-resolution depth (meta
    color_curve: ToneCurve(**color_curve) colours the 16-bit depth the same)."""
    import cv2

    H, W = rgb.shape[:2]
    if depth.shape != (H, W):
        raise ValueError(f"depth {depth.shape} does not match the image {(H, W)}")
    ih, iw = inference_hw(H, W, size)
    curve = tone_curve(depth)
    meta = dict(
        model=entry.id,
        model_label=entry.label,
        input_size=[H, W],
        inference_size=[ih, iw],
        flip=bool(flip),
        colormap=cmap,
        color_scale="tone curve",
        color_curve=curve.as_dict(),  # depths lo / hi, blend w: see ToneCurve
    )
    out = new_output_dir(settings.output_dir, settings.keep_outputs_s)
    stem = f"{safe_stem(name)}_{entry.id}"
    png = out / f"{stem}_depth16.png"
    jpeg = max(H, W) <= JPEG_MAX_SIDE
    color_path = out / f"{stem}_depth_{cmap.lower()}.{'jpg' if jpeg else 'png'}"
    meta["color_file"] = color_path.name
    meta = write_depth16(png, depth, meta)
    sidecar = out / f"{stem}_depth16.json"
    sidecar.write_text(json.dumps(meta, indent=1, sort_keys=True))
    color = colorize(depth, cmap, bgr=True, curve=curve)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        ok, buf = cv2.imencode(".png", color, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    del color
    if not ok:
        raise RuntimeError(f"{color_path.suffix} encoding failed")
    color_path.write_bytes(buf.tobytes())

    hw = preview_hw(H, W, settings.preview_side)
    depth_preview = colorize(resize_area(depth, hw), cmap, curve=curve)
    input_preview = resize_area(rgb, hw)
    flip_note = " + flip" if flip else ""
    if (ih, iw) == (H, W):  # input, inference and output are one size
        parts = [f"{W}x{H} ({megapixels(H, W):.1f} MP), native resolution{flip_note}"]
    else:
        parts = [
            f"input {W}x{H} ({megapixels(H, W):.1f} MP)",
            f"inference {iw}x{ih} ({megapixels(ih, iw):.1f} MP){flip_note}",
            f"output {W}x{H} ({megapixels(H, W):.1f} MP)",
        ]
    parts += [
        entry.label,
        f"{seconds:.1f} s"
        + (f" (+ {load_seconds:.0f} s model load)" if load_seconds >= 0.5 else ""),
        f"peak GPU {format_bytes(peak)}",
    ]
    if hw != (H, W):
        parts.append(f"preview {hw[1]}x{hw[0]}, full resolution in the downloads")
    return DepthResult(
        depth_preview=depth_preview,
        input_preview=input_preview,
        files=[str(png), str(color_path), str(sidecar)],
        info=" · ".join(parts),
        meta=meta,
    )


def list_examples(folder: Path = HERE / "examples") -> List[str]:
    if not folder.is_dir():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in suffixes)


@dataclasses.dataclass(frozen=True)
class Example:
    """One example photo: the file the model runs on, a small thumbnail, optionally
    precomputed results (an RGB preview and, per model id, a depth preview with the
    seconds and device of the run that made it) and the photo's credit."""

    name: str
    path: str
    thumb: str
    megapixels: Optional[float] = None
    rgb_preview: Optional[str] = None
    results: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
    credit: Optional[str] = None  # who took the photo, and where it is from

    @property
    def title(self) -> str:
        """A readable name: "bamboo-river" -> "Bamboo river"."""
        name = self.name.replace("-", " ").replace("_", " ")
        return name[:1].upper() + name[1:]

    @property
    def caption(self) -> str:
        """The title; the megapixel count is added only when the example has no
        thumbnail of its own: the example thumbnails carry it as a badge."""
        if self.megapixels and self.thumb == self.path:
            return f"{self.title} · {self.megapixels:g} MP"
        return self.title

    def result(self, model_id: str) -> Optional[Mapping[str, Any]]:
        """The precomputed result of ``model_id`` (with the RGB preview), else None."""
        r = self.results.get(model_id)
        return r if r is not None and self.rgb_preview else None


EXAMPLES_FORMAT = "l-dpt-examples/1"
EXAMPLES_DIR = Path(tempfile.gettempdir()) / "ldpt_examples"  # gradio serves temp files


def read_examples(folder: Path) -> List[Example]:
    """The examples of a folder: its examples.json ({"format": EXAMPLES_FORMAT,
    "examples": [{"name", "file", "thumb", "megapixels", "credit", "results": {"rgb":
    <preview>, "models": {<model id>: {"preview", "seconds", "device"}}}}, ...]}, paths
    relative to the folder; missing files skipped), else every image in it as its own
    thumbnail."""
    index = folder / "examples.json"
    if not index.is_file():
        return [Example(Path(p).stem, p, p) for p in list_examples(folder)]
    doc = json.loads(index.read_text())
    if doc.get("format") != EXAMPLES_FORMAT:
        raise SetupError(
            f"{index}: format {doc.get('format')!r}, not {EXAMPLES_FORMAT}"
        )
    out = []
    for e in doc.get("examples", []):
        path, thumb = folder / e["file"], folder / e.get("thumb", e["file"])
        if path.is_file():
            thumb = thumb if thumb.is_file() else path
            mp = e.get("megapixels")
            res = e.get("results") or {}
            rgb = folder / res["rgb"] if res.get("rgb") else None
            models = {
                k: dict(v, preview=str(folder / v["preview"]))
                for k, v in (res.get("models") or {}).items()
                if (folder / v["preview"]).is_file()
            }
            out.append(
                Example(
                    e["name"],
                    str(path),
                    str(thumb),
                    mp and float(mp),
                    str(rgb) if rgb is not None and rgb.is_file() else None,
                    models,
                    (e.get("credit") or "").strip() or None,
                )
            )
    return out


EXAMPLES_DIR_ENV = "LDPT_EXAMPLES_DIR"


def local_examples_dir(env: Mapping[str, str] = os.environ) -> Path:
    """The local folder of examples: $LDPT_EXAMPLES_DIR (app.py --examples), else the
    Space's own examples/ folder (make_examples.py)."""
    value = env.get(EXAMPLES_DIR_ENV)
    return Path(value).expanduser() if value else HERE / "examples"


def resolve_examples(
    env: Mapping[str, str] = os.environ,
    snapshot: Callable[..., str] = _hf_snapshot,
    local: Optional[Path] = None,
) -> List[Example]:
    """The examples: $LDPT_EXAMPLES_REPO (a dataset repo laid out as read_examples
    expects, at $LDPT_EXAMPLES_REVISION, read with $HF_TOKEN) downloaded into
    EXAMPLES_DIR, else the local folder (default local_examples_dir: a folder of
    photos works). A failed download leaves the local examples (logged); no folder
    and no repo: no examples."""
    local = local_examples_dir(env) if local is None else local
    repo = env.get("LDPT_EXAMPLES_REPO")
    if repo:
        try:
            folder = snapshot(
                repo_id=repo,
                repo_type="dataset",
                revision=env.get("LDPT_EXAMPLES_REVISION") or None,
                token=env.get("HF_TOKEN") or None,
                local_dir=str(EXAMPLES_DIR),
            )
            return read_examples(Path(folder))
        except Exception as e:
            log.error("examples from %s not loaded: %s", repo, e)
    return read_examples(local)
