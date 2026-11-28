"""
Depth as images, one convention for scripts/demo_depth.py, scripts/highres_infer.py and
the demo (demo/hf_space):

- colour: near blue, far red (matplotlib "Spectral", reversed, without its deep-purple
  and deep-red ends) through a tone curve fitted to each image (ToneCurve): its 1 and
  99 % depths anchor it; a blend of linear and log depth (at most 40 % log), chosen so
  that the median depth lands in blue-teal, opens up the near layers of a near-heavy
  photo a little and keeps a far-heavy one even; soft knees roll the tails off instead
  of clipping them. The colours depend on the depth only up to a scale and a shift, and
  rise monotonically from near to far;
- 16-bit: a grayscale PNG mapped linearly from the depth's minimum to its maximum,
  ``depth = lo + value / 65535 * (hi - lo)``, with lo / hi (and any metadata) as JSON in
  its tEXt chunk "l-dpt".

    from ldpt.utils.depth_image import colorize, tone_curve, write_depth16

    rgb = colorize(depth)                   # HxW relative depth -> HxWx3 uint8 RGB
    curve = tone_curve(depth)               # the fitted curve (lo, hi, w) ...
    small = colorize(preview, curve=curve)  # ... colours a preview of it alike
    info = write_depth16("d.png", depth)    # {"lo": ..., "hi": ..., ...}

Only numpy is imported here; matplotlib (the colormaps) and OpenCV (the PNG codec) are
imported when used. The curve is fitted on a strided grid of at most FIT_SAMPLES pixels
and applied as a lookup table, in blocks of CHUNK_PIXELS pixels: a 100 MP depth needs
no full-size float copy.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import dataclasses
import functools
import json
import math
import os
import struct
import warnings
import zlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "COLORMAPS",
    "DEPTH16_FORMAT",
    "DISPLAY_CMAP",
    "DISPLAY_MEDIAN_T",
    "DISPLAY_PERCENTILES",
    "DISPLAY_T_RANGE",
    "PNG_TEXT_KEY",
    "ToneCurve",
    "colorize",
    "colormap_lut",
    "decode_depth16",
    "depth16_png",
    "depth_range",
    "encode_depth16",
    "read_depth16",
    "tone_curve",
    "write_depth16",
]

# colormaps by name (matplotlib's), oriented for the tone curve's position t: 0 near,
# 1 far
COLORMAPS: Dict[str, str] = {
    "Spectral": "Spectral_r",  # near blue, far red
    "Turbo": "turbo",  # near blue, far red
    "Inferno": "inferno_r",  # near bright yellow, far dark purple
    "Gray": "gray_r",  # near white, far dark
}
DISPLAY_CMAP = "Spectral"
DISPLAY_PERCENTILES = (1.0, 99.0)  # the depths at the tone curve's u = 0 and u = 1
DISPLAY_MEDIAN_T = 0.14  # where the median depth lands (blue-teal on Spectral)
DISPLAY_T_RANGE = (0.015, 0.87)  # the part of a colormap in use: t0 near, t1 far
FIT_SAMPLES = 2_000_000  # pixels the curve is fitted on, at most (a strided grid)
CHUNK_PIXELS = 1 << 22  # pixels per block
_LOG_ALPHA = 0.05  # L(u) = log(1 + u / a) / log(1 + 1 / a): opens u < ~10 a
# the blend: 0 linear, 0.4 the most log, < 0 mirrored (opens the far band)
_W_RANGE = (-0.1, 0.4)
_KNEES = (0.05, 0.25)  # widths of the soft knees at the near / far end
_LUT_NODES = 1 << 16  # entries of an image's lookup table ...
_LUT_MARGIN = 2.0  # ... spanning u in [-2, 3]
_LOG_FLOOR = 1e-12


# ---- colour


@functools.lru_cache(maxsize=None)
def _colormap_rgb(cmap: str) -> Tuple[np.ndarray, np.ndarray]:
    """(positions, [N, 3] float RGB) of a COLORMAPS entry's N colours (matplotlib's
    table, colour i at i / (N - 1)); ValueError for another name."""
    if cmap not in COLORMAPS:
        raise ValueError(f"unknown colormap {cmap!r}: one of {list(COLORMAPS)}")
    import matplotlib

    cm = matplotlib.colormaps[COLORMAPS[cmap]]
    rgb = np.asarray(cm(np.arange(cm.N)), np.float64)[:, :3]
    x = np.linspace(0.0, 1.0, cm.N)
    x.setflags(write=False)
    rgb.setflags(write=False)
    return x, rgb


def _rgb_at(cmap: str, t: np.ndarray) -> np.ndarray:
    """uint8 RGB of a COLORMAPS entry at positions t in [0, 1] (linear between its
    colours: no steps).

    Shapes:
        t: [N] -> [N, 3] uint8
    """
    x, rgb = _colormap_rgb(cmap)
    out = np.stack([np.interp(t, x, rgb[:, c]) for c in range(3)], axis=-1)
    return np.round(out * 255.0).astype(np.uint8)


def colormap_lut(cmap: str = DISPLAY_CMAP, size: int = 256) -> np.ndarray:
    """[size, 3] uint8 RGB of a COLORMAPS entry from t = 0 (near) to t = 1 (far);
    ValueError for another name. colorize uses its DISPLAY_T_RANGE part."""
    return _rgb_at(cmap, np.linspace(0.0, 1.0, size))


def _blend(u: np.ndarray, w: float) -> np.ndarray:
    """(1 - w) u + w L(u) for 0 <= w <= 1; below u = 0 linear with its slope at 0."""
    a = _LOG_ALPHA
    log_u = np.log1p(np.maximum(u, 0.0) / a) / math.log1p(1.0 / a)
    slope0 = 1.0 / (a * math.log1p(1.0 / a))  # L'(0)
    return np.where(u >= 0, (1.0 - w) * u + w * log_u, u * ((1.0 - w) + w * slope0))


def _tone(u: np.ndarray, w: float) -> np.ndarray:
    """Display position t in DISPLAY_T_RANGE of the normalised depth u (float64)."""
    v = _blend(u, w) if w >= 0 else 1.0 - _blend(1.0 - u, -w)
    k_lo, k_hi = _KNEES  # tanh shoulders, C1 at the joins, the identity between
    v = np.where(v < k_lo, k_lo - k_lo * np.tanh((k_lo - v) / k_lo), v)
    v = np.where(
        v > 1.0 - k_hi, 1.0 - k_hi + k_hi * np.tanh((v - 1.0 + k_hi) / k_hi), v
    )
    t0, t1 = DISPLAY_T_RANGE
    return t0 + (t1 - t0) * v


def _solve_w(u_med: float) -> float:
    """The blend w in _W_RANGE that puts u_med at DISPLAY_MEDIAN_T (bisection; t
    rises with w), or the nearer end of _W_RANGE."""
    lo, hi = _W_RANGE

    def miss(w: float) -> float:
        return float(_tone(np.array([u_med]), w)[0]) - DISPLAY_MEDIAN_T

    if not math.isfinite(u_med) or miss(lo) >= 0:
        return lo
    if miss(hi) <= 0:
        return hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if miss(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


@dataclasses.dataclass(frozen=True)
class ToneCurve:
    """The display curve of one depth map (tone_curve fits it): depth d -> position t
    on the colormap, t0 near to t1 far (DISPLAY_T_RANGE):

        u = (d - lo) / (hi - lo)       lo / hi: the 1 and 99 % depths
        v = (1 - w) u + w L(u),        L(u) = log(1 + u / a) / log(1 + 1 / a), a = 0.05
            w < 0 mirrors it: v = 1 - [(1 + w)(1 - u) - w L(1 - u)]
        t = t0 + (t1 - t0) knee(v)     tanh shoulders below v = 0.05 and above 0.75

    w in [-0.1, 0.4] (0 linear, 0.4 the most log, -0.1 the mirror) puts the median
    depth at DISPLAY_MEDIAN_T, or takes the nearer end of that range. t rises
    monotonically with d and depends on it only through u: the same colours for any
    positive scale and shift of the depth. as_dict() / ToneCurve(**d) round-trip it
    through JSON."""

    lo: float
    hi: float
    w: float

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        """Positions t (float64) of ``depth``; NaN stays NaN."""
        u = (np.asarray(depth, np.float64) - self.lo) / (self.hi - self.lo)
        return _tone(u, self.w)

    def as_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


def _fit_sample(depth: np.ndarray, max_samples: int) -> np.ndarray:
    """The finite values (float64) of a strided grid of at most ``max_samples`` pixels
    (a view of the map: no full-size copy)."""
    step = max(1, math.ceil((depth.size / max(1, max_samples)) ** (1.0 / depth.ndim)))
    sample = np.asarray(depth[(slice(None, None, step),) * depth.ndim], np.float64)
    sample = sample.reshape(-1)
    return sample[np.isfinite(sample)]


def _percentile_range(
    sample: np.ndarray, percentiles: Tuple[float, float]
) -> Tuple[float, float]:
    if sample.size == 0:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(sample, percentiles))
    if not hi > lo:
        lo, hi = float(sample.min()), float(sample.max())
    if not hi > lo:
        hi = lo + max(1.0, abs(lo))
    return lo, hi


def depth_range(
    depth: np.ndarray,
    percentiles: Tuple[float, float] = DISPLAY_PERCENTILES,
    max_samples: int = FIT_SAMPLES,
    *,
    log: bool = False,
) -> Tuple[float, float]:
    """(lo, hi): ``percentiles`` of the finite depth values (of log(depth) with
    ``log=True``) on a strided grid of at most ``max_samples`` pixels, the samples the
    tone curve is fitted on. Too few distinct values: the minimum and maximum; a flat
    map gives (v, v + max(1, |v|))."""
    sample = _fit_sample(depth, max_samples)
    if log:
        sample = np.log(np.maximum(sample, _LOG_FLOOR))
    return _percentile_range(sample, percentiles)


def tone_curve(
    depth: np.ndarray,
    percentiles: Tuple[float, float] = DISPLAY_PERCENTILES,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    *,
    max_samples: int = FIT_SAMPLES,
) -> ToneCurve:
    """The ToneCurve of ``depth`` (relative, larger = farther): lo / hi = its
    ``percentiles`` (depth_range; ``lo`` / ``hi`` given replace them), w from its
    median. ValueError when hi <= lo."""
    sample = _fit_sample(depth, max_samples)
    p_lo, p_hi = _percentile_range(sample, percentiles)
    lo = p_lo if lo is None else float(lo)
    hi = p_hi if hi is None else float(hi)
    if not hi > lo:
        raise ValueError(f"tone curve: hi ({hi}) must exceed lo ({lo})")
    median = float(np.median(sample)) if sample.size else lo
    return ToneCurve(lo=lo, hi=hi, w=_solve_w((median - lo) / (hi - lo)))


def colorize(
    depth: np.ndarray,
    cmap: str = DISPLAY_CMAP,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    *,
    log: Optional[bool] = None,
    percentiles: Tuple[float, float] = DISPLAY_PERCENTILES,
    bgr: bool = False,
    curve: Optional[ToneCurve] = None,
) -> np.ndarray:
    """Relative depth (larger = farther) -> uint8 colour image: with the default
    "Spectral", near blue, far red.

    The colours follow ``curve``, by default the tone curve fitted to ``depth``
    (tone_curve at ``percentiles``; ``lo`` / ``hi``, depths, replace the percentile
    depths). Pass the curve of a full-resolution map to colour its preview alike.
    NaN and +inf take the far end, -inf the near end. ``bgr`` writes the channels in
    OpenCV order. ``log`` is deprecated and has no effect: the curve blends linear and
    log depth itself (with ``log=True``, ``lo`` / ``hi`` are read as log depths).

    Shapes:
        depth: [H, W] -> [H, W, 3] uint8
    """
    if log is not None:
        warnings.warn(
            "colorize(log=...) is deprecated and has no effect: the tone curve blends "
            "linear and log depth itself",
            DeprecationWarning,
            stacklevel=2,
        )
        if log:
            lo = None if lo is None else math.exp(lo)
            hi = None if hi is None else math.exp(hi)
    if curve is None:
        curve = tone_curve(depth, percentiles, lo, hi)
    elif lo is not None or hi is not None:
        raise ValueError("colorize: pass either curve or lo / hi")
    # the curve tabulated on _LUT_NODES depths over u in [-2, 3], then looked up
    span = curve.hi - curve.lo
    start = curve.lo - _LUT_MARGIN * span
    u = np.linspace(-_LUT_MARGIN, 1.0 + _LUT_MARGIN, _LUT_NODES)
    table = _rgb_at(cmap, _tone(u, curve.w))
    if bgr:
        table = np.ascontiguousarray(table[:, ::-1])
    float_type = np.float64 if depth.dtype == np.float64 else np.float32
    offset = float_type(start)
    scale = float_type((_LUT_NODES - 1) / ((1.0 + 2.0 * _LUT_MARGIN) * span))
    last = _LUT_NODES - 1
    out = np.empty(depth.shape + (3,), np.uint8)
    rows = max(1, CHUNK_PIXELS // max(1, math.prod(depth.shape[1:])))
    with np.errstate(invalid="ignore", over="ignore"):
        for r in range(0, depth.shape[0], rows):
            x = depth[r : r + rows].astype(float_type)  # a copy: edited in place
            x -= offset
            x *= scale
            x += float_type(0.5)
            np.nan_to_num(x, copy=False, nan=last, posinf=last, neginf=0.0)
            np.clip(x, 0, last, out=x)
            np.take(table, x.astype(np.uint16), 0, out[r : r + rows], "clip")
    return out


# ---- 16-bit PNG: depth = lo + value / 65535 * (hi - lo), lo / hi in a tEXt chunk

DEPTH16_FORMAT = "l-dpt-depth16/1"
PNG_TEXT_KEY = "l-dpt"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def encode_depth16(depth: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """(uint16 map, lo, hi): linear from the minimum (0) to the maximum (65535),
    rounded. Finite depth only (ValueError).

    Shapes:
        depth: [H, W] -> [H, W] uint16
    """
    lo, hi = float(np.min(depth)), float(np.max(depth))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("non-finite depth: the 16-bit PNG holds finite values only")
    out = np.empty(depth.shape, np.uint16)
    scale = 65535.0 / (hi - lo) if hi > lo else 0.0
    rows = max(1, CHUNK_PIXELS // max(1, depth.shape[1]))
    for r in range(0, depth.shape[0], rows):
        x = (depth[r : r + rows].astype(np.float64) - lo) * scale
        out[r : r + rows] = np.clip(np.round(x), 0, 65535).astype(np.uint16)
    return out, lo, hi


def decode_depth16(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Inverse of encode_depth16, up to half a level ((hi - lo) / 131070)."""
    return (lo + values.astype(np.float64) / 65535.0 * (hi - lo)).astype(np.float32)


def png_text_chunk(key: str, text: str) -> bytes:
    data = key.encode("latin-1") + b"\0" + text.encode("latin-1")
    body = b"tEXt" + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def add_png_text(png: bytes, key: str, text: str) -> bytes:
    """``png`` with a tEXt chunk right after IHDR."""
    if png[:8] != _PNG_SIGNATURE or png[12:16] != b"IHDR":
        raise ValueError("not a PNG stream")
    end = 8 + 12 + struct.unpack(">I", png[8:12])[0]
    return png[:end] + png_text_chunk(key, text) + png[end:]


def png_texts(png: bytes) -> Dict[str, str]:
    """The tEXt chunks of a PNG stream before its image data (CRC checked)."""
    if png[:8] != _PNG_SIGNATURE:
        raise ValueError("not a PNG stream")
    texts, pos = {}, 8
    while pos + 12 <= len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        kind = png[pos + 4 : pos + 8]
        data = png[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", png[pos + 8 + length : pos + 12 + length])
        if kind == b"tEXt":
            if zlib.crc32(kind + data) != crc:
                raise ValueError("corrupt tEXt chunk")
            key, _, text = data.partition(b"\0")
            texts[key.decode("latin-1")] = text.decode("latin-1")
        if kind in (b"IDAT", b"IEND"):  # our tEXt chunk comes before the data
            break
        pos += 12 + length
    return texts


def depth16_png(
    depth: np.ndarray, meta: Optional[Mapping[str, Any]] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """(PNG file bytes, info): the 16-bit grayscale PNG of ``depth`` and the JSON of its
    tEXt chunk "l-dpt" (format, lo, hi, the decoding formula, then ``meta``)."""
    import cv2

    values, lo, hi = encode_depth16(depth)
    info = dict(
        format=DEPTH16_FORMAT,
        lo=lo,
        hi=hi,
        mapping="minmax",
        decode="depth = lo + value / 65535 * (hi - lo)",
        depth="relative depth, larger = farther, up to an affine map",
    )
    info.update(meta or {})
    ok, buf = cv2.imencode(".png", values, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    text = json.dumps(info, ensure_ascii=True, sort_keys=True)
    return add_png_text(buf.tobytes(), PNG_TEXT_KEY, text), info


def write_depth16(
    path: os.PathLike, depth: np.ndarray, meta: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Write depth16_png(depth, meta) to ``path``; returns its info."""
    data, info = depth16_png(depth, meta)
    Path(path).write_bytes(data)
    return info


def read_depth16(path: os.PathLike) -> Tuple[np.ndarray, Dict[str, Any]]:
    """(float32 depth, info) of a write_depth16 file."""
    import cv2

    data = Path(path).read_bytes()
    info = json.loads(png_texts(data)[PNG_TEXT_KEY])
    values = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if values is None or values.dtype != np.uint16 or values.ndim != 2:
        raise ValueError(f"{path} is not a 16-bit grayscale PNG")
    return decode_depth16(values, info["lo"], info["hi"]), info
