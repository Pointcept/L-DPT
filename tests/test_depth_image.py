"""
ldpt/utils/depth_image.py (CPU): the one depth display (the tone curve fitted to each
image, near blue, far red) and the 16-bit PNG shared by the scripts and the demo; numpy
only at import. The Space's own tests (test_hf_space.py) cover its colormap choices,
the edge values and the PNG chunks through its re-exports.
"""

import json
import math
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

from helpers import REPO

from ldpt.utils import depth_image as di


def positions(colours, cmap="Spectral", n=1024):
    """(t, distance) of each colour ([..., 3] uint8): its nearest position on the
    whole colormap (0 near ... 1 far) and its RGB distance there."""
    lut = di.colormap_lut(cmap, n).astype(np.float32)
    colours = colours.reshape(-1, 3).astype(np.float32)
    dist = ((colours[:, None, :] - lut[None]) ** 2).sum(-1)
    return dist.argmin(1) / (n - 1), np.sqrt(dist.min(1))


def distinct(rgb):
    return np.unique(rgb.reshape(-1, 3), axis=0)


def with_median(u_median, n=10001, lo=2.0, hi=50.0):
    """A [1, n] depth from lo to hi whose median sits at u = u_median of its range."""
    k = math.log(u_median) / math.log(0.5)
    u = np.linspace(0.0, 1.0, n) ** k
    return (lo + (hi - lo) * u)[None].astype(np.float32)


def test_default_display():
    rng = np.random.default_rng(0)
    depth = np.exp(rng.normal(size=(40, 30))).astype(np.float32)  # exp head: > 0
    lo, hi = di.depth_range(depth)
    ref = np.percentile(depth.astype(np.float64), [1, 99])
    assert (lo, hi) == (float(ref[0]), float(ref[1]))
    curve = di.tone_curve(depth)
    assert (curve.lo, curve.hi) == (lo, hi)
    rgb = di.colorize(depth)
    assert rgb.shape == (40, 30, 3) and rgb.dtype == np.uint8
    assert np.array_equal(di.colorize(depth, "Spectral", lo, hi), rgb)
    assert np.array_equal(di.colorize(depth, curve=curve), rgb)
    assert np.array_equal(
        di.colorize(depth, curve=di.ToneCurve(**curve.as_dict())), rgb
    )
    assert np.array_equal(di.colorize(depth, bgr=True), rgb[..., ::-1])
    # near is blue, far is red
    near = rgb[np.unravel_index(depth.argmin(), depth.shape)].astype(int)
    far = rgb[np.unravel_index(depth.argmax(), depth.shape)].astype(int)
    assert near[2] > near[0] + 50 and far[0] > far[2] + 50
    # the image is the colormap at the curve's positions (to the lookup table's step)
    exact = di._rgb_at("Spectral", curve(depth.reshape(-1))).reshape(rgb.shape)
    assert np.abs(exact.astype(int) - rgb).max() <= 1


@pytest.mark.parametrize(
    "u_median, w",
    [
        (0.0005, "max"),
        (0.02, "max"),
        (0.08, "inside"),
        (0.15, "inside"),
        (0.5, "min"),
        (0.9, "min"),
    ],
)
def test_monotone_in_depth(u_median, w):
    """For every blend (the most log for a near-heavy map, linear, the mirror for a
    far-heavy one) t rises with depth, also beyond the 1-99 % range, and the colours
    walk along the colormap from near to far; inside its range the median lands at
    DISPLAY_MEDIAN_T."""
    depth = with_median(u_median)
    curve = di.tone_curve(depth)
    w_min, w_max = di._W_RANGE
    if w == "max":
        assert curve.w == w_max
    elif w == "min":
        assert curve.w == w_min
    else:
        assert w_min < curve.w < w_max
        t_median = curve(np.median(depth))
        assert t_median == pytest.approx(di.DISPLAY_MEDIAN_T, abs=1e-6)
    span = curve.hi - curve.lo
    ramp = np.linspace(curve.lo - 4 * span, curve.hi + 4 * span, 200001)
    t = curve(ramp)
    assert (np.diff(t) >= 0).all() and t[-1] - t[0] > 0.8
    rgb = di.colorize(ramp[None, ::100].astype(np.float32), curve=curve)
    pos, dist = positions(rgb)  # 2001 colours, near to far
    assert dist.max() < 3  # each lies on the colormap ...
    assert (np.diff(pos) >= -3 / 1023).all()  # ... and none steps back
    assert pos[-1] - pos[0] > 0.8


def test_gentle_curve():
    """The curve's parameters (a = 0.05, w at most 0.4): over the 1-99 % range no
    blend is steeper than 2.6 times the linear map from t0 to t1 (a = 0.005 with w up
    to 1 reached 29 times), and the linear blend is the linear map there."""
    assert (di._LOG_ALPHA, di._W_RANGE) == (0.05, (-0.1, 0.4))
    u = np.linspace(0.0, 1.0, 200001)
    t0, t1 = di.DISPLAY_T_RANGE
    steepest = []
    for w in np.linspace(*di._W_RANGE, 11):
        gain = np.diff(di._tone(u, w)) / np.diff(u) / (t1 - t0)
        assert gain.min() > 0
        steepest.append(gain.max())
    assert max(steepest) == pytest.approx(2.58, abs=0.01)  # at w = 0.4, near u = 0.015
    assert steepest[2] == pytest.approx(1.0, abs=1e-6)  # w = 0: linear between knees


def test_scale_and_shift_invariant():
    """The colours follow relative depth only: a positive scale and a shift of the
    map give the same curve (in its units) and the same image, to one level."""
    depth = np.exp(np.random.default_rng(3).normal(size=(120, 90)))
    rgb = di.colorize(depth.astype(np.float32))
    base = di.tone_curve(depth.astype(np.float32))
    for scale, shift in [(7.0, 0.0), (0.01, 3.0), (250.0, -40.0)]:
        other = (depth * scale + shift).astype(np.float32)
        curve = di.tone_curve(other)
        assert curve.w == pytest.approx(base.w, abs=1e-5)
        assert curve.lo == pytest.approx(base.lo * scale + shift, rel=1e-5, abs=1e-5)
        assert curve.hi == pytest.approx(base.hi * scale + shift, rel=1e-5, abs=1e-5)
        diff = np.abs(di.colorize(other).astype(int) - rgb)
        assert diff.max() <= 1 and (diff > 0).mean() < 0.02


def test_colormap_ends_are_cut():
    """Spectral's deep purple-blue (t < DISPLAY_T_RANGE[0]) and deep red
    (t > DISPLAY_T_RANGE[1]) never appear, however far the outliers reach."""
    t0, t1 = di.DISPLAY_T_RANGE
    assert (t0, t1) == (0.015, 0.87)
    depth = np.exp(3 * np.random.default_rng(4).normal(size=(64, 64)))
    depth[0, :4] = [-1e30, -1e6, 1e6, 1e30]  # outliers on both sides (a shift)
    curve = di.tone_curve(depth)
    t = curve(depth)
    assert t0 <= t.min() < t0 + 0.002 and t1 - 0.002 < t.max() <= t1
    rgb = di.colorize(depth)
    pos, dist = positions(distinct(rgb))
    assert dist.max() < 3
    assert pos.min() >= t0 - 1.5 / 1023 and pos.max() <= t1 + 1.5 / 1023
    spectral = di.colormap_lut("Spectral", 2).astype(int)  # its own ends
    for end in spectral:  # (94, 79, 162) deep purple-blue, (158, 1, 66) deep red
        assert np.abs(rgb.astype(int) - end).sum(-1).min() > 8


def test_lo_hi_percentiles_and_the_deprecated_log():
    depth = np.exp(np.random.default_rng(5).normal(size=(77, 13))).astype(np.float32)
    rgb = di.colorize(depth)
    # percentiles and lo / hi (depths) move the anchors
    five = di.tone_curve(depth, (5.0, 95.0))
    ref = np.percentile(depth.astype(np.float64), [5, 95])
    assert (five.lo, five.hi) == (float(ref[0]), float(ref[1]))
    assert np.array_equal(
        di.colorize(depth, percentiles=(5.0, 95.0)), di.colorize(depth, curve=five)
    )
    anchored = di.tone_curve(depth, lo=0.5, hi=3.0)
    assert (anchored.lo, anchored.hi) == (0.5, 3.0)
    assert np.array_equal(
        di.colorize(depth, "Spectral", 0.5, 3.0), di.colorize(depth, curve=anchored)
    )
    with pytest.raises(ValueError, match="exceed"):
        di.tone_curve(depth, lo=3.0, hi=0.5)
    with pytest.raises(ValueError, match="curve or lo"):
        di.colorize(depth, lo=0.5, curve=anchored)
    # log: deprecated and without effect; with log=True lo / hi are log depths
    with pytest.warns(DeprecationWarning, match="log"):
        assert np.array_equal(di.colorize(depth, log=False), rgb)
    log_lo, log_hi = di.depth_range(depth, log=True)
    with pytest.warns(DeprecationWarning):
        via_log = di.colorize(depth, "Spectral", log_lo, log_hi, log=True)
    assert np.abs(via_log.astype(int) - rgb).max() <= 1


def test_blocks_match_whole(monkeypatch):
    depth = np.exp(np.random.default_rng(1).normal(size=(61, 47))).astype(np.float32)
    depth[3, 4], depth[5, 6], depth[7, 8] = np.nan, np.inf, -np.inf
    whole, png = di.colorize(depth), di.depth16_png(depth[10:])[0]
    monkeypatch.setattr(di, "CHUNK_PIXELS", 100)  # ~2 rows per block
    assert np.array_equal(di.colorize(depth), whole)
    assert di.depth16_png(depth[10:])[0] == png
    # the fit: every 6th row and column for at most 100 samples (2867 / 6 ** 2 < 100)
    curve = di.tone_curve(depth, max_samples=100)
    grid = depth[::6, ::6].astype(np.float64)
    grid = grid[np.isfinite(grid)]
    assert (curve.lo, curve.hi) == tuple(float(v) for v in np.percentile(grid, [1, 99]))


def test_100_megapixels_in_blocks():
    """A 100 MP depth (a broadcast view: its rows alike, no memory of its own) is
    fitted on a strided grid and coloured block by block: next to the uint8 image the
    peak allocation stays far below one float32 copy of the map (400 MB)."""
    row = np.exp(np.linspace(-3.0, 3.0, 10_000, dtype=np.float32) ** 3 / 9)
    depth = np.broadcast_to(row, (10_000, 10_000))
    tracemalloc.start()
    try:
        rgb = di.colorize(depth)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert rgb.shape == (10_000, 10_000, 3) and depth.size == 100_000_000
    assert peak - rgb.nbytes < 100e6
    reference = di.colorize(row[None], curve=di.tone_curve(depth))[0]
    for r in (0, 4_194, 9_999):  # rows of different blocks
        assert np.array_equal(rgb[r], reference)


def test_flat_and_non_finite_maps():
    flat = di.colorize(np.full((4, 5), 3.0, np.float32))
    assert flat.shape == (4, 5, 3) and (flat == flat[0, 0]).all()
    empty = di.colorize(np.full((3, 4), np.nan, np.float32))
    assert (empty == empty[0, 0]).all()


def test_depth16_round_trip(tmp_path):
    depth = np.exp(np.random.default_rng(2).normal(size=(37, 53))).astype(np.float32)
    data, info = di.depth16_png(depth, dict(model="m"))
    assert info["format"] == di.DEPTH16_FORMAT and info["model"] == "m"
    assert json.loads(di.png_texts(data)[di.PNG_TEXT_KEY]) == info
    path = tmp_path / "d.png"
    assert di.write_depth16(path, depth, dict(model="m")) == info
    assert path.read_bytes() == data
    back, read_info = di.read_depth16(path)
    step = (info["hi"] - info["lo"]) / 65535
    assert read_info == info and back.dtype == np.float32
    assert np.abs(back - depth).max() <= 0.5 * step + 1e-6 * info["hi"]


def test_imports_numpy_only():
    code = (
        "import sys; import ldpt.utils.depth_image; "
        "print(sorted(m for m in ('torch', 'cv2', 'matplotlib') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"
