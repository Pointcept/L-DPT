"""
The Hugging Face Space (demo/hf_space, CPU): resolution choices, the megapixel limit,
colorize and the 16-bit depth PNG (ldpt/utils/depth_image.py, which space_utils
re-exports), models.json and the environment overrides, the model
cache, the outputs with a stub predictor, make_examples, the inference subset of
sync_ldpt.sh (imports and runs api.load / predict without the training dependencies),
and, when gradio is installed, the Blocks app with one click handler on the stub.
"""

import gc
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import types
import weakref
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pytest
import torch

from ldpt.models.l_dpt import api
from ldpt.utils import depth_image

REPO = Path(__file__).resolve().parents[1]
SPACE = REPO / "demo" / "hf_space"
# model ids: the entry of models.json and the one two_model_registry adds
ZS = "l-dpt_large_depth_zeroshot"
MIX = "l-dpt_large_depth_mixed"
# after the repo's ldpt is imported, so a synced demo/hf_space/ldpt cannot shadow it
sys.path.append(str(SPACE))

import space_utils as su  # noqa: E402

# ---- megapixels and GPU time


@pytest.mark.parametrize(
    "hw, size", [((50, 70), 768), ((90, 40), 768), ((1000, 1333), 517), ((7, 9), None)]
)
def test_inference_hw_matches_api(hw, size):
    x = torch.zeros(1, 1, *hw)
    ref = x if size is None else api.resize_short_side(x, size)
    assert su.inference_hw(*hw, size) == tuple(ref.shape[-2:])


def test_megapixel_limits():
    su.check_megapixels(10000, 10500, 105)
    with pytest.raises(su.InputError, match=r"10001x10500 \(105.0 MP\).*up to 105 MP"):
        su.check_megapixels(10500, 10001, 105)
    su.check_inference(12000, 9000, 768, 1.0)  # 768 x 1024 = 0.79 MP
    with pytest.raises(su.InputError, match="Downscale the photo"):
        su.check_inference(12000, 9000, None, 105)
    # flip TTA above max_flip_mp
    su.check_inference(6000, 8000, None, 105, flip=True, max_flip_mp=48)  # 48 MP
    su.check_inference(6000, 8000, None, 105, flip=False, max_flip_mp=32)
    with pytest.raises(su.InputError, match="mirrored pass runs up to 32 MP"):
        su.check_inference(6000, 8000, None, 105, flip=True, max_flip_mp=32)
    # an estimate at the GPU-seconds limit is refused, not capped
    limit = 300
    su.check_inference(7000, 10000, None, 105, max_gpu_seconds=limit)  # 289.5 s
    with pytest.raises(su.InputError, match=r"an estimated 559 GPU seconds.* 300 s"):
        su.check_inference(7000, 10000, None, 105, flip=True, max_gpu_seconds=limit)
    with pytest.raises(su.InputError, match=r"an estimated 379 GPU seconds"):
        su.check_inference(8500, 10000, None, 105, max_gpu_seconds=limit)
    # ZeroGPU: the native input limit (105 MP, no flip) stays under its task limit
    est = su.estimate_gpu_seconds(10000, 10500, None, False)
    assert est < su.GPU_SECONDS_MAX
    su.check_inference(10000, 10500, None, 105, max_gpu_seconds=su.GPU_SECONDS_MAX)


def test_gpu_seconds():
    t = [su.gpu_seconds(h, w, None, False) for h, w in ((768, 1024), (3000, 4000))]
    assert su.GPU_SECONDS_BASE <= t[0] < t[1]
    assert su.gpu_seconds(3000, 4000, None, True) > t[1]
    assert t[1] == math.ceil(su.estimate_gpu_seconds(3000, 4000, None, False))
    assert su.estimate_gpu_seconds(14000, 15000, None, True) > su.GPU_SECONDS_MAX
    assert su.gpu_seconds(14000, 15000, None, True) == su.GPU_SECONDS_MAX


def test_zerogpu_slot_from_the_peak_memory():
    """A photo whose estimated peak exceeds ZEROGPU_LARGE_PEAK_GIB (38 GiB: 121 MP by
    the fit) runs on the full GPU, the others on the default half slot: every photo up
    to the default input limit (105 MP). The fit is within 0.3 GiB of the peaks of the
    timing guide (the RTX PRO 6000's; the peak does not depend on the GPU) from 8 MP
    up, above them at 8 and 102 MP."""
    assert su.PEAK_GIB == {2: 14.8, 8: 18.6, 32: 21.0, 60: 24.5, 102: 33.1}
    assert su.PEAK_GIB == su.RTX_PRO_6000.peak_gib
    for mp, gib in su.PEAK_GIB.items():
        if mp >= 8:
            assert su.estimate_peak_gib(mp) == pytest.approx(gib, abs=0.3), mp
    for mp in (2, 8, 102):  # below 8 MP and at the ends: on the safe side
        assert su.estimate_peak_gib(mp) > su.PEAK_GIB[mp], mp
    assert su.estimate_peak_gib(su.ZERO_GPU_MAX_INFER_MP) < 34 < 38
    # the half slot (47.5 GiB) less spaces' own 42 GiB bound and 4 GiB more
    assert su.ZEROGPU_LARGE_PEAK_GIB == 38.0
    assert su.ZEROGPU_LARGE_PEAK_GIB < 45097156608 / 2**30 < 101973950464 / 2**31
    assert su.estimate_peak_gib(120.7) < 38 < su.estimate_peak_gib(120.9)
    sizes = {mp: su.zerogpu_size_at(mp) for mp in (0.01, 2, 8, 32, 60, 105, 120.7)}
    assert set(sizes.values()) == {su.GPU_LARGE}
    assert su.zerogpu_size_at(120.9) == su.GPU_XLARGE
    # a request: at its inference resolution, the flip not counted (same peak)
    assert su.zerogpu_size(6000, 8000) == su.GPU_LARGE  # 48 MP
    assert su.zerogpu_size(10000, 10000) == su.GPU_LARGE
    assert su.zerogpu_size(10000, 10500) == su.GPU_LARGE  # the 105 MP limit
    assert su.zerogpu_size(11000, 11000) == su.GPU_XLARGE  # 121 MP
    assert su.zerogpu_size(11000, 11000, 768) == su.GPU_LARGE  # inference at 0.6 MP
    assert su.zerogpu_timing(su.GPU_LARGE) is su.ZEROGPU
    assert su.zerogpu_timing(su.GPU_XLARGE) is su.ZEROGPU_XLARGE


def test_zerogpu_slot_durations_and_quota():
    """The full GPU's duration: MARGIN x its own fit (the half slot's one-pass fit of its
    run, (0.59, 0.060), scaled to the 379.9 s measured at 100 MP), + BASE; the quota a
    request reserves: a large duration x 1.5 (spaces' duration_factor), an xlarge one
    twice. The default slot runs every photo up to 105 MP; a tiled photo there is
    reserved from its tiled fit (measured up to 102 MP), not from the one-pass fit."""
    xl, zg = su.ZEROGPU_XLARGE, su.ZEROGPU
    assert xl.seconds == {100: 379.9} and xl.peak_gib == {100: 40.3}
    assert xl.fit[0] / 0.59 == pytest.approx(xl.fit[1] / 0.060, rel=0.02)
    assert xl.estimate(100) == pytest.approx(379.9, rel=0.02)
    # the release code tiles a photo that large: the full GPU's one-pass fit stays above
    # the half slot's tiled fit from 79 MP up
    for mp in (79, 100, 121, 200):
        assert xl.estimate(mp, True) > zg.estimate(mp, True), mp
    assert (su.GPU_SECONDS_XLARGE_PER_MP, su.GPU_SECONDS_XLARGE_PER_MP2) == (
        0.44,
        0.046,
    )
    assert (su.GPU_SECONDS_XLARGE_PER_MP, su.GPU_SECONDS_XLARGE_PER_MP2) == (
        pytest.approx(
            (su.GPU_SECONDS_MARGIN * xl.fit[0], su.GPU_SECONDS_MARGIN * xl.fit[1]),
            rel=0.02,
        )
    )
    hw = (10000, 10000)  # 100 MP
    large = su.gpu_seconds(*hw, None, False)
    xlarge = su.gpu_seconds(*hw, None, False, su.GPU_XLARGE)
    assert large == su.gpu_seconds(*hw, None, False, su.GPU_LARGE) == 480
    assert large == math.ceil(20 + 2.1 * 100 + 0.025 * 100**2)  # the tiled fit
    assert xlarge == math.ceil(20 + 0.44 * 100 + 0.046 * 100**2) == 524
    assert su.gpu_seconds(*hw, None, True, su.GPU_XLARGE) == math.ceil(
        20 + 2 * (0.44 * 100 + 0.046 * 100**2)
    )
    assert su.estimate_gpu_seconds(*hw, 768, False, su.GPU_XLARGE) < 21
    assert su.quota_seconds(*hw, None, False) == round(1.5 * large) == 720
    assert su.quota_seconds(11000, 11000, None, False) == 2 * su.gpu_seconds(
        11000, 11000, None, False, su.GPU_XLARGE
    )
    assert su.ZEROGPU_QUOTA_FACTOR == {su.GPU_LARGE: 1.5, su.GPU_XLARGE: 2.0}
    small = su.gpu_seconds(6528, 4896, None, False)  # 32 MP: the half slot
    assert su.quota_seconds(6528, 4896, None, False) == round(1.5 * small) == 170
    assert su.quota_seconds(8944, 6708, None, False) == 354  # 60 MP
    # the rows the Space quotes: each size on its slot, measured, at the size it stands
    # for (100 MP: the 102 MP photo)
    assert su.zerogpu_timing_rows() == [
        (2, 1.2, False),
        (8, 14, False),
        (32, 70, False),
        (60, 160, False),
        (100, 370, False),
    ]
    # ZeroGPU's limit applies to the duration on the request's slot
    su.check_inference(*hw, None, 105, max_gpu_seconds=500)  # 480 s on the half slot
    with pytest.raises(su.InputError, match="an estimated 480 GPU seconds"):
        su.check_inference(*hw, None, 105, max_gpu_seconds=450)


def test_encoder_windows_floor():
    """A photo with both sides above 3072 px is expected to take at least per_window
    seconds per encoder window, and every tiled photo is reserved that much (its share
    of a full window per axis of at most 192 tokens): the photos just above 3072 px,
    where the windows overlap most, are not underestimated by the megapixel fits, on
    the RTX PRO 6000 (measured) and in the ZeroGPU reservation (MARGIN x the half slot's
    ZEROGPU_PER_RTX x the RTX PRO 6000's per window). A photo with one side a partial
    window is expected from the tiled fit (the floor is above its measurement). Larger
    photos are reserved from the tiled fit, a photo up to 3072 px from the one-pass
    fit; both cover the photos measured on the Space."""
    assert su.encoder_windows(3072, 3072) == 0  # one pass
    assert su.encoder_windows(3088, 3088) == 4  # 193 tokens: 2 x 2 windows
    assert su.encoder_windows(3264, 2448) == pytest.approx(2 * 153 / 192)
    assert su.encoder_windows(11664, 8750) == 24  # 6 x 4
    assert su.encoder_windows(6336, 9504) == 15  # 3 x 5
    assert su.encoder_windows(5136, 5136) == 9
    # the RTX PRO 6000's measured times (median of 3) against its estimate
    rtx = su.RTX_PRO_6000
    assert rtx.per_window == su.RTX_SECONDS_PER_WINDOW == 5.8
    for (h, w), seconds in {
        (3088, 3088): 21.69,
        (3088, 4112): 22.28,
        (4112, 4112): 23.31,
        (5136, 5136): 36.49,
        (2304, 3088): 7.23,
        (2448, 3264): 7.576,
    }.items():
        est = su.expected_seconds(h, w, rtx)
        assert seconds * 0.95 < est < seconds * 1.5, (h, w, est)
        fit = rtx.estimate(h * w / 1e6, True)
        if min(h, w) > 3072:
            assert fit < est  # the floor applies
        else:  # one side a partial window: the tiled fit, nearer than the floor
            floor = rtx.per_window * su.encoder_windows(h, w)
            assert est == fit and abs(est - seconds) < floor - seconds, (h, w)
    # ZeroGPU: the half slot's per window, and the reservation covers it with MARGIN
    zg = su.ZEROGPU
    assert zg.per_window == pytest.approx(1.8 * 5.8)
    assert su.estimate_gpu_seconds(3088, 3088, None, False) == pytest.approx(
        20 + 1.3 * 1.8 * 5.8 * 4
    )
    assert su.estimate_gpu_seconds(3088, 3088, None, True) == pytest.approx(
        20 + 2 * 1.3 * 1.8 * 5.8 * 4
    )
    assert su.expected_seconds(3088, 3088) == pytest.approx(1.8 * 5.8 * 4)
    # large photos: the tiled megapixel fit is the larger
    for hw in ((6000, 10000), (10000, 10000), (11664, 8750)):
        mp = hw[0] * hw[1] / 1e6
        fit = su.GPU_SECONDS_TILED_PER_MP * mp + su.GPU_SECONDS_TILED_PER_MP2 * mp * mp
        assert su.estimate_gpu_seconds(*hw, None, False) == pytest.approx(20 + fit)
    # one encoder pass (no side above 3072 px): the one-pass fit, no floor
    for hw in ((1632, 1232), (3072, 2304), (3072, 3072)):
        mp = hw[0] * hw[1] / 1e6
        fit = su.GPU_SECONDS_PER_MP * mp + su.GPU_SECONDS_PER_MP2 * mp * mp
        assert su.estimate_gpu_seconds(*hw, None, False) == pytest.approx(20 + fit)
    assert su.estimate_gpu_seconds(1632, 1232, None, False) < 22
    # the photos measured on the Space (the slower of two runs): the estimate is within
    # 5 % of each (the 8 MP photo, one side a partial window: the tiled fit), and the
    # reservation without BASE covers each
    for hw, seconds in {
        (1632, 1232): 1.2,
        (3264, 2448): 14.6,
        (6528, 4896): 70.5,
        (8944, 6708): 161.4,
        (11664, 8750): 365.1,
    }.items():
        est = su.expected_seconds(*hw)
        assert seconds * 0.95 < est < seconds * 1.05, (hw, est)
        assert su.estimate_gpu_seconds(*hw, None, False) - 20 > seconds, hw
    assert su.format_duration(su.expected_seconds(3264, 2448), True) == "14 s"
    # the progress text (quoted_seconds) near a measured size quotes the measurement,
    # as the card, the GPU-time line and the examples title (not the fit's 360 s / 71 s
    # / 1.1 s); elsewhere, or on the other encoder path, expected_seconds
    for hw, text in {
        (1632, 1232): "1.2 s",
        (3264, 2448): "14 s",
        (6528, 4896): "70 s",
        (8944, 6708): "160 s",
        (11664, 8750): "370 s",
    }.items():
        seconds, est = su.quoted_seconds(*hw)
        assert not est and su.format_duration(seconds, True) == text, hw
    assert su.format_duration(su.expected_seconds(11664, 8750), True) == "360 s"
    for hw in ((3072, 2600), (8944, 8944)):  # 7.99 MP in one pass; 80 MP
        assert su.quoted_seconds(*hw) == (su.expected_seconds(*hw), True), hw
    assert su.quoted_seconds(11664, 8750, rtx) == (205.12, False)


def test_timing_guide():
    """The GPU-time model and every time the UI quotes come from the timing guide; each
    fit matches its measurements, so a new profiling run that changes the
    measurements also has to refit."""
    for gpu in (su.ZEROGPU, su.RTX_PRO_6000, su.ZEROGPU_XLARGE):
        assert gpu.seconds
        for mp, seconds in gpu.seconds.items():
            assert gpu.estimate(mp) == pytest.approx(seconds, rel=0.1), (gpu, mp)
    zg = su.ZEROGPU
    a, b = zg.fit
    assert (su.GPU_SECONDS_PER_MP, su.GPU_SECONDS_PER_MP2) == pytest.approx(
        (su.GPU_SECONDS_MARGIN * a, su.GPU_SECONDS_MARGIN * b), rel=0.02
    )
    assert (su.GPU_SECONDS_PER_MP, su.GPU_SECONDS_PER_MP2) == (0.56, 0.091)
    a, b = zg.tiled_fit
    assert (su.GPU_SECONDS_TILED_PER_MP, su.GPU_SECONDS_TILED_PER_MP2) == (
        pytest.approx((su.GPU_SECONDS_MARGIN * a, su.GPU_SECONDS_MARGIN * b), rel=0.02)
    )
    assert (su.GPU_SECONDS_TILED_PER_MP, su.GPU_SECONDS_TILED_PER_MP2) == (2.1, 0.025)
    assert su.expected_seconds(3000, 2000) == zg.estimate(6)  # one pass
    # the RTX PRO 6000's fits: one encoder pass up to a 3072 px side, else tiled
    rtx = su.RTX_PRO_6000
    assert rtx.fit == (0.24, 0.039) and rtx.tiled_fit == (0.92, 0.012)
    assert su.expected_seconds(4000, 2000, rtx) == rtx.estimate(8, True)
    # the half ZeroGPU slot, measured on the Space with the encoder tiled from 8 MP up
    # (the times to two significant digits): ZEROGPU_PER_RTX times the RTX PRO 6000 on
    # the same photos, the same peaks; its one-pass fit (photos up to 3072 px, one
    # measurement) is the RTX PRO 6000's times ZEROGPU_PER_RTX
    assert zg.seconds == {2: 1.2, 8: 14, 32: 70, 60: 160, 102: 370}
    assert zg.peak_gib == {2: 14.8, 8: 18.6, 32: 21.0, 60: 25.2, 102: 33.1}
    assert zg.tiled_fit == (1.6, 0.019)
    assert su.ZEROGPU_PER_RTX == 1.8
    for mp in (2, 8, 32, 102):  # the 60 MP photos differ (3 x 4 and 5 x 3 windows)
        ratio = zg.seconds[mp] / rtx.seconds[mp]
        assert ratio == pytest.approx(su.ZEROGPU_PER_RTX, rel=0.06), (mp, ratio)
        assert zg.peak_gib[mp] == rtx.peak_gib[mp], mp
    assert zg.fit == tuple(float(f"{1.8 * c:.2g}") for c in rtx.fit) == (0.43, 0.070)
    assert su.expected_seconds(6528, 4896) == zg.estimate(6528 * 4896 / 1e6, True)
    assert su.expected_seconds(3072, 2000, rtx) == rtx.estimate(6.144, False)
    assert su.encoder_tiled(3073, 10) and not su.encoder_tiled(3072, 3072)
    assert rtx.estimate(7) == rtx.estimate(7, False)  # below TILED_MP
    assert rtx.estimate(7.1) == rtx.estimate(7.1, True)
    assert su.TILED_MP == pytest.approx(3072 * 2304 / 1e6)
    from ldpt.models.l_dpt.api import ENCODER_TILES

    assert su.ENCODER_TILE_PX == 16 * ENCODER_TILES.tile
    # rows: the measurement within 5 % (102 MP for 100 MP), else the fit
    assert su.timing_rows(su.ZEROGPU) == [
        (2, 1.2, False),
        (8, 14, False),
        (32, 70, False),
        (60, 160, False),
        (102, 370, False),
    ]
    assert su.timing_rows(su.RTX_PRO_6000)[-1] == (102, 205.12, False)
    assert su.measured_at({102: 1.0}, 90) is None
    # seconds only: below 10 s two significant digits (as the docs state them), else
    # to the second, an estimate from 100 s up to two significant digits
    seconds = (0.4, 0.638, 8.0, 8.3, 9.96, 79.2, 425.67, 748, 2400)
    assert [su.format_duration(s) for s in seconds] == [
        "0.40 s",
        "0.64 s",
        "8.0 s",
        "8.3 s",
        "10 s",
        "79 s",
        "426 s",
        "748 s",
        "2400 s",
    ]
    assert [su.format_duration(s, estimate=True) for s in (8.0, 87, 748, 992)] == [
        "8.0 s",
        "87 s",
        "750 s",
        "990 s",
    ]


def _seconds_stated(text: str) -> Dict[float, float]:
    """{megapixels: seconds} of each "<t> s at <mp> MP"."""
    return {
        float(mp): float(t) for t, mp in re.findall(r"([\d.]+) s at ([\d.]+) MP", text)
    }


# the number style of the docs: times in seconds, one value, no hedge
HEDGES = re.compile(
    r"\babout\b|\bapprox|~|≈|\bmin\b|\bminutes?\b|\d+ ?- ?\d+ (?:s|GiB)\b"
)


def _paragraph(text: str, start: str) -> str:
    i = text.index(start)
    return text[i : text.find("\n\n", i)].replace("\n", " ")


def _section(text: str, heading: str) -> str:
    """The text of a Markdown section, from its heading to the next of its level."""
    i = text.index(heading)
    j = text.find("\n" + heading.split(" ")[0] + " ", i + len(heading))
    return text[i : j if j >= 0 else len(text)]


# a row of the time and memory table: | 2 MP | 15 GiB | 0.64 s |
TIME_ROW = re.compile(r"^\| (\d+) MP \| ([\d.]+) GiB \|([^|]*)\|$", re.M)
# a row of the attention order table: | 2 MP | 0.64 s | 0.69 s |
ORDER_ROW = re.compile(r"^\| (\d+) MP((?: \| [^|]+ s){2}) \|$", re.M)
# the 100 MP row's note: the megapixels the guide measured that row at
ROW_100_NOTE = "The 100 MP row is a {rtx:g} MP photo"


def _time_table(text: str):
    """{row MP: (peak GiB, RTX PRO 6000 cell)}."""
    rows = {
        int(mp): (float(g), *(c.strip() for c in cells))
        for mp, g, *cells in TIME_ROW.findall(text)
    }
    assert list(rows) == [2, 8, 32, 60, 100], rows
    return rows


def _cell_seconds(cell: str) -> float:
    return float(re.fullmatch(r"([\d.]+) s", cell.strip()).group(1))


def test_timing_guide_in_docs():
    """README.md and docs/inference.md state the RTX PRO 6000's times and the peaks of
    the timing guide (space_utils), the Space card the ZeroGPU times, in seconds and as
    one value each: after a new profiling run the guide changes first, and this test
    names every doc that still states the old numbers."""
    readme = (REPO / "README.md").read_text()
    inference = (REPO / "docs" / "inference.md").read_text()
    timing = _section(inference, "## Time and memory")
    card_text = (SPACE / "README.md").read_text()
    card = " ".join(
        _paragraph(card_text, start)
        for start in ("**GPU time** on this Space", "**Quota.**")
    )
    rtx = su.RTX_PRO_6000
    note = ROW_100_NOTE.format(rtx=su.measured_mp(rtx.seconds, 100))
    tables = {
        "README.md": _time_table(readme),
        "docs/inference.md": _time_table(timing),
    }
    for where, rows in tables.items():
        for mp, (peak, rtx_cell) in rows.items():
            # the RTX PRO 6000's peak (the same on every GPU; README: whole GiB)
            want = su.measured_at(su.PEAK_GIB, mp)
            want = round(want) if where == "README.md" else want
            assert peak == want, (where, mp, peak)
            assert rtx_cell == su.format_duration(su.measured_at(rtx.seconds, mp))
        text = readme if where == "README.md" else timing
        assert note in " ".join(text.split()), (where, note)  # across line breaks
    readme_rows, docs_rows = tables["README.md"], tables["docs/inference.md"]
    for mp, (peak, *cells) in readme_rows.items():  # README: whole GiB
        assert peak == round(docs_rows[mp][0]) and cells == list(docs_rows[mp][1:]), mp
    # every measurement of the RTX PRO 6000 from 2 MP up has its row
    assert {su.measured_mp(rtx.seconds, mp) for mp in readme_rows} >= set(rtx.seconds)
    # the attention orders on the RTX PRO 6000: the one api.predict picks (the time
    # table's column), then cuDNN attention first, which is slower
    order = {
        int(mp): re.findall(r"([\d.]+) s", cells)
        for mp, cells in ORDER_ROW.findall(timing)
    }
    assert list(order) == list(docs_rows), order
    for mp, (rtx_torch, rtx_cudnn) in order.items():
        assert f"{rtx_torch} s" == docs_rows[mp][1]
        assert float(rtx_torch) < float(rtx_cudnn)
    assert not HEDGES.search(timing), timing
    # the Quick start's memory sentence: the peak allocation, in whole GiB
    peaks = {mp: round(su.measured_at(su.PEAK_GIB, mp)) for mp in (2, 32, 100)}
    assert peaks == {2: 15, 32: 21, 100: 33}
    assert (
        f"the peak allocation is {peaks[2]} GiB at 2 MP, {peaks[32]} GiB at 32 MP and "
        f"{peaks[100]} GiB at 100 MP" in " ".join(readme.split())
    )

    # the Space card: the times of each size on its slot (an estimate as such), and
    # what it reserves (the measured photos at 32 and 60 MP; to two significant digits)
    for mp, seconds, estimated in su.zerogpu_timing_rows():
        stated = f"{su.format_duration(seconds, estimated)} at {mp:g} MP"
        assert stated + (" (estimate)" if estimated else "") in card, (mp, card)
    for mp, hw in ((32, (6528, 4896)), (60, (8944, 6708)), (100, (10000, 10000))):
        reserved = su.format_duration(su.quota_seconds(*hw, None, False), True)
        assert f"{reserved} for a {mp} MP photo" in card, (mp, reserved)

    def limit(a, b):  # the largest whole MP of an a:b photo reserving at most 300 s
        def quota(mp):
            short = round(math.sqrt(mp * 1e6 * b / a))
            return su.quota_seconds(round(short * a / b), short, None, False)

        return max(mp for mp in range(1, 106) if quota(mp) <= 300)

    # the encoder windows follow the shape (16:9 photos reach 5 x 3 windows above 47 MP,
    # square ones 4 x 4 above 51 MP): the limit of the common photo shapes
    free = min(limit(*shape) for shape in ((1, 1), (4, 3), (3, 2), (16, 9)))
    # every photo of at most free MP with a side ratio up to 3:1 fits (at a short side,
    # the reservation grows with the long side)
    for short in range(1, math.isqrt(free * 10**6) + 1):
        long = min(free * 10**6 // short, 3 * short)
        assert su.quota_seconds(long, short, None, False) <= 300, (long, short)
    assert f"a free account has 300 s, enough for photos up to {free} MP" in card
    assert not HEDGES.search(card), card


def test_read_rgb(tmp_path):
    bgr = np.zeros((90, 120, 3), np.uint8)
    bgr[...] = (10, 20, 30)
    path = tmp_path / "x.png"
    cv2.imwrite(str(path), bgr)
    rgb = su.read_rgb(path, 105)
    assert rgb.shape == (90, 120, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (30, 20, 10)
    with pytest.raises(su.InputError, match="up to 0.01 MP"):
        su.read_rgb(path, 0.01)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(su.InputError, match="Cannot read"):
        su.read_rgb(bad, 105)


# EXIF orientation -> the stored pixels as displayed (PIL's exif_transpose agrees)
ORIENTATIONS = {
    2: lambda a: a[:, ::-1],  # mirrored left-right
    5: lambda a: a.transpose(1, 0, 2),  # mirrored along the main diagonal
    6: lambda a: np.rot90(a, k=-1),  # rotate 90 degrees clockwise
    8: lambda a: np.rot90(a, k=1),  # rotate 90 degrees counter-clockwise
}


@pytest.mark.parametrize("orientation", sorted(ORIENTATIONS))
def test_read_rgb_exif_orientation(tmp_path, orientation):
    """Four distinct quadrants (aligned to the 16 px JPEG blocks, so they decode almost
    exactly): a wrong rotation direction or a missing mirror moves them."""
    Image = pytest.importorskip("PIL.Image")
    ImageOps = pytest.importorskip("PIL.ImageOps")
    stored = np.empty((32, 64, 3), np.uint8)
    stored[:16, :32], stored[:16, 32:] = (220, 20, 20), (20, 220, 20)
    stored[16:, :32], stored[16:, 32:] = (20, 20, 220), (220, 220, 20)
    exif = Image.Exif()
    exif[274] = orientation
    path = tmp_path / "rot.jpg"
    Image.fromarray(stored).save(path, exif=exif, quality=95, subsampling=0)
    expected = ORIENTATIONS[orientation](stored)
    with Image.open(path) as im:
        shown = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    got = su.read_rgb(path, 105)
    assert got.shape == shown.shape == expected.shape
    h, w = expected.shape[:2]
    for y in (h // 4, 3 * h // 4):
        for x in (w // 4, 3 * w // 4):
            for image in (got, shown):
                diff = np.abs(image[y, x].astype(int) - expected[y, x].astype(int))
                assert diff.max() < 16, (orientation, y, x, image[y, x])


# ---- colorize


@pytest.mark.parametrize(
    "cmap, near, far",
    [
        ("Spectral", "blue", "red"),
        ("Turbo", "blue", "red"),
        ("Inferno", "bright", "dark"),
        ("Gray", "bright", "dark"),
    ],
)
def test_colormap_choices(cmap, near, far):
    """The Space's colormaps (the package's, so the scripts write the same images):
    Spectral (the default) and Turbo near blue, far red; Inferno and Gray near bright,
    far dark; each on the DISPLAY_T_RANGE part of its colormap, in RGB or BGR."""
    assert su.colorize is depth_image.colorize
    assert su.tone_curve is depth_image.tone_curve
    assert su.write_depth16 is depth_image.write_depth16
    assert list(su.COLORMAPS) == ["Spectral", "Turbo", "Inferno", "Gray"]
    assert su.DISPLAY_CMAP == "Spectral"
    depth = np.tile(np.linspace(1.0, 9.0, 256, dtype=np.float32), (3, 1))
    curve = su.tone_curve(depth)
    rgb = su.colorize(depth, cmap)
    assert rgb.shape == (3, 256, 3) and rgb.dtype == np.uint8
    assert (su.colorize(depth, cmap, bgr=True) == rgb[..., ::-1]).all()
    # the colormap at the curve's positions, inside the cut, near (t = 0) -> far
    t = curve(depth[0])
    t0, t1 = depth_image.DISPLAY_T_RANGE
    assert t0 <= t[0] < t[-1] <= t1
    exact = depth_image._rgb_at(cmap, t)
    assert np.abs(exact.astype(int) - rgb[0]).max() <= 1
    lut = su.colormap_lut(cmap, 1024).astype(int)  # the whole colormap, t 0 -> 1
    assert np.abs(rgb[0, 0] - lut[round(t[0] * 1023)]).max() <= 2
    a, b = rgb[0, 0].astype(int), rgb[0, -1].astype(int)
    if near == "blue":
        assert a[2] > a[0] + 30 and b[0] > b[2] + 30
    else:
        assert a.sum() > b.sum() + 300
    with pytest.raises(ValueError, match="colormap"):
        su.colorize(depth, "jet")


def test_colorize_chunks_and_edge_values(monkeypatch):
    """Blocks of CHUNK_PIXELS give the whole image; NaN and +inf take the far end,
    -inf the near end; a preview shares the full map's curve."""
    depth = np.random.default_rng(0).random((61, 47)).astype(np.float32)
    depth[3, 4], depth[5, 6], depth[7, 8] = np.nan, np.inf, -np.inf
    whole = su.colorize(depth, "Turbo")
    monkeypatch.setattr(depth_image, "CHUNK_PIXELS", 100)  # ~2 rows per chunk
    assert (su.colorize(depth, "Turbo") == whole).all()
    curve = su.tone_curve(depth)
    ends = su.colorize(np.array([[-1e30, 1e30]], np.float32), "Turbo", curve=curve)
    assert (whole[3, 4] == ends[0, 1]).all() and (whole[5, 6] == ends[0, 1]).all()
    assert (whole[7, 8] == ends[0, 0]).all()
    small = su.colorize(depth[::2, ::2], "Turbo", curve=curve)
    assert (small == whole[::2, ::2]).all()
    flat = su.colorize(np.full((4, 5), 3.0, np.float32), "Spectral")
    assert flat.shape == (4, 5, 3)


# ---- 16-bit PNG


def test_depth16_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    depth = np.exp(rng.normal(size=(37, 53))).astype(np.float32)  # exp head: > 0
    path = tmp_path / "d.png"
    meta = su.write_depth16(path, depth, dict(model="m"))
    back, info = su.read_depth16(path)
    assert info == meta and info["model"] == "m" and info["mapping"] == "minmax"
    assert info["lo"] == float(depth.min()) and info["hi"] == float(depth.max())
    step = (info["hi"] - info["lo"]) / 65535
    assert back.shape == depth.shape and back.dtype == np.float32
    assert np.abs(back - depth).max() <= 0.5 * step + 1e-6 * info["hi"]
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert raw.dtype == np.uint16 and raw.min() == 0 and raw.max() == 65535
    Image = pytest.importorskip("PIL.Image")
    with Image.open(path) as im:  # a standard tEXt chunk
        assert json.loads(im.text[su.PNG_TEXT_KEY]) == meta


def test_depth16_edge_cases(tmp_path):
    depth = np.array([[1.0, 2.0], [1.5, 2.0]], np.float32)
    values, lo, hi = su.encode_depth16(depth)
    assert (lo, hi) == (1.0, 2.0) and values.tolist() == [[0, 65535], [32768, 65535]]
    for bad in (np.inf, np.nan, -np.inf):  # policy: finite depth only
        with pytest.raises(ValueError, match="non-finite"):
            su.encode_depth16(np.array([[1.0, bad]], np.float32))
    values, lo, hi = su.encode_depth16(np.full((2, 3), 0.5, np.float32))
    assert (values == 0).all() and lo == hi == 0.5
    with pytest.raises(ValueError, match="not a PNG"):
        su.add_png_text(b"GIF89a" + bytes(40), "k", "v")
    path = tmp_path / "d.png"
    su.write_depth16(path, np.ones((3, 3), np.float32))
    data = bytearray(path.read_bytes())
    data[45] ^= 0xFF  # inside the tEXt chunk after IHDR
    with pytest.raises(ValueError, match="corrupt"):
        su.png_texts(bytes(data))


# ---- models.json and the environment


def test_shipped_registry():
    """The Space serves one model, the zero-shot release, from the release repo pinned
    as presets.WEIGHTS pins it (repo, commit, file, sha256), with the release
    architecture (presets.RELEASE_L) in "expect"."""
    from ldpt.models.l_dpt import presets

    reg = su.Registry.load()
    assert [m.id for m in reg.models] == [ZS]
    m = reg.default
    assert m.id == ZS and m.recommended
    # a local file: the variable api.load reads (then LDPT_MODEL_<ID>)
    assert m.env == "LDPT_CKPT_LARGE_DEPTH_ZEROSHOT" == presets.WEIGHTS[ZS].env
    assert m.legacy_env == "LDPT_MODEL_L_DPT_LARGE_DEPTH_ZEROSHOT"
    assert su.ckpt_env(MIX) == presets.WEIGHTS[MIX].env
    assert not m.placeholder and m.path is None
    w = presets.WEIGHTS[ZS]
    assert (m.repo, m.revision, m.filename, m.sha256) == (
        w.repo,
        w.revision,
        w.file,
        w.sha256,
    )
    assert m.expect == {k: presets.RELEASE_L[k] for k in su.EXPECT_DEFAULTS}
    assert reg.choices() == [("L-DPT Large (zero-shot) · recommended", ZS)]
    with pytest.raises(su.InputError, match="unknown model"):
        reg.get("nope")


def two_model_registry(folder):
    """The shipped registry + an entry MIX pinned like presets.WEIGHTS (the service
    and the multi-model UI are tested with two models)."""
    import json as _json
    from ldpt.models.l_dpt import presets

    doc = _json.loads((SPACE / "models.json").read_text())
    w = presets.WEIGHTS[MIX]
    doc["models"].append(
        dict(
            id=MIX,
            label="L-DPT Large (mixed)",
            description="Adds the NYUv2, KITTI and DIODE training sets.",
            recommended=False,
            source=dict(
                repo=w.repo, filename=w.file, revision=w.revision, sha256=w.sha256
            ),
            expect=doc["models"][0]["expect"],
        )
    )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "models.json").write_text(_json.dumps(doc))
    return su.Registry.load(folder / "models.json")


def _entry(**kw):
    e = dict(id="a", label="A", description="d", source=dict(path="a.pth"))
    e.update(kw)
    return e


@pytest.mark.parametrize(
    "doc, match",
    [
        ([], "expected"),
        (dict(models=[]), "non-empty"),
        (dict(models=[_entry()], extra=1), "expected"),
        (dict(models=[_entry(id="A b")]), "must match"),
        (dict(models=[_entry(), _entry(label="B")]), "id 'a' already used"),
        (dict(models=[_entry(), _entry(id="b")]), "label 'A' already used"),
        (dict(models=[_entry(id="x-1"), _entry(id="x_1", label="B")]), "env"),
        (dict(models=[_entry(colour=1)]), "unknown keys"),
        (dict(models=[{"id": "a", "label": "A", "source": {}}]), "missing"),
        (
            dict(models=[_entry(source=dict(path="a", repo="r", filename="f"))]),
            "either",
        ),
        (dict(models=[_entry(source=dict(repo="r"))]), "either"),
        (dict(models=[_entry(source=dict(path=""))]), "non-empty"),
        (dict(models=[_entry(source=dict(path="a", sha256="0" * 64))]), "either"),
        (
            dict(models=[_entry(source=dict(repo="r", filename="f", sha256="AB"))]),
            "64 lowercase hex",
        ),
        (dict(models=[_entry(recommended="yes")]), "true or false"),
        (dict(models=[_entry(label=" ")]), "label must"),
        (dict(models=[_entry(expect=dict(colour=1))]), "expect must be"),
        (dict(models=[_entry(expect=["head"])]), "expect must be"),
        (dict(models=[_entry(expect=dict(head=["exp"]))]), "JSON scalars"),
    ],
)
def test_registry_validation(doc, match):
    with pytest.raises(ValueError, match=match):
        su.parse_registry(doc)


def test_registry_comments_and_defaults():
    doc = {
        "_note": "x",
        "models": [
            _entry(_why="comment"),
            _entry(
                id="b",
                label="B",
                recommended=True,
                source=dict(repo="u/r", filename="f"),
            ),
        ],
    }
    a, b = su.parse_registry(doc)
    assert b.revision == "main" and not a.recommended and not b.placeholder
    assert a.sha256 is None and b.sha256 is None
    assert su.Registry([a, b], Path(".")).default is b
    assert su.Registry([a], Path(".")).default is a
    assert a.expect == {} and b.expect == {}


def test_expect_check(tmp_path):
    """models.json "expect" against the release config: a missing key is the model's
    default (EXPECT_DEFAULTS, pinned against LatentDPT here)."""
    import inspect

    from ldpt.models.l_dpt.l_dpt_v1m1_base import LatentDPT

    params = inspect.signature(LatentDPT.__init__).parameters
    assert {k: params[k].default for k in su.EXPECT_DEFAULTS} == su.EXPECT_DEFAULTS
    (entry,) = su.parse_registry(
        dict(models=[_entry(expect=dict(head="exp", dec_qk_bound=1000))])
    )
    su.check_expect(entry, dict(head="exp", dec_qk_bound=1000.0), Path("r.pth"))
    with pytest.raises(su.SetupError, match=r"entry 'a': head='sigmoid' \(models"):
        su.check_expect(entry, dict(dec_qk_bound=1000), Path("r.pth"))
    with pytest.raises(su.SetupError, match="dec_qk_bound=None"):
        su.check_expect(entry, dict(head="exp"), Path("r.pth"))
    release = tmp_path / "release.pth"  # the default reader: the release file's config
    torch.save(dict(format="l-dpt-release/1", config=dict(head="exp")), release)
    assert su._release_config(release) == dict(head="exp")


def test_resolve_release(tmp_path):
    (tmp_path / "local.pth").write_bytes(b"x")
    other = tmp_path / "other.pth"
    other.write_bytes(b"y")
    local, hub, todo, pinned = su.parse_registry(
        dict(
            models=[
                _entry(id="loc", label="L", source=dict(path="local.pth")),
                _entry(
                    id="hub",
                    label="H",
                    source=dict(repo="u/r", filename="f.pth", revision="abc"),
                ),
                _entry(id="t", label="T", source=dict(repo="TODO/r", filename="f")),
                _entry(
                    id="pin",
                    label="P",
                    source=dict(repo="u/r", filename="p.pth", sha256="a" * 64),
                ),
            ]
        )
    )
    calls = []

    def download(**kw):
        calls.append(kw)
        return str(other)

    base = tmp_path
    assert su.resolve_release(local, base, {}, download) == tmp_path / "local.pth"
    assert su.resolve_release(hub, base, {"HF_TOKEN": "tok"}, download) == other
    assert calls == [dict(repo_id="u/r", filename="f.pth", revision="abc", token="tok")]
    with pytest.raises(su.SetupError, match="set LDPT_CKPT_T to its release file"):
        su.resolve_release(todo, base, {}, download)
    # $LDPT_CKPT_<NAME> wins over every source, then $LDPT_MODEL_<ID>
    env = {"LDPT_CKPT_T": str(other), "LDPT_MODEL_LOC": str(other)}
    assert su.resolve_release(todo, base, env, download) == other
    assert su.resolve_release(local, base, env, download) == other
    env["LDPT_CKPT_LOC"] = str(tmp_path / "local.pth")
    assert su.resolve_release(local, base, env, download) == tmp_path / "local.pth"
    with pytest.raises(su.SetupError, match="LDPT_CKPT_HUB: .* is not a file"):
        su.resolve_release(hub, base, {"LDPT_CKPT_HUB": str(tmp_path / "no")})
    with pytest.raises(su.SetupError, match="is not a file"):
        su.resolve_release(local, tmp_path / "elsewhere", {}, download)

    # a pinned download is hashed (unless verify=False); a local override is not
    hashes = {other: "a" * 64}
    assert su.resolve_release(pinned, base, {}, download, sha256=hashes.get) == other
    hashes[other] = "b" * 64
    with pytest.raises(su.SetupError, match="not the release file models.json pins"):
        su.resolve_release(pinned, base, {}, download, sha256=hashes.get)
    assert su.resolve_release(pinned, base, {}, download, False, hashes.get) == other
    env = {"LDPT_CKPT_PIN": str(tmp_path / "local.pth")}
    assert su.resolve_release(pinned, base, env, download, sha256=hashes.get) == (
        tmp_path / "local.pth"
    )

    def missing(**kw):
        raise OSError("404 Client Error: Repository Not Found")

    with pytest.raises(su.SetupError, match=r"H \(u/r/f.pth at abc\): download failed"):
        su.resolve_release(hub, base, {}, missing)


def test_resolve_dinov3(tmp_path):
    f = tmp_path / "dinov3.pth"
    f.write_bytes(b"x")
    folder = tmp_path / "dinov3-hf"
    folder.mkdir()
    calls, snaps = [], []

    def download(**kw):
        calls.append(kw)
        return str(f)

    def snapshot(**kw):
        snaps.append(kw)
        return str(folder)

    assert su.resolve_dinov3({"LDPT_DINOV3": str(f)}, download, snapshot) == f
    assert su.resolve_dinov3({"LDPT_DINOV3": str(folder)}, download, snapshot) == folder
    env = dict(LDPT_DINOV3_REPO="me/dino", LDPT_DINOV3_FILE="d.pth", HF_TOKEN="tok")
    assert su.resolve_dinov3(env, download, snapshot) == f
    assert calls == [
        dict(repo_id="me/dino", filename="d.pth", revision=None, token="tok")
    ]
    # default: Meta's gated transformers-format repo at the pinned commit
    assert su.resolve_dinov3({"HF_TOKEN": "tok"}, download, snapshot) == folder
    assert snaps == [
        dict(
            repo_id="facebook/dinov3-vit7b16-pretrain-lvd1689m",
            revision="b80367753773648a6793235ab9c65cdbb029506f",
            allow_patterns=su.DINOV3_HF_FILES,
            token="tok",
        )
    ]
    env = dict(LDPT_DINOV3_HF_REPO="me/dino-hf")  # another repo: no pinned commit
    assert su.resolve_dinov3(env, download, snapshot) == folder
    assert snaps[-1]["repo_id"] == "me/dino-hf" and snaps[-1]["revision"] is None
    assert snaps[-1]["token"] is None

    def gated(**kw):
        raise OSError("401 Client Error: Cannot access gated repo")

    with pytest.raises(su.SetupError, match=r"request access at .*\(no HF_TOKEN"):
        su.resolve_dinov3({}, download, gated)
    with pytest.raises(su.SetupError, match="not redistributable"):
        su.resolve_dinov3({"LDPT_DINOV3_HF_REPO": "none"}, download, snapshot)
    with pytest.raises(su.SetupError, match="set both"):
        su.resolve_dinov3({"LDPT_DINOV3_REPO": "me/dino"}, download, snapshot)
    with pytest.raises(su.SetupError, match="LDPT_DINOV3: .* is not a file"):
        su.resolve_dinov3({"LDPT_DINOV3": str(tmp_path / "missing.pth")}, download)


def test_settings_from_env():
    s = su.Settings.from_env({})
    assert (s.max_mp, s.precision, s.chunk_tokens, s.verify) == (
        105,
        None,  # each model's own (the release files: bf16)
        32768,
        True,
    )
    assert s.selection_impl == "stream" and s.preload  # app.py --lazy: no
    s = su.Settings.from_env(
        dict(
            LDPT_MAX_MP="50",
            LDPT_CHUNK_TOKENS="0",
            LDPT_VERIFY="0",
            LDPT_PRELOAD="0",
            LDPT_PRECISION="fp32",
        )
    )
    assert (s.max_mp, s.chunk_tokens, s.verify, s.preload) == (50, None, False, False)
    assert s.precision == "fp32"
    s = su.Settings.from_env({})
    assert (s.max_infer_mp, s.max_flip_mp, s.max_gpu_seconds) == (105, None, None)
    # ZeroGPU (SPACES_ZERO_GPU, as spaces.config reads it): 105 MP, flip 32 MP, the
    # seconds limit, preload; the LDPT_* variables still override
    assert not su.zero_gpu_active({"SPACES_ZERO_GPU": "false"})
    s = su.Settings.from_env({"SPACES_ZERO_GPU": "true"})
    assert (s.max_infer_mp, s.max_flip_mp, s.max_gpu_seconds, s.preload) == (
        105,
        32,
        su.GPU_SECONDS_MAX,
        True,
    )
    s = su.Settings.from_env(
        dict(
            SPACES_ZERO_GPU="1",
            LDPT_MAX_INFER_MP="50",
            LDPT_MAX_FLIP_MP="none",
            LDPT_PRELOAD="0",
        )
    )
    assert (s.max_infer_mp, s.max_flip_mp, s.preload) == (50, None, False)


# ---- the service with a stub model


class StubModel:
    def __init__(self, encoder=None):
        self.encoder = object() if encoder is None else encoder


def stub_predict(model, rgb, size, flip):
    """Relative depth growing down the image, with a little of the red channel."""
    assert isinstance(model, StubModel)
    h, w = rgb.shape[:2]
    rows = np.linspace(1.0, 2.0, h, dtype=np.float32)[:, None]
    return rows + rgb[..., 0].astype(np.float32) / 2550.0


def stub_service(tmp_path, loads=None, configs=None, **settings):
    """The shipped models.json, one stub release file per model whose config is what
    its entry expects (``configs``: {model id: config values replacing those})."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    dinov3 = tmp_path / "dinov3.pth"
    dinov3.write_bytes(b"d")
    registry = two_model_registry(tmp_path / "registry")
    releases = {m.id: tmp_path / f"release-{m.id}.pth" for m in registry.models}
    for path in releases.values():
        path.write_bytes(b"r")
    env = {m.env: str(releases[m.id]) for m in registry.models}
    env["LDPT_DINOV3"] = str(dinov3)
    release_configs = {
        releases[m.id]: dict(m.expect, **(configs or {}).get(m.id, {}))
        for m in registry.models
    }
    loads = [] if loads is None else loads

    def loader(rel, enc, shared):
        """The first load reads the DINOv3 weights, the next ones share its encoder."""
        assert rel in release_configs
        first = not any(ref() is not None for ref in loads)
        assert (enc, shared is None) == ((dinov3, True) if first else (None, False))
        model = StubModel(shared)
        loads.append(weakref.ref(model))
        return model

    settings.setdefault("output_dir", tmp_path / "out")
    settings.setdefault("preview_side", 64)
    return su.DepthService(
        registry,
        su.Settings(**settings),
        env=env,
        loader=loader,
        predictor=stub_predict,
        download=lambda **kw: pytest.fail("no download expected"),
        snapshot=lambda **kw: pytest.fail("no download expected"),
        config_reader=lambda path: release_configs[path],
    )


def test_service_keeps_the_models_sharing_the_encoder(tmp_path):
    loads = []
    service = stub_service(tmp_path, loads)
    rgb = np.zeros((20, 30, 3), np.uint8)
    assert not service.is_loaded(ZS)
    service.ensure(ZS)
    assert service.ensure(ZS) == 0.0 and len(loads) == 1
    depth, seconds, peak = service.predict(ZS, rgb, None, False)
    assert depth.shape == (20, 30) and seconds >= 0 and peak is None  # no GPU
    assert len(loads) == 1
    service.predict(MIX, rgb, None, False)  # loads MIX on ZS's encoder
    assert len(loads) == 2 and service.is_loaded(ZS) and service.is_loaded(MIX)
    assert loads[1]().encoder is loads[0]().encoder
    service.predict(ZS, rgb, None, False)  # both stay: nothing loads again
    assert service.ensure_all() == 0.0 and len(loads) == 2
    with pytest.raises(su.InputError, match="unknown model"):
        service.ensure("nope")
    service.release()
    gc.collect()
    assert service.models == {} and all(ref() is None for ref in loads)
    service.ensure(MIX)  # after a release the next load reads DINOv3 again
    assert len(loads) == 3


def test_service_ensure_all_loads_the_default_first(tmp_path):
    loads = []
    service = stub_service(tmp_path, loads)
    service.registry.models.reverse()  # the default (recommended) is no longer first
    assert service.registry.models[0].id != service.registry.default.id == ZS
    assert service.ensure_all() >= 0.0
    assert list(service.models) == [ZS, MIX]
    assert len(loads) == 2 and service.load_errors == {}


def test_service_ensure_all_keep_going_and_freeze(tmp_path, caplog):
    service = stub_service(tmp_path)
    load = service.loader

    def flaky(release, dinov3, encoder):
        if MIX in str(release):
            raise OSError("download interrupted")
        return load(release, dinov3, encoder)

    service.loader = flaky
    with pytest.raises(OSError, match="interrupted"):
        service.ensure_all()  # without keep_going a failure raises
    assert list(service.models) == [ZS]
    with caplog.at_level("ERROR"):
        service.ensure_all(keep_going=True)
    assert list(service.models) == [ZS] and f"'{MIX}' not loaded" in caplog.text
    assert service.load_errors == {MIX: "OSError: download interrupted"}
    service.freeze()
    rgb = np.zeros((20, 30, 3), np.uint8)
    assert service.predict(ZS, rgb, None, False)[0].shape == (20, 30)
    with pytest.raises(su.SetupError, match=r"L-DPT Large \(mixed\) is not available"):
        service.predict(MIX, rgb, None, False)
    with pytest.raises(su.InputError, match="unknown model"):
        service.ensure("nope")
    # the default's failure always raises
    service = stub_service(tmp_path / "b")
    service.loader = lambda release, dinov3, encoder: (_ for _ in ()).throw(
        OSError("offline")
    )
    with pytest.raises(OSError, match="offline"):
        service.ensure_all(keep_going=True)


def test_service_setup_errors(tmp_path):
    service = stub_service(tmp_path)
    service.env = {}

    def offline(**kw):
        raise OSError("offline")

    service.download = service.snapshot = offline
    with pytest.raises(
        su.SetupError,
        match="download failed.*or set LDPT_CKPT_LARGE_DEPTH_ZEROSHOT",
    ):
        service.ensure(ZS)
    service.env = {
        "LDPT_CKPT_LARGE_DEPTH_ZEROSHOT": str(tmp_path / f"release-{ZS}.pth")
    }
    with pytest.raises(su.SetupError, match="dinov3-vit7b16.*download failed"):
        service.ensure(ZS)
    service.env = dict(service.env, LDPT_DINOV3_HF_REPO="none")
    with pytest.raises(su.SetupError, match="not redistributable"):
        service.ensure(ZS)
    assert service.models == {}


def test_service_checks_expect_before_loading(tmp_path):
    loads = []
    configs = {MIX: dict(head="sigmoid")}
    service = stub_service(tmp_path, loads, configs=configs)
    service.ensure(ZS)
    with pytest.raises(su.SetupError, match=rf"'{MIX}': head='sigmoid' \(models"):
        service.ensure(MIX)
    assert list(service.models) == [ZS] and len(loads) == 1  # nothing loaded
    configs[MIX] = dict(attn_fp32=None)
    service = stub_service(tmp_path / "b", loads, configs=configs)
    with pytest.raises(su.SetupError, match=r"attn_fp32=None \(models.json expects"):
        service.ensure(MIX)
    configs.clear()
    service = stub_service(tmp_path / "c", loads, configs=configs)
    service.ensure(MIX)
    assert service.is_loaded(MIX)


def test_service_refuses_nonfinite_depth(tmp_path):
    def nan_predict(model, rgb, size, flip):
        depth = stub_predict(model, rgb, size, flip)
        depth[1, 2], depth[3, 4] = np.nan, np.inf
        return depth

    service = stub_service(tmp_path)
    service.predictor = nan_predict
    with pytest.raises(su.InputError, match="2 non-finite depth values at 30x20"):
        service.predict(ZS, np.zeros((20, 30, 3), np.uint8), None, False)


def test_service_out_of_memory(tmp_path):
    class OutOfMemoryError(RuntimeError):
        pass

    def oom(model, rgb, size, flip):
        raise OutOfMemoryError("CUDA out of memory")

    service = stub_service(tmp_path)
    service.predictor = oom
    with pytest.raises(su.InputError, match=r"Out of GPU memory at 1365x1024"):
        service.predict(ZS, np.zeros((3000, 4000, 3), np.uint8), 1024, False)


def test_service_out_of_memory_frees_the_frames(tmp_path, monkeypatch):
    """The OOM traceback (frames holding the forward's tensors) is dropped before
    gc.collect + empty_cache, and the user-facing error carries no context."""

    class OutOfMemoryError(RuntimeError):
        pass

    class Activations:
        pass

    refs, alive_at_empty_cache = [], []

    def oom(model, rgb, size, flip):
        activations = Activations()  # what a failed forward still references
        refs.append(weakref.ref(activations))
        raise OutOfMemoryError("CUDA out of memory")

    service = stub_service(tmp_path)
    service.ensure(ZS)
    service.predictor = oom
    cuda = types.SimpleNamespace(
        synchronize=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
        empty_cache=lambda: alive_at_empty_cache.append(refs[-1]() is not None),
    )
    monkeypatch.setattr(su, "_cuda_ready", lambda: True)
    monkeypatch.setattr(su, "_torch", lambda: types.SimpleNamespace(cuda=cuda))
    with pytest.raises(su.InputError, match="Out of GPU memory") as exc:
        service.predict(ZS, np.zeros((30, 40, 3), np.uint8), None, False)
    assert alive_at_empty_cache == [False]
    assert exc.value.__context__ is None and exc.value.__cause__ is None


# what torch raised for a 100 MP photo on ZeroGPU's half slot (its allocator could not
# allocate, and its out-of-memory report could not query NVML in the container)
ALLOCATOR_ASSERT = (
    'NVML_SUCCESS == r INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/'
    'CUDACachingAllocator.cpp":1131, please report a bug to PyTorch.'
)


def test_service_allocator_assert_is_out_of_memory(tmp_path, monkeypatch):
    """The allocator's NVML assert reads as running out of GPU memory, like
    torch.cuda.OutOfMemoryError (also when empty_cache fails after it); any other
    RuntimeError passes through."""
    assert su.is_out_of_memory(RuntimeError(ALLOCATOR_ASSERT))
    assert su.is_out_of_memory(RuntimeError("... CUDACachingAllocator.cpp ..."))
    assert su.is_out_of_memory(type("OutOfMemoryError", (RuntimeError,), {})("x"))
    assert not su.is_out_of_memory(RuntimeError("shape mismatch"))
    assert not su.is_out_of_memory(ValueError(ALLOCATOR_ASSERT))

    def fails(message):
        def predictor(model, rgb, size, flip):
            raise RuntimeError(message)

        return predictor

    service = stub_service(tmp_path)
    service.predictor = fails(ALLOCATOR_ASSERT)
    with pytest.raises(su.InputError, match=r"Out of GPU memory at 4000x3000 \(12.0"):
        service.predict(ZS, np.zeros((3000, 4000, 3), np.uint8), None, False)
    service.predictor = fails("shape mismatch")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        service.predict(ZS, np.zeros((30, 40, 3), np.uint8), None, False)

    def empty_cache():
        raise RuntimeError(ALLOCATOR_ASSERT)

    cuda = types.SimpleNamespace(
        synchronize=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
        empty_cache=empty_cache,
    )
    monkeypatch.setattr(su, "_cuda_ready", lambda: True)
    monkeypatch.setattr(su, "_torch", lambda: types.SimpleNamespace(cuda=cuda))
    service.predictor = fails(ALLOCATOR_ASSERT)
    with pytest.raises(su.InputError, match="Out of GPU memory at 40x30"):
        service.predict(ZS, np.zeros((30, 40, 3), np.uint8), None, False)


def test_user_error_unwraps_a_zerogpu_task_exception():
    """spaces re-raises a GPU task's exception in the app's process wrapped (a
    ZeroGPUTaskException, the task's exception in ``exception``): the message shown is
    the InputError / SetupError inside."""

    class ZeroGPUTaskException(Exception):  # spaces.zero.errors' shape
        def __init__(self, message, exception):
            super().__init__(message)
            self.exception = exception

    oom = su.InputError("Out of GPU memory at 160x90 (0.0 MP).")
    assert su.user_error(oom) is oom
    setup = su.SetupError("no weights")
    assert su.user_error(setup) is setup
    assert su.user_error(ZeroGPUTaskException("InputError", oom)) is oom
    assert su.user_error(ZeroGPUTaskException("RuntimeError", RuntimeError())) is None
    assert su.user_error(RuntimeError(ALLOCATOR_ASSERT)) is None


def test_render_result(tmp_path):
    service = stub_service(tmp_path)
    rgb = np.random.default_rng(0).integers(0, 256, (200, 300, 3), dtype=np.uint8)
    depth, seconds, peak = service.predict(ZS, rgb, 768, True)
    entry = service.registry.get(ZS)
    res = su.render_result(
        rgb,
        depth,
        entry=entry,
        size=768,
        flip=True,
        cmap="Inferno",
        seconds=seconds,
        peak=peak,
        settings=service.settings,
        name="/x/My photo!.jpg",
        load_seconds=12.0,
    )
    assert res.depth_preview.shape == res.input_preview.shape == (43, 64, 3)
    png, jpg, sidecar = map(Path, res.files)
    assert png.name == f"My_photo_{ZS}_depth16.png"
    assert jpg.name == f"My_photo_{ZS}_depth_inferno.jpg"
    back, meta = su.read_depth16(png)
    assert np.abs(back - depth).max() < 1e-4
    assert json.loads(sidecar.read_text()) == meta == res.meta
    assert meta["color_file"] == jpg.name and meta["mapping"] == "minmax"
    assert meta["inference_size"] == [768, 1152] and meta["flip"] is True
    # the preview and the download share the full-resolution depth's tone curve
    curve = su.ToneCurve(**meta["color_curve"])
    assert curve == su.tone_curve(depth) and meta["color_scale"] == "tone curve"
    small = su.resize_area(depth, (43, 64))
    assert np.array_equal(res.depth_preview, su.colorize(small, "Inferno", curve=curve))
    full = cv2.imread(str(jpg))
    assert full.shape == (200, 300, 3)  # full resolution, previews downsampled
    for part in (
        "input 300x200 (0.1 MP)",
        "inference 1152x768 (0.9 MP) + flip",
        "output 300x200",
        entry.label,
        "+ 12 s model load",
        "peak GPU n/a",
        "preview 64x43",
    ):
        assert part in res.info
    with pytest.raises(ValueError, match="does not match"):
        su.render_result(
            rgb,
            depth[:10],
            entry=entry,
            size=None,
            flip=False,
            cmap="Gray",
            seconds=0,
            peak=None,
            settings=service.settings,
        )


def test_render_result_png_above_the_jpeg_limit(tmp_path, monkeypatch):
    """A side above JPEG's 65,500 px: the colorized download is a lossless PNG."""
    assert cv2.imencode(".jpg", np.zeros((2, su.JPEG_MAX_SIDE, 3), np.uint8))[0]
    assert not cv2.imencode(".jpg", np.zeros((2, su.JPEG_MAX_SIDE + 1, 3)))[0]
    monkeypatch.setattr(su, "JPEG_MAX_SIDE", 299)
    service = stub_service(tmp_path)
    rgb = np.random.default_rng(0).integers(0, 256, (200, 300, 3), dtype=np.uint8)
    depth, seconds, peak = service.predict(ZS, rgb, None, False)
    res = su.render_result(
        rgb,
        depth,
        entry=service.registry.get(ZS),
        size=None,
        flip=False,
        cmap="Turbo",
        seconds=seconds,
        peak=peak,
        settings=service.settings,
        name="pano.png",
    )
    png, color, sidecar = map(Path, res.files)
    assert color.name == f"pano_{ZS}_depth_turbo.png" == res.meta["color_file"]
    assert json.loads(sidecar.read_text())["color_file"] == color.name
    assert res.meta["color_scale"] == "tone curve"
    curve = su.ToneCurve(**res.meta["color_curve"])
    assert curve == su.tone_curve(depth)
    expected = su.colorize(depth, "Turbo", bgr=True, curve=curve)
    assert np.array_equal(cv2.imread(str(color)), expected)


def test_output_dirs_are_pruned(tmp_path):
    old = su.new_output_dir(tmp_path, 3600)
    os.utime(old, (0, 0))
    new = su.new_output_dir(tmp_path, 3600)
    assert not old.exists() and new.is_dir()


# ---- make_examples


def test_examples_from_a_repo_and_locally(tmp_path, caplog):
    import json as _json

    repo = tmp_path / "repo"
    (repo / "images").mkdir(parents=True)
    (repo / "thumbs").mkdir()
    for name in ("big", "small", "gone"):
        if name != "gone":
            cv2.imwrite(str(repo / "images" / f"{name}.jpg"), np.zeros((8, 8, 3)))
    cv2.imwrite(str(repo / "thumbs" / "big.jpg"), np.zeros((4, 4, 3)))
    (repo / "results" / "big").mkdir(parents=True)
    for f in ("rgb", ZS):
        cv2.imwrite(str(repo / "results" / "big" / f"{f}.jpg"), np.zeros((4, 4, 3)))
    doc = dict(
        format=su.EXAMPLES_FORMAT,
        examples=[
            dict(
                name="big",
                file="images/big.jpg",
                thumb="thumbs/big.jpg",
                megapixels=102.1,
                credit=" Someone on Unsplash ",
                results=dict(
                    rgb="results/big/rgb.jpg",
                    models={
                        ZS: dict(
                            preview=f"results/big/{ZS}.jpg",
                            seconds=311.7,
                            device="GB200",
                        ),
                        MIX: dict(preview="results/big/missing.jpg", seconds=1),
                    },
                ),
            ),
            dict(name="small", file="images/small.jpg", thumb="thumbs/small.jpg"),
            dict(name="gone", file="images/gone.jpg"),
        ],
    )
    (repo / "examples.json").write_text(_json.dumps(doc))
    calls = []

    def snapshot(**kw):
        calls.append(kw)
        return str(repo)

    env = dict(LDPT_EXAMPLES_REPO="me/ex", LDPT_EXAMPLES_REVISION="abc", HF_TOKEN="t")
    ex = su.resolve_examples(env, snapshot, local=tmp_path / "none")
    assert [e.name for e in ex] == ["big", "small"]  # a missing file is skipped
    assert ex[0].thumb.endswith("thumbs/big.jpg") and ex[0].caption == "Big"  # badge
    assert ex[0].credit == "Someone on Unsplash" and ex[1].credit is None
    assert ex[1].thumb == ex[1].path and ex[1].caption == "Small"  # no thumbnail file
    assert su.Example("misty-forest_lake", "a.jpg", "a.jpg", 90.6).caption == (
        "Misty forest lake · 90.6 MP"
    )
    # precomputed results: only those whose preview exists; none without an RGB preview
    assert ex[0].rgb_preview.endswith("results/big/rgb.jpg")
    assert set(ex[0].results) == {ZS} and ex[0].result(MIX) is None
    assert ex[0].result(ZS)["seconds"] == 311.7
    assert ex[0].result(ZS)["preview"].endswith(f"results/big/{ZS}.jpg")
    assert ex[1].results == {} and ex[1].result(ZS) is None
    assert calls[0]["repo_type"] == "dataset" and calls[0]["revision"] == "abc"
    assert calls[0]["token"] == "t" and calls[0]["local_dir"] == str(su.EXAMPLES_DIR)
    # no repo: the local folder, each image its own thumbnail; a failed download too
    local = tmp_path / "local"
    local.mkdir()
    cv2.imwrite(str(local / "a.jpg"), np.zeros((8, 8, 3)))
    assert [e.name for e in su.resolve_examples({}, snapshot, local=local)] == ["a"]

    def offline(**kw):
        raise OSError("offline")

    with caplog.at_level("ERROR"):
        ex = su.resolve_examples(env, offline, local=local)
    assert [e.name for e in ex] == ["a"] and "not loaded" in caplog.text
    (repo / "examples.json").write_text(_json.dumps(dict(format="other")))
    with pytest.raises(su.SetupError, match="format"):
        su.read_examples(repo)


def test_local_examples_folder(tmp_path):
    """$LDPT_EXAMPLES_DIR (app.py --examples) is a folder of photos for the gallery;
    without it the Space's examples/ folder, which a clone does not have: no gallery."""
    photos = tmp_path / "photos"
    photos.mkdir()
    for name in ("b.png", "a.JPG"):
        cv2.imwrite(str(photos / name), np.zeros((8, 8, 3)))
    (photos / "notes.txt").write_text("not an image")
    env = {su.EXAMPLES_DIR_ENV: str(photos)}
    assert su.local_examples_dir(env) == photos
    assert su.local_examples_dir({}) == SPACE / "examples"
    no_download = lambda **kw: pytest.fail("no download expected")  # noqa: E731
    ex = su.resolve_examples(env, no_download)
    assert [(e.name, e.thumb == e.path) for e in ex] == [("a", True), ("b", True)]
    env[su.EXAMPLES_DIR_ENV] = str(tmp_path / "missing")
    assert su.resolve_examples(env, no_download) == []


def test_make_examples(tmp_path):
    import make_examples

    src = tmp_path / "photo.png"
    cv2.imwrite(str(src), np.full((300, 400, 3), 128, np.uint8))
    out = make_examples.make_examples([src], tmp_path / "ex", mps=[0.05, 0.02, 1.0])
    assert [p.name for p in out] == ["photo_0.05mp.jpg", "photo_0.02mp.jpg"]
    for path, mp in zip(out, (0.05, 0.02)):
        h, w = cv2.imread(str(path)).shape[:2]
        assert abs(h * w / 1e6 - mp) / mp < 0.02 and abs(w / h - 4 / 3) < 0.02
    assert su.list_examples(tmp_path / "ex") == sorted(map(str, out))
    assert su.list_examples(tmp_path / "missing") == []

    # two photos with one stem (case-insensitive) are refused before any write; the
    # same photo twice is made once
    a, b = tmp_path / "trip1" / "IMG_1.png", tmp_path / "trip2" / "img_1.png"
    for path in (a, b):
        path.parent.mkdir()
        cv2.imwrite(str(path), np.full((300, 400, 3), 128, np.uint8))
    with pytest.raises(SystemExit, match="would write the same examples"):
        make_examples.make_examples([a, b], tmp_path / "ex2", mps=[0.05])
    assert not (tmp_path / "ex2").exists()
    out = make_examples.make_examples([a, a], tmp_path / "ex3", mps=[0.05])
    assert [p.name for p in out] == ["IMG_1_0.05mp.jpg"]


# ---- the inference subset of the ldpt package

BLOCKED = [
    "wandb",
    "tensorboard",
    "tensorboardX",
    "addict",
    "yapf",
    "termcolor",
    "h5py",
    "OpenEXR",
    "Imath",
    "torchmetrics",
    "transformers",
    "torchvision",
    "xformers",
    "flash_attn",
    "cv2",
    "matplotlib",
    "PIL",
    "huggingface_hub",
    "gradio",
    "spaces",
]

SUBSET_RUN = r"""
import importlib.abc, json, os, sys

space, work, blocked = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in blocked or name.startswith("torch.utils.tensorboard"):
            raise ImportError(f"blocked for the test: {name}")
        return None


sys.meta_path.insert(0, Block())
sys.path.insert(0, space)
import numpy as np
import torch

from ldpt.models.builder import MODELS, build_model
from ldpt.models.dinov3.model import DinoVisionTransformer
from ldpt.models.l_dpt import api, convert


class Tiny(DinoVisionTransformer):
    def __init__(self):
        super().__init__(
            patch_size=16, in_chans=3, pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate", pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32", embed_dim=64, depth=4, num_heads=4,
            ffn_ratio=3, qkv_bias=False, drop_path_rate=0.0, layerscale_init=1.0e-05,
            norm_layer="layernormbf16", ffn_layer="swiglu64", ffn_bias=True,
            proj_bias=True, n_storage_tokens=4, mask_k_bias=True,
        )


MODELS.register_module("dinov3_tiny_space_test", module=Tiny)
cfg = dict(
    type="L-DPT-v1m1", encoder=dict(type="dinov3_tiny_space_test"),
    encoder_inter_layers=[0, 1, 2, 3], dec_dim=32, dec_num_heads=2,
    dec_num_layers=(1, 1, 1, 1), dec_qk_norm=True,
)
torch.manual_seed(0)
model = build_model(cfg)
model.encoder.init_weights()
torch.save(model.encoder.state_dict(), os.path.join(work, "enc.pth"))
decoder = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
torch.save(dict(state_dict=decoder), os.path.join(work, "dev.pth"))
release = os.path.join(work, "rel.pth")
convert.convert(os.path.join(work, "dev.pth"), release, config=cfg)
# as the Space loads (DepthService._api_load), with a smaller token chunk
loaded = api.load(
    release, os.path.join(work, "enc.pth"), device="cpu", precision="bf16",
    selection_impl="stream", chunk_tokens=64,
)
image = np.random.default_rng(0).integers(0, 256, (40, 56, 3), dtype=np.uint8)
depths = [api.predict(loaded, image, size=s, flip=True) for s in (None, 48)]
assert all(d.shape == (40, 56) for d in depths), [d.shape for d in depths]
# the colour and 16-bit PNG module imports without matplotlib / OpenCV
from ldpt.utils import depth_image

values, lo, hi = depth_image.encode_depth16(depths[0])
assert values.dtype == np.uint16 and lo <= hi
modules = {
    n: getattr(m, "__file__", None) for n, m in sys.modules.items()
    if n == "ldpt" or n.startswith("ldpt.")
}
finite = all(bool(np.isfinite(d).all()) for d in depths)
print(json.dumps(dict(modules=modules, finite=finite)))
"""


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_sync_subset_runs_without_training_deps(tmp_path):
    space = tmp_path / "space"
    env = dict(os.environ, PY=sys.executable)
    run = subprocess.run(
        ["bash", str(SPACE / "sync_ldpt.sh"), str(space / "ldpt")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "files from" in run.stdout
    listed = (space / "ldpt" / "SUBSET.txt").read_text().splitlines()[2:]
    copied = sorted(
        str(p.relative_to(space)) for p in (space / "ldpt").rglob("*.py") if p.is_file()
    )
    assert listed == copied and "ldpt/models/l_dpt/api.py" in copied
    assert "ldpt/utils/depth_image.py" in copied  # space_utils' colour and 16-bit PNG
    assert "ldpt/models/dinov3/hf_format.py" in copied  # the transformers-format reader
    assert not any(
        p.startswith(("ldpt/engines", "ldpt/datasets", "ldpt/hooks")) for p in copied
    )
    assert "ldpt/utils/config.py" not in copied  # training config parser (convert CLI)
    assert not list((space / "ldpt").rglob("__pycache__"))

    work = tmp_path / "work"
    work.mkdir()
    out = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            SUBSET_RUN,
            str(space),
            str(work),
            ",".join(BLOCKED),
        ],
        cwd=work,
        env=dict(os.environ, LDPT_CACHE=str(tmp_path / "cache")),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["finite"]
    files = [f for f in result["modules"].values() if f is not None]
    assert files and all(Path(f).is_relative_to(space / "ldpt") for f in files)

    with pytest.raises(subprocess.CalledProcessError):  # refuses odd destinations
        subprocess.run(
            ["bash", str(SPACE / "sync_ldpt.sh"), str(tmp_path / "pkg")],
            env=env,
            capture_output=True,
            check=True,
        )


# ---- the Gradio app (only where gradio is installed)


def _demo(tmp_path, monkeypatch, **settings):
    pytest.importorskip("gradio")
    monkeypatch.setattr(sys, "path", list(sys.path))  # app.py prepends its folder
    app = importlib.import_module("app")
    demo = app.build_demo(stub_service(tmp_path, **settings))
    fns = [f for f in demo.fns.values() if f.api_name == "depth"]
    assert len(fns) == 1
    return app, demo, fns[0].fn


def test_gradio_app_click(tmp_path, monkeypatch):
    app, demo, run_depth = _demo(tmp_path, monkeypatch)
    import gradio as gr

    image = tmp_path / "photo.png"
    cv2.imwrite(
        str(image), np.random.default_rng(0).integers(0, 256, (90, 160, 3), np.uint8)
    )
    descs = []  # the progress text quotes su.quoted_seconds (370 s for 102 MP)
    with monkeypatch.context() as m:
        m.setattr(su, "quoted_seconds", lambda h, w, gpu: (370.0, False))
        out = run_depth(
            str(image), ZS, progress=lambda f, desc=None: descs.append(desc)
        )
    assert "Inference at 160x90 (estimate 370 s)" in descs
    by_type = {type(k).__name__: v for k, v in out.items()}
    assert by_type["Image"].shape == (36, 64, 3)  # preview of 160 x 90 at 64 px
    assert (
        "160x90 (0.0 MP), native resolution" in by_type["Markdown"]
        and "L-DPT Large (zero-shot)" in by_type["Markdown"]
    )
    files = by_type["File"]
    assert len(files) == 3 and all(Path(f).is_file() for f in files)
    assert cv2.imread(files[1]).shape == (90, 160, 3)
    assert "inference" not in by_type["Markdown"]  # native: one size, no flip
    assert "flip" not in by_type["Markdown"]
    meta = json.loads(Path(files[2]).read_text())
    assert (meta["colormap"], meta["color_scale"], meta["flip"]) == (
        su.DISPLAY_CMAP,
        "tone curve",
        False,
    )
    # no resolution, flip or colormap choice in the UI: the only choice is the model
    choices = [
        b.label for b in demo.blocks.values() if isinstance(b, (gr.Radio, gr.Checkbox))
    ]
    assert choices == ["Model"]
    assert not [
        b for b in demo.blocks.values() if getattr(b, "label", "") == "Colormap"
    ]
    if hasattr(gr, "ImageSlider"):
        before, after = by_type["ImageSlider"]
        assert before.shape == after.shape == (36, 64, 3)
    assert list(app.SERVICE.models) == [ZS]
    out = run_depth(str(image), MIX)
    assert (
        "L-DPT Large (mixed)"
        in [v for k, v in out.items() if type(k).__name__ == "Markdown"][0]
    )
    assert list(app.SERVICE.models) == [ZS, MIX]

    with pytest.raises(gr.Error, match="Upload an image"):
        run_depth(None, ZS)


def test_gradio_app_example_gallery(tmp_path, monkeypatch):
    """The examples are a gallery of thumbnails; a click loads the full file and shows
    the example's precomputed result for the chosen model, if it has one; a model switch
    shows that model's; an upload forgets the example."""
    gr = pytest.importorskip("gradio")
    files = []
    for i in range(3):
        f = tmp_path / f"ex{i}.jpg"
        cv2.imwrite(str(f), np.full((8, 12, 3), 40 * i, np.uint8))
        files.append(str(f))
    res = {ZS: dict(preview=files[1], seconds=311.7, device="NVIDIA GB200")}
    examples = [su.Example(f"ex{i}", f, f, 1.5) for i, f in enumerate(files)]
    examples[2] = su.Example("ex2", files[2], files[2], 102.1, files[0], res, "Ann")
    monkeypatch.setattr(su, "resolve_examples", lambda: examples)
    _, demo, _ = _demo(tmp_path, monkeypatch)
    (gallery,) = [b for b in demo.blocks.values() if isinstance(b, gr.Gallery)]
    assert len(gallery.value) == 3
    fns = {getattr(f.fn, "__name__", ""): f.fn for f in demo.fns.values()}
    named = lambda out: {
        getattr(k, "label", None) or type(k).__name__: v for k, v in out.items()
    }
    out = named(fns["pick"](ZS, types.SimpleNamespace(index=2)))
    assert out["Photo"] == files[2] and out["State"] == 2
    assert out["Depth"] == files[1]
    md = [v for k, v in out.items() if k in ("Markdown",)]
    assert (
        md
        and "Precomputed: Ex2 · 102.1 MP" in md[0]
        and "computed on NVIDIA GB200 (native resolution) · Photo: Ann." in md[0]
        and "312 s" not in md[0]  # the example's own run time is not shown
    )
    if hasattr(gr, "ImageSlider"):
        assert out["Photo / depth"] == (files[0], files[1])
    out = named(fns["pick"](MIX, types.SimpleNamespace(index=2)))
    assert out["Depth"] is None  # no precomputed MIX result: the outputs are cleared
    assert out["Markdown"] == "Ex2 · Photo: Ann"  # the credit stays
    out = named(fns["on_model"](ZS, 2))
    assert out["Depth"] == files[1]
    out = named(fns["on_model"](MIX, None))
    assert set(out) == {"Markdown"}  # an upload: only the description changes
    out = named(fns["pick"](ZS, types.SimpleNamespace(index=0)))
    assert out["Photo"] == files[0] and out["Depth"] is None


def test_gradio_app_single_model(tmp_path, monkeypatch):
    """With one model (the shipped registry) the UI shows no model choice."""
    gr = pytest.importorskip("gradio")
    monkeypatch.setattr(sys, "path", list(sys.path))
    app = importlib.import_module("app")
    service = stub_service(tmp_path)
    service.registry = su.Registry.load()  # the Space's one model
    demo = app.build_demo(service)
    assert not [b for b in demo.blocks.values() if isinstance(b, gr.Radio)]
    (run,) = [f.fn for f in demo.fns.values() if f.api_name == "depth"]
    image = tmp_path / "photo.png"
    cv2.imwrite(str(image), np.zeros((90, 160, 3), np.uint8))
    out = run(str(image), ZS)
    assert (
        "L-DPT Large (zero-shot)"
        in [v for k, v in out.items() if type(k).__name__ == "Markdown"][0]
    )


def test_gradio_app_limits(tmp_path, monkeypatch):
    _, _, run_depth = _demo(tmp_path, monkeypatch, max_mp=0.01, max_infer_mp=0.5)
    import gradio as gr

    image = tmp_path / "photo.png"
    cv2.imwrite(str(image), np.zeros((90, 160, 3), np.uint8))  # 0.0144 MP
    with pytest.raises(gr.Error, match="up to 0.01 MP"):
        run_depth(str(image), ZS)
    _, _, run_depth = _demo(tmp_path, monkeypatch, max_infer_mp=0.01)
    with pytest.raises(gr.Error, match="Downscale the photo"):
        run_depth(str(image), ZS)
    _, _, run_depth = _demo(tmp_path, monkeypatch, max_flip_mp=0.01)
    run_depth(str(image), ZS)  # no flip: the flip limit does not apply
    _, _, run_depth = _demo(tmp_path, monkeypatch, max_gpu_seconds=10)
    with pytest.raises(gr.Error, match="GPU seconds"):  # the base alone is 20 s
        run_depth(str(image), ZS)


class ZeroGPUTaskException(Exception):
    """spaces.zero.errors.ZeroGPUTaskException's shape: how spaces re-raises a GPU
    task's exception in the app's process."""

    def __init__(self, message, exception):
        super().__init__(message)
        self.exception = exception


def _zero_gpu_app(tmp_path, monkeypatch, service=None, zero_gpu=True):
    """app.py imported afresh with a stub ``spaces`` (spaces.GPU records each wrapped
    function's size and each call's duration, and wraps a task's exception as spaces
    does) and SPACES_ZERO_GPU set (``zero_gpu``) or not."""
    pytest.importorskip("gradio")
    service = service or stub_service(tmp_path)
    wrapped, calls = {}, []

    def GPU(duration=None, size=None):
        def wrap(fn):
            assert fn.__name__ not in wrapped  # spaces keeps one wrapper per function
            wrapped[fn.__name__] = size

            def run(*args):
                calls.append((size, duration(*args)))
                try:
                    return fn(*args)
                except Exception as e:
                    raise ZeroGPUTaskException(type(e).__name__, e) from e

            return run

        return wrap

    monkeypatch.setitem(sys.modules, "spaces", types.SimpleNamespace(GPU=GPU))
    if zero_gpu:
        monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    else:
        monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    monkeypatch.setattr(
        su.DepthService, "from_env", classmethod(lambda cls, env=None: service)
    )
    monkeypatch.delitem(sys.modules, "app", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    app = importlib.import_module("app")
    return app, service, wrapped, calls


def test_gradio_app_zero_gpu_loads_at_import(tmp_path, monkeypatch):
    """On ZeroGPU every model is loaded when app.py is imported (before demo.launch;
    a load after it would fail), and the inference runs in spaces.GPU with the
    estimated duration."""
    app, service, wrapped, calls = _zero_gpu_app(tmp_path, monkeypatch)
    assert app.ZERO_GPU and app.SERVICE is service
    # two entry points: the default slot (no size given) and the full GPU
    assert wrapped == {"_predict": None, "_predict_full_gpu": su.GPU_XLARGE}
    assert set(service.models) == {m.id for m in service.registry.models}
    assert service.frozen  # nothing loads after the import
    assert app.MAX_FILE_SIZE == "400mb"
    demo = app.build_demo()  # keeps the loaded service
    assert app.SERVICE is service
    (fn,) = [f.fn for f in demo.fns.values() if f.api_name == "depth"]
    image = tmp_path / "photo.png"
    cv2.imwrite(str(image), np.zeros((90, 160, 3), np.uint8))
    loaded = dict(service.models)
    for m in service.registry.models:
        fn(str(image), m.id)
    assert service.models == loaded  # nothing loaded after the import
    assert calls == [(None, su.gpu_seconds(90, 160, None, False))] * len(loaded)


def test_gradio_app_zero_gpu_routes_large_photos_to_the_full_gpu(tmp_path, monkeypatch):
    """The entry point is chosen per request before the GPU is requested: the full GPU
    (size="xlarge", its own duration) above the peak-memory threshold, the default slot
    below it; out of GPU memory inside the task (the allocator's assert included) shows
    as the InputError's message, not as the wrapped exception's class."""
    gr = pytest.importorskip("gradio")
    app, service, wrapped, calls = _zero_gpu_app(tmp_path, monkeypatch)
    for hw, slot in (((10000, 10500), su.GPU_LARGE), ((11000, 11000), su.GPU_XLARGE)):
        entry = app.gpu_entry(*hw, None)
        assert su.zerogpu_size(*hw) == slot
        assert entry is (
            app.predict_on_full_gpu if slot == su.GPU_XLARGE else app.predict_on_gpu
        )
    assert app.predict_on_gpu is not app.predict_on_full_gpu
    demo = app.build_demo()
    (fn,) = [f.fn for f in demo.fns.values() if f.api_name == "depth"]
    image = tmp_path / "photo.png"
    cv2.imwrite(str(image), np.zeros((90, 160, 3), np.uint8))
    fn(str(image), ZS)  # 0.01 MP: the default slot
    # a threshold below every photo's estimate: this one runs on the full GPU
    monkeypatch.setattr(su, "ZEROGPU_LARGE_PEAK_GIB", 1.0)
    assert app.gpu_entry(90, 160, None) is app.predict_on_full_gpu
    out = fn(str(image), ZS)
    assert (
        "160x90 (0.0 MP)"
        in [v for k, v in out.items() if type(k).__name__ == "Markdown"][0]
    )
    assert calls == [
        (None, su.gpu_seconds(90, 160, None, False)),
        (su.GPU_XLARGE, su.gpu_seconds(90, 160, None, False, su.GPU_XLARGE)),
    ]

    def allocator_assert(model, rgb, size, flip):
        raise RuntimeError(ALLOCATOR_ASSERT)

    service.predictor = allocator_assert
    with pytest.raises(gr.Error, match=r"Out of GPU memory at 160x90 \(0.0 MP\)"):
        fn(str(image), ZS)

    def bug(model, rgb, size, flip):
        raise KeyError("a bug")

    service.predictor = bug  # anything else is not a message for the user
    with pytest.raises(ZeroGPUTaskException):
        fn(str(image), ZS)


def test_app_local_run_never_uses_spaces(tmp_path, monkeypatch):
    """Off ZeroGPU (no SPACES_ZERO_GPU, even with spaces installed) app.py never calls
    spaces.GPU: both entry points are the plain functions and every photo takes the
    default one."""
    pytest.importorskip("gradio")
    app, service, wrapped, calls = _zero_gpu_app(
        tmp_path, monkeypatch, service=None, zero_gpu=False
    )
    assert not app.ZERO_GPU and app.SERVICE is None and wrapped == {}
    assert app.predict_on_gpu is app._predict
    assert app.predict_on_full_gpu is app._predict_full_gpu
    for hw in ((90, 160), (10000, 10000), (10000, 10500)):
        assert app.gpu_entry(*hw, None) is app.predict_on_gpu
    monkeypatch.setattr(su, "ZEROGPU_LARGE_PEAK_GIB", 1.0)
    assert app.gpu_entry(90, 160, None) is app.predict_on_gpu
    demo = app.build_demo(stub_service(tmp_path / "local"))
    (fn,) = [f.fn for f in demo.fns.values() if f.api_name == "depth"]
    image = tmp_path / "photo.png"
    cv2.imwrite(str(image), np.zeros((90, 160, 3), np.uint8))
    fn(str(image), ZS)
    assert calls == [] and wrapped == {}


def test_gradio_app_without_image_slider(tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    monkeypatch.delattr(gr, "ImageSlider", raising=False)
    _, _, run_depth = _demo(tmp_path, monkeypatch)
    image = tmp_path / "photo.png"
    cv2.imwrite(str(image), np.zeros((90, 160, 3), np.uint8))
    out = run_depth(str(image), MIX)
    images = [v for k, v in out.items() if type(k).__name__ == "Image"]
    assert len(images) == 3  # depth, then the side-by-side image and depth
    assert all(v.shape == (36, 64, 3) for v in images)
    assert not any(type(k).__name__ == "ImageSlider" for k in out)


# ---- running the demo locally: app.py's command line and a real launch


def _main(tmp_path, monkeypatch, argv, service=None):
    """app.main(argv) on a stub service, with demo.launch recorded instead of run."""
    gr = pytest.importorskip("gradio")
    for var in (su.EXAMPLES_DIR_ENV, "SPACE_ID"):  # main sets the first; restored
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    service = service or stub_service(tmp_path / "service")
    monkeypatch.setattr(
        su.DepthService, "from_env", classmethod(lambda cls, env=None: service)
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "SERVICE", None)
    launched = []
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kw: launched.append(kw))
    demo = app.main(argv)
    assert isinstance(demo, gr.Blocks) and len(launched) == 1
    return app, service, launched[0]


def test_app_main_loads_then_launches(tmp_path, monkeypatch, capsys):
    """python app.py: the default model is loaded before the server starts; host,
    port and share go to launch() only when given (a Space sets GRADIO_SERVER_NAME
    and GRADIO_SERVER_PORT itself); the output and example folders may be served."""
    app, service, kw = _main(tmp_path, monkeypatch, [])
    assert list(service.models) == [ZS]  # loaded at startup, before launch
    out = capsys.readouterr().out
    assert "Loading L-DPT Large (zero-shot)" in out and "Loaded" in out
    assert not {"server_name", "server_port", "share"} & set(kw)
    assert kw["max_file_size"] == app.MAX_FILE_SIZE
    assert str(service.settings.output_dir) in kw["allowed_paths"]
    assert str((SPACE / "examples").resolve()) in kw["allowed_paths"]
    assert ("theme" in kw and "css" in kw) == app.STYLE_IN_LAUNCH  # Gradio 6

    photos = tmp_path / "photos"
    photos.mkdir()
    cv2.imwrite(str(photos / "p.jpg"), np.zeros((8, 8, 3), np.uint8))
    argv = ["--host", "0.0.0.0", "--port", "7999", "--share", "--examples", str(photos)]
    app, _, kw = _main(tmp_path, monkeypatch, argv)
    assert (kw["server_name"], kw["server_port"], kw["share"]) == (
        "0.0.0.0",
        7999,
        True,
    )
    assert str(photos.resolve()) in kw["allowed_paths"]  # served from any folder
    assert os.environ[su.EXAMPLES_DIR_ENV] == str(photos)

    with pytest.raises(SystemExit):
        app.parse_args(["--examples", str(tmp_path / "missing")])


def test_app_main_lazy_and_setup_errors(tmp_path, monkeypatch, capsys):
    """--lazy (or LDPT_PRELOAD=0) starts the server without loading; a missing
    DINOv3 or weights file stops the start with the reason, not a traceback."""
    _, service, _ = _main(tmp_path, monkeypatch, ["--lazy"])
    assert service.models == {}
    _, service, _ = _main(
        tmp_path, monkeypatch, [], stub_service(tmp_path / "s2", preload=False)
    )
    assert service.models == {}

    broken = stub_service(tmp_path / "s3")
    broken.env = dict(broken.env, LDPT_DINOV3=str(tmp_path / "nowhere.pth"))
    with pytest.raises(SystemExit, match="LDPT_DINOV3: .*nowhere.pth is not a file"):
        _main(tmp_path, monkeypatch, [], broken)
    assert "Loading" in capsys.readouterr().out


def test_app_texts_off_a_space(tmp_path, monkeypatch):
    """A local run shows the GPU times of one local GPU, not those of 'this Space'."""
    gr = pytest.importorskip("gradio")
    monkeypatch.setattr(sys, "path", list(sys.path))
    app = importlib.import_module("app")
    photo = tmp_path / "ex.jpg"
    cv2.imwrite(str(photo), np.zeros((8, 8, 3), np.uint8))
    result = {ZS: dict(preview=str(photo), seconds=1.0, device="NVIDIA GB200")}
    examples = [su.Example("ex", str(photo), str(photo), 1.0, str(photo), result)]
    monkeypatch.setattr(su, "resolve_examples", lambda: examples)

    def texts(space_id):
        if space_id:
            monkeypatch.setenv("SPACE_ID", space_id)
        else:
            monkeypatch.delenv("SPACE_ID", raising=False)
        demo = app.build_demo(stub_service(tmp_path / "s"))
        return " ".join(
            str(b.value)
            for b in demo.blocks.values()
            if isinstance(b, (gr.HTML, gr.Markdown))
        )

    local, space = texts(None), texts("someone/L-DPT")
    assert "this Space" not in local and "RTX PRO 6000" in local
    assert "runs it again on your GPU" in local
    assert "GPU time on this Space" in space and "RTX PRO 6000" not in space
    assert "runs it again here" in space
    # a folder of photos (--examples): nothing precomputed to promise
    examples[0] = su.Example("ex", str(photo), str(photo))
    local = texts(None)
    assert "Click a photo to load it" in local and "GB200" not in local


def test_app_texts_from_the_timing_guide(monkeypatch):
    """The hero links (Code only once CODE_URL is a URL), the GPU-time line (from the
    timing guide, so a change there is the only edit) and the gallery title."""
    pytest.importorskip("gradio")
    monkeypatch.setattr(sys, "path", list(sys.path))
    app = importlib.import_module("app")
    hero = app.hero_html(105)
    assert "up to 105 MP" in hero and "ViT-7B/16" in hero and "Weights</a>" in hero
    assert ("Code</a>" in hero) == app.CODE_URL.startswith("https://")
    monkeypatch.setattr(
        app, "LINKS", app.LINKS + (("Code", "https://example.org/l-dpt"),)
    )
    assert '<a href="https://example.org/l-dpt" target="_blank">Code</a>' in (
        app.hero_html(105)
    )

    space, local = app.timing_html(True, True), app.timing_html(False)
    # each size on its slot (all on the half slot, all measured; 100 MP: the 102 MP
    # photo), and what 100 MP reserves of the quota (1.5 times its 480 s duration, to
    # two significant digits)
    assert (
        "1.2 s at 2 MP · 14 s at 8 MP · 70 s at 32 MP · 160 s at 60 MP · "
        "370 s at 100 MP." in space
    )
    assert (
        "A 100 MP photo reserves 720 s of your daily ZeroGPU quota before it starts, "
        "more than a free account has." in space
    )
    assert su.quota_seconds(10000, 10000, None, False) == 720
    assert app.ZEROGPU_FREE_QUOTA_S == 300  # Hugging Face's free quota: 5 minutes
    assert "precomputed" in space and "precomputed" not in app.timing_html(True)
    assert (
        "on one RTX PRO 6000 Blackwell: 0.64 s at 2 MP · 7.6 s at 8 MP · 39 s at 32 MP "
        "· 106 s at 60 MP · 205 s at 102 MP." in local
    )
    assert "(estimate)" not in space + local
    assert not HEDGES.search(space + local), (space, local)
    assert (
        "15 GiB at 2 MP, 19 GiB at 8 MP, 21 GiB at 32 MP, 24 GiB at 60 MP, "
        "33 GiB at 102 MP." in local
    )
    # the measured times and peaks as the Space card and the README write them (the
    # README's 100 MP row: the guide's 102 MP measurement)
    card = (SPACE / "README.md").read_text()
    assert f"a free account has {app.ZEROGPU_FREE_QUOTA_S} s" in card
    card = _paragraph(card, "**GPU time** on this Space")
    assert _seconds_stated(space) == _seconds_stated(card)
    rows = _time_table((REPO / "README.md").read_text())
    mp_of = {mp: su.measured_mp(su.RTX_PRO_6000.seconds, mp) for mp in rows}
    readme_rtx = {mp_of[mp]: _cell_seconds(r[1]) for mp, r in rows.items()}
    assert _seconds_stated(local) == readme_rtx
    peaks = ", ".join(f"{r[0]:.0f} GiB at {mp_of[mp]:g} MP" for mp, r in rows.items())
    assert f"Peak GPU memory allocated: {peaks}." in local
    faster = su.GpuTiming("ZeroGPU", {8: 4.0}, {}, (0.24, 0.035))
    monkeypatch.setattr(su, "ZEROGPU", faster)
    monkeypatch.setattr(su, "ZEROGPU_XLARGE", su.GpuTiming("full", {}, {}, (1, 0.05)))
    assert "4.0 s at 8 MP" in app.timing_html(True)
    assert "370 s at 100 MP (estimate)" in app.timing_html(True)  # the half slot
    assert "A 100 MP photo reserves 720 s of your daily" in app.timing_html(True)
    monkeypatch.setattr(su, "ZEROGPU_LARGE_PEAK_GIB", 30.0)  # 100 MP: the full GPU
    assert "600 s at 100 MP (estimate)" in app.timing_html(True)
    assert "A 100 MP photo runs on a full GPU, which counts twice" in (
        app.timing_html(True)
    )
    monkeypatch.undo()

    res = {ZS: dict(preview="d.jpg", seconds=311.7, device="NVIDIA GB200")}
    big = [su.Example("a", "a.jpg", "a.jpg", 102.1, "rgb.jpg", res)]
    title = app.examples_title(big, space=True)
    assert "our run on one NVIDIA GB200 shows at once" in title
    # on the half slot: its measurement at 102 MP (within 5 %), as the GPU-time line
    assert "the 102 MP photos take 370 s and need a Hugging Face PRO" in title
    # no measurement within 5 %: the estimate of a square photo that large
    res80 = [su.Example("a", "a.jpg", "a.jpg", 80.0, "rgb.jpg", res)]
    assert su.expected_seconds(8944, 8944) == pytest.approx(249.6, abs=0.1)
    assert "the 80 MP photos take an estimated 250 s and need" in (
        app.examples_title(res80, space=True)
    )
    small = [su.Example("a", "a.jpg", "a.jpg", 2.0, "rgb.jpg", res)]
    assert app.examples_title(small, space=True).endswith("runs it again here.")
    assert app.examples_title(big, space=False).endswith("runs it again on your GPU.")
    plain = [su.Example("a", "a.jpg", "a.jpg")]
    assert "Click a photo to load it" in app.examples_title(plain, space=True)


def test_app_extra_is_the_space_gradio():
    """pip install -e ".[demo,app]" gets the Gradio of the online demo (the Space's
    sdk_version), the version these tests and the Space run."""
    import re

    card = (SPACE / "README.md").read_text()
    (sdk,) = re.findall(r"^sdk_version: (\S+)$", card, re.M)
    pyproject = (REPO / "pyproject.toml").read_text()
    assert re.findall(r"^app = \[(.*)\]$", pyproject, re.M) == [f'"gradio=={sdk}"']


LAUNCH_RUN = r"""
import runpy, sys

repo, tests, app = sys.argv[1:4]
sys.path[:0] = [repo, tests]  # the installed package; the tiny encoder of the test
import helpers  # noqa: F401,E402  (registers dinov3_tiny_test)

sys.argv = [app, *sys.argv[4:]]
runpy.run_path(app, run_name="__main__")
"""


def _tiny_release(work):
    """A release file of the tiny model whose config passes the expect check of
    models.json, and its encoder weights."""
    from helpers import init_encoder_, randomize_decoder_

    from ldpt.models.builder import build_model
    from ldpt.models.l_dpt import convert

    entry = su.Registry.load().default
    cfg = dict(
        type="L-DPT-v1m1",
        encoder=dict(type="dinov3_tiny_test"),
        encoder_inter_layers=[0, 1, 2, 3],
        dec_dim=32,
        dec_num_heads=2,
        dec_num_layers=(1, 1, 1, 1),
        **entry.expect,
    )
    torch.manual_seed(0)
    model = build_model(dict(cfg, criterion=False))  # inference only, as a release
    init_encoder_(model.encoder)
    randomize_decoder_(model)
    encoder, dev, release = work / "enc.pth", work / "dev.pth", work / "rel.pth"
    torch.save(model.encoder.state_dict(), encoder)
    decoder = {
        k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")
    }
    torch.save(dict(state_dict=decoder), dev)
    convert.convert(dev, release, config=cfg)
    return release, encoder


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_json(url: str):
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def _queue_call(base: str, fn_index: int, data, event_data=None):
    """One event through Gradio's queue as the browser sends it (event_data: what a
    gallery click carries, which gradio_client cannot send); the completed message."""
    import urllib.request
    import uuid

    session = uuid.uuid4().hex
    body = dict(
        data=data,
        event_data=event_data,
        fn_index=fn_index,
        trigger_id=None,
        session_hash=session,
    )
    request = urllib.request.Request(
        base + "gradio_api/queue/join",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    urllib.request.urlopen(request, timeout=30).read()
    url = base + f"gradio_api/queue/data?session_hash={session}"
    with urllib.request.urlopen(url, timeout=120) as stream:
        for line in stream:
            if line.startswith(b"data:"):
                message = json.loads(line[5:])
                if message.get("msg") == "process_completed":
                    return message
    raise AssertionError("no process_completed message")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs a POSIX system")
def test_app_runs_locally(tmp_path):
    """The README's local demo, end to end on the CPU: `python demo/hf_space/app.py
    --port P --examples DIR` with the tiny model as the release file and DINOv3 (the
    variables of the docs), from another working directory and another temp folder;
    the server comes up on 127.0.0.1 after the model is loaded, gradio_client runs a
    photo through /depth, a click on a gallery photo outside the working directory
    and the temp folder loads it, and the page carries the local texts."""
    pytest.importorskip("gradio")
    gradio_client = pytest.importorskip("gradio_client")
    work, photos, cwd, tmp = (tmp_path / d for d in ("w", "photos", "cwd", "tmp"))
    for d in (work, photos, cwd, tmp):
        d.mkdir()
    release, encoder = _tiny_release(work)
    rng = np.random.default_rng(0)
    photo = photos / "street.jpg"
    cv2.imwrite(str(photo), rng.integers(0, 256, (72, 104, 3), np.uint8))
    port = _free_port()
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LDPT_", "GRADIO_", "SPACE", "HF_TOKEN"))
    }
    env.update(
        LDPT_DINOV3=str(encoder),
        LDPT_CKPT_LARGE_DEPTH_ZEROSHOT=str(release),
        LDPT_CACHE=str(work / "cache"),
        TMPDIR=str(tmp),  # tempfile.gettempdir(): the outputs and Gradio's uploads
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        GRADIO_ANALYTICS_ENABLED="False",
        PYTHONUNBUFFERED="1",  # the log shows each line at once
    )
    log = (tmp_path / "app.log").open("w")
    argv = ["--port", str(port), "--examples", str(photos)]
    proc = subprocess.Popen(
        [sys.executable, "-c", LAUNCH_RUN, str(REPO), str(REPO / "tests")]
        + [str(SPACE / "app.py")]
        + argv,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}/"
    try:
        import time

        deadline = time.monotonic() + 180
        while "Running on local URL" not in (tmp_path / "app.log").read_text():
            assert proc.poll() is None, (tmp_path / "app.log").read_text()[-3000:]
            assert time.monotonic() < deadline, "the app did not start"
            time.sleep(0.5)
        started = (tmp_path / "app.log").read_text()
        assert "Loaded L-DPT Large (zero-shot)" in started  # before the server
        assert started.index("Loaded") < started.index("Running on local URL")
        assert f"127.0.0.1:{port}" in started

        client = gradio_client.Client(base, verbose=False)
        depth, compare, info, files = client.predict(
            gradio_client.handle_file(str(photo)), api_name="/depth"
        )
        assert "104x72 (0.0 MP), native resolution" in info
        assert cv2.imread(depth).shape == (72, 104, 3)
        paths = [f if isinstance(f, str) else f["path"] for f in files]
        assert sorted(Path(f).suffix for f in paths) == [".jpg", ".json", ".png"]
        depth16 = next(f for f in paths if f.endswith("_depth16.png"))
        values, _ = depth_image.read_depth16(depth16)
        assert values.shape == (72, 104) and np.isfinite(values).all()

        config = _get_json(base + "config")
        page = json.dumps(config)
        assert "RTX PRO 6000" in page and "this Space" not in page
        gallery = [c for c in config["components"] if c.get("type") == "gallery"]
        assert len(gallery) == 1 and len(gallery[0]["props"]["value"]) == 1
        (pick,) = [
            i
            for i, d in enumerate(config["dependencies"])
            if d.get("api_name") == "pick"
        ]
        done = _queue_call(base, pick, [None], {"index": 0, "value": None})
        assert done["success"], done
        loaded = done["output"]["data"][0]
        assert loaded["orig_name"] == "street.jpg" or "street" in loaded["path"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
