"""
The tiled encoder of L-DPT-v1m1 (CPU; ldpt/models/l_dpt/encoder_tiles.py and
LatentDPT.encode_select_tiled / infer(image, encoder_tiles)): tile origins and blend
ramps, the plain forward bit for bit at or below the tile, one tile = the plain path,
and the tiled path itself on the tiny DINOv3-shaped encoder with a small tile.
"""

import math

import pytest
import torch

from helpers import randomize_decoder_
from test_l_dpt_golden import BASE, randomize_

from ldpt.models.builder import build_model
from ldpt.models.l_dpt.api import ENCODER_TILES
from ldpt.models.l_dpt.encoder_tiles import (
    EncoderTiles,
    tile_ramp,
    tile_starts,
    tile_windows,
)

DEPTH = dict(BASE, criterion=False, head="exp")
SEMSEG = dict(BASE, task="semseg", out_dim=8, num_classes=5, criterion=False)
SMALL = EncoderTiles(tile=8, overlap=2, rope_scale=2.0)  # 128 px tiles


def tiny(cfg=DEPTH, seed=0, **opts):
    return randomize_(build_model(dict(cfg, **opts)), seed).eval()


def image(h_tokens, w_tokens, batch=1, seed=0, crop=(0, 0)):
    """A normalised image of h x w tokens (16 px patches), ``crop`` pixels short of
    them at the bottom / right (the model pads)."""
    g = torch.Generator().manual_seed(seed)
    hw = (16 * h_tokens - crop[0], 16 * w_tokens - crop[1])
    return torch.randn(batch, 3, *hw, generator=g)


# ---- geometry


def test_default_setting():
    assert ENCODER_TILES == EncoderTiles() == EncoderTiles(192, 64, 2.0)
    assert not ENCODER_TILES.applies((192, 192))  # 3072 x 3072 px: one pass
    assert ENCODER_TILES.applies((193, 100)) and ENCODER_TILES.applies((100, 193))
    # tiles per axis (rows, columns) of the timing photos (H x W px): 102 MP 6 x 4,
    # 60 MP 3 x 5, 8 MP 2 x 1 (a 3264 px side, 204 tokens)
    sizes = {(11664, 8750): [6, 4], (6336, 9504): [3, 5], (3264, 2448): [2, 1]}
    for (H, W), counts in sizes.items():
        hw = (math.ceil(H / 16), math.ceil(W / 16))
        assert [len(tile_starts(n, 192, 64)) for n in hw] == counts


@pytest.mark.parametrize(
    "tile, overlap", [(192, 64), (192, 32), (8, 2), (9, 4), (6, 3)]
)
def test_tile_starts_cover_the_axis(tile, overlap):
    """Origins from 0 to n - tile, rising, neighbours overlapping by overlap tokens or
    more, k = ceil((n - overlap) / (tile - overlap)) of them, and as many as the
    mirrored axis has (each mirrored origin within one token, rounding)."""
    for n in range(1, 4 * tile + 7):
        starts = tile_starts(n, tile, overlap)
        if n <= tile:
            assert starts == [0]
            continue
        k = math.ceil((n - overlap) / (tile - overlap))
        assert len(starts) == k >= 2
        assert starts[0] == 0 and starts[-1] == n - tile
        assert all(b > a for a, b in zip(starts, starts[1:]))
        assert all(a + tile - b >= overlap for a, b in zip(starts, starts[1:]))
        # one tile fewer would leave less than the overlap somewhere
        assert (k - 2) * (tile - overlap) < n - tile
        mirrored = [n - tile - s for s in reversed(starts)]
        assert all(abs(a - b) <= 1 for a, b in zip(starts, mirrored))
        if (n - tile) % (k - 1) == 0:  # no rounding: exactly symmetric
            assert starts == mirrored


def test_tile_ramp():
    r = tile_ramp(10, True, True, 4)
    assert r.dtype == torch.float32
    assert r.tolist() == [0.125, 0.375, 0.625, 0.875] + [1.0] * 2 + [
        0.875,
        0.625,
        0.375,
        0.125,
    ]
    assert torch.equal(r, r.flip(0))  # symmetric
    assert torch.equal(
        tile_ramp(10, True, False, 4), tile_ramp(10, False, True, 4).flip(0)
    )
    assert torch.equal(tile_ramp(10, False, False, 4), torch.ones(10))
    # a falling and a rising ramp over the same tokens sum to 1
    up, down = tile_ramp(8, True, False, 4)[:4], tile_ramp(8, False, True, 4)[4:]
    assert torch.equal(up + down, torch.ones(4))


@pytest.mark.parametrize("hw", [(8, 8), (8, 21), (13, 9), (20, 30), (14, 14), (37, 8)])
def test_tile_weights_cover_the_grid(hw):
    """Every token is in a tile with a positive weight; the tiles lie inside the
    grid; weights are at most 1 and 1 on the image border; the summed weights are
    the same on the transposed grid, and 1 wherever two tiles overlap by exactly the
    ramp."""
    tiles = EncoderTiles(tile=8, overlap=3, rope_scale=1.0)
    h, w = hw
    total = torch.zeros(h, w)
    count = torch.zeros(h, w)
    windows = tile_windows(hw, tiles)
    for y0, x0, th, tw, win in windows:
        assert (th, tw) == (min(8, h), min(8, w)) and win.shape == (th, tw)
        assert 0 <= y0 <= h - th and 0 <= x0 <= w - tw
        assert (win > 0).all() and (win <= 1).all()
        if y0 == 0:
            assert torch.equal(win[0], tile_ramp(tw, x0 > 0, x0 + tw < w, 3))
        total[y0 : y0 + th, x0 : x0 + tw] += win
        count[y0 : y0 + th, x0 : x0 + tw] += 1
    assert (count >= 1).all() and (total > 0).all()
    assert len(windows) == len(tile_starts(h, 8, 3)) * len(tile_starts(w, 8, 3))
    transposed = torch.zeros(w, h)
    for y0, x0, th, tw, win in tile_windows((w, h), tiles):
        transposed[y0 : y0 + th, x0 : x0 + tw] += win
    torch.testing.assert_close(transposed, total.T)  # (another summation order)
    # 13 tokens = 8 + 8 - 3: one overlap of exactly 3 tokens, weights sum to 1
    exact = torch.zeros(13)
    for s in tile_starts(13, 8, 3):
        exact[s : s + 8] += tile_ramp(8, s > 0, s + 8 < 13, 3)
    assert torch.equal(exact, torch.ones(13))


def test_encoder_tiles_checks():
    for bad in (dict(overlap=97), dict(overlap=0), dict(tile=1, overlap=1)):
        with pytest.raises(ValueError, match="overlap"):
            EncoderTiles(**bad)
    for bad in (dict(tile=192.0), dict(overlap=True)):
        with pytest.raises(ValueError, match="must be an int"):
            EncoderTiles(**bad)
    for bad in (0, -1.0, math.inf, True):
        with pytest.raises(ValueError, match="rope_scale"):
            EncoderTiles(rope_scale=bad)
    assert EncoderTiles(tile=8, overlap=4).overlap == 4  # tile / 2 is allowed


# ---- the model


@torch.no_grad()
@pytest.mark.parametrize("cfg", [DEPTH, SEMSEG], ids=["depth", "semseg"])
@pytest.mark.parametrize(
    "hw, crop", [((8, 8), (0, 0)), ((8, 5), (3, 9)), ((6, 8), (0, 0)), ((3, 7), (5, 0))]
)
def test_no_op_at_or_below_the_tile(cfg, hw, crop):
    """A token grid with no side above the tile runs the plain forward bit for bit:
    no tile, no RoPE scale (also with the release setting on these small inputs)."""
    model = tiny(cfg)
    x = image(*hw, batch=2, crop=crop)
    plain = model.infer(x)
    for tiles in (SMALL, ENCODER_TILES, EncoderTiles(8, 4, 1.0)):
        assert torch.equal(model.infer(x, encoder_tiles=tiles), plain)
    assert torch.equal(model.infer(x, encoder_tiles=None), plain)
    assert torch.equal(model.infer(x, encoder_tiles=False), plain)
    assert model.encoder.rope_embed.eval_scale is None


@torch.no_grad()
@pytest.mark.parametrize("selection_impl", ["concat", "stream"])
@pytest.mark.parametrize("selection", ["stage", "shared"])
def test_one_tile_is_the_plain_path(selection_impl, selection):
    """The blend itself is exact: one tile (rope_scale 1.0) gives the plain
    selections bit for bit, and with rope_scale 2 those of the encoder run with its
    RoPE scale 2."""
    model = tiny(selection=selection, selection_impl=selection_impl)
    x, hw = model.pad(image(7, 8, batch=2, crop=(4, 0)))
    one = EncoderTiles(tile=8, overlap=2, rope_scale=1.0)
    assert len(tile_windows(hw, one)) == 1
    plain = model._selections(x)
    tiled = model.encode_select_tiled(x, one)
    assert len(tiled) == len(plain) == model.num_stages
    for a, b in zip(tiled, plain):
        assert a.dtype == b.dtype and torch.equal(a, b)
    assert torch.equal(model.decode(tiled, hw), model.decode(plain, hw))
    model.encoder.rope_embed.eval_scale = 2.0
    scaled = model._selections(x)
    model.encoder.rope_embed.eval_scale = None
    for a, b in zip(model.encode_select_tiled(x, SMALL), scaled):
        assert torch.equal(a, b)
    assert not torch.equal(scaled[0], plain[0])  # the scale does change the encoder
    assert model.encoder.rope_embed.eval_scale is None


def reference_tiled(model, x, tiles):
    """The tiled forward written out per token: each tile's selections (encoder RoPE
    scale set by hand) times its weight (products of 1-D ramps built here), summed,
    divided by the summed weight, then one decoder pass."""
    image, (h, w) = model.pad(x)
    B, P, T, O = x.shape[0], model.patch_size, tiles.tile, tiles.overlap

    def ramp(n, lo, hi):
        return [
            min(
                (i + 0.5) / O if lo else 1.0,
                (n - i - 0.5) / O if hi else 1.0,
                1.0,
            )
            for i in range(n)
        ]

    def origins(n):
        if n <= T:
            return [0]
        k = math.ceil((n - O) / (T - O))
        return [round(i * (n - T) / (k - 1)) for i in range(k)]

    sums = [torch.zeros(B, h, w, model.dec_dim) for _ in range(model.num_stages)]
    weight = torch.zeros(h, w)
    model.encoder.rope_embed.eval_scale = tiles.rope_scale
    try:
        for y0 in origins(h):
            for x0 in origins(w):
                th, tw = min(T, h), min(T, w)
                crop = image[..., y0 * P : (y0 + th) * P, x0 * P : (x0 + tw) * P]
                sel = model._selections(crop)
                wy = torch.tensor(ramp(th, y0 > 0, y0 + th < h))
                wx = torch.tensor(ramp(tw, x0 > 0, x0 + tw < w))
                win = wy[:, None] * wx[None, :]
                for s, t in enumerate(sel):
                    sums[s][:, y0 : y0 + th, x0 : x0 + tw] += (
                        t.float().reshape(B, th, tw, -1) * win[None, :, :, None]
                    )
                weight[y0 : y0 + th, x0 : x0 + tw] += win
    finally:
        model.encoder.rope_embed.eval_scale = None
    selections = [
        (t / weight[None, :, :, None]).reshape(B, h * w, -1).to(sel[0].dtype)
        for t in sums
    ]
    pred = model.decode(selections, (h, w))[:, :, : x.shape[-2], : x.shape[-1]]
    return torch.exp(pred.clamp(-30, 30))


@torch.no_grad()
@pytest.mark.parametrize("hw, crop", [((16, 21), (0, 0)), ((9, 30), (7, 5))])
def test_tiled_forward_small_tile(hw, crop, monkeypatch):
    """A tile of 8 tokens forces the tiled path on a tiny model: the encoder runs on
    each tile with RoPE scale 2 (restored after), the decoder once on the whole grid,
    and the output is the per-token reference's; batch items stay independent."""
    model = tiny()
    x = image(*hw, batch=2, crop=crop)
    rope = model.encoder.rope_embed
    seen = []
    forward = rope.forward

    def spy(**kwargs):
        seen.append((kwargs["H"], kwargs["W"], rope.eval_scale))
        return forward(**kwargs)

    monkeypatch.setattr(rope, "forward", spy)
    tiled = model.infer(x, encoder_tiles=SMALL)
    monkeypatch.setattr(rope, "forward", forward)
    h, w = hw
    n_tiles = len(tile_starts(h, 8, 2)) * len(tile_starts(w, 8, 2))
    assert n_tiles > 1 and len(seen) == n_tiles  # one RoPE table per tile ...
    assert {s[2] for s in seen} == {2.0} and rope.eval_scale is None  # ... scaled
    assert {s[:2] for s in seen} == {(min(8, h), min(8, w))}
    assert tiled.shape == (2, 1) + x.shape[-2:] and torch.isfinite(tiled).all()
    plain = model.infer(x)
    assert (tiled - plain).abs().max() > 1e-3  # another encoder input
    torch.testing.assert_close(
        tiled, reference_tiled(model, x, SMALL), rtol=1e-5, atol=0
    )
    for b in range(2):
        alone = model.infer(x[b : b + 1], encoder_tiles=SMALL)
        torch.testing.assert_close(alone, tiled[b : b + 1], rtol=1e-5, atol=0)


@torch.no_grad()
def test_tiled_forward_options():
    """stream and concat selections, and a shared selection, agree on the tiled path;
    a user's encoder RoPE scale is restored; training mode and an encoder without
    RoPE refuse a RoPE scale."""
    x = image(11, 19, crop=(3, 3))
    concat = tiny().infer(x, encoder_tiles=SMALL)
    stream = tiny(selection_impl="stream").infer(x, encoder_tiles=SMALL)
    torch.testing.assert_close(stream, concat, rtol=1e-5, atol=1e-5)
    shared = tiny(selection="shared")
    out = shared.infer(x, encoder_tiles=SMALL)
    assert out.shape == concat.shape and torch.isfinite(out).all()
    model = tiny()
    model.encoder.rope_embed.eval_scale = 1.5
    model.infer(x, encoder_tiles=SMALL)
    assert model.encoder.rope_embed.eval_scale == 1.5
    with pytest.raises(ValueError, match="model.eval"):
        model.train().encode_select_tiled(model.pad(x)[0], SMALL)
    stub = dict(
        BASE,
        encoder=dict(type="feature_stub_test", embed_dim=32),
        criterion=False,
        head="exp",
    )
    stub_model = randomize_decoder_(build_model(stub)).eval()
    with pytest.raises(ValueError, match="rope_embed"):
        stub_model.infer(x, encoder_tiles=SMALL)
    out = stub_model.infer(x, encoder_tiles=EncoderTiles(8, 2, rope_scale=1.0))
    assert out.shape == concat.shape
    with pytest.raises(TypeError, match="encoder_tiles"):
        model.eval().infer(x, encoder_tiles="auto")  # the API's word, not the model's


@torch.no_grad()
def test_tiled_semseg():
    model = tiny(SEMSEG)
    x = image(10, 17)
    out = model.infer(x, encoder_tiles=SMALL)
    assert out.shape == (1, 5) + x.shape[-2:] and torch.isfinite(out).all()
