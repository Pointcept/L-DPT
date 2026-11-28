"""Edge-aware depth resampling in the train Resize (docs/depth_loss.md 4.6).

Resize(depth_resample="nearest") is the default (torchvision resizes the depth
Mask by nearest); "edge_aware" resizes the depth Mask by resize_depth_edge_aware:
bilinear in inverse depth where the 5 x 5 source windows are valid and edge free, else
the nearest source pixel (nearest-exact).
"""

import copy

import pytest
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

from ldpt.datasets.transforms import (
    TRANSFORMS,
    Compose,
    Resize,
    depth_edge_free,
    resize_depth_edge_aware,
)

H, W = 48, 64


def grid(h=H, w=W):
    return torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )


def plane(h=H, w=W):
    """A slanted plane: inverse depth linear in the pixel coordinates. [1, h, w]"""
    yy, xx = grid(h, w)
    return (1.0 / (0.2 + 0.004 * xx + 0.002 * yy))[None]


def sample(depth, seed=0, segment=True):
    """A data dict as the train transform sees it before Resize."""
    g = torch.Generator().manual_seed(seed)
    _, h, w = depth.shape
    data = dict(
        image=tv_tensors.Image(torch.randn(3, h, w, generator=g)),
        depth=tv_tensors.Mask(depth.clone()),
        name="hypersim/scene/0",
    )
    if segment:
        data["segment"] = tv_tensors.Mask(torch.randint(0, 5, (1, h, w), generator=g))
    return data


def nearest_exact(depth, size):
    return F.interpolate(depth[None], size=size, mode="nearest-exact")[0]


def test_nearest_bit_identical_to_today():
    """depth_resample="nearest" (and the default) is the plain torchvision transform
    of every target; the dict keeps its key order."""
    depth = plane()
    depth[:, 10:20, 30:40] = 0.0
    for kwargs in (dict(size=96), dict(size=37), dict(size=(50, 70))):
        today = T.Resize(interpolation=T.InterpolationMode.BICUBIC, **kwargs)
        data = sample(depth)
        want = dict(data)
        keys = ["image", "depth", "segment"]
        for k, v in zip(keys, today([data[k] for k in keys])):
            want[k] = v
        for options in ({}, dict(depth_resample="nearest")):
            got = Resize(interpolation="bicubic", **kwargs, **options)(sample(depth))
            assert list(got) == list(want)
            for k in keys:
                assert type(got[k]) is type(want[k]) and torch.equal(got[k], want[k])
            assert got["name"] == want["name"]


def test_plane_upsampled_within_1e5():
    """A plane in inverse depth upsampled x 2 is reproduced in the interior to 1e-5
    (bilinear in inverse depth is exact on it; nearest is not)."""
    data = Resize(size=2 * H, depth_resample="edge_aware")(sample(plane()))
    out = data["depth"]
    assert isinstance(out, tv_tensors.Mask) and out.dtype == torch.float32
    assert out.shape == (1, 2 * H, 2 * W) == (1, *data["image"].shape[-2:])
    yy, xx = grid(2 * H, 2 * W)
    y, x = (yy + 0.5) / 2 - 0.5, (xx + 0.5) / 2 - 0.5  # source coordinates
    exact = 1.0 / (0.2 + 0.004 * x + 0.002 * y)
    inner = (slice(8, -8), slice(8, -8))
    err = ((out[0] - exact).abs() / exact)[inner].max()
    assert err < 1e-5
    near = nearest_exact(plane(), (2 * H, 2 * W))[0]
    assert ((near - exact).abs() / exact)[inner].max() > 100 * err


@pytest.mark.parametrize("scale", [2.0, 1.33, 0.6])
def test_step_edge_takes_source_values(scale):
    """Across a step edge (2 m | 6 m) every output is a source value: nothing is
    interpolated across the edge (inside a flat side the inverse-depth interpolation
    may round by an ulp)."""
    _, xx = grid()
    step = torch.where(xx < W / 2, 2.0, 6.0)[None]
    size = (int(H * scale), int(W * scale))
    out = resize_depth_edge_aware(step, size)
    assert out.shape == (1, *size)
    near_2 = (out - 2.0).abs() <= 2.0 * 1e-6
    near_6 = (out - 6.0).abs() <= 6.0 * 1e-6
    assert (near_2 | near_6).all() and near_2.any() and near_6.any()
    # the two output columns at the step are exactly nearest
    near = nearest_exact(step, size)
    step_at = near[..., 1:] != near[..., :-1]
    edge = torch.zeros_like(near, dtype=torch.bool)
    edge[..., 1:] |= step_at
    edge[..., :-1] |= step_at
    assert edge.any() and torch.equal(out[edge], near[edge])


def test_no_interpolation_across_holes():
    """Invalid pixels (0) stay 0 and are never mixed into a valid output: every output
    whose bilinear footprint touches an invalid source pixel is the nearest-exact
    source value; the hole has 4 x its area of zeros after a x 2 upsampling."""
    depth = plane()
    depth[:, 15:25, 20:35] = 0.0
    size = (2 * H, 2 * W)
    out = resize_depth_edge_aware(depth, size)
    near = nearest_exact(depth, size)
    assert int((out == 0).sum()) == 10 * 15 * 4
    assert torch.equal(out == 0, near == 0)
    touches = (
        F.interpolate(
            (depth == 0).float()[None], size=size, mode="bilinear", align_corners=False
        )[0]
        > 0
    )
    assert touches.sum() > 0 and torch.equal(out[touches], near[touches])
    valid = out > 0
    assert out[valid].min() >= depth[depth > 0].min() * (1 - 1e-6)
    assert out[valid].max() <= depth.max() * (1 + 1e-6)
    # negative values (RandomCrop's padding) are invalid too and come out as 0
    padded = depth.clone()
    padded[:, :, -6:] = -1.0
    out = resize_depth_edge_aware(padded, size)
    assert torch.equal(out == 0, nearest_exact(padded, size) <= 0)
    assert (out >= 0).all()


def test_sparse_map_is_nearest_exact():
    """A 5 % sparse map (LiDAR) has no fully valid 5 x 5 window: nearest-exact
    everywhere, also from the transform."""
    g = torch.Generator().manual_seed(0)
    sparse = torch.where(torch.rand(1, H, W, generator=g) < 0.05, plane(), 0.0)
    assert not depth_edge_free(sparse).any()
    for size in ((2 * H, 2 * W), (int(1.5 * H), int(1.5 * W)), (H // 2, W // 2)):
        assert torch.equal(
            resize_depth_edge_aware(sparse, size), nearest_exact(sparse, size)
        )
    data = Resize(size=2 * H, depth_resample="edge_aware")(sample(sparse))
    assert torch.equal(data["depth"], nearest_exact(sparse, (2 * H, 2 * W)))


def test_edge_free_rule():
    """depth_edge_free: a plane is free inside (the 2-px border is not: its window
    leaves the image); a step marks a band around the edge; a hole marks its 5 x 5
    neighbourhood."""
    ok = depth_edge_free(plane())
    assert ok.shape == (1, H, W) and ok.dtype == torch.bool
    assert ok[:, 2:-2, 2:-2].all() and not ok[:, :2].any() and not ok[:, :, -2:].any()
    _, xx = grid()
    step = torch.where(xx < W / 2, 2.0, 6.0)[None]
    ok = depth_edge_free(step)
    assert not ok[:, 2:-2, W // 2 - 2 : W // 2 + 2].any()
    assert ok[:, 2:-2, 2 : W // 2 - 4].all() and ok[:, 2:-2, W // 2 + 4 : -2].all()
    holed = plane()
    holed[:, 20, 30] = 0.0
    ok = depth_edge_free(holed)
    assert not ok[:, 18:23, 28:33].any() and ok[:, 2:17, 2:-2].all()


@pytest.mark.parametrize("size", [96, 37, (50, 70), 2 * W])
def test_edge_aware_transform_sizes_and_other_targets(size):
    """edge_aware: the image and every other Mask as with "nearest" (bitwise); the
    depth Mask has the image's new size; key order kept; the name untouched."""
    depth = plane()
    depth[:, 5:9, 10:14] = 0.0
    want = Resize(size=size, interpolation="bicubic")(sample(depth))
    got = Resize(size=size, interpolation="bicubic", depth_resample="edge_aware")(
        sample(depth)
    )
    assert list(got) == list(want)
    assert torch.equal(got["image"], want["image"])
    assert torch.equal(got["segment"], want["segment"])
    assert isinstance(got["segment"], tv_tensors.Mask)
    assert (
        got["depth"].shape[-2:] == got["image"].shape[-2:] == want["depth"].shape[-2:]
    )
    assert torch.equal(
        got["depth"], resize_depth_edge_aware(depth, got["image"].shape[-2:])
    )
    assert got["name"] == "hypersim/scene/0"


def test_edge_aware_only_on_a_target_depth_mask():
    """The depth Mask goes through resize_depth_edge_aware only when it is a target:
    with target_keys=["image"] (the val transform's Update) it is left unchanged;
    without an image the depth's own torchvision size is used."""
    depth = plane()
    data = sample(depth, segment=False)
    data["target_keys"] = ["image"]
    got = Resize(size=96, depth_resample="edge_aware")(data)
    assert torch.equal(got["depth"], depth) and got["image"].shape[-2:] == (96, 128)
    only = dict(depth=tv_tensors.Mask(depth.clone()))
    got = Resize(size=37, depth_resample="edge_aware")(only)
    want = T.Resize(size=37)(tv_tensors.Mask(depth.clone()))
    assert got["depth"].shape == want.shape
    assert torch.equal(got["depth"], resize_depth_edge_aware(depth, want.shape[-2:]))


def test_option_from_config_and_validation():
    """The option builds from a config (inside RandomChoice as the train list has it)
    and an unknown value raises."""
    cfg = [
        dict(
            type="RandomChoice",
            transforms=[
                dict(
                    type="Resize",
                    size=96,
                    interpolation="bicubic",
                    depth_resample="edge_aware",
                )
            ],
        )
    ]
    data = Compose(cfg)(sample(plane()))
    assert torch.equal(data["depth"], resize_depth_edge_aware(plane(), (96, 128)))
    assert TRANSFORMS.build(dict(type="Resize", size=8)).depth_resample == "nearest"
    with pytest.raises(ValueError, match="depth_resample"):
        Resize(size=96, depth_resample="bilinear")


def test_deepcopy_and_input_untouched():
    """The input depth is not modified in place (MultiAugment copies dicts)."""
    depth = plane()
    data = sample(depth)
    before = copy.deepcopy(data["depth"])
    resize_depth_edge_aware(data["depth"], (96, 128))
    assert torch.equal(data["depth"], before)
