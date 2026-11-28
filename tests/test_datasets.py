"""DefaultDataset split loading (CPU, no data)."""

import json

import pytest

from ldpt.datasets.default import DefaultDataset


def test_split_files(tmp_path):
    (tmp_path / "splits").mkdir()
    for name, items in [("a.json", [["x.png"]]), ("b.json", [["y.png"], ["z.png"]])]:
        (tmp_path / "splits" / name).write_text(json.dumps(items))
    ds = DefaultDataset(tmp_path, ["splits/a.json", "splits/b.json"])
    assert ds.data_list == [["x.png"], ["y.png"], ["z.png"]]
    with pytest.raises(FileNotFoundError, match="missing.json"):
        DefaultDataset(tmp_path, ["splits/a.json", "splits/missing.json"])


def write_hypersim_frame(root, semantic, scene="ai_000_000", camera="00", frame=0):
    """One frame in the ml-hypersim layout: tonemap.jpg, depth_meters.hdf5 (distance)
    and semantic.hdf5 (NYU-40 ids, -1 unlabelled)."""
    import cv2
    import h5py
    import numpy as np

    images = root / "scenes" / scene / "images"
    preview = images / f"scene_cam_{camera}_final_preview"
    geometry = images / f"scene_cam_{camera}_geometry_hdf5"
    preview.mkdir(parents=True)
    geometry.mkdir(parents=True)
    h, w = semantic.shape
    rng = np.random.default_rng(0)
    cv2.imwrite(
        str(preview / f"frame.{frame:04d}.tonemap.jpg"),
        rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
    )
    with h5py.File(geometry / f"frame.{frame:04d}.depth_meters.hdf5", "w") as f:
        f["dataset"] = rng.uniform(0.5, 10, (h, w)).astype(np.float16)
    with h5py.File(geometry / f"frame.{frame:04d}.semantic.hdf5", "w") as f:
        f["dataset"] = semantic.astype(np.int16)
    rel = f"scenes/{scene}/images/scene_cam_{camera}"
    return dict(
        image=f"{rel}_final_preview/frame.{frame:04d}.tonemap.jpg",
        depth=f"{rel}_geometry_hdf5/frame.{frame:04d}.depth_meters.hdf5",
        segment=f"{rel}_geometry_hdf5/frame.{frame:04d}.semantic.hdf5",
    )


def test_hypersim_semseg(tmp_path):
    """HypersimDataset on a synthetic frame through the segmentation recipe's
    transforms: NYU-40 ids 1..37 -> train ids 0..36 (HYPERSIM_CLASS_NAMES), -1, 0 and
    38..40 (otherstructure / otherfurniture / otherprop) ignored."""
    import numpy as np
    import torch

    from helpers import REPO
    from ldpt.datasets.builder import build_dataset
    from ldpt.datasets.hypersim import HypersimDataset
    from ldpt.utils.config import Config

    values = np.arange(-1, 41)  # every value of semantic.hdf5
    semantic = np.resize(values, (48, 64))
    entry = write_hypersim_frame(tmp_path, semantic)
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "train.json").write_text(json.dumps([entry]))

    assert len(HypersimDataset.CLASS_NAMES) == 37
    assert HypersimDataset.CLASS_NAMES[0] == "wall"  # NYU-40 id 1
    assert HypersimDataset.CLASS_NAMES[36] == "bag"  # NYU-40 id 37
    # the segmentation recipe's transforms on Hypersim's 37 classes
    cfg = Config.fromfile(f"{REPO}/configs/cocostuff/semseg-l-dpt-v1m1-1-large.py")
    for split in ("train", "val"):
        ds_cfg = dict(
            type="HypersimDataset",
            data_root=str(tmp_path),
            split="splits/train.json",
            transform=cfg[f"{split}_transform"],
        )
        ds = build_dataset(ds_cfg)
        assert len(ds) == 1
        raw = ds.get_data(0)
        assert raw["name"] == "hypersim/ai_000_000-cam_00-frame_0000"
        expected = np.where((semantic >= 1) & (semantic <= 37), semantic - 1, -1)
        assert np.array_equal(raw["segment"], expected)
        sample = ds[0]
        assert set(sample) == {"image", "segment", "name"}
        segment = torch.as_tensor(sample["segment"])
        assert set(segment.unique().tolist()) <= set(range(-1, 37))
        if split == "train":  # random scale, 768 crop (padded with ignore)
            assert sample["image"].shape == (3, 768, 768)
            assert segment.shape[-2:] == (768, 768)
        else:  # image to short side 768, labels at their size
            assert sample["image"].shape == (3, 768, 1024)
            assert segment.shape[-2:] == (48, 64)
            assert torch.equal(segment.reshape(48, 64), torch.as_tensor(expected))


# ------------------------------------------------ evaluation protocols (synthetic)


def window(shape, rows, cols):
    """Bool [H, W], True on rows x cols (slices)."""
    import numpy as np

    out = np.zeros(shape, dtype=bool)
    out[rows, cols] = True
    return out


def write_split(root, items):
    (root / "splits").mkdir(exist_ok=True)
    (root / "splits" / "test.json").write_text(json.dumps(items))
    return "splits/test.json"


def test_parse_depth_range():
    import math

    from ldpt.datasets.default import parse_depth_range

    assert parse_depth_range(None) is None
    assert parse_depth_range("None") is None and parse_depth_range("null") is None
    assert parse_depth_range((1e-3, 10)) == (1e-3, 10.0)
    assert parse_depth_range([1e-5, None]) == (1e-5, math.inf)
    assert parse_depth_range((1e-5, float("inf"))) == (1e-5, math.inf)
    for bad in ((10, 1), (-1, 1), (1,), "1,2", (1, 1)):
        with pytest.raises(ValueError, match="depth_range"):
            parse_depth_range(bad)


def test_nyuv2_masks(tmp_path):
    """Marigold's NYUv2 mask (eigen_crop=True, depth_range=(1e-3, 10)): rows 45:471,
    cols 41:601 and 1e-3 < d < 10 (the raw GT's 10.0 m pixels dropped); the mixed
    model's training labels (eigen_crop="legacy"): rows 159:438, cols 23:616 and
    1e-6 <= d <= 10."""
    import cv2
    import numpy as np

    from ldpt.datasets.nyuv2 import NYUv2Dataset

    (tmp_path / "test" / "kitchen").mkdir(parents=True)
    cv2.imwrite(
        str(tmp_path / "test/kitchen/rgb_00001.jpg"), np.zeros((480, 640, 3), np.uint8)
    )
    mm = np.full((480, 640), 5000, np.uint16)  # 5 m
    mm[300, :] = 10000  # 10.0 m, the raw GT's cap
    mm[301, :] = 1  # 1e-3 m
    mm[302, :] = 0  # no depth
    cv2.imwrite(str(tmp_path / "test/kitchen/sync_depth_00001.png"), mm)
    split = write_split(
        tmp_path,
        [
            dict(
                image="test/kitchen/rgb_00001.jpg",
                depth="test/kitchen/sync_depth_00001.png",
            )
        ],
    )
    ds = NYUv2Dataset(tmp_path, split, eigen_crop=True, depth_range=(1e-3, 10.0))
    data = ds.get_data(0)
    expected = window((480, 640), slice(45, 471), slice(41, 601))
    expected[300:303] = False
    assert np.array_equal(data["depth"] > 0, expected)
    assert expected.sum() == (426 - 3) * 560
    assert (data["depth_min"], data["depth_max"]) == (1e-3, 10.0)
    assert ds.crop_protocol() == "eigen-nyu" and ds.range_protocol() == "(0.001,10)"

    legacy = NYUv2Dataset(tmp_path, split, eigen_crop="legacy").get_data(0)
    expected = window((480, 640), slice(159, 438), slice(23, 616))
    expected[302] = False  # 10.0 m and 1e-3 m are kept
    assert np.array_equal(legacy["depth"] > 0, expected)
    assert (legacy["depth_min"], legacy["depth_max"]) == (1e-6, 10.0)


def test_kitti_masks(tmp_path):
    """Marigold's KITTI mask (kb_crop, eigen_crop, depth_range=(1e-5, 80)): the
    352 x 1216 benchmark crop (image and GT), its rows 117:321, cols 43:1172, and
    d < 80 (80.0 m dropped); the mixed model's training labels: kb + Garg rows
    143:349, 1e-6 <= d <= 80."""
    import cv2
    import numpy as np

    from ldpt.datasets.kitti import KITTIDataset

    rel_img = "image/2011_09_26/drive_0002_sync/image_02/data/0000000069.png"
    rel_gt = "depth/drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png"
    for rel in (rel_img, rel_gt):
        (tmp_path / rel).parent.mkdir(parents=True)
    cv2.imwrite(str(tmp_path / rel_img), np.zeros((375, 1242, 3), np.uint8))
    gt = np.full((375, 1242), 20 * 256, np.uint16)  # 20 m
    gt[23 + 200, :] = 80 * 256  # 80.0 m (row 200 of the crop)
    cv2.imwrite(str(tmp_path / rel_gt), gt)
    split = write_split(tmp_path, [dict(image=rel_img, depth=rel_gt)])
    ds = KITTIDataset(
        tmp_path, split, kb_crop=True, eigen_crop=True, depth_range=(1e-5, 80.0)
    )
    data = ds.get_data(0)
    assert data["image"].shape == (352, 1216, 3)
    expected = window((352, 1216), slice(117, 321), slice(43, 1172))
    expected[200] = False
    assert np.array_equal(data["depth"] > 0, expected)
    assert (data["depth_min"], data["depth_max"]) == (1e-5, 80.0)
    assert ds.crop_protocol() == "kb+eigen"

    train = KITTIDataset(tmp_path, split, kb_crop=True, garg_crop=True).get_data(0)
    expected = window((352, 1216), slice(143, 349), slice(43, 1172))
    assert np.array_equal(train["depth"] > 0, expected)  # 80.0 m kept


def test_eth3d_range(tmp_path):
    """Marigold's ETH3D mask (depth_range=(1e-5, None)): the whole frame, d > 1e-5,
    no upper bound (inf = no depth); the class range (default): 0.1 <= d <= 200."""
    import math

    import cv2
    import numpy as np

    from ldpt.datasets.eth3d import ETH3DDataset

    (tmp_path / "courtyard" / "images").mkdir(parents=True)
    (tmp_path / "courtyard" / "depth").mkdir(parents=True)
    cv2.imwrite(
        str(tmp_path / "courtyard/images/a.jpg"), np.zeros((40, 60, 3), np.uint8)
    )
    depth = np.full((40, 60), 5.0, np.float32)
    depth[0] = 300.0
    depth[1] = 0.05
    depth[2] = np.inf
    depth[3] = 0.0
    depth.tofile(tmp_path / "courtyard/depth/a.bin")
    split = write_split(
        tmp_path, [dict(image="courtyard/images/a.jpg", depth="courtyard/depth/a.bin")]
    )
    ds = ETH3DDataset(tmp_path, split, depth_range=(1e-5, None))
    data = ds.get_data(0)
    expected = np.ones((40, 60), bool)
    expected[2:4] = False
    assert np.array_equal(data["depth"] > 0, expected)
    assert data["depth_min"] == 1e-5 and data["depth_max"] == math.inf
    assert ds.range_protocol() == "(1e-05,inf)"
    default = ETH3DDataset(tmp_path, split).get_data(0)
    expected[0:2] = False
    assert np.array_equal(default["depth"] > 0, expected)
    assert (default["depth_min"], default["depth_max"]) == (0.1, 200.0)


@pytest.mark.parametrize("scene", ["indoors", "outdoor"])
def test_diode_masks(tmp_path, scene):
    """Marigold's DIODE protocol (depth_range=(0.6, 350), mask_only, no crop): every
    pixel of DIODE's depth mask, whatever its depth (0.5 m and 400 m kept), the range
    only in depth_min / depth_max (the prediction's clip). The mixed model's training
    labels: indoor the Eigen ratios (rows 72:753, cols 65:961 of 768 x 1024) and
    d <= 10 m, outdoor the Garg ratios (rows 313:761, cols 36:987) and d <= 80 m."""
    import cv2
    import numpy as np

    from ldpt.datasets.diode import DIODEIndoorDataset, DIODEOutdoorDataset

    cls = DIODEIndoorDataset if scene == "indoors" else DIODEOutdoorDataset
    rel = f"val/{scene}/scene_00019/scan_00183/00019_00183_{scene}_000_010"
    (tmp_path / rel).parent.mkdir(parents=True)
    cv2.imwrite(str(tmp_path / f"{rel}.png"), np.zeros((768, 1024, 3), np.uint8))
    depth = np.full((768, 1024, 1), 5.0, np.float32)
    depth[400, :] = 0.5
    depth[401, :] = 400.0
    depth[402, :] = 50.0
    mask = np.ones((768, 1024), np.float32)
    mask[500, :] = 0
    np.save(tmp_path / f"{rel}_depth.npy", depth)
    np.save(tmp_path / f"{rel}_depth_mask.npy", mask)
    split = write_split(tmp_path, [dict(image=f"{rel}.png", depth=f"{rel}_depth.npy")])
    ds = cls(tmp_path, split, depth_range=(0.6, 350.0), mask_only=True)
    data = ds.get_data(0)
    expected = np.ones((768, 1024), bool)
    expected[500] = False
    assert np.array_equal(data["depth"] > 0, expected)
    assert (data["depth_min"], data["depth_max"]) == (0.6, 350.0)
    assert ds.crop_protocol() == "none"
    assert ds.range_protocol() == "mask,clip[0.6,350]"
    assert data["name"].startswith(
        "DIODE-INDOOR/" if scene == "indoors" else "DIODE-OUTDOOR/"
    )

    if scene == "indoors":
        train = cls(tmp_path, split, eigen_crop=True).get_data(0)
        expected = window((768, 1024), slice(72, 753), slice(65, 961))
        expected[[401, 402, 500]] = False  # beyond 10 m; outside the depth mask
        assert train["depth_max"] == 10.0
    else:
        train = cls(tmp_path, split, garg_crop=True).get_data(0)
        expected = window((768, 1024), slice(313, 761), slice(36, 987))
        expected[[401, 500]] = False  # beyond 80 m; outside the depth mask
        assert train["depth_max"] == 80.0
    assert np.array_equal(train["depth"] > 0, expected)
