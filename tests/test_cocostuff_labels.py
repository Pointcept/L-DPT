"""
COCO-Stuff label mapping (CPU, no data): label_mapping="official" maps the official
0-based stuffthingmaps PNG values to the standard 171 train ids (mmseg clsID_to_trID),
"legacy" is the mapping that reads the thing ids as 1-based (COCOStuffDataset), and
the class names are in the official train-id order.
"""

import json

import cv2
import numpy as np
import pytest

from ldpt.datasets.builder import build_dataset
from ldpt.datasets.cocostuff import COCOStuffDataset

# PNG label values present in the official COCO-Stuff stuffthingmaps (train)
LABEL_SET = (
    set(range(0, 11))
    | set(range(12, 25))
    | {26, 27}
    | set(range(30, 44))
    | set(range(45, 65))
    | {66, 69}
    | set(range(71, 82))
    | set(range(83, 90))
    | set(range(91, 182))
    | {255}
)


def old_mapping():
    """The "legacy" mapping, written out independently of ldpt/datasets/cocostuff.py."""
    # fmt: off
    things_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 66, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 83, 84, 85, 86, 87, 88, 89, 91
    ]
    # fmt: on
    stuff_ids = list(range(92, 183))
    mapping = np.full(256, -1, dtype=np.int32)
    for train_id, raw_id in enumerate(things_ids + stuff_ids):
        mapping[raw_id] = train_id
    return mapping


def test_official_mapping():
    mapping = COCOStuffDataset.create_label_mapping("official")
    assert mapping.shape == (256,) and mapping.dtype == np.int32
    # every class value of the data, in increasing order, is the next train id
    values = sorted(LABEL_SET - {255})
    assert len(values) == 171
    for train_id, value in enumerate(values):
        assert mapping[value] == train_id, value
    assert mapping[0] == 0  # person
    assert mapping[89] == 79  # toothbrush, the last thing
    assert mapping[91] == 80  # banner, the first stuff class
    assert mapping[181] == 170  # wood
    # 255 (unlabeled) and every value outside the label set are ignored
    for value in set(range(256)) - LABEL_SET | {255}:
        assert mapping[value] == -1, value
    assert sorted(set(mapping[sorted(LABEL_SET)].tolist())) == list(range(-1, 171))
    # the default
    assert np.array_equal(COCOStuffDataset.create_label_mapping(), mapping)


def test_legacy_mapping():
    legacy = COCOStuffDataset.create_label_mapping("legacy")
    assert np.array_equal(legacy, old_mapping())
    # what it did to the data: person ignored, ids one class off, train id 170 empty
    assert legacy[0] == -1 and legacy[1] == 0 and legacy[91] == 79
    assert legacy[181] == 169 and legacy[182] == 170 and 182 not in LABEL_SET
    official = COCOStuffDataset.create_label_mapping("official")
    assert all(legacy[v] == official[v] - 1 for v in LABEL_SET - {0, 255})


def test_unknown_mapping():
    with pytest.raises(ValueError, match="label_mapping"):
        COCOStuffDataset.create_label_mapping("1-based")
    with pytest.raises(ValueError, match="label_mapping"):
        COCOStuffDataset(data_root="missing", split="x.json", label_mapping="mmseg")


def test_class_names_official_order():
    names = COCOStuffDataset.CLASS_NAMES
    assert len(names) == 171 and len(set(names)) == 171
    # train id = rank of the PNG value (labels.txt id - 1): things, then stuff
    expected = {
        0: "person",  # PNG 0
        10: "fire hydrant",  # PNG 10
        11: "stop sign",  # PNG 12 (PNG 11 "street sign" is unused)
        24: "backpack",  # PNG 26
        56: "chair",  # PNG 61
        62: "tv",  # PNG 71
        79: "toothbrush",  # PNG 89
        80: "banner",  # PNG 91
        81: "blanket",  # PNG 92
        94: "clouds",  # PNG 105
        167: "waterdrops",  # PNG 178
        170: "wood",  # PNG 181
    }
    for train_id, name in expected.items():
        assert names[train_id] == name, train_id


def test_dataset_maps_png(tmp_path):
    """get_data reads the PNG and applies the mapping of the config."""
    values = np.array([[0, 1, 11, 91], [181, 182, 255, 12]], dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), values)
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((2, 4, 3), dtype=np.uint8))
    (tmp_path / "split.json").write_text(
        json.dumps([dict(image="a.jpg", segment="a.png")])
    )
    common = dict(type="COCOStuffDataset", data_root=str(tmp_path), split="split.json")
    official = build_dataset(dict(common)).get_data(0)["segment"]
    assert official.tolist() == [[0, 1, -1, 80], [170, -1, -1, 11]]
    legacy = build_dataset(dict(common, label_mapping="legacy")).get_data(0)
    assert legacy["segment"].tolist() == [[-1, 0, -1, 79], [169, 170, -1, 10]]
    assert legacy["name"] == "cocostuff/a"
