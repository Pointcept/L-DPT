"""
VOC head-row init (CPU): CheckpointLoader(seg_head_rows=...) initialises the 21-class
VOC seg_head from the COCO-Stuff seg_head of a loaded checkpoint: each VOC object class
from the row of its COCO-Stuff class, background from the mean of the other 151 rows.
Off (default): the loader is unchanged (random head, the COCO-Stuff head unused).
"""

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ldpt.datasets.meta import COCOSTUFF_CLASS_NAMES, VOC2012_CLASS_NAMES
from ldpt.hooks.misc import (
    VOC2012_FROM_COCOSTUFF,
    CheckpointLoader,
    seg_head_from_rows,
)

# VOC train id -> COCO-Stuff train id (official mapping), written out independently
# of the name table: the standard COCO 80-class order
VOC_TO_COCO_IDS = {
    1: 4,  # aeroplane / airplane
    2: 1,  # bicycle
    3: 14,  # bird
    4: 8,  # boat
    5: 39,  # bottle
    6: 5,  # bus
    7: 2,  # car
    8: 15,  # cat
    9: 56,  # chair
    10: 19,  # cow
    11: 60,  # dining table
    12: 16,  # dog
    13: 17,  # horse
    14: 3,  # motorbike / motorcycle
    15: 0,  # person
    16: 58,  # potted plant
    17: 18,  # sheep
    18: 57,  # sofa / couch
    19: 6,  # train
    20: 62,  # tv monitor / tv
}


def test_class_map():
    assert len(VOC2012_CLASS_NAMES) == 21 and len(COCOSTUFF_CLASS_NAMES) == 171
    assert set(VOC2012_FROM_COCOSTUFF) == set(VOC2012_CLASS_NAMES[1:])
    for voc, coco in VOC_TO_COCO_IDS.items():
        name = VOC2012_FROM_COCOSTUFF[VOC2012_CLASS_NAMES[voc]]
        assert COCOSTUFF_CLASS_NAMES.index(name) == coco, name


def test_rows():
    g = torch.Generator().manual_seed(0)
    weight = torch.randn(171, 8, 1, 1, generator=g)
    bias = torch.randn(171, generator=g)
    names = (VOC2012_CLASS_NAMES, COCOSTUFF_CLASS_NAMES, VOC2012_FROM_COCOSTUFF)
    others = sorted(set(range(171)) - set(VOC_TO_COCO_IDS.values()))
    assert len(others) == 151
    for source in (weight, bias):
        rows = seg_head_from_rows(source, *names)
        assert rows.shape == (21, *source.shape[1:])
        for voc, coco in VOC_TO_COCO_IDS.items():
            assert torch.equal(rows[voc], source[coco])  # copied
        torch.testing.assert_close(rows[0], source[others].mean(0))  # background
    with pytest.raises(ValueError, match="rows"):
        seg_head_from_rows(weight[:170], *names)
    with pytest.raises(ValueError, match="names"):
        seg_head_from_rows(
            weight, VOC2012_CLASS_NAMES, COCOSTUFF_CLASS_NAMES, {"cat": "lion"}
        )


class TinySeg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.body = nn.Linear(3, 3)
        self.seg_head = nn.Conv2d(8, num_classes, kernel_size=1)


def load(tmp_path, **kwargs):
    """A COCO-Stuff checkpoint (DDP keys, 171-class head) loaded into a 21-class model
    the way the VOC config loads it (seg_head renamed to dummy)."""
    g = torch.Generator().manual_seed(1)
    coco = {
        "module.body.weight": torch.randn(3, 3, generator=g),
        "module.body.bias": torch.randn(3, generator=g),
        "module.seg_head.weight": torch.randn(171, 8, 1, 1, generator=g),
        "module.seg_head.bias": torch.randn(171, generator=g),
    }
    path = tmp_path / "model_last.pth"
    torch.save({"state_dict": coco}, path)
    torch.manual_seed(2)
    model = TinySeg(21)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    trainer = SimpleNamespace(
        model=model,
        cfg=SimpleNamespace(checkpoint=[str(path)], resume=False),
        logger=logging.getLogger(__name__),
    )
    hook = CheckpointLoader(
        keywords="module.seg_head", replacement="module.dummy", **kwargs
    )
    hook.trainer = trainer
    hook.before_train()
    return model, coco, init


def test_loader_head_rows(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    rows = dict(classes="voc2012_from_cocostuff", source="dummy")
    model, coco, _ = load(tmp_path, seg_head_rows=rows)
    assert torch.equal(model.body.weight, coco["module.body.weight"])
    names = (VOC2012_CLASS_NAMES, COCOSTUFF_CLASS_NAMES, VOC2012_FROM_COCOSTUFF)
    for param in ("weight", "bias"):
        source = coco[f"module.seg_head.{param}"]
        got = getattr(model.seg_head, param).detach()
        assert torch.equal(got, seg_head_from_rows(source, *names))
        assert torch.equal(got[15], source[0])  # person
    assert "tv monitor <- tv" in caplog.text
    assert "background <- mean of the unmapped rows" in caplog.text


def test_loader_default_unchanged(tmp_path):
    model, coco, init = load(tmp_path)  # seg_head_rows=None: the random head stays
    assert torch.equal(model.body.weight, coco["module.body.weight"])
    assert torch.equal(model.seg_head.weight, init["seg_head.weight"])
    assert torch.equal(model.seg_head.bias, init["seg_head.bias"])


def test_loader_head_rows_errors(tmp_path):
    with pytest.raises(ValueError, match="classes must be one of"):
        CheckpointLoader(seg_head_rows=dict(classes="ade20k_from_cocostuff"))
    with pytest.raises(ValueError, match="seg_head_rows"):
        CheckpointLoader(seg_head_rows=dict(classes="voc2012_from_cocostuff", x=1))
    with pytest.raises(KeyError, match="not in the loaded checkpoints"):
        load(tmp_path, seg_head_rows=dict(classes="voc2012_from_cocostuff", source="x"))
