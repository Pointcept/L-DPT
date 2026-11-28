"""
Evaluator hooks (CPU): SemSegEvaluator flip TTA, multi-scale ("scale@r") TTA and
miou_present, batched TTA (one forward for the image and its flip) against one forward
each, the best-checkpoint
metric of the evaluators and its direction in CheckpointSaver,
CheckpointSaver(keep_epoch_weights), and the metrics history (metrics.jsonl, best epoch
per group, Pareto front).
"""

import logging
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from test_l_dpt_golden import DEPTH_LOSS, ENC, SEG_LOSS, randomize_

from ldpt.hooks.evaluator import (
    DepthEvaluator,
    SemSegEvaluator,
    pareto_front,
    read_metrics_history,
    scale_image,
    tta_predictions,
)
from ldpt.hooks.misc import CheckpointSaver, load_checkpoint
from ldpt.models.builder import build_model

NUM_CLASSES = 5
SEG = dict(
    type="L-DPT-v1m1",
    task="semseg",
    num_classes=NUM_CLASSES,
    criterion=SEG_LOSS,
    **ENC,
    dec_dim=32,
    dec_num_heads=2,
    dec_num_layers=(1, 1, 1, 1),
    dec_attn_downscale=2,
)


def fake_trainer(model, samples, **kwargs):
    loader = torch.utils.data.DataLoader(samples, batch_size=2)
    cfg = SimpleNamespace(evaluate=True, log_interval=1, enable_wandb=False)
    return SimpleNamespace(
        model=model,
        val_loader=loader,
        comm_info={},
        cfg=cfg,
        logger=logging.getLogger(__name__),
        writer=None,
        epoch=0,
        **kwargs,
    )


def evaluate(evaluator, model, samples):
    evaluator.trainer = fake_trainer(model, samples)
    evaluator.after_epoch()
    return evaluator.trainer.comm_info


def seg_samples(flip=False, n=4, hw=(40, 56)):
    """Random images and labels (H, W not multiples of the patch: padded)."""
    g = torch.Generator().manual_seed(0)
    samples = []
    for i in range(n):
        image = torch.randn(3, *hw, generator=g)
        segment = torch.randint(0, NUM_CLASSES, (1, *hw), generator=g)
        segment[torch.rand(segment.shape, generator=g) < 0.1] = -1
        if flip:
            image, segment = image.flip(-1), segment.flip(-1)
        samples.append(dict(image=image, segment=segment, name=f"seg/{i}"))
    return samples


DEPTH = dict(
    type="L-DPT-v1m1",
    criterion=DEPTH_LOSS,
    **ENC,
    dec_dim=32,
    dec_num_heads=2,
    dec_num_layers=(1, 1, 1, 1),
    dec_attn_downscale=2,
)


@pytest.fixture(scope="module")
def seg_model():
    return randomize_(build_model(SEG)).eval()


@pytest.fixture(scope="module")
def depth_model():
    return randomize_(build_model(DEPTH)).eval()


def miou(pred, segment):
    label = segment.flatten()
    keep = label != -1
    hist = torch.bincount(
        NUM_CLASSES * label[keep] + pred.flatten()[keep], minlength=NUM_CLASSES**2
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    hist = hist.double()
    diag = hist.diag()
    return (diag / (hist.sum(0) + hist.sum(1) - diag + 1e-10)).mean().item()


@torch.inference_mode()
def test_semseg_flip_tta(seg_model):
    samples = seg_samples()
    image = torch.stack([s["image"] for s in samples])
    segment = torch.stack([s["segment"] for s in samples])
    logits, flipped = [], []
    for i in range(0, len(samples), 2):  # the evaluator's batches
        logits.append(seg_model(dict(image=image[i : i + 2]))["pred_segment"])
        out = seg_model(dict(image=image[i : i + 2].flip(-1)))["pred_segment"]
        flipped.append(out.flip(-1))
    logits, flipped = torch.cat(logits), torch.cat(flipped)

    plain = evaluate(SemSegEvaluator(NUM_CLASSES), seg_model, samples)
    assert plain["current_metric_name"] == "miou"
    assert plain["current_metric_greater"] is True
    assert plain["current_metric_value"] == miou(logits.argmax(1), segment)

    # TTA: mean softmax of the original and the flipped-back flipped prediction
    tta = SemSegEvaluator(NUM_CLASSES, tta=[["flip"]])
    probs = torch.stack([logits.softmax(1), flipped.softmax(1)], dim=1).mean(1)
    expected = miou(probs.argmax(1), segment)
    assert evaluate(tta, seg_model, samples)["current_metric_value"] == expected

    # the flip-TTA prediction is flip-equivariant, the plain one is not
    samples_flipped = seg_samples(flip=True)
    tta_flipped = evaluate(tta, seg_model, samples_flipped)["current_metric_value"]
    assert tta_flipped == expected
    plain_flipped = evaluate(SemSegEvaluator(NUM_CLASSES), seg_model, samples_flipped)
    assert plain_flipped["current_metric_value"] != plain["current_metric_value"]


def test_semseg_best_metric(seg_model):
    with pytest.raises(ValueError, match="best_metric"):
        SemSegEvaluator(NUM_CLASSES, best_metric="iou")
    with pytest.raises(ValueError, match="tta op"):
        SemSegEvaluator(NUM_CLASSES, tta=[["rotate"]])
    info = evaluate(
        SemSegEvaluator(NUM_CLASSES, best_metric="allacc"), seg_model, seg_samples()
    )
    assert info["current_metric_name"] == "allacc"
    assert 0 < info["current_metric_value"] < 1
    # the record of metrics_file: one group, every metric, the per-class IoU
    evaluator = SemSegEvaluator(NUM_CLASSES)
    info = evaluate(evaluator, seg_model, seg_samples())
    record = evaluator.results
    assert record["score"] == info["current_metric_value"]
    assert record["counted"] == ["val"] and record["best_metric"] == "miou"
    assert record["groups"]["val"]["num_samples"] == 4
    assert record["groups"]["val"]["miou"] == info["current_metric_value"]
    assert set(record["groups"]["val"]) == {
        "num_samples",
        "num_absent_classes",
        "miou",
        "macc",
        "allacc",
        "miou_present",
    }
    assert len(record["per_class_iou"]) == NUM_CLASSES
    # every class in the GT: the mean over present classes is the mean
    assert record["groups"]["val"]["num_absent_classes"] == 0
    assert record["groups"]["val"]["miou_present"] == record["groups"]["val"]["miou"]


class LogitsModel(nn.Module):
    """Returns the image as the logits (the image has NUM_CLASSES channels)."""

    def forward(self, input_dict):
        return dict(pred_segment=input_dict["image"])


def test_semseg_miou_present(caplog):
    """miou_present / num_absent_classes are logged next to miou, which is unchanged
    (the mean over every class, union 0 counting as IoU 0)."""
    g = torch.Generator().manual_seed(0)
    samples = []
    for i in range(4):
        segment = torch.randint(0, 3, (1, 8, 12), generator=g)  # GT classes 0, 1, 2
        segment[0, 0] = -1
        pred = torch.randint(0, 3, (8, 12), generator=g)
        pred[pred == 2] = 3  # class 3 only predicted, class 4 nowhere
        logits = torch.nn.functional.one_hot(pred, NUM_CLASSES).permute(2, 0, 1)
        samples.append(dict(image=logits.float(), segment=segment, name=f"seg/{i}"))
    caplog.set_level(logging.INFO)
    evaluator = SemSegEvaluator(NUM_CLASSES)
    info = evaluate(evaluator, LogitsModel(), samples)
    segment = torch.stack([s["segment"] for s in samples])
    pred = torch.stack([s["image"] for s in samples]).argmax(1)
    # the main metric: bit-identical to the formula before miou_present
    assert info["current_metric_value"] == miou(pred, segment)
    group = evaluator.results["groups"]["val"]
    assert group["miou"] == info["current_metric_value"]
    iou = torch.tensor(evaluator.results["per_class_iou"], dtype=torch.float64)
    assert iou[2] == 0 and iou[3] == 0 and iou[4] == 0  # 2, 3: union > 0; 4: union 0
    assert group["miou"] == iou.mean().item()
    assert group["miou_present"] == iou[:4].mean().item()
    assert group["miou_present"] == pytest.approx(group["miou"] * 5 / 4)
    assert group["num_absent_classes"] == 1
    assert "1 classes in neither GT nor prediction" in caplog.text
    assert "Mean IoU of the 4 present classes" in caplog.text


class ForwardLog(nn.Module):
    """The wrapped model, recording the batch size of every forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.batches = []

    def forward(self, input_dict):
        self.batches.append(input_dict["image"].shape[0])
        return self.model(input_dict)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("task", ["depth", "seg"])
@pytest.mark.parametrize(
    "tta, batches",
    [
        ([["flip"]], [4]),  # image + flip: one forward
        ([["flip"], ["upsample"], ["flip", "upsample"]], [4, 4]),  # one per shape
    ],
)
@torch.inference_mode()
def test_tta_batched(task, precision, tta, batches):
    """Batched TTA = one forward per variant, up to rounding (bf16 autocast encoder)."""
    cfg = dict(DEPTH if task == "depth" else SEG, precision=precision)
    model = ForwardLog(randomize_(build_model(cfg)).eval())
    key = "pred_depth" if task == "depth" else "pred_segment"
    image = torch.randn(2, 3, 40, 56, generator=torch.Generator().manual_seed(0))
    batched = tta_predictions(model, dict(image=image), key, tta)
    assert model.batches == batches
    model.batches.clear()
    separate = tta_predictions(model, dict(image=image), key, tta, batched=False)
    assert model.batches == [2] * (1 + len(tta))
    assert len(batched) == len(separate) == 1 + len(tta)
    for a, b in zip(batched, separate):
        assert a.shape == separate[0].shape
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


# DINOv3's 5 scales x flip (the image itself is always prediction 0)
MS_FLIP = [["flip"]] + [
    ops
    for r in (0.9, 0.95, 1.05, 1.1)
    for ops in ([f"scale@{r}"], [f"scale@{r}", "flip"])
]


def test_scale_op():
    """scale@r: int(H r + 0.5) x int(W r + 0.5), bilinear; r = 1 is the image itself."""
    image = torch.randn(2, 3, 40, 56, generator=torch.Generator().manual_seed(0))
    assert scale_image(image, 1.0) is image
    assert scale_image(image, 1.005) is image  # 40.2, 56.28: the same size
    assert scale_image(image, 1.01).shape == (2, 3, 40, 57)  # 40.4, 56.56
    assert scale_image(image, 0.9).shape == (2, 3, 36, 50)  # 36.0, 50.4
    assert scale_image(image, 1.1).shape == (2, 3, 44, 62)  # 44.0, 61.6
    assert scale_image(image, 0.5).shape == (2, 3, 20, 28)
    expected = nn.functional.interpolate(
        image, size=(36, 50), mode="bilinear", align_corners=False
    )
    assert torch.equal(scale_image(image, 0.9), expected)
    for op in ["scale@0", "scale@-1", "scale@x", "scale@", "scale@nan", "scale@inf"]:
        with pytest.raises(ValueError, match="tta op"):
            SemSegEvaluator(NUM_CLASSES, tta=[[op]])
    # depth: no scale op (only the segmentation evaluator resizes back)
    with pytest.raises(ValueError, match="tta op"):
        DepthEvaluator(relative=True, tta=[["scale@0.9"]])


@torch.inference_mode()
def test_tta_predictions_scale(seg_model):
    """Scale variants keep their size; the other ops of the entry are undone; one
    forward per shape when batched; r = 1 is the plain prediction."""
    model = ForwardLog(seg_model)
    image = torch.randn(2, 3, 40, 56, generator=torch.Generator().manual_seed(0))
    run = lambda x: seg_model(dict(image=x))["pred_segment"]  # noqa: E731
    tta = [["scale@1.0"], ["scale@0.9"], ["scale@0.9", "flip"], ["flip", "scale@1.1"]]
    preds = tta_predictions(model, dict(image=image), "pred_segment", tta, False)
    assert model.batches == [2] * 5
    assert torch.equal(preds[1], preds[0])  # identity at r = 1
    assert torch.equal(preds[0], run(image))
    assert preds[2].shape == preds[3].shape == (2, NUM_CLASSES, 36, 50)
    assert preds[4].shape == (2, NUM_CLASSES, 44, 62)
    small = scale_image(image, 0.9)
    assert torch.equal(preds[2], run(small))
    assert torch.equal(preds[3], run(small.flip(-1)).flip(-1))  # flipped back
    assert torch.equal(preds[4], run(scale_image(image.flip(-1), 1.1)).flip(-1))
    model.batches.clear()
    batched = tta_predictions(model, dict(image=image), "pred_segment", MS_FLIP)
    assert model.batches == [4] * 5  # image + flip, then each scale + its flip
    separate = tta_predictions(model, dict(image=image), "pred_segment", MS_FLIP, False)
    for a, b in zip(batched, separate):
        assert a.shape == b.shape
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def scaled_label_samples(n=4, hw=(40, 56), label_hw=(30, 42)):
    """As the val transform: image resized, labels at their own (native) size."""
    g = torch.Generator().manual_seed(2)
    samples = []
    for i in range(n):
        segment = torch.randint(0, NUM_CLASSES, (1, *label_hw), generator=g)
        segment[torch.rand(segment.shape, generator=g) < 0.1] = -1
        image = torch.randn(3, *hw, generator=g)
        samples.append(dict(image=image, segment=segment, name=f"seg/{i}"))
    return samples


@torch.inference_mode()
def test_semseg_scale_tta(seg_model):
    """5 scales x flip: the softmax of every variant (flipped back) resized to the
    label size, averaged there; scale@1 alone takes the same-size (flip-TTA) path."""
    samples = scaled_label_samples()
    image = torch.stack([s["image"] for s in samples])
    segment = torch.stack([s["segment"] for s in samples])
    size = segment.shape[-2:]
    probs = []
    for i in range(0, len(samples), 2):  # the evaluator's batches
        x = image[i : i + 2]
        preds = tta_predictions(
            seg_model, dict(image=x), "pred_segment", MS_FLIP, False
        )
        probs.append(
            torch.stack(
                [
                    nn.functional.interpolate(
                        p.softmax(1), size=size, mode="bilinear", align_corners=False
                    )
                    for p in preds
                ],
                dim=1,
            ).mean(1)
        )
    expected = miou(torch.cat(probs).argmax(1), segment)
    evaluator = SemSegEvaluator(NUM_CLASSES, tta=MS_FLIP, tta_batched=False)
    assert evaluate(evaluator, seg_model, samples)["current_metric_value"] == expected
    batched = evaluate(SemSegEvaluator(NUM_CLASSES, tta=MS_FLIP), seg_model, samples)
    assert batched["current_metric_value"] == pytest.approx(expected, abs=1e-3)
    # scale@1 only: every prediction at the image size, the flip-TTA path (mean
    # softmax, then resized); two equal softmaxes give the plain argmax
    one = SemSegEvaluator(NUM_CLASSES, tta=[["scale@1.0"]], tta_batched=False)
    one = evaluate(one, seg_model, samples)
    logits = torch.cat(
        [
            seg_model(dict(image=image[i : i + 2]))["pred_segment"]
            for i in range(0, len(samples), 2)
        ]
    )
    resized = nn.functional.interpolate(
        logits.softmax(1), size=size, mode="bilinear", align_corners=False
    )
    assert one["current_metric_value"] == miou(resized.argmax(1), segment)


def test_scale_tta_options():
    """The 5-scale x flip option string of --options parses to MS_FLIP."""
    from ldpt.utils.config import DictAction

    text = (
        "[[flip],[scale@0.9],[scale@0.9,flip],[scale@0.95],[scale@0.95,flip],"
        "[scale@1.05],[scale@1.05,flip],[scale@1.1],[scale@1.1,flip]]"
    )
    assert DictAction._parse_iterable(text) == MS_FLIP


def depth_eval_samples(n=6, hw=(40, 56)):
    """Random images and GT depth (10% invalid) in groups a and b."""
    g = torch.Generator().manual_seed(1)
    samples = []
    for i in range(n):
        depth = torch.rand(1, *hw, generator=g) * 5 + 0.5
        depth[torch.rand(depth.shape, generator=g) < 0.1] = 0
        image = torch.randn(3, *hw, generator=g)
        samples.append(dict(image=image, depth=depth, name=f"{'ab'[i % 2]}/{i}"))
    return samples


@pytest.mark.parametrize("tta", [[["flip"]], [["flip"], ["upsample"]]])
def test_evaluators_tta_batched(depth_model, seg_model, tta):
    """The evaluators' metrics with batched TTA match one forward per variant."""
    samples = depth_eval_samples()
    ref = DepthEvaluator(relative=True, tta=tta, tta_batched=False)
    evaluate(ref, depth_model, samples)
    out = DepthEvaluator(relative=True, tta=tta)
    evaluate(out, depth_model, samples)
    assert out.results["score"] == pytest.approx(ref.results["score"], rel=1e-5)
    assert set(out.results["groups"]) == {"a", "b"}
    for group, metrics in ref.results["groups"].items():
        assert out.results["groups"][group] == pytest.approx(metrics, rel=1e-4)

    samples = seg_samples()
    ref = SemSegEvaluator(NUM_CLASSES, tta=tta, tta_batched=False)
    ref = evaluate(ref, seg_model, samples)["current_metric_value"]
    out = evaluate(SemSegEvaluator(NUM_CLASSES, tta=tta), seg_model, samples)
    assert out["current_metric_value"] == pytest.approx(ref, abs=1e-3)


class ScaledDepth(nn.Module):
    """pred_depth = channel 0 of the image (the samples put a scaled GT there)."""

    def forward(self, input_dict):
        return dict(pred_depth=input_dict["image"][:, :1])


def depth_samples(groups=("a", "b"), scales=(1.1, 1.5)):
    """Group a: pred = 1.1 * GT (2 images), group b: pred = 1.5 * GT (1 image)."""
    a, b = groups
    sa, sb = scales
    g = torch.Generator().manual_seed(0)
    samples = []
    for name, scale in [(f"{a}/0", sa), (f"{a}/1", sa), (f"{b}/0", sb)]:
        depth = torch.rand(1, 8, 8, generator=g) + 1.0
        image = (depth * scale).expand(3, 8, 8).clone()
        samples.append(dict(image=image, depth=depth, name=name))
    return samples


@pytest.mark.parametrize(
    "kwargs, groups, best_metric, greater, expected",
    [
        # default: geometric mean over groups, not images
        ({}, ("a", "b"), "abs_rel", False, (0.1 * 0.5) ** 0.5),
        (dict(best_metric_mean="arithmetic"), ("a", "b"), "abs_rel", False, 0.3),
        # delta1, arithmetic mean
        (
            dict(best_metric="delta1", best_metric_mean="arithmetic"),
            ("a", "b"),
            "delta1",
            True,
            (1.0 + 0.0) / 2,
        ),
        # the combined DIODE group is logged, not counted; a plain DIODE group counts
        ({}, ("DIODE-INDOOR", "DIODE-OUTDOOR"), "abs_rel", False, (0.1 * 0.5) ** 0.5),
        ({}, ("nyuv2", "DIODE"), "abs_rel", False, (0.1 * 0.5) ** 0.5),
        # score_groups: only these count; the others (e.g. held-out synthetic) logged
        (dict(score_groups=["nyuv2"]), ("nyuv2", "hypersim"), "abs_rel", False, 0.1),
        (dict(score_groups=["DIODE"]), ("DIODE-INDOOR", "x"), "abs_rel", False, 0.1),
    ],
)
def test_depth_best_metric(kwargs, groups, best_metric, greater, expected):
    evaluator = DepthEvaluator(relative=False, **kwargs)
    info = evaluate(evaluator, ScaledDepth(), depth_samples(groups))
    assert info["current_metric_name"] == best_metric
    assert info["current_metric_greater"] is greater
    assert info["current_metric_value"] == pytest.approx(expected, abs=1e-6)
    with pytest.raises(ValueError, match="best_metric"):
        DepthEvaluator(best_metric="d1")
    with pytest.raises(ValueError, match="best_metric_mean"):
        DepthEvaluator(best_metric_mean="harmonic")
    with pytest.raises(ValueError, match="tta op"):
        DepthEvaluator(tta=[["rotate"]])


@pytest.mark.parametrize(
    "greater, values, best_epochs",
    [
        (True, [0.5, 0.3, 0.6], [1, 1, 3]),
        (False, [0.5, 0.3, 0.4], [1, 2, 2]),  # e.g. abs_rel: lower is better
    ],
)
def test_checkpoint_saver_best(tmp_path, greater, values, best_epochs):
    (tmp_path / "model").mkdir()
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = fake_trainer(
        model,
        [],
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1),
        scaler=None,
        best_metric_value=-torch.inf,
    )
    trainer.cfg.save_path, trainer.cfg.enable_amp = str(tmp_path), False
    saver = CheckpointSaver()
    saver.trainer = trainer
    for epoch, (value, best_epoch) in enumerate(zip(values, best_epochs)):
        trainer.epoch = epoch
        trainer.comm_info.update(
            current_metric_value=value,
            current_metric_name="m",
            current_metric_greater=greater,
        )
        saver.after_epoch()
        best = load_checkpoint(tmp_path / "model" / "model_best.pth")
        assert best["epoch"] == best_epoch
        assert best["best_metric_value"] == values[best_epoch - 1]


def test_checkpoint_saver_epoch_weights(tmp_path):
    """keep_epoch_weights: epoch_<N>.pth per epoch, weights only, no skipped keys."""
    (tmp_path / "model").mkdir()
    model = nn.ModuleDict(dict(encoder=nn.Linear(2, 2), head=nn.Linear(2, 1)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = fake_trainer(
        model,
        [],
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1),
        scaler=None,
        best_metric_value=-torch.inf,
    )
    trainer.cfg.save_path, trainer.cfg.enable_amp = str(tmp_path), False
    trainer.cfg.evaluate = False
    saver = CheckpointSaver(skip_keywords=["encoder"], keep_epoch_weights=True)
    saver.trainer = trainer
    for epoch in range(2):
        trainer.epoch = epoch
        with torch.no_grad():
            model.head.weight.fill_(epoch)
        saver.after_epoch()
    files = sorted(p.name for p in (tmp_path / "model").iterdir())
    assert files == ["epoch_1.pth", "epoch_2.pth", "model_last.pth"]
    last = load_checkpoint(tmp_path / "model" / "model_last.pth")
    for epoch in (1, 2):
        weights = load_checkpoint(tmp_path / "model" / f"epoch_{epoch}.pth")
        assert set(weights) == {"epoch", "state_dict"} and weights["epoch"] == epoch
        assert set(weights["state_dict"]) == {"head.weight", "head.bias"}
        assert (weights["state_dict"]["head.weight"] == epoch - 1).all()
    for key, value in last["state_dict"].items():
        assert torch.equal(weights["state_dict"][key], value)
    with pytest.raises(ValueError, match="keep_epoch_weights"):
        CheckpointSaver(save_freq=1, keep_epoch_weights=True)


def test_pareto_front():
    rows = [dict(a=0.1, b=0.5), dict(a=0.3, b=0.2), dict(a=0.3, b=0.4)]
    assert pareto_front(rows, ["a", "b"], greater=False) == [0, 1]
    assert pareto_front(rows, ["a", "b"], greater=True) == [0, 2]
    assert pareto_front(rows, ["a"], greater=False) == [0]
    assert pareto_front([dict(a=1.0), dict(a=1.0)], ["a"], greater=False) == [0, 1]


def test_metrics_history(tmp_path, caplog):
    """metrics.jsonl: one line per evaluation; resume reads it back and appends."""
    caplog.set_level(logging.INFO)
    path = tmp_path / "metrics.jsonl"
    path.write_text("stale line of an earlier run\n")

    def run(scales_per_epoch, resume, first_epoch=0):
        evaluator = DepthEvaluator()
        trainer = fake_trainer(ScaledDepth(), [])
        trainer.cfg.save_path, trainer.cfg.resume = str(tmp_path), resume
        evaluator.trainer = trainer
        evaluator.before_train()
        for epoch, scales in enumerate(scales_per_epoch, start=first_epoch):
            trainer.epoch = epoch
            samples = depth_samples(("DIODE-INDOOR", "kitti"), scales)
            trainer.val_loader = torch.utils.data.DataLoader(samples, batch_size=2)
            evaluator.after_epoch()
        return evaluator

    # AbsRel (DIODE-INDOOR, kitti): (0.1, 0.5), (0.3, 0.2), (0.3, 0.4)
    run([(1.1, 1.5), (1.3, 1.2), (1.3, 1.4)], resume=False)  # fresh: file anew
    history = read_metrics_history(path)
    assert list(history) == [1, 2, 3]
    record = history[2]
    assert record["best_metric"] == "abs_rel"
    assert record["best_metric_mean"] == "geometric"
    assert record["score"] == pytest.approx((0.3 * 0.2) ** 0.5, abs=1e-6)
    assert record["counted"] == ["DIODE-INDOOR", "kitti"]  # combined DIODE: logged
    assert set(record["groups"]) == {"DIODE-INDOOR", "DIODE", "kitti"}
    group = record["groups"]["DIODE-INDOOR"]
    assert set(group) == {"num_samples", *DepthEvaluator.METRICS}
    assert group["num_samples"] == 2
    assert group["abs_rel"] == pytest.approx(0.3, abs=1e-6)
    assert "DIODE-INDOOR 0.1000 (epoch 1)" in caplog.text
    assert "kitti 0.2000 (epoch 2)" in caplog.text
    assert "Best score so far: 0.2236 (epoch 1)" in caplog.text
    assert "Pareto front of ['DIODE-INDOOR', 'kitti']: [1, 2]" in caplog.text

    # resumed from the epoch 2 checkpoint: epoch 3 is evaluated again
    caplog.clear()
    evaluator = run([(1.05, 1.05)], resume=True, first_epoch=2)
    assert list(evaluator.history) == [1, 2, 3]
    assert len(path.read_text().splitlines()) == 4
    history = read_metrics_history(path)
    assert list(history) == [1, 2, 3]
    assert history[3]["score"] == pytest.approx(0.05, abs=1e-6)
    assert "kitti 0.0500 (epoch 3)" in caplog.text
    assert "Best score so far: 0.0500 (epoch 3)" in caplog.text
    assert "Pareto front of ['DIODE-INDOOR', 'kitti']: [3]" in caplog.text

    # metrics_file=None: no file
    path.unlink()
    evaluator = DepthEvaluator(metrics_file=None)
    evaluator.trainer = fake_trainer(ScaledDepth(), depth_samples())
    evaluator.trainer.cfg.save_path = str(tmp_path)
    evaluator.after_epoch()
    assert evaluator.results["epoch"] == 1 and not path.exists()


# ------------------------------------------------------------ robust-fit evaluation


def depth_map(seed, hw=(32, 32)):
    """GT depth uniform in 1 .. 10 m."""
    return 1.0 + 9.0 * torch.rand(1, *hw, generator=torch.Generator().manual_seed(seed))


def pred_samples(pairs):
    """[(name, prediction [1, H, W], GT [1, H, W])] -> samples for ScaledDepth (the
    prediction is image channel 0)."""
    return [
        dict(image=pred.expand(3, *pred.shape[1:]).clone(), depth=gt, name=name)
        for name, pred, gt in pairs
    ]


def with_outliers(depth, frac, seed=0):
    """depth with round(frac * N) random pixels at 10x, and their mask."""
    n = depth.numel()
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    out = torch.zeros(n, dtype=torch.bool)
    out[idx[: round(frac * n)]] = True
    out = out.view(depth.shape)
    return torch.where(out, 10.0 * depth, depth), out


def test_depth_evaluator_option_validation():
    for bad, match in (
        (dict(relative=True, align_solver="lstsq"), "align_solver"),
        (dict(relative=True, align_protocols=["trim10"]), "align protocol"),
        (dict(relative=True, align_protocols=["disp", "disp"]), "repeats"),
        (dict(align_protocols=["disp"]), "relative"),
        (dict(relative=True, score_groups=["nyuv2@trim5"]), "never counted"),
    ):
        with pytest.raises(ValueError, match=match):
            DepthEvaluator(**bad)
    ev = DepthEvaluator(relative=True, align_solver="lsq", align_protocols=["disp"])
    assert ev.protocol("nyuv2").startswith("align=depth-lsq ")
    assert ev.protocol("nyuv2@disp").startswith("align=disparity-lsq ")
    assert (
        DepthEvaluator(relative=True)
        .protocol("a")
        .startswith("align=scale-shift/centered ")
    )


def test_align_solver_lsq():
    """ "lsq" = exact fp64 least squares with a non-negative scale: "centered" on a
    well-conditioned batch; an anti-correlated or flat image scores as the
    constant GT mean (s = 0, num_rejected) instead of leaving the group, so a
    checkpoint that flips or flattens images never scores better than one that
    predicts them poorly, and a group never drops out of the score."""
    from ldpt.models.utils.depth import (
        compute_scale_and_shift,
        lsq_scale_and_shift,
        nonneg_lsq_scale_and_shift,
    )

    g = torch.Generator().manual_seed(2)
    gts = [depth_map(i) for i in range(4)]
    preds = [0.05 * d + 0.1 + 0.02 * torch.randn(d.shape, generator=g) for d in gts]
    preds[3] = 1.0 - 0.05 * gts[3]  # anti-correlated
    p, d = torch.cat(preds), torch.cat(gts)
    s, t, valid = lsq_scale_and_shift(p, d, d > 0)
    s2, t2, _ = compute_scale_and_shift(p, d, d > 0, ridge=0)
    assert valid.tolist() == [True, True, True, False]
    torch.testing.assert_close(s[:3], s2[:3], rtol=1e-6, atol=0)
    torch.testing.assert_close(t[:3], t2[:3], rtol=1e-6, atol=1e-6)
    s3, t3, valid3, rejected = nonneg_lsq_scale_and_shift(p, d, d > 0)
    assert valid3.all() and rejected.tolist() == [False, False, False, True]
    assert torch.equal(s3[:3], s[:3]) and torch.equal(t3[:3], t[:3])
    assert s3[3].item() == 0.0
    assert t3[3].item() == pytest.approx(gts[3].mean().item(), rel=1e-6)

    names = ["a/0", "a/1", "b/2", "b/3"]
    samples = pred_samples(zip(names, preds, gts))
    records = {}
    for solver in ("centered", "lsq"):
        evaluator = DepthEvaluator(relative=True, align_solver=solver)
        evaluate(evaluator, ScaledDepth(), samples[:3])
        records[solver] = evaluator.results
        evaluator = DepthEvaluator(relative=True, align_solver=solver)
        evaluate(evaluator, ScaledDepth(), samples)
        records[solver + "+anti"] = evaluator.results
    for group in ("a", "b"):
        a, b = records["centered"]["groups"][group], records["lsq"]["groups"][group]
        assert (b.pop("num_excluded"), b.pop("num_rejected")) == (0, 0)
        assert "num_excluded" not in a and "num_rejected" not in a
        assert b == pytest.approx(a, rel=1e-6, abs=1e-7)
    lsq, centered = records["lsq+anti"]["groups"], records["centered+anti"]["groups"]
    assert lsq["b"]["num_samples"] == 2 and lsq["b"]["num_excluded"] == 0
    assert lsq["b"]["num_rejected"] == 1 and lsq["a"]["num_rejected"] == 0
    assert centered["b"]["num_samples"] == 2  # median / MAD fallback
    assert "align_protocols" not in records["lsq"]
    # the anti-correlated image scores as the constant GT mean
    mean = gts[3].mean()
    const = ((mean - gts[3]).abs() / gts[3]).mean().item()
    b_ok = records["lsq"]["groups"]["b"]["abs_rel"]
    assert lsq["b"]["abs_rel"] == pytest.approx((b_ok + const) / 2, rel=1e-5)

    # checkpoints: image 3 predicted poorly (noisy), flipped or flat; "lsq" ranks
    # the flipped and the flat one below the poor one, and counts both groups
    noisy = 0.05 * gts[3] + 0.4 * torch.randn(gts[3].shape, generator=g)
    flat = torch.full_like(gts[3], 0.5)
    scores = {}
    for kind, pred3 in (("noisy", noisy), ("anti", preds[3]), ("flat", flat)):
        pairs = list(zip(names, preds[:3] + [pred3], gts))
        evaluator = DepthEvaluator(relative=True, align_solver="lsq")
        evaluate(evaluator, ScaledDepth(), pred_samples(pairs))
        scores[kind] = evaluator.results["score"]
        assert evaluator.results["counted"] == ["a", "b"]
        assert evaluator.results["groups"]["b"]["num_samples"] == 2
    assert scores["noisy"] < scores["anti"] and scores["noisy"] < scores["flat"]
    # a group whose every image is flipped still counts
    pairs = [(n, 1.0 - 0.05 * d, d) for n, d in zip(names[2:], gts[2:])]
    evaluator = DepthEvaluator(relative=True, align_solver="lsq")
    evaluate(
        evaluator, ScaledDepth(), pred_samples(list(zip(names, preds, gts))[:2] + pairs)
    )
    assert evaluator.results["counted"] == ["a", "b"]
    assert evaluator.results["groups"]["b"]["num_rejected"] == 2


def test_align_solver_lsq_free():
    """ "lsq_free" = Marigold's evaluation fit (np.linalg.lstsq of GT on [pred, 1],
    no constraint): bit for bit "lsq" where "lsq" accepts the fit; an anti-correlated
    image keeps its negative scale (num_rejected counts it, nothing is excluded); a
    flat one scores as the GT mean, as lstsq's minimum-norm solution does."""
    import numpy as np

    from ldpt.models.utils.depth import (
        free_lsq_scale_and_shift,
        nonneg_lsq_scale_and_shift,
    )

    g = torch.Generator().manual_seed(5)
    gts = [depth_map(40 + i) for i in range(4)]
    preds = [0.05 * d + 0.1 + 0.02 * torch.randn(d.shape, generator=g) for d in gts]
    preds[2] = 1.0 - 0.05 * gts[2] + 0.01 * torch.randn(gts[2].shape, generator=g)
    preds[3] = torch.full_like(gts[3], 0.5)
    p, d = torch.cat(preds), torch.cat(gts)
    mask = d > 0
    mask[0, :4] = False
    s, t, valid, signed = free_lsq_scale_and_shift(p, d, mask)
    s0, t0, valid0, rejected0 = nonneg_lsq_scale_and_shift(p, d, mask)
    assert valid.all() and torch.equal(signed, rejected0)
    assert signed.tolist() == [False, False, True, True]
    assert torch.equal(s[:2], s0[:2]) and torch.equal(t[:2], t0[:2])
    for i in range(3):  # numpy's lstsq, as Marigold
        m = mask[i].numpy()
        a = np.stack([p[i].numpy()[m], np.ones(m.sum())], 1).astype(np.float64)
        ref, *_ = np.linalg.lstsq(a, d[i].numpy()[m].astype(np.float64), rcond=None)
        assert s[i].item() == pytest.approx(ref[0], rel=1e-5)
        assert t[i].item() == pytest.approx(ref[1], rel=1e-5, abs=1e-5)
    assert s[2].item() < 0
    assert s[3].item() == 0.0
    assert t[3].item() == pytest.approx(gts[3].mean().item(), rel=1e-6)

    names = ["a/0", "a/1", "b/2", "b/3"]
    samples = pred_samples(zip(names, preds, gts))
    results = {}
    for solver in ("lsq", "lsq_free"):
        evaluator = DepthEvaluator(relative=True, align_solver=solver)
        evaluate(evaluator, ScaledDepth(), samples)
        results[solver] = evaluator.results["groups"]
        assert evaluator.protocol("a").startswith(
            "align=depth-" + solver.replace("_", "-") + " "
        )
    free, nonneg = results["lsq_free"], results["lsq"]
    assert free["a"] == nonneg["a"]
    assert free["b"]["num_rejected"] == 2 and free["b"]["num_excluded"] == 0
    assert free["b"]["num_samples"] == 2
    # the inverted image scores better than the constant mean "lsq" gives it
    assert free["b"]["abs_rel"] < nonneg["b"]["abs_rel"]


def test_depth_max_inf():
    """A set without an upper bound (ETH3D, depth_max inf): the headline clip and the
    disp protocol stay finite (disparity floor 1e-3, Marigold's)."""
    d = depth_map(50)
    pred = 1.0 / (1.0 / d + 0.05)
    pred[0, 0, 0] = 1e3
    gt = d.clone()
    gt[0, 0, 0] = 0.0
    mask = gt > 0
    evaluator = DepthEvaluator(relative=True, align_solver="lsq_free")
    input_dict = dict(
        depth_min=torch.tensor([1e-5]), depth_max=torch.tensor([math.inf])
    )
    clamped = evaluator.clamp_range(pred * 1e4, input_dict)
    assert torch.equal(clamped, pred * 1e4)
    aligned, valid, _ = evaluator.align_protocol("disp", pred, gt, mask, input_dict)
    assert valid.item() and torch.isfinite(aligned).all()
    assert aligned[0, 0, 0].item() == pytest.approx(1e3)


def marigold_align(gt_arr, pred_arr, valid_mask_arr, max_resolution=None):
    """prs-eth/Marigold src/util/alignment.py align_depth_least_square
    (return_scale_shift=True), verbatim but for the return: (aligned, scale, shift,
    the mask it fits on)."""
    import numpy as np

    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)
    return aligned_pred, scale.item(), shift.item(), valid_mask


def test_max_res_columns_eth3d():
    """Marigold's ETH3D fit (script/depth/eval/32_eval_eth3d.sh: --alignment_max_res
    1024) on the 4032 x 6048 frame: align_depth_least_square's torch.nn.Upsample
    (nearest) on the [1, H, W] arrays keeps every row and 1024 of the 6048 columns,
    0, 5, 11, 17, ..., 6042; max_res_columns is that index set."""
    import numpy as np

    from ldpt.hooks.evaluator import max_res_columns

    height, width = 4032, 6048
    cols = max_res_columns(height, width, 1024)
    assert cols.dtype == torch.long and cols.numel() == 1024
    assert cols[:6].tolist() == [0, 5, 11, 17, 23, 29]
    assert cols[-3:].tolist() == [6030, 6036, 6042]
    assert torch.all(cols[1:] > cols[:-1])
    # Marigold's downscaler on an [H, W] map of column indices: the same columns,
    # every row
    index = np.ascontiguousarray(
        np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    )
    scale_factor = np.min(1024 / np.array((height, width)))
    down = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
    out = down(torch.as_tensor(index).unsqueeze(0)).numpy()
    assert out.shape == (1, height, 1024)
    assert (out[0] == cols.numpy().astype(np.float32)[None]).all()
    # no dimension above max_res: every pixel (Marigold: scale_factor >= 1)
    assert max_res_columns(480, 640, 1024) is None
    assert max_res_columns(1024, 1024, 1024) is None


def test_align_max_res_matches_marigold():
    """align_max_res=dict(eth3d=12) on 24 x 60 images = Marigold's
    align_depth_least_square(max_resolution=12) (12 of the 60 columns): the fit mask
    is Marigold's downscaled mask, the scale and shift its lstsq solution, and the
    metrics use the full mask; a group without the option (nyuv2) fits every valid
    pixel, also within a mixed batch."""
    import numpy as np

    from ldpt.hooks.evaluator import max_res_columns

    hw, max_res = (24, 60), 12
    cols = max_res_columns(*hw, max_res)
    assert cols.tolist() == list(range(0, 60, 5))
    g = torch.Generator().manual_seed(11)
    names = ["eth3d/0", "nyuv2/0", "eth3d/1", "eth3d/2"]
    gts, preds = [], []
    for i in range(len(names)):
        d = depth_map(70 + i, hw)
        d[torch.rand(d.shape, generator=g) < 0.2] = 0.0  # invalid pixels
        bend = torch.linspace(0.6, 1.4, hw[1]).view(1, 1, -1)  # column-wise
        p = 0.05 * d.clamp_min(1.0) ** 0.8 * bend + 0.1
        preds.append(p + 0.01 * torch.randn(d.shape, generator=g))
        gts.append(d)
    p, d = torch.cat(preds), torch.cat(gts)
    mask = d > 0
    evaluator = DepthEvaluator(
        relative=True, align_solver="lsq_free", align_max_res=dict(eth3d=max_res)
    )
    fit = evaluator.fit_mask(names, mask)
    s, t, valid, _ = evaluator.align(p, d, fit)
    assert valid.all()
    expected, full_fit = {}, {}
    for i, name in enumerate(names):
        group = name.split("/")[0]
        gt_arr, pred_arr, mask_arr = d[i].numpy(), p[i].numpy(), mask[i].numpy()
        aligned, scale, shift, fit_arr = marigold_align(
            gt_arr, pred_arr, mask_arr, max_res if group == "eth3d" else None
        )
        if group == "eth3d":  # Marigold's mask = ours on its columns, none elsewhere
            assert np.array_equal(fit[i][:, cols].numpy(), fit_arr[0])
            assert fit[i].sum().item() == fit_arr.sum() < mask[i].sum().item()
        else:
            assert torch.equal(fit[i], mask[i])
        assert s[i].item() == pytest.approx(scale, rel=1e-4)
        assert t[i].item() == pytest.approx(shift, rel=1e-4, abs=1e-5)
        a = np.clip(aligned, 1e-6, None)[mask_arr]  # scored on every valid pixel
        expected.setdefault(group, []).append(
            np.mean(np.abs(a - gt_arr[mask_arr]) / gt_arr[mask_arr])
        )
        aligned, *_ = marigold_align(gt_arr, pred_arr, mask_arr)
        a = np.clip(aligned, 1e-6, None)[mask_arr]
        full_fit.setdefault(group, []).append(
            np.mean(np.abs(a - gt_arr[mask_arr]) / gt_arr[mask_arr])
        )

    evaluate(evaluator, ScaledDepth(), pred_samples(zip(names, preds, gts)))
    groups = evaluator.results["groups"]
    for group in ("eth3d", "nyuv2"):
        assert groups[group]["abs_rel"] == pytest.approx(
            np.mean(expected[group]), rel=1e-5
        )
    # the column subsample changes the ETH3D number (not the NYUv2 one)
    assert np.mean(full_fit["eth3d"]) != pytest.approx(np.mean(expected["eth3d"]))
    assert evaluator.protocol("eth3d").startswith("align=depth-lsq-free/max-res-12 ")
    assert evaluator.protocol("nyuv2").startswith("align=depth-lsq-free ")


def test_align_max_res_options():
    """align_max_res: None / "None" (--options) / {group: int >= 1}; relative only."""
    assert DepthEvaluator(relative=True, align_max_res="None").align_max_res == {}
    assert DepthEvaluator(relative=True).align_max_res == {}
    evaluator = DepthEvaluator(
        relative=True, align_max_res=dict(eth3d=1024, kitti=None)
    )
    assert evaluator.align_max_res == dict(eth3d=1024)
    mask = torch.ones(2, 8, 8, dtype=torch.bool)
    assert evaluator.fit_mask(["kitti/0", "eth3d/0"], mask) is mask  # 8 <= 1024
    for bad in (1024, dict(eth3d=0), dict(eth3d=10.5), dict(eth3d=True)):
        with pytest.raises(ValueError, match="align_max_res"):
            DepthEvaluator(relative=True, align_max_res=bad)
    with pytest.raises(ValueError, match="relative"):
        DepthEvaluator(align_max_res=dict(eth3d=1024))


def test_trim5_protocol():
    """Prediction exact (affine in depth) against a GT with outliers at 10x: trim5
    recovers the exact fit, lsq does not; the metrics use every valid pixel, so the
    outliers stay in them (AbsRel 0.9 per outlier)."""
    d = depth_map(10)
    p = (d - 0.5) / 20.0
    for frac in (0.01, 0.05):
        bad, out = with_outliers(d, frac)
        evaluator = DepthEvaluator(
            relative=True, align_solver="lsq", align_protocols=["trim5"]
        )
        evaluate(evaluator, ScaledDepth(), pred_samples([("x/0", p, bad)]))
        groups = evaluator.results["groups"]
        floor = 0.9 * out.float().mean().item()
        assert groups["x@trim5"]["abs_rel"] == pytest.approx(floor, abs=1e-5)
        assert groups["x"]["abs_rel"] > 0.05 + floor
        if frac == 0.01:  # spec thresholds
            assert groups["x@trim5"]["abs_rel"] < 0.01
    # clean GT: trim5 drops 5% of exact pixels and refits the same line
    evaluate(evaluator, ScaledDepth(), pred_samples([("x/0", p, d)]))
    groups = evaluator.results["groups"]
    assert groups["x@trim5"]["abs_rel"] == pytest.approx(
        groups["x"]["abs_rel"], abs=1e-3
    )


def test_disp_protocol():
    """disp fits 1 / p to 1 / g: a prediction affine in inverse depth is exact under
    disp and not under lsq, a depth-affine one the reverse; an aligned disparity
    <= 0 is clamped to 1 / depth_max (the pixel ends at depth_max)."""
    d = depth_map(11)
    disp_affine = 1.0 / (1.0 / d + 0.5)
    depth_affine = 0.05 * d + 0.1
    evaluator = DepthEvaluator(
        relative=True, align_solver="lsq", align_protocols=["disp"]
    )
    samples = pred_samples([("u/0", disp_affine, d), ("v/0", depth_affine, d)])
    evaluate(evaluator, ScaledDepth(), samples)
    groups = evaluator.results["groups"]
    assert groups["u@disp"]["abs_rel"] < 1e-3 and groups["u"]["abs_rel"] > 0.05
    assert groups["v"]["abs_rel"] < 1e-3 and groups["v@disp"]["abs_rel"] > 0.05

    # 1 / p = 1 / g + 0.05 (shift -0.05 in disparity); one pixel (no GT, so not in
    # the fit) predicted very far: s x + t < 0 there
    pred = 1.0 / (1.0 / d + 0.05)
    pred[0, 0, 0] = 1e3
    gt = d.clone()
    gt[0, 0, 0] = 0.0
    mask = gt > 0
    for input_dict, far in ((dict(depth_max=torch.tensor([20.0])), 20.0), ({}, 1e3)):
        aligned, valid, rejected = evaluator.align_protocol(
            "disp", pred, gt, mask, input_dict
        )
        assert valid.item() and not rejected.item()
        assert aligned[0, 0, 0].item() == pytest.approx(far)
        torch.testing.assert_close(aligned[mask], d[mask], rtol=1e-4, atol=0)


def test_ransac_protocol():
    """Seeded per image: two evaluations agree bit for bit (and a batch gives the
    images' own results); 30% outliers: RANSAC recovers the clean fit, LSQ does not;
    an anti-correlated prediction takes the mean ratio and is not excluded."""
    from ldpt.models.utils.depth import lsq_scale_and_shift, ransac_scale_and_shift

    d = depth_map(12)
    p = (d - 0.5) / 20.0  # exact: s = 20, t = 0.5
    bad, _ = with_outliers(d, 0.3, seed=1)
    m = bad > 0
    s, t, valid = ransac_scale_and_shift(p, bad, m)
    assert valid.item()
    assert s.item() == pytest.approx(20.0, rel=0.01)
    assert t.item() == pytest.approx(0.5, abs=0.01 * 20)
    s_lsq, _, _ = lsq_scale_and_shift(p, bad, m)
    assert abs(s_lsq.item() / 20.0 - 1) > 0.1

    anti = 1.0 - 0.05 * d
    s, t, valid = ransac_scale_and_shift(anti, d, d > 0)
    assert valid.item() and t.item() == 0.0
    assert s.item() == pytest.approx((d.mean() / anti.mean()).item(), rel=1e-6)

    # the chunking of the trials does not change the pick
    ref = ransac_scale_and_shift(p, bad, m)
    for kw in (dict(chunk=7), dict(max_elements=3 * 1024)):
        out = ransac_scale_and_shift(p, bad, m, **kw)
        assert all(torch.equal(a, b) for a, b in zip(out, ref)), kw

    # per image: alone or in a batch, the same (s, t)
    g = torch.Generator().manual_seed(3)
    gts = torch.cat([depth_map(i) for i in (13, 14, 15)])
    preds = 0.05 * gts + 0.02 * torch.randn(gts.shape, generator=g)
    batch = ransac_scale_and_shift(preds, gts, gts > 0)
    for b in range(3):
        single = ransac_scale_and_shift(
            preds[b : b + 1], gts[b : b + 1], gts[b : b + 1] > 0
        )
        assert batch[0][b] == single[0][0] and batch[1][b] == single[1][0]

    names = ["k/0", "k/1", "k/2", "k/3"]
    samples = pred_samples(zip(names, [p, anti, p, preds[:1]], [bad, d, d, gts[:1]]))
    results = []
    for _ in range(2):
        evaluator = DepthEvaluator(relative=True, align_protocols=["ransac"])
        evaluate(evaluator, ScaledDepth(), samples)
        results.append(evaluator.results["groups"])
    assert results[0] == results[1]
    assert results[0]["k@ransac"]["num_samples"] == 4
    assert results[0]["k@ransac"]["num_excluded"] == 0


def ransac_samples(hw=(24, 60)):
    """Samples of four groups with their val sets' ranges (depth_min / depth_max): an
    affine prediction with noise and 10x outliers, a flat one (the mean-ratio
    fallback), an anti-correlated one, GT with < 10 valid pixels (excluded), GT
    pixels without depth and beyond the range."""
    g = torch.Generator().manual_seed(21)
    ranges = {
        "nyuv2": (1e-3, 10.0),
        "eth3d": (1e-5, math.inf),
        "DIODE-INDOOR": (0.6, 350.0),
        "DIODE-OUTDOOR": (0.6, 350.0),
    }
    samples = []
    for i, group in enumerate(["nyuv2", "eth3d", "DIODE-INDOOR", "DIODE-OUTDOOR"] * 3):
        gt = depth_map(90 + i, hw)
        gt[torch.rand(gt.shape, generator=g) < 0.2] = 0.0  # no depth
        noisy, _ = with_outliers(gt, 0.1, seed=i)
        kind = i // 4
        if kind == 0:
            pred = 0.05 * noisy + 0.1 + 0.01 * torch.randn(gt.shape, generator=g)
        elif kind == 1:
            pred = torch.full_like(gt, 0.5) if i % 2 else 1.0 - 0.05 * gt
        else:
            pred = 0.2 * noisy**0.9 + 0.3 * torch.rand(gt.shape, generator=g)
            if group == "nyuv2":
                gt = torch.where(torch.arange(gt.numel()).view(gt.shape) < 5, gt, 0)
        lo, hi = ranges[group]
        samples.append(
            dict(
                image=pred.expand(3, *hw).clone(),
                depth=gt,
                name=f"{group}/{i}",
                depth_min=lo,
                depth_max=hi,
            )
        )
    return samples


def test_align_solver_ransac():
    """align_solver="ransac" (the depth recipes' fit, seed 0) gives every group the
    numbers of its "<group>@ransac" twin, bit for bit: with the "ransac" protocol in
    the same evaluator, and logged beside the "lsq_free" fit with an ETH3D column
    subsample (align_max_res) and the other protocols, with flip TTA. The same seeded
    fit on every valid pixel, the same clip to the set's range, the same mean-ratio
    fallback (num_rejected) and exclusions (num_excluded); ransac_seed reaches the
    fit."""
    from ldpt.models.utils.depth import ransac_scale_and_shift

    samples = ransac_samples()
    ransac = DepthEvaluator(
        relative=True, tta=[["flip"]], align_solver="ransac", ransac_seed=0
    )
    evaluate(ransac, ScaledDepth(), samples)
    groups = ransac.results["groups"]
    assert list(groups) == [
        "nyuv2",
        "eth3d",
        "DIODE-INDOOR",
        "DIODE",
        "DIODE-OUTDOOR",
    ]
    assert groups["nyuv2"]["num_excluded"] == 1
    assert sum(m["num_rejected"] for m in groups.values()) > 0
    assert "align_protocols" not in ransac.results
    assert ransac.protocol("eth3d").startswith("align=depth-ransac ")

    both = DepthEvaluator(
        relative=True, tta=[["flip"]], align_solver="ransac", align_protocols=["ransac"]
    )
    logged = DepthEvaluator(
        relative=True,
        tta=[["flip"]],
        align_solver="lsq_free",
        align_max_res=dict(eth3d=12),
        align_protocols=["trim5", "disp", "ransac", "roe"],
    )
    for evaluator in (both, logged):
        evaluate(evaluator, ScaledDepth(), samples)
        twins = evaluator.results["groups"]
        for group, metrics in groups.items():
            assert twins[f"{group}@ransac"] == metrics, group
            assert evaluator.protocol(f"{group}@ransac") == ransac.protocol(group)
    assert both.results["groups"]["eth3d"] == groups["eth3d"]
    assert logged.results["groups"]["eth3d"] != groups["eth3d"]

    p = torch.stack([s["image"][0] for s in samples[:4]])
    d = torch.stack([s["depth"][0] for s in samples[:4]])
    seeded = DepthEvaluator(relative=True, align_solver="ransac", ransac_seed=7)
    out = seeded.align(p, d, d > 0)
    ref = ransac_scale_and_shift(p, d, d > 0, seed=7, return_rejected=True)
    assert all(torch.equal(a, b) for a, b in zip(out, ref))


def test_rejected_fits_never_excluded():
    """A flat and an anti-correlated prediction: the headline "lsq" and every
    protocol score them through their fallback (num_rejected) and exclude nothing,
    so each group keeps its images whatever the model; only GT with < 10 valid
    pixels is excluded."""
    from ldpt.models.utils.depth import ransac_scale_and_shift

    d = depth_map(30)
    flat = torch.full_like(d, 0.5)
    s, t, valid, rejected = ransac_scale_and_shift(flat, d, d > 0, return_rejected=True)
    assert valid.item() and rejected.item() and t.item() == 0.0
    assert s.item() == pytest.approx((d.mean() / 0.5).item(), rel=1e-6)

    sparse = torch.zeros_like(d)
    sparse[0, 0, :5] = 3.0  # 5 valid pixels
    pairs = [
        ("x/0", 0.05 * d + 0.1, d),
        ("x/1", flat, depth_map(31)),
        ("x/2", 1.0 - 0.05 * depth_map(32), depth_map(32)),
        ("x/3", 0.05 * d, sparse),
    ]
    evaluator = DepthEvaluator(
        relative=True, align_solver="lsq", align_protocols=["trim5", "disp", "ransac"]
    )
    evaluate(evaluator, ScaledDepth(), pred_samples(pairs))
    for group, metrics in evaluator.results["groups"].items():
        assert metrics["num_samples"] == 3 and metrics["num_excluded"] == 1, group
        assert metrics["num_rejected"] == 2, group


@pytest.mark.parametrize(
    "kwargs, counted",
    [
        ({}, ["nyuv2", "DIODE-INDOOR", "DIODE-OUTDOOR"]),
        (dict(score_groups=["nyuv2"]), ["nyuv2"]),
        (
            dict(best_metric_mean="arithmetic"),
            ["nyuv2", "DIODE-INDOOR", "DIODE-OUTDOOR"],
        ),
    ],
)
def test_protocol_groups_never_counted(tmp_path, kwargs, counted):
    """Every headline group (and the combined DIODE one) gets "@trim5", "@disp" and
    "@ransac" twins in metrics.jsonl after the headline groups; the score and
    counted see the headline groups only."""
    g = torch.Generator().manual_seed(4)
    pairs = []
    for i, (group, scale) in enumerate(
        [("nyuv2", 1.1), ("nyuv2", 1.1), ("DIODE-INDOOR", 1.5), ("DIODE-OUTDOOR", 1.2)]
    ):
        gt = depth_map(20 + i)
        pred = scale * gt + 0.3 * torch.randn(gt.shape, generator=g)
        pairs.append((f"{group}/{i}", pred, gt))
    protocols = ["trim5", "disp", "ransac"]
    evaluator = DepthEvaluator(
        relative=True, align_solver="lsq", align_protocols=protocols, **kwargs
    )
    evaluator.trainer = fake_trainer(ScaledDepth(), pred_samples(pairs))
    evaluator.trainer.cfg.save_path, evaluator.trainer.cfg.resume = str(tmp_path), False
    evaluator.before_train()
    evaluator.after_epoch()
    (record,) = read_metrics_history(tmp_path / "metrics.jsonl").values()
    headline = ["nyuv2", "DIODE-INDOOR", "DIODE", "DIODE-OUTDOOR"]
    assert list(record["groups"]) == headline + [
        f"{group}@{p}" for p in protocols for group in headline
    ]
    assert record["counted"] == counted
    assert record["align_protocols"] == protocols
    values = [record["groups"][group]["abs_rel"] for group in counted]
    if kwargs.get("best_metric_mean") == "arithmetic":
        expected = sum(values) / len(values)
    else:
        expected = math.prod(values) ** (1 / len(values))
    assert record["score"] == pytest.approx(expected, rel=1e-12)
    for group, metrics in record["groups"].items():
        keys = {"num_samples", "num_excluded", "num_rejected", *DepthEvaluator.METRICS}
        assert set(metrics) == keys
        assert metrics["num_excluded"] == 0 and metrics["num_rejected"] == 0
    assert record["groups"]["DIODE@disp"]["num_samples"] == 2
    # the protocols differ from the headline (noisy prediction)
    assert (
        record["groups"]["nyuv2@disp"]["abs_rel"]
        != record["groups"]["nyuv2"]["abs_rel"]
    )
    # per-image metrics: headline only
    assert set(evaluator.per_image) == {name for name, _, _ in pairs}
    assert evaluator.per_image["nyuv2/0"]["abs_rel"] != pytest.approx(0)


# ------------------------------------------------- the "roe" evaluation protocol


def brute_force_l1(p, g):
    """min over all lines through two points (p_i, g_i) of sum_j |s p_j + t - g_j| /
    g_j, with s > 0 (an exact minimiser of a weighted L1 line fit passes through two
    data points); fp64."""
    p, g = p.double(), g.double()
    i, j = torch.triu_indices(p.numel(), p.numel(), offset=1)
    dp = p[j] - p[i]
    ok = dp.abs() > 1e-12
    s = (g[j] - g[i])[ok] / dp[ok]
    t = g[i][ok] - s * p[i][ok]
    keep = s > 0
    s, t = s[keep], t[keep]
    obj = ((s[:, None] * p[None] + t[:, None] - g[None]).abs() / g[None]).sum(1)
    return obj.min().item()


def test_roe_protocol():
    """The "roe" protocol (MoGe's evaluation fit, roe_eval_scale_and_shift): an
    exact affine map of the target scores 0; on a 64-pixel image the "@roe" AbsRel is
    the brute-force minimum of the 1/g-weighted L1 fit / N; a flat or anti-correlated
    prediction is rejected (s = 0, t = the 1/g-weighted GT median), scored and counted
    in num_rejected; < 10 valid pixels: excluded; "@roe" groups never count."""
    from ldpt.models.utils.depth import roe_eval_scale_and_shift

    evaluator = DepthEvaluator(relative=True, align_protocols=["roe"])
    assert evaluator.protocol("nyuv2@roe").startswith("align=depth-roe ")

    # exact affine map: 0
    d = depth_map(40)
    p = (d - 0.5) / 20.0
    evaluator = DepthEvaluator(
        relative=True, align_solver="lsq", align_protocols=["roe"]
    )
    evaluate(evaluator, ScaledDepth(), pred_samples([("x/0", p, d)]))
    assert evaluator.results["groups"]["x@roe"]["abs_rel"] < 1e-5
    assert evaluator.results["counted"] == ["x"]

    # brute force on 64 pixels (8 x 8) with noise and outliers: L1 optimum
    g = torch.Generator().manual_seed(41)
    d = depth_map(42, hw=(8, 8))
    p = 0.05 * d + 0.1 + 0.03 * torch.randn(d.shape, generator=g)
    p[0, 0, :4] += 1.0  # gross outliers
    s, t, valid, rejected = roe_eval_scale_and_shift(p, d, d > 0, return_rejected=True)
    assert valid.item() and not rejected.item() and s.item() > 0
    obj = ((s.double() * p.double() + t.double() - d).abs() / d).sum().item()
    best = brute_force_l1(p.flatten(), d.flatten())
    assert obj <= best * (1 + 1e-5) + 1e-9
    evaluator = DepthEvaluator(
        relative=True, align_solver="lsq", align_protocols=["roe"]
    )
    evaluate(evaluator, ScaledDepth(), pred_samples([("y/0", p, d)]))
    groups = evaluator.results["groups"]
    assert groups["y@roe"]["abs_rel"] == pytest.approx(best / 64, rel=1e-5)
    assert groups["y@roe"]["abs_rel"] < groups["y"]["abs_rel"]  # LSQ is not L1-optimal

    # rejected fits: flat and anti-correlated -> s = 0, t = weighted GT median
    d2, d3 = depth_map(43), depth_map(44)
    flat, anti = torch.full_like(d2, 0.5), 1.0 - 0.05 * d3
    for pred, gt in ((flat, d2), (anti, d3)):
        s, t, valid, rejected = roe_eval_scale_and_shift(
            pred, gt, gt > 0, return_rejected=True
        )
        assert valid.item() and rejected.item() and s.item() == 0.0
        v, order = gt.flatten().double().sort()
        cw = (1.0 / v).cumsum(0)
        median = v[torch.searchsorted(cw, 0.5 * cw[-1:])].item()
        assert t.item() == pytest.approx(median, rel=1e-6)
        # the weighted median is the best constant of the 1/g-weighted L1
        const = ((t.double() - gt).abs() / gt).sum()
        for c in (median * 0.99, median * 1.01):
            assert const <= ((c - gt.double()).abs() / gt).sum() + 1e-9
    sparse = torch.zeros_like(d2)
    sparse[0, 0, :5] = 3.0  # 5 valid pixels
    s, _, valid, rejected = roe_eval_scale_and_shift(
        p, sparse[..., :8, :8], sparse[..., :8, :8] > 0, return_rejected=True
    )
    assert not valid.item() and not rejected.item() and math.isnan(s.item())

    pairs = [
        ("x/0", 0.05 * d2 + 0.1, d2),
        ("x/1", flat, d2),
        ("x/2", anti, d3),
        ("x/3", 0.05 * d2, sparse),
    ]
    evaluator = DepthEvaluator(
        relative=True,
        align_solver="lsq",
        align_protocols=["trim5", "disp", "ransac", "roe"],
    )
    evaluate(evaluator, ScaledDepth(), pred_samples(pairs))
    groups = evaluator.results["groups"]
    assert list(groups) == ["x", "x@trim5", "x@disp", "x@ransac", "x@roe"]
    assert groups["x@roe"]["num_samples"] == 3 and groups["x@roe"]["num_excluded"] == 1
    assert groups["x@roe"]["num_rejected"] == 2
    assert evaluator.results["counted"] == ["x"]


def test_roe_protocol_never_counted(tmp_path):
    """With the depth recipes' protocols every headline group gets an "@roe" twin
    last in metrics.jsonl; the score and counted see the headline only."""
    g = torch.Generator().manual_seed(5)
    pairs = []
    for i, (group, scale) in enumerate(
        [("nyuv2", 1.1), ("DIODE-INDOOR", 1.5), ("DIODE-OUTDOOR", 1.2)]
    ):
        gt = depth_map(50 + i)
        pred = scale * gt + 0.3 * torch.randn(gt.shape, generator=g)
        pairs.append((f"{group}/{i}", pred, gt))
    protocols = ["trim5", "disp", "ransac", "roe"]
    evaluator = DepthEvaluator(
        relative=True, align_solver="lsq", align_protocols=protocols
    )
    evaluator.trainer = fake_trainer(ScaledDepth(), pred_samples(pairs))
    evaluator.trainer.cfg.save_path, evaluator.trainer.cfg.resume = str(tmp_path), False
    evaluator.before_train()
    evaluator.after_epoch()
    (record,) = read_metrics_history(tmp_path / "metrics.jsonl").values()
    headline = ["nyuv2", "DIODE-INDOOR", "DIODE", "DIODE-OUTDOOR"]
    assert list(record["groups"]) == headline + [
        f"{group}@{p}" for p in protocols for group in headline
    ]
    counted = ["nyuv2", "DIODE-INDOOR", "DIODE-OUTDOOR"]
    assert record["counted"] == counted
    values = [record["groups"][group]["abs_rel"] for group in counted]
    assert record["score"] == pytest.approx(
        math.prod(values) ** (1 / len(values)), rel=1e-12
    )
    for group in headline:
        roe = record["groups"][group + "@roe"]
        assert roe["num_excluded"] == 0 and roe["num_rejected"] == 0
        assert roe["abs_rel"] != record["groups"][group]["abs_rel"]
