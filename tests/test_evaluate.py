"""
Offline evaluation (scripts/evaluate.py, ldpt/engines/evaluate.py), CPU, tiny DINOv3.

A 2-epoch run with evaluation and CheckpointSaver(skip_keywords=["encoder"],
keep_epoch_weights=True) writes decoder-only epoch_<N>.pth files and metrics.jsonl.
The script, given the run's config.py and those files (the encoder comes from the
config's checkpoint), must reproduce the metrics the run logged for each epoch.
"""

import importlib.util
import json
import math
import os

import pytest
import torch

from helpers import REPO, DINOv3TinyTest, init_encoder_

from ldpt.datasets.builder import DATASETS
from ldpt.engines.default import default_config_parser, default_setup
from ldpt.engines.evaluate import (
    average_states,
    encoder_sources,
    load_decoder,
    summarize,
)
from ldpt.engines.trainer import build_trainer
from ldpt.hooks.evaluator import read_metrics_history
from ldpt.hooks.misc import load_checkpoint
from ldpt.models.builder import build_model
from ldpt.utils.config import Config

CONFIG = """
_base_ = ["{base}"]

checkpoint = "{pretrained}"
enable_wandb = False
evaluate = True
seed = 0
epoch = 2
batch_size = 2
num_workers = 0

model = dict(
    type="L-DPT-v1m1",
    encoder=dict(type="dinov3_tiny_test"),
    encoder_inter_layers=[0, 1, 2, 3],
    encoder_weight_dtype="bf16",
    dec_dim=32,
    dec_num_heads=2,
    dec_num_layers=(1, 1, 1, 1),
    dec_attn_downscale=2,
    precision="bf16",
    criterion=dict(type="relative_depth_loss", roe_points=64),
)

base_lr = 1e-3
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.1)
scheduler = dict(
    type="OneCycleLR",
    max_lr=base_lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

data = dict(
    train=dict(type="EvalTestDataset", length=4),
    val=dict(type="EvalTestDataset", length=6, groups=("nyuv2", "kitti"), seed=1),
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module.encoder."),
    dict(type="ModelHook"),
    dict(type="DepthEvaluator", relative=True, tta=[["flip"]]),
    dict(type="CheckpointSaver", skip_keywords=["encoder"], keep_epoch_weights=True),
]
"""


@DATASETS.register_module("EvalTestDataset", force=True)
class EvalTestDataset(torch.utils.data.Dataset):
    def __init__(self, length=4, hw=(32, 48), groups=("train",), seed=0):
        g = torch.Generator().manual_seed(seed)
        self.image = torch.randn(length, 3, *hw, generator=g)
        self.depth = torch.rand(length, 1, *hw, generator=g) * 10 + 0.5
        self.groups = groups

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        name = f"{self.groups[idx % len(self.groups)]}/{idx}"
        return dict(image=self.image[idx], depth=self.depth[idx], name=name)

    def info(self):
        return f"{len(self)} synthetic samples"


def load_script():
    path = os.path.join(REPO, "scripts", "evaluate.py")
    spec = importlib.util.spec_from_file_location("evaluate_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A finished 2-epoch run: (save_path, pretrained encoder file)."""
    root = tmp_path_factory.mktemp("eval")
    pretrained = root / "dinov3_tiny_pretrain.pth"
    torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), pretrained)
    config = root / "config_src.py"
    config.write_text(
        CONFIG.format(
            base=f"{REPO}/configs/_base_/default_runtime.py", pretrained=pretrained
        )
    )
    save_path = root / "exp"
    cfg = default_setup(
        default_config_parser(str(config), dict(save_path=str(save_path)))
    )
    build_trainer(dict(**cfg.train, cfg=cfg)).train()
    return save_path, pretrained


def test_evaluate_script(run, tmp_path):
    save_path, pretrained = run
    model_dir = save_path / "model"
    history = read_metrics_history(save_path / "metrics.jsonl")
    assert list(history) == [1, 2]
    for epoch in (1, 2):
        weights = load_checkpoint(model_dir / f"epoch_{epoch}.pth")
        assert set(weights) == {"epoch", "state_dict"}
        assert not any("encoder" in k for k in weights["state_dict"])

    output = tmp_path / "eval.json"
    ckpts = [
        model_dir / name for name in ("model_best.pth", "epoch_2.pth", "epoch_1.pth")
    ]
    load_script().main(
        ["--config-file", str(save_path / "config.py"), "--checkpoints"]
        + [str(p) for p in ckpts]
        + ["--output", str(output), "--per-image"]
    )
    out = json.loads(output.read_text())
    assert out["encoder"] == [str(pretrained)]
    assert out["evaluator"]["type"] == "DepthEvaluator"
    assert out["evaluator"]["tta"] == [["flip"]]
    assert set(out["evaluator"]["protocol"]) == {"nyuv2", "kitti"}
    results = out["results"]
    names = [os.path.basename(r["checkpoint"]) for r in results]
    assert names == ["epoch_1.pth", "epoch_2.pth", "model_best.pth"]  # natural order
    # each epoch file reproduces what the run logged after that epoch
    for record, epoch in zip(results[:2], (1, 2)):
        assert record["epoch"] == epoch
        logged = history[epoch]
        assert record["score"] == pytest.approx(logged["score"], rel=1e-6)
        assert record["counted"] == logged["counted"] == ["nyuv2", "kitti"]
        for group, metrics in logged["groups"].items():
            assert record["groups"][group] == pytest.approx(metrics, rel=1e-6)
        assert len(record["per_image"]) == 6
    best_epoch = min(history, key=lambda e: history[e]["score"])  # abs_rel: lowest
    assert results[2]["epoch"] == best_epoch
    assert results[2]["groups"] == results[best_epoch - 1]["groups"]
    assert out["summary"]["best_score"]["epoch"] == best_epoch
    assert set(out["summary"]["best_per_group"]) == {"nyuv2", "kitti"}

    # --encoder: the DINOv3 file directly; one checkpoint, default output path
    load_script().main(
        ["--config-file", str(save_path / "config.py")]
        + [
            "--checkpoints",
            str(model_dir / "epoch_1.pth"),
            "--encoder",
            str(pretrained),
        ]
    )
    out = json.loads((model_dir / "eval.json").read_text())
    assert out["results"][0]["groups"] == results[0]["groups"]


def test_evaluate_script_on_a_release_file(run, tmp_path):
    """A release file (convert.py output of an epoch file) evaluates to the epoch's
    logged metrics with the run's config (docs/evaluation.md)."""
    from ldpt.models.l_dpt import convert
    from ldpt.utils.config import Config

    save_path, _ = run
    release = tmp_path / "release.pth"
    model_cfg = Config.fromfile(str(save_path / "config.py")).model
    convert.convert(save_path / "model" / "epoch_2.pth", release, config=model_cfg)
    output = tmp_path / "eval.json"
    load_script().main(
        ["--config-file", str(save_path / "config.py"), "--checkpoints", str(release)]
        + ["--output", str(output)]
    )
    (record,) = json.loads(output.read_text())["results"]
    logged = read_metrics_history(save_path / "metrics.jsonl")[2]
    assert record["score"] == pytest.approx(logged["score"], rel=1e-6)
    for group, metrics in logged["groups"].items():
        assert record["groups"][group] == pytest.approx(metrics, rel=1e-6)


def test_evaluate_script_average(run, tmp_path):
    """--average: one record for the mean weights of the given checkpoints."""
    save_path, _ = run
    model_dir = save_path / "model"
    e1, e2 = (load_checkpoint(model_dir / f"epoch_{n}.pth") for n in (1, 2))
    mean = average_states([model_dir / "epoch_1.pth", model_dir / "epoch_2.pth"])
    assert mean.keys() == e1["state_dict"].keys()
    for k, v in mean.items():
        a, b = e1["state_dict"][k], e2["state_dict"][k]
        assert v.dtype == a.dtype
        assert torch.allclose(v, (a.float() + b.float()) / 2)

    def run_script(*names):
        output = tmp_path / "eval.json"
        ckpts = [str(model_dir / n) for n in names]
        load_script().main(
            ["--config-file", str(save_path / "config.py"), "--checkpoints", *ckpts]
            + ["--average", "--output", str(output)]
        )
        (record,) = json.loads(output.read_text())["results"]
        assert record["averaged"] == ckpts and record["epoch"] is None
        return record

    both = run_script("epoch_1.pth", "epoch_2.pth")
    assert both["checkpoint"] == "average of 2"
    # the average of a file with itself is the file
    same = run_script("epoch_1.pth", "epoch_1.pth")
    logged = read_metrics_history(save_path / "metrics.jsonl")[1]
    assert same["score"] == pytest.approx(logged["score"], rel=1e-6)


def test_load_decoder(run, tmp_path):
    save_path, _ = run
    cfg = Config.fromfile(str(save_path / "config.py"))
    model = build_model(cfg.model)
    last = load_checkpoint(save_path / "model" / "model_last.pth")
    assert load_decoder(model, save_path / "model" / "model_last.pth") == 2
    # a plain state dict with the DDP prefix
    state = {"module." + k: v for k, v in last["state_dict"].items()}
    torch.save(state, tmp_path / "plain.pth")
    assert load_decoder(model, tmp_path / "plain.pth") is None
    for k, v in last["state_dict"].items():
        assert torch.equal(model.state_dict()[k], v)
    # a missing decoder key or an unknown key is an error
    del state["module.unpatchfy.0.proj.weight"]
    torch.save(state, tmp_path / "missing.pth")
    with pytest.raises(RuntimeError, match="unpatchfy.0.proj.weight"):
        load_decoder(model, tmp_path / "missing.pth")
    torch.save(dict(last["state_dict"], extra=torch.zeros(1)), tmp_path / "extra.pth")
    with pytest.raises(RuntimeError, match="extra"):
        load_decoder(model, tmp_path / "extra.pth")


def test_encoder_sources():
    """The encoder comes from the DINOv3 file, also in VOC (not the COCO decoder)."""
    dinov3 = "checkpoints/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
    for rel in (
        "hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py",
        "hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal.py",
        "cocostuff/semseg-l-dpt-v1m1-1-large.py",
        "voc2012/semseg-l-dpt-v1m1-1-large.py",
    ):
        cfg = Config.fromfile(os.path.join(REPO, "configs", rel))
        paths, keywords, replacements = encoder_sources(cfg)
        assert paths == [dinov3], rel
        assert (keywords, replacements) == (["module."], ["module.encoder."]), rel


def test_summarize():
    def record(name, epoch, a, b):
        groups = dict(a=dict(abs_rel=a), b=dict(abs_rel=b))
        score = (a * b) ** 0.5
        return dict(
            checkpoint=name, epoch=epoch, score=score, counted=["a", "b"], groups=groups
        )

    records = [
        record("e1", 1, 0.1, 0.5),
        record("e2", 2, 0.3, 0.2),
        record("e3", 3, 0.3, 0.4),
    ]
    summary = summarize(records, "abs_rel", greater=False)
    assert summary["best_score"]["checkpoint"] == "e1"
    assert summary["best_per_group"]["a"] == dict(checkpoint="e1", epoch=1, value=0.1)
    assert summary["best_per_group"]["b"] == dict(checkpoint="e2", epoch=2, value=0.2)
    assert [r["checkpoint"] for r in summary["pareto_front"]] == ["e1", "e2"]


@DATASETS.register_module("EvalSegTestDataset", force=True)
class EvalSegTestDataset(torch.utils.data.Dataset):
    def __init__(self, length=4, hw=(32, 48), num_classes=5):
        g = torch.Generator().manual_seed(0)
        self.image = torch.randn(length, 3, *hw, generator=g)
        self.segment = torch.randint(0, num_classes, (length, 1, *hw), generator=g)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        return dict(image=self.image[idx], segment=self.segment[idx], name=f"seg/{idx}")

    def info(self):
        return f"{len(self)} synthetic samples"


SEG_CONFIG = """
_base_ = ["{base}"]

checkpoint = "{pretrained}"
num_workers = 0
model = dict(
    type="L-DPT-v1m1",
    task="semseg",
    num_classes=5,
    encoder=dict(type="dinov3_tiny_test"),
    encoder_inter_layers=[0, 1, 2, 3],
    dec_dim=32,
    dec_num_heads=2,
    dec_num_layers=(1, 1, 1, 1),
    dec_attn_downscale=2,
    criterion=[dict(type="ce_loss", loss_weight=1.0, ignore_index=-1)],
)
data = dict(val=dict(type="EvalSegTestDataset"))
hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module.encoder."),
    dict(type="SemSegEvaluator", num_classes=5, tta=[["flip"]]),
]
"""


def test_evaluate_script_semseg(tmp_path):
    """SemSegEvaluator configs: one group "val", mIoU as the score."""
    pretrained = tmp_path / "dinov3_tiny_pretrain.pth"
    torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), pretrained)
    config = tmp_path / "config.py"
    config.write_text(
        SEG_CONFIG.format(
            base=f"{REPO}/configs/_base_/default_runtime.py", pretrained=pretrained
        )
    )
    model = build_model(Config.fromfile(str(config)).model)
    decoder = {k: v for k, v in model.state_dict().items() if "encoder" not in k}
    torch.save(dict(epoch=3, state_dict=decoder), tmp_path / "epoch_3.pth")
    output = tmp_path / "eval.json"
    load_script().main(
        ["--config-file", str(config), "--checkpoints", str(tmp_path / "epoch_3.pth")]
        + ["--output", str(output)]
    )
    (record,) = json.loads(output.read_text())["results"]
    assert record["epoch"] == 3 and record["counted"] == ["val"]
    assert record["score"] == record["groups"]["val"]["miou"]
    assert record["groups"]["val"]["num_samples"] == 4
    assert len(record["per_class_iou"]) == 5

    # 5 scales x flip through --options (hooks.4.tta=...)
    ms = "hooks.1.tta=[[flip]" + "".join(
        f",[scale@{r}],[scale@{r},flip]" for r in (0.9, 0.95, 1.05, 1.1)
    )
    ms += "]"
    load_script().main(
        ["--config-file", str(config), "--checkpoints", str(tmp_path / "epoch_3.pth")]
        + ["--output", str(output), "--options", ms]
    )
    (ms_record,) = json.loads(output.read_text())["results"]
    assert ms_record["groups"]["val"]["num_samples"] == 4
    assert ms_record["score"] != record["score"]  # not the flip-only evaluation


# a run like the depth recipes: exp head with the fp32 readout and an fp32 attention
# island, the L-DPT depth loss with label types, gradient clipping (logs grad_norm),
# the aux_scale warm-up gated on drop_frac and the sensitivity protocols
RECIPE_CHANGES = (
    (
        'criterion=dict(type="relative_depth_loss", roe_points=64),',
        """head="exp",
    readout_fp32=True,
    attn_fp32="last",
    criterion=dict(
        type="relative_depth_loss",
        align="roe",
        head="exp",
        min_valid=1000,
        roe_points=256,
        label_types=dict(
            A=["hypersim"], B=["blendedmvs"], C=["nyuv2", "kitti"]
        ),
        type_trim=dict(A=0.0, B=0.05, C=0.05),
        local_levels=dict(A=[4, 16, 64], B=[4, 16], C=[4]),
        local_cell=128.0,
        grad_types=["A"],
        mage_weight=4.0,
        mage_num_scales=4,
        log_logit_stats=True,
    ),""",
    ),
    # one training image per label type (and a second type-A one)
    (
        'train=dict(type="EvalTestDataset", length=4),',
        'train=dict(type="EvalTestDataset", length=4, '
        'groups=("hypersim", "blendedmvs", "nyuv2")),',
    ),
    ("num_workers = 0\n", "num_workers = 0\nclip_grad = 1.0\n"),
    (
        '    dict(type="DepthEvaluator", relative=True, tta=[["flip"]]),',
        """    dict(type="InformationWriter", log_interval=1),
    dict(
        type="DepthEvaluator",
        relative=True,
        tta=[["flip"]],
        align_solver="lsq",
        align_protocols=["trim5", "disp", "ransac", "roe"],
    ),""",
    ),
    (
        '    dict(type="CheckpointSaver", skip_keywords=["encoder"], '
        "keep_epoch_weights=True),\n",
        """    dict(type="CheckpointSaver", skip_keywords=["encoder"], keep_epoch_weights=True),
    dict(
        type="CriterionWeightWarmup",
        key="aux_scale",
        gate_key="drop_frac",
        min_hold=1,
        ramp=2,
        gate_depth_loss=100.0,
        gate_clamp_frac=1.0,
        max_hold=10,
        log_interval=1,
    ),
""",
    ),
)
PROTOCOLS = ["trim5", "disp", "ransac", "roe"]


@pytest.fixture(scope="module")
def run_recipe(tmp_path_factory):
    """A finished 2-epoch run with the exp-head model, the L-DPT depth loss, its
    warm-up and the sensitivity protocols."""
    root = tmp_path_factory.mktemp("eval_recipe")
    pretrained = root / "dinov3_tiny_pretrain.pth"
    torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), pretrained)
    text = CONFIG.format(
        base=f"{REPO}/configs/_base_/default_runtime.py", pretrained=pretrained
    )
    for old, new in RECIPE_CHANGES:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    config = root / "config_src.py"
    config.write_text(text)
    save_path = root / "exp"
    cfg = default_setup(
        default_config_parser(str(config), dict(save_path=str(save_path)))
    )
    trainer = build_trainer(dict(**cfg.train, cfg=cfg))
    trainer.train()
    return save_path, trainer


def test_evaluate_script_recipe_run(run_recipe, tmp_path):
    """The run trains (finite loss, every ROE_KEYS key and grad_norm logged, the
    warm-up ramps aux_scale 0 -> 1 over the 4 steps), logs the sensitivity groups,
    which never count, and scripts/evaluate.py reproduces every group of every epoch
    (the seeded "ransac" and the deterministic "roe" protocols exactly). The tiny
    model is uncorrelated with the random GT: some fits are rejected; they are scored
    by the fallback (num_rejected), never excluded, so every group keeps its 3
    images in every epoch."""
    from ldpt.models.losses.depth import RelativeDepthLoss

    save_path, trainer = run_recipe
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    assert model.head == "exp" and model.criterion.align == "roe"
    assert model.criterion.aux_scale == 1.0  # ramp finished
    (warmup,) = [
        h for h in trainer.hooks if type(h).__name__ == "CriterionWeightWarmup"
    ]
    assert warmup.opened and warmup.progress >= 1.0

    history = read_metrics_history(save_path / "metrics.jsonl")
    assert list(history) == [1, 2]
    groups_expected = {"nyuv2", "kitti"} | {
        f"{g}@{p}" for g in ("nyuv2", "kitti") for p in PROTOCOLS
    }
    for record in history.values():
        assert set(record["groups"]) == groups_expected
        assert record["counted"] == ["nyuv2", "kitti"]
        assert record["align_protocols"] == PROTOCOLS
        for group, metrics in record["groups"].items():
            assert metrics["num_excluded"] == 0 and metrics["num_samples"] == 3
            assert 0 <= metrics["num_rejected"] <= 3

    (writer,) = [h for h in trainer.hooks if type(h).__name__ == "InformationWriter"]
    keys = set(writer.model_output_keys)
    assert set(RelativeDepthLoss.ROE_KEYS) <= keys and "grad_norm" in keys
    assert {"logit_mean", "logit_std", "logit_absmax"} <= keys
    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    )
    events = event_accumulator.EventAccumulator(str(save_path))
    events.Reload()
    for key in ("loss", "depth_loss", "local_loss", "mage_loss", "drop_frac"):
        values = [e.value for e in events.Scalars(f"train_batch/{key}")]
        assert len(values) == 4 and all(math.isfinite(v) for v in values), key
    grad_norm = [e.value for e in events.Scalars("train/grad_norm")]
    assert len(grad_norm) == 2 and all(0 < v < float("inf") for v in grad_norm)
    assert len(events.Scalars("train_batch/grad_norm")) == 4  # 2 steps per epoch
    # gm_image_frac: the 2 type-A images of the 4 (hypersim/0, hypersim/3)
    assert [e.value for e in events.Scalars("train/gm_image_frac")] == [0.5, 0.5]
    # val/nyuv2@roe/abs_rel: TensorBoard writes "@" in a tag as "_" (wandb keeps it)
    assert "val/nyuv2_roe/abs_rel" in events.Tags()["scalars"]

    output = tmp_path / "eval.json"
    ckpts = [save_path / "model" / f"epoch_{epoch}.pth" for epoch in (1, 2)]
    load_script().main(
        ["--config-file", str(save_path / "config.py"), "--checkpoints"]
        + [str(p) for p in ckpts]
        + ["--output", str(output)]
    )
    out = json.loads(output.read_text())
    for record, epoch in zip(out["results"], (1, 2)):
        logged = history[epoch]
        assert record["counted"] == logged["counted"]
        assert record["score"] == pytest.approx(logged["score"], rel=1e-6)
        assert set(record["groups"]) == set(logged["groups"])
        for group, metrics in logged["groups"].items():
            assert record["groups"][group] == pytest.approx(metrics, rel=1e-6)
            if group.endswith(("@ransac", "@roe")):
                assert record["groups"][group] == metrics
    protocol = out["evaluator"]["protocol"]
    assert set(protocol) == set(history[1]["groups"])
    assert protocol["nyuv2"].startswith("align=depth-lsq ")
    assert protocol["kitti@ransac"].startswith("align=depth-ransac ")
    assert protocol["kitti@roe"].startswith("align=depth-roe ")

    # the evaluator is hook 3 here (InformationWriter is 2)
    load_script().main(
        ["--config-file", str(save_path / "config.py"), "--checkpoints", str(ckpts[0])]
        + ["--output", str(output), "--options", "hooks.3.align_protocols=[]"]
    )
    (record,) = json.loads(output.read_text())["results"]
    assert set(record["groups"]) == {"nyuv2", "kitti"}
    assert "align_protocols" not in record
    assert record["score"] == pytest.approx(history[1]["score"], rel=1e-6)
