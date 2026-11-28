"""
The COCO-Stuff recipes beyond their config keys (CPU, tiny models with the recipes'
structural options):
- recipe 1's MuonAdamW groups: which parameters take the orthogonalised step, which
  stay on AdamW, which have no weight decay;
- the anneal (1a, from recipe 1's epoch 9) keeps its recipe's parameter groups at the
  anneal peaks; OneCycleLR over the one epoch: max_lr / 10 -> max_lr at 3 % ->
  max_lr / 1e4, and the peak is within 10 % of the lr the recipe trained at when the
  start file was written;
- CheckpointLoader(allow_missing=False) loads [DINOv3, epoch file] into the whole model
  and raises when the start file leaves a model key at its init;
- VOC recipe 1's loader: [DINOv3, the COCO-Stuff recipe-1 average] fills every key of
  the 21-class model with 64 pixel features (tap BatchNorm statistics included), the
  head from the rows of the 171-class COCO-Stuff head, and raises on a start file
  without the tap BatchNorms.
"""

import importlib.util
import logging
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from helpers import REPO, init_encoder_, randomize_decoder_
from test_l_dpt_golden import SEG_LOSS

from ldpt.hooks.builder import HOOKS
from ldpt.datasets.meta import COCOSTUFF_CLASS_NAMES, VOC2012_CLASS_NAMES
from ldpt.hooks.misc import VOC2012_FROM_COCOSTUFF, CheckpointLoader, seg_head_from_rows
from ldpt.models.builder import build_model
from ldpt.utils.config import Config, ConfigDict
from ldpt.utils.optimizer import MuonAdamW, build_optimizer, keyword_groups
from ldpt.utils.scheduler import build_scheduler

CONFIGS = os.path.join(REPO, "configs")
R1 = "cocostuff/semseg-l-dpt-v1m1-1-large.py"
R1A = "cocostuff/semseg-l-dpt-v1m1-1a-large-anneal.py"
DINOV3 = "checkpoints/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
ITERS = 3696  # COCO-Stuff train (118,287 images) at batch 32
# anneal -> (recipe, start epoch, start file, keyword-group peaks, the param groups'
# (source keyword group, weight decay, peak))
CASES = {
    R1A: (
        R1,
        9,
        "exp/cocostuff/semseg-l-dpt-v1m1-1-large/model/epoch_9.pth",
        [4e-4, 4e-5, 4e-4],
        [(0, 0.1, 4e-4), (0, 0.1, 4e-4), (1, 0.1, 4e-5), (2, 0.0, 4e-4)],
    ),
}
TINY = dict(
    type="L-DPT-v1m1",
    encoder=dict(type="dinov3_tiny_test"),
    encoder_inter_layers=[0, 1, 2, 3],
    task="semseg",
    num_classes=5,
    criterion=SEG_LOSS,
    dec_dim=64,
    dec_num_heads=4,
    dec_num_layers=(1, 1, 1, 1),
    dec_attn_downscale=2,
)
OPTIONS = ("dec_qk_norm", "dec_qk_tau_max", "tap_bn", "out_dim")


def config(rel):
    return Config.fromfile(os.path.join(CONFIGS, rel))


def tiny(rel, seed=0):
    """A tiny L-DPT with the config's structural options (names as the real model)."""
    cfg = config(rel)
    options = {k: cfg.model[k] for k in OPTIONS if k in cfg.model}
    torch.manual_seed(seed)
    model = build_model(dict(TINY, **options))
    init_encoder_(model.encoder, seed)
    return model


def names_of(model, group):
    ids = {id(p): n for n, p in model.named_parameters()}
    return {ids[id(p)] for p in group["params"]}


def lr_curve(scheduler_cfg, model, optimizer_cfg, param_dicts, total, at=None):
    """lr of every optimizer group at every step (or at the steps in `at`)."""
    opt = build_optimizer(optimizer_cfg, model, param_dicts)
    sched = build_scheduler(ConfigDict(scheduler_cfg, total_steps=total), opt)
    opt.step()  # no gradients: only the optimizer-before-scheduler order matters
    rows = {0: sched.get_last_lr()} if at is None or 0 in at else {}
    for step in range(1, total if at is None else max(at) + 1):
        sched.step()
        if at is None or step in at:
            rows[step] = sched.get_last_lr()
    return opt, rows


# ---------------------------------------------------------------- recipe 1


def test_recipe_1_routing():
    """MuonAdamW over recipe 1's groups: the decoder's attention, MLP and selection
    matrices take the orthogonalised step (default group, weight decay 0.1); the
    QK-norm temperatures, readout, classifier and the frozen encoder stay on AdamW;
    norms, tap BatchNorms, biases and LayerScale have no weight decay."""
    cfg, model = config(R1), tiny(R1)
    opt = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
    assert isinstance(opt, MuonAdamW)
    groups = opt.param_groups
    assert [(g["source"], g["muon"]) for g in groups] == [
        (0, False),
        (0, True),
        (1, False),
        (2, False),
    ]
    adam0, matrices, encoder, no_decay = (names_of(model, g) for g in groups)
    assert groups[1]["weight_decay"] == 0.1 and groups[3]["weight_decay"] == 0.0
    for s in range(4):
        block = f"dec_blocks.{s}.0"
        for part in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj"):
            assert f"{block}.{part}.weight" in matrices
        assert {f"{block}.mlp.fc1.weight", f"{block}.mlp.fc2.weight"} <= matrices
        assert f"neck_blocks.{s}.weight" in matrices
        assert f"{block}.attn.logit_scale" in adam0  # decayed as in the run
        assert {f"{block}.norm1.weight", f"{block}.ls1.gamma"} <= no_decay
        assert f"{block}.attn.out_proj.bias" in no_decay
        assert {f"tap_bn.{s}.weight", f"tap_bn.{s}.bias"} <= no_decay
    assert {"unpatchfy.0.proj.weight", "seg_head.weight"} <= adam0
    assert all(n.startswith("encoder.") for n in encoder)
    assert not any(p.requires_grad for g in groups[2:3] for p in g["params"])
    assert "kept on AdamW, frozen" in opt.routing


# ---------------------------------------------------------------- anneals


@pytest.mark.parametrize("rel", list(CASES))
def test_groups_are_the_recipes(rel):
    """Every parameter in the same keyword group as in its recipe, at the anneal lr."""
    recipe, _, _, peaks, _ = CASES[rel]
    cfg, base, model = config(rel), config(recipe), tiny(rel)
    mine = keyword_groups(model, cfg.param_dicts, cfg.optimizer.lr)
    theirs = keyword_groups(model, base.param_dicts, base.optimizer.lr)
    assert [g["names"] for g in mine] == [g["names"] for g in theirs]
    assert [g.get("weight_decay") for g in mine] == [
        g.get("weight_decay") for g in theirs
    ]
    assert [g["lr"] if "lr" in g else cfg.optimizer.lr for g in mine] == peaks


@pytest.mark.parametrize("rel", list(CASES))
def test_optimizer_groups_and_peaks(rel):
    """The built optimizer (MuonAdamW) and OneCycleLR: group sources, weight decays
    and peaks; the first step at max_lr / 10."""
    _, _, _, _, expected = CASES[rel]
    cfg, model = config(rel), tiny(rel)
    opt = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
    assert isinstance(opt, MuonAdamW)
    sched = build_scheduler(ConfigDict(cfg.scheduler, total_steps=ITERS), opt)
    got = [
        (g.get("source", i), g["weight_decay"], g["max_lr"])
        for i, g in enumerate(opt.param_groups)
    ]
    assert len(got) == len(expected)
    for (s, wd, peak), (s0, wd0, peak0) in zip(got, expected):
        assert s == s0 and wd == wd0 and peak == pytest.approx(peak0, rel=1e-12)
    torch.testing.assert_close(
        sched.get_last_lr(), [p / 10 for _, _, p in expected], rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("rel", list(CASES))
def test_one_epoch_schedule(rel):
    """trainer: total_steps = len(train_loader) x epoch = 3,696; warm-up to the peak in
    3 % of the epoch (111 steps), cosine down to max_lr / 1e4 at the last step."""
    _, _, _, _, expected = CASES[rel]
    cfg = config(rel)
    total = ITERS * cfg.epoch
    _, rows = lr_curve(cfg.scheduler, tiny(rel), cfg.optimizer, cfg.param_dicts, total)
    peaks = [p for _, _, p in expected]
    curve = torch.tensor([rows[s] for s in range(total)], dtype=torch.float64)
    top = int(curve[:, 0].argmax())
    assert abs(top - 0.03 * total) <= 2  # torch's phase end: pct_start * total - 1
    torch.testing.assert_close(
        curve[top], torch.tensor(peaks, dtype=torch.float64), rtol=1e-6, atol=0
    )
    torch.testing.assert_close(
        curve[-1], torch.tensor(peaks, dtype=torch.float64) / 1e4
    )
    diffs = curve[1:, 0] - curve[:-1, 0]
    assert (diffs[:top] > 0).all() and (diffs[top:] < 0).all()


@pytest.mark.parametrize("rel", list(CASES))
def test_peak_is_the_recipes_lr_at_the_start(rel):
    """The peak is within 10 % of the lr the recipe trained at when the start file was
    written (recipe 1 after epoch 9: 3.8e-4)."""
    recipe, start, _, peaks, _ = CASES[rel]
    base = config(recipe)
    total = ITERS * base.epoch
    step = ITERS * start
    _, rows = lr_curve(
        base.scheduler, tiny(recipe), base.optimizer, base.param_dicts, total, at={step}
    )
    group0 = rows[step][0]
    assert group0 == pytest.approx({R1A: 3.8e-4}[rel], rel=0.03)
    assert peaks[0] == pytest.approx(group0, rel=0.1)


# ---------------------------------------------------------------- start-file load


def files(tmp_path, source, extra_encoder_key=True):
    """A DINOv3-style encoder file (plain keys, plus the 7B file's unexpected
    local_cls_norm) and a CheckpointSaver epoch file (DDP keys, no encoder)."""
    encoder = {k: v.clone() for k, v in source.encoder.state_dict().items()}
    if extra_encoder_key:
        encoder["local_cls_norm.weight"] = torch.ones(3)
    epoch = {
        f"module.{k}": v.clone()
        for k, v in source.state_dict().items()
        if not k.startswith("encoder.")
    }
    paths = tmp_path / "dinov3.pth", tmp_path / "epoch_14.pth"
    torch.save(encoder, paths[0])
    torch.save({"epoch": 14, "state_dict": epoch}, paths[1])
    return [str(p) for p in paths], epoch


def run_loader(hook_cfg, model, ckpts):
    hook = HOOKS.build(hook_cfg)
    hook.trainer = SimpleNamespace(
        model=model,
        cfg=SimpleNamespace(checkpoint=ckpts, resume=False),
        logger=logging.getLogger(__name__),
    )
    hook.before_train()
    return hook


@pytest.mark.parametrize("rel", list(CASES))
def test_anneal_loader_fills_the_model(rel, tmp_path):
    """The config's loader: the encoder from file 0, every decoder key from file 1
    (tap BatchNorm statistics included)."""
    source = randomize_decoder_(tiny(rel, seed=1), seed=1)
    if rel == R1A:
        for bn in source.tap_bn:
            bn.running_mean.normal_()
            bn.running_var.uniform_(0.5, 2.0)
            bn.num_batches_tracked.fill_(7)
    ckpts, _ = files(tmp_path, source)
    model = tiny(rel, seed=2)  # another init: everything must be overwritten
    run_loader(config(rel).hooks[0], model, ckpts)
    want = source.state_dict()
    for k, v in model.state_dict().items():
        assert torch.equal(v, want[k]), k


def test_loader_raises_on_a_start_file_that_leaves_keys_at_init(tmp_path):
    """allow_missing=False: the epoch file of a decoder without QK-norm and tap
    BatchNorms (no logit_scale / tap_bn) into the recipe-1 model, or a file without the
    decoder, raises; the default only logs."""
    torch.manual_seed(1)
    plain = build_model(
        dict(TINY, out_dim=64)
    )  # recipe 1's readout, no QK-norm / tap_bn
    init_encoder_(plain.encoder, 1)
    plain = randomize_decoder_(plain, seed=1)
    one_hook = config(R1A).hooks[0]
    assert one_hook.allow_missing is False
    plain_files, _ = files(tmp_path, plain)
    with pytest.raises(RuntimeError, match="allow_missing=False.*logit_scale"):
        run_loader(one_hook, tiny(R1A, seed=2), plain_files)
    # the DINOv3 file twice (a wrong second path): the whole decoder stays at init
    with pytest.raises(RuntimeError, match="allow_missing=False"):
        run_loader(one_hook, tiny(R1A, seed=3), [plain_files[0]] * 2)
    # default (allow_missing=True): the missing keys are only logged
    hook = dict(one_hook)
    hook.pop("allow_missing")
    run_loader(hook, tiny(R1A, seed=3), [plain_files[0]] * 2)


@pytest.mark.parametrize("value", [None, 0, "False"])
def test_allow_missing_must_be_bool(value):
    with pytest.raises(ValueError, match="allow_missing"):
        CheckpointLoader(allow_missing=value)


def test_unexpected_keys_stay_allowed(tmp_path):
    model = nn.Module()
    model.encoder = nn.Linear(2, 2)
    model.head = nn.Linear(2, 3)
    torch.save(
        {"weight": torch.ones(2, 2), "bias": torch.ones(2), "x": torch.ones(1)},
        tmp_path / "enc.pth",
    )
    torch.save(
        {"module.head.weight": torch.ones(3, 2), "module.head.bias": torch.ones(3)},
        tmp_path / "dec.pth",
    )
    run_loader(
        dict(
            type="CheckpointLoader",
            keywords=["module.", "module."],
            replacement=["module.encoder.", "module."],
            allow_missing=False,
        ),
        model,
        [str(tmp_path / "enc.pth"), str(tmp_path / "dec.pth")],
    )
    assert torch.equal(model.head.weight, torch.ones(3, 2))
    assert torch.equal(model.encoder.bias, torch.ones(2))


# ---------------------------------------------------------------- VOC recipe 1

VOC_1 = ["voc2012/semseg-l-dpt-v1m1-1-large.py"]


def tiny_with_classes(rel, num_classes, seed):
    """tiny(rel) with another head width (171: COCO-Stuff, 21: VOC)."""
    cfg = config(rel)
    options = {k: cfg.model[k] for k in OPTIONS if k in cfg.model}
    torch.manual_seed(seed)
    model = build_model(dict(TINY, **options, num_classes=num_classes))
    init_encoder_(model.encoder, seed)
    return model


def average_script():
    path = os.path.join(REPO, "scripts", "average_weights.py")
    spec = importlib.util.spec_from_file_location("average_weights_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coco_average(tmp_path, rel, drop=()):
    """[DINOv3-style encoder file, average.pth]: scripts/average_weights.py over two
    decoder-only epoch files (DDP keys) of the 171-class COCO-Stuff model with rel's
    structural options and random tap BatchNorm statistics; keys starting with one of
    `drop` (model-relative) are left out of both epoch files."""
    paths = []
    for seed, name in ((1, "epoch_9.pth"), (2, "epoch_1.pth")):
        coco = randomize_decoder_(tiny_with_classes(rel, 171, seed), seed=seed)
        for bn in coco.tap_bn:
            bn.running_mean.normal_()
            bn.running_var.uniform_(0.5, 2.0)
            bn.num_batches_tracked.fill_(3 + seed)
        state = {
            f"module.{k}": v.clone()
            for k, v in coco.state_dict().items()
            if not k.startswith(("encoder.",) + tuple(drop))
        }
        paths.append(tmp_path / name)
        torch.save({"epoch": 9 if seed == 1 else 1, "state_dict": state}, paths[-1])
    encoder = {k: v.clone() for k, v in coco.encoder.state_dict().items()}
    torch.save(encoder, tmp_path / "dinov3.pth")
    out = tmp_path / "anneal" / "model" / "average.pth"
    argv = ["--inputs", *map(str, paths), "--output", str(out)]
    assert average_script().main(argv) == 0
    return [str(tmp_path / "dinov3.pth"), str(out)], encoder


@pytest.mark.parametrize("rel", VOC_1)
def test_voc_1_loader_fills_the_model(rel, tmp_path):
    """VOC recipe 1's hook 0 (seg_head_rows, allow_missing=False) on [DINOv3, the
    COCO-Stuff recipe-1 average]: the encoder from file 0; every decoder key from the
    average, the tap BatchNorms' running statistics and counter included; the 21-class
    head (64 input channels) from the rows of the 171-class COCO-Stuff head."""
    cfg = config(rel)
    assert cfg.model.out_dim == 64 and cfg.model.tap_bn and cfg.model.dec_qk_norm
    loader = cfg.hooks[0]
    assert loader.allow_missing is False and loader.seg_head_rows is not None
    ckpts, encoder = coco_average(tmp_path, rel)
    average = {
        k[len("module.") :]: v
        for k, v in torch.load(ckpts[1], weights_only=True)["state_dict"].items()
    }
    model = tiny_with_classes(rel, 21, seed=5)  # another init: all must be overwritten
    assert model.seg_head.weight.shape == (21, 64, 1, 1)
    run_loader(loader, model, ckpts)
    names = (VOC2012_CLASS_NAMES, COCOSTUFF_CLASS_NAMES, VOC2012_FROM_COCOSTUFF)
    for k, v in model.state_dict().items():
        if k.startswith("encoder."):
            want = encoder[k[len("encoder.") :]]
        elif k.startswith("seg_head."):
            want = seg_head_from_rows(average[k], *names)
        else:
            want = average[k]
        assert torch.equal(v, want), k
    for i, bn in enumerate(model.tap_bn):
        assert bn.num_batches_tracked.item() == 5  # from the last input (seed 2)
        assert not torch.equal(bn.running_mean, torch.zeros_like(bn.running_mean)), i


def test_voc_1_loader_raises_without_tap_bn(tmp_path):
    """allow_missing=False: a start file without the tap BatchNorms would leave them at
    their init; the load raises instead."""
    rel = VOC_1[0]
    ckpts, _ = coco_average(tmp_path, rel, drop=("tap_bn.",))
    with pytest.raises(RuntimeError, match="allow_missing=False.*tap_bn"):
        run_loader(config(rel).hooks[0], tiny_with_classes(rel, 21, seed=5), ckpts)
