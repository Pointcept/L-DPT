"""
Resume with CheckpointSaver(skip_keywords=["encoder"]) (CPU smoke, tiny DINOv3).

model_last.pth then holds no encoder keys. Resuming the way train.sh -r auto does
(saved config.py, resume=True, checkpoint=model_last.pth) must load the decoder and
the training state strictly from model_last.pth and the frozen encoder from the
pretrained file the fresh run started from.

With CheckpointSaver(save_freq_iter=N), a run that crashes mid-epoch resumes at the
last saved iteration: same remaining batches, and (deterministic model) the same final
weights, optimizer and scheduler state as a run that never stopped.
"""

import shutil

import pytest
import torch

from helpers import REPO, DINOv3TinyTest, init_encoder_

from ldpt.datasets.builder import DATASETS
from ldpt.engines.default import default_config_parser, default_setup
from ldpt.engines.trainer import ResumableSampler, build_trainer
from ldpt.hooks.builder import HOOKS
from ldpt.hooks.default import HookBase
from ldpt.utils.config import Config
from ldpt.hooks.misc import (
    CheckpointLoader,
    CheckpointSaver,
    InformationWriter,
    load_checkpoint,
)
from ldpt.models.losses import LOSSES

CONFIG = """
_base_ = ["{base}"]

checkpoint = "{pretrained}"
enable_wandb = False
evaluate = False
seed = 0
epoch = 2
batch_size = 2
num_workers = 0

model = dict(
    type="L-DPT-v1m1",
    encoder=dict(type="dinov3_tiny_test"),
    encoder_inter_layers=[0, 1, 2, 3],
    dec_dim=32,
    dec_num_heads=2,
    dec_num_layers=(1, 1, 1, 1),
    dec_attn_downscale=2,
    criterion=dict(type="relative_depth_loss", roe_points=64),
)

base_lr = 1e-3
param_dicts = [dict(keyword="encoder", lr=base_lr / 10)]
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.1)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

data = dict(train=dict(type="ResumeTestDataset"))

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module.encoder."),
    dict(type="ModelHook"),
    dict(type="CheckpointSaver", save_freq=None, skip_keywords=["encoder"]),
    dict(type="StopAfterEpochTest", epoch=1),
]
"""


@DATASETS.register_module("ResumeTestDataset", force=True)
class ResumeTestDataset(torch.utils.data.Dataset):
    def __init__(self, length=4, hw=(32, 48)):
        g = torch.Generator().manual_seed(0)
        self.image = torch.randn(length, 3, *hw, generator=g)
        self.depth = torch.rand(length, 1, *hw, generator=g) * 10 + 0.5

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        return dict(image=self.image[idx], depth=self.depth[idx], index=idx)

    def info(self):
        return f"{len(self)} synthetic samples"


@LOSSES.register_module("DeterministicDepthLossTest", force=True)
class DeterministicDepthLoss(torch.nn.Module):
    """Mean absolute error over the valid pixels: a loss without random numbers, for
    the bitwise mid-epoch resume tests."""

    def forward(self, pred, target, mask):
        err = torch.where(mask, (pred - target).abs(), 0.0)
        return dict(loss=err.sum() / mask.sum().clamp_min(1))


@HOOKS.register_module("StopAfterEpochTest", force=True)
class StopAfterEpoch(HookBase):
    """Simulate a crash right after `epoch` was saved (the trainer exits)."""

    def __init__(self, epoch):
        self.epoch = epoch

    def after_epoch(self):
        if self.trainer.epoch + 1 == self.epoch:
            raise RuntimeError("simulated crash")


@HOOKS.register_module("CrashAtTest", force=True)
class CrashAt(HookBase):
    """Simulate a crash after training batch `iter` of `epoch` (fresh run only)."""

    def __init__(self, epoch, iter):
        self.epoch = epoch
        self.iter = iter

    def after_step(self):
        at = (self.trainer.epoch, self.trainer.comm_info["iter"])
        if not self.trainer.cfg.resume and at == (self.epoch, self.iter):
            raise RuntimeError("simulated crash")


@HOOKS.register_module("RecordBatchesTest", force=True)
class RecordBatches(HookBase):
    """(epoch, iter, sample indices) of every trained batch."""

    def __init__(self):
        self.batches = []

    def before_step(self):
        info = self.trainer.comm_info
        index = info["input_dict"]["index"].tolist()
        self.batches.append((self.trainer.epoch, info["iter"], index))


def build(config_file, **options):
    """train.py main() + main_worker() without launching or training."""
    cfg = default_setup(default_config_parser(str(config_file), options))
    return build_trainer(dict(**cfg.train, cfg=cfg))


def build_resume(save_path, checkpoint=None):
    """What train.sh -r auto runs when model_last.pth exists."""
    checkpoint = checkpoint or save_path / "model" / "model_last.pth"
    return build(
        save_path / "config.py",
        save_path=str(save_path),
        resume=True,
        checkpoint=str(checkpoint),
    )


def find_hook(trainer, cls):
    return next(h for h in trainer.hooks if isinstance(h, cls))


def checkpoint_loader(trainer):
    return find_hook(trainer, CheckpointLoader)


def assert_state_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


@pytest.fixture(scope="module")
def fresh_run(tmp_path_factory):
    """Fresh run from a pretrained encoder file, crashed after epoch 1 of 2."""
    root = tmp_path_factory.mktemp("resume")
    pretrained = root / "dinov3_tiny_pretrain.pth"
    torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), pretrained)
    config = root / "config_src.py"
    config.write_text(
        CONFIG.format(
            base=f"{REPO}/configs/_base_/default_runtime.py", pretrained=pretrained
        )
    )
    save_path = root / "exp"
    trainer = build(config, save_path=str(save_path))
    with pytest.raises(SystemExit):  # ExceptionWriter turns the crash into exit(1)
        trainer.train()
    encoder = torch.load(pretrained, map_location="cpu", weights_only=True)
    assert_state_equal(trainer.model.encoder.state_dict(), encoder)
    return save_path, encoder


@pytest.fixture
def save_path(fresh_run, tmp_path):
    """A private copy of the crashed experiment directory."""
    return shutil.copytree(fresh_run[0], tmp_path / "exp")


def test_resume_skipped_encoder(fresh_run, save_path):
    encoder = fresh_run[1]
    last = load_checkpoint(save_path / "model" / "model_last.pth")
    assert last["epoch"] == 1 and "iter" not in last  # epoch checkpoint
    assert last["state_dict"] and not any("encoder" in k for k in last["state_dict"])

    trainer = build_resume(save_path)
    # no save_freq_iter: single-process runs keep RandomSampler
    assert isinstance(trainer.train_loader.sampler, torch.utils.data.RandomSampler)
    model = trainer.model
    torch.nn.init.zeros_(model.encoder.patch_embed.proj.weight)  # not the pretrained
    checkpoint_loader(trainer).before_train()

    state = model.state_dict()
    assert_state_equal({k: state[k] for k in last["state_dict"]}, last["state_dict"])
    assert_state_equal(model.encoder.state_dict(), encoder)
    assert trainer.start_epoch == 1
    assert trainer.best_metric_value == last["best_metric_value"]
    optimizer = trainer.optimizer.state_dict()
    assert optimizer["param_groups"] == last["optimizer"]["param_groups"]
    assert optimizer["state"].keys() == last["optimizer"]["state"].keys()
    assert len(optimizer["state"]) > 0
    for i in optimizer["state"]:
        assert_state_equal(optimizer["state"][i], last["optimizer"]["state"][i])
    assert trainer.scheduler.state_dict() == last["scheduler"]

    # the resumed run finishes epoch 2; the frozen encoder stays the pretrained one
    trainer.train()
    last = load_checkpoint(save_path / "model" / "model_last.pth")
    assert last["epoch"] == 2
    assert not any("encoder" in k for k in last["state_dict"])
    assert_state_equal(model.encoder.state_dict(), encoder)


@pytest.mark.parametrize("tamper", ["missing", "unexpected"])
def test_resume_strict_decoder(save_path, tamper):
    last = load_checkpoint(save_path / "model" / "model_last.pth")
    if tamper == "missing":
        del last["state_dict"]["unpatchfy.0.proj.weight"]
    else:
        last["state_dict"]["dec_extra.weight"] = torch.zeros(1)
    tampered = save_path / "model" / "model_tampered.pth"
    torch.save(last, tampered)

    trainer = build_resume(save_path, tampered)
    with pytest.raises(RuntimeError, match="unpatchfy.0.proj.weight|dec_extra"):
        checkpoint_loader(trainer).before_train()


def test_resume_needs_fresh_run_config(save_path):
    trainer = build_resume(save_path)
    (save_path / "config.py").unlink()  # parsed already; the loader re-reads it
    with pytest.raises(FileNotFoundError, match="config.py"):
        checkpoint_loader(trainer).before_train()


# ---------------------------------------------------------------- mid-epoch resume

MID_EPOCH = """
_base_ = ["{base}"]

model = dict(
    encoder_train_mode=False,  # no random RoPE anywhere: runs are deterministic
    dec_pos_embed_rope_rescale_coords=None,
    # nor random fit pixels (the L-DPT depth loss draws them; a resume does not
    # restore the random generators)
    criterion=dict(_delete_=True, type="DeterministicDepthLossTest"),
)
data = dict(train=dict(type="ResumeTestDataset", length=12))  # 6 iters / epoch

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module.encoder."),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=1),
    dict(type="CheckpointSaver", skip_keywords=["encoder"], save_freq_iter=4),
    dict(type="RecordBatchesTest"),
] + {crash}
"""


def test_resumable_sampler():
    data = list(range(11))
    for rank in range(2):
        ref = torch.utils.data.distributed.DistributedSampler(
            data, num_replicas=2, rank=rank
        )
        sampler = ResumableSampler(data, num_replicas=2, rank=rank)
        for epoch in (0, 3):
            ref.set_epoch(epoch)
            sampler.set_epoch(epoch)
            sampler.start_index = 0
            assert list(sampler) == list(ref)  # same order as DistributedSampler
            sampler.start_index = 4
            assert list(sampler) == list(ref)[4:]
            assert len(sampler) == len(ref) == 6  # len stays the full epoch


def test_resume_mid_epoch(tmp_path):
    pretrained = tmp_path / "dinov3_tiny_pretrain.pth"
    torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), pretrained)
    base = tmp_path / "base.py"
    base.write_text(
        CONFIG.format(
            base=f"{REPO}/configs/_base_/default_runtime.py", pretrained=pretrained
        )
    )

    def run(name, crash):
        config = tmp_path / f"{name}.py"
        config.write_text(MID_EPOCH.format(base=base, crash=crash))
        trainer = build(config, save_path=str(tmp_path / name))
        assert isinstance(trainer.train_loader.sampler, ResumableSampler)
        return trainer

    reference = run("reference", crash="[]")
    reference.train()

    # saves at iterations 4 (epoch 1), 6 (end of epoch 1), 8 (epoch 2, iter 2);
    # crashes after iteration 10 (epoch 2, iter 4)
    crashed = run("exp", crash='[dict(type="CrashAtTest", epoch=1, iter=3)]')
    with pytest.raises(SystemExit):
        crashed.train()
    save_path = tmp_path / "exp"
    last = load_checkpoint(save_path / "model" / "model_last.pth")
    assert (last["epoch"], last["iter"]) == (1, 2)
    assert not any("encoder" in k for k in last["state_dict"])

    resumed = build_resume(save_path)
    resumed.train()
    assert (resumed.start_epoch, resumed.start_iter) == (1, 2)

    # data order: the resumed run trains exactly the batches the crash left out
    trained = find_hook(crashed, RecordBatches).batches
    rest = find_hook(resumed, RecordBatches).batches
    assert len(trained) == 10 and rest[0][:2] == (1, 2)
    assert trained[:8] + rest == find_hook(reference, RecordBatches).batches
    assert find_hook(resumed, InformationWriter).curr_iter == 12

    # training state: as if the run had never stopped
    assert_state_equal(resumed.model.state_dict(), reference.model.state_dict())
    ref_opt, opt = reference.optimizer.state_dict(), resumed.optimizer.state_dict()
    assert opt["param_groups"] == ref_opt["param_groups"]
    assert opt["state"].keys() == ref_opt["state"].keys()
    for i in opt["state"]:
        assert_state_equal(opt["state"][i], ref_opt["state"][i])
    assert resumed.scheduler.state_dict() == reference.scheduler.state_dict()
    last = load_checkpoint(save_path / "model" / "model_last.pth")
    assert last["epoch"] == 2 and "iter" not in last


def test_checkpoint_saver_refuses_non_finite():
    good = {"a": torch.ones(3), "steps": torch.tensor(5)}
    CheckpointSaver.check_finite(good)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(FloatingPointError, match="not finite"):
            CheckpointSaver.check_finite({**good, "b": torch.tensor([1.0, bad])})


def test_fine_tune_start_from_two_files_and_resume(fresh_run, tmp_path):
    """A second stage (e.g. configs/hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal.py):
    checkpoint = [DINOv3 file, a decoder-only weights file] with hook 0 keywords
    ["module.", "module."] -> ["module.encoder.", "module."] starts with every key
    loaded (encoder from the first, decoder from the second); its resume (train.sh:
    config.py of the fresh run, checkpoint=model_last.pth) reloads the skipped encoder
    keys from that list."""
    src_path, encoder = fresh_run
    last = load_checkpoint(src_path / "model" / "model_last.pth")
    start = tmp_path / "epoch_1.pth"  # as CheckpointSaver(keep_epoch_weights) writes
    torch.save(dict(epoch=1, state_dict=last["state_dict"]), start)
    pre = Config.fromfile(str(src_path / "config.py")).checkpoint
    save_path = tmp_path / "ft"
    trainer = build(
        src_path / "config.py",
        save_path=str(save_path),
        resume=False,
        checkpoint=[pre, str(start)],
        **{
            "hooks.0.keywords": ["module.", "module."],
            "hooks.0.replacement": ["module.encoder.", "module."],
        },
    )
    model = trainer.model
    torch.nn.init.zeros_(model.encoder.patch_embed.proj.weight)
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.zeros_(p)
    loader = checkpoint_loader(trainer)
    weight, _ = loader.merge_weights(
        trainer.cfg.checkpoint, loader.keywords, loader.replacement
    )
    info = model.load_state_dict(weight, strict=False)
    assert info.missing_keys == [] and info.unexpected_keys == []
    torch.nn.init.zeros_(model.encoder.patch_embed.proj.weight)
    loader.before_train()
    state = model.state_dict()
    assert_state_equal({k: state[k] for k in last["state_dict"]}, last["state_dict"])
    assert_state_equal(model.encoder.state_dict(), encoder)
    assert trainer.start_epoch == 0
    with pytest.raises(SystemExit):  # StopAfterEpochTest: crash after epoch 1
        trainer.train()
    # the requeue: config.py of the fine-tune keeps the two files
    assert Config.fromfile(str(save_path / "config.py")).checkpoint == [pre, str(start)]
    trainer = build_resume(save_path)
    torch.nn.init.zeros_(trainer.model.encoder.patch_embed.proj.weight)
    checkpoint_loader(trainer).before_train()
    assert trainer.start_epoch == 1
    ft_last = load_checkpoint(save_path / "model" / "model_last.pth")
    state = trainer.model.state_dict()
    assert_state_equal(
        {k: state[k] for k in ft_last["state_dict"]}, ft_last["state_dict"]
    )
    assert_state_equal(trainer.model.encoder.state_dict(), encoder)
