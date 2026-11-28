"""CriterionWeightWarmup (ldpt/hooks/misc.py): gated warm-up of a criterion weight
(aux_scale of the L-DPT depth loss, the default key; any other key works the same).

A fake trainer (CPU, one process) feeds the hook depth_loss / drop_frac sequences.
"""

import logging
import types

import pytest
import torch

from ldpt.hooks.misc import CriterionWeightWarmup
from ldpt.models.losses.depth import RelativeDepthLoss


class FakeTrainer:
    def __init__(
        self, resume=False, start_epoch=0, start_iter=0, iters_per_epoch=100, state=None
    ):
        criterion = RelativeDepthLoss(aux_scale=4.0)
        self.model = types.SimpleNamespace(
            module=types.SimpleNamespace(criterion=criterion)
        )
        self.cfg = types.SimpleNamespace(resume=resume)
        self.start_epoch, self.start_iter = start_epoch, start_iter
        self.train_loader = range(iters_per_epoch)
        self.logger = logging.getLogger("test_criterion_warmup")
        self.writer = None
        self.comm_info = {}
        if state is not None:  # what CheckpointLoader.resume hands over
            self.comm_info["resume_hook_state"] = {"CriterionWeightWarmup": state}

    @property
    def weight(self):
        return self.model.module.criterion.aux_scale

    def step(self, hook, depth_loss, drop_frac):
        self.comm_info["model_output_dict"] = dict(
            loss=torch.tensor(9.0),
            depth_loss=torch.tensor(depth_loss),
            drop_frac=torch.tensor(drop_frac),
        )
        hook.after_step()


def make(**kwargs):
    trainer = FakeTrainer(**kwargs)
    hook = CriterionWeightWarmup(
        min_hold=10, ramp=20, max_hold=30, ema=0.5, log_interval=5
    )
    hook.trainer = trainer
    hook.before_train()
    return trainer, hook


def test_held_at_zero_while_gate_closed(caplog):
    trainer, hook = make()
    assert trainer.weight == 0.0 and hook.final == 4.0
    with caplog.at_level(logging.WARNING, logger="test_criterion_warmup"):
        for _ in range(40):  # trapped: flat-prediction loss, every image dropped
            trainer.step(hook, 1.0, 1.0)
    assert trainer.weight == 0.0 and hook.progress == 0.0 and not hook.opened
    assert "gate never opened" in caplog.text  # past max_hold


def test_min_hold_then_linear_ramp():
    trainer, hook = make()
    weights = []
    for _ in range(40):  # aligned from the start: the gate opens at min_hold
        trainer.step(hook, 0.1, 0.0)
        weights.append(trainer.weight)
    assert weights[:9] == [0.0] * 9  # iterations 1-9 < min_hold
    assert weights[9] == pytest.approx(4.0 / 20)  # iteration 10: first ramp step
    assert weights[28] == pytest.approx(4.0)  # 20 gated iterations later
    assert weights[-1] == 4.0
    assert all(b >= a for a, b in zip(weights, weights[1:]))


def test_closed_gate_pauses_never_decreases():
    trainer, hook = make()
    for _ in range(15):
        trainer.step(hook, 0.1, 0.0)
    paused = trainer.weight
    assert 0.0 < paused < 4.0
    for _ in range(10):  # depth_loss EMA back above the gate
        trainer.step(hook, 2.0, 0.0)
    assert trainer.weight == paused


def test_gate_uses_both_averages():
    trainer, hook = make()
    for _ in range(30):  # low loss but most images dropped: stay at 0
        trainer.step(hook, 0.1, 0.5)
    assert trainer.weight == 0.0


def test_resume_restores_the_checkpoint_state():
    trainer, hook = make()
    for _ in range(20):  # 11 ramp steps (iterations 10-20)
        trainer.step(hook, 0.1, 0.0)
    state = hook.state_dict()
    assert state["progress"] == pytest.approx(11 / 20) and state["opened"]
    resumed, hook2 = make(resume=True, start_iter=20, state=state)
    assert hook2.curr_iter == 20 and hook2.averages == hook.averages
    assert resumed.weight == pytest.approx(4.0 * 11 / 20)
    for _ in range(9):
        resumed.step(hook2, 0.1, 0.0)
    assert resumed.weight == pytest.approx(4.0)
    resumed.step(hook2, 5.0, 1.0)  # finished: the hook no longer reads the outputs
    assert resumed.weight == 4.0


def test_fresh_run_ignores_any_state():
    """A fresh run (resume=False) starts at 0, whatever was handed over or saved."""
    trainer, hook = make(state=dict(progress=1.0, opened=True, averages=[0.1, 0.0]))
    assert trainer.weight == 0.0 and hook.progress == 0.0 and hook.averages is None


def test_resume_without_state_holds_again():
    """A checkpoint without the hook's state (e.g. from a run without the hook)."""
    trainer, hook = make(resume=True, start_iter=5000)
    assert trainer.weight == 0.0
    trainer.step(hook, 0.1, 0.0)  # trained model: gate open at once, ramp from 0
    assert trainer.weight == pytest.approx(4.0 / 20)


# ------------------------------------------------- the defaults (the depth recipes)


def test_defaults_are_the_depth_recipes(caplog):
    """The defaults: key aux_scale, gated on depth_loss <= 0.20 and drop_frac <= 0.20
    (another output such as clamp_frac is ignored), the log lines name drop_frac,
    the state has the same keys; a missing drop_frac raises."""
    hook = CriterionWeightWarmup()
    assert (hook.key, hook.gate_key) == ("aux_scale", "drop_frac")
    assert (hook.gate_depth_loss, hook.gate_clamp_frac) == (0.20, 0.20)
    assert (hook.min_hold, hook.ramp, hook.max_hold) == (1000, 1000, 3000)
    trainer, hook = make()
    with caplog.at_level(logging.INFO, logger="test_criterion_warmup"):
        for _ in range(10):
            trainer.comm_info["model_output_dict"] = dict(
                loss=torch.tensor(9.0),
                depth_loss=torch.tensor(0.1),
                drop_frac=torch.tensor(0.0),
                clamp_frac=torch.tensor(1.0),
            )
            hook.after_step()
    assert trainer.weight == pytest.approx(4.0 / 20)
    assert "drop_frac_ema 0.0000" in caplog.text
    assert set(hook.state_dict()) == {"progress", "opened", "averages"}
    trainer.comm_info["model_output_dict"] = dict(depth_loss=torch.tensor(0.1))
    with pytest.raises(KeyError, match="drop_frac"):
        hook.after_step()


class StubCriterion(torch.nn.Module):
    """What the hook needs of the L-DPT depth loss: an aux_scale attribute (the
    weight of the local and gradient terms); its loss dict carries depth_loss and
    drop_frac."""

    def __init__(self, aux_scale=1.0):
        super().__init__()
        self.aux_scale = aux_scale

    def forward(self, depth_loss, drop_frac):
        local_loss, mage_loss = torch.tensor(0.5), torch.tensor(0.1)
        return dict(
            loss=depth_loss + self.aux_scale * (local_loss + 4.0 * mage_loss),
            depth_loss=depth_loss,
            local_loss=local_loss,
            mage_loss=mage_loss,
            drop_frac=drop_frac,
        )


class StubTrainer(FakeTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.module.criterion = StubCriterion()

    @property
    def weight(self):
        return self.model.module.criterion.aux_scale

    def step(self, hook, depth_loss, drop_frac):
        criterion = self.model.module.criterion
        out = criterion(torch.tensor(depth_loss), torch.tensor(drop_frac))
        self.comm_info["model_output_dict"] = out
        hook.after_step()
        return out


def make_stub(**kwargs):
    trainer = StubTrainer(**kwargs)
    hook = CriterionWeightWarmup(
        key="aux_scale",
        gate_key="drop_frac",
        min_hold=10,
        ramp=20,
        gate_depth_loss=0.20,
        gate_clamp_frac=0.20,
        max_hold=30,
        ema=0.5,
        log_interval=5,
    )
    hook.trainer = trainer
    hook.before_train()
    return trainer, hook


def test_hold_ramp_and_gate(caplog):
    """key="aux_scale", gate_key="drop_frac": held at 0 until min_hold and both EMAs
    under their bounds (depth_loss <= 0.20, drop_frac <= gate_clamp_frac), then
    ramped linearly to the criterion's configured 1.0; a closed gate pauses it."""
    trainer, hook = make_stub()
    assert hook.final == 1.0 and trainer.weight == 0.0
    out = trainer.step(hook, 0.1, 0.0)
    assert out["loss"] == out["depth_loss"]  # aux_scale 0: the data term alone
    for _ in range(14):  # every image dropped: closed, past min_hold
        trainer.step(hook, 0.1, 1.0)
    assert trainer.weight == 0.0 and not hook.opened
    for _ in range(14):  # loss above the bound: closed
        trainer.step(hook, 0.5, 0.0)
    assert trainer.weight == 0.0 and not hook.opened
    weights = []
    with caplog.at_level(logging.INFO, logger="test_criterion_warmup"):
        for _ in range(25):
            trainer.step(hook, 0.1, 0.0)
            weights.append(trainer.weight)
    assert hook.opened and 0.0 in weights[:3] and weights[-1] == 1.0
    assert "drop_frac_ema" in caplog.text and "clamp_frac" not in caplog.text
    first = next(i for i, w in enumerate(weights) if w > 0)
    assert weights[first] == pytest.approx(1 / 20)
    assert weights[first + 19] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(weights, weights[1:]))


def test_pause_and_resume():
    """A drop_frac EMA back over its bound pauses the ramp; the state saved mid-ramp
    restores aux_scale on resume and the ramp continues to 1."""
    trainer, hook = make_stub()
    for _ in range(15):  # opens at iteration 10: 6 ramp steps
        trainer.step(hook, 0.1, 0.0)
    assert trainer.weight == pytest.approx(6 / 20)
    for _ in range(5):
        trainer.step(hook, 0.1, 1.0)
    assert trainer.weight == pytest.approx(6 / 20)
    state = hook.state_dict()
    assert set(state) == {"progress", "opened", "averages"} and state["opened"]
    resumed, hook2 = make_stub(resume=True, start_iter=20, state=state)
    assert hook2.curr_iter == 20 and hook2.averages == hook.averages
    assert resumed.weight == pytest.approx(6 / 20)
    for _ in range(20):
        resumed.step(hook2, 0.1, 0.0)
    assert resumed.weight == pytest.approx(1.0)
    resumed.step(hook2, 5.0, 1.0)  # finished: the outputs are no longer read
    assert resumed.weight == 1.0


def test_missing_gate_key_raises():
    """A criterion without the gate key in its outputs: KeyError naming it."""
    trainer, hook = make_stub()
    trainer.comm_info["model_output_dict"] = dict(
        depth_loss=torch.tensor(0.1), clamp_frac=torch.tensor(0.0)
    )
    with pytest.raises(KeyError, match="drop_frac"):
        hook.after_step()


TRAIN_CONFIG = """
_base_ = ["{base}"]

checkpoint = "{pretrained}"
enable_wandb = False
evaluate = False
seed = 0
epoch = 3
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
    criterion=dict(
        type="relative_depth_loss", min_valid=100, roe_points=64, aux_scale=4.0
    ),
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
    dict(type="InformationWriter", log_interval=1),
    dict(type="CheckpointSaver", save_freq=None, skip_keywords=["encoder"], save_freq_iter=1),
    dict(
        type="CriterionWeightWarmup", min_hold=2, ramp=4, gate_depth_loss=100.0,
        gate_clamp_frac=1.0, log_interval=1,
    ),
    dict(type="CrashAtTest", epoch=0, iter=1),
]
"""


def test_in_trainer_with_resume(tmp_path):
    """The hook in the real trainer (tiny model, 3 epochs x 2 iterations): crash after
    iteration 2, resume from the iteration-1 checkpoint (train.sh -r auto)."""
    from helpers import REPO, DINOv3TinyTest, init_encoder_
    from test_resume import build, build_resume  # registers the test dataset / hooks

    from ldpt.hooks.misc import load_checkpoint

    pretrained = tmp_path / "dinov3_tiny_pretrain.pth"
    torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), pretrained)
    config = tmp_path / "config_src.py"
    config.write_text(
        TRAIN_CONFIG.format(
            base=f"{REPO}/configs/_base_/default_runtime.py", pretrained=pretrained
        )
    )
    save_path = tmp_path / "exp"

    def run(trainer):
        hook = next(h for h in trainer.hooks if isinstance(h, CriterionWeightWarmup))
        seen, original = [], hook.after_step

        def record():
            original()
            seen.append(trainer.model.criterion.aux_scale)

        hook.after_step = record
        try:
            trainer.train()
        except SystemExit:  # the simulated crash
            pass
        return seen

    # iteration 1 < min_hold; the gate opens at 2; crash after 2 (no epoch save)
    assert run(build(config, save_path=str(save_path))) == [0.0, 1.0]
    last = load_checkpoint(save_path / "model" / "model_last.pth")
    assert last["iter"] == 1  # saved after iteration 1, before the hook's update
    assert last["hook_state"]["CriterionWeightWarmup"] == dict(
        progress=0.0, opened=False, averages=None
    )
    # resumed at iteration 2: the ramp continues as in an uninterrupted run
    assert run(build_resume(save_path)) == [1.0, 2.0, 3.0, 4.0, 4.0]
    last = load_checkpoint(save_path / "model" / "model_last.pth")
    assert (
        last["epoch"] == 3
        and last["hook_state"]["CriterionWeightWarmup"]["progress"] == 1.0
    )


def test_real_criterion():
    """The hook with the real L-DPT depth loss (relative_depth_loss): it
    sets aux_scale to 0 (loss == depth_loss, local / gradient terms still logged),
    stays closed while the anti-correlated prediction drops every image (drop_frac
    1), opens on a good prediction and ramps aux_scale to the configured 1.0, where
    the loss is depth_loss + local_loss + 4 mage_loss again."""
    from test_depth_loss_roe import ROE, batch

    criterion = RelativeDepthLoss(**ROE)
    trainer = FakeTrainer()
    trainer.model.module.criterion = criterion
    hook = CriterionWeightWarmup(
        key="aux_scale",
        gate_key="drop_frac",
        min_hold=3,
        ramp=4,
        gate_depth_loss=0.20,
        gate_clamp_frac=0.20,
        max_hold=30,
        ema=0.5,
        log_interval=5,
    )
    hook.trainer = trainer
    hook.before_train()
    assert hook.final == 1.0 and criterion.aux_scale == 0.0
    pred, target, mask, names = batch("ABC", hw=64)

    def step(p):
        torch.manual_seed(0)
        out = criterion(p, target, mask, names=names)
        trainer.comm_info["model_output_dict"] = out
        hook.after_step()
        return out

    out = step(pred)
    assert out["drop_frac"] == 0.0 and out["depth_loss"] < 0.20
    assert out["loss"] == out["depth_loss"] and out["local_loss"] > 0
    assert out["mage_loss"] > 0
    for _ in range(6):  # anti-correlated: every fit dropped, the gate stays closed
        out = step(1.0 / pred)
        assert out["drop_frac"] == 1.0
    assert criterion.aux_scale == 0.0 and not hook.opened
    weights = []
    for _ in range(8):
        out = step(pred)
        weights.append(criterion.aux_scale)
    assert hook.opened and weights[-1] == 1.0 and weights[0] < 1.0
    assert all(b >= a for a, b in zip(weights, weights[1:]))
    expected = out["depth_loss"] + out["local_loss"] + 4.0 * out["mage_loss"]
    assert out["loss"].item() == pytest.approx(expected.item(), rel=1e-6)
