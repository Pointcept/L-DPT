"""
Model and trainer options of the release recipes that are off by default (CPU; CUDA
cases where a GPU is visible), and the shared tiny-run helpers of the option tests:

- model.attn_fp32 / model.block_fp32: fp32 islands in the bf16 decoder (SDPA inputs
  fp32, no autocast in the block, the same math without autocast, compiled blocks);
  the release models run attn_fp32="last_stage";
- model.dec_qk_bound: an idle bound leaves the eval prediction bitwise unchanged (the
  release models use 1000; the other dec_qk_bound tests are in
  test_structural_options.py);
- relative_depth_loss(log_logit_stats=True): detached statistics of the head's logit;
- helpers: train.py's option parsing (parse), a plain L config (plain_l_config), tiny
  training runs with the L-DPT depth loss (loss_run), resume, a step recorder hook.
"""

import copy
import logging
from functools import partial

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from helpers import REPO, DINOv3TinyTest, init_encoder_
from test_evaluate import CONFIG as EVAL_CONFIG
from test_l_dpt_golden import BASE, DEPTH_LOSS, make_data, randomize_
from test_resume import build

from ldpt.engines.default import default_argument_parser, default_config_parser
from ldpt.hooks.builder import HOOKS
from ldpt.hooks.default import HookBase
from ldpt.models.builder import build_model
from ldpt.models.dinov3.layers import RopePositionEmbedding
from ldpt.models.l_dpt.modules import Attention, Block
from ldpt.models.losses.depth import RelativeDepthLoss

RELEASE_ZS = f"{REPO}/configs/hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py"
LOGIT_KEYS = {"logit_mean", "logit_std", "logit_absmax"}


def plain_l_config(tmp_path):
    """The zero-shot release recipe with the plain L decoder (sigmoid head, the default
    loss, no fp32 island, no logit bound; 270,995,712 decoder parameters): a full
    config for the option-string tests."""
    path = tmp_path / "plain_l.py"
    path.write_text(
        f"_base_ = [{RELEASE_ZS!r}]\n"
        "model = dict(head='sigmoid', readout_fp32=False, attn_fp32=None, "
        "dec_qk_bound=None, criterion=dict(_delete_=True, type='relative_depth_loss'))\n"
    )
    return path


@HOOKS.register_module("RecordStepsTest", force=True)
class RecordSteps(HookBase):
    """Per step: the optimizer's (betas, eps) per group before the step, the loss;
    the weights before the first step (after the checkpoint is loaded)."""

    def __init__(self):
        self.hyper = []
        self.losses = []
        self.start = {}

    def before_train(self):
        self.start = {
            n: p.detach().clone() for n, p in self.trainer.model.named_parameters()
        }

    def before_step(self):
        groups = self.trainer.optimizer.param_groups
        self.hyper.append([(tuple(g["betas"]), g["eps"]) for g in groups])

    def after_step(self):
        loss = self.trainer.comm_info["model_output_dict"]["loss"]
        self.losses.append(loss.detach().clone())


def add_recorder(trainer):
    trainer.register_hooks([dict(type="RecordStepsTest")])
    return trainer.hooks[-1]


def capture_log(trainer):
    """Messages of the trainer's logger (the root logger keeps the first run's file)."""
    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    trainer.logger.addHandler(handler)
    return messages, handler


def train(trainer, crash_ok=False):
    messages, handler = capture_log(trainer)
    try:
        trainer.train()
    except SystemExit:  # ExceptionWriter: a (simulated) crash
        if not crash_ok:
            raise
    finally:
        trainer.logger.removeHandler(handler)
    return messages


def parse(config_file, options):
    """train.py's parsing (train.sh -o "a=1 b=2" passes them split on spaces)."""
    args = default_argument_parser().parse_args(
        ["--config-file", str(config_file), "--options"] + list(options)
    )
    return default_config_parser(args.config_file, args.options)


def resume(save_path, checkpoint="model_last.pth", **options):
    return build(
        save_path / "config.py",
        save_path=str(save_path),
        resume=True,
        checkpoint=str(save_path / "model" / checkpoint),
        **options,
    )


# test_evaluate's tiny config with the L-DPT depth loss (label types A and B, the
# training samples named after them), gradient clipping (logs grad_norm) and
# InformationWriter (hook 2)
LOSS_RUN_CHANGES = (
    (
        'criterion=dict(type="relative_depth_loss", roe_points=64),',
        """criterion=dict(
        type="relative_depth_loss",
        min_valid=100,
        roe_points=256,
        label_types=dict(A=["hypersim"], B=["blendedmvs"]),
        type_trim=dict(A=0.0, B=0.05),
        local_levels=dict(A=[4, 16], B=[4]),
        local_cell=128.0,
        grad_types=["A"],
        mage_weight=4.0,
    ),""",
    ),
    (
        'train=dict(type="EvalTestDataset", length=4),',
        'train=dict(type="EvalTestDataset", length=4, '
        'groups=("hypersim", "blendedmvs")),',
    ),
    ("num_workers = 0\n", "num_workers = 0\nclip_grad = 1.0\n"),
    (
        '    dict(type="DepthEvaluator", relative=True, tta=[["flip"]]),',
        """    dict(type="InformationWriter", log_interval=1),
    dict(type="DepthEvaluator", relative=True, tta=[["flip"]]),""",
    ),
)


def loss_run_config(tmp_path):
    """test_evaluate's tiny config (bf16 decoder) with LOSS_RUN_CHANGES."""
    pretrained = tmp_path / "dinov3_tiny_pretrain.pth"
    if not pretrained.exists():
        torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), pretrained)
    text = EVAL_CONFIG.format(
        base=f"{REPO}/configs/_base_/default_runtime.py", pretrained=pretrained
    )
    for old, new in LOSS_RUN_CHANGES:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    config = tmp_path / "config_loss_run.py"
    config.write_text(text)
    return config


def loss_run(tmp_path, name, **options):
    """2 epochs x 4 iterations, InformationWriter every 2nd iteration."""
    options = {
        "evaluate": False,
        "hooks.2.log_interval": 2,
        "data.train.length": 8,
        "save_path": str(tmp_path / name),
        **options,
    }
    trainer = build(loss_run_config(tmp_path), **options)
    rec = add_recorder(trainer)
    messages = train(trainer)
    return trainer, rec, messages


# ---------------------------------------------------------------- fp32 islands


def island_model(precision="bf16", **options):
    """The golden test's tiny L-DPT (stages of 1, 1, 2, 2 blocks), random weights."""
    torch.manual_seed(0)
    config = dict(BASE, criterion=DEPTH_LOSS, precision=precision)
    config.update(dec_num_layers=(1, 1, 2, 2), **options)
    return randomize_(build_model(config))


def float_dtypes(tree):
    return {
        t.dtype
        for t in tree_leaves(tree)
        if isinstance(t, torch.Tensor) and t.is_floating_point()
    }


class OpTrace(TorchDispatchMode):
    """Per decoder block "s.i": the ATen ops it runs (as dispatched below autocast
    and autograd) with their floating input / output dtypes, and whether autocast
    was on when the block was called (forward hooks mark the running block)."""

    def __init__(self, model):
        super().__init__()
        self.block, self.ops, self.autocast, self.handles = None, [], {}, []
        for name, blk in model.dec_blocks.named_modules():
            if isinstance(blk, Block):
                self.handles += [
                    blk.register_forward_pre_hook(partial(self.enter, name)),
                    blk.register_forward_hook(self.leave),
                ]

    def enter(self, name, module, args):
        self.block = name
        self.autocast[name] = torch.is_autocast_enabled("cpu")

    def leave(self, module, args, output):
        self.block = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if self.block is not None:
            inputs, outputs = float_dtypes((args, kwargs)), float_dtypes(out)
            self.ops.append((self.block, str(func), inputs, outputs))
        return out


@pytest.mark.parametrize(
    "options, attn32, block32",
    [
        (dict(attn_fp32="last"), {"3.1"}, set()),
        (dict(attn_fp32="last_stage"), {"3.0", "3.1"}, set()),
        (dict(attn_fp32="all"), {"0.0", "1.0", "2.0", "2.1", "3.0", "3.1"}, set()),
        (dict(block_fp32="last"), set(), {"3.1"}),
        (dict(block_fp32="last_stage", attn_dtype="bf16"), set(), {"3.0", "3.1"}),
    ],
)
def test_fp32_islands_dispatch(options, attn32, block32):
    """bf16 decoder: the SDPA kernel of an attn_fp32 block gets fp32 q / k / v (the
    others bf16); a block_fp32 block runs without autocast and no op in it sees a
    bf16 tensor; the training step's backward reaches every decoder parameter."""
    model = island_model(**options).train()
    data = make_data("depth", (64, 80))
    trace = OpTrace(model)
    torch.manual_seed(1)
    with trace:
        loss = model(data)["loss"]
    for h in trace.handles:
        h.remove()
    loss.backward()
    blocks = sorted(trace.autocast)
    assert len(blocks) == 6
    for blk in blocks:
        sdpa = [o for o in trace.ops if o[0] == blk and "scaled_dot_product" in o[1]]
        assert len(sdpa) == 1, blk
        want = torch.float32 if blk in attn32 | block32 else torch.bfloat16
        assert sdpa[0][2] == {want}, blk
        assert trace.autocast[blk] == (blk not in block32), blk
        if blk in block32:
            ops = [o for o in trace.ops if o[0] == blk]
            assert all(torch.bfloat16 not in o[2] | o[3] for o in ops), blk
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_fp32_islands_same_math_without_autocast():
    """precision fp32 (no autocast): attn_fp32="all" and block_fp32="last_stage" give
    bitwise the outputs, loss and gradients of the model without them."""
    data = make_data("depth", (64, 80))
    results = []
    for options in ({}, dict(attn_fp32="all", block_fp32="last_stage")):
        model = island_model("fp32", **options)
        model.eval()
        with torch.no_grad():
            pred = model(copy.deepcopy(data))["pred_depth"]
        model.train()
        torch.manual_seed(1)
        loss = model(copy.deepcopy(data))["loss"]
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        results.append((pred, loss, grads))
    (p0, l0, g0), (p1, l1, g1) = results
    assert torch.equal(p0, p1) and torch.equal(l0, l1)
    assert all(torch.equal(a, b) for a, b in zip(g0, g1))


def attention_grads(attn, x, rope, autocast, forward=None):
    """Output and the gradients of x and of the weights after (output^2).sum()."""
    x = x.clone().requires_grad_(True)
    attn.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        y = (forward or attn)(x, rope=rope)
    y.float().square().sum().backward()
    return [y, x.grad] + [p.grad for p in attn.parameters()]


def test_attn_fp32_attention():
    """Attention(force_fp32) under bf16 autocast: forward and backward bitwise those
    without autocast, also compiled (aot_eager: AOT autograd would trace the backward
    of an inlined island in bf16) and with attn_dtype bf16."""
    torch.manual_seed(0)
    rope = RopePositionEmbedding(embed_dim=16, num_heads=2, base=100)(H=4, W=6)
    x = torch.randn(2, 24, 32)
    torch._dynamo.reset()
    for attn_dtype in ("fp32", "bf16"):
        attn = Attention(32, num_heads=2, downscale=2, attn_dtype=attn_dtype)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert attn(x, rope=rope).dtype == torch.bfloat16
        attn.force_fp32 = True
        want = attention_grads(attn, x, rope, autocast=False)
        assert want[0].dtype == torch.float32
        compiled = torch.compile(attn.forward, backend="aot_eager")
        for forward in (None, compiled):
            got = attention_grads(attn, x, rope, autocast=True, forward=forward)
            assert all(torch.equal(a, b) for a, b in zip(want, got)), forward


def test_fp32_islands_options():
    """The scopes on stages of 1, 1, 2, 2 blocks; invalid values raise."""
    for scope, want in (
        ("last", {(3, 1)}),
        ("last_stage", {(3, 0), (3, 1)}),
        ("all", {(0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)}),
    ):
        model = island_model(attn_fp32=scope)
        got = {
            (s, i)
            for s, stage in enumerate(model.dec_blocks)
            for i, blk in enumerate(stage)
            if blk.attn.force_fp32
        }
        assert got == want and model.fp32_blocks == set()
        if scope != "all":  # block_fp32 blocks: their attention is an island too
            model = island_model(block_fp32=scope)
            assert model.fp32_blocks == want
            assert all(
                blk.attn.force_fp32 == ((s, i) in want)
                for s, stage in enumerate(model.dec_blocks)
                for i, blk in enumerate(stage)
            )
    for bad in (dict(attn_fp32="None"), dict(block_fp32="all"), dict(attn_fp32=True)):
        with pytest.raises(ValueError):
            island_model(**bad)


def test_fp32_islands_compiled_blocks():
    """compile_blocks (aot_eager: AOT autograd, as inductor) with attn_fp32 /
    block_fp32 islands in the bf16 decoder: the training loss and every gradient
    bitwise those of the eager model; the only graph break is the island attention,
    which runs eagerly (Attention.fp32_forward)."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    options = dict(attn_fp32="last_stage", block_fp32="last")
    eager = island_model(**options).train()
    compiled = copy.deepcopy(eager)
    compiled.compile_blocks(backend="aot_eager")
    data = make_data("depth", (64, 80))
    out = []
    for m in (eager, compiled):
        torch.manual_seed(1)
        loss = m(copy.deepcopy(data))["loss"]
        loss.backward()
        out.append((loss, [p.grad for p in m.parameters() if p.requires_grad]))
    breaks = torch._dynamo.utils.counters["graph_break"]
    assert breaks and all("fp32_forward" in reason for reason in breaks)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] >= 2
    assert torch.equal(out[1][0], out[0][0])
    assert all(torch.equal(a, b) for a, b in zip(out[0][1], out[1][1]))


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_qk_bound_eval_bit_identical_device(device):
    """dec_qk_bound in eval (the other dec_qk_bound tests are in
    test_structural_options.py; on CUDA when a GPU is visible). One fixed batch in eval
    mode, fp32 and bf16 decoder: a fresh model with
    dec_qk_bound 200 or 3e5 (head dim 8: r = 921, whose fp32 reciprocal times r rounds
    below 1) gives the prediction of the model without it, largest absolute difference
    0; an active bound (0.5) changes it."""
    image = make_data(None, (64, 80))["image"].to(device)
    for precision in ("fp32", "bf16"):
        preds = []
        for bound in (None, 200.0, 3e5, 0.5):
            model = island_model(precision, dec_qk_bound=bound).to(device).eval()
            with torch.no_grad():
                preds.append(model.infer(image))
        off, *idle, active = preds
        for pred in idle:
            assert (pred - off).abs().max().item() == 0.0, precision
        assert (active - off).abs().max().item() > 0.0, precision


# ---------------------------------------------------------------- loss: logit stats


def test_logit_stats_toy():
    """Medians over the images of the per-image logit mean / std, max |logit|; only
    valid pixels (the invalid ones hold p = 1e-9, |logit| ~ 13.8), p clamped to
    [1e-6, 1 - 1e-6] (a valid p = 1.0 reads as logit 13.80 in fp32)."""
    g = torch.Generator().manual_seed(0)
    hw = (40, 48)
    shift = torch.tensor([-2.0, 0.5, 3.0]).view(3, 1, 1, 1)
    scale = torch.tensor([0.5, 2.0, 1.0]).view(3, 1, 1, 1)
    z = shift + scale * torch.randn(3, 1, *hw, generator=g)
    depth = 1.0 + 4.0 * z.sigmoid()  # an affine map of the prediction: every fit holds
    depth[:, :, :5] = 0.0  # invalid rows
    mask = depth > 0
    pred = torch.where(mask, z.sigmoid(), torch.full_like(z, 1e-9))
    crit = RelativeDepthLoss(roe_points=64, log_logit_stats=True)

    def reference(p):
        pc = p.clamp(1e-6, 1 - 1e-6).double()
        zz = (pc / (1 - pc)).log()
        means, stds = [], []
        for zb, mb in zip(zz.flatten(1), mask.flatten(1)):
            means.append(zb[mb].mean())
            stds.append(zb[mb].std(unbiased=False))
        absmax = zz[mask].abs().max()
        return torch.stack(means).median(), torch.stack(stds).median(), absmax

    out = crit(pred, depth, mask)
    assert out["drop_frac"] == 0.0
    mean, std, absmax = reference(pred)
    assert out["logit_mean"].item() == pytest.approx(mean.item(), abs=1e-4)
    assert out["logit_std"].item() == pytest.approx(std.item(), rel=1e-4)
    assert out["logit_absmax"].item() == pytest.approx(absmax.item(), rel=1e-4)
    # the known logits: per-image means -2, 0.5, 3 and stds 0.5, 2, 1
    assert out["logit_mean"].item() == pytest.approx(0.5, abs=0.1)
    assert out["logit_std"].item() == pytest.approx(1.0, abs=0.1)
    assert out["logit_absmax"].item() < 13  # the invalid pixels are not read

    saturated = pred.clone()
    saturated[2, 0, 10, 10] = 1.0  # sigmoid of z > 17 in fp32
    out = crit(saturated, depth, mask)
    assert out["logit_absmax"].item() == pytest.approx(13.80, abs=0.02)
    assert not any(out[k].requires_grad for k in LOGIT_KEYS)
