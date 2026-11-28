"""
Segmentation model / optimizer options (CPU):
- the LR keywords of every segmentation recipe: "encoder" puts the frozen encoder alone
  at lr / 10, so the decoder attention (dec_blocks.*.attn.*) trains at the full lr;
- model tap_bn (off by default; the COCO-Stuff recipes 1 / 1a): a learned BatchNorm per
  tapped encoder layer before the necks (off: no module, bitwise the model without it;
  the golden tests cover the default path);
- param_dicts keyword lists; evaluate.py's weight average with BatchNorm counters.
"""

import copy
import glob
import os

import pytest
import torch
import torch.nn as nn

from helpers import REPO, init_encoder_
from test_l_dpt_golden import SEG_LOSS, make_data
from test_resume import assert_state_equal

from ldpt.engines.evaluate import average_states
from ldpt.models.builder import build_model
from ldpt.utils.config import Config
from ldpt.utils.optimizer import keyword_groups

CONFIGS = os.path.join(REPO, "configs")
RELEASE_SEG = sorted(  # every segmentation recipe
    os.path.relpath(p, CONFIGS)
    for folder in ("cocostuff", "voc2012")
    for p in glob.glob(os.path.join(CONFIGS, folder, "*.py"))
)
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
C_ENC = 64  # DINOv3TinyTest width
HW = (32, 48)
BN_KEYS = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")


def config(rel):
    return Config.fromfile(os.path.join(CONFIGS, rel))


def tiny(seed=0, **options):
    torch.manual_seed(seed)
    model = build_model(dict(TINY, **options))
    init_encoder_(model.encoder, seed)
    return model


@torch.no_grad()
def randomize_bn_(model, seed=0):
    """tap_bn statistics and affine away from their init."""
    g = torch.Generator().manual_seed(seed)
    for bn in model.tap_bn:
        bn.running_mean.copy_(0.3 * torch.randn(bn.num_features, generator=g))
        bn.running_var.copy_(0.5 + torch.rand(bn.num_features, generator=g))
        bn.weight.copy_(1.0 + 0.2 * torch.randn(bn.num_features, generator=g))
        bn.bias.copy_(0.1 * torch.randn(bn.num_features, generator=g))
    return model


def group_of(model, rel):
    cfg = config(rel)
    groups = keyword_groups(model, cfg.param_dicts, cfg.optimizer.lr)
    return {n: i for i, g in enumerate(groups) for n in g["names"]}


# ---------------------------------------------------------------- LR keyword


def test_release_seg_lr_keywords_leave_decoder_attention_at_full_lr():
    """The param_dicts of every segmentation recipe: the keyword "encoder" puts the
    frozen encoder alone into the lr / 10 group, the no-weight-decay group takes norms,
    tap BatchNorms, biases and LayerScale, so the decoder's q / k / v / out_proj weights
    and QK-norm temperatures stay in the default group at the full lr."""
    model = tiny(dec_qk_norm=True, tap_bn=True, out_dim=64)
    attn = [n for n, _ in model.named_parameters() if n.startswith("dec_blocks.")]
    attn = [n for n in attn if ".attn." in n]
    assert len(attn) == 4 * 6  # per block: logit_scale, q / k / v, out_proj (w, b)
    assert len(RELEASE_SEG) == 3
    for rel in RELEASE_SEG:
        cfg = config(rel)
        assert [dict(r) for r in cfg.param_dicts] == [
            dict(keyword="encoder", lr=cfg.base_lr / 10),
            dict(keyword=["norm", "tap_bn", "bias", "gamma"], weight_decay=0.0),
        ], rel
        where = group_of(model, rel)
        assert {where[n] for n in attn if not n.endswith(".bias")} == {0}, rel
        assert {where[n] for n in attn if n.endswith(".bias")} == {2}, rel
        assert {where[n] for n in where if n.startswith("encoder.")} == {1}, rel
        assert list(cfg.scheduler.max_lr) == [
            cfg.base_lr,
            cfg.base_lr / 10,
            cfg.base_lr,
        ], rel


# ---------------------------------------------------------------- tap_bn


def test_tap_bn_off_is_the_old_model():
    off, default = tiny(tap_bn=False), tiny()
    assert off.tap_bn is None
    assert not any("tap_bn" in k for k in off.state_dict())
    assert_state_equal(off.state_dict(), default.state_dict())
    data = make_data(None, HW)
    off.eval(), default.eval()
    with torch.no_grad():
        assert torch.equal(off(data)["pred_segment"], default(data)["pred_segment"])


def test_tap_bn_keys_init_and_rng():
    torch.manual_seed(0)
    on = build_model(dict(TINY, tap_bn=True))
    rng_on = torch.get_rng_state()
    torch.manual_seed(0)
    off = build_model(TINY)
    assert torch.equal(rng_on, torch.get_rng_state())  # BatchNorm draws nothing
    s_on, s_off = on.state_dict(), off.state_dict()
    extra = sorted(set(s_on) - set(s_off))
    assert extra == sorted(f"tap_bn.{i}.{k}" for i in range(4) for k in BN_KEYS)
    assert set(s_off) <= set(s_on)
    for k in s_off:  # every other decoder value as without the option (the tiny
        if not k.startswith("encoder."):  # encoder is uninitialised until init_weights)
            assert torch.equal(s_on[k], s_off[k]), k
    for bn in on.tap_bn:
        assert type(bn) is nn.BatchNorm1d and bn.num_features == C_ENC
        assert bn.affine and bn.track_running_stats
        assert (bn.momentum, bn.eps) == (0.1, 1e-5)
        assert torch.equal(bn.weight, torch.ones(C_ENC))
        assert torch.equal(bn.bias, torch.zeros(C_ENC))
    n_on = sum(p.numel() for p in on.parameters() if p.requires_grad)
    n_off = sum(p.numel() for p in off.parameters() if p.requires_grad)
    assert n_on - n_off == 2 * 4 * C_ENC


@pytest.mark.parametrize("value", ["True", 1, None])
def test_tap_bn_must_be_bool(value):
    with pytest.raises(ValueError, match="tap_bn"):
        build_model(dict(TINY, tap_bn=value))


def test_tap_bn_normalises_each_tap():
    """Train: every tapped layer standardised over the batch's tokens per channel (the
    raw taps from the same encoder), the running statistics updated (momentum 0.1,
    unbiased variance); eval: the running statistics and the affine."""
    model, ref = tiny(tap_bn=True), tiny()
    image = torch.randn(2, 3, *HW, generator=torch.Generator().manual_seed(3))
    model.train(), ref.train()  # the encoder's RoPE augmentation: the same draws
    torch.manual_seed(1)
    raw = ref.encode(image)
    torch.manual_seed(1)
    got = model.encode(image)
    assert got.shape == raw.shape == (2, (HW[0] // 16) * (HW[1] // 16), 4 * C_ENC)
    assert got.dtype == torch.float32 and got.requires_grad and not raw.requires_grad
    x = raw.reshape(-1, 4, C_ENC)  # [tokens, tap, channel]
    mean, var = x.mean(0), x.var(0, unbiased=False)
    want = ((x - mean) / torch.sqrt(var + 1e-5)).reshape(raw.shape)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    for i, bn in enumerate(model.tap_bn):
        torch.testing.assert_close(bn.running_mean, 0.1 * mean[i])
        torch.testing.assert_close(bn.running_var, 0.9 + 0.1 * x[:, i].var(0))
        assert bn.num_batches_tracked.item() == 1

    randomize_bn_(model)
    model.eval(), ref.eval()
    with torch.no_grad():
        raw, got = ref.encode(image), model.encode(image)
    x = raw.reshape(-1, 4, C_ENC)
    rm = torch.stack([bn.running_mean for bn in model.tap_bn])
    rv = torch.stack([bn.running_var for bn in model.tap_bn])
    w = torch.stack([bn.weight for bn in model.tap_bn])
    b = torch.stack([bn.bias for bn in model.tap_bn])
    want = ((x - rm) / torch.sqrt(rv + 1e-5) * w + b).reshape(raw.shape)
    torch.testing.assert_close(got, want.detach(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("selection", ["stage", "shared"])
def test_tap_bn_stream_matches_concat(selection):
    """The per-layer encode_select path (high-res inference) applies the same BN."""
    model = randomize_bn_(tiny(tap_bn=True, selection=selection)).eval()
    image = torch.randn(1, 3, 40, 56, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        concat = model.infer(image)
        model.selection_impl = "stream"
        stream = model.infer(image)
    assert concat.shape == (1, 5, 40, 56)
    torch.testing.assert_close(stream, concat, atol=1e-4, rtol=1e-4)
    off = tiny(selection=selection).eval()
    with torch.no_grad():
        assert not torch.allclose(off.infer(image), concat, atol=1e-3)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_tap_bn_trains(precision):
    """Gradients reach the BN affine (fp32 also under the bf16 decoder), none the
    frozen encoder; the BN runs in fp32 under an autocast."""
    model = tiny(tap_bn=True, precision=precision).train()
    out = model(make_data("segment", HW, seed=2))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for bn in model.tap_bn:
        for p in (bn.weight, bn.bias):
            assert p.grad is not None and p.grad.dtype == torch.float32
            assert p.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.encoder.parameters())
    feat = torch.randn(2, 6, C_ENC)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert model._tap_norm(feat.bfloat16(), 0).dtype == torch.float32


def test_tap_bn_sync_batchnorm():
    """The trainer's sync_bn converts the tap BNs (and nothing else) to SyncBatchNorm,
    which takes the [tokens, C] input; outside DDP it is the same batch norm."""
    model = tiny(tap_bn=True)
    ref = copy.deepcopy(model)
    sync = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    assert all(type(m) is nn.SyncBatchNorm for m in sync.tap_bn)
    assert sum(isinstance(m, nn.SyncBatchNorm) for m in sync.modules()) == 4
    assert sync.state_dict().keys() == ref.state_dict().keys()
    data = make_data("segment", HW, seed=2)
    sync.train(), ref.train()
    torch.manual_seed(5)
    a = sync(data)["loss"]
    torch.manual_seed(5)
    b = ref(data)["loss"]
    assert torch.equal(a, b)
    assert_state_equal(sync.state_dict(), ref.state_dict())  # running statistics
    sync.eval(), ref.eval()
    with torch.no_grad():
        assert torch.equal(sync(data)["pred_segment"], ref(data)["pred_segment"])


# ---------------------------------------------------------------- param groups


def test_keyword_lists():
    model = tiny()
    names = [n for n, _ in model.named_parameters()]
    groups = keyword_groups(
        model,
        [
            dict(keyword="encoder", lr=1e-4),
            dict(keyword=["norm", "gamma"], weight_decay=0.0),
        ],
        1e-3,
    )
    want = [n for n in names if not n.startswith("encoder.")]
    want = [n for n in want if "norm" in n or "gamma" in n]
    assert groups[2]["names"] == want and len(want) == 4 * 6  # norm1/2 (w, b), ls1/2
    # a one-word list is the string
    one = keyword_groups(model, [dict(keyword=["attn"], lr=1e-4)], 1e-3)
    plain = keyword_groups(model, [dict(keyword="attn", lr=1e-4)], 1e-3)
    assert [g["names"] for g in one] == [g["names"] for g in plain]
    for bad in ([], [""], ["norm", 1], None, "", 3):
        with pytest.raises(ValueError, match="keyword"):
            keyword_groups(model, [dict(keyword=bad)], 1e-3)


def test_average_states_takes_last_bn_counter(tmp_path):
    a = dict(w=torch.ones(2), n=torch.tensor(3), c=torch.tensor(1))
    b = dict(w=torch.zeros(2), n=torch.tensor(5), c=torch.tensor(1))
    a["tap_bn.0.num_batches_tracked"], b["tap_bn.0.num_batches_tracked"] = a.pop(
        "n"
    ), b.pop("n")
    paths = [tmp_path / "a.pth", tmp_path / "b.pth"]
    for path, state in zip(paths, (a, b)):
        torch.save(dict(epoch=1, state_dict=state), path)
    avg = average_states(paths)
    assert torch.equal(avg["w"], torch.full((2,), 0.5))
    assert avg["tap_bn.0.num_batches_tracked"].item() == 5
    b["c"] = torch.tensor(2)  # any other integer tensor must agree
    torch.save(dict(epoch=1, state_dict=b), paths[1])
    with pytest.raises(RuntimeError, match="non-float c"):
        average_states(paths)
