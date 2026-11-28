"""
MuonAdamW (ldpt/utils/optimizer.py), CPU.

Newton-Schulz singular values; one step on a rank-1 coherent gradient (spectral step
lr x scale, not Adam's lr sqrt(mn)); the update rule against a hand-written reference
(Nesterov momentum, row blocks, decoupled decay); the AdamW part bit for bit
torch.optim.AdamW; OneCycleLR leaves the matrix groups' momentum alone while their lr
follows the cycle; state_dict round trip bit for bit, an AdamW state refused; the
routing of a tiny LatentDPT; the default AdamW path unchanged; a trainer crash +
mid-epoch resume equal to an uninterrupted run.
"""

import copy
import io
import math

import pytest
import torch
import torch.nn as nn

from helpers import REPO, DINOv3TinyTest, init_encoder_
from test_resume import CONFIG, MID_EPOCH, RecordBatches, build, build_resume
from test_resume import assert_state_equal, find_hook

from ldpt.models.builder import build_model
from ldpt.utils.config import ConfigDict
from ldpt.utils.optimizer import (
    MuonAdamW,
    build_optimizer,
    muon_param_groups,
    muon_scale,
    newton_schulz,
)
from ldpt.utils.scheduler import build_scheduler

TINY = dict(
    type="L-DPT-v1m1",
    encoder=dict(type="dinov3_tiny_test"),
    encoder_inter_layers=[0, 1, 2, 3],
    dec_dim=32,
    dec_num_heads=2,
    dec_num_layers=(1, 1, 1, 1),
    dec_attn_downscale=2,
)


def cfg(**kw):
    return ConfigDict(dict(type="MuonAdamW", lr=1e-3, weight_decay=0.1, **kw))


class Toy(nn.Module):
    """A fused qkv weight, a hidden MLP, a norm, a small output layer, a conv."""

    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(32, 96, bias=False)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.norm = nn.LayerNorm(32)
        self.out = nn.Linear(32, 4)  # a side below 16: AdamW
        self.conv = nn.Conv2d(4, 4, 3)
        self.zero = nn.Linear(32, 32, bias=False)  # zero at build: AdamW
        nn.init.zeros_(self.zero.weight)


def toy_grads(model, seed):
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g)


def names_of(model, group):
    ids = {id(p): n for n, p in model.named_parameters()}
    return {ids[id(p)] for p in group["params"]}


# ---------------------------------------------------------------- Newton-Schulz


@pytest.mark.parametrize("shape", [(32, 128), (128, 32), (48, 96), (16, 256)])
def test_newton_schulz_singular_values(shape):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, *shape, generator=g)
    y = newton_schulz(x, 5)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    s = torch.linalg.svdvals(y.float())
    assert s.min() > 0.6 and s.max() < 1.3, (s.min(), s.max())
    # the direction of U V^T
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    cos = (y.float() * (u @ vh)).sum((-2, -1)) / (
        y.float().norm(dim=(-2, -1)) * u.shape[-1] ** 0.5
    )
    assert cos.min() > 0.97
    with torch.autocast("cpu", dtype=torch.bfloat16):  # autocast off inside
        assert torch.equal(newton_schulz(x, 5), y)


def test_muon_scale():
    assert muon_scale("match_rms", 1024, 4096) == pytest.approx(0.2 * 64)
    assert muon_scale("spectral", 4096, 1024) == pytest.approx(2.0)
    assert muon_scale("spectral", 1024, 4096) == 1.0
    with pytest.raises(ValueError):
        muon_scale("rms", 4, 4)


# ---------------------------------------------------------------- update rule


def test_rank_one_coherent_step():
    """A sign-coherent rank-1 gradient: the orthogonalised step has spectral norm
    about lr x scale, Adam's about lr sqrt(m n) (a sign matrix)."""
    m, n, lr = 64, 256, 1e-3
    g = torch.Generator().manual_seed(0)
    u = torch.randn(m, generator=g).sign()
    v = torch.randn(n, generator=g).sign()
    grad = torch.outer(u, v) * torch.rand(m, n, generator=g).add(0.5)  # coherent signs
    w0 = torch.randn(m, n, generator=g) * 0.02
    steps = {}
    for kind in ("muon", "adam"):
        w = nn.Parameter(w0.clone())
        w.grad = grad.clone()
        if kind == "muon":
            opt = MuonAdamW([dict(params=[w], muon=True)], lr=lr, weight_decay=0.0)
        else:
            opt = torch.optim.AdamW([w], lr=lr, weight_decay=0.0)
        opt.step()
        steps[kind] = torch.linalg.matrix_norm(w.detach() - w0, ord=2).item()
    scale = muon_scale("match_rms", m, n)
    assert 0.6 * lr * scale < steps["muon"] < 1.3 * lr * scale
    assert steps["adam"] == pytest.approx(lr * math.sqrt(m * n), rel=0.05)
    assert steps["muon"] < 0.05 * steps["adam"]


@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("mode", ["match_rms", "spectral"])
def test_update_rule_reference(nesterov, mode):
    """Two steps against the formula: B <- lerp(B, g, 1 - mu); O = NS(lerp(g, B, mu))
    (or NS(B)) per row block; W <- W (1 - lr wd) - lr s O."""
    model = Toy()
    lr, wd, mu = 0.01, 0.1, 0.95
    groups, _ = muon_param_groups(
        model, lr=lr, weight_decay=wd, muon_split=[("attn.qkv", 3)]
    )
    opt = MuonAdamW(groups, lr=lr, weight_decay=wd, nesterov=nesterov, muon_scale=mode)
    muon = [g for g in opt.param_groups if g["muon"]]
    assert len(muon) == 1 and names_of(model, muon[0]) == {
        "attn.qkv.weight",
        "fc1.weight",
        "fc2.weight",
    }
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    bufs = {n: torch.zeros_like(p) for n, p in ref.items()}
    blocks = {"attn.qkv.weight": 3, "fc1.weight": 1, "fc2.weight": 1}
    for step in range(2):
        toy_grads(model, step)
        for n, k in blocks.items():
            p = dict(model.named_parameters())[n]
            bufs[n].lerp_(p.grad, 1 - mu)
            u = p.grad.lerp(bufs[n], mu) if nesterov else bufs[n]
            r, c = p.shape[0] // k, p.shape[1]
            o = newton_schulz(u.reshape(k, r, c)).reshape(p.shape).float()
            ref[n] = ref[n] * (1 - lr * wd) - lr * muon_scale(mode, r, c) * o
        opt.step()
        for n in blocks:
            p = dict(model.named_parameters())[n]
            torch.testing.assert_close(p.detach(), ref[n], rtol=0, atol=1e-6)
            assert torch.equal(opt.state[p]["momentum_buffer"], bufs[n])


def test_adamw_part_is_torch_adamw():
    """MuonAdamW's AdamW groups step bit for bit as torch.optim.AdamW on them."""
    torch.manual_seed(0)
    model = Toy()
    ref = copy.deepcopy(model)
    param_dicts = [dict(keyword="fc2", lr=3e-4, weight_decay=0.05)]
    opt = build_optimizer(cfg(betas=(0.9, 0.95), eps=1e-6), model, param_dicts)
    names = {id(p): n for n, p in model.named_parameters()}
    ref_params = dict(ref.named_parameters())
    adam_groups = [
        dict(
            params=[ref_params[names[id(p)]] for p in g["params"]],
            **{k: g[k] for k in ("lr", "weight_decay") if k in g},
        )
        for g in opt.param_groups
        if not g["muon"]
    ]
    ref_opt = torch.optim.AdamW(
        adam_groups, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-6
    )
    adam_names = {
        names[id(p)] for g in opt.param_groups if not g["muon"] for p in g["params"]
    }
    assert "fc2.bias" in adam_names and "zero.weight" in adam_names
    for step in range(3):
        toy_grads(model, step)
        toy_grads(ref, step)
        opt.step()
        ref_opt.step()
        for n, p in model.named_parameters():
            if n in adam_names:
                assert torch.equal(p, ref_params[n]), n
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    assert torch.equal(
                        opt.state[p][key], ref_opt.state[ref_params[n]][key]
                    )


# ---------------------------------------------------------------- schedule


@pytest.mark.parametrize("muon_lr", [None, 0.02])
def test_one_cycle_leaves_muon_momentum(muon_lr):
    model = Toy()
    param_dicts = [dict(keyword="fc2", lr=1e-4)]
    opt = build_optimizer(cfg(muon_lr=muon_lr), model, param_dicts)
    sched = build_scheduler(
        ConfigDict(
            dict(
                type="OneCycleLR",
                max_lr=[1e-3, 1e-4],
                total_steps=20,
                pct_start=0.2,
                anneal_strategy="cos",
                div_factor=10.0,
                final_div_factor=1000.0,
            )
        ),
        opt,
    )
    beta1 = []
    for _ in range(19):
        toy_grads(model, 0)
        opt.step()
        sched.step()
        groups = opt.param_groups
        by_source = {g["source"]: g for g in groups if not g["muon"]}
        for g in groups:
            if g["muon"]:
                assert "betas" not in g and "momentum" not in g
                assert g["muon_momentum"] == 0.95
                src = by_source[g["source"]]
                # muon_lr keeps its ratio to the keyword group's lr
                ratio = 1.0 if muon_lr is None else muon_lr / (1e-3, 1e-4)[g["source"]]
                assert g["lr"] == pytest.approx(src["lr"] * ratio, rel=1e-12)
        beta1.append(groups[0]["betas"][0])
    assert min(beta1) < 0.9 < max(beta1)  # the AdamW groups still cycle beta1
    with pytest.raises(ValueError, match="one value per keyword group"):
        build_scheduler(
            ConfigDict(
                dict(type="OneCycleLR", max_lr=[1e-3, 1e-4, 1e-5], total_steps=20)
            ),
            build_optimizer(cfg(), Toy(), param_dicts),
        )


def test_default_adamw_path_unchanged():
    """type AdamW: the groups and steps of the code before MuonAdamW, and OneCycleLR is
    torch's (lr and betas every step)."""
    torch.manual_seed(0)
    model = Toy()
    ref = copy.deepcopy(model)
    param_dicts = [ConfigDict(dict(keyword="fc2", lr=1e-4))]
    opt = build_optimizer(
        ConfigDict(dict(type="AdamW", lr=1e-3, weight_decay=0.1)), model, param_dicts
    )
    assert type(opt) is torch.optim.AdamW
    rp = dict(ref.named_parameters())
    ref_opt = torch.optim.AdamW(
        [
            dict(params=[p for n, p in rp.items() if "fc2" not in n], lr=1e-3),
            dict(params=[p for n, p in rp.items() if "fc2" in n], lr=1e-4),
        ],
        lr=1e-3,
        weight_decay=0.1,
    )
    kw = dict(total_steps=10, pct_start=0.2, div_factor=10.0, final_div_factor=1000.0)
    sched = build_scheduler(
        ConfigDict(dict(type="OneCycleLR", max_lr=[1e-3, 1e-4], **kw)), opt
    )
    ref_sched = torch.optim.lr_scheduler.OneCycleLR(ref_opt, max_lr=[1e-3, 1e-4], **kw)
    for step in range(9):
        toy_grads(model, step)
        toy_grads(ref, step)
        opt.step()
        ref_opt.step()
        sched.step()
        ref_sched.step()
        for a, b in zip(opt.param_groups, ref_opt.param_groups):
            assert {k: v for k, v in a.items() if k != "params"} == {
                k: v for k, v in b.items() if k != "params"
            }
        for n, p in model.named_parameters():
            assert torch.equal(p, rp[n]), n
    # no param_dicts: model.parameters(), one group
    opt = build_optimizer(ConfigDict(dict(type="AdamW", lr=1e-3)), Toy(), None)
    assert len(opt.param_groups) == 1 and "muon" not in opt.param_groups[0]


# ---------------------------------------------------------------- checkpoints


def train(model, opt, sched, steps, seed0):
    for s in range(steps):
        toy_grads(model, seed0 + s)
        opt.step()
        sched.step()


def make(seed=0):
    torch.manual_seed(seed)
    model = Toy()
    opt = build_optimizer(cfg(), model, [dict(keyword="fc2", lr=1e-4)])
    sched = build_scheduler(
        ConfigDict(dict(type="OneCycleLR", max_lr=[1e-3, 1e-4], total_steps=20)), opt
    )
    return model, opt, sched


def test_state_dict_round_trip_bit_for_bit():
    model, opt, sched = make()
    train(model, opt, sched, 3, 0)
    buf = io.BytesIO()
    torch.save(
        dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict()),
        buf,
    )
    buf.seek(0)
    saved = torch.load(buf, weights_only=False)
    resumed, ropt, rsched = make(seed=1)  # other init: everything comes from the file
    resumed.load_state_dict(saved["model"])
    ropt.load_state_dict(saved["opt"])
    rsched.load_state_dict(saved["sched"])
    train(model, opt, sched, 4, 3)
    train(resumed, ropt, rsched, 4, 3)
    for (n, p), q in zip(model.named_parameters(), resumed.parameters()):
        assert torch.equal(p, q), n
    a, b = opt.state_dict(), ropt.state_dict()
    assert a["param_groups"] == b["param_groups"]
    for i in a["state"]:
        assert_state_equal(a["state"][i], b["state"][i])
    # the inner AdamW follows the loaded groups (lr written by the scheduler)
    assert all(
        g is h
        for g, h in zip(
            ropt._adamw.param_groups, [g for g in ropt.param_groups if not g["muon"]]
        )
    )
    assert ropt._adamw.state is ropt.state
    clone = copy.deepcopy(ropt)  # a deep copy steps the copy's parameters
    clone.step()


def test_adamw_state_refused():
    model, opt, _ = make()
    adamw = build_optimizer(
        ConfigDict(dict(type="AdamW", lr=1e-3)), model, [dict(keyword="fc2", lr=1e-4)]
    )
    toy_grads(model, 0)
    adamw.step()
    with pytest.raises(ValueError, match="an AdamW .not a MuonAdamW. state"):
        opt.load_state_dict(adamw.state_dict())
    # the same group sizes, another matrix in the matrix part (zero.weight is no
    # longer zero)
    other = build_optimizer(
        cfg(muon_exclude=["fc1"]), model, [dict(keyword="fc2", lr=1e-4)]
    )
    other.step()
    with pytest.raises(ValueError, match="another routing"):
        opt.load_state_dict(other.state_dict())


# ---------------------------------------------------------------- routing


def test_latent_dpt_routing():
    model = build_model(dict(TINY, freeze_encoder=False))
    opt = build_optimizer(cfg(), model, [dict(keyword="encoder", lr=1e-4)])
    muon = set().union(*(names_of(model, g) for g in opt.param_groups if g["muon"]))
    adam = set().union(*(names_of(model, g) for g in opt.param_groups if not g["muon"]))
    assert muon.isdisjoint(adam)
    assert muon | adam == {n for n, _ in model.named_parameters()}
    for n in muon:
        assert n.endswith(("_proj.weight", "fc1.weight", "fc2.weight")) or n.startswith(
            "neck_blocks."
        ), n
        assert not n.startswith("encoder.")  # trained encoder: excluded by default
    for s in range(4):
        for part in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert f"dec_blocks.{s}.0.attn.{part}.weight" in muon
        assert {
            f"dec_blocks.{s}.0.mlp.fc1.weight",
            f"dec_blocks.{s}.0.mlp.fc2.weight",
        } <= muon
        assert f"neck_blocks.{s}.weight" in muon
        assert f"dec_blocks.{s}.0.ls1.gamma" in adam
        assert f"dec_blocks.{s}.0.norm1.weight" in adam
        assert f"dec_blocks.{s}.0.attn.out_proj.bias" in adam
    assert {"unpatchfy.0.proj.weight", "unpatchfy.0.proj.bias"} <= adam
    assert "encoder.blocks.0.attn.qkv.weight" in adam
    assert "kept on AdamW, excluded by 'encoder.'" in opt.routing

    seg = build_model(
        dict(
            TINY,
            task="semseg",
            out_dim=16,
            num_classes=5,
            criterion=[dict(type="ce_loss", loss_weight=1.0, ignore_index=-1)],
        )
    )
    opt = build_optimizer(cfg(), seg, None)
    adam = set().union(*(names_of(seg, g) for g in opt.param_groups if not g["muon"]))
    assert {"seg_head.weight", "seg_head.bias", "unpatchfy.0.proj.weight"} <= adam
    assert "frozen" in opt.routing  # the frozen encoder's matrices


def test_toy_routing_reasons():
    model = Toy()
    _, routing = muon_param_groups(
        model, lr=1e-3, weight_decay=0.0, muon_exclude=["fc2", "nothing"]
    )
    assert "excluded by 'fc2'" in routing
    assert "a side below 16: out.weight (4, 32)" in routing
    assert "zero at build: zero.weight" in routing
    assert "matching nothing: ['nothing']" in routing
    with pytest.raises(ValueError, match="do not divide"):
        muon_param_groups(model, lr=1e-3, weight_decay=0.0, muon_split=[("fc1", 5)])
    with pytest.raises(ValueError, match="momentum"):
        muon_param_groups(
            model, [dict(keyword="fc1", momentum=0.9)], lr=1e-3, weight_decay=0.0
        )


# ---------------------------------------------------------------- trainer


def test_trainer_mid_epoch_resume(tmp_path):
    """The trainer with MuonAdamW: crash + mid-epoch resume ends bit for bit where an
    uninterrupted run ends (weights, optimizer, scheduler)."""
    pretrained = tmp_path / "dinov3_tiny_pretrain.pth"
    torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), pretrained)
    base = tmp_path / "base.py"
    adamw = 'optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.1)'
    assert adamw in CONFIG
    base.write_text(
        CONFIG.replace(
            adamw,
            'optimizer = dict(type="MuonAdamW", lr=base_lr, weight_decay=0.1, '
            'muon_scale="match_rms")\nclip_grad = 1.0',
        ).format(
            base=f"{REPO}/configs/_base_/default_runtime.py", pretrained=pretrained
        )
    )

    def run(name, crash):
        config = tmp_path / f"{name}.py"
        config.write_text(MID_EPOCH.format(base=base, crash=crash))
        return build(config, save_path=str(tmp_path / name))

    reference = run("reference", crash="[]")
    assert isinstance(reference.optimizer, MuonAdamW)
    assert any(g["muon"] for g in reference.optimizer.param_groups)
    reference.train()
    crashed = run("exp", crash='[dict(type="CrashAtTest", epoch=1, iter=3)]')
    with pytest.raises(SystemExit):
        crashed.train()
    resumed = build_resume(tmp_path / "exp")
    resumed.train()
    assert (resumed.start_epoch, resumed.start_iter) == (1, 2)
    trained = find_hook(crashed, RecordBatches).batches
    rest = find_hook(resumed, RecordBatches).batches
    assert trained[:8] + rest == find_hook(reference, RecordBatches).batches
    assert_state_equal(resumed.model.state_dict(), reference.model.state_dict())
    ref_opt, opt = reference.optimizer.state_dict(), resumed.optimizer.state_dict()
    assert opt["param_groups"] == ref_opt["param_groups"]
    for i in opt["state"]:
        assert_state_equal(opt["state"][i], ref_opt["state"][i])
    assert resumed.scheduler.state_dict() == reference.scheduler.state_dict()
