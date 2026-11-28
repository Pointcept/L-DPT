"""
model.dec_qk_norm (QK-norm, cosine attention with a learned capped temperature, an
experimental option that bounds the attention logits): in every decoder
attention q and k are L2-normalised per head and token (fp32, before RoPE) and scaled
by sqrt(tau_h sqrt(d)), so SDPA's 1 / sqrt(d) gives the logit tau_h cos(q, R k) with
tau_h = min((sqrt(d) / 3) exp(30 rho_h), dec_qk_tau_max), rho_h = attn.logit_scale[h].

Off (the default): no parameter, no branch, bitwise the model without it (also
test_l_dpt_golden.py and test_structural_options.test_off_bit_identical_to_reference).
On: one key per attention block, |logit| <= tau_max for any weights and input, the
attention of the model without it at init, no gradient along the scale of W_q / W_k.
CPU.
"""

import copy
import json
import math

import pytest
import torch
import torch.nn.functional as F

from helpers import REPO, DINOv3TinyTest, init_encoder_
from test_evaluate import CONFIG as EVAL_CONFIG
from test_evaluate import load_script
from test_l_dpt_golden import BASE, DEPTH_LOSS, make_data
from test_l_dpt_options import island_model, parse, plain_l_config, resume, train
from test_l_dpt_options import loss_run
from test_resume import CONFIG as RESUME_CONFIG
from test_resume import MID_EPOCH, assert_state_equal, build

from ldpt.engines.evaluate import load_decoder
from ldpt.hooks.evaluator import read_metrics_history
from ldpt.models.builder import build_model
from ldpt.models.dinov3.layers import RopePositionEmbedding
from ldpt.models.dinov3.layers.attention import rope_apply
from ldpt.models.l_dpt.modules import (
    QK_NORM_EPS,
    QK_TAU_GAIN,
    QK_TAU_MAX,
    Attention,
    Block,
)
from ldpt.utils.config import Config

BF16_BOUND = QK_TAU_MAX * (1 + 2**-8) ** 2  # 201.57: bf16 operands round by 2^-8
# rho at which every head sits at the cap (tau_init e^(30 rho) >> 200)
RHO_WALL = 1.0


def rope_of(h, w, dim, heads):
    r = RopePositionEmbedding(
        embed_dim=dim, num_heads=heads, base=100, normalize_coords="min"
    ).eval()
    with torch.no_grad():
        return r(H=h, W=w)


def tokens(kind, b, n, dim, seed=0):
    """White tokens, or a common-mode stream (98 % of the energy in the token mean,
    as the decoder stream of the runaway)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, dim, generator=g)
    if kind == "common":
        x = 7.0 * torch.randn(1, 1, dim, generator=g) + x
    return x


def logits_of(attn, h, rope, dtype=None):
    """fp64 logits q.k / sqrt(d) of q / k as qkv returns them (cast to dtype first,
    as SDPA gets them with attn_dtype bf16)."""
    q, k, _ = attn.qkv(h, rope=rope)
    if dtype is not None:
        q, k = q.to(dtype), k.to(dtype)
    return q.double() @ k.double().transpose(-1, -2) / math.sqrt(q.shape[-1])


def to_qk_norm(attn, **kw):
    """attn's weights in an Attention with qk_norm (rho = 0)."""
    dim = attn.q_proj.in_features
    new = Attention(
        dim,
        num_heads=attn.num_heads,
        qkv_bias=attn.q_proj.bias is not None,
        proj_bias=attn.out_proj.bias is not None,
        downscale=dim // attn.q_proj.out_features,
        attn_dtype="fp32" if attn.attn_dtype is None else "bf16",
        force_fp32=attn.force_fp32,
        qk_norm=True,
        **kw,
    )
    missing, unexpected = new.load_state_dict(attn.state_dict(), strict=False)
    assert missing == ["logit_scale"] and not unexpected
    return new


def qkn_model(precision="bf16", **options):
    """island_model with dec_qk_norm; randomize_ draws logit_scale like a weight
    (most heads at the cap), so it is reset to small temperatures around tau_init."""
    model = island_model(precision, dec_qk_norm=True, **options)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("attn.logit_scale"):
                p.copy_(torch.linspace(-0.02, 0.03, p.numel()))
    return model


def logit_scales(model):
    return {n: p for n, p in model.named_parameters() if n.endswith("attn.logit_scale")}


# ---------------------------------------------------------------- off / on, keys, RNG


def test_qk_norm_off_and_keys():
    """qk_norm=False is bitwise the plain module (same keys); qk_norm=True adds one key
    "logit_scale" (zeros, registered first), draws no random number and keeps every
    other initial weight; tau_init = sqrt(d) / 3 (2.667 at d = 64, 1.886 at d = 32)."""
    dim, heads = 256, 4
    torch.manual_seed(0)
    ref = Attention(dim, num_heads=heads, attn_dtype="bf16")
    after_ref = torch.rand(1)
    torch.manual_seed(0)
    off = Attention(dim, num_heads=heads, attn_dtype="bf16", qk_norm=False)
    torch.manual_seed(0)
    on = Attention(dim, num_heads=heads, attn_dtype="bf16", qk_norm=True)
    after_on = torch.rand(1)
    assert torch.equal(after_ref, after_on)
    x, rope = tokens("white", 2, 64, dim), rope_of(8, 8, dim, heads)
    with torch.no_grad():
        assert torch.equal(ref(x, rope=rope), off(x, rope=rope))
    assert list(ref.state_dict()) == list(off.state_dict())
    assert not hasattr(off, "logit_scale") and not off.qk_norm
    keys = list(on.state_dict())
    assert keys[0] == "logit_scale" and keys[1:] == list(ref.state_dict())
    assert all(torch.equal(v, on.state_dict()[k]) for k, v in ref.state_dict().items())
    assert torch.equal(on.logit_scale, torch.zeros(heads))
    assert on.tau_init == pytest.approx(8 / 3, abs=1e-12)
    assert on.tau_max == QK_TAU_MAX == 200.0
    assert torch.equal(on.temperature(), torch.full((heads,), 8 / 3))
    assert Attention(1024, 32, qk_norm=True).tau_init == pytest.approx(1.8856, 1e-4)


def test_qk_norm_model_keys_and_params():
    """LatentDPT: off (default or explicit) adds no state-dict key; on has
    one key dec_blocks.{s}.{i}.attn.logit_scale [heads] per attention block, first
    among the block's attn.* keys, every other initial weight and the RNG stream
    after __init__ unchanged; dec_conv_mixer blocks have none."""
    config = dict(BASE, criterion=DEPTH_LOSS, dec_num_layers=(1, 1, 2, 2))
    models, after = {}, {}
    for name, options in (
        ("default", {}),
        ("off", dict(dec_qk_norm=False, dec_qk_tau_max=200.0)),
        ("on", dict(dec_qk_norm=True)),
        ("on_conv", dict(dec_qk_norm=True, dec_conv_mixer="last")),
    ):
        torch.manual_seed(0)
        models[name] = build_model(dict(config, **options))
        after[name] = torch.rand(1)
    assert all(torch.equal(after["default"], a) for a in after.values())
    ref = models["default"].state_dict()
    for name in ("off", "on"):
        sd = models[name].state_dict()
        extra = [k for k in sd if k not in ref]
        assert set(ref) <= set(sd)
        # (the encoder's cls / storage / mask tokens are torch.empty)
        decoder = [k for k in ref if not k.startswith("encoder.")]
        assert all(torch.equal(ref[k], sd[k]) for k in decoder), name
        if name == "off":
            assert list(sd) == list(ref)
            continue
        blocks = [(s, i) for s, n in enumerate((1, 1, 2, 2)) for i in range(n)]
        assert extra == [f"dec_blocks.{s}.{i}.attn.logit_scale" for s, i in blocks]
        for s, i in blocks:
            attn_keys = [k for k in sd if k.startswith(f"dec_blocks.{s}.{i}.attn.")]
            assert attn_keys[0] == f"dec_blocks.{s}.{i}.attn.logit_scale"
            assert sd[attn_keys[0]].shape == (BASE["dec_num_heads"],)
    conv = models["on_conv"]
    assert "dec_blocks.3.1.attn.logit_scale" not in conv.state_dict()
    assert "dec_blocks.3.0.attn.logit_scale" in conv.state_dict()
    # the parameter counts of the L config (16 blocks x 16 heads)
    count = {}
    for options in ({}, dict(dec_qk_norm=True)):
        cfg = Config.fromfile(
            f"{REPO}/configs/hybrid/depth-l-dpt-v1m1-1-large-mixed.py"
        )
        cfg.model.update(options, dec_qk_bound=None)
        with torch.device("meta"):
            m = build_model(cfg.model)
        count[bool(options)] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert count == {False: 270_995_712, True: 270_995_968}


# ---------------------------------------------------------------- the formula


def test_qk_norm_formula():
    """The fp32 module equals the fp64 formula softmax(tau_h cos(q, R k)) v with RoPE
    applied after the normalisation."""
    dim, heads, d = 256, 4, 64
    torch.manual_seed(1)
    a = Attention(dim, num_heads=heads, qk_norm=True)
    x = tokens("white", 2, 256, dim)
    rope = rope_of(16, 16, dim, heads)
    with torch.no_grad():
        a.logit_scale.copy_(torch.linspace(-0.1, 0.15, heads))
        tau = a.temperature()
        want_tau = (math.sqrt(d) / 3) * torch.exp(QK_TAU_GAIN * a.logit_scale.double())
        assert torch.allclose(tau.double(), want_tau.clamp(max=200.0), rtol=1e-6)
        assert tau.max() == 200.0 and tau.min() < 1  # one head at the cap
        h = F.layer_norm(x, (dim,))
        B, N = h.shape[:2]
        q, k, v = [
            p(h).reshape(B, N, heads, d).transpose(1, 2).double()
            for p in (a.q_proj, a.k_proj, a.v_proj)
        ]
        sin, cos = [t.double() for t in rope]
        qn = rope_apply(F.normalize(q, dim=-1), sin, cos)
        kn = rope_apply(F.normalize(k, dim=-1), sin, cos)
        want_logits = tau.double()[None, :, None, None] * (qn @ kn.transpose(-1, -2))
        got_logits = logits_of(a, h, rope)
        want = (want_logits.softmax(-1) @ v).transpose(1, 2).reshape(B, N, dim)
        want = want @ a.out_proj.weight.double().T + a.out_proj.bias.double()
        got = a(h, rope=rope).double()
    assert (got_logits - want_logits).abs().max() < 1e-4
    assert (got - want).abs().max() < 1e-5
    # |q'| = |k'| = sqrt(tau_h sqrt(d)) per head and token
    q2, k2, _ = a.qkv(h, rope=rope)
    radius = torch.sqrt(tau * math.sqrt(d))[None, :, None]
    for t in (q2, k2):
        torch.testing.assert_close(t.norm(dim=-1), radius.expand_as(t[..., 0]))


# ---------------------------------------------------------------- at initialisation


@pytest.mark.parametrize("kind", ["white", "common"])
def test_qk_norm_at_init_matches_the_old_attention(kind):
    """L's attention (dim 1024, 16 heads, default init), 32 x 32 tokens: with rho = 0
    the logit std is within 1 % of the unnormalised attention's on white tokens (5 %
    on a common-mode stream, where |q| / |k| vary more over the tokens: 3.4 % here),
    the attention output within 5 % and the block
    output (LayerScale 1e-5) within 1e-6."""
    dim, heads = 1024, 16
    rope = rope_of(32, 32, dim, heads)
    torch.manual_seed(2)
    blk = Block(dim, num_heads=heads, init_values=1e-5, attn_dtype="fp32")
    x = tokens(kind, 1, 1024, dim, seed=3)
    qa = to_qk_norm(blk.attn)
    blk_qkn = copy.deepcopy(blk)
    blk_qkn.attn = qa
    with torch.no_grad():
        h = blk.norm1(x)
        std_ref = logits_of(blk.attn, h, rope).std().item()
        std_new = logits_of(qa, h, rope).std().item()
        a_ref, a_new = blk.attn(h, rope=rope), qa(h, rope=rope)
        y_ref, y_new = blk(x, rope=rope), blk_qkn(x, rope=rope)
    assert abs(std_new / std_ref - 1) < (0.01 if kind == "white" else 0.05), (
        std_new,
        std_ref,
    )
    assert ((a_new - a_ref).norm() / a_ref.norm()).item() < 0.05
    assert ((y_new - y_ref).norm() / y_ref.norm()).item() < 1e-6


# ---------------------------------------------------------------- the bound


@pytest.mark.parametrize("attn_dtype", ["fp32", "bf16"])
def test_qk_norm_bound_on_adversarial_weights(attn_dtype):
    """W_q = W_k x 1000 (cos = 1 on the diagonal), rho far past the cap, inputs from
    1e-20 to 1e4: |logit| <= 200 (fp32, 200.002) and <= 200 (1 + 2^-8)^2 (bf16
    operands; also under bf16 autocast, where qkv returns bf16 q / k whatever
    attn_dtype); zero tokens give a finite output (eps)."""
    dim, heads = 256, 4
    limit = QK_TAU_MAX * (1 + 1e-5) if attn_dtype == "fp32" else BF16_BOUND
    dtype = torch.bfloat16 if attn_dtype == "bf16" else None
    rope = rope_of(16, 16, dim, heads)
    torch.manual_seed(4)
    a = Attention(dim, num_heads=heads, attn_dtype=attn_dtype, qk_norm=True)
    worst = worst_amp = 0.0
    with torch.no_grad():
        a.q_proj.weight.mul_(1000.0)
        a.k_proj.weight.copy_(a.q_proj.weight)
        a.logit_scale.fill_(5.0)  # tau_init e^150 before the clamp
        assert torch.equal(a.temperature(), torch.full((heads,), QK_TAU_MAX))
        for kind, scale in (("white", 1.0), ("common", 1e4), ("white", 1e-20)):
            x = tokens(kind, 1, 256, dim, seed=5) * scale
            worst = max(worst, logits_of(a, x, rope, dtype).abs().max().item())
            assert torch.isfinite(a(x, rope=rope)).all()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = a(x, rope=rope)
                logits = logits_of(a, x, rope, dtype)
                assert logits.dtype == torch.float64
                worst_amp = max(worst_amp, logits.abs().max().item())
            assert torch.isfinite(y.float()).all()
        z = a(torch.zeros(1, 256, dim), rope=rope)
    assert QK_TAU_MAX * 0.99 < worst <= limit, worst
    assert QK_TAU_MAX * 0.99 < worst_amp <= BF16_BOUND, worst_amp
    assert torch.isfinite(z).all()


def test_qk_norm_bound_on_a_rank1_runaway_kernel():
    """The runaway path: a rank-1 kernel along the common stream on one head of W_q
    and W_k (amplitude 1000). Unnormalised logits reach > 1e5; with QK-norm they stay
    at tau_init (2.67) at rho = 0 and below 201.57 at the cap (bf16 operands)."""
    dim, heads, d = 256, 4, 64
    rope = rope_of(16, 16, dim, heads)
    torch.manual_seed(9)
    blk = Block(dim, num_heads=heads, init_values=1.0, attn_dtype="bf16")
    x = tokens("common", 1, 256, dim, seed=10)
    with torch.no_grad():
        h = blk.norm1(x)
        c = F.normalize(h.mean(dim=(0, 1)), dim=0)
        u = F.normalize(torch.randn(d), dim=0)
        for W in (blk.attn.q_proj.weight, blk.attn.k_proj.weight):
            W[:d] += 1000.0 * torch.outer(u, c)
        raw = logits_of(blk.attn, h, rope, torch.bfloat16).abs().max().item()
        qa = to_qk_norm(blk.attn)
        at_init = logits_of(qa, h, rope, torch.bfloat16).abs().max().item()
        qa.logit_scale.fill_(RHO_WALL)
        at_wall = logits_of(qa, h, rope, torch.bfloat16).abs().max().item()
    assert raw > 1e5
    assert at_init <= 8 / 3 * (1 + 2**-8) ** 2
    assert at_wall <= BF16_BOUND


# ---------------------------------------------------------------- gradients


def test_qk_norm_gradients():
    """Per head <W, dL/dW> = 0 for q_proj / k_proj (no gradient along the scale of the
    weights: the runaway's amplitude direction); d / d rho non-zero below the cap and
    exactly 0 at it; the output invariant to the scale of W_q and W_k."""
    dim, heads, d = 256, 4, 64
    rope = rope_of(16, 16, dim, heads)
    torch.manual_seed(6)
    blk = Block(dim, num_heads=heads, init_values=1.0, attn_dtype="fp32")
    x = tokens("common", 2, 256, dim, seed=7)
    target = torch.randn(2, 256, dim, generator=torch.Generator().manual_seed(8))
    h = blk.norm1(x).detach()

    def grads(attn):
        attn.zero_grad()
        ((attn(h, rope=rope) - target) ** 2).mean().backward()
        return attn

    ref = grads(blk.attn)
    qa = grads(to_qk_norm(blk.attn))
    for name in ("q_proj", "k_proj"):
        W, G = getattr(qa, name).weight, getattr(qa, name).weight.grad
        Wr, Gr = getattr(ref, name).weight, getattr(ref, name).weight.grad
        radial = (W.reshape(heads, d, dim) * G.reshape(heads, d, dim)).sum((1, 2))
        radial_ref = (Wr.reshape(heads, d, dim) * Gr.reshape(heads, d, dim)).sum((1, 2))
        assert radial.abs().max() < 1e-9 + 1e-4 * radial_ref.abs().max(), name
        assert torch.isfinite(G).all() and G.norm() > 0
    assert (qa.logit_scale.grad.abs() > 0).all()
    with torch.no_grad():
        qa.logit_scale.fill_(RHO_WALL)
    grads(qa)
    assert torch.equal(qa.logit_scale.grad, torch.zeros(heads))
    qa = to_qk_norm(blk.attn)
    with torch.no_grad():
        y0 = qa(h, rope=rope)
        qa.q_proj.weight.mul_(37.0)
        qa.k_proj.weight.mul_(1 / 53.0)
        y1 = qa(h, rope=rope)
    assert (y1 - y0).abs().max() < 1e-5


def test_qk_norm_model_trains_finite():
    """The tiny L-DPT with dec_qk_norm under bf16 autocast: loss and every gradient
    finite, logit_scale has a gradient; the zero-input eps does not leak NaN."""
    model = qkn_model().train()
    loss = model(make_data("depth", (64, 80)))["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert all(p.grad.abs().sum() > 0 for p in logit_scales(model).values())
    assert QK_NORM_EPS == 1e-6


# ---------------------------------------------------------------- options


def test_qk_norm_option_validation():
    """dec_qk_norm must be a bool; dec_qk_tau_max a number in [tau_init, inf); with
    dec_qk_bound it raises (it replaces the bound)."""
    config = dict(BASE, criterion=DEPTH_LOSS, dec_num_layers=(1, 1, 1, 1))
    for bad in (1, 0, "True", "None", None):
        with pytest.raises(ValueError, match="dec_qk_norm must be True or False"):
            build_model(dict(config, dec_qk_norm=bad))
    for bound in (200.0, 1000):
        with pytest.raises(ValueError, match="dec_qk_bound must be None"):
            build_model(dict(config, dec_qk_norm=True, dec_qk_bound=bound))
    # the bound off by its "None" string composes
    build_model(dict(config, dec_qk_norm=True, dec_qk_bound="None"))
    # head dim 8 in the tiny decoder: tau_init = 0.943
    for bad in (0, 0.5, -1.0, math.inf, math.nan, True, "abc", "None", None):
        with pytest.raises(ValueError, match="qk_tau_max"):
            build_model(dict(config, dec_qk_norm=True, dec_qk_tau_max=bad))
    model = build_model(dict(config, dec_qk_norm=True, dec_qk_tau_max=400))
    assert {b.attn.tau_max for s in model.dec_blocks for b in s} == {400.0}
    with pytest.raises(ValueError, match="qk_norm bounds every logit"):
        Attention(32, num_heads=2, qk_bound=200.0, qk_norm=True)
    with pytest.raises(ValueError, match="qk_norm must be True or False"):
        Attention(32, num_heads=2, qk_norm=1)


def test_qk_norm_option_strings(tmp_path):
    """train.sh -o / --options on a plain L config: model.dec_qk_norm=True reaches every
    block, model.dec_qk_tau_max=400 its cap; other strings raise."""
    save = f"save_path={tmp_path / 'exp'}"
    tiny = dict(
        encoder=dict(type="dinov3_tiny_test"), encoder_inter_layers=[0, 1, 2, 3]
    )
    for options, on, tau_max in (
        ([], False, None),
        (["model.dec_qk_norm=True"], True, 200.0),
        (["model.dec_qk_norm=true", "model.dec_qk_tau_max=400"], True, 400.0),
        (["model.dec_qk_norm=False"], False, None),
    ):
        cfg = parse(plain_l_config(tmp_path), [save] + options)
        with torch.device("meta"):
            model = build_model(dict(cfg.model, **tiny))
        for stage in model.dec_blocks:
            for blk in stage:
                assert blk.attn.qk_norm is on
                assert getattr(blk.attn, "tau_max", None) == tau_max
                if on:
                    assert blk.attn.tau_init == pytest.approx(8 / 3)
    for bad in (["model.dec_qk_norm=1"], ["model.dec_qk_norm=None"]):
        cfg = parse(plain_l_config(tmp_path), [save] + bad)
        with pytest.raises(ValueError, match="dec_qk_norm"):
            with torch.device("meta"):
                build_model(dict(cfg.model, **tiny))


# ---------------------------------------------------------------- composition


def test_qk_norm_with_attn_fp32():
    """dec_qk_norm composes with the fp32 island (attn_fp32 / block_fp32 call qkv):
    without an autocast the island is the same math (loss and gradients within 1e-6);
    under bf16 the island's logits stay bounded and everything is finite."""
    data = make_data("depth", (64, 80))
    out = {}
    for name, options in (
        ("plain", {}),
        ("attn_fp32", dict(attn_fp32="last_stage")),
        ("block_fp32", dict(block_fp32="last")),
    ):
        model = qkn_model("fp32", **options).train()
        torch.manual_seed(1)
        loss = model(copy.deepcopy(data))["loss"]
        loss.backward()
        out[name] = (loss, [p.grad for p in model.parameters() if p.requires_grad])
    for name in ("attn_fp32", "block_fp32"):
        torch.testing.assert_close(out[name][0], out["plain"][0], rtol=1e-6, atol=1e-6)
        for a, b in zip(out[name][1], out["plain"][1]):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    model = qkn_model("bf16", attn_fp32="last").train()
    attn = model.dec_blocks[3][1].attn
    assert attn.force_fp32 and attn.qk_norm
    loss = model(copy.deepcopy(data))["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad
    )


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_qk_norm_compiled_blocks(backend):
    """compile_blocks: the training loss and every gradient (logit_scale's too) are
    bitwise those of the eager model."""
    torch._dynamo.reset()
    eager = qkn_model().train()
    compiled = copy.deepcopy(eager)
    compiled.compile_blocks(backend=backend)
    data = make_data("depth", (64, 80))
    out = []
    for m in (eager, compiled):
        torch.manual_seed(1)
        loss = m(copy.deepcopy(data))["loss"]
        loss.backward()
        out.append((loss, [p.grad for p in m.parameters() if p.requires_grad]))
    assert torch.equal(out[1][0], out[0][0])
    assert all(torch.equal(a, b) for a, b in zip(out[0][1], out[1][1]))
    torch._dynamo.reset()


def test_qk_norm_state_dict_round_trip(tmp_path):
    """A QK-norm model's weights load strictly into a model with the option (same
    eval output) and fail the strict load and the evaluator's key check without it
    (unexpected logit_scale keys); a plain checkpoint fails the other way."""
    image = make_data(None, (48, 112))["image"]
    normed = qkn_model("fp32")
    path = tmp_path / "epoch_1.pth"
    torch.save(dict(epoch=1, state_dict=normed.state_dict()), path)
    config = dict(BASE, criterion=DEPTH_LOSS, dec_num_layers=(1, 1, 2, 2))
    again = build_model(dict(config, dec_qk_norm=True))
    assert load_decoder(again, path) == 1
    with torch.no_grad():
        assert torch.equal(normed.eval().infer(image), again.eval().infer(image))
    plain = build_model(config)
    with pytest.raises(RuntimeError, match="attn.logit_scale"):
        plain.load_state_dict(normed.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="unexpected keys \\['dec_blocks.0.0.attn"):
        load_decoder(plain, path)
    with pytest.raises(RuntimeError, match="attn.logit_scale"):
        again.load_state_dict(plain.state_dict(), strict=True)


def test_qk_norm_tiny_run_optimizer_group(tmp_path):
    """A tiny loss run with model.dec_qk_norm=True: finite losses, the
    temperatures moved, logit_scale in the default parameter group (weight decay kept,
    no extra group)."""
    on, rec, _ = loss_run(tmp_path, "qk-norm", **{"model.dec_qk_norm": True})
    off, _, _ = loss_run(tmp_path, "off")
    assert len(rec.losses) == 8 and all(torch.isfinite(x) for x in rec.losses)
    scales = logit_scales(on.model)
    assert len(scales) == sum(on.model.dec_num_layers)
    assert all(p.abs().max() > 0 for p in scales.values())
    groups_on, groups_off = on.optimizer.param_groups, off.optimizer.param_groups
    assert len(groups_on) == len(groups_off)
    assert [g["weight_decay"] for g in groups_on] == [
        g["weight_decay"] for g in groups_off
    ]
    ids = {id(p) for p in scales.values()}
    (group,) = [g for g in groups_on if ids & {id(p) for p in g["params"]}]
    assert ids <= {id(p) for p in group["params"]}
    assert group is groups_on[0] and group["weight_decay"] > 0


# ---------------------------------------------------------------- trainer, evaluate, resume

QKN_OPTIONS = {"model.dec_qk_norm": True}


@pytest.fixture(scope="module")
def qkn_run(tmp_path_factory):
    """test_evaluate's tiny bf16 run (2 epochs, evaluation, epoch weights) with
    dec_qk_norm: save_path."""
    root = tmp_path_factory.mktemp("qk_norm")
    pretrained = root / "dinov3_tiny_pretrain.pth"
    torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), pretrained)
    config = root / "config_src.py"
    config.write_text(
        EVAL_CONFIG.format(
            base=f"{REPO}/configs/_base_/default_runtime.py", pretrained=pretrained
        )
    )
    save_path = root / "exp"
    train(build(config, save_path=str(save_path), **QKN_OPTIONS))
    return save_path


def test_qk_norm_evaluate_script(qkn_run, tmp_path):
    """scripts/evaluate.py on the run's config.py reproduces the logged metrics; the
    model without the option refuses the files."""
    history = read_metrics_history(qkn_run / "metrics.jsonl")
    files = [qkn_run / "model" / f"epoch_{e}.pth" for e in (1, 2)]
    output = tmp_path / "eval.json"
    load_script().main(
        ["--config-file", str(qkn_run / "config.py"), "--checkpoints"]
        + [str(p) for p in files]
        + ["--output", str(output)]
    )
    results = json.loads(output.read_text())["results"]
    for record, epoch in zip(results, (1, 2)):
        assert record["epoch"] == epoch
        assert record["score"] == pytest.approx(history[epoch]["score"], rel=1e-6)
        for group, metrics in history[epoch]["groups"].items():
            assert record["groups"][group] == pytest.approx(metrics, rel=1e-6)
    plain = copy.deepcopy(Config.fromfile(str(qkn_run / "config.py")).model)
    plain.pop("dec_qk_norm")
    with pytest.raises(RuntimeError, match="unexpected keys"):
        load_decoder(build_model(plain), files[0])


def test_qk_norm_resume_exact(tmp_path):
    """Mid-epoch resume of a deterministic run with dec_qk_norm (taken from the fresh
    run's config.py): bitwise the uninterrupted run's final state."""
    pretrained = tmp_path / "dinov3_tiny_pretrain.pth"
    torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), pretrained)
    base = tmp_path / "base.py"
    base.write_text(
        RESUME_CONFIG.format(
            base=f"{REPO}/configs/_base_/default_runtime.py", pretrained=pretrained
        )
    )

    def run(name, crash):
        config = tmp_path / f"{name}.py"
        config.write_text(MID_EPOCH.format(base=base, crash=crash))
        return build(config, save_path=str(tmp_path / name), **QKN_OPTIONS)

    reference = run("reference", crash="[]")
    train(reference)
    train(
        run("exp", crash='[dict(type="CrashAtTest", epoch=1, iter=3)]'), crash_ok=True
    )
    resumed = resume(tmp_path / "exp")
    train(resumed)
    assert resumed.model.dec_blocks[0][0].attn.qk_norm
    assert any(p.abs().max() > 0 for p in logit_scales(resumed.model).values())
    assert_state_equal(resumed.model.state_dict(), reference.model.state_dict())
