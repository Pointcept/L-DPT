"""
Structural model options (CPU; CUDA cases where a GPU is visible):

- model.dec_qk_bound: q / k clipped per head and token to the norm sqrt(B sqrt(d))
  before RoPE in every decoder attention, so every logit is <= B; no parameters, same
  state dict (the release models use B = 1000);
- model.dec_conv_mixer "last" / "last_stage" (experimental): those blocks mix tokens
  with LocalMix (3x3 depthwise conv on the token grid, replicate padding, fp32, +
  Linear) instead of self-attention;
- both off: the model and the training step are bitwise those of the reference
  implementation (copies of the methods without the options, below);
- the exp head and the fp32 readout (model.head, model.readout_fp32).
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
from test_l_dpt_golden import BASE, DEPTH_LOSS, correlated_depth, make_data
from test_l_dpt_options import (
    OpTrace,
    island_model,
    parse,
    plain_l_config,
    resume,
    train,
    loss_run,
)
from test_resume import CONFIG as RESUME_CONFIG
from test_resume import MID_EPOCH, assert_state_equal, build

import ldpt.models.l_dpt.modules as modules
from ldpt.engines.default import default_argument_parser
from ldpt.engines.evaluate import load_decoder
from ldpt.utils.sdpa import sdpa_context
from ldpt.hooks.evaluator import read_metrics_history
from ldpt.models.builder import build_model
from ldpt.models.dinov3.layers import RopePositionEmbedding, rope_is_random
from ldpt.models.dinov3.layers.attention import rope_apply
from ldpt.models.l_dpt.l_dpt_v1m1_base import LatentDPT
from ldpt.models.l_dpt.modules import Attention, Block, LocalMix, clip_norm
from ldpt.models.losses.depth import RelativeDepthLoss
from ldpt.utils.config import Config

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
# an idle bound whose radius r has an inexact fp32 reciprocal at head dim 8 (the tiny
# decoders): a factor r / max(|q|, r) would be 1 - 6e-8 below r, not 1.0
IDLE_BOUND = 3e5


# ---------------------------------------------------------------- reference


def ref_qkv(self, x, rope=None):
    """Attention.qkv without the options (reference copy)."""
    B, N, _ = x.shape
    C = self.q_proj.out_features
    q, k, v = [
        proj(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        for proj in (self.q_proj, self.k_proj, self.v_proj)
    ]
    if rope is not None:
        sin, cos = rope
        dtype = q.dtype
        q = rope_apply(q.to(sin.dtype), sin, cos).to(dtype)
        k = rope_apply(k.to(sin.dtype), sin, cos).to(dtype)
    return q, k, v


def ref_block_forward(self, x, rope=None, chunk_tokens=None):
    """Block.forward without the options (reference copy)."""
    x = self.norm1(x)
    x = x + self.ls1(self.attn(x, rope=rope))
    if chunk_tokens is None or torch.is_grad_enabled():
        return x + self.ls2(self.mlp(self.norm2(x)))
    for a in range(0, x.shape[1], chunk_tokens):
        x_chunk = x[:, a : a + chunk_tokens]
        x_chunk += self.ls2(self.mlp(self.norm2(x_chunk)))
    return x


def ref_decode(self, selections, patch_hw):
    """LatentDPT.decode without the options (reference copy, no argument checks)."""
    h, w = patch_hw
    with self._autocast(selections[0].device):
        rope_random = rope_is_random(self.rope)
        rope = None if rope_random else self.rope(H=h, W=w)
        chunk_tokens = self._eval_chunk_tokens()
        query = selections[0].float()
        for s, blocks in enumerate(self.dec_blocks):
            if s > 0:
                query = query + selections[s]
            for i, blk in enumerate(blocks):
                if rope_random:
                    rope = self.rope(H=h, W=w)
                if (s, i) in self.fp32_blocks:
                    with torch.autocast(query.device.type, enabled=False):
                        query = blk(query.float(), rope=rope, chunk_tokens=chunk_tokens)
                else:
                    query = blk(query, rope=rope, chunk_tokens=chunk_tokens)
        b, _, c = query.shape
        query = query.reshape(b, h, w, c).permute(0, 3, 1, 2)
        pred = self.unpatchfy(query)
    return pred.float()


def assert_runs_equal(ref, other, name):
    """Losses, weights, gradients and optimizer state bitwise equal."""
    (ta, ra), (tb, rb) = ref, other
    assert len(ra.losses) == len(rb.losses) == 8
    assert all(torch.equal(a, b) for a, b in zip(ra.losses, rb.losses)), name
    for (n, a), b in zip(ta.model.named_parameters(), tb.model.parameters()):
        assert torch.equal(a, b), (name, n)
        assert (a.grad is None) == (b.grad is None), (name, n)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad), (name, n)
    sa, sb = ta.optimizer.state_dict(), tb.optimizer.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    for i in sa["state"]:
        assert_state_equal(sa["state"][i], sb["state"][i])


@pytest.mark.parametrize("compiled", [False, True])
def test_off_bit_identical_to_reference(tmp_path, monkeypatch, compiled):
    """A tiny loss run (bf16 decoder, clip 1.0): with the options absent,
    explicitly off, or with a bound no norm reaches (IDLE_BOUND), losses, weights,
    gradients and optimizer state are bitwise those of the reference code."""
    extra = dict(compile_blocks=dict(backend="eager")) if compiled else {}
    torch._dynamo.reset()
    with monkeypatch.context() as m:
        m.setattr(Attention, "qkv", ref_qkv)
        m.setattr(Block, "forward", ref_block_forward)
        m.setattr(LatentDPT, "decode", ref_decode)
        ref = loss_run(tmp_path, "ref", **extra)[:2]
    torch._dynamo.reset()
    for name, options in (
        ("default", {}),
        ("off", {"model.dec_qk_bound": None, "model.dec_conv_mixer": None}),
        ("idle_bound", {"model.dec_qk_bound": IDLE_BOUND}),
    ):
        run = loss_run(tmp_path, name, **extra, **options)[:2]
        assert_runs_equal(ref, run, name)
        assert all(blk.mixer == "attn" for s in run[0].model.dec_blocks for blk in s)
    attn = run[0].model.dec_blocks[0][0].attn
    assert attn.qk_bound == IDLE_BOUND
    assert attn.q_proj.out_features // attn.num_heads == 8


# ---------------------------------------------------------------- dec_qk_bound
# Saturated heads (logits spread over +-B) are kept to a few dozen tokens: on x86 CPUs
# their fp32 softmax weights go subnormal and the backward of the attention is 10-17
# times slower at 4096 tokens (the forward and GPUs are not affected).


class SDPACapture:
    """The q / k every scaled_dot_product_attention call gets (the call then runs);
    with a model, only the calls inside its decoder blocks (not the encoder's)."""

    def __init__(self, monkeypatch, model=None):
        self.calls, self.handles = [], []
        self.inside = model is None
        original = F.scaled_dot_product_attention

        def capture(q, k, v, *args, **kwargs):
            if self.inside:
                self.calls.append((q.detach(), k.detach()))
            return original(q, k, v, *args, **kwargs)

        def enter(*_):
            self.inside = True

        def leave(*_):
            self.inside = False

        for blk in [] if model is None else [b for s in model.dec_blocks for b in s]:
            self.handles += [
                blk.register_forward_pre_hook(enter),
                blk.register_forward_hook(leave),
            ]
        monkeypatch.setattr(F, "scaled_dot_product_attention", capture)

    def close(self):
        for handle in self.handles:
            handle.remove()

    def max_logits(self):
        """Per call: max |q.k| / sqrt(d) in fp64 and the q / k dtype."""
        return [
            (
                float((q.double() @ k.double().transpose(-1, -2)).abs().amax())
                / math.sqrt(q.shape[-1]),
                q.dtype,
            )
            for q, k in self.calls
        ]


def adversarial_block(device, bound=200.0, **options):
    """A Block (dim 128, 2 heads of d = 64: r = 40 at B = 200) whose q / k weights,
    biases and norm1 are blown up: raw logits above 1e6."""
    torch.manual_seed(0)
    blk = Block(128, num_heads=2, init_values=0.3, qkv_bias=True, qk_bound=bound)
    for name, value in options.items():
        setattr(blk.attn, name, value)
    with torch.no_grad():
        blk.attn.q_proj.weight.mul_(1000.0)
        blk.attn.k_proj.weight.mul_(-700.0)
        blk.attn.q_proj.bias.fill_(300.0)
        blk.attn.k_proj.bias.fill_(-300.0)
        blk.norm1.weight.mul_(50.0)
        blk.norm1.bias.fill_(20.0)
    return blk.to(device)


@pytest.mark.parametrize("device", DEVICES)
def test_qk_bound_holds_on_adversarial_inputs(device, monkeypatch):
    """The real Block, fp32 and bf16 autocast, with and without RoPE, attn_dtype
    fp32 / bf16 and the attn_fp32 island: the q / k SDPA gets give logits <= B
    (1 + 1e-5) in fp32 and <= B x 1.01 in bf16 (the cast after the clip), while the
    unclipped ones exceed 1e6."""
    B = 200.0
    rope = RopePositionEmbedding(embed_dim=128, num_heads=2, base=100)(H=4, W=6)
    rope = tuple(t.to(device) for t in rope)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 24, 128, generator=g) * 1e3 + 5e2
    x[:, 7] *= 1e3  # one outlier token
    x = x.to(device)
    variants = (
        {},
        dict(attn_dtype=torch.bfloat16),
        dict(force_fp32=True, attn_dtype=torch.bfloat16),
    )
    for options in variants:
        blk = adversarial_block(device, B, **options)
        for rope_ in (None, rope):
            for autocast in (False, True):
                capture = SDPACapture(monkeypatch)
                with (
                    torch.no_grad(),
                    torch.autocast(device, dtype=torch.bfloat16, enabled=autocast),
                ):
                    y = blk(x, rope=rope_)
                    q, k, _ = blk.attn.qkv(blk.norm1(x), rope=rope_, bound=False)
                monkeypatch.undo()
                ((logit, dtype),) = capture.max_logits()
                tol = 1e-5 if dtype == torch.float32 else 1e-2
                case = (options, rope_ is not None, autocast, dtype)
                assert logit <= B * (1 + tol), case
                assert logit > 0.3 * B, case  # the rows are not degenerate
                raw = float((q.double() @ k.double().transpose(-1, -2)).abs().amax())
                assert raw / 8 > 1e6, case
                assert torch.isfinite(y).all(), case
                if "force_fp32" in options:  # the island: fp32 whatever the rest
                    want = torch.float32
                elif autocast or "attn_dtype" in options:
                    want = torch.bfloat16
                else:
                    want = torch.float32
                assert dtype == want, case


@pytest.mark.parametrize("device", DEVICES)
def test_qk_bound_clip_and_gradient(device):
    """r = sqrt(B sqrt(d)); tokens above r get norm r and keep their direction, those
    below stay bitwise; bound=False gives the raw q / k; the loss has no radial
    gradient on the q weights of fully clipped heads (scaling them changes nothing)."""
    assert Attention(1024, num_heads=16, qk_bound=200.0).qk_radius == 40.0
    torch.manual_seed(0)
    attn = Attention(128, num_heads=2, qk_bound=200.0).to(device)
    with torch.no_grad():
        attn.q_proj.weight[:64].mul_(30.0)  # head 0 of q far above r = 40
    x = (torch.randn(2, 24, 128) * 3).to(device)
    with torch.no_grad():
        q, k, _ = attn.qkv(x)
        q_raw, k_raw, _ = attn.qkv(x, bound=False)
    n_raw = q_raw.norm(dim=-1)
    assert (n_raw[:, 0] > 40).all() and (n_raw[:, 1] < 40).all()
    torch.testing.assert_close(q[:, 0].norm(dim=-1), torch.full_like(n_raw[:, 0], 40.0))
    cos = F.cosine_similarity(q[:, 0], q_raw[:, 0], dim=-1)
    torch.testing.assert_close(cos, torch.ones_like(cos))
    assert torch.equal(q[:, 1], q_raw[:, 1]) and torch.equal(k, k_raw)
    plain = Attention(128, num_heads=2).to(device)
    plain.load_state_dict(attn.state_dict())
    assert torch.equal(plain.qkv(x)[0], q_raw)

    rope = RopePositionEmbedding(embed_dim=128, num_heads=2, base=100)(H=4, W=6)
    rope = tuple(t.to(device) for t in rope)
    radial = {}
    for scale in (30.0, 1.0):  # every head clipped / none clipped
        torch.manual_seed(0)
        a = Attention(128, num_heads=2, qk_bound=200.0).to(device)
        with torch.no_grad():
            a.q_proj.weight.mul_(scale)
            a.k_proj.weight.mul_(scale)
        out = a(x, rope=rope)
        (out * torch.randn_like(out)).sum().backward()
        gw, w = a.q_proj.weight.grad, a.q_proj.weight.detach()
        radial[scale] = float((gw * w).sum().abs() / (gw.norm() * w.norm()))
    assert radial[30.0] < 1e-6 and radial[1.0] > 1e-3, radial


def first_clip(t, r):
    """The clip of the first implementation, t r / max(|t|, r): its factor rounds to
    1 - 6e-8 below r for some r."""
    return t * (r / t.norm(dim=-1, keepdim=True).clamp_min(r))


@pytest.mark.parametrize("device", DEVICES)
def test_qk_bound_idle_exact(device):
    """Below the radius q / k are bitwise unchanged for every bound: clip_norm for the
    integer B in 1..4000 (and 3e5, 1e6) at head dims 8 to 64, where the first
    implementation's factor was inexact for about 13 % of B (300 at d = 64 among them);
    at the module level (d = 64, B = 300) qkv with an idle bound gives bitwise the q / k
    (dtype, strides) of bound=False, with and without RoPE and bf16 autocast."""
    g = torch.Generator().manual_seed(0)
    for d in (8, 16, 32, 64):
        u = torch.randn(256, d, generator=g)
        u = (u / u.norm(dim=-1, keepdim=True)).to(device)
        inexact = 0
        for B in list(range(1, 4001)) + [3e5, 1e6]:
            r = math.sqrt(B * math.sqrt(d))
            t = u * (torch.rand(256, 1, generator=g).to(device) * 0.999 * r)
            assert torch.equal(clip_norm(t, r), t), (d, B)
            inexact += not torch.equal(first_clip(t, r), t)
        assert inexact > 300, (d, inexact)
    torch.manual_seed(0)
    attn = Attention(128, num_heads=2, qk_bound=300.0).to(device)  # r = 48.99
    rope = RopePositionEmbedding(embed_dim=128, num_heads=2, base=100)(H=4, W=6)
    rope = tuple(t.to(device) for t in rope)
    x = torch.randn(2, 24, 128).to(device)
    for rope_ in (None, rope):
        for autocast in (False, True):
            with (
                torch.no_grad(),
                torch.autocast(device, dtype=torch.bfloat16, enabled=autocast),
            ):
                got = attn.qkv(x, rope=rope_)
                want = attn.qkv(x, rope=rope_, bound=False)
            assert (got[0].norm(dim=-1) < 0.5 * attn.qk_radius).all()
            for a, b in zip(got, want):
                assert torch.equal(a, b) and a.dtype == b.dtype
                assert a.stride() == b.stride(), (a.stride(), b.stride())


def grads_of(attn, x, rope, autocast):
    """Output, input gradient and weight gradients after (output^2).sum(), with or
    without bf16 autocast on x's device."""
    x = x.clone().requires_grad_(True)
    attn.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=autocast):
        y = attn(x, rope=rope)
    y.float().square().sum().backward()
    return [y, x.grad] + [p.grad for p in attn.parameters()]


@pytest.mark.parametrize("device", DEVICES)
def test_qk_bound_inactive_and_fp32_island(device):
    """A bound no norm reaches: Attention outputs and gradients bitwise those without
    it (fp32, bf16 autocast, attn_dtype bf16, island). The attn_fp32 island with an
    active bound under bf16 autocast: bitwise the bounded module without autocast."""
    torch.manual_seed(0)
    rope = RopePositionEmbedding(embed_dim=16, num_heads=2, base=100)(H=4, W=6)
    rope = tuple(t.to(device) for t in rope)
    x = torch.randn(2, 24, 32).to(device)
    for attn_dtype in ("fp32", "bf16"):
        for force_fp32 in (False, True):
            pair = []
            for bound in (None, IDLE_BOUND):  # no draw from the RNG for the bound
                torch.manual_seed(0)
                a = Attention(
                    32,
                    num_heads=2,
                    downscale=2,
                    attn_dtype=attn_dtype,
                    force_fp32=force_fp32,
                    qk_bound=bound,
                )
                pair.append(a.to(device))
            for autocast in (False, True):
                want = grads_of(pair[0], x, rope, autocast)
                got = grads_of(pair[1], x, rope, autocast)
                assert all(torch.equal(a, b) for a, b in zip(want, got))
    # active bound: r = sqrt(0.5 sqrt(8)) = 1.19 clips most of these q / k
    torch.manual_seed(0)
    attn = Attention(32, num_heads=2, downscale=2, attn_dtype="bf16", qk_bound=0.5)
    attn = attn.to(device)
    attn.force_fp32 = True
    with torch.no_grad():
        q, _, _ = attn.qkv(x, bound=False)
    assert (q.norm(dim=-1) > attn.qk_radius).float().mean() > 0.5
    want = grads_of(attn, x, rope, autocast=False)
    got = grads_of(attn, x, rope, autocast=True)
    assert want[0].dtype == got[0].dtype == torch.float32
    assert all(torch.equal(a, b) for a, b in zip(want, got))


def test_qk_bound_model_fp32_inactive_bit_identical():
    """precision fp32 (the golden path), random weights: dec_qk_bound=IDLE_BOUND gives
    bitwise the eval prediction, loss and gradients of the model without it."""
    data = make_data("depth", (64, 80))
    results = []
    for options in ({}, dict(dec_qk_bound=IDLE_BOUND)):
        model = island_model("fp32", **options)
        with torch.no_grad():
            pred = model.eval()(copy.deepcopy(data))["pred_depth"]
        model.train()
        torch.manual_seed(1)
        loss = model(copy.deepcopy(data))["loss"]
        loss.backward()
        results.append(
            (pred, loss, [p.grad for p in model.parameters() if p.grad is not None])
        )
    (p0, l0, g0), (p1, l1, g1) = results
    assert torch.equal(p0, p1) and torch.equal(l0, l1)
    assert len(g0) == len(g1) and all(torch.equal(a, b) for a, b in zip(g0, g1))


def test_qk_bound_model_with_attn_fp32(monkeypatch):
    """bf16 decoder with dec_qk_bound=2 (active: most q / k clipped) and
    attn_fp32=last_stage: every SDPA call of a training step gets bounded logits
    (fp32 operands in the island blocks 3.x, bf16 elsewhere); every decoder
    parameter gets a finite gradient."""
    model = island_model(dec_qk_bound=2.0, attn_fp32="last_stage").train()
    capture = SDPACapture(monkeypatch, model)
    torch.manual_seed(1)
    loss = model(make_data("depth", (64, 80)))["loss"]
    monkeypatch.undo()
    capture.close()
    loss.backward()
    logits = capture.max_logits()
    assert [dtype for _, dtype in logits] == [torch.bfloat16] * 4 + [torch.float32] * 2
    for logit, dtype in logits:
        assert 1.0 < logit <= 2.0 * (1 + (1e-5 if dtype == torch.float32 else 1e-2))
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n


@pytest.mark.parametrize("device", DEVICES)
def test_qk_bound_sdpa_backends(device):
    """The bound sits before SDPA: every backend the device runs (CPU: flash, math;
    CUDA: also mem-efficient and cuDNN) gives the attention output of the math backend
    on the same bounded bf16 q / k (B = 20 on blown-up weights: every head clipped)."""
    torch.manual_seed(0)
    blk = adversarial_block(device, bound=20.0, attn_dtype=torch.bfloat16)
    rope = RopePositionEmbedding(embed_dim=128, num_heads=2, base=100)(H=4, W=6)
    rope = tuple(t.to(device) for t in rope)
    x = (torch.randn(2, 24, 128) * 1e3).to(device)
    ran = []
    for autocast in (False, True):
        outs = {}
        for backend in ("math", "flash", "efficient", "cudnn"):
            try:
                with (
                    sdpa_context([backend]),
                    torch.no_grad(),
                    torch.autocast(device, dtype=torch.bfloat16, enabled=autocast),
                ):
                    outs[backend] = blk.attn(blk.norm1(x), rope=rope).float()
            except RuntimeError as e:  # no kernel for this backend on the device
                assert "backend" in str(e).lower() or "kernel" in str(e).lower(), e
        for backend, y in outs.items():
            error = (y - outs["math"]).abs().max() / outs["math"].abs().max()
            assert error < 1e-2, (backend, autocast, float(error))
        ran.append(sorted(outs))
    assert all({"math", "flash"} <= set(r) for r in ran), ran
    if device == "cuda":
        assert all(len(r) >= 3 for r in ran), ran


# ---------------------------------------------------------------- dec_conv_mixer


def test_conv_mixer_init_keys_and_rng():
    """Each scope: the conv blocks' keys (local.dw.*, local.out_proj.*; no attn.*), every
    other parameter equal to the baseline's for the same seed, the global RNG state
    after __init__ unchanged, distinct out_proj inits, the block at init norm1(x) +
    O(LayerScale), the module repr."""
    for scope, want in (("last", {(3, 1)}), ("last_stage", {(3, 0), (3, 1)})):
        states = []
        for options in ({}, dict(dec_conv_mixer=scope)):
            torch.manual_seed(0)
            cfg = dict(type="L-DPT-v1m1", encoder=dict(type="dinov3_tiny_test"))
            cfg.update(encoder_inter_layers=[0, 1, 2, 3], dec_dim=64, dec_num_heads=4)
            cfg.update(dec_num_layers=(1, 1, 2, 2), dec_attn_downscale=2, **options)
            model = build_model(cfg)
            states.append((model, model.state_dict(), torch.get_rng_state()))
        (_, sb, rb), (model, sm, rm) = states
        assert torch.equal(rb, rm)
        assert model.conv_blocks == want
        conv = [f"dec_blocks.{s}.{i}." for s, i in sorted(want)]
        assert sorted(set(sb) - set(sm)) == sorted(
            p + f"attn.{n}"
            for p in conv
            for n in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "out_proj.weight",
                "out_proj.bias",
            )
        )
        assert sorted(set(sm) - set(sb)) == sorted(
            p + f"local.{n}"
            for p in conv
            for n in ("dw.weight", "dw.bias", "out_proj.weight", "out_proj.bias")
        )
        for key in set(sb) & set(sm):
            if not key.startswith("encoder."):  # partly torch.empty in the tiny ViT
                assert torch.equal(sb[key], sm[key]), key
        for p in conv:
            dw = sm[p + "local.dw.weight"]
            assert dw.shape == (64, 1, 3, 3) and dw[:, 0, 1, 1].eq(1).all()
            assert dw.sum() == 64 and not sm[p + "local.dw.bias"].any()
        if len(conv) == 2:
            assert not torch.equal(*(sm[p + "local.out_proj.weight"] for p in conv))
        blk = model.dec_blocks[3][1]
        assert blk.mixer == "conv" and not hasattr(blk, "attn")
        x = torch.randn(2, 20, 64) * 3
        with torch.no_grad():
            y = blk(x, hw=(4, 5))
        assert (y - blk.norm1(x)).abs().max() < 1e-3
        assert "LocalMix" in repr(blk)


def test_conv_mixer_parameter_counts_l(tmp_path):
    """The L decoder (270,995,712 parameters): CONV1 -3,135,488 (one block),
    CONV4 -12,541,952 (four); the new parameters are in the default optimizer group
    (no "encoder" in their names)."""
    counts = {}
    for scope in (None, "last", "last_stage"):
        options = [f"model.dec_conv_mixer={scope}"] if scope else []
        cfg = parse(
            plain_l_config(tmp_path), [f"save_path={tmp_path / 'exp'}"] + options
        )
        with torch.device("meta"):
            model = build_model(cfg.model)
        names = [n for n, _ in model.named_parameters() if not n.startswith("encoder.")]
        counts[scope] = sum(
            p.numel()
            for n, p in model.named_parameters()
            if not n.startswith("encoder.")
        )
        local = [n for n in names if ".local." in n]
        assert len(local) == 4 * len(model.conv_blocks)
        assert not any("encoder" in n for n in local)
    assert counts[None] == 270_995_712
    assert counts["last"] - counts[None] == -3_135_488
    assert counts["last_stage"] - counts[None] == -12_541_952
    assert sum(p.numel() for p in LocalMix(1024).parameters()) == 1_059_840


def test_local_mix():
    """Identity kernel at init; replicate padding exactly (a token-constant stream
    stays constant; the convolution equals the 9-tap sum over the clamped grid); the
    interior of a crop with a 1-token halo equals the full grid; under bf16 autocast
    the depthwise convolution's input, output and weight gradient are fp32 and
    out_proj runs in bf16; the gradients have the parameters' strides; the grid is
    checked."""
    torch.manual_seed(0)
    h, w, c = 5, 7, 16
    lm = LocalMix(c)
    x = torch.randn(2, h * w, c)
    with torch.no_grad():
        torch.testing.assert_close(lm(x, (h, w)), lm.out_proj(x), rtol=0, atol=1e-6)
        lm.dw.weight.normal_(0, 0.3)
        lm.dw.bias.normal_(0, 0.3)
        const = torch.randn(1, 1, c).expand(2, h * w, c).contiguous()
        yc = lm(const, (h, w))
        assert torch.equal(yc, yc[:, :1].expand_as(yc))
        # the 9 taps over the grid with clamped (= replicated) indices
        grid = x.reshape(2, h, w, c)
        rows, cols = torch.arange(h), torch.arange(w)
        want = lm.dw.bias.expand(2, h, w, c).clone()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                shifted = grid[:, (rows + di).clamp(0, h - 1)][
                    :, :, (cols + dj).clamp(0, w - 1)
                ]
                want += shifted * lm.dw.weight[:, 0, di + 1, dj + 1]
        want = lm.out_proj(want.reshape(2, h * w, c))
        torch.testing.assert_close(lm(x, (h, w)), want, rtol=1e-5, atol=1e-5)
        # zero padding: the same interior, another border
        nchw = grid.permute(0, 3, 1, 2)
        rep = lm.dw(nchw)
        zero = F.conv2d(nchw, lm.dw.weight, lm.dw.bias, padding=1, groups=c)
        inner = (slice(None), slice(None), slice(1, -1), slice(1, -1))
        torch.testing.assert_close(rep[inner], zero[inner], rtol=0, atol=1e-6)
        assert (rep - zero).abs().max() > 1e-2
        g = torch.randn(1, 20 * 24, c)
        full = lm(g, (20, 24)).reshape(1, 20, 24, c)
        crop = g.reshape(1, 20, 24, c)[:, 4:14, 6:18].reshape(1, -1, c)
        part = lm(crop, (10, 12)).reshape(1, 10, 12, c)
        torch.testing.assert_close(
            part[:, 1:-1, 1:-1], full[:, 5:13, 7:17], rtol=0, atol=1e-6
        )
        for bad in (None, (h, w + 1)):
            with pytest.raises(ValueError, match="grid"):
                lm(x, bad)

    seen = {}
    lm.dw.register_forward_hook(lambda m, i, o: seen.update(i=i[0].dtype, o=o.dtype))
    xb = (torch.randn(2, h * w, c) * 3).requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        yb = lm(xb.to(torch.bfloat16), (h, w))
    yb.float().square().sum().backward()
    assert seen == dict(i=torch.float32, o=torch.float32) and yb.dtype == torch.bfloat16
    assert lm.dw.weight.grad.dtype == torch.float32
    assert torch.isfinite(lm.dw.weight.grad).all() and lm.dw.weight.grad.abs().sum() > 0
    # the gradients have the parameters' strides (DDP's bucket views), also from an
    # fp32 input, where the [B, C, h, w] view of the tokens is channels-last
    for p in lm.parameters():
        assert p.grad.stride() == p.stride(), p.shape
    lm.zero_grad(set_to_none=True)
    lm(xb, (h, w)).square().sum().backward()
    assert all(p.grad.stride() == p.stride() for p in lm.parameters())


@pytest.mark.parametrize(
    "options, conv, attn32, block32",
    [
        (dict(dec_conv_mixer="last"), {"3.1"}, set(), set()),
        (dict(dec_conv_mixer="last_stage"), {"3.0", "3.1"}, set(), set()),
        # attn_fp32 does not apply to a conv block, block_fp32 does
        (
            dict(dec_conv_mixer="last", attn_fp32="last_stage", block_fp32="last"),
            {"3.1"},
            {"3.0"},
            {"3.1"},
        ),
    ],
)
def test_conv_blocks_dispatch(options, conv, attn32, block32, monkeypatch):
    """bf16 training step: a conv block calls no softmax, no SDPA and no RoPE; its
    depthwise convolution sees fp32 only; with block_fp32 it runs outside autocast with
    no bf16 op; the attention blocks are unchanged (SDPA once, RoPE on q and k); every
    decoder parameter gets a finite gradient."""
    model = island_model(**options).train()
    trace = OpTrace(model)
    rope_calls = []

    def counted(*args, **kwargs):
        rope_calls.append(trace.block)
        return rope_apply(*args, **kwargs)

    monkeypatch.setattr(modules, "rope_apply", counted)
    torch.manual_seed(1)
    with trace:
        loss = model(make_data("depth", (64, 80)))["loss"]
    for handle in trace.handles:
        handle.remove()
    loss.backward()
    assert sorted(trace.autocast) == ["0.0", "1.0", "2.0", "2.1", "3.0", "3.1"]
    for blk in trace.autocast:
        ops = [o for o in trace.ops if o[0] == blk]
        sdpa = [o for o in ops if "scaled_dot_product" in o[1]]
        convs = [o for o in ops if "convolution" in o[1]]
        if blk in conv:
            assert not sdpa and not any("softmax" in o[1] for o in ops), blk
            assert rope_calls.count(blk) == 0, blk
            assert len(convs) == 1 and convs[0][2] | convs[0][3] == {torch.float32}
        else:
            assert len(sdpa) == 1 and not convs, blk
            assert rope_calls.count(blk) == 2, blk
            want = torch.float32 if blk in attn32 else torch.bfloat16
            assert sdpa[0][2] == {want}, blk
        assert trace.autocast[blk] == (blk not in block32), blk
        if blk in block32:
            assert all(torch.bfloat16 not in o[2] | o[3] for o in ops), blk
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n


@pytest.mark.parametrize(
    "options",
    [
        dict(dec_qk_bound=2.0, attn_fp32="last_stage"),
        dict(dec_conv_mixer="last_stage"),
        dict(dec_conv_mixer="last", dec_qk_bound=2.0, block_fp32="last"),
    ],
    ids=["qk_bound+attn_fp32", "conv_last_stage", "conv+qk_bound+block_fp32"],
)
def test_structural_compiled_blocks(options):
    """compile_blocks (aot_eager: AOT autograd, as inductor): conv blocks stay eager,
    attention blocks are compiled; the training loss and every gradient are bitwise
    those of the eager model."""
    torch._dynamo.reset()
    eager = island_model(**options).train()
    compiled = copy.deepcopy(eager)
    compiled.compile_blocks(backend="aot_eager")
    for stage in compiled.dec_blocks:
        for blk in stage:
            assert ("forward" in vars(blk)) == (blk.mixer == "attn")
    data = make_data("depth", (64, 80))
    out = []
    for m in (eager, compiled):
        torch.manual_seed(1)
        loss = m(copy.deepcopy(data))["loss"]
        loss.backward()
        out.append((loss, [p.grad for p in m.parameters() if p.requires_grad]))
    assert torch.equal(out[1][0], out[0][0])
    assert all(torch.equal(a, b) for a, b in zip(out[0][1], out[1][1]))


@pytest.mark.parametrize(
    "options",
    [
        dict(dec_qk_bound=2.0),
        dict(dec_conv_mixer="last_stage"),
        dict(dec_conv_mixer="last", dec_qk_bound=2.0, block_fp32="last"),
    ],
    ids=["qk_bound", "conv_last_stage", "conv+qk_bound"],
)
def test_structural_eval_grids_and_chunks(options):
    """Evaluation at the training grid and at other grids (non-square, other token
    counts): the conv blocks get the image's token grid, and chunk_tokens (chunked
    encoder blocks and decoder FFN) gives the unchunked prediction."""
    model = island_model("fp32", **options).eval()
    grids = []
    for blk in (b for s in model.dec_blocks for b in s if b.mixer == "conv"):
        blk.register_forward_pre_hook(
            lambda m, a, kw: grids.append(kw.get("hw")), with_kwargs=True
        )
    for hw in ((64, 80), (48, 112), (40, 56)):  # 4 x 5, 3 x 7, 3 x 4 (padded) tokens
        image = make_data(None, hw)["image"]
        with torch.no_grad():
            model.chunk_tokens = None
            a = model.infer(image)
            model.chunk_tokens = 7
            b = model.infer(image)
        model.chunk_tokens = None
        assert a.shape == (2, 1, *hw) and torch.isfinite(a).all()
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)
        tokens = (math.ceil(hw[0] / 16), math.ceil(hw[1] / 16))
        assert all(g == tokens for g in grids[-2 * len(model.conv_blocks) :])
    assert len(grids) == 6 * len(model.conv_blocks)


def test_structural_state_dict_round_trip(tmp_path):
    """dec_qk_bound keeps the state dict: a baseline checkpoint loads strictly into a
    bound model and back. A conv model's weights load strictly into a model with the
    same option (same eval output), and fail the strict load and the evaluator's
    key check (load_decoder) without it."""
    base = island_model("fp32")
    bound = island_model("fp32", dec_qk_bound=2.0)
    bound.load_state_dict(base.state_dict(), strict=True)
    base.load_state_dict(bound.state_dict(), strict=True)

    image = make_data(None, (48, 112))["image"]
    options = dict(dec_conv_mixer="last", dec_qk_bound=2.0)
    conv = island_model("fp32", **options)
    path = tmp_path / "epoch_1.pth"
    torch.save(dict(epoch=1, state_dict=conv.state_dict()), path)
    # island_model's architecture, default weights before the load
    config = dict(BASE, criterion=DEPTH_LOSS, dec_num_layers=(1, 1, 2, 2))
    again = build_model(dict(config, **options))
    assert load_decoder(again, path) == 1
    with torch.no_grad():
        assert torch.equal(conv.eval().infer(image), again.eval().infer(image))
    with pytest.raises(RuntimeError, match="local.dw.weight"):
        base.load_state_dict(conv.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="missing decoder keys"):
        load_decoder(build_model(config), path)


# ---------------------------------------------------------------- trainer, evaluate, resume

# active on the tiny decoder (head dim 8): r = sqrt(0.5 sqrt(8)) = 1.19
RUN_OPTIONS = {"model.dec_qk_bound": 0.5, "model.dec_conv_mixer": "last"}


@pytest.fixture(scope="module")
def structural_run(tmp_path_factory):
    """test_evaluate's tiny bf16 run (2 epochs, evaluation, epoch weights) with both
    options: (save_path, pretrained)."""
    root = tmp_path_factory.mktemp("structural")
    pretrained = root / "dinov3_tiny_pretrain.pth"
    torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), pretrained)
    config = root / "config_src.py"
    config.write_text(
        EVAL_CONFIG.format(
            base=f"{REPO}/configs/_base_/default_runtime.py", pretrained=pretrained
        )
    )
    save_path = root / "exp"
    train(build(config, save_path=str(save_path), **RUN_OPTIONS))
    return save_path, pretrained


def test_structural_evaluate_script(structural_run, tmp_path):
    """scripts/evaluate.py on the run's config.py (it carries the options) reproduces
    the metrics the run logged; the model without the options refuses the files."""
    save_path, _ = structural_run
    history = read_metrics_history(save_path / "metrics.jsonl")
    files = [save_path / "model" / f"epoch_{e}.pth" for e in (1, 2)]
    output = tmp_path / "eval.json"
    load_script().main(
        ["--config-file", str(save_path / "config.py"), "--checkpoints"]
        + [str(p) for p in files]
        + ["--output", str(output)]
    )
    results = json.loads(output.read_text())["results"]
    for record, epoch in zip(results, (1, 2)):
        assert record["epoch"] == epoch
        assert record["score"] == pytest.approx(history[epoch]["score"], rel=1e-6)
        for group, metrics in history[epoch]["groups"].items():
            assert record["groups"][group] == pytest.approx(metrics, rel=1e-6)
    plain = copy.deepcopy(Config.fromfile(str(save_path / "config.py")).model)
    plain.pop("dec_conv_mixer")
    with pytest.raises(RuntimeError, match="missing decoder keys"):
        load_decoder(build_model(plain), files[0])
    # the with / without diagnostic: --options model.dec_qk_bound=None on the run's
    # config.py (as scripts/evaluate.py merges it) builds the model without the bound,
    # which loads the same files
    args = load_script().parse_args(
        ["--config-file", str(save_path / "config.py"), "--checkpoints", "x.pth"]
        + ["--options", "model.dec_qk_bound=None"]
    )
    cfg = Config.fromfile(args.config_file)
    cfg.merge_from_dict(args.options)
    unbound = build_model(cfg.model)
    attn_blocks = [b for s in unbound.dec_blocks for b in s if b.mixer == "attn"]
    assert unbound.conv_blocks == {(3, 0)} and len(attn_blocks) == 3
    assert all(b.attn.qk_bound is None for b in attn_blocks)
    assert load_decoder(unbound, files[0]) == 1


def test_structural_resume_exact(tmp_path):
    """Mid-epoch resume of a deterministic run with both options (taken from the
    fresh run's config.py): bitwise the uninterrupted run's final state."""
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
        return build(config, save_path=str(tmp_path / name), **RUN_OPTIONS)

    reference = run("reference", crash="[]")
    train(reference)
    train(
        run("exp", crash='[dict(type="CrashAtTest", epoch=1, iter=3)]'), crash_ok=True
    )
    resumed = resume(tmp_path / "exp")
    train(resumed)
    assert resumed.model.conv_blocks == {(3, 0)}
    assert resumed.model.dec_blocks[0][0].attn.qk_bound == 0.5
    assert_state_equal(resumed.model.state_dict(), reference.model.state_dict())


# ---------------------------------------------------------------- option strings


def test_structural_option_strings(tmp_path):
    """The options as train.sh -o / --options pass them, on a plain L config: values,
    the blocks they reach (r = 40 at B = 200, d = 64), dec_qk_bound=None / null as
    off, invalid values. (The L decoder on the tiny encoder: the ViT-7B adds nothing
    here but build time.)"""
    save = f"save_path={tmp_path / 'exp'}"
    config_l = plain_l_config(tmp_path)
    tiny = dict(
        encoder=dict(type="dinov3_tiny_test"), encoder_inter_layers=[0, 1, 2, 3]
    )
    for options, bound, conv in (
        (["model.dec_qk_bound=200.0"], 200.0, set()),
        (["model.dec_conv_mixer=last", "model.dec_qk_bound=1000"], 1000.0, {(3, 3)}),
        (
            ["model.dec_conv_mixer=last_stage", "model.dec_qk_bound=200.0"],
            200.0,
            {(3, 0), (3, 1), (3, 2), (3, 3)},
        ),
    ):
        cfg = parse(config_l, [save] + options)
        assert cfg.model.get("dec_qk_bound") == bound
        with torch.device("meta"):
            model = build_model(dict(cfg.model, **tiny))
        assert model.conv_blocks == conv and model.dec_num_layers == (4, 4, 4, 4)
        for s, stage in enumerate(model.dec_blocks):
            for i, blk in enumerate(stage):
                assert blk.mixer == ("conv" if (s, i) in conv else "attn")
                if blk.mixer == "attn":
                    assert blk.attn.qk_bound == bound
                    if bound == 200.0:
                        assert blk.attn.qk_radius == 40.0
    # the command-line strings through train.py's argument parser
    model_cfg = dict(cfg.model, **tiny, dec_qk_bound=200.0, dec_conv_mixer=None)

    def build_with(option):
        args = default_argument_parser().parse_args(
            ["--config-file", str(config_l), "--options", option]
        )
        ((key, value),) = args.options.items()
        with torch.device("meta"):
            return build_model(dict(model_cfg, **{key.split(".")[1]: value}))

    # "None" / "null" turn off a bound the config carries (a bound run's config.py)
    for off in ("model.dec_qk_bound=None", "model.dec_qk_bound=null"):
        model = build_with(off)
        assert all(b.attn.qk_bound is None for s in model.dec_blocks for b in s)
    for bad in (
        "model.dec_qk_bound=0",
        "model.dec_qk_bound=-1.0",
        "model.dec_qk_bound=True",
        "model.dec_qk_bound=inf",
        "model.dec_qk_bound=nan",
        "model.dec_qk_bound=abc",
        "model.dec_conv_mixer=all",
        "model.dec_conv_mixer=None",
    ):
        with pytest.raises(ValueError, match="qk_bound|dec_conv_mixer"):
            build_with(bad)


def test_attn_fp32_on_conv_blocks_only_raises():
    """An attn_fp32 scope that holds only dec_conv_mixer blocks would be a silent
    no-op: ValueError. A scope with attention blocks in it, and block_fp32 on a conv
    block, build."""
    for conv, attn_fp32 in (("last", "last"), ("last_stage", "last_stage")):
        with pytest.raises(ValueError, match="no attention"):
            island_model(dec_conv_mixer=conv, attn_fp32=attn_fp32)
    model = island_model(dec_conv_mixer="last", attn_fp32="last_stage")
    assert model.dec_blocks[3][0].attn.force_fp32
    island_model(dec_conv_mixer="last_stage", attn_fp32="all", block_fp32="last")


# ------------------------------------------------- head and readout


def head_model(head="sigmoid", readout_fp32=False, precision="bf16"):
    """island_model with the head options and a criterion naming the same head."""
    criterion = dict(DEPTH_LOSS, head=head)
    return island_model(
        precision, head=head, readout_fp32=readout_fp32, criterion=criterion
    )


def fitted_data(model, hw=(64, 80)):
    """make_data's depth batch with targets the loss's fits accept (an affine map of
    the model's eval prediction, test_l_dpt_golden.correlated_depth)."""
    return correlated_depth(model, make_data("depth", hw), seed=5)


class ReadoutCapture:
    """Input and output of the readout (model.unpatchfy) of every forward."""

    def __init__(self, model):
        self.inputs, self.outputs = [], []
        self.handle = model.unpatchfy.register_forward_hook(self.hook)

    def hook(self, module, args, output):
        self.inputs.append(args[0].detach().clone())
        self.outputs.append(output.detach().clone())


def state_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_head_exp_is_exp_of_clamped_logit(monkeypatch):
    """head="exp": infer returns exp(clamp(u, -30, 30)) of the readout logit u (the
    decode output cropped to the input); "sigmoid" returns sigmoid(u) of the same
    logit; same parameters and state dict."""
    sig, exp = head_model("sigmoid"), head_model("exp")
    assert state_equal(sig, exp) and exp.head == "exp"
    image = make_data("depth", (40, 56))["image"]  # padded to 48 x 64
    logits = []
    decode = LatentDPT.decode

    def recording(self, *args):
        out = decode(self, *args)
        logits.append(out[:, :, :40, :56])
        return out

    monkeypatch.setattr(LatentDPT, "decode", recording)
    with torch.no_grad():
        p_sig, p_exp = sig.eval().infer(image), exp.eval().infer(image)
    assert torch.equal(logits[0], logits[1])
    assert torch.equal(p_sig, torch.sigmoid(logits[0]))
    assert torch.equal(p_exp, torch.exp(logits[1].clamp(-30.0, 30.0)))
    assert p_exp.shape == (2, 1, 40, 56) and (p_exp > 0).all()

    # the clamp: a logit beyond +-30 gives exp(+-30)
    u = torch.linspace(-50.0, 50.0, 2 * 48 * 64).reshape(2, 1, 48, 64)
    monkeypatch.setattr(LatentDPT, "decode", lambda self, *a: u.clone())
    with torch.no_grad():
        p = exp.infer(image)
    assert torch.equal(p, torch.exp(u[:, :, :40, :56].clamp(-30.0, 30.0)))
    assert p.min() == torch.exp(torch.tensor(-30.0)) and p.max() == torch.exp(
        torch.tensor(30.0)
    )


def test_head_exp_trains():
    """A training step with head="exp": finite loss, the readout gets a gradient."""
    model = head_model("exp", readout_fp32=True)
    data = fitted_data(model)
    model.train()
    torch.manual_seed(1)
    loss = model(data)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    grad = model.unpatchfy[0].proj.weight.grad
    assert grad is not None and grad.dtype == torch.float32 and grad.abs().sum() > 0


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_head_options_default_bit_identical(precision):
    """head="sigmoid", readout_fp32=False passed explicitly: bitwise the model without
    them (eval prediction, loss, gradients), same state dict."""
    data = fitted_data(island_model(precision))
    results = []
    for options in ({}, dict(head="sigmoid", readout_fp32=False)):
        model = island_model(precision, **options)
        with torch.no_grad():
            pred = model.eval()(copy.deepcopy(data))["pred_depth"]
        model.train()
        torch.manual_seed(1)
        loss = model(copy.deepcopy(data))["loss"]
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        results.append((model, pred, loss, grads))
    (m0, p0, l0, g0), (m1, p1, l1, g1) = results
    assert state_equal(m0, m1) and l0 > 0
    assert torch.equal(p0, p1) and torch.equal(l0, l1)
    assert len(g0) == len(g1) and all(torch.equal(a, b) for a, b in zip(g0, g1))


def test_readout_fp32_under_bf16_autocast():
    """precision="bf16" (CPU bf16 autocast): with readout_fp32 the readout sees the
    fp32 copy of the decoder stream and returns its fp32 convolution + pixel shuffle;
    without it the readout output is bf16 and differs. Same state dict."""
    image = make_data("depth", (64, 80))["image"]
    runs = {}
    for readout_fp32 in (False, True):
        model = head_model(readout_fp32=readout_fp32).eval()
        capture = ReadoutCapture(model)
        with torch.no_grad():
            pred = model.infer(image)
        capture.handle.remove()
        runs[readout_fp32] = (model, capture.inputs[0], capture.outputs[0], pred)
    (m_bf, x_bf, y_bf, p_bf), (m_32, x_32, y_32, p_32) = runs[False], runs[True]
    assert state_equal(m_bf, m_32)
    assert x_32.dtype == torch.float32 and torch.equal(x_32, x_bf.float())
    assert y_bf.dtype == torch.bfloat16 and y_32.dtype == torch.float32
    proj = m_32.unpatchfy[0].proj
    want = F.pixel_shuffle(
        F.conv2d(x_32, proj.weight, proj.bias, padding=proj.padding), 16
    )
    assert torch.equal(y_32, want)
    assert not torch.equal(y_32, y_bf.float())
    assert not torch.equal(p_32, p_bf)
    # the bf16 path rounds the logit; the fp32 path is within bf16 rounding of it
    assert torch.allclose(y_32, y_bf.float(), rtol=2e-2, atol=2e-2)


def test_readout_fp32_without_autocast_identical():
    """precision="fp32" (no autocast): readout_fp32=True is the same math, bitwise
    (eval prediction, loss, gradients)."""
    data = fitted_data(head_model(precision="fp32"))
    results = []
    for readout_fp32 in (False, True):
        model = head_model(readout_fp32=readout_fp32, precision="fp32")
        with torch.no_grad():
            pred = model.eval()(copy.deepcopy(data))["pred_depth"]
        model.train()
        torch.manual_seed(1)
        loss = model(copy.deepcopy(data))["loss"]
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        results.append((pred, loss, grads))
    (p0, l0, g0), (p1, l1, g1) = results
    assert l0 > 0 and torch.equal(p0, p1) and torch.equal(l0, l1)
    assert len(g0) == len(g1) and all(torch.equal(a, b) for a, b in zip(g0, g1))


def test_head_options_validated():
    """An unknown head raises; head="exp" on semseg raises; a criterion whose head
    differs from the model's raises (the default criterion names "sigmoid", so an
    exp model needs the criterion's head="exp")."""
    with pytest.raises(ValueError, match="head"):
        island_model(head="softplus")
    with pytest.raises(ValueError, match="relative_depth"):
        build_model(dict(BASE, task="semseg", out_dim=8, num_classes=5, head="exp"))
    for model_head, criterion_head in (("exp", "sigmoid"), ("sigmoid", "exp")):
        criterion = dict(DEPTH_LOSS, head=criterion_head)
        with pytest.raises(ValueError, match="differs"):
            island_model(head=model_head, criterion=criterion)
    assert RelativeDepthLoss().head == "sigmoid"
    with pytest.raises(ValueError, match="differs"):
        island_model(head="exp", criterion=dict(type="relative_depth_loss"))
    assert island_model(criterion=DEPTH_LOSS).head == "sigmoid"
    exp = island_model(head="exp", criterion=dict(DEPTH_LOSS, head="exp"))
    assert exp.head == "exp"
