"""
Vendored DINOv3 changes (CPU):
- get_intermediate_layers: RoPE computed once when it is deterministic, tapped layers
  normalised when collected (vs. the original per-block RoPE + norm afterwards);
  iter_intermediate_layers yields the same tensors one by one (bit-exact);
- token-chunked inference blocks (forward_chunked, chunk_tokens): close, not bit-exact;
- frozen-encoder weight storage in bf16 vs fp32 weights under bf16 autocast (bit-exact).
"""

import copy

import pytest
import torch

from helpers import init_encoder_
from test_l_dpt_golden import BASE, DEPTH_LOSS, make_data, randomize_

from ldpt.models.builder import build_model
from ldpt.models.l_dpt.l_dpt_v1m1_base import cast_linear_conv_

ENCODERS = {
    "tiny7b": (dict(type="dinov3_tiny_test"), [0, 1, 2, 3]),
    "vitb": (dict(type="dinov3_vitb"), [2, 5, 8, 11]),
}


def original_intermediate_layers(encoder, x, n):
    """Upstream DINOv3: RoPE per block, keep pre-norm outputs, norm at the end."""
    x, (H, W) = encoder.prepare_tokens_with_masks(x)
    outputs = []
    for i, blk in enumerate(encoder.blocks):
        rope = encoder.rope_embed(H=H, W=W) if encoder.rope_embed is not None else None
        x = blk(x, rope)
        if i in n:
            outputs.append(x)
    normed = []
    for out in outputs:
        if encoder.untie_cls_and_patch_norms:
            cls_reg = encoder.cls_norm(out[:, : encoder.n_storage_tokens + 1])
            patch = encoder.norm(out[:, encoder.n_storage_tokens + 1 :])
            normed.append(torch.cat((cls_reg, patch), dim=1))
        else:
            normed.append(encoder.norm(out))
    return tuple(out[:, encoder.n_storage_tokens + 1 :] for out in normed)


def build_encoder(name, seed=0):
    cfg, layers = ENCODERS[name]
    return init_encoder_(build_model(cfg), seed), layers


def run(encoder, x, layers, autocast, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        return encoder.get_intermediate_layers(x, n=layers, return_class_token=False)


@pytest.mark.parametrize("name", list(ENCODERS))
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_intermediate_layers_match_original(name, training, autocast):
    encoder, layers = build_encoder(name)
    encoder.train(training)
    x = torch.randn(2, 3, 64, 96, generator=torch.Generator().manual_seed(0))
    out = run(encoder, x, layers, autocast, seed=7)
    torch.manual_seed(7)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        ref = original_intermediate_layers(encoder, x, layers)
    assert len(out) == len(ref) == len(layers)
    for a, b in zip(out, ref):
        assert torch.isfinite(a).all()
        assert torch.equal(a, b)


def test_untied_norms_match_original():
    from ldpt.models.dinov3.model import DinoVisionTransformer

    encoder = DinoVisionTransformer(
        embed_dim=64, depth=3, num_heads=4, n_storage_tokens=2,
        untie_cls_and_patch_norms=True, pos_embed_rope_dtype="fp32",
    )  # fmt: skip
    init_encoder_(encoder).eval()
    x = torch.randn(1, 3, 48, 64)
    out = run(encoder, x, [0, 2], autocast=False)
    with torch.no_grad():
        ref = original_intermediate_layers(encoder, x, [0, 2])
    for a, b in zip(out, ref):
        assert torch.equal(a, b)


@pytest.mark.parametrize("name", list(ENCODERS))
@pytest.mark.parametrize("training", [False, True])
def test_iter_intermediate_layers_match(name, training):
    encoder, layers = build_encoder(name)
    encoder.train(training)
    x = torch.randn(2, 3, 64, 96, generator=torch.Generator().manual_seed(0))
    ref = run(encoder, x, layers, autocast=True, seed=7)
    torch.manual_seed(7)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = list(encoder.iter_intermediate_layers(x, n=layers))
    assert len(out) == len(ref)
    for a, b in zip(out, ref):
        assert torch.equal(a, b)


@pytest.mark.parametrize("name", list(ENCODERS))
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("chunk", [5, 24, 1000])  # chunks of the patch tokens
@pytest.mark.parametrize("norm", [True, False])  # False: blocks not run in place
def test_chunked_blocks_match(name, autocast, chunk, norm):
    encoder, layers = build_encoder(name)
    encoder.eval()
    x = torch.randn(2, 3, 64, 96, generator=torch.Generator().manual_seed(0))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        ref = encoder.get_intermediate_layers(x, n=layers, norm=norm)
        out = encoder.get_intermediate_layers(
            x, n=layers, norm=norm, chunk_tokens=chunk
        )
    for a, b in zip(out, ref):
        assert a.dtype == b.dtype
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_chunked_block_without_rope_and_with_grad():
    encoder, _ = build_encoder("tiny7b")
    encoder.eval()
    blk = encoder.blocks[0]
    x = torch.randn(2, 30, encoder.embed_dim)
    with torch.no_grad():
        torch.testing.assert_close(blk.forward_chunked(x, None, 7), blk(x))
    with pytest.raises(RuntimeError, match="inference only"):
        blk.forward_chunked(x, None, 7)
    # with autograd the encoder ignores chunk_tokens (full-size path, bit-exact)
    x = torch.randn(1, 3, 32, 48)
    a = encoder.get_intermediate_layers(x, n=[1], chunk_tokens=4)[0]
    b = encoder.get_intermediate_layers(x, n=[1])[0]
    assert a.requires_grad and torch.equal(a, b)


def test_no_global_dynamo_override():
    import torch._dynamo

    import ldpt.models.dinov3.layers.block  # noqa: F401

    assert torch._dynamo.config.automatic_dynamic_shapes is True


# ---- bf16 storage of the frozen encoder weights ----


@pytest.mark.parametrize("name", list(ENCODERS))
@pytest.mark.parametrize("training", [False, True])
def test_bf16_weights_bit_exact(name, training):
    fp32, layers = build_encoder(name)
    for p in fp32.parameters():
        p.requires_grad = False
    bf16 = cast_linear_conv_(copy.deepcopy(fp32), torch.bfloat16)
    fp32.train(training), bf16.train(training)
    x = torch.randn(2, 3, 64, 96, generator=torch.Generator().manual_seed(1))
    out_fp32 = run(fp32, x, layers, autocast=True, seed=3)
    out_bf16 = run(bf16, x, layers, autocast=True, seed=3)
    for a, b in zip(out_fp32, out_bf16):
        assert a.dtype == b.dtype == torch.float32
        assert torch.equal(a, b)


def test_bf16_weights_cast_only_linear_and_conv():
    encoder, _ = build_encoder("vitb")
    n_bytes = sum(p.nbytes for p in encoder.parameters())
    cast_linear_conv_(encoder, torch.bfloat16)
    for name, p in encoder.named_parameters():
        is_gemm = (
            name.startswith("patch_embed.proj")
            or ".attn.qkv" in name
            or ".attn.proj" in name
            or ".mlp.fc" in name
        )
        assert p.dtype == (torch.bfloat16 if is_gemm else torch.float32), name
    for name, b in encoder.named_buffers():
        assert b.dtype == torch.float32, name  # rope periods, bias masks
    assert sum(p.nbytes for p in encoder.parameters()) < 0.51 * n_bytes


DEPTH = dict(BASE, criterion=DEPTH_LOSS)
HW = (72, 88)


def ldpt_pair(**kwargs):
    ref = randomize_(build_model(DEPTH))
    model = build_model(dict(DEPTH, **kwargs))
    model.load_state_dict(ref.state_dict(), strict=True)
    return ref, model


def test_ldpt_bf16_encoder_weights():
    ref, model = ldpt_pair(encoder_weight_dtype="bf16")
    assert model.encoder.blocks[0].attn.qkv.weight.dtype == torch.bfloat16
    assert model.encoder.blocks[0].norm1.weight.dtype == torch.float32
    assert model.neck_blocks[0].weight.dtype == torch.float32  # decoder untouched
    data = make_data("depth", HW)
    # eval
    ref.eval(), model.eval()
    with torch.no_grad():
        a, b = ref(copy.deepcopy(data)), model(copy.deepcopy(data))
    assert torch.equal(a["pred_depth"], b["pred_depth"])
    # train (random encoder + decoder RoPE)
    ref.train(), model.train()
    torch.manual_seed(5)
    a = ref(copy.deepcopy(data))["loss"]
    torch.manual_seed(5)
    b = model(copy.deepcopy(data))["loss"]
    assert torch.equal(a, b)


@pytest.mark.parametrize("assign", [False, True])
def test_ldpt_bf16_encoder_weights_survive_load(assign):
    ref, model = ldpt_pair(encoder_weight_dtype="bf16")
    state = {k: v.clone() for k, v in ref.state_dict().items()}  # fp32 encoder weights
    model.load_state_dict(state, strict=True, assign=assign)
    assert model.encoder.blocks[0].attn.qkv.weight.dtype == torch.bfloat16
    assert model.encoder.patch_embed.proj.weight.dtype == torch.bfloat16
    ref.eval(), model.eval()
    data = make_data(None, HW)
    with torch.no_grad():
        a, b = ref(copy.deepcopy(data)), model(copy.deepcopy(data))
    assert torch.equal(a["pred_depth"], b["pred_depth"])


def test_ldpt_bf16_encoder_weights_need_frozen_encoder():
    with pytest.raises(ValueError, match="freeze_encoder"):
        build_model(dict(DEPTH, encoder_weight_dtype="bf16", freeze_encoder=False))
