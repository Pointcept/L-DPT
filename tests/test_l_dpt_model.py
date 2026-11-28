"""
L-DPT-v1m1 flags and bookkeeping (CPU).
"""

import copy
import math

import pytest
import torch

from helpers import init_encoder_, randomize_decoder_
from test_l_dpt_golden import (
    BASE,
    DEC,
    DEPTH_LOSS,
    ENC,
    correlated_depth,
    make_data,
    randomize_,
)

from ldpt.models.builder import build_model
from ldpt.models.dinov3.layers import RopePositionEmbedding
from ldpt.models.l_dpt.presets import (
    INPUT_SIZE,
    PRESETS,
    RELEASE_L,
    WEIGHTS,
    get_preset,
)

STUB = dict(
    encoder=dict(type="feature_stub_test"), encoder_inter_layers=[9, 19, 29, 39]
)
HF_DEC = dict(
    dec_dim=1024, dec_num_heads=16, dec_num_layers=(2, 2, 2, 2), dec_attn_downscale=2
)
DEPTH = dict(BASE, criterion=DEPTH_LOSS)
HW = (72, 88)


def test_encoder_train_mode():
    cfg = DEPTH
    model = build_model(cfg).train()
    assert model.training and model.encoder.training  # the default
    model = build_model(dict(cfg, encoder_train_mode=False)).train()
    assert model.training and model.dec_blocks.training and model.rope.training
    assert not model.encoder.training
    assert not any(m.training for m in model.encoder.modules())
    model.eval()
    assert not model.training and not model.encoder.training


def test_train_rope_is_random():
    """The training forward samples RoPE (else the golden train cases are weak)."""
    model = randomize_(build_model(DEPTH)).train()
    data = make_data("depth", HW)
    torch.manual_seed(1)
    a = model(copy.deepcopy(data))["loss"]
    torch.manual_seed(2)
    b = model(copy.deepcopy(data))["loss"]
    assert not torch.equal(a, b)


def test_attn_dtype_and_legacy_attn_impl():
    cfg = dict(type="L-DPT-v1m1", **ENC, **DEC)
    get = lambda m: m.dec_blocks[0][0].attn.attn_dtype  # noqa: E731
    assert get(build_model(cfg)) is None
    assert get(build_model(dict(cfg, attn_dtype="bf16"))) == torch.bfloat16
    assert get(build_model(dict(cfg, attn_impl="auto"))) is None
    assert get(build_model(dict(cfg, attn_impl="flash"))) == torch.bfloat16
    assert (
        get(build_model(dict(cfg, attn_impl="flash", attn_dtype="bf16")))
        == torch.bfloat16
    )
    with pytest.raises(ValueError, match="conflicts"):
        build_model(dict(cfg, attn_impl="flash", attn_dtype="fp32"))
    with pytest.raises(ValueError, match="attn_dtype"):
        build_model(dict(cfg, attn_dtype="fp16"))


def test_state_dict_keys():
    cfg = dict(type="L-DPT-v1m1", **STUB, **HF_DEC)
    with torch.device("meta"):
        keys = set(build_model(cfg).state_dict())
        shared = set(build_model(dict(cfg, selection="shared")).state_dict())
        seg = set(build_model(dict(cfg, task="semseg", num_classes=7)).state_dict())
    assert len(keys) == 131
    assert {"rope.periods", "neck_blocks.3.weight", "unpatchfy.0.proj.weight"} <= keys
    assert "dec_blocks.3.1.attn.q_proj.weight" in keys
    assert "neck_block.weight" in shared and not any("neck_blocks" in k for k in shared)
    assert seg - keys == {"seg_head.weight", "seg_head.bias"}


def n_params(**kwargs):
    with torch.device("meta"):
        model = build_model(dict(type="L-DPT-v1m1", **STUB, **kwargs))
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 2)


# decoder sizes S / B / L: (dim, heads, layers, attn downscale)
PAPER = {
    "S": dict(
        dec_dim=256, dec_num_heads=8, dec_num_layers=(2,) * 4, dec_attn_downscale=1
    ),
    "B": dict(
        dec_dim=768, dec_num_heads=12, dec_num_layers=(2,) * 4, dec_attn_downscale=2
    ),
    "L": dict(
        dec_dim=1024, dec_num_heads=16, dec_num_layers=(4,) * 4, dec_attn_downscale=1
    ),
}


def test_param_counts():
    # 23M / 98M / 269M with a 1x1 readout; the released 3x3 readout: L = 271M
    k1 = [n_params(**PAPER[v], readout_kernel_size=1) for v in "SBL"]
    k3 = [n_params(**PAPER[v], readout_kernel_size=3) for v in "SBL"]
    assert k1 == [23.16, 97.79, 268.90]
    assert k3 == [23.68, 99.36, 271.00]
    assert n_params(**HF_DEC) == 153.46  # 1024 / 16 heads / (2, 2, 2, 2) / downscale 2


def test_out_dim_defaults():
    cfg = dict(type="L-DPT-v1m1", **ENC, **DEC)
    assert build_model(cfg).out_dim == 1
    assert build_model(dict(cfg, task="semseg", num_classes=3)).out_dim == 32
    with pytest.raises(ValueError, match="one channel"):
        build_model(dict(cfg, out_dim=4))
    with pytest.raises(ValueError, match="num_classes"):
        build_model(dict(cfg, task="semseg"))  # num_classes missing


def test_errors_have_hints():
    cfg = dict(type="L-DPT-v1m1", **ENC, **DEC)
    with pytest.raises(ValueError, match="task must be one of"):
        build_model(dict(cfg, task="depth"))
    with pytest.raises(ValueError, match="out of range"):
        build_model(dict(cfg, encoder_inter_layers=[9, 19, 29, 39]))
    with pytest.raises(ValueError, match="one encoder tap per decoder stage"):
        build_model(dict(cfg, encoder_inter_layers=[0, 1, 2]))
    with pytest.raises(ValueError, match="RoPE"):
        build_model(dict(cfg, dec_num_heads=3))
    with pytest.raises(ValueError, match="precision"):
        build_model(dict(cfg, precision="fp16"))
    with pytest.raises(ValueError, match="odd"):
        build_model(dict(cfg, readout_kernel_size=2))
    with pytest.raises(TypeError, match="registry cfg"):
        build_model(dict(cfg, encoder="dinov3_vit7b"))
    model = build_model(cfg)
    with pytest.raises(ValueError, match="pad it first"):
        model.encode(torch.randn(1, 3, 40, 48))
    with pytest.raises(ValueError, match="needs the target"):
        model.train()(dict(image=torch.randn(1, 3, 32, 32)))


def test_meta_build_has_no_storage():
    cfg = dict(type="L-DPT-v1m1", **ENC, **DEC, encoder_weight_dtype="bf16")
    with torch.device("meta"):
        model = build_model(cfg)
    assert all(t.is_meta for t in model.state_dict().values())
    assert model.encoder.blocks[0].attn.qkv.weight.dtype == torch.bfloat16
    assert model.encoder.norm.weight.dtype == torch.float32


def test_precision_bf16():
    """Decoder under bf16 autocast: fp32 outputs and loss, close to fp32, trains."""
    cfg = DEPTH
    fp32 = randomize_(build_model(cfg)).eval()
    bf16 = build_model(dict(cfg, precision="bf16")).eval()
    bf16.load_state_dict(fp32.state_dict())
    data = correlated_depth(fp32, make_data("depth", HW, gt_hw=(80, 100)), seed=3)
    with torch.no_grad():
        torch.manual_seed(0)  # the same fit pixels for both
        a = fp32(copy.deepcopy(data))
        torch.manual_seed(0)
        b = bf16(copy.deepcopy(data))
        image, _ = bf16.pad(data["image"])
        selections = bf16.select(bf16.encode(image))
    assert selections[0].dtype == torch.bfloat16
    assert b["pred_depth"].dtype == torch.float32 and b["loss"].dtype == torch.float32
    assert 0 < (a["pred_depth"] - b["pred_depth"]).abs().max() < 2e-2
    assert a["loss"] > 0
    assert abs(a["loss"].item() - b["loss"].item()) < 1e-2 * a["loss"].item()
    bf16.train()
    out = bf16(make_data("depth", HW))
    out["loss"].backward()
    grads = [p.grad for p in bf16.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


# L-DPT depth loss criterion of an exp-head model (test_depth_loss_roe.py's ROE, smaller)
EXP_LOSS = dict(
    type="relative_depth_loss",
    align="roe",
    head="exp",
    min_valid=100,
    roe_points=256,
    label_types=dict(A=["hypersim"], B=["blendedmvs"], C=[]),
    type_trim=dict(A=0.0, B=0.05, C=0.05),
    local_levels=dict(A=[4, 16], B=[4], C=[4]),
    local_cell=256.0,
    grad_types=["A"],
    mage_weight=4.0,
    mage_num_scales=4,
)
EXP = dict(BASE, head="exp", readout_fp32=True, precision="bf16")


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_inference_only_matches_trained_build(precision):
    """criterion=False (inference only, release files / api.load): an exp-head model
    builds without its training criterion, has the same state dict and is bitwise the
    model built with it in eval: infer, and forward with the target (the loss is
    skipped, the prediction is unchanged). Training raises."""
    cfg = dict(EXP, precision=precision)
    ref = randomize_(build_model(dict(cfg, criterion=EXP_LOSS))).eval()
    with pytest.raises(ValueError, match="differs"):  # the default loss is "sigmoid"
        build_model(cfg)
    model = build_model(dict(cfg, criterion=False))
    assert model.criterion is None and ref.criterion is not None
    assert list(model.state_dict()) == list(ref.state_dict())
    model.load_state_dict(ref.state_dict(), strict=True)
    model.eval()
    data = make_data("depth", HW, gt_hw=(80, 100))
    data["name"] = ["hypersim/scene/0", "blendedmvs/scene/1"]  # label_types routing
    with torch.no_grad():
        assert torch.equal(model.infer(data["image"]), ref.infer(data["image"]))
        out, want = model(copy.deepcopy(data)), ref(copy.deepcopy(data))
    assert set(out) == {"pred_depth"} and "loss" in want
    assert torch.equal(out["pred_depth"], want["pred_depth"])
    with pytest.raises(ValueError, match="criterion=False"):
        model.train()(copy.deepcopy(data))


def test_criterion_none_is_the_default_loss():
    """criterion=None (the default) still builds relative_depth_loss() / the seg
    criteria; False builds none; True is not an option."""
    from ldpt.models.losses.depth import RelativeDepthLoss

    cfg = dict(type="L-DPT-v1m1", **ENC, **DEC)
    model = build_model(cfg)
    assert isinstance(model.criterion, RelativeDepthLoss)
    assert model.criterion.head == "sigmoid"
    seg = dict(cfg, task="semseg", out_dim=8, num_classes=5)
    assert build_model(seg).criterion is not None
    seg_only = build_model(dict(seg, criterion=False)).eval()
    assert seg_only.criterion is None
    data = make_data("segment", (32, 48))
    with torch.no_grad():
        assert set(seg_only(data)) == {"pred_segment"}
    with pytest.raises(ValueError, match="criterion=True"):
        build_model(dict(cfg, criterion=True))


# decoder keys and parameters per preset (the release models)
PRESET_DECODERS = {
    "l-dpt_large_depth_zeroshot": (251, 270_995_712),
    "l-dpt_large_depth_mixed": (251, 270_995_712),
    "l-dpt_large_semseg_cocostuff": (161, 772_017_131),
}


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets(name):
    kwargs = get_preset(name)
    assert (
        kwargs is not PRESETS[name]
        and kwargs["encoder"] is not PRESETS[name]["encoder"]
    )
    assert INPUT_SIZE[name] == 768
    with torch.device("meta"):
        model = build_model(dict(type="L-DPT-v1m1", **kwargs))
    decoder = {
        k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")
    }
    n = sum(
        p.numel() for k, p in model.named_parameters() if not k.startswith("encoder.")
    )
    assert (len(decoder), n) == PRESET_DECODERS[name]
    assert model.encoder_inter_layers == [9, 19, 29, 39] and model.enc_dim == 4096


def test_release_presets():
    """Both depth models share one architecture (they differ only in weights): the
    model of the depth recipes, built inference-only; INPUT_SIZE 768; weight slots
    pinned to the release repo. Every preset has a weight slot."""
    zs, mix = "l-dpt_large_depth_zeroshot", "l-dpt_large_depth_mixed"
    assert set(PRESETS) == {zs, mix, "l-dpt_large_semseg_cocostuff"}
    assert PRESETS[zs] is PRESETS[mix] is RELEASE_L
    kwargs = get_preset(mix)
    assert kwargs == get_preset(zs) and kwargs is not RELEASE_L
    assert {k: kwargs[k] for k in RELEASE_KEYS} == RELEASE_KEYS
    with torch.device("meta"):
        model = build_model(dict(type="L-DPT-v1m1", **kwargs))
    assert model.criterion is None and model.head == "exp" and model.readout_fp32
    blocks = [blk for stage in model.dec_blocks for blk in stage]
    assert [blk.attn.qk_bound for blk in blocks] == [1000.0] * 16
    assert [blk.attn.force_fp32 for blk in blocks] == [False] * 12 + [True] * 4
    assert INPUT_SIZE[zs] == INPUT_SIZE[mix] == 768
    assert set(WEIGHTS) == set(PRESETS)
    for name, env in (
        (zs, "LDPT_CKPT_LARGE_DEPTH_ZEROSHOT"),
        (mix, "LDPT_CKPT_LARGE_DEPTH_MIXED"),
    ):
        source = WEIGHTS[name]
        assert source.format == "release" and source.env == env
        assert source.file == f"{name}.pth"
        assert source.published and source.repo == "pointcept-bot/L-DPT"
        assert len(source.revision) == 40 and len(source.sha256) == 64
        assert source.size > 1_000_000_000


# the release architecture as the task states it (and the run configs)
RELEASE_KEYS = dict(
    dec_dim=1024,
    dec_num_heads=16,
    dec_num_layers=(4, 4, 4, 4),
    dec_attn_downscale=1,
    dec_ffn_ratio=4.0,
    dec_pos_embed_rope_base=100,
    dec_pos_embed_rope_normalize_coords="min",
    readout_kernel_size=3,
    precision="bf16",
    attn_dtype="bf16",
    head="exp",
    readout_fp32=True,
    attn_fp32="last_stage",
    dec_qk_bound=1000,
    dec_qk_norm=False,
    encoder_weight_dtype="bf16",
)


def test_unknown_preset_suggests():
    with pytest.raises(ValueError, match="did you mean 'l-dpt_large_depth_zeroshot'"):
        get_preset("l-dpt-large-depth-zeroshot")


def test_legacy_preset_names():
    """The names stored in the first release files ("l-zs", "l-mix") read as the
    current presets (canonical_preset) but are not presets themselves."""
    from ldpt.models.l_dpt.presets import LEGACY_PRESETS, canonical_preset

    assert LEGACY_PRESETS == {
        "l-zs": "l-dpt_large_depth_zeroshot",
        "l-mix": "l-dpt_large_depth_mixed",
    }
    assert set(LEGACY_PRESETS.values()) <= set(PRESETS)
    assert not set(LEGACY_PRESETS) & set(PRESETS)
    for old, new in LEGACY_PRESETS.items():
        assert canonical_preset(old) == new and canonical_preset(new) == new
        with pytest.raises(ValueError, match=f"renamed '{new}'"):
            get_preset(old)
    assert canonical_preset(None) is None and canonical_preset("x") == "x"


# ---- decoder RoPE evaluation scale (dec_pos_embed_rope_eval_scale)

ROPE_TINY = dict(
    encoder=dict(type="dinov3_tiny_test"),
    encoder_inter_layers=[0, 1, 2, 3],
    dec_dim=32,
    dec_num_heads=2,
    dec_num_layers=(1, 2, 1, 1),
    dec_pos_embed_rope_jitter_coords=1.2,
    dec_pos_embed_rope_rescale_coords=8,
)


def test_rope_eval_scale():
    kw = dict(embed_dim=64, num_heads=2, base=100, normalize_coords="min")
    base = RopePositionEmbedding(**kw).eval()
    scaled = RopePositionEmbedding(**kw, eval_scale=2.0).eval()
    H, W = 3, 5
    ch = torch.arange(0.5, H) / 3
    cw = torch.arange(0.5, W) / 3
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1).flatten(0, 1)
    coords = (2.0 * coords - 1.0) * 2.0
    angles = (
        (2 * math.pi * coords[:, :, None] / scaled.periods[None, None, :])
        .flatten(1, 2)
        .tile(2)
    )
    sin, cos = scaled(H=H, W=W)
    assert torch.allclose(sin, torch.sin(angles), atol=1e-6)
    assert torch.allclose(cos, torch.cos(angles), atol=1e-6)
    assert not torch.allclose(sin, base(H=H, W=W)[0])
    # training is unchanged
    base.train(), scaled.train()
    assert all(torch.equal(a, b) for a, b in zip(base(H=H, W=W), scaled(H=H, W=W)))


def test_model_rope_eval_scale():
    torch.manual_seed(0)
    ref = build_model(dict(type="L-DPT-v1m1", **ROPE_TINY))
    init_encoder_(ref.encoder, seed=0)
    randomize_decoder_(ref, seed=1)
    state = ref.state_dict()
    x = torch.randn(1, 3, 48, 80)
    outs = {}
    for r in (None, 1.0, 2.0):
        m = build_model(
            dict(type="L-DPT-v1m1", **ROPE_TINY, dec_pos_embed_rope_eval_scale=r)
        )
        assert m.state_dict().keys() == state.keys()
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            outs[r] = m.infer(x)
    assert torch.equal(outs[None], outs[1.0])
    assert not torch.allclose(outs[None], outs[2.0])
    for bad in (0, -1.0, True, "2", float("inf")):
        with pytest.raises(ValueError, match="dec_pos_embed_rope_eval_scale"):
            build_model(
                dict(type="L-DPT-v1m1", **ROPE_TINY, dec_pos_embed_rope_eval_scale=bad)
            )
