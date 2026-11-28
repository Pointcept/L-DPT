"""
DINOv3 ViT weights in the transformers format (ldpt.models.dinov3.hf_format): tiny
backbones are converted with the key mapping of transformers' convert_dinov3_vit_to_hf.py
(copied below, so transformers is not needed), written as safetensors folders and read
back into exactly the original state; api.load gives the same prediction from such a
folder as from the .pth; the ViT-7B key coverage against the public parameter count of
facebook/dinov3-vit7b16-pretrain-lvd1689m (6,716,035,072 F32).
"""

import json
import re

import numpy as np
import pytest
import torch

from ldpt.models.builder import MODELS, build_model
from ldpt.models.dinov3 import hf_format
from ldpt.models.dinov3.model import DINOv3ViT7B, DinoVisionTransformer
from ldpt.models.l_dpt import api, convert
from ldpt.models.l_dpt.weights import encoder_state, encoder_weights

safetensors_torch = pytest.importorskip("safetensors.torch")

# ---- transformers' converter (src/transformers/models/dinov3_vit/
# convert_dinov3_vit_to_hf.py, Apache-2.0): key mapping, split_qkv and the key loop

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"cls_token": r"embeddings.cls_token",
    r"mask_token": r"embeddings.mask_token",
    r"storage_tokens": r"embeddings.register_tokens",
    r"patch_embed.proj": r"embeddings.patch_embeddings",
    r"periods": r"inv_freq",
    r"rope_embed": r"rope_embeddings",
    r"blocks.(\d+).attn.proj": r"layer.\1.attention.o_proj",
    r"blocks.(\d+).attn.": r"layer.\1.attention.",
    r"blocks.(\d+).ls(\d+).gamma": r"layer.\1.layer_scale\2.lambda1",
    r"blocks.(\d+).mlp.fc1": r"layer.\1.mlp.up_proj",
    r"blocks.(\d+).mlp.fc2": r"layer.\1.mlp.down_proj",
    r"blocks.(\d+).mlp": r"layer.\1.mlp",
    r"blocks.(\d+).norm": r"layer.\1.norm",
    r"w1": r"gate_proj",
    r"w2": r"up_proj",
    r"w3": r"down_proj",
}


def convert_old_keys_to_new_keys(state_dict_keys):
    old_text = "\n".join(state_dict_keys)
    new_text = old_text
    for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        new_text = re.sub(pattern, replacement, new_text)
    return dict(zip(old_text.split("\n"), new_text.split("\n")))


def split_qkv(state_dict):
    for key in [x for x in state_dict if "qkv" in x]:
        qkv = state_dict.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        state_dict[key.replace("qkv", "q_proj")] = q
        state_dict[key.replace("qkv", "k_proj")] = k
        state_dict[key.replace("qkv", "v_proj")] = v
    return state_dict


def to_transformers(original, model_prefix=False):
    """The converter's state dict (model_prefix: transformers 5's ``model.layer.N``)."""
    original = split_qkv(dict(original))
    new_keys = convert_old_keys_to_new_keys(list(original))
    out = {}
    for key, new_key in new_keys.items():
        tensor = original[key]
        if "bias_mask" in key or "attn.k_proj.bias" in key or "local_cls_norm" in key:
            continue
        if key.startswith("projectors."):
            continue
        if "embeddings.mask_token" in new_key:
            tensor = tensor.unsqueeze(1)
        if "inv_freq" in new_key:
            continue
        if model_prefix and new_key.startswith("layer."):
            new_key = f"model.{new_key}"
        out[new_key] = tensor
    return out


# ---- tiny backbones and transformers folders


def tiny(gated: bool, qkv_bias: bool, seed: int = 0) -> DinoVisionTransformer:
    torch.manual_seed(seed)
    model = DinoVisionTransformer(
        patch_size=16,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=64,
        depth=3,
        num_heads=4,
        ffn_ratio=3 if gated else 4,
        qkv_bias=qkv_bias,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu64" if gated else "mlp",
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    with torch.no_grad():  # nothing left at its init constant, so a mix-up shows
        for name, p in model.named_parameters():
            p.copy_(torch.randn_like(p))
        for name, b in model.named_buffers():
            if name.endswith("bias_mask"):  # the bias transformers applies: q, v
                o = b.numel() // 3
                b.fill_(1.0)
                b[o : 2 * o] = 0.0
    return model


def original_state(model):
    """The model's state as the official file stores it: the key bias masked to 0, and
    the training-only local_cls_norm the converter drops."""
    state = {k: v.clone() for k, v in model.state_dict().items()}
    for k in list(state):
        if k.endswith("qkv.bias"):
            o = state[k].numel() // 3
            state[k][o : 2 * o] = 0.0
    dim = model.embed_dim
    state["local_cls_norm.weight"] = torch.ones(dim)
    state["local_cls_norm.bias"] = torch.zeros(dim)
    return state


def hf_config(model, gated: bool, qkv_bias: bool):
    """config.json as save_pretrained writes it: only the non-default values."""
    values = dict(
        hidden_size=model.embed_dim,
        intermediate_size=(
            model.blocks[0].mlp.w1.out_features
            if gated
            else model.blocks[0].mlp.fc1.out_features
        ),
        num_hidden_layers=model.n_blocks,
        num_attention_heads=model.num_heads,
        query_bias=qkv_bias,
        value_bias=qkv_bias,
        num_register_tokens=4,
        use_gated_mlp=gated,
        hidden_act="silu" if gated else "gelu",
    )
    diff = {k: v for k, v in values.items() if hf_format.CONFIG_DEFAULTS.get(k) != v}
    return dict(diff, model_type="dinov3_vit", architectures=["DINOv3ViTModel"])


def write_folder(folder, state, config, shards: int = 1):
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(config))
    keys = list(state)
    if shards == 1:
        safetensors_torch.save_file(
            {k: v.contiguous() for k, v in state.items()},
            str(folder / "model.safetensors"),
            metadata={"format": "pt"},
        )
        return folder
    weight_map = {}
    for i in range(shards):
        name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        part = {k: state[k].contiguous() for k in keys[i::shards]}
        safetensors_torch.save_file(part, str(folder / name), metadata={"format": "pt"})
        weight_map.update({k: name for k in part})
    index = dict(metadata=dict(total_size=0), weight_map=weight_map)
    (folder / "model.safetensors.index.json").write_text(json.dumps(index))
    return folder


def assert_same(a, b):
    assert sorted(a) == sorted(b), (sorted(set(a) - set(b)), sorted(set(b) - set(a)))
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        assert torch.equal(a[k], b[k]), k


# ---- round trips


@pytest.mark.parametrize("model_prefix", [False, True])
@pytest.mark.parametrize("shards", [1, 3])
def test_round_trip_gated_without_qkv_bias(tmp_path, model_prefix, shards):
    """The 7B's layout (SwiGLU, no qkv bias): exactly the original state, from the
    2025 repos' ``layer.N`` keys and transformers 5's ``model.layer.N``."""
    model = tiny(gated=True, qkv_bias=False)
    original = original_state(model)
    hf = to_transformers(original, model_prefix)
    assert not any(k.startswith("local_cls_norm") for k in hf)
    folder = write_folder(tmp_path / "hf", hf, hf_config(model, True, False), shards)
    assert hf_format.is_hf_folder(folder)
    back = hf_format.read_state(folder)
    original.pop("local_cls_norm.weight"), original.pop("local_cls_norm.bias")
    assert_same(back, original)
    assert torch.equal(back["rope_embed.periods"], model.rope_embed.periods)


def test_round_trip_with_qkv_bias(tmp_path):
    """L / H layout: q / v bias kept, the dropped key bias back as 0 with a mask that
    leaves it out; fc1 / fc2 MLP."""
    model = tiny(gated=False, qkv_bias=True)
    original = original_state(model)
    folder = write_folder(
        tmp_path / "hf",
        to_transformers(original),
        hf_config(model, False, True),
    )
    back = hf_format.read_state(folder)
    original.pop("local_cls_norm.weight"), original.pop("local_cls_norm.bias")
    assert_same(back, original)
    mask = back["blocks.0.attn.qkv.bias_mask"]
    assert mask.tolist() == [1.0] * 64 + [0.0] * 64 + [1.0] * 64


def test_dtypes_device_and_encoder_state(tmp_path):
    model = tiny(gated=True, qkv_bias=False)
    original = original_state(model)
    folder = write_folder(
        tmp_path / "hf", to_transformers(original), hf_config(model, True, False)
    )
    dtypes = {"blocks.1.attn.qkv.weight": torch.bfloat16}
    back = hf_format.read_state(folder, dtypes=dtypes, device="cpu")
    assert back["blocks.1.attn.qkv.weight"].dtype == torch.bfloat16
    assert torch.equal(
        back["blocks.1.attn.qkv.weight"],
        original["blocks.1.attn.qkv.weight"].to(torch.bfloat16),
    )
    state = encoder_state(folder, dtypes=dtypes)  # as api.load reads it
    assert set(state) == {"encoder." + k for k in back}
    assert not any("local_cls_norm" in k for k in state)


def test_errors(tmp_path):
    model = tiny(gated=True, qkv_bias=False)
    config = hf_config(model, True, False)
    hf = to_transformers(original_state(model))
    bad = dict(hf, **{"layer.0.attention.extra.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="unexpected key 'layer.0.attention.extra"):
        hf_format.read_state(write_folder(tmp_path / "a", bad, config))
    missing = {k: v for k, v in hf.items() if k != "layer.2.attention.k_proj.weight"}
    with pytest.raises(ValueError, match="incomplete q / k / v.*blocks.2.attn.qkv"):
        hf_format.read_state(write_folder(tmp_path / "b", missing, config))
    twice = dict(hf, **{"model.layer.0.norm1.weight": hf["layer.0.norm1.weight"] + 0})
    with pytest.raises(ValueError, match="two tensors map to blocks.0.norm1.weight"):
        hf_format.read_state(write_folder(tmp_path / "c", twice, config))
    other = write_folder(tmp_path / "d", hf, dict(config, model_type="dinov2"))
    with pytest.raises(ValueError, match="model_type 'dinov2'"):
        hf_format.read_state(other)
    # config.json against a known backbone
    folder = write_folder(tmp_path / "e", hf, config)
    with pytest.raises(ValueError, match="is not dinov3_vit7b: hidden_size=64"):
        hf_format.read_state(folder, encoder_type="dinov3_vit7b")
    with pytest.raises(ValueError, match="is not dinov3_vit7b"):
        encoder_weights("dinov3_vit7b", folder)  # verify=True checks config.json
    assert encoder_weights("dinov3_vit7b", folder, verify=False) == folder
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="without config.json"):
        encoder_weights("dinov3_vit7b", tmp_path / "empty")
    assert not hf_format.is_hf_folder(tmp_path / "empty")


def test_check_config_7b():
    config = {**hf_format.CONFIG_DEFAULTS, **hf_format.HF_ARCHITECTURES["dinov3_vit7b"]}
    hf_format.check_config(config, "dinov3_vit7b")
    with pytest.raises(ValueError, match="rope_theta=10000.0"):
        hf_format.check_config(dict(config, rope_theta=10000.0), "dinov3_vit7b")
    with pytest.raises(ValueError, match="no transformers format known"):
        hf_format.check_config(config, "dinov3_vitx")


def test_7b_key_coverage():
    """The ViT-7B of model.py through the converter's mapping and back: every key and
    shape of the backbone, and the transformers state has the public parameter count of
    facebook/dinov3-vit7b16-pretrain-lvd1689m (meta tensors: no memory)."""
    with torch.device("meta"):
        model = DINOv3ViT7B()
    original = dict(model.state_dict())
    dim = model.embed_dim
    original["local_cls_norm.weight"] = torch.empty(dim, device="meta")
    original["local_cls_norm.bias"] = torch.empty(dim, device="meta")
    hf = to_transformers(original)
    assert sum(v.numel() for v in hf.values()) == 6_716_035_072
    config = {**hf_format.CONFIG_DEFAULTS, **hf_format.HF_ARCHITECTURES["dinov3_vit7b"]}
    shapes, parts = {}, {}
    for key, value in hf.items():
        name, part = hf_format.original_key(key, gated=True)
        if part is None:
            shapes[name] = tuple(value.shape)
        else:
            parts.setdefault(name, []).append(tuple(value.shape))
    assert all(len(v) == 3 for v in parts.values())
    shapes.update({k: (sum(s[0] for s in v), *v[0][1:]) for k, v in parts.items()})
    shapes["mask_token"] = (1, dim)
    periods = hf_format.rope_periods(
        config, hf_format.PERIODS_DTYPE["dinov3_vit7b"], "dinov3_vit7b"
    )
    shapes["rope_embed.periods"] = tuple(periods.shape)
    assert shapes == {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert periods.dtype == torch.float32 == model.rope_embed.periods.dtype
    # the official file's values: the CPU's pow but for two GPU-rounded ones (1 ulp)
    cpu = hf_format.rope_periods(config)
    assert (periods != cpu).nonzero().flatten().tolist() == [22, 27]
    ulp = torch.finfo(torch.float32).eps * cpu
    assert ((periods - cpu).abs() <= ulp).all()
    assert float(periods[0]) == 1.0 and float(periods[16]) == 10.0


# ---- api.load from a transformers folder


class _TinyEncoder(DinoVisionTransformer):
    def __init__(self):
        super().__init__(
            patch_size=16,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=64,
            depth=4,
            num_heads=4,
            ffn_ratio=3,
            qkv_bias=False,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="swiglu64",
            n_storage_tokens=4,
            mask_k_bias=True,
        )


def test_api_load_from_folder_matches_pth(tmp_path):
    if "dinov3_tiny_hf_test" not in MODELS.module_dict:
        MODELS.register_module("dinov3_tiny_hf_test", module=_TinyEncoder)
    cfg = dict(
        type="L-DPT-v1m1",
        encoder=dict(type="dinov3_tiny_hf_test"),
        encoder_inter_layers=[0, 1, 2, 3],
        dec_dim=32,
        dec_num_heads=2,
        dec_num_layers=(1, 1, 1, 1),
        head="exp",
        criterion=False,
    )
    torch.manual_seed(0)
    model = build_model(cfg)
    model.encoder.init_weights()
    enc = {k: v for k, v in model.encoder.state_dict().items()}
    torch.save(enc, tmp_path / "enc.pth")
    decoder = {
        k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")
    }
    torch.save(dict(state_dict=decoder), tmp_path / "dev.pth")
    release = tmp_path / "rel.pth"
    convert.convert(tmp_path / "dev.pth", release, config=cfg)
    folder = write_folder(
        tmp_path / "hf",
        to_transformers(enc, model_prefix=True),
        hf_config(model.encoder, True, False),
        shards=2,
    )
    image = np.random.default_rng(0).integers(0, 256, (40, 56, 3), dtype=np.uint8)
    out = []
    for weights in (tmp_path / "enc.pth", folder):
        loaded = api.load(release, weights, device="cpu", verify=False)
        assert loaded.weights_info["encoder"] == str(weights)
        out.append(api.predict(loaded, image, size=48))
    assert np.array_equal(out[0], out[1])
