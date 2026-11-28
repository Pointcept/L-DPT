"""
DINOv3 ViT weights in the Hugging Face transformers format -> the original key layout.

Meta publishes the DINOv3 backbones twice: the original ``.pth`` files of the DINOv3
release (the key layout of ``model.py``) and gated Hugging Face repos in the transformers
format (``DINOv3ViTModel``: ``config.json`` + ``model*.safetensors``), e.g.
``facebook/dinov3-vit7b16-pretrain-lvd1689m``. ``read_state(folder)`` reads such a
folder (a ``snapshot_download`` of the repo) into the original layout (``iter_state``:
one tensor at a time), the inverse of transformers' ``convert_dinov3_vit_to_hf.py``:

    transformers (both layer prefixes)          original
    embeddings.cls_token                        cls_token
    embeddings.mask_token [1, 1, D]             mask_token [1, D]
    embeddings.register_tokens                  storage_tokens
    embeddings.patch_embeddings.*               patch_embed.proj.*
    [model.]layer.N.norm{1,2}.*                 blocks.N.norm{1,2}.*
    [model.]layer.N.attention.{q,k,v}_proj.*    blocks.N.attn.qkv.* (q; k; v rows)
    [model.]layer.N.attention.o_proj.*          blocks.N.attn.proj.*
    [model.]layer.N.layer_scale{1,2}.lambda1    blocks.N.ls{1,2}.gamma
    [model.]layer.N.mlp.{gate,up,down}_proj.*   blocks.N.mlp.{w1,w2,w3}.* (gated MLP)
    [model.]layer.N.mlp.{up,down}_proj.*        blocks.N.mlp.{fc1,fc2}.*
    norm.*                                      norm.*

The repos uploaded in 2025 name the blocks ``layer.N``; transformers 5 renames them to
``model.layer.N`` on load and saves them that way: both are read. Not in the transformers
format, so rebuilt here: ``rope_embed.periods`` (computed as ``RopePositionEmbedding``
does, in the dtype of the official file: fp32 for the 7B, bf16 for L and H; the 7B
file's two GPU-rounded values are pinned) and, for
backbones with a qkv bias, the zero key bias and ``bias_mask`` (1 on the query and value
rows, 0 on the key rows: the bias transformers applies). Dropped by the converter and
not used here: ``local_cls_norm.*`` (training only). The ViT-7B has no qkv bias, so its
state is the original file's exactly (tests/test_dinov3_hf_format.py; the full round trip
of the official 7B file was checked tensor by tensor).

Tensors are read one at a time from the memory-mapped safetensors files and cast /
moved as they are read (``dtypes``, ``device``), so a 27 GB fp32 7B never sits in memory
as a whole. Needs ``safetensors`` (imported when a folder is read).

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor

__all__ = [
    "HF_ARCHITECTURES",
    "check_config",
    "is_hf_folder",
    "iter_state",
    "read_config",
    "read_state",
    "rope_periods",
]

PathLike = Union[str, os.PathLike]
MODEL_TYPE = "dinov3_vit"

# DINOv3ViTConfig defaults (transformers): config.json stores only the values that
# differ from them (save_pretrained's use_diff)
CONFIG_DEFAULTS: Dict[str, Any] = dict(
    patch_size=16,
    hidden_size=384,
    intermediate_size=1536,
    num_hidden_layers=12,
    num_attention_heads=6,
    hidden_act="gelu",
    layer_norm_eps=1e-5,
    rope_theta=100.0,
    num_channels=3,
    query_bias=True,
    key_bias=False,
    value_bias=True,
    proj_bias=True,
    mlp_bias=True,
    use_gated_mlp=False,
    num_register_tokens=0,
)

# the transformers config of each backbone of model.py (as the official converter
# writes it), and the dtype of rope_embed.periods in the official .pth file
HF_ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "dinov3_vit7b": dict(
        hidden_size=4096,
        intermediate_size=8192,
        num_hidden_layers=40,
        num_attention_heads=32,
        query_bias=False,
        value_bias=False,
        num_register_tokens=4,
        use_gated_mlp=True,
        hidden_act="silu",
    ),
    "dinov3_vith": dict(
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        num_register_tokens=4,
        use_gated_mlp=True,
        hidden_act="silu",
    ),
    "dinov3_vitl": dict(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_register_tokens=4,
        use_gated_mlp=False,
        hidden_act="gelu",
    ),
    "dinov3_vitb": dict(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_register_tokens=4,
        use_gated_mlp=False,
        hidden_act="gelu",
    ),
}
# dtype of rope_embed.periods in the official .pth files (checked: 7B, H and L);
# others: fp32, the dtype the backbones of model.py declare
PERIODS_DTYPE: Dict[str, torch.dtype] = {
    "dinov3_vit7b": torch.float32,
    "dinov3_vith": torch.bfloat16,
    "dinov3_vitl": torch.bfloat16,
}
# the official 7B file's periods were computed on a GPU: two of its 32 fp32 values are
# one ulp off the CPU's pow; the file's values (index: float.hex)
PERIODS_FILE_VALUES: Dict[str, Dict[int, str]] = {
    "dinov3_vit7b": {22: "0x1.7b6b76p+4", 27: "0x1.8592f4p+5"},
}
# compared with the expected architecture (the keys and shapes are checked by the
# strict load; these change the output without changing a shape)
_CHECKED = (
    "patch_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "hidden_act",
    "layer_norm_eps",
    "rope_theta",
    "query_bias",
    "key_bias",
    "value_bias",
    "proj_bias",
    "mlp_bias",
    "use_gated_mlp",
    "num_register_tokens",
)

_LAYER = re.compile(r"^(?:model\.)?layer\.(\d+)\.(.+)$")
_TOP = {
    "embeddings.cls_token": "cls_token",
    "embeddings.mask_token": "mask_token",
    "embeddings.register_tokens": "storage_tokens",
    "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
    "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
    "norm.weight": "norm.weight",
    "norm.bias": "norm.bias",
}
_IN_LAYER = {
    "norm1": "norm1",
    "norm2": "norm2",
    "attention.o_proj": "attn.proj",
    "layer_scale1.lambda1": "ls1.gamma",
    "layer_scale2.lambda1": "ls2.gamma",
}
_MLP = {
    True: {"gate_proj": "w1", "up_proj": "w2", "down_proj": "w3"},
    False: {"up_proj": "fc1", "down_proj": "fc2"},
}
_QKV = ("q_proj", "k_proj", "v_proj")


def is_hf_folder(path: PathLike) -> bool:
    """A folder in the transformers format: config.json + safetensors weights."""
    path = Path(path)
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and bool(_shards(path, missing_ok=True))
    )


def read_config(path: PathLike) -> Dict[str, Any]:
    """config.json of a transformers folder, the DINOv3ViTConfig defaults filled."""
    data = json.loads((Path(path) / "config.json").read_text())
    if data.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"{path}: config.json has model_type {data.get('model_type')!r}, not "
            f"{MODEL_TYPE!r} (a DINOv3 ViT in the transformers format)"
        )
    return {**CONFIG_DEFAULTS, **data}


def check_config(config: Mapping[str, Any], encoder_type: str, where: str = "") -> None:
    """ValueError unless ``config`` (read_config) is the backbone ``encoder_type``."""
    if encoder_type not in HF_ARCHITECTURES:
        raise ValueError(
            f"no transformers format known for {encoder_type!r}: one of "
            f"{sorted(HF_ARCHITECTURES)}"
        )
    expected = {**CONFIG_DEFAULTS, **HF_ARCHITECTURES[encoder_type]}
    wrong = [
        f"{k}={config.get(k)!r} (expected {expected[k]!r})"
        for k in _CHECKED
        if config.get(k) != expected[k]
    ]
    if wrong:
        raise ValueError(
            f"{where or 'config.json'} is not {encoder_type}: " + ", ".join(wrong)
        )


def rope_periods(
    config: Mapping[str, Any],
    dtype: torch.dtype = torch.float32,
    encoder_type: Optional[str] = None,
) -> Tensor:
    """``rope_embed.periods`` as RopePositionEmbedding._init_weights computes it (base
    ``rope_theta``, on the CPU, in ``dtype``), with the official file's values of
    ``encoder_type`` where they differ (PERIODS_FILE_VALUES)."""
    head = config["hidden_size"] // config["num_attention_heads"]
    exponent = 2 * torch.arange(head // 4, dtype=dtype) / (head // 2)
    periods = config["rope_theta"] ** exponent
    for i, value in PERIODS_FILE_VALUES.get(encoder_type, {}).items():
        periods[i] = float.fromhex(value)
    return periods


def _shards(path: Path, missing_ok: bool = False):
    index = path / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        files = [path / n for n in names]
        absent = [str(f) for f in files if not f.is_file()]
        if absent:
            raise FileNotFoundError(
                f"{path}: shards listed in {index.name} missing: {absent}"
            )
        return files
    single = path / "model.safetensors"
    if single.is_file():
        return [single]
    if missing_ok:
        return []
    raise FileNotFoundError(
        f"{path}: no model.safetensors(.index.json): not a transformers folder"
    )


def _tensors(path: Path) -> Iterator[Tuple[str, Any]]:
    """(key, tensor) of every tensor of the folder's safetensors files, one at a time."""
    from safetensors import safe_open  # only when a transformers folder is read

    for shard in _shards(path):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def original_key(key: str, gated: bool) -> Tuple[str, Optional[str]]:
    """(original key, qkv part) of a transformers key; the part ("q_proj", ...) is set
    for the attention projections that make up ``blocks.N.attn.qkv``. KeyError for a
    key the format does not have."""
    if key in _TOP:
        return _TOP[key], None
    m = _LAYER.match(key)
    if m is None:
        raise KeyError(key)
    n, rest = m.groups()
    name, _, param = rest.rpartition(".")
    if param not in ("weight", "bias") and rest not in _IN_LAYER:
        raise KeyError(key)
    if rest in _IN_LAYER:  # layer scales: no .weight / .bias
        return f"blocks.{n}.{_IN_LAYER[rest]}", None
    if name in _IN_LAYER:
        return f"blocks.{n}.{_IN_LAYER[name]}.{param}", None
    if name.startswith("attention.") and name[len("attention.") :] in _QKV:
        return f"blocks.{n}.attn.qkv.{param}", name[len("attention.") :]
    if name.startswith("mlp.") and name[len("mlp.") :] in _MLP[gated]:
        return f"blocks.{n}.mlp.{_MLP[gated][name[len('mlp.'):]]}.{param}", None
    raise KeyError(key)


def read_state(
    path: PathLike,
    *,
    encoder_type: Optional[str] = None,
    dtypes: Optional[Mapping[str, torch.dtype]] = None,
    device: Union[str, torch.device, None] = None,
) -> Dict[str, Tensor]:
    """The original-layout state dict of a transformers DINOv3 ViT folder (iter_state
    collected)."""
    return dict(
        iter_state(path, encoder_type=encoder_type, dtypes=dtypes, device=device)
    )


def iter_state(
    path: PathLike,
    *,
    encoder_type: Optional[str] = None,
    dtypes: Optional[Mapping[str, torch.dtype]] = None,
    device: Union[str, torch.device, None] = None,
) -> Iterator[Tuple[str, Tensor]]:
    """(original key, tensor) of a transformers DINOv3 ViT folder, one at a time: a
    layer's q / k / v are joined into ``qkv`` as soon as all three are read.

    ``encoder_type`` (e.g. "dinov3_vit7b") checks config.json against that backbone (a
    type HF_ARCHITECTURES lacks: not checked) and picks the dtype of
    ``rope_embed.periods`` (PERIODS_DTYPE); ``dtypes`` maps an original key to the
    dtype to store it in; ``device`` moves each tensor as it is read. Raises ValueError
    on a key the format does not have, a duplicate, or an incomplete q / k / v set.
    """
    path = Path(path)
    config = read_config(path)
    if encoder_type in HF_ARCHITECTURES:
        check_config(config, encoder_type, str(path / "config.json"))
    gated = bool(config["use_gated_mlp"])
    dim = config["hidden_size"]
    dtypes = dtypes or {}
    query_bias, key_bias, value_bias = (
        bool(config[k]) for k in ("query_bias", "key_bias", "value_bias")
    )
    if (query_bias or key_bias or value_bias) and not (query_bias and value_bias):
        raise ValueError(f"{path}: a qkv bias on only some of q / v is not supported")
    seen = set()

    def out(key: str, value: Tensor) -> Tuple[str, Tensor]:
        if key in seen:
            raise ValueError(f"{path}: two tensors map to {key}")
        seen.add(key)
        dtype = dtypes.get(key)
        if dtype is not None or device is not None:
            value = value.to(device=device, dtype=dtype)
        return key, value

    def qkv_complete(name: str, group: Mapping[str, Tensor]) -> bool:
        if not name.endswith(".bias"):
            return len(group) == 3
        need = {p for p, on in zip(_QKV, (query_bias, key_bias, value_bias)) if on}
        return set(group) == need

    parts: Dict[str, Dict[str, Tensor]] = {}  # qkv key -> {"q_proj": tensor, ...}
    for key, value in _tensors(path):
        try:
            name, part = original_key(key, gated)
        except KeyError:
            raise ValueError(
                f"{path}: unexpected key {key!r} for a DINOv3 ViT in the transformers "
                f"format"
            ) from None
        if part is None:
            if name == "mask_token":  # [1, 1, D] -> [1, D]
                value = value.reshape(1, dim)
            yield out(name, value)
            continue
        group = parts.setdefault(name, {})
        if part in group or name in seen:
            raise ValueError(f"{path}: two tensors for {key}")
        group[part] = value
        if not qkv_complete(name, group):
            continue
        group = parts.pop(name)
        if name.endswith(".bias"):  # the converter drops the (masked) key bias
            ref = next(iter(group.values()))
            group = {p: group.get(p, torch.zeros_like(ref)) for p in _QKV}
            mask = torch.tensor(
                [float(on) for on in (query_bias, key_bias, value_bias)]
            )
            yield out(name + "_mask", mask.repeat_interleave(dim))  # qkv.bias_mask
        yield out(name, torch.cat([group[p] for p in _QKV], dim=0))
        del group
    if parts:
        raise ValueError(
            f"{path}: incomplete q / k / v projections: "
            + ", ".join(f"{n} has {sorted(g)}" for n, g in sorted(parts.items()))
        )
    dtype = PERIODS_DTYPE.get(encoder_type, torch.float32)
    periods = rope_periods(config, dtype, encoder_type)
    yield out("rope_embed.periods", periods)
