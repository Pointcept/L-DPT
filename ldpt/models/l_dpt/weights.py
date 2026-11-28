"""
Weights of L-DPT: find or download a decoder checkpoint, check it, read it.

- Decoder: ``fetch(preset)`` returns a local, sha256-checked copy of the preset's pinned
  file: the path in the preset's environment variable
  (``LDPT_CKPT_LARGE_DEPTH_ZEROSHOT``) when set, else
  ``huggingface_hub.hf_hub_download`` at the pinned revision (HF_HOME, HF_HUB_OFFLINE
  work as usual). Training checkpoints (``module.`` prefix, optimizer state) and
  release files (convert.py) both read through ``decoder_state``.
- Encoder: DINOv3 weights are not redistributed. ``encoder_weights`` takes a local file
  (argument, else ``LDPT_DINOV3``) and checks the sha256 prefix of its official name, or
  a folder in the transformers format (a download of Meta's gated Hugging Face repo, e.g.
  facebook/dinov3-vit7b16-pretrain-lvd1689m) and checks its config.json;
  ``encoder_state`` reads either into the original key layout
  (``ldpt.models.dinov3.hf_format``).

A file is hashed once; the result is remembered by (path, size, mtime) in
``~/.cache/l_dpt/verified.json`` (``LDPT_CACHE`` overrides the folder). Checkpoints are
read with ``torch.load(mmap=True, weights_only=True)``: nothing is copied into memory
until it is used.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
from torch import Tensor

from ..dinov3 import hf_format
from .presets import DINOV3_SHA256_PREFIX, WeightSource, weight_source

__all__ = [
    "decoder_state",
    "encoder_state",
    "encoder_weights",
    "fetch",
    "read_checkpoint",
    "sha256_of",
]

log = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]
ENCODER_ENV = "LDPT_DINOV3"
# DINOv3 checkpoint keys the inference encoder does not have (training-only norm)
ENCODER_DROPPED = ("local_cls_norm.",)


def _cache_dir() -> Path:
    root = os.environ.get("LDPT_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "l_dpt"
    )
    return Path(root)


def sha256_of(path: PathLike) -> str:
    """The sha256 of a file (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_sha256(path: PathLike) -> str:
    """sha256 of ``path``, hashed once per (path, size, mtime) and cached."""
    path = Path(path).resolve()
    stat = path.stat()
    record = _cache_dir() / "verified.json"
    try:
        known = json.loads(record.read_text())
    except (OSError, ValueError):
        known = {}
    entry = known.get(str(path))
    if entry and entry[:2] == [stat.st_size, stat.st_mtime_ns]:
        return entry[2]
    log.info("hashing %s (%.1f GB, once)", path, stat.st_size / 1e9)
    digest = sha256_of(path)
    known[str(path)] = [stat.st_size, stat.st_mtime_ns, digest]
    try:
        record.parent.mkdir(parents=True, exist_ok=True)
        tmp = record.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(known, indent=1))
        tmp.replace(record)
    except OSError:  # a read-only cache only costs a re-hash next time
        pass
    return digest


def fetch(name: str, *, revision: Optional[str] = None, verify: bool = True) -> Path:
    """A local, checked copy of the released decoder weights of preset ``name``.

    The file named by the preset's environment variable wins over a download; either way
    its sha256 must match the pinned one (``verify=False`` skips that). ``revision``
    downloads another revision (then the pinned sha256 is not checked). A slot that is
    not published yet (``WeightSource.published``) has nothing to download or check:
    its environment variable with ``verify=False``, or api.load's weights=, is the way.
    """
    source = weight_source(name)
    if source is None:
        raise ValueError(
            f"preset {name!r} has no released weights; pass weights=<path> to "
            f"api.load, or train it (configs/hybrid/*.py)"
        )
    pinned = revision is None or revision == source.revision
    local = os.environ.get(source.env)
    if not source.published and (not local or verify):
        raise ValueError(
            f"the weights of preset {name!r} are not published yet (no pinned file or "
            f"sha256): pass weights=<release file> to api.load, or set {source.env} "
            f"and load with verify=False"
        )
    if local:
        path = Path(local).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{source.env}={local} is not a file")
    else:
        from huggingface_hub import hf_hub_download  # only when weights are fetched

        path = Path(
            hf_hub_download(
                repo_id=source.repo,
                filename=source.file,
                revision=source.revision if pinned else revision,
            )
        )
    if verify and pinned:
        _check_pinned(path, source)
    return path


def _check_pinned(path: Path, source: WeightSource) -> None:
    size = path.stat().st_size
    if size != source.size or verified_sha256(path) != source.sha256:
        raise RuntimeError(
            f"{path} is not the pinned {source.url} (sha256 {source.sha256[:12]}..., "
            f"{source.size} bytes); unset {source.env} or delete the download and load "
            f"again, or pass weights=<path> to load another file"
        )


def encoder_weights(
    encoder_type: str, path: Optional[PathLike] = None, *, verify: bool = True
) -> Path:
    """The local DINOv3 weights for ``encoder_type`` (``path``, else ``$LDPT_DINOV3``):
    an official file, checked against the sha256 prefix of its official name when that
    is known, or a transformers-format folder (hf_format), whose config.json must be
    that backbone (``verify``; the keys and shapes are checked by the strict load)."""
    path = path or os.environ.get(ENCODER_ENV)
    if not path:
        raise ValueError(
            f"{encoder_type} weights are not redistributed: pass encoder_weights=<path> "
            f"or set {ENCODER_ENV} to the official DINOv3 file (e.g. "
            f"dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth from Meta's DINOv3 release) "
            f"or to a download of its Hugging Face repo (transformers format, e.g. "
            f"facebook/dinov3-vit7b16-pretrain-lvd1689m)"
        )
    path = Path(path).expanduser()
    if path.is_dir():
        if not hf_format.is_hf_folder(path):
            raise FileNotFoundError(
                f"{path} is a folder without config.json + model*.safetensors (the "
                f"transformers format of a DINOv3 Hugging Face repo)"
            )
        if verify:
            hf_format.check_config(
                hf_format.read_config(path), encoder_type, str(path / "config.json")
            )
        return path
    if not path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {path}")
    prefix = DINOV3_SHA256_PREFIX.get(encoder_type)
    if verify and prefix is not None:
        digest = verified_sha256(path)
        if not digest.startswith(prefix):
            raise RuntimeError(
                f"{path} is not the official {encoder_type} file: sha256 {digest[:8]}..., "
                f"expected {prefix}... (pass verify=False to use it anyway)"
            )
    return path


def read_checkpoint(path: PathLike) -> Dict[str, Any]:
    """A checkpoint as a dict, memory-mapped (``torch.load(mmap=True, weights_only=True)``)."""
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def state_dict_of(ckpt: Mapping[str, Any]) -> Mapping[str, Tensor]:
    """The state dict inside a checkpoint (``"state_dict"``, ``"model"``, or the dict)."""
    for key in ("state_dict", "model"):
        value = ckpt.get(key)
        if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
            return value
    return ckpt


def decoder_state(ckpt: Mapping[str, Any]) -> Dict[str, Tensor]:
    """Decoder keys of a training or release checkpoint: DDP ``module.`` prefix removed,
    ``encoder.*`` (frozen DINOv3, saved by some runs) and ``criterion.*`` (the state
    of a training loss module, if it has one; an inference model has no loss)
    dropped."""
    state = {}
    for k, v in state_dict_of(ckpt).items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if not k.startswith(("encoder.", "criterion.")):
            state[k] = v
    return state


def encoder_state(
    path: PathLike,
    *,
    prefix: str = "encoder.",
    dtypes: Optional[Mapping[str, Any]] = None,
    device: Union[str, torch.device, None] = None,
    encoder_type: Optional[str] = None,
) -> Dict[str, Tensor]:
    """An official DINOv3 file, or a transformers-format folder (read into the original
    layout by hf_format; ``encoder_type`` sets the dtype of its rebuilt RoPE periods),
    as ``encoder.*`` keys (training-only keys dropped).

    ``dtypes`` maps a key (without ``prefix``) to the dtype to store it in and
    ``device`` moves each tensor; both happen tensor by tensor from the memory map, so
    a 7B fp32 file is never materialised as a whole.
    """
    if Path(path).is_dir():
        raw = hf_format.read_state(
            path, encoder_type=encoder_type, dtypes=dtypes, device=device
        )
        return {
            prefix + k: v for k, v in raw.items() if not k.startswith(ENCODER_DROPPED)
        }
    state = {}
    for k, v in state_dict_of(read_checkpoint(path)).items():
        if k.startswith(ENCODER_DROPPED):
            continue
        dtype = (dtypes or {}).get(k)
        if dtype is not None or device is not None:
            v = v.to(device=device, dtype=dtype)
        state[prefix + k] = v
    return state
