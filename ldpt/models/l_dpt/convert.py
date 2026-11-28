"""
Training checkpoint -> L-DPT release file, and back.

A release file (``torch.save``, loadable with ``weights_only=True``) holds:
    format       "l-dpt-release/2"
    model        "L-DPT-v1m1"
    preset       preset name (--preset), or None (--config); the short names of
                 LEGACY_PRESETS are read as the current ones
    config       the LatentDPT kwargs without the training loss (a preset dict, or the
                 ``model`` of a config file, minus ``criterion``)
    state_dict   decoder keys only (no ``module.`` prefix, no frozen encoder, no loss
                 module), fp32
    provenance   source file name and sha256, epoch, best metric, conversion time;
                 for an average written by scripts/average_weights.py (its
                 SRC.avg.json beside SRC) also "averaged": the names and sha256 of
                 its inputs and their weights
Optimizer, scheduler and scaler state are dropped. A release file builds its model
inference-only (``criterion=False``), so loading it never builds the training loss
(the loss code is not needed). "l-dpt-release/1" files (also storing the
criterion) still load: the criterion is ignored. Both directions check the decoder keys
strictly: every key of the model built from ``config`` filled once, with its shape.

CLI:
    python -m ldpt.models.l_dpt.convert SRC DST --preset l-dpt_large_depth_zeroshot
    python -m ldpt.models.l_dpt.convert SRC DST --preset l-dpt_large_semseg_cocostuff
    python -m ldpt.models.l_dpt.convert SRC DST --config exp/hybrid/<run>/config.py

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from ..builder import build_model
from .presets import PRESETS, canonical_preset, get_preset
from .weights import PathLike, decoder_state, read_checkpoint, verified_sha256

__all__ = [
    "FORMAT",
    "FORMATS",
    "ConvertError",
    "build_meta",
    "check_decoder_state",
    "convert",
    "ensure_release",
    "inference_kwargs",
    "load_release",
    "release_kwargs",
]

FORMAT = "l-dpt-release/2"  # written; config without the criterion
# read: /1 also stores the training criterion in the config (dropped on load)
FORMATS = (FORMAT, "l-dpt-release/1")
MODEL = "L-DPT-v1m1"


class ConvertError(ValueError):
    """A checkpoint that does not match its L-DPT config exactly."""


def plain(value: Any) -> Any:
    """Config values as plain Python (dict / list / tuple / scalars) for torch.save."""
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(plain(v) for v in value)
    return value


def model_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    """LatentDPT kwargs from a model cfg (``type`` checked and removed)."""
    kwargs = plain(config)
    kind = kwargs.pop("type", MODEL)
    if kind != MODEL:
        raise ConvertError(f"config builds {kind!r}, not {MODEL!r}")
    return kwargs


def release_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The kwargs a release file stores: ``model_kwargs`` without ``criterion`` (the
    training loss; it has no state-dict keys and no effect on the prediction)."""
    kwargs = model_kwargs(config)
    kwargs.pop("criterion", None)
    return kwargs


def inference_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    """``release_kwargs`` built inference-only (``criterion=False``: no loss)."""
    return dict(release_kwargs(config), criterion=False)


def build_meta(config: Mapping[str, Any], **overrides: Any):
    """LatentDPT of ``config`` on the meta device (no memory, no init work)."""
    with torch.device("meta"):
        return build_model(dict(type=MODEL, **model_kwargs(config), **overrides))


def check_decoder_state(
    config: Mapping[str, Any], state: Mapping[str, Tensor], model=None
) -> None:
    """Strict: ``state`` fills every decoder key of the model of ``config`` (or of
    ``model``, already built from it) once, with the model's shapes, and holds nothing
    else. Without ``model`` it builds ``config`` inference-only (the loss has no
    state-dict keys)."""
    model = build_meta(inference_kwargs(config)) if model is None else model
    expected = {
        k: tuple(v.shape)
        for k, v in model.state_dict().items()
        if not k.startswith("encoder.")
    }
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shapes = [
        f"{k}: {tuple(state[k].shape)} vs {expected[k]}"
        for k in sorted(set(expected) & set(state))
        if tuple(state[k].shape) != expected[k]
    ]
    if missing or unexpected or shapes:
        problems = []
        if missing:
            problems.append(
                f"missing {missing[:8]}{' ...' if len(missing) > 8 else ''}"
            )
        if unexpected:
            problems.append(
                f"unexpected {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}"
            )
        if shapes:
            problems.append(f"shape mismatch {shapes[:8]}")
        raise ConvertError(
            "decoder state does not match the config (wrong preset / config?): "
            + "; ".join(problems)
        )


def convert(
    src: PathLike,
    dst: PathLike,
    *,
    preset: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the release file of the training checkpoint ``src`` to ``dst``; returns its
    provenance. Pass the run's ``preset`` or its model ``config``; the file stores its
    kwargs without the criterion (``release_kwargs``)."""
    if (preset is None) == (config is None):
        raise ValueError(
            "pass exactly one of preset= (e.g. 'l-dpt_large_depth_zeroshot') or "
            "config= (the run's model dict); a training checkpoint does not record "
            "its model kwargs"
        )
    kwargs = release_kwargs(get_preset(preset) if preset is not None else config)
    ckpt = read_checkpoint(src)
    if ckpt.get("format") in FORMATS:
        raise ConvertError(f"{src} is already a release file ({ckpt['format']})")
    state = decoder_state(ckpt)
    check_decoder_state(kwargs, state)
    provenance = dict(
        source=os.path.basename(os.fspath(src)),
        source_sha256=verified_sha256(src),
        epoch=ckpt.get("epoch"),
        best_metric_value=_scalar(ckpt.get("best_metric_value")),
        dropped=sorted(set(ckpt) - {"state_dict", "epoch", "best_metric_value"}),
        converted=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    )
    averaged = _averaged(src)
    if averaged is not None:
        provenance["averaged"] = averaged
    release = dict(
        format=FORMAT,
        model=MODEL,
        preset=preset,
        config=kwargs,
        state_dict={k: v.detach().contiguous().clone() for k, v in state.items()},
        provenance=provenance,
    )
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    torch.save(release, tmp)
    tmp.replace(dst)
    return provenance


def _norm(value: Any) -> Any:
    """torch.save keeps tuples, a config file may give lists: compare as lists."""
    if isinstance(value, Mapping):
        return {k: _norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    return value


def ensure_release(
    src: PathLike,
    dst: PathLike,
    *,
    preset: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """``convert`` unless ``dst`` already is the release file of ``src`` (provenance
    sha256, current FORMAT) with the same config (the preset's or ``config``'s
    ``release_kwargs``). Returns (status, provenance): status "up to date", "new" or
    "stale: <why>"; a written file is loaded back strictly."""
    if (preset is None) == (config is None):
        raise ValueError("pass exactly one of preset= or config=")
    kwargs = release_kwargs(get_preset(preset) if preset is not None else config)
    status = "new"
    if os.path.isfile(dst):
        old = read_checkpoint(dst)
        provenance = old.get("provenance") or {}
        stale = []
        if old.get("format") not in FORMATS:
            stale.append("not a release file")
        elif old.get("format") != FORMAT:
            stale.append(f"format {old.get('format')}, not {FORMAT}")
        if provenance.get("source_sha256") != verified_sha256(src):
            stale.append(f"made from {provenance.get('source')}, not this checkpoint")
        mine, theirs = _norm(kwargs), _norm(release_kwargs(old.get("config") or {}))
        differ = sorted(
            k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)
        )
        if differ:
            stale.append(f"config differs in {differ}")
        if canonical_preset(old.get("preset")) != preset:
            stale.append(f"preset {old.get('preset')!r}, not {preset!r}")
        del old
        if not stale:
            return "up to date", provenance
        status = "stale: " + "; ".join(stale)
    if preset is not None:
        provenance = convert(src, dst, preset=preset)
    else:
        provenance = convert(src, dst, config=kwargs)
    load_release(dst)  # strict back-load
    return status, provenance


def load_release(
    path: PathLike,
) -> Tuple[Dict[str, Any], Dict[str, Tensor], Dict[str, Any]]:
    """(config, decoder state, info) of a release file (any of FORMATS), checked
    strictly; ``config`` is its ``release_kwargs`` (a /1 file's criterion dropped)."""
    ckpt = read_checkpoint(path)
    if ckpt.get("format") not in FORMATS:
        raise ConvertError(
            f"{path} is not an L-DPT release file ({FORMAT}): convert the training "
            f"checkpoint first (python -m ldpt.models.l_dpt.convert SRC DST "
            f"--preset l-dpt_large_depth_zeroshot), or load it with its preset: "
            f"api.load('l-dpt_large_depth_zeroshot', weights=path)"
        )
    if ckpt.get("model") != MODEL:
        raise ConvertError(f"{path} holds {ckpt.get('model')!r}, not {MODEL!r}")
    config, state = release_kwargs(ckpt["config"]), decoder_state(ckpt)
    check_decoder_state(config, state)
    info = dict(
        preset=canonical_preset(ckpt.get("preset")),
        provenance=ckpt.get("provenance", {}),
        format=ckpt["format"],
    )
    return config, state, info


# the record scripts/average_weights.py writes beside its output
AVERAGE_SIDECAR = ".avg.json"


def _averaged(src: PathLike) -> Optional[Dict[str, Any]]:
    """The inputs of an averaged checkpoint (file names, sha256, weights) from its
    scripts/average_weights.py record (``<src>.avg.json``), None without one."""
    record = Path(os.fspath(src) + AVERAGE_SIDECAR)
    if not record.is_file():
        return None
    data = json.loads(record.read_text())
    return dict(
        sources=[
            dict(source=os.path.basename(s["path"]), sha256=s["sha256"])
            for s in data["sources"]
        ],
        weights=list(data["weights"]),
    )


def _scalar(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.item()
    return float(value) if hasattr(value, "__float__") else value


def _config_from_file(path: str) -> Dict[str, Any]:
    from ldpt.utils.config import Config  # the training config parser

    return model_kwargs(Config.fromfile(path).model)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ldpt.models.l_dpt.convert",
        description="L-DPT training checkpoint -> release file (decoder + config).",
    )
    parser.add_argument(
        "src", help="training checkpoint (model_best.pth / model_last.pth)"
    )
    parser.add_argument("dst", help="release file to write")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--preset",
        help=f"preset of the run (presets.PRESETS: {', '.join(PRESETS)})",
    )
    group.add_argument("--config", help="the run's config file (its `model` dict)")
    args = parser.parse_args(argv)
    config = _config_from_file(args.config) if args.config else None
    provenance = convert(args.src, args.dst, preset=args.preset, config=config)
    load_release(args.dst)  # strict back-load
    size = os.path.getsize(args.dst) / 1e6
    print(f"wrote {args.dst} ({size:.1f} MB): {provenance}")


if __name__ == "__main__":
    main()
