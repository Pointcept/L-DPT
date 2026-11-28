"""
Weighted average of decoder checkpoints into one CheckpointSaver-style file (CPU).

L-DPT Large (mixed) is the uniform average of two epoch files: epoch 4 of
configs/hybrid/depth-l-dpt-v1m1-1-large-mixed.py and the end of the one-epoch anneal
started from it (configs/hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal.py):

    python scripts/average_weights.py \\
        --inputs exp/hybrid/depth-l-dpt-v1m1-1-large-mixed/model/epoch_4.pth \\
                 exp/hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal/model/epoch_1.pth \\
        --output exp/hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal/model/average.pth

Every floating tensor of the state dicts becomes sum_i w_i x_i (summed in float64,
stored in the --keep-from file's dtype; default weights uniform); every non-float
tensor, the key order and the other top-level entries ("epoch", ...) come from the
--keep-from file (default: the last input). Before anything is written the inputs
must be decoder-only state dicts (no encoder.* key, a DDP "module." prefix aside) with
identical keys, shapes and dtypes; otherwise it exits non-zero and writes nothing.
OUTPUT is written atomically, with OUTPUT.avg.json beside it (sources with size and
sha256, weights, tensor counts, ||x_1 - x_0|| / ||x_0||). The output is a training-
style file: evaluate it with scripts/evaluate.py or convert it with
`python -m ldpt.models.l_dpt.convert OUTPUT <release file> --preset
l-dpt_large_depth_mixed` (the release file's provenance lists the inputs of
OUTPUT.avg.json).
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

SIDECAR = ".avg.json"


def sha256_of(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def state_of(ckpt: Mapping[str, Any], name: str) -> Mapping[str, torch.Tensor]:
    state = ckpt.get("state_dict") if isinstance(ckpt, Mapping) else None
    if not isinstance(state, Mapping):
        raise SystemExit(f"{name}: not a CheckpointSaver file (no 'state_dict' entry)")
    return state


def check_states(states: Sequence[Mapping[str, torch.Tensor]], names) -> List[str]:
    """Problems that forbid the average (empty: none)."""
    problems = []
    for name, state in zip(names, states):
        bare = [k[len("module.") :] if k.startswith("module.") else k for k in state]
        enc = [k for k in bare if k.startswith("encoder.")]
        if enc:
            problems.append(f"{name}: not decoder-only ({len(enc)} encoder keys)")
        bad = [k for k, v in state.items() if not isinstance(v, torch.Tensor)]
        if bad:
            problems.append(f"{name}: non-tensor entries {bad[:5]}")
    ref, ref_name = states[0], names[0]
    for name, state in zip(names[1:], states[1:]):
        only_ref = sorted(set(ref) - set(state))
        only_this = sorted(set(state) - set(ref))
        if only_ref or only_this:
            problems.append(
                f"keys differ: only in {ref_name} {only_ref[:5]} ({len(only_ref)}), "
                f"only in {name} {only_this[:5]} ({len(only_this)})"
            )
            continue
        shapes = [
            f"{k}: {tuple(ref[k].shape)} vs {tuple(v.shape)}"
            for k, v in state.items()
            if tuple(v.shape) != tuple(ref[k].shape)
        ]
        if shapes:
            problems.append(f"{name}: shapes differ ({len(shapes)}): {shapes[:5]}")
        dtypes = [
            f"{k}: {ref[k].dtype} vs {v.dtype}"
            for k, v in state.items()
            if v.dtype != ref[k].dtype
        ]
        if dtypes:
            problems.append(f"{name}: dtypes differ ({len(dtypes)}): {dtypes[:5]}")
    return problems


def average(
    states: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float], keep: int
):
    """(averaged state dict in the key order of states[keep], stats)."""
    base = states[keep]
    out: Dict[str, torch.Tensor] = {}
    n_float = n_other = 0
    other_differ = []
    diff_sq = ref_sq = 0.0
    for k, v in base.items():
        if v.is_floating_point():
            acc = torch.zeros(v.shape, dtype=torch.float64)
            for w, state in zip(weights, states):
                acc.add_(state[k].to(torch.float64), alpha=float(w))
            out[k] = acc.to(v.dtype).contiguous()
            n_float += 1
            x0 = states[0][k].to(torch.float64)
            diff_sq += (states[1][k].to(torch.float64) - x0).pow(2).sum().item()
            ref_sq += x0.pow(2).sum().item()
        else:
            out[k] = v.clone()
            n_other += 1
            if any(not torch.equal(state[k], v) for state in states):
                other_differ.append(k)
    stats = dict(
        tensors=len(out),
        float_tensors=n_float,
        nonfloat_tensors=n_other,
        nonfloat_differ=other_differ,
        rel_dist_1_vs_0=math.sqrt(diff_sq / ref_sq) if ref_sq > 0 else None,
    )
    return out, stats


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--inputs", nargs="+", required=True, help="checkpoint files")
    ap.add_argument("--output", required=True, help="the averaged file")
    ap.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="one non-negative weight per input, summing to 1 (default: uniform)",
    )
    ap.add_argument(
        "--keep-from",
        type=int,
        default=-1,
        help="input index giving non-float tensors, dtypes and top-level entries "
        "(default -1: the last)",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = [Path(p) for p in args.inputs]
    if len(paths) < 2:
        raise SystemExit("--inputs: two files or more")
    weights = args.weights or [1.0 / len(paths)] * len(paths)
    if len(weights) != len(paths):
        raise SystemExit(f"--weights: {len(weights)} values for {len(paths)} inputs")
    if abs(sum(weights) - 1.0) > 1e-9 or min(weights) < 0:
        raise SystemExit(f"--weights {weights}: non-negative, summing to 1")
    keep = args.keep_from % len(paths)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"no such input: {missing}")

    ckpts = [
        torch.load(p, map_location="cpu", mmap=True, weights_only=True) for p in paths
    ]
    names = [str(p) for p in paths]
    states = [state_of(c, n) for c, n in zip(ckpts, names)]
    problems = check_states(states, names)
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        raise SystemExit("inputs cannot be averaged; nothing written")
    state, stats = average(states, weights, keep)
    result = {k: v for k, v in ckpts[keep].items() if k != "state_dict"}
    result["state_dict"] = state
    result = {k: result[k] for k in ckpts[keep]}  # the key order of --keep-from

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    torch.save(result, tmp)
    os.replace(tmp, out)
    record = dict(
        output=str(out),
        sources=[
            dict(path=str(p), size=p.stat().st_size, sha256=sha256_of(p)) for p in paths
        ],
        weights=list(weights),
        keep_from=keep,
        entries={
            k: v
            for k, v in result.items()
            if k != "state_dict" and isinstance(v, (int, float, str, type(None)))
        },
        **stats,
    )
    sidecar = out.with_name(out.name + SIDECAR)
    tmp = sidecar.with_name(sidecar.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, sidecar)
    print(
        f"wrote {out}: {stats['float_tensors']} float tensors averaged with weights "
        f"{list(weights)}, {stats['nonfloat_tensors']} non-float from input {keep}; "
        f"||x1 - x0|| / ||x0|| = {stats['rel_dist_1_vs_0']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
