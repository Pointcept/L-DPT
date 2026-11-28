"""
Offline evaluation of saved checkpoints (scripts/evaluate.py).

Builds the config's model, loads the frozen encoder as a fresh run does (the config's
`checkpoint` with its CheckpointLoader keywords / replacement) and, for each given
checkpoint, the decoder: the files CheckpointSaver writes (model_last.pth,
model_best.pth, epoch_<N>.pth; decoder-only with skip_keywords, or full). Each is
evaluated by the config's evaluator hook (DepthEvaluator / SemSegEvaluator, same TTA
and metrics as during training) on the config's val sets, distributed when launched on
several GPUs (each rank a share of the val set, gathered to rank 0). With
average=True the uniform average of the checkpoints' weights is evaluated instead
(post-hoc weight averaging, e.g. of the last epochs' epoch_<N>.pth).
"""

import os
import re
from types import SimpleNamespace

import torch
import torch.utils.data

from ldpt.datasets import build_dataset
from ldpt.hooks.builder import HOOKS
from ldpt.hooks.evaluator import DefaultEvaluator, pareto_front
from ldpt.hooks.misc import CheckpointLoader, load_checkpoint, merge_weights
from ldpt.models import build_model
from ldpt.utils import comm
from ldpt.utils.device import get_device
from ldpt.utils.logger import get_root_logger


def find_hook(cfg, base):
    """Config of the first hook in cfg.hooks whose class is base or a subclass."""
    for hook in cfg.hooks:
        cls = HOOKS.get(hook["type"])
        if cls is not None and issubclass(cls, base):
            return hook
    return None


def build_evaluator(cfg):
    """The config's evaluator hook, without metrics file and evaluation_only exit."""
    hook = find_hook(cfg, DefaultEvaluator)
    if hook is None:
        raise ValueError("the config has no evaluator hook (DepthEvaluator, ...)")
    return HOOKS.build(dict(hook, evaluation_only=False, metrics_file=None))


def encoder_sources(cfg):
    """(paths, keywords, replacements) the encoder is loaded from.

    The entries of cfg.checkpoint that the config's CheckpointLoader maps into
    module.encoder. (VOC: the DINOv3 file, not the COCO-Stuff decoder); all entries
    if none does.
    """
    ckpts = cfg.checkpoint
    if ckpts is None:
        raise ValueError("the config has no checkpoint to load the encoder from")
    ckpts = list(ckpts) if isinstance(ckpts, (list, tuple)) else [ckpts]
    loader = find_hook(cfg, CheckpointLoader) or {}

    def per_ckpt(x, default):
        x = default if x is None else x
        return list(x) if isinstance(x, (list, tuple)) else [x] * len(ckpts)

    keywords = per_ckpt(loader.get("keywords"), "module.")
    replacements = per_ckpt(loader.get("replacement"), None)
    replacements = [kw if rp is None else rp for kw, rp in zip(keywords, replacements)]
    entries = list(zip(ckpts, keywords, replacements))
    picked = [e for e in entries if e[2].startswith("module.encoder.")] or entries
    return tuple(list(x) for x in zip(*picked))


def load_encoder(model, cfg, encoder=None, logger=None):
    """Load the frozen encoder: `encoder` (a DINOv3 file) or the config's sources.

    Only keys under encoder. are loaded. A missing encoder parameter is an error, a
    missing buffer a warning (kept at init, as in a fresh run).
    """
    if encoder is not None:
        paths, keywords, replacements = [encoder], ["module."], ["module.encoder."]
    else:
        paths, keywords, replacements = encoder_sources(cfg)
    weight, _ = merge_weights(
        paths, keywords, replacements, strip_module=True, logger=logger
    )
    weight = {k: v for k, v in weight.items() if k.startswith("encoder.")}
    if not weight:
        raise RuntimeError(f"no encoder keys in {paths}")
    info = model.load_state_dict(weight, strict=False)
    missing = [k for k in info.missing_keys if k.startswith("encoder.")]
    params = {n for n, _ in model.named_parameters()}
    if any(k in params for k in missing):
        raise RuntimeError(f"encoder parameters missing in {paths}: {missing}")
    if missing and logger is not None:
        logger.warning(f"Encoder buffers kept at init: {missing}")
    return paths


def read_state(path):
    """(state dict without the DDP "module." prefix, epoch or None) of a file."""
    checkpoint = load_checkpoint(path)
    state = checkpoint.get("state_dict", checkpoint)
    state = {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state.items()
    }
    return state, checkpoint.get("epoch")


def average_states(paths):
    """Uniform average of the files' weights (fp32 sum, stored dtype); non-float
    tensors must be equal in every file, except BatchNorm's num_batches_tracked (model
    tap_bn): the last file's (a counter, unused at the default momentum). Running
    means / variances are averaged like the weights."""
    total, dtypes = None, None
    for path in paths:
        state, _ = read_state(path)
        if total is None:
            dtypes = {k: v.dtype for k, v in state.items()}
            total = {
                k: v.float().clone() if v.is_floating_point() else v.clone()
                for k, v in state.items()
            }
            continue
        if state.keys() != total.keys():
            raise RuntimeError(f"{path}: keys differ from {paths[0]}")
        for k, v in state.items():
            if v.is_floating_point():
                total[k] += v.float()
            elif k.endswith("num_batches_tracked"):
                total[k] = v.clone()
            elif not torch.equal(v, total[k]):
                raise RuntimeError(f"{path}: non-float {k} differs from {paths[0]}")
    return {
        k: (v / len(paths)).to(dtypes[k]) if v.is_floating_point() else v
        for k, v in total.items()
    }


def load_decoder(model, path, state=None):
    """Load a CheckpointSaver file (or a plain state dict) into the model.

    Every non-encoder key of the model must be in the file and every key of the file
    in the model; encoder keys in the file (a full checkpoint) are loaded too.
    state: the weights to load instead of the file's (path only names them).
    Returns the file's epoch (None if it has none).
    """
    epoch = None
    if state is None:
        state, epoch = read_state(path)
    info = model.load_state_dict(state, strict=False)
    missing = [k for k in info.missing_keys if not k.startswith("encoder.")]
    if missing or info.unexpected_keys:
        raise RuntimeError(
            f"{path}: missing decoder keys {missing}, unexpected keys "
            f"{info.unexpected_keys}"
        )
    return epoch


def build_val_loader(cfg):
    dataset = build_dataset(cfg.data.val)
    sampler = None
    if comm.get_world_size() > 1:
        # padded with repeated samples to equal shares; the evaluators deduplicate
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=False
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size_val_per_gpu,
        shuffle=False,
        num_workers=cfg.num_workers_per_gpu,
        pin_memory=get_device().type == "cuda",
        sampler=sampler,
    )


def natural_key(path):
    """epoch_2.pth before epoch_10.pth."""
    parts = re.split(r"(\d+)", os.path.basename(path))
    return os.path.dirname(path), [int(p) if p.isdigit() else p for p in parts]


def evaluate_checkpoints(
    cfg, checkpoints, encoder=None, per_image=False, average=False, logger=None
):
    """Evaluate each checkpoint in turn (average: their mean weights, once).

    Returns (records, evaluator, encoder paths); records only on rank 0, else [].

    cfg: resolved config after default_setup (batch_size_val_per_gpu, ...).
    A record is the evaluator's metrics history line (score, counted, groups, ...)
    with "checkpoint" and the file's "epoch"; per_image adds the per-image metrics
    (DepthEvaluator); the average's record has "averaged" (the files) instead.
    """
    logger = logger or get_root_logger()
    cfg.evaluate = True
    model = build_model(cfg.model)
    encoder_paths = load_encoder(model, cfg, encoder, logger)
    model = model.to(get_device())
    evaluator = build_evaluator(cfg)
    runner = SimpleNamespace(
        cfg=cfg,
        model=model,
        val_loader=build_val_loader(cfg),
        comm_info={},
        logger=logger,
        writer=None,
        epoch=0,
    )
    evaluator.trainer = runner
    items = [(str(path), None) for path in checkpoints]
    if average:
        logger.info(f"Averaging the weights of {len(checkpoints)} checkpoints")
        name = f"average of {len(checkpoints)}"
        items = [(name, average_states(checkpoints))]
    records = []
    for i, (path, state) in enumerate(items):
        logger.info(f"=> [{i + 1}/{len(items)}] {path}")
        epoch = load_decoder(model, path, state)
        runner.epoch = (epoch if isinstance(epoch, int) else i + 1) - 1
        evaluator.after_epoch()
        if comm.is_main_process():
            record = dict(evaluator.results, checkpoint=path, epoch=epoch)
            if average:
                record["averaged"] = [str(p) for p in checkpoints]
            if per_image:
                record["per_image"] = dict(getattr(evaluator, "per_image", {}))
            records.append(record)
    return records, evaluator, encoder_paths


def summarize(records, metric, greater):
    """Best checkpoint by the score, per group (diagnostics), and the Pareto front
    of `metric` over the counted groups."""
    pick = max if greater else min

    def ref(r, **extra):
        return dict(checkpoint=r["checkpoint"], epoch=r["epoch"], **extra)

    best = pick(records, key=lambda r: r["score"])
    per_group = {}
    for group in records[0]["groups"]:
        have = [r for r in records if group in r["groups"]]
        r = pick(have, key=lambda r: r["groups"][group][metric])
        per_group[group] = ref(r, value=r["groups"][group][metric])
    counted = records[0]["counted"]
    have = [r for r in records if all(g in r["groups"] for g in counted)]
    rows = [{g: r["groups"][g][metric] for g in counted} for r in have]
    front = [ref(have[i]) for i in pareto_front(rows, counted, greater)]
    return dict(
        best_score=ref(best, score=best["score"]),
        best_per_group=per_group,
        pareto_front=front,
    )


def format_table(records, metric):
    """One row per checkpoint: epoch, score, `metric` per group."""
    groups = list(records[0]["groups"])
    name_w = max(len(os.path.basename(r["checkpoint"])) for r in records)
    col_w = max(8, *(len(g) for g in groups))
    head = f"{'checkpoint':<{name_w}}  {'epoch':>5}  {'score':>8}  " + "  ".join(
        f"{g:>{col_w}}" for g in groups
    )
    lines = [f"{metric} per group (score: the evaluator's selection score)", head]
    for r in records:
        cells = [
            f"{r['groups'][g][metric]:>{col_w}.4f}" if g in r["groups"] else " " * col_w
            for g in groups
        ]
        epoch = "-" if r["epoch"] is None else r["epoch"]
        lines.append(
            f"{os.path.basename(r['checkpoint']):<{name_w}}  {epoch:>5}  "
            f"{r['score']:>8.4f}  " + "  ".join(cells)
        )
    return "\n".join(lines)
