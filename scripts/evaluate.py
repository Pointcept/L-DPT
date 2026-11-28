"""
Evaluate saved checkpoints on a config's val sets and write a JSON table.

The config's evaluator hook (DepthEvaluator / SemSegEvaluator) runs with its TTA and
metrics as during training; the encoder comes from the config's `checkpoint` (the
DINOv3 file), the decoder from each checkpoint (CheckpointSaver files: epoch_<N>.pth
from keep_epoch_weights, model_best.pth, model_last.pth; decoder-only or full).

    # every kept epoch of a run, 8 GPUs (as train.py: one process per GPU)
    RUN=exp/hybrid/depth-l-dpt-v1m1-0-large-zeroshot
    python scripts/evaluate.py --config-file $RUN/config.py \\
        --checkpoints $RUN/model/epoch_*.pth --num-gpus 8 --output $RUN/eval.json

    # a release file (convert.py output) with the recipe's evaluation
    python scripts/evaluate.py \\
        --config-file configs/hybrid/depth-l-dpt-v1m1-1-large-mixed.py \\
        --checkpoints l-dpt_large_depth_mixed.pth --num-gpus 8 \\
        --output eval_large_depth_mixed.json

    # post-hoc weight averaging: the mean of the last 5 epochs' decoder weights
    python scripts/evaluate.py --config-file $RUN/config.py \\
        --checkpoints $RUN/model/epoch_2[1-5].pth --average \\
        --num-gpus 8 --output $RUN/eval_avg21-25.json

    # --options overrides config keys, e.g. another seed of the RANSAC fit
    python scripts/evaluate.py \\
        --config-file configs/hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py \\
        --checkpoints l-dpt_large_depth_zeroshot.pth --options hooks.4.ransac_seed=1

Output: {"config", "options", "evaluator", "encoder", "results": [one record per
checkpoint: checkpoint, epoch, score, counted, groups (every metric of every group),
optional per_image; --average: one record with "averaged"], "summary": {best_score,
best_per_group, pareto_front}}.
best_per_group and the Pareto front are diagnostics; report one checkpoint for all
datasets.

The model options of the config must match the checkpoint's: model.attn_fp32 /
model.block_fp32 (fp32 islands) and model.dec_qk_bound (the logit bound: without it the
same keys load and the logits are unbounded) act in eval too; model.dec_conv_mixer
changes the keys (the strict key check fails without it). The recipes and a run's
config.py carry them. --options model.dec_qk_bound=None evaluates a bound model
without the bound.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpt.engines.default import default_argument_parser, default_setup  # noqa: E402
from ldpt.engines.evaluate import (  # noqa: E402
    evaluate_checkpoints,
    format_table,
    natural_key,
    summarize,
)
from ldpt.engines.launch import launch  # noqa: E402
from ldpt.utils import comm  # noqa: E402
from ldpt.utils.config import Config  # noqa: E402
from ldpt.utils.logger import get_root_logger  # noqa: E402


def parse_args(argv=None):
    parser = default_argument_parser(epilog=__doc__)
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, help="decoder checkpoint files"
    )
    parser.add_argument(
        "--output", default=None, help="JSON file (default: eval.json beside the first)"
    )
    parser.add_argument(
        "--encoder",
        default=None,
        help="DINOv3 file for the encoder (default: the config's checkpoint)",
    )
    parser.add_argument(
        "--per-image", action="store_true", help="add per-image metrics (depth)"
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help="evaluate the uniform average of the checkpoints' weights instead",
    )
    parser.add_argument("--log-file", default=None, help="also log to this file")
    return parser.parse_args(argv)


def main_worker(cfg, args):
    cfg = default_setup(cfg)
    logger = get_root_logger(log_file=args.log_file)
    checkpoints = sorted(args.checkpoints, key=natural_key)
    records, evaluator, encoder = evaluate_checkpoints(
        cfg, checkpoints, args.encoder, args.per_image, args.average, logger
    )
    if not comm.is_main_process():
        return
    metric, greater = evaluator.best_metric, evaluator.best_metric_greater
    out = dict(
        config=args.config_file,
        options=args.options,
        evaluator=dict(
            type=type(evaluator).__name__,
            tta=evaluator.tta,
            best_metric=metric,
            best_metric_mean=getattr(evaluator, "best_metric_mean", None),
        ),
        encoder=encoder,
        world_size=comm.get_world_size(),
        results=records,
        summary=summarize(records, metric, greater),
    )
    if hasattr(evaluator, "protocol"):  # DepthEvaluator: alignment, crop, TTA
        out["evaluator"]["protocol"] = {
            g: evaluator.protocol(g) for g in records[0]["groups"]
        }
    output = args.output or os.path.join(os.path.dirname(checkpoints[0]), "eval.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=1)
    logger.info(format_table(records, metric))
    best = out["summary"]["best_score"]
    logger.info(f"Best score: {best['score']:.4f} ({best['checkpoint']})")
    logger.info(f"Wrote {output}")


def main(argv=None):
    args = parse_args(argv)
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg, args),
    )


if __name__ == "__main__":
    main()
