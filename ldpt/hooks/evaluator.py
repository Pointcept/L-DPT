"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import json
import math
from collections import defaultdict
from functools import lru_cache, partial
import numpy as np
import wandb
import torch
import torch.nn.functional as F

import ldpt.utils.comm as comm
from ldpt.models.utils.depth import (
    compute_scale_and_shift,
    free_lsq_scale_and_shift,
    nonneg_lsq_scale_and_shift,
    ransac_scale_and_shift,
    roe_eval_scale_and_shift,
    trimmed_scale_and_shift,
)
from ldpt.utils.device import get_device

from .default import HookBase
from .builder import HOOKS


# test-time augmentation ops on the image and their inverse on the prediction
TTA_PRE_PROCESS = {
    "flip": partial(torch.flip, dims=[-1]),
    "upsample": partial(
        F.interpolate, mode="bilinear", align_corners=False, scale_factor=2
    ),
}
TTA_POST_PROCESS = {
    "flip": partial(torch.flip, dims=[-1]),
    "upsample": partial(
        F.interpolate, mode="bilinear", align_corners=False, scale_factor=0.5
    ),
}


# multi-scale op "scale@<r>" (SemSegEvaluator only): the image resized by r
SCALE_OP = "scale@"


def check_choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")


def scale_of(op):
    """r of a "scale@<r>" TTA op (finite, > 0); None for any other op."""
    if not (isinstance(op, str) and op.startswith(SCALE_OP)):
        return None
    try:
        r = float(op[len(SCALE_OP) :])
    except ValueError:
        r = math.nan
    if not (math.isfinite(r) and r > 0):
        raise ValueError(f"tta op {op!r}: expected {SCALE_OP}<r> with a number r > 0")
    return r


def scale_image(image, r):
    """[B, C, H, W] -> [B, C, int(H r + 0.5), int(W r + 0.5)] (bilinear); unchanged
    (the same tensor) when the size does not change, e.g. r = 1."""
    h, w = image.shape[-2:]
    size = (max(1, int(h * r + 0.5)), max(1, int(w * r + 0.5)))
    if size == (h, w):
        return image
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def tta_predictions(model, input_dict, key, tta=None, batched=True):
    """Prediction of the image and of each TTA variant, mapped back to the image.

    Args:
        model: called as model(input_dict with "image" replaced) -> dict with `key`.
        tta: None or op lists (TTA_PRE_PROCESS keys or "scale@<r>"), e.g. [["flip"]],
            applied in list order. "scale@<r>" resizes the image by r (bilinear,
            int(H r + 0.5) x int(W r + 0.5); r = 1: the image itself) and is not
            undone here: that prediction keeps the scaled size (SemSegEvaluator
            resizes each softmax to the label size); the other ops of the entry are
            undone as usual (["scale@0.9", "flip"]: flipped back, at 0.9 x the size).
        batched: True: one forward per image shape, the images of that shape
            concatenated along the batch (flip: image + flipped image in one forward;
            "upsample" and each scale form their own batch). Same math as one
            forward each, up to the rounding of other batch sizes (the frozen encoder
            runs under bf16 autocast). False: one forward per variant.

    Returns:
        [prediction of the image] + one prediction per tta entry, mapped back.
    """
    image = input_dict["image"]
    images = [image]
    for ops in tta or []:
        x = image
        for op in ops:
            r = scale_of(op)
            x = TTA_PRE_PROCESS[op](x) if r is None else scale_image(x, r)
        images.append(x)
    if batched:
        by_shape = {}
        for i, x in enumerate(images):
            by_shape.setdefault(tuple(x.shape), []).append(i)
        groups = list(by_shape.values())
    else:
        groups = [[i] for i in range(len(images))]
    preds = [None] * len(images)
    for group in groups:
        batch = torch.cat([images[i] for i in group])
        out = model(dict(input_dict, image=batch))[key]
        for i, pred in zip(group, out.split(image.shape[0])):
            preds[i] = pred
    for i, ops in enumerate(tta or [], start=1):
        for op in reversed(ops):
            if scale_of(op) is None:
                preds[i] = TTA_POST_PROCESS[op](preds[i])
    return preds


def pareto_front(rows, groups, greater):
    """Indices of the rows that no other row beats on every group.

    rows: list of {group: value}; greater: True if higher is better. A row is
    dominated by one that is at least as good in every group and better in one.
    """
    sign = -1.0 if greater else 1.0  # compare as "lower is better"
    vals = [[sign * row[g] for g in groups] for row in rows]
    front = []
    for i, a in enumerate(vals):
        dominated = any(
            all(y <= x for x, y in zip(a, b)) and any(y < x for x, y in zip(a, b))
            for j, b in enumerate(vals)
            if j != i
        )
        if not dominated:
            front.append(i)
    return front


def read_metrics_history(path):
    """metrics.jsonl -> {epoch: record}, ordered by epoch. An epoch evaluated again
    (after resuming from an earlier checkpoint) keeps its last line."""
    history = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                history[record["epoch"]] = record
    return dict(sorted(history.items()))


@HOOKS.register_module()
class DefaultEvaluator(HookBase):
    """Evaluates after every epoch. Subclasses report the metric that picks
    model_best.pth in comm_info: current_metric_value, current_metric_name and
    current_metric_greater (True: higher is better; CheckpointSaver compares).

    Rank 0 also keeps every evaluation (record()): one JSON line per evaluation in
    <save_path>/<metrics_file> (epoch, the selection score, every metric of every
    group), and logs the best epoch per group so far (diagnostics: model_best.pth
    follows the score only). A resumed run reads the file back and appends; a fresh
    run starts it anew (as train.log). metrics_file=None: no file.
    """

    def __init__(self, evaluation_only=False, metrics_file="metrics.jsonl"):
        super().__init__()
        self.evaluation_only = evaluation_only
        self.metrics_file = metrics_file
        self.history = {}  # epoch -> record (rank 0)
        self.results = None  # record of the last evaluation (rank 0)

    def metrics_path(self):
        save_path = getattr(self.trainer.cfg, "save_path", None)
        if not self.metrics_file or save_path is None:
            return None
        return os.path.join(save_path, self.metrics_file)

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")
        self.history = {}
        path = self.metrics_path()
        if path is None or not comm.is_main_process() or not os.path.isfile(path):
            return
        if self.trainer.cfg.resume:
            self.history = read_metrics_history(path)
            self.trainer.logger.info(
                f"Read {len(self.history)} evaluations (epochs "
                f"{list(self.history)}) from {path}"
            )
        else:
            os.remove(path)  # fresh run

    def record(self, groups, score, counted, **extra):
        """Rank 0: keep this evaluation, append it to metrics_file, log the history.

        Args:
            groups: {group: {"num_samples": n, metric: value, ...}}
            score: the value that picks model_best.pth
            counted: the groups the score is computed from
        """
        epoch = self.trainer.epoch + 1
        record = dict(
            epoch=epoch,
            best_metric=self.best_metric,
            score=score,
            counted=list(counted),
            groups=groups,
            **extra,
        )
        self.results = record
        self.history[epoch] = record
        path = self.metrics_path()
        if path is not None:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        self.log_history()

    def log_history(self):
        """Best epoch per group so far, of the score, and the Pareto front."""
        metric, greater = self.best_metric, self.best_metric_greater
        pick = max if greater else min
        records = [r for r in self.history.values() if r["best_metric"] == metric]
        last = records[-1]
        lines = []
        for group in last["groups"]:
            have = [r for r in records if group in r["groups"]]
            best = pick(have, key=lambda r: r["groups"][group][metric])
            value = best["groups"][group][metric]
            lines.append(f"{group} {value:.4f} (epoch {best['epoch']})")
        self.trainer.logger.info(
            f"Best {metric} per group so far (diagnostics; model_best follows the "
            f"score): " + ", ".join(lines)
        )
        best = pick(records, key=lambda r: r["score"])
        info = f"Best score so far: {best['score']:.4f} (epoch {best['epoch']})"
        counted = last["counted"]
        if len(counted) > 1:
            have = [r for r in records if all(g in r["groups"] for g in counted)]
            rows = [{g: r["groups"][g][metric] for g in counted} for r in have]
            front = [have[i]["epoch"] for i in pareto_front(rows, counted, greater)]
            info += f"; epochs on the {metric} Pareto front of {counted}: {front}"
        self.trainer.logger.info(info)

    def before_epoch(self):
        if self.evaluation_only:
            self.after_epoch()
            # stop the process of training
            comm.synchronize()
            sys.exit(0)

    def after_epoch(self):
        # Skip evaluation if cfg.evaluate is False
        if not self.trainer.cfg.evaluate:
            return

        # Start evaluation
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        self.eval_before()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    device = get_device()
                    input_dict[key] = input_dict[key].to(
                        device, non_blocking=(device.type == "cuda")
                    )
            # inference_mode (as in DepthVisualizer): models must not cache
            # tensors created here for later use in training
            with torch.inference_mode():
                self.eval_step(input_dict, step=i)
        self.eval_finalize()
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def eval_before(self):
        pass

    def eval_step(self, input_dict, step):
        pass

    def eval_finalize(self):
        pass


@HOOKS.register_module()
class SemSegEvaluator(DefaultEvaluator):
    METRICS = dict(miou=True, macc=True, allacc=True)  # True: higher is better
    GROUP = "val"  # one group: the whole val set (metrics_file)

    def __init__(
        self,
        num_classes,
        ignore_index=-1,
        tta=None,
        tta_batched=True,
        best_metric="miou",
        **kwargs,
    ):
        """Semantic segmentation evaluator (confusion matrix over the val set).

        Args:
            tta: None (default) or op lists as DepthEvaluator, e.g. [["flip"]]: the
                softmax of each augmented prediction (mapped back, e.g. flipped back)
                is averaged with the softmax of the original prediction (always
                included: do not list []). Also "scale@<r>" (the image resized by r,
                see tta_predictions), composable with flip: with a scale op every
                softmax is resized (bilinear) to the label size and averaged there;
                without one (flip / upsample: all at the image size) the mean softmax
                is resized to the label size. DINOv3's 5 scales x flip (r = 0.9 ...
                1.1) is [["flip"]] + [[f"scale@{r}"], [f"scale@{r}", "flip"]] for r in
                (0.9, 0.95, 1.05, 1.1): 9 entries, 10 predictions.
            tta_batched: True (default): the image and its same-shape TTA variants
                run in one forward (see tta_predictions); False: one forward each.
            best_metric: "miou" (default), "macc" or "allacc"; the highest picks
                model_best.pth.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        for ops in tta or []:
            for op in ops:
                if scale_of(op) is None:
                    check_choice("tta op", op, TTA_PRE_PROCESS)
        self.tta = tta
        self.tta_batched = tta_batched
        self.best_metric = best_metric
        check_choice("best_metric", best_metric, self.METRICS)
        self.best_metric_greater = self.METRICS[best_metric]

    def eval_before(self):
        self.trainer.comm_info["semseg_results"] = []

    def eval_step(self, input_dict, step):
        gt_segment = input_dict.pop("segment")  # not compute loss in eval
        preds = tta_predictions(
            self.trainer.model, input_dict, "pred_segment", self.tta, self.tta_batched
        )
        pred_segment = preds[0]
        if self.tta:
            probs = [pred.softmax(dim=1) for pred in preds]
            if any(p.shape != probs[0].shape for p in probs):
                # scale@r: each softmax at the label size, then averaged
                size = gt_segment.shape[-2:]
                probs = [
                    (
                        p
                        if p.shape[-2:] == size
                        else F.interpolate(
                            p, size=size, mode="bilinear", align_corners=False
                        )
                    )
                    for p in probs
                ]
            pred_segment = torch.stack(probs, dim=1).mean(dim=1)

        # User guarantees unique name exists
        names = input_dict["name"]

        if pred_segment.shape[-2:] != gt_segment.shape[-2:]:
            pred_segment = F.interpolate(
                pred_segment,
                size=gt_segment.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Manual calculation for deduplication
        pred_segment_idx = pred_segment.argmax(dim=1)
        gt_segment_idx = gt_segment.squeeze(1)

        for i, name in enumerate(names):
            pred = pred_segment_idx[i].flatten()
            label = gt_segment_idx[i].flatten()

            keep = (label != self.ignore_index) & (label < self.num_classes)
            hist = torch.bincount(
                self.num_classes * label[keep] + pred[keep],
                minlength=self.num_classes**2,
            ).reshape(self.num_classes, self.num_classes)

            # Store in comm_info
            self.trainer.comm_info["semseg_results"].append((name, hist.cpu()))

        if step % self.trainer.cfg.log_interval == 0:
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=step, max_iter=len(self.trainer.val_loader)
            )
            self.trainer.logger.info(info)

    def eval_finalize(self):
        comm.synchronize()

        local_results = self.trainer.comm_info.pop("semseg_results")
        all_results_lists = comm.gather(local_results)

        if comm.is_main_process():
            unique_results = {}
            for res_list in all_results_lists:
                for name, hist in res_list:
                    if name not in unique_results:
                        unique_results[name] = hist

            total_hist = torch.zeros(
                (self.num_classes, self.num_classes), dtype=torch.float64
            )
            for hist in unique_results.values():
                total_hist += hist

            diag = torch.diag(total_hist)
            row_sum = total_hist.sum(dim=1)
            col_sum = total_hist.sum(dim=0)
            union = row_sum + col_sum - diag

            iou = diag / (union + 1e-10)
            miou = iou.mean().item()  # over all classes, union 0 counting as IoU 0
            per_class_iou = iou.tolist()
            macc = (diag / (row_sum + 1e-10)).mean().item()
            allacc = (diag.sum() / (total_hist.sum() + 1e-10)).item()
            # logged only: the mean over the classes with union > 0 (in GT or in the
            # prediction), and the number of classes in neither (equal to miou and 0
            # when every class occurs in the val GT)
            present = union > 0
            miou_present = iou[present].mean().item() if present.any() else math.nan
            absent = (~present).nonzero().flatten().tolist()
            metrics = dict(
                miou=miou, macc=macc, allacc=allacc, miou_present=miou_present
            )

            self.trainer.logger.info("Evaluation results:")
            for i, value in enumerate(per_class_iou):
                self.trainer.logger.info(f"Class {i:>3}: {value:.4f}")
            self.trainer.logger.info(f"Mean IoU: {miou:.4f}")
            self.trainer.logger.info(f"Mean Acc: {macc:.4f}  All Acc: {allacc:.4f}")
            self.trainer.logger.info(
                f"Mean IoU of the {int(present.sum())} present classes: "
                f"{miou_present:.4f}"
            )
            if absent:
                self.trainer.logger.warning(
                    f"{len(absent)} classes in neither GT nor prediction (union 0, "
                    f"IoU 0 in Mean IoU): {absent}"
                )

            current_epoch = self.trainer.epoch + 1
            if self.trainer.writer is not None:
                for key, value in metrics.items():
                    self.trainer.writer.add_scalar(f"val/{key}", value, current_epoch)
                if self.trainer.cfg.enable_wandb:
                    wandb.log(
                        {
                            "Epoch": current_epoch,
                            **{f"val/{k}": v for k, v in metrics.items()},
                        },
                        step=wandb.run.step,
                    )
            self.trainer.comm_info["current_metric_value"] = metrics[self.best_metric]
            self.trainer.comm_info["current_metric_name"] = self.best_metric
            self.trainer.comm_info["current_metric_greater"] = self.best_metric_greater
            group = dict(
                num_samples=len(unique_results),
                num_absent_classes=len(absent),
                **metrics,
            )
            self.record(
                {self.GROUP: group},
                metrics[self.best_metric],
                [self.GROUP],
                per_class_iou=per_class_iou,
            )

        comm.synchronize()


@lru_cache(maxsize=None)
def max_res_columns(height, width, max_res):
    """The columns of a height x width image that Marigold's least-squares fit uses with
    --alignment_max_res max_res (src/util/alignment.py,
    align_depth_least_square(max_resolution=max_res)): scale factor min(max_res /
    height, max_res / width); below 1, torch.nn.Upsample(scale_factor, mode="nearest")
    on the [1, H, W] GT, prediction and mask, i.e. a 1-D nearest subsample along W
    (every row is kept). Computed with that call (CPU, float32, as there) on the column
    indices: a sorted LongTensor, or None (scale factor >= 1: every pixel). For 4032 x
    6048 (ETH3D): 1024 of the 6048 columns, 0, 5, 11, 17, ..., 6042."""
    scale_factor = np.min(max_res / np.array((height, width)))
    if scale_factor >= 1:
        return None
    down = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
    index = torch.arange(width, dtype=torch.float32).view(1, 1, width)
    return down(index).view(-1).long()


def parse_align_max_res(align_max_res):
    """The evaluator's align_max_res option -> {group: max_res int}; None (also the
    strings "None" / "null" of --options) -> {}."""
    if align_max_res is None or (
        isinstance(align_max_res, str) and align_max_res.lower() in ("none", "null")
    ):
        return {}
    if not isinstance(align_max_res, dict):
        raise ValueError(
            f"align_max_res must be None or {{group: max_res}}, got {align_max_res!r}"
        )
    out = {}
    for group, max_res in align_max_res.items():
        if max_res is None or str(max_res).lower() in ("none", "null"):
            continue
        if isinstance(max_res, bool) or int(max_res) != max_res or max_res < 1:
            raise ValueError(
                f"align_max_res[{group!r}] must be an int >= 1: {max_res!r}"
            )
        out[str(group)] = int(max_res)
    return out


def get_gt_protocols(dataset):
    """Map dataset group (sample name prefix) -> dict(crop=, range=) of a val set:
    its crop_protocol() and range_protocol() ("none" / "unknown" without them)."""
    protocols = {}
    for ds in getattr(dataset, "datasets", [dataset]):
        if not hasattr(ds, "crop_protocol") or len(ds) == 0:
            continue
        name = ds.get_data_name(0)
        if name is not None:
            gt_range = ds.range_protocol() if hasattr(ds, "range_protocol") else None
            protocols[name.split("/")[0]] = dict(
                crop=ds.crop_protocol(), range=gt_range or "unknown"
            )
    return protocols


@HOOKS.register_module()
class DepthEvaluator(DefaultEvaluator):
    TTA_PRE_PROCESS = TTA_PRE_PROCESS
    TTA_POST_PROCESS = TTA_POST_PROCESS
    # logged only, not counted in the best-checkpoint mean (DIODE-INDOOR + -OUTDOOR)
    COMBINED_GROUP = "DIODE"
    ALIGN_SOLVERS = ("centered", "lsq", "lsq_free", "ransac")
    # sensitivity protocols: logged as "<group>@<protocol>", never counted
    ALIGN_PROTOCOLS = ("trim5", "disp", "ransac", "roe")
    PROTOCOL_SEP = "@"
    # eval_depth metrics; True: higher is better
    METRICS = dict(
        delta1=True,
        delta2=True,
        delta3=True,
        abs_rel=False,
        sq_rel=False,
        rmse=False,
        rmse_log=False,
        log10=False,
        silog=False,
    )

    def __init__(
        self,
        relative=False,
        tta=None,
        tta_batched=True,
        align_solver="centered",
        best_metric="abs_rel",
        best_metric_mean="geometric",
        score_groups=None,
        align_protocols=(),
        ransac_seed=0,
        align_max_res=None,
        **kwargs,
    ):
        """Depth evaluator.

        Args:
            relative: align prediction to GT with per-image scale & shift.
            tta: None / [["flip"], ["upsample"], ["flip", "upsample"]], averaged
                with the original prediction.
            tta_batched: True (default): the image and its same-shape TTA variants
                (flip) run in one forward (see tta_predictions); False: one forward
                each.
            align_solver: scale & shift solver for relative depth (per image, in
                depth; the prediction is then clamped to the dataset's range).
                - "centered": compute_scale_and_shift with ridge=0, centred fp64
                  moments (default): scale clamped to [1e-3, 1e3], a degenerate or
                  negative fit falls back to median / MAD matching
                - "lsq": exact fp64 least squares with a non-negative scale and no
                  clamp (nonneg_lsq_scale_and_shift): an image whose fit is
                  rejected (var(p) <= 1e-6 or a scale <= 0) gets the constrained
                  optimum s = 0, t = mean GT, i.e. it scores as the constant GT
                  mean (counted as num_rejected): every image with >= 10 valid
                  pixels counts, whatever the model, so flipping or flattening
                  an image never improves the score (an excluded image would
                  leave the mean).
                - "lsq_free": exact fp64 least squares with no constraint
                  (free_lsq_scale_and_shift), Marigold's evaluation fit
                  (align_depth_least_square, np.linalg.lstsq): bit for bit "lsq"
                  where that fit is accepted; a negative scale is kept (the
                  prediction enters inverted) and a flat prediction scores as the
                  GT mean. num_rejected counts the fits "lsq" rejects (a scale <= 0,
                  kept here, or a flat prediction, which scores the same in both).
                - "ransac": seeded depth RANSAC (ransac_scale_and_shift, seed
                  ransac_seed for every image; InfiniDepth's sklearn
                  RANSACRegressor protocol, reproducible): 1000 two-point trials,
                  inliers within the median absolute deviation of the GT, then
                  exact least squares on the inliers of the best trial; a fit that
                  is degenerate or has a scale <= 0, or a constant prediction (no
                  trial), takes the mean ratio s = mean GT / mean prediction, t = 0
                  (counted as num_rejected). The same fit, prediction and clip as
                  the "ransac" protocol below, so without align_max_res its groups
                  equal "<group>@ransac" bit for bit; the solver of the depth
                  recipes (docs/evaluation.md).
            best_metric: one of METRICS; its mean over the evaluated groups (sample
                name prefixes, e.g. nyuv2, kitti, eth3d, DIODE-INDOOR, DIODE-OUTDOOR;
                the combined DIODE group is logged only) picks model_best.pth, the
                highest for delta*, the lowest for the error metrics. Default
                "abs_rel" (delta1 is near saturation on these sets).
            best_metric_mean: "geometric" (default: a relative change counts the same
                in every group, so DIODE-OUTDOOR's ~0.3 AbsRel does not outweigh
                NYUv2's ~0.05) or "arithmetic".
            score_groups: None (default): every group but the combined DIODE one
                counts in the mean; a list: only these groups count, the others are
                logged (e.g. a held-out synthetic set kept out of the selection).
            align_protocols: sensitivity protocols (relative only), each logged as
                extra groups "<group>@<protocol>" (nyuv2@trim5, DIODE@ransac, ...)
                that never count in the score (a score_groups entry with "@" is an
                error), on the same flip-averaged prediction:
                - "trim5": exact LSQ in depth, the 5% largest relative residuals
                  |s p + t - g| / g dropped, LSQ again (trimmed_scale_and_shift);
                - "disp": LSQ in disparity, 1 / clamp(p, 1e-3) against 1 / g (Marigold
                  least_square_disparity), the aligned disparity clamped to
                  >= 1 / depth_max (1e-3 without depth_max or with depth_max inf)
                  and inverted;
                - "ransac": align_solver "ransac" (ransac_scale_and_shift, seed
                  ransac_seed for every image), on every valid pixel;
                - "roe": MoGe's evaluation fit, the exact 1/g-weighted L1 fit
                  (ROE, no truncation) on 4096 deterministic samples of the valid
                  pixels (roe_eval_scale_and_shift; the fit of the L-DPT depth
                  loss without the truncation and the refit).
                The prediction is clamped to the dataset's range and the metrics
                use every valid pixel, as for the headline groups; a rejected fit
                takes the fallback (trim5 / disp: s = 0, the constant GT mean in
                depth / disparity; ransac: the mean ratio; roe: s = 0, the
                1/g-weighted GT median) and counts in
                num_rejected.
            ransac_seed: seed of the "ransac" solver and protocol (default 0).
            align_max_res: None (default: every fit uses every valid pixel) or
                {group: max_res} (relative only), Marigold's --alignment_max_res per
                group (sample name prefix; its ETH3D script passes 1024): the headline
                fit of that group's images uses the valid pixels of every row and of
                the columns Marigold's fit keeps (max_res_columns: a nearest
                subsample along W by min(max_res / H, max_res / W)); the metrics
                still use every valid pixel (an image with < 10 valid pixels in those
                columns is excluded), and the align_protocols fit on every valid
                pixel. Logged in the protocol as "align=<solver>/max-res-<N>".
        With align_solver "lsq" / "lsq_free" / "ransac" or align_protocols, every
        group also records num_excluded (images without metrics: < 10 valid GT
        pixels, so the same in every epoch) and num_rejected (images whose fit was
        replaced by the solver's fallback: "centered" median / MAD, "lsq"
        the GT mean, "ransac" the mean ratio; with "lsq_free" the fits that "lsq"
        would replace; see align_protocols), and the record align_protocols;
        otherwise the record has none of the three.
        """
        super().__init__(**kwargs)
        check_choice("align_solver", align_solver, self.ALIGN_SOLVERS)
        align_protocols = list(align_protocols or [])
        for protocol in align_protocols:
            check_choice("align protocol", protocol, self.ALIGN_PROTOCOLS)
        if len(set(align_protocols)) != len(align_protocols):
            raise ValueError(f"align_protocols repeats a protocol: {align_protocols}")
        if align_protocols and not relative:
            raise ValueError("align_protocols need relative=True")
        for group in score_groups or []:
            if self.PROTOCOL_SEP in group:
                raise ValueError(
                    f"score_groups: {group!r} is a sensitivity protocol group; "
                    f"groups with {self.PROTOCOL_SEP!r} are never counted"
                )
        self.align_protocols = align_protocols
        self.ransac_seed = int(ransac_seed)
        self.align_max_res = parse_align_max_res(align_max_res)
        if self.align_max_res and not relative:
            raise ValueError("align_max_res needs relative=True")
        # the record lists excluded images only where a fit can exclude them
        self.record_excluded = align_solver in ("lsq", "lsq_free", "ransac") or bool(
            align_protocols
        )
        self.relative = relative
        for ops in tta or []:
            for op in ops:
                check_choice("tta op", op, TTA_PRE_PROCESS)
        self.tta = tta
        self.tta_batched = tta_batched
        self.align_solver = align_solver
        self.best_metric = best_metric
        check_choice("best_metric", best_metric, self.METRICS)
        self.best_metric_greater = self.METRICS[best_metric]
        check_choice("best_metric_mean", best_metric_mean, ("geometric", "arithmetic"))
        self.best_metric_mean = best_metric_mean
        self.score_groups = None if score_groups is None else list(score_groups)
        self.gt_protocols = {}
        self.per_image = {}  # name -> metrics of the last evaluation (rank 0)

    def protocol(self, ds_name):
        """Short description of how ds_name is evaluated, for logging."""
        ds_name, _, protocol = ds_name.partition(self.PROTOCOL_SEP)
        if protocol:
            align = dict(trim5="depth-trim5", disp="disparity-lsq")
            align = align.get(protocol, f"depth-{protocol}")
        elif not self.relative:
            align = "none"
        elif self.align_solver in ("lsq", "lsq_free", "ransac"):
            align = "depth-" + self.align_solver.replace("_", "-")
        else:
            align = f"scale-shift/{self.align_solver}"
        if not protocol and ds_name in self.align_max_res:
            align += f"/max-res-{self.align_max_res[ds_name]}"
        gt = [
            gt_protocol
            for name, gt_protocol in self.gt_protocols.items()
            if name == ds_name or name.startswith(ds_name + "-")
        ]
        crop = "+".join(sorted({p["crop"] for p in gt})) or "unknown"
        gt_range = "+".join(sorted({p["range"] for p in gt})) or "unknown"
        tta = "+".join(["orig"] + ["/".join(ops) for ops in self.tta or []])
        return f"align={align} crop={crop} range={gt_range} tta={tta}"

    @staticmethod
    def eval_depth(pred, target):
        assert pred.shape == target.shape

        thresh = torch.max((target / pred), (pred / target))

        d1 = torch.sum(thresh < 1.25).float() / thresh.numel()
        d2 = torch.sum(thresh < 1.25**2).float() / thresh.numel()
        d3 = torch.sum(thresh < 1.25**3).float() / thresh.numel()

        diff = pred - target
        diff_log = torch.log(pred) - torch.log(target)

        abs_rel = torch.mean(torch.abs(diff) / target)
        sq_rel = torch.mean(torch.pow(diff, 2) / target)

        rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
        rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

        log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
        silog = torch.sqrt(
            torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)
        )
        result_dict = dict(
            delta1=d1,
            delta2=d2,
            delta3=d3,
            abs_rel=abs_rel,
            sq_rel=sq_rel,
            rmse=rmse,
            rmse_log=rmse_log,
            log10=log10,
            silog=silog,
        )
        return result_dict

    def eval_before(self):
        self.trainer.comm_info["depth_results"] = []
        self.gt_protocols = get_gt_protocols(self.trainer.val_loader.dataset)

    def align(self, pred, gt, mask):
        """Headline (scale, shift, valid, rejected) per image with align_solver;
        rejected: the fit was replaced by the solver's fallback."""
        if self.align_solver == "ransac":
            return ransac_scale_and_shift(
                pred, gt, mask, seed=self.ransac_seed, return_rejected=True
            )
        if self.align_solver == "centered":
            return compute_scale_and_shift(pred, gt, mask, ridge=0, return_unfit=True)
        if self.align_solver == "lsq_free":
            return free_lsq_scale_and_shift(pred, gt, mask)
        return nonneg_lsq_scale_and_shift(pred, gt, mask)

    def fit_mask(self, names, mask):
        """The pixels of the headline fit ([B, H, W] bool): mask, restricted for the
        images of a group in align_max_res to Marigold's fit columns
        (max_res_columns of the image's H x W)."""
        if not self.align_max_res:
            return mask
        fit = mask
        height, width = mask.shape[-2:]
        for i, name in enumerate(names):
            max_res = self.align_max_res.get(name.split("/")[0])
            cols = None if max_res is None else max_res_columns(height, width, max_res)
            if cols is None:
                continue
            keep = torch.zeros(width, dtype=torch.bool, device=mask.device)
            keep[cols.to(mask.device)] = True
            if fit is mask:
                fit = mask.clone()
            fit[i] &= keep
        return fit

    @staticmethod
    def clamp_range(pred, input_dict):
        """Clamp to the per-sample [depth_min, depth_max] ([B], over [B, H, W]);
        depth_max inf: no upper clamp."""
        if "depth_min" in input_dict:
            pred = pred.clamp(min=input_dict["depth_min"].view(-1, 1, 1))
        if "depth_max" in input_dict:
            pred = pred.clamp(max=input_dict["depth_max"].view(-1, 1, 1))
        return pred

    def align_protocol(self, protocol, pred, gt, mask, input_dict):
        """(prediction aligned by a sensitivity protocol and clamped to the dataset's
        range, valid [B], rejected [B]); see __init__."""
        if protocol == "disp":
            x = 1.0 / pred.clamp(min=1e-3)
            y = torch.where(mask, 1.0 / gt, 0.0)
            scale, shift, valid, rejected = nonneg_lsq_scale_and_shift(x, y, mask)
            disp = x * scale[:, None, None] + shift[:, None, None]
            if "depth_max" in input_dict:
                # 1 / depth_max; no upper bound (inf): Marigold's floor 1e-3
                floor = 1.0 / input_dict["depth_max"].view(-1, 1, 1)
                disp = disp.clamp(min=torch.where(floor > 0, floor, 1e-3))
            else:
                disp = disp.clamp(min=1e-3)
            aligned = 1.0 / disp
        else:
            if protocol == "trim5":
                scale, shift, valid, rejected = trimmed_scale_and_shift(
                    pred, gt, mask, 0.05, return_rejected=True
                )
            elif protocol == "roe":
                scale, shift, valid, rejected = roe_eval_scale_and_shift(
                    pred, gt, mask, return_rejected=True
                )
            else:  # "ransac"
                scale, shift, valid, rejected = ransac_scale_and_shift(
                    pred, gt, mask, seed=self.ransac_seed, return_rejected=True
                )
            aligned = pred * scale[:, None, None] + shift[:, None, None]
        aligned = self.clamp_range(aligned.clamp(min=1e-6), input_dict)
        return aligned, valid, rejected

    def collect(self, names, pred, gt, mask, valid, suffix="", rejected=None):
        """Metrics of the valid images into comm_info["depth_results"] as (name,
        group, metrics, rejected); excluded images as (name, group, None, False).
        rejected: None or [B] bool, the images whose fit took the fallback."""
        rejected = [False] * len(names) if rejected is None else rejected.tolist()
        valid_names = []
        metrics_list = []
        for i, (name, ok) in enumerate(zip(names, valid.tolist())):
            if not ok:
                continue
            m = mask[i]
            metrics = self.eval_depth(pred[i][m], gt[i][m])
            valid_names.append(name)
            # native metric dtype (no fp64 cast: MPS has no fp64)
            metrics_list.append(torch.stack(list(metrics.values())))

        results = self.trainer.comm_info["depth_results"]
        flags = dict(zip(names, rejected))
        if metrics_list:
            # one host transfer per batch; tolist() gives the exact .item() values
            metric_keys = list(metrics.keys())
            for name, values in zip(valid_names, torch.stack(metrics_list).tolist()):
                dataset_name = name.split("/")[0]
                metrics_cpu = dict(zip(metric_keys, values))
                flag = bool(flags[name])
                results.append((name, dataset_name + suffix, metrics_cpu, flag))
                if dataset_name.startswith(self.COMBINED_GROUP + "-"):
                    # Also store DIODE-INDOOR and DIODE-OUTDOOR as one group
                    results.append(
                        (name, self.COMBINED_GROUP + suffix, dict(metrics_cpu), flag)
                    )
        for name, ok in zip(names, valid.tolist()):
            if ok:
                continue
            dataset_name = name.split("/")[0]
            results.append((name, dataset_name + suffix, None, False))
            if dataset_name.startswith(self.COMBINED_GROUP + "-"):
                results.append((name, self.COMBINED_GROUP + suffix, None, False))

    def eval_step(self, input_dict, step):
        gt_depth = input_dict.pop("depth")  # not compute loss in forward
        names = input_dict.pop("name")
        preds = tta_predictions(
            self.trainer.model, input_dict, "pred_depth", self.tta, self.tta_batched
        )
        # ensemble: mean of the original and the mapped-back TTA predictions
        pred_depth = preds[0] if len(preds) == 1 else torch.stack(preds, 1).mean(1)

        if pred_depth.shape != gt_depth.shape:
            pred_depth = F.interpolate(
                pred_depth,
                size=gt_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if len(pred_depth.shape) == 4:
            pred_depth = pred_depth.squeeze(1)
        if len(gt_depth.shape) == 4:
            gt_depth = gt_depth.squeeze(1)

        mask = gt_depth > 0
        # Check valid pixels per image
        enough = mask.flatten(1).sum(1) >= 10
        valid_images = enough

        rejected = None
        if self.relative:
            # fit on fit_mask (align_max_res), score on mask
            scale, shift, valid_alignment, rejected = self.align(
                pred_depth, gt_depth, self.fit_mask(names, mask)
            )
            valid_images = valid_images & valid_alignment

            aligned = (pred_depth * scale[:, None, None] + shift[:, None, None]).clamp(
                min=1e-6
            )
        else:
            aligned = pred_depth
        aligned = self.clamp_range(aligned, input_dict)
        self.collect(names, aligned, gt_depth, mask, valid_images, "", rejected)

        for protocol in self.align_protocols:
            aligned, valid, rejected = self.align_protocol(
                protocol, pred_depth, gt_depth, mask, input_dict
            )
            suffix = self.PROTOCOL_SEP + protocol
            self.collect(
                names, aligned, gt_depth, mask, enough & valid, suffix, rejected
            )

        if step % self.trainer.cfg.log_interval == 0:
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=step, max_iter=len(self.trainer.val_loader)
            )
            self.trainer.logger.info(info)

    def eval_finalize(self):
        comm.synchronize()
        local_results = self.trainer.comm_info.pop("depth_results")
        all_results_lists = comm.gather(local_results)

        if comm.is_main_process():
            # Deduplicate by (name, dataset_name) -> metrics: a sample counts once in
            # each of its groups (e.g. DIODE-INDOOR and the combined DIODE)
            unique_results, unique_rejected = {}, {}
            for res_list in all_results_lists:
                for name, ds_name, metrics, rejected in res_list:
                    key = (name, ds_name)  # Use tuple as key
                    if key not in unique_results:
                        unique_results[key] = metrics
                        unique_rejected[key] = rejected

            # Group by dataset; excluded images (metrics None) are only counted
            dataset_results = defaultdict(list)
            excluded, rejected = defaultdict(int), defaultdict(int)
            for (name, ds_name), metrics in unique_results.items():
                if metrics is None:
                    excluded[ds_name] += 1
                else:
                    dataset_results[ds_name].append(metrics)
                    rejected[ds_name] += unique_rejected[(name, ds_name)]
            # headline groups first, then the groups of each protocol (stable sort:
            # the order of first appearance within each)
            order = ["", *self.align_protocols]
            dataset_results = dict(
                sorted(
                    dataset_results.items(),
                    key=lambda item: order.index(
                        item[0].partition(self.PROTOCOL_SEP)[2]
                    ),
                )
            )

            major_metric_values = []
            # the combined group repeats the split groups; a plain DIODE group counts
            combined = any(
                n.startswith(self.COMBINED_GROUP + "-") for n in dataset_results
            )
            # per-image metrics of the last evaluation (scripts/evaluate.py): headline
            self.per_image = {
                name: metrics
                for (name, ds_name), metrics in unique_results.items()
                if metrics is not None
                and self.PROTOCOL_SEP not in ds_name
                and not (combined and ds_name == self.COMBINED_GROUP)
            }
            groups, counted, num_rejected = {}, [], 0

            for ds_name, metrics_list in dataset_results.items():
                if not metrics_list:
                    continue

                # Aggregate
                keys = metrics_list[0].keys()
                avg_metrics = {k: 0.0 for k in keys}
                for m in metrics_list:
                    for k in keys:
                        avg_metrics[k] += m[k]

                num_samples = len(metrics_list)
                for k in keys:
                    avg_metrics[k] /= num_samples

                n_out, n_rej = excluded[ds_name], rejected[ds_name]
                out = ""
                if self.record_excluded:
                    kind = "rejected fits"
                    if (
                        self.align_solver == "lsq_free"
                        and self.PROTOCOL_SEP not in ds_name
                    ):
                        kind = "fits with a scale <= 0 or a flat prediction"
                    out = f", {n_out} excluded, {n_rej} {kind}"
                self.trainer.logger.info(
                    f"Evaluation results of {ds_name} "
                    f"({num_samples} samples{out}, {self.protocol(ds_name)}):"
                )
                for key, value in avg_metrics.items():
                    self.trainer.logger.info(f"{key:>8}: {value:.4f}")

                current_epoch = self.trainer.epoch + 1
                if self.trainer.writer is not None:
                    for key, value in avg_metrics.items():
                        self.trainer.writer.add_scalar(
                            f"val/{ds_name}/{key}", value, current_epoch
                        )
                    if self.trainer.cfg.enable_wandb:
                        wandb.log(
                            {
                                "Epoch": current_epoch,
                                **{
                                    f"val/{ds_name}/{k}": v
                                    for k, v in avg_metrics.items()
                                },
                            },
                            step=wandb.run.step,
                        )
                group = dict(num_samples=num_samples)
                if self.record_excluded:
                    group["num_excluded"] = n_out
                    group["num_rejected"] = n_rej
                groups[ds_name] = dict(group, **avg_metrics)
                if self.PROTOCOL_SEP in ds_name:  # sensitivity protocol: logged only
                    counts = False
                elif self.score_groups is None:
                    counts = not (combined and ds_name == self.COMBINED_GROUP)
                else:
                    counts = ds_name in self.score_groups
                if counts:
                    major_metric_values.append(avg_metrics[self.best_metric])
                    counted.append(ds_name)
                    num_rejected += n_rej

            empty = sorted(set(excluded) - set(dataset_results))
            if empty:
                self.trainer.logger.warning(
                    f"No image of {empty} could be evaluated (all excluded)"
                )
            absent = [g for g in self.score_groups or [] if g not in counted]
            if absent:
                self.trainer.logger.warning(
                    f"score_groups {absent} were not evaluated; the score uses {counted}"
                )

            if major_metric_values:
                n = len(major_metric_values)
                if self.best_metric_mean == "geometric":
                    major_metric_avg = math.prod(major_metric_values) ** (1 / n)
                else:
                    major_metric_avg = sum(major_metric_values) / n
                rej = ""
                if self.align_solver == "lsq_free":
                    rej = (
                        f", {num_rejected} fits with a scale <= 0 or a flat prediction"
                    )
                elif self.record_excluded:
                    rej = f", {num_rejected} rejected fits scored by the fallback"
                self.trainer.logger.info(
                    f"{self.best_metric_mean.capitalize()} mean {self.best_metric} of "
                    f"{n} groups (picks model_best{rej}): {major_metric_avg:.4f}"
                )
                comm_info = self.trainer.comm_info
                comm_info["current_metric_value"] = major_metric_avg
                comm_info["current_metric_name"] = self.best_metric
                comm_info["current_metric_greater"] = self.best_metric_greater
                extra = dict(best_metric_mean=self.best_metric_mean)
                if self.align_protocols:
                    extra["align_protocols"] = list(self.align_protocols)
                self.record(groups, major_metric_avg, counted, **extra)

        comm.synchronize()
