"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import pickle
import shutil
import time
import zipfile
import wandb
import torch
import torch.utils.data
from collections import OrderedDict
import ldpt.utils.comm as comm
from ldpt.utils.timer import Timer
from ldpt.utils.config import Config
from ldpt.datasets.meta import COCOSTUFF_CLASS_NAMES, VOC2012_CLASS_NAMES

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = (
            _remain_epoch * len(self.trainer.train_loader) - self.trainer.start_iter
        )

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self, log_interval=1):
        self.curr_iter = 0
        self.log_interval = log_interval
        self.model_output_keys = []
        # every key of the epoch's steps, in order of first appearance: the epoch
        # summary also has keys only some steps log
        self.epoch_keys = {}

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = (
            self.trainer.start_epoch * len(self.trainer.train_loader)
            + self.trainer.start_iter
        )
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("params/*", step_metric="Iter")
            wandb.define_metric("train_batch/*", step_metric="Iter")
            wandb.define_metric("train/*", step_metric="Epoch")

    def before_step(self):
        self.curr_iter += 1
        # this step's line is logged
        self.trainer.comm_info["log_iter"] = self.curr_iter % self.log_interval == 0
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            if "grad_norm" in self.trainer.comm_info:
                # pre-clip total gradient norm of the step (trainer clip_grad)
                grad_norm = self.trainer.comm_info.pop("grad_norm")
                model_output_dict = dict(model_output_dict, grad_norm=grad_norm)
            self.model_output_keys = model_output_dict.keys()
            self.epoch_keys.update(dict.fromkeys(self.model_output_keys))
            values = [v.detach().reshape(()) for v in model_output_dict.values()]
            if values:
                # one device-to-host transfer instead of one .item() per key
                device = values[0].device
                values = torch.stack([v.to(device) for v in values]).tolist()
            for key, value in zip(self.model_output_keys, values):
                self.trainer.storage.put_scalar(key, value)

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.param_groups[0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        if self.curr_iter % self.log_interval == 0:
            self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )
            if self.trainer.cfg.enable_wandb:

                wandb.log(
                    {"Iter": self.curr_iter, "params/lr": lr}, step=self.curr_iter
                )
                for key in self.model_output_keys:
                    wandb.log(
                        {
                            "Iter": self.curr_iter,
                            f"train_batch/{key}": self.trainer.storage.history(key).val,
                        },
                        step=wandb.run.step,
                    )

    def after_epoch(self):
        keys = list(self.epoch_keys) or list(self.model_output_keys)
        self.epoch_keys = {}
        epoch_info = "Train result: "
        for key in keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )

            if self.trainer.cfg.enable_wandb:

                for key in keys:
                    wandb.log(
                        {
                            "Epoch": self.trainer.epoch + 1,
                            f"train/{key}": self.trainer.storage.history(key).avg,
                        },
                        step=wandb.run.step,
                    )


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(
        self,
        save_freq=None,
        skip_keywords=None,
        save_freq_iter=None,
        keep_epoch_weights=False,
    ):
        """Checkpoint saver with selective parameter saving.

        Every epoch writes model/model_last.pth (training state included); the best
        evaluation writes model_best.pth. "Best" is the metric the evaluator reports
        (DepthEvaluator / SemSegEvaluator best_metric, with its direction).

        Args:
            save_freq: every save_freq epochs, also copy model_last.pth (training
                state included) to epoch_<N>.pth. None means only save last.
            skip_keywords: Keywords to skip when saving state_dict. Can be:
                - None: Save all parameters (default)
                - str: Single keyword to skip
                - list: List of keywords to skip
                Parameters containing any of these keywords will not be saved.
                Useful for skipping frozen backbone weights (e.g., ["encoder."]).
            save_freq_iter: also write model_last.pth every save_freq_iter training
                iterations (counted over the whole run), with "iter" = the batches of
                "epoch" already trained; resuming from it continues mid-epoch with the
                same data order (same GPU count and batch size). None (default):
                epoch checkpoints only.
            keep_epoch_weights: True: every epoch also writes model/epoch_<N>.pth
                with the model weights only ("epoch", "state_dict"; no optimizer;
                without the skip_keywords keys, i.e. decoder-only in the release
                configs), for choosing or averaging checkpoints after training
                (scripts/evaluate.py). Exclusive with save_freq (same file names).
                Default False.
        """
        if keep_epoch_weights and save_freq:
            raise ValueError(
                "keep_epoch_weights and save_freq both write epoch_<N>.pth; set one"
            )
        self.save_freq = save_freq
        self.save_freq_iter = save_freq_iter
        self.keep_epoch_weights = keep_epoch_weights

        # Normalize skip_keywords to list
        if skip_keywords is None:
            self.skip_keywords = []
        elif isinstance(skip_keywords, str):
            self.skip_keywords = [skip_keywords]
        elif isinstance(skip_keywords, list):
            self.skip_keywords = skip_keywords
        else:
            raise ValueError(
                f"skip_keywords must be None, str, or list, got {type(skip_keywords)}"
            )

    def _filter_state_dict(self, state_dict):
        """Filter state_dict based on skip_keywords."""
        if not self.skip_keywords:
            return state_dict

        filtered_dict = {}
        skipped_count = 0

        for key, value in state_dict.items():
            # Check if key contains any skip keyword
            should_skip = any(keyword in key for keyword in self.skip_keywords)

            if should_skip:
                skipped_count += 1
            else:
                filtered_dict[key] = value

        if skipped_count > 0:
            self.trainer.logger.info(
                f"CheckpointSaver: Skipped {skipped_count} parameters matching {self.skip_keywords}"
            )

        return filtered_dict

    @staticmethod
    def check_finite(state_dict):
        """Raise instead of saving non-finite weights: a diverged run must not
        overwrite the last good model_last.pth (mid-epoch saves come minutes apart)."""
        bad = [
            k
            for k, v in state_dict.items()
            if torch.is_floating_point(v) and not torch.isfinite(v).all()
        ]
        if bad:
            raise FloatingPointError(
                f"{len(bad)} weight tensors are not finite (e.g. {bad[:3]}); "
                "checkpoint not written"
            )

    def save_last(self, epoch, state_dict=None, **extra):
        """Write model_last.pth atomically; epoch = the epoch to resume at."""
        if state_dict is None:
            state_dict = self._filter_state_dict(self.trainer.model.state_dict())
        self.check_finite(state_dict)
        filename = os.path.join(self.trainer.cfg.save_path, "model", "model_last.pth")
        self.trainer.logger.info("Saving checkpoint to: " + filename)
        # training state of hooks that keep one (state_dict / load_state_dict), restored
        # by CheckpointLoader.resume; the entry is absent when no hook has state
        hook_state = {
            type(h).__name__: h.state_dict()
            for h in getattr(self.trainer, "hooks", ())
            if callable(getattr(h, "state_dict", None))
        }
        if hook_state:
            extra = dict(extra, hook_state=hook_state)
        torch.save(
            {
                "epoch": epoch,
                **extra,
                "state_dict": state_dict,
                "optimizer": self.trainer.optimizer.state_dict(),
                "scheduler": self.trainer.scheduler.state_dict(),
                "scaler": (
                    self.trainer.scaler.state_dict()
                    if self.trainer.cfg.enable_amp
                    else None
                ),
                "best_metric_value": self.trainer.best_metric_value,
            },
            filename + ".tmp",
        )
        os.replace(filename + ".tmp", filename)
        return filename

    def save_epoch_weights(self, epoch, state_dict):
        """Write model/epoch_<epoch>.pth (weights only) atomically."""
        filename = os.path.join(
            self.trainer.cfg.save_path, "model", f"epoch_{epoch}.pth"
        )
        self.trainer.logger.info("Saving epoch weights to: " + filename)
        torch.save({"epoch": epoch, "state_dict": state_dict}, filename + ".tmp")
        os.replace(filename + ".tmp", filename)

    def after_step(self):
        if not self.save_freq_iter or not comm.is_main_process():
            return
        epoch_iters = len(self.trainer.train_loader)
        done = self.trainer.comm_info["iter"] + 1  # batches of this epoch trained
        if done == epoch_iters:
            return  # after_epoch saves the end of the epoch
        if (self.trainer.epoch * epoch_iters + done) % self.save_freq_iter == 0:
            self.save_last(self.trainer.epoch, iter=done)

    def is_better(self, value, greater):
        best = self.trainer.best_metric_value
        if greater:
            return value > best
        # the trainer starts at -inf; for "lower is better" any first value wins
        return best == -torch.inf or value < best

    def after_epoch(self):
        if comm.is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                greater = self.trainer.comm_info.get("current_metric_greater", True)
                if self.is_better(current_metric_value, greater):
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            epoch = self.trainer.epoch + 1
            state_dict = self._filter_state_dict(self.trainer.model.state_dict())
            filename = self.save_last(epoch, state_dict)
            if self.keep_epoch_weights:
                self.save_epoch_weights(epoch, state_dict)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )


def load_checkpoint(path):
    """Load a checkpoint to CPU.

    mmap pages tensors in lazily, so a 27 GB encoder checkpoint never sits fully
    in RAM. weights_only covers plain state dicts and trainer checkpoints
    (state_dict / optimizer / scheduler / scaler / epoch / best_metric_value).
    """
    path = str(path)
    # legacy (non-zip, pre torch 1.6) files cannot be memory-mapped
    mmap = zipfile.is_zipfile(path)
    try:
        return torch.load(path, map_location="cpu", mmap=mmap, weights_only=True)
    except pickle.UnpicklingError:
        # file pickles non-tensor objects; only load checkpoints you trust
        return torch.load(path, map_location="cpu", mmap=mmap, weights_only=False)


def merge_weights(ckpts, keywords, replacement, strip_module, logger=None):
    """Load checkpoint(s) into one CPU state dict.

    In each key (prefixed "module." first if it is not) the first `keywords` is
    replaced by `replacement` (per checkpoint when lists; default "module." and
    the keyword itself); strip_module then drops the "module." prefix (a model
    without DDP wrapper). Returns the state dict and the last loaded checkpoint.
    """
    if not isinstance(ckpts, (list, tuple)):
        ckpts = [ckpts]

    def _norm_list(x, n, default=None):
        # x: None / str / list[str]
        if isinstance(x, (list, tuple)):
            xs = list(x)
        elif x is None:
            xs = [default]
        else:
            xs = [x]
        if len(xs) == 1 and n > 1:
            xs = xs * n
        if len(xs) != n:
            raise ValueError(f"Length mismatch: got {len(xs)} items, expect {n}.")
        return xs

    keywords_list = _norm_list(keywords, len(ckpts), default="module.")
    replacement_list = _norm_list(replacement, len(ckpts), default=None)
    replacement_list = [
        (kw if rp is None else rp) for kw, rp in zip(keywords_list, replacement_list)
    ]  # replacement by default same as keywords

    if logger is not None:
        logger.info(
            f"Loading {len(ckpts)} checkpoint(s) with per-ckpt (keyword -> "
            f"replacement): {list(zip(keywords_list, replacement_list))}"
        )

    merged_weight = OrderedDict()
    for ckpt_path, kw, rp in zip(ckpts, keywords_list, replacement_list):
        if logger is not None:
            logger.info(f"Loading weight at: {ckpt_path}")
        checkpoint = load_checkpoint(ckpt_path)
        weight_ = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        for key, value in weight_.items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx

            if kw in key:
                key = key.replace(kw, rp, 1)

            if strip_module:
                key = key[7:]  # module.xxx.xxx -> xxx.xxx

            # keep on CPU: load_state_dict copies into the param's device
            merged_weight[key] = value
    return merged_weight, checkpoint


# VOC 2012 class -> the COCO-Stuff class of the same object (all 20 VOC objects are
# COCO things); VOC "background" has no counterpart. Names as in
# VOC2012_CLASS_NAMES / COCOSTUFF_CLASS_NAMES (the train-id order of
# COCOStuffDataset(label_mapping="official")).
VOC2012_FROM_COCOSTUFF = {
    "aeroplane": "airplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "dining table": "dining table",
    "dog": "dog",
    "horse": "horse",
    "motorbike": "motorcycle",
    "person": "person",
    "potted plant": "potted plant",
    "sheep": "sheep",
    "sofa": "couch",
    "train": "train",
    "tv monitor": "tv",
}
# CheckpointLoader(seg_head_rows=dict(classes=<key>)): target classes, source classes,
# target -> source name map
SEG_HEAD_ROW_MAPS = {
    "voc2012_from_cocostuff": (
        VOC2012_CLASS_NAMES,
        COCOSTUFF_CLASS_NAMES,
        VOC2012_FROM_COCOSTUFF,
    ),
}


def seg_head_from_rows(source, target_names, source_names, name_map):
    """Rows of a 1x1 seg head for other classes, from a head trained on source_names.

    source: tensor [len(source_names), ...] (weight or bias). A target class in
    name_map gets the row of its source class; every other target class (VOC:
    background) the mean of the source rows that no target class maps to (VOC from
    COCO-Stuff: the 151 classes that are not VOC objects).
    """
    if source.shape[0] != len(source_names):
        raise ValueError(
            f"seg_head_rows: source head has {source.shape[0]} rows, the class list "
            f"{len(source_names)}"
        )
    index = {name: i for i, name in enumerate(source_names)}
    unknown = set(name_map) - set(target_names) | set(name_map.values()) - set(index)
    if unknown:
        raise ValueError(f"seg_head_rows: names not in the class lists: {unknown}")
    mapped = [index[name_map[name]] for name in target_names if name in name_map]
    others = [i for i in range(len(source_names)) if i not in set(mapped)]
    rest = source[others].mean(dim=0)
    rows = [
        source[index[name_map[name]]] if name in name_map else rest
        for name in target_names
    ]
    return torch.stack(rows)


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    """Load cfg.checkpoint before training.

    Fresh run: cfg.checkpoint is a path or a list of paths. In each key the first
    `keywords` is replaced by `replacement` ("module." -> "module.encoder." loads a
    plain DINOv3 file into model.encoder); missing keys are allowed unless strict.

    seg_head_rows (fresh run only; None: off): dict(classes="voc2012_from_cocostuff",
    source="dummy") initialises model.seg_head (weight and bias) from the loaded
    head under `source` (the model-relative prefix the replacement gave it: VOC
    renames the COCO-Stuff "module.seg_head" to "module.dummy") row by row: each VOC
    object class from its COCO-Stuff class (VOC2012_FROM_COCOSTUFF), background from
    the mean of the other 151 rows, instead of the random init. The source head must
    use the official COCO-Stuff train ids (label_mapping="official"). A resume loads
    seg_head from model_last.pth like every other key.

    allow_missing (fresh run only): True (default) or False: raise when a model key is
    in none of the loaded files, e.g. a second stage whose start file ([DINOv3,
    <run>/model/epoch_<N>.pth]) does not fill the decoder, which would otherwise train
    from the init. Unexpected keys stay allowed (the DINOv3 file's local_cls_norm.*).

    Resume (cfg.resume=True, cfg.checkpoint=model_last.pth): keys load unchanged and
    strictly; epoch (and iter of a mid-epoch checkpoint, see
    CheckpointSaver(save_freq_iter)), best metric, optimizer, scheduler and scaler are
    restored. Keys that CheckpointSaver(skip_keywords) left out of the file (the frozen
    encoder) are reloaded as the fresh run loaded them: from the checkpoint in
    save_path/config.py, with this hook's keywords / replacement.
    """

    def __init__(
        self,
        keywords=None,
        replacement=None,
        strict=False,
        seg_head_rows=None,
        allow_missing=True,
    ):
        self.keywords = keywords
        self.replacement = replacement
        self.strict = strict
        if not isinstance(allow_missing, bool):
            raise ValueError(
                f"allow_missing must be True or False, got {allow_missing!r}"
            )
        self.allow_missing = allow_missing
        if seg_head_rows is not None:
            seg_head_rows = dict(seg_head_rows)
            unknown = set(seg_head_rows) - {"classes", "source"}
            if unknown or "classes" not in seg_head_rows:
                raise ValueError(
                    "seg_head_rows: dict(classes=..., source='dummy'), got "
                    f"{seg_head_rows}"
                )
            if seg_head_rows["classes"] not in SEG_HEAD_ROW_MAPS:
                raise ValueError(
                    f"seg_head_rows classes must be one of {list(SEG_HEAD_ROW_MAPS)}, "
                    f"got {seg_head_rows['classes']!r}"
                )
            seg_head_rows.setdefault("source", "dummy")
        self.seg_head_rows = seg_head_rows

    def init_seg_head_rows(self, weight):
        """seg_head_rows: write the target head into the merged state dict."""
        cfg = self.seg_head_rows
        target_names, source_names, name_map = SEG_HEAD_ROW_MAPS[cfg["classes"]]
        prefix = "" if comm.get_world_size() == 1 else "module."  # as merge_weights
        model_state = self.trainer.model.state_dict()
        for param in ("weight", "bias"):
            src, dst = f"{prefix}{cfg['source']}.{param}", f"{prefix}seg_head.{param}"
            if src not in weight:
                raise KeyError(
                    f"seg_head_rows: {src} not in the loaded checkpoints (the "
                    "replacement must rename the source seg_head to it)"
                )
            rows = seg_head_from_rows(weight[src], target_names, source_names, name_map)
            if rows.shape != model_state[dst].shape:
                raise ValueError(
                    f"seg_head_rows: {dst} is {tuple(model_state[dst].shape)}, the "
                    f"rows give {tuple(rows.shape)}"
                )
            weight[dst] = rows
        self.trainer.logger.info(
            f"seg_head_rows {cfg['classes']}: seg_head from {cfg['source']} ("
            + ", ".join(
                f"{name} <- {name_map.get(name, 'mean of the unmapped rows')}"
                for name in target_names
            )
            + ")"
        )

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        ckpts = self.trainer.cfg.checkpoint
        if ckpts is None:
            self.trainer.logger.info("No checkpoint specified for loading.")
            return
        if self.trainer.cfg.resume:
            self.resume(ckpts)
            return

        weight, _ = self.merge_weights(ckpts, self.keywords, self.replacement)
        if self.seg_head_rows is not None:
            self.init_seg_head_rows(weight)
        load_state_info = self.trainer.model.load_state_dict(weight, strict=self.strict)
        del weight
        self.trainer.logger.info(f"Missing keys: {load_state_info.missing_keys}")
        self.trainer.logger.info(f"Unexpected keys: {load_state_info.unexpected_keys}")
        if not self.allow_missing and load_state_info.missing_keys:
            raise RuntimeError(
                f"CheckpointLoader(allow_missing=False): {ckpts} leave "
                f"{len(load_state_info.missing_keys)} model keys at init: "
                f"{load_state_info.missing_keys}"
            )

    def merge_weights(self, ckpts, keywords, replacement):
        """Load checkpoint(s) into one CPU state dict keyed like the trainer's model.

        Returns the state dict and the last loaded checkpoint.
        """
        return merge_weights(
            ckpts,
            keywords,
            replacement,
            strip_module=comm.get_world_size() == 1,  # no DDP wrapper
            logger=self.trainer.logger,
        )

    def fresh_run_checkpoint(self):
        """cfg.checkpoint of the fresh run (on resume, cfg.checkpoint is model_last)."""
        path = os.path.join(self.trainer.cfg.save_path, "config.py")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Resume needs {path} (written by the fresh run) to reload the keys "
                "CheckpointSaver skipped."
            )
        ckpts = Config.fromfile(path).get("checkpoint")
        if ckpts is None:
            raise ValueError(f"{path} has no checkpoint to reload skipped keys from.")
        return ckpts

    def resume(self, ckpts):
        trainer = self.trainer
        trainer.logger.info("Resume training: keys load unchanged and strictly.")
        weight, checkpoint = self.merge_weights(ckpts, "module.", None)

        skip_keywords = [
            kw
            for h in trainer.hooks
            if isinstance(h, CheckpointSaver)
            for kw in h.skip_keywords
        ]
        skipped = {
            k
            for k in trainer.model.state_dict().keys()
            if k not in weight and any(kw in k for kw in skip_keywords)
        }
        if skipped:
            fresh_ckpts = self.fresh_run_checkpoint()
            trainer.logger.info(
                f"Reloading {len(skipped)} keys skipped by CheckpointSaver "
                f"{skip_keywords} from the fresh-run checkpoint {fresh_ckpts}"
            )
            pretrained, _ = self.merge_weights(
                fresh_ckpts, self.keywords, self.replacement
            )
            weight.update((k, pretrained[k]) for k in skipped if k in pretrained)
            del pretrained

        load_state_info = trainer.model.load_state_dict(weight, strict=False)
        del weight
        missing = [k for k in load_state_info.missing_keys if k not in skipped]
        if missing or load_state_info.unexpected_keys:
            raise RuntimeError(
                f"Resume from {ckpts}: missing keys {missing}, "
                f"unexpected keys {load_state_info.unexpected_keys}"
            )
        if load_state_info.missing_keys:
            # not in the fresh-run checkpoint either: init values, as in the fresh run
            trainer.logger.warning(
                f"Skipped keys kept at init: {load_state_info.missing_keys}"
            )

        trainer.start_epoch = checkpoint["epoch"]
        trainer.start_iter = checkpoint.get("iter", 0)  # mid-epoch checkpoint
        # read by the hooks that keep training state (their before_train runs after this)
        trainer.comm_info["resume_hook_state"] = checkpoint.get("hook_state", {})
        trainer.logger.info(
            f"Resuming train at epoch {trainer.start_epoch + 1}, "
            f"iter {trainer.start_iter}"
        )
        trainer.best_metric_value = checkpoint["best_metric_value"]
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler"])
        if trainer.cfg.enable_amp:
            trainer.scaler.load_state_dict(checkpoint["scaler"])


@HOOKS.register_module()
class CriterionWeightWarmup(HookBase):
    """Gated warm-up of one weight (``key``) of the model's criterion: the weight is
    held at 0 until the data term alone has aligned the predictions, then ramped
    linearly to its configured value.

    - hold: weight 0 for at least ``min_hold`` iterations and until the exponential
      moving averages (``ema`` per iteration, over all ranks) of the model outputs
      ``depth_loss`` <= ``gate_depth_loss`` and ``gate_key`` <= ``gate_clamp_frac``;
    - ramp: every iteration with the gate open, progress += 1 / ``ramp`` (it never
      decreases; a closed gate pauses it), weight = configured value * progress.

    The defaults are the depth recipes' values (``key="aux_scale",
    gate_key="drop_frac"``, gate 0.20 / 0.20): the local and gradient terms of the
    L-DPT depth loss stay off until few fits are rejected (docs/depth_loss.md
    section 5). ``gate_clamp_frac`` is the bound of ``gate_key`` whatever that key
    is.

    A warning is logged every ``log_interval`` iterations once ``max_hold`` iterations
    have passed without the gate ever opening (the weight is then still 0). The state
    (progress, averages, whether the gate opened) is saved in model_last.pth by
    CheckpointSaver and restored by CheckpointLoader.resume, so a resumed run
    continues from its checkpoint (a mid-epoch save runs before this hook's update of
    that step: one update behind); a fresh run starts at 0. Every rank computes the
    same weight from the all-reduced averages. Once progress is 1 the hook does
    nothing.
    """

    def __init__(
        self,
        key="aux_scale",
        min_hold=1000,
        ramp=1000,
        gate_depth_loss=0.20,
        gate_clamp_frac=0.20,
        max_hold=3000,
        ema=0.01,
        log_interval=50,
        gate_key="drop_frac",
    ):
        self.key = key
        self.gate_key = str(gate_key)
        self.min_hold = min_hold
        self.ramp = ramp
        self.gate_depth_loss = gate_depth_loss
        self.gate_clamp_frac = gate_clamp_frac
        self.max_hold = max_hold
        self.ema = ema
        self.log_interval = log_interval
        self.final = None
        self.progress = 0.0
        self.opened = False  # the gate has been open at least once
        self.averages = None  # EMAs of (depth_loss, gate_key)
        self.curr_iter = 0

    @property
    def criterion(self):
        model = self.trainer.model
        return getattr(model, "module", model).criterion

    def state_dict(self):
        return dict(progress=self.progress, opened=self.opened, averages=self.averages)

    def load_state_dict(self, state):
        self.progress = float(state["progress"])
        self.opened = bool(state["opened"])
        averages = state["averages"]
        self.averages = None if averages is None else [float(a) for a in averages]

    def before_train(self):
        criterion = self.criterion
        if not hasattr(criterion, self.key):
            raise AttributeError(
                f"criterion {type(criterion).__name__} has no {self.key}"
            )
        self.final = float(getattr(criterion, self.key))
        self.curr_iter = (
            self.trainer.start_epoch * len(self.trainer.train_loader)
            + self.trainer.start_iter
        )
        state = None
        if self.trainer.cfg.resume:  # CheckpointLoader.resume ran first
            state = self.trainer.comm_info.get("resume_hook_state", {}).get(
                type(self).__name__
            )
            if state is not None:
                self.load_state_dict(state)
        self.set_weight()
        self.trainer.logger.info(
            f"CriterionWeightWarmup: {self.key} {self.final} -> "
            f"{getattr(criterion, self.key):.4f} (progress {self.progress:.4f} at iter "
            f"{self.curr_iter}, {'restored from the checkpoint' if state else 'fresh'}); "
            f"gate depth_loss <= {self.gate_depth_loss}, {self.gate_key} <= "
            f"{self.gate_clamp_frac} after {self.min_hold} iterations, ramp {self.ramp}"
        )

    def set_weight(self):
        setattr(self.criterion, self.key, self.final * self.progress)

    def after_step(self):
        self.curr_iter += 1
        if self.progress >= 1.0:
            return
        out = self.trainer.comm_info["model_output_dict"]
        if "depth_loss" not in out or self.gate_key not in out:
            raise KeyError(
                f"CriterionWeightWarmup gates on depth_loss and {self.gate_key}: the "
                f"model output has {sorted(out)}"
            )
        depth_loss = out["depth_loss"].detach().float().reshape(())
        gate_value = (
            out[self.gate_key].detach().float().reshape(()).to(depth_loss.device)
        )
        values = torch.stack([depth_loss, gate_value])
        if comm.get_world_size() > 1:
            torch.distributed.all_reduce(values)
            values /= comm.get_world_size()
        values = values.tolist()
        if self.averages is None:
            self.averages = values
        else:
            self.averages = [
                a + self.ema * (v - a) for a, v in zip(self.averages, values)
            ]
        depth_loss, gate_value = self.averages
        gate = (
            self.curr_iter >= self.min_hold
            and depth_loss <= self.gate_depth_loss
            and gate_value <= self.gate_clamp_frac
        )
        if gate:
            self.opened = True
            self.progress = min(1.0, self.progress + 1.0 / self.ramp)
            self.set_weight()
        if self.progress >= 1.0 or self.curr_iter % self.log_interval == 0:
            weight = getattr(self.criterion, self.key)
            self.trainer.logger.info(
                f"CriterionWeightWarmup: iter {self.curr_iter} {self.key} {weight:.4f} "
                f"progress {self.progress:.4f} gate {'open' if gate else 'closed'} "
                f"depth_loss_ema {depth_loss:.4f} {self.gate_key}_ema {gate_value:.4f}"
            )
            if not self.opened and self.curr_iter >= self.max_hold:
                self.trainer.logger.warning(
                    f"CriterionWeightWarmup: gate never opened in {self.curr_iter} "
                    f"iterations (max_hold {self.max_hold}): {self.key} stays 0"
                )
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar(
                    f"params/{self.key}", weight, self.curr_iter
                )
