"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import itertools
import weakref
import wandb
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
from pathlib import Path

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .default import worker_init_fn, create_ddp_model
from ldpt.hooks import build_hooks, HookBase
from ldpt.datasets import build_dataset
from ldpt.models import build_model
from ldpt.utils import comm
from ldpt.utils.events import EventStorage, ExceptionWriter
from ldpt.utils.logger import get_root_logger
from ldpt.utils.optimizer import build_optimizer
from ldpt.utils.scheduler import build_scheduler
from ldpt.utils.device import get_device
from ldpt.utils.registry import Registry
from ldpt.utils.sdpa import sdpa_context

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.start_iter = 0  # batches of start_epoch already trained (mid-epoch resume)
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.writer: SummaryWriter
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


class ResumableSampler(torch.utils.data.distributed.DistributedSampler):
    """DistributedSampler that can start an epoch part-way (mid-epoch resume).

    The order is DistributedSampler's, a function of (seed, epoch) only, so a resumed
    run sees the same batches. DefaultTrainer.before_epoch sets start_index to skip the
    indices of the batches already trained; len() stays the full epoch, so
    len(train_loader) and comm_info["iter"] keep counting the whole epoch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.start_index, None)


@TRAINERS.register_module()
class DefaultTrainer(TrainerBase):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.cfg = cfg
        if cfg.get("allow_tf32") is not None:
            torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
        self.logger.info("=> Loading writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Loading config ...")
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def before_train(self):
        super().before_train()
        self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")

    def before_epoch(self):
        super().before_epoch()
        # resumed from a mid-epoch checkpoint: skip the batches it already trained
        start_iter = self.start_iter if self.epoch == self.start_epoch else 0
        sampler = self.train_loader.sampler
        if isinstance(sampler, ResumableSampler):
            sampler.set_epoch(self.epoch)
            sampler.start_index = start_iter * self.train_loader.batch_size
        elif start_iter:
            raise RuntimeError(
                "resuming mid-epoch needs the ResumableSampler, which single-process "
                "runs use only with CheckpointSaver(save_freq_iter=...)"
            )
        if start_iter:
            self.logger.info(
                f"Resume epoch {self.epoch + 1} at iter {start_iter}: skip "
                f"{start_iter} of {len(self.train_loader)} batches"
            )
        self.model.train()
        self.data_iterator = enumerate(self.train_loader, start=start_iter)

    def run_step(self):
        device = get_device()
        auto_cast = partial(torch.amp.autocast, device_type=device.type)

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(
                    device, non_blocking=(device.type == "cuda")
                )

        with (
            sdpa_context(self.cfg.get("sdpa_backends")),
            auto_cast(enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]),
        ):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                self.clip_grad()
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                self.clip_grad()
            self.optimizer.step()
            self.scheduler.step()
        self.comm_info["model_output_dict"] = output_dict

    def clip_grad(self):
        """Clip the gradient norm to cfg.clip_grad; the total norm before clipping
        goes to comm_info["grad_norm"] (InformationWriter logs it as grad_norm)."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.clip_grad
        )
        self.comm_info["grad_norm"] = grad_norm.detach()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if self.cfg.get("compile_blocks"):
            # training steps only (model.compile_blocks); evaluation stays eager
            options = self.cfg.compile_blocks
            model.compile_blocks(**(options if isinstance(options, dict) else {}))
            self.logger.info(f"Compiled blocks for the training steps: {options}")
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")

        device = get_device()
        model = create_ddp_model(
            model.to(device),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            run_id_file = Path(self.cfg.save_path) / "wandb" / "wandb_run_id.txt"

            # Try to resume existing wandb run
            if self.cfg.resume and run_id_file.exists():
                run_id = run_id_file.read_text().strip()
                wandb.init(
                    project=self.cfg.wandb_project,
                    id=run_id,
                    resume="must",
                    dir=self.cfg.save_path,
                    settings=wandb.Settings(api_key=self.cfg.wandb_key),
                )
                self.logger.info(f"Resuming wandb run with id: {run_id}")
                return writer

            # Start a new wandb run
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
            # Save run_id for future resume
            run_id_file.write_text(wandb.run.id)
            self.logger.info(f"Started new wandb run with id: {wandb.run.id}")
        return writer

    def build_train_loader(self):
        train_dataset = build_dataset(self.cfg.data.train)
        self.logger.info(train_dataset.info())
        # mid-epoch checkpoints need an order that depends on the epoch only;
        # single-process runs otherwise keep RandomSampler (global RNG)
        saves_mid_epoch = any(
            h.get("type") == "CheckpointSaver" and h.get("save_freq_iter")
            for h in self.cfg.hooks
        )
        if comm.get_world_size() > 1 or saves_mid_epoch:
            train_sampler = ResumableSampler(
                train_dataset, num_replicas=comm.get_world_size(), rank=comm.get_rank()
            )
        else:
            train_sampler = None

        init_fn = partial(
            worker_init_fn,
            num_workers=self.cfg.num_workers_per_gpu,
            rank=comm.get_rank(),
            seed=self.cfg.seed,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_workers_per_gpu,
            sampler=train_sampler,
            pin_memory=True if get_device().type != "mps" else False,
            worker_init_fn=init_fn,
            drop_last=len(train_dataset) > self.cfg.batch_size,
            # persistent workers need worker processes (num_workers=0 raises)
            persistent_workers=self.cfg.num_workers_per_gpu > 0,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_dataset = build_dataset(self.cfg.data.val)
            self.logger.info(val_dataset.info())
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset
                )
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_workers_per_gpu,
                pin_memory=True if get_device().type != "mps" else False,
                sampler=val_sampler,
                worker_init_fn=partial(
                    worker_init_fn,
                    num_workers=self.cfg.num_workers_per_gpu,
                    rank=comm.get_rank(),
                    seed=None,
                ),
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        device = get_device()
        grad_scaler = partial(torch.amp.GradScaler, device=device)
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


def build_trainer(cfg):
    """Build trainer."""
    return TRAINERS.build(cfg)
