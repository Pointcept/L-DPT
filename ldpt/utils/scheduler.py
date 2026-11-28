"""
Scheduler

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import contextlib
import copy

import torch.optim.lr_scheduler as lr_scheduler
from .registry import Registry

SCHEDULERS = Registry("schedulers")


def _per_group(optimizer, value, name, scale=False):
    """A MuonAdamW value given per keyword group (cfg.param_dicts order, as for AdamW)
    or as one number, expanded to one per param group (each keyword group's AdamW
    and matrix parts; scale: times the group's lr_factor, so a matrix part with its
    own muon_lr keeps its ratio to the keyword group's lr). Other optimizers:
    unchanged."""
    groups = optimizer.param_groups
    if not any("source" in g for g in groups):
        return value
    n = max(g["source"] for g in groups) + 1
    if not isinstance(value, (list, tuple)):
        value = [value] * n
    elif len(value) != n:
        raise ValueError(
            f"OneCycleLR {name}: give one value per keyword group ({n}: "
            f"cfg.param_dicts + 1), got {len(value)}"
        )
    return [value[g["source"]] * (g["lr_factor"] if scale else 1.0) for g in groups]


@contextlib.contextmanager
def _muon_momentum_free(optimizer, cycle_momentum):
    """torch's OneCycleLR writes beta1 (or momentum) into every param group; a
    MuonAdamW matrix group has neither (its momentum is muon_momentum): it gets a
    throwaway key for the duration, removed after. The AdamW groups cycle as usual."""
    muon = (
        [g for g in optimizer.param_groups if g.get("muon")] if cycle_momentum else []
    )
    key = "betas" if "betas" in optimizer.defaults else "momentum"
    for g in muon:
        g[key] = (0.0, 0.0) if key == "betas" else 0.0
    try:
        yield
    finally:
        for g in muon:
            g.pop(key, None)


@SCHEDULERS.register_module()
class OneCycleLR(lr_scheduler.OneCycleLR):
    r"""
    torch.optim.lr_scheduler.OneCycleLR, Block total_steps

    With MuonAdamW: max_lr (and the momentum bounds) per keyword group or one number;
    the lr of every matrix group follows the cycle, its momentum (muon_momentum) does
    not.
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
    ):
        max_lr = _per_group(optimizer, max_lr, "max_lr", scale=True)
        base_momentum = _per_group(optimizer, base_momentum, "base_momentum")
        max_momentum = _per_group(optimizer, max_momentum, "max_momentum")
        with _muon_momentum_free(optimizer, cycle_momentum):
            super().__init__(
                optimizer=optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
                cycle_momentum=cycle_momentum,
                base_momentum=base_momentum,
                max_momentum=max_momentum,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                three_phase=three_phase,
                last_epoch=last_epoch,
            )

    def get_lr(self):
        with _muon_momentum_free(self.optimizer, self.cycle_momentum):
            return super().get_lr()


def build_scheduler(cfg, optimizer):
    cfg = copy.deepcopy(cfg)
    cfg.optimizer = optimizer
    return SCHEDULERS.build(cfg=cfg)
