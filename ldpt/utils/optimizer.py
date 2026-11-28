"""
Optimizer

``build_optimizer`` builds ``cfg.optimizer`` over the keyword groups of ``cfg.param_dicts``
(every parameter in the first group whose keyword occurs in its name, else group 0). A
keyword is a name substring or a list of them (any one occurs), e.g. a no-weight-decay
group ``dict(keyword=["norm", "tap_bn", "bias", "gamma"], weight_decay=0.0)`` after the
encoder's group.

``MuonAdamW`` (``optimizer=dict(type="MuonAdamW", ...)``, the segmentation recipes
``configs/cocostuff/semseg-l-dpt-v1m1-1-large.py`` and its anneal): the hidden 2-D
weights step with Nesterov momentum orthogonalised by Newton-Schulz iterations (Keller
Jordan's optimizer for hidden layers), every other parameter with torch's AdamW, in one
optimizer (one state, one checkpoint). Every keyword group is split in two: its AdamW
part (group order and keys as with AdamW, so ``param_groups[0]`` is still the default
group) and its matrix part (``muon=True``). ``type="AdamW"`` does not touch any of it.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import contextlib
import copy
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ldpt.utils.logger import get_root_logger
from ldpt.utils.registry import Registry

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.SGD, name="SGD")
OPTIMIZERS.register_module(module=torch.optim.Adam, name="Adam")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")


def _keywords(rule) -> Tuple[str, ...]:
    """A param_dicts rule's keyword as a tuple: one name substring or a non-empty list
    of them."""
    keyword = rule["keyword"]
    words = (keyword,) if isinstance(keyword, str) else keyword
    if (
        not isinstance(words, (list, tuple))
        or not words
        or not all(isinstance(k, str) and k for k in words)
    ):
        raise ValueError(
            f"param_dicts keyword is a name substring or a list of them, got {keyword!r}"
        )
    return tuple(words)


def keyword_groups(model, param_dicts, lr) -> List[Dict[str, Any]]:
    """cfg.param_dicts as param groups with their parameter names: group 0 (lr) takes
    every parameter no keyword matches; group i + 1 takes those whose name contains
    param_dicts[i].keyword (a substring, or any of a list of them; first match wins)
    with its lr / momentum / weight_decay."""
    groups = [dict(names=[], params=[], lr=lr)]
    words = [_keywords(rule) for rule in param_dicts]
    for rule in param_dicts:
        group = dict(names=[], params=[])
        for key in ("lr", "momentum", "weight_decay"):
            if key in rule.keys():
                group[key] = rule[key]
        groups.append(group)
    for n, p in model.named_parameters():
        i = next((i + 1 for i, ks in enumerate(words) if any(k in n for k in ks)), 0)
        groups[i]["names"].append(n)
        groups[i]["params"].append(p)
    return groups


def build_optimizer(cfg, model, param_dicts=None):
    cfg = copy.deepcopy(cfg)
    if cfg.get("type") == "MuonAdamW":
        return build_muon_adamw(cfg, model, param_dicts)
    if param_dicts is None:
        cfg.params = model.parameters()
    else:
        cfg.params = keyword_groups(model, param_dicts, cfg.lr)
        logger = get_root_logger()
        for i in range(len(cfg.params)):
            param_names = cfg.params[i].pop("names")
            message = ""
            for key in cfg.params[i].keys():
                if key != "params":
                    message += f" {key}: {cfg.params[i][key]};"
            logger.info(f"Params Group {i+1} -{message} Params: {param_names}.")
    return OPTIMIZERS.build(cfg=cfg)


# ---------------------------------------------------------------- MuonAdamW

# the quintic Newton-Schulz coefficients of Keller Jordan's implementation (also
# PyTorch's from 2.9)
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_EPS = 1e-7
MUON_SCALES = ("match_rms", "spectral")
# build keys that route parameters (consumed by muon_param_groups, not the optimizer)
ROUTING_KEYS = (
    "muon_lr",
    "muon_weight_decay",
    "muon_exclude",
    "muon_split",
    "muon_min_dim",
)
DEFAULT_MUON_EXCLUDE = ("encoder.",)  # a trained encoder stays on AdamW


def newton_schulz(x: Tensor, steps: int = 5) -> Tensor:
    """Every matrix of x orthogonalised by `steps` quintic Newton-Schulz iterations in
    bf16 with autocast off: Frobenius normalisation, a tall matrix transposed. The
    result is about U V^T of x = U S V^T, singular values about 0.7-1.2 (the
    coefficients trade accuracy for speed).
    Shapes: x [B, R, C], any float dtype -> [B, R, C] bf16."""
    if x.ndim != 3:
        raise ValueError(f"newton_schulz takes [B, R, C], got {tuple(x.shape)}")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps < 100:
        raise ValueError(f"newton_schulz needs an int 1 <= steps < 100, got {steps!r}")
    a, b, c = NS_COEFFICIENTS
    device_type = x.device.type
    off = (
        torch.autocast(device_type, enabled=False)
        if torch.amp.is_autocast_available(device_type)
        else contextlib.nullcontext()
    )
    with off:
        y = x.to(torch.bfloat16)
        tall = y.size(-2) > y.size(-1)
        if tall:
            y = y.mT
        y = y / torch.linalg.vector_norm(y, dim=(-2, -1), keepdim=True).clamp_min(
            NS_EPS
        )
        for _ in range(steps):
            gram = y @ y.mT
            y = torch.baddbmm(
                y, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), y, beta=a
            )
        return y.mT if tall else y


def muon_scale(mode: str, rows: int, cols: int) -> float:
    """The factor on the orthogonalised step of a rows x cols block: "match_rms"
    0.2 sqrt(max(rows, cols)) (Moonshot AI's rule: the RMS of an AdamW step, so AdamW's
    lr carries over; the segmentation recipes), "spectral" sqrt(max(1, rows / cols))
    (Keller Jordan's: the same relative spectral step at every width; needs its own lr,
    about 0.005-0.02)."""
    if mode == "match_rms":
        return 0.2 * math.sqrt(max(rows, cols))
    if mode == "spectral":
        return math.sqrt(max(1.0, rows / cols))
    raise ValueError(f"muon_scale must be one of {MUON_SCALES}, got {mode!r}")


def _muon_route(name, p, tables, exclude, split, min_dim):
    """(row blocks, reason): row blocks > 0 routes p to the matrix part; else reason
    says why it stays on AdamW (None for a tensor that is not a 2-D weight)."""
    if p.ndim != 2 or name.rsplit(".", 1)[-1] != "weight":
        return 0, None
    if not p.requires_grad:
        return 0, "frozen"
    if id(p) in tables:
        return 0, "embedding table"
    hit = next((k for k in exclude if k in name), None)
    if hit is not None:
        return 0, f"excluded by {hit!r}"
    blocks = next((n for key, n in split if key in name), 1)
    if p.shape[0] % blocks:
        raise ValueError(
            f"muon_split cuts {name} {tuple(p.shape)} into {blocks} row blocks, but "
            f"its {p.shape[0]} rows do not divide"
        )
    if min(p.shape[0] // blocks, p.shape[1]) < min_dim:
        return 0, f"a side below {min_dim}"
    if not bool(p.detach().any()):
        # an orthogonalised first step would be full-size whatever the gradient's scale
        return 0, "zero at build"
    return blocks, None


def muon_param_groups(
    model: nn.Module,
    param_dicts=None,
    *,
    lr: float,
    weight_decay: float,
    muon_lr: Optional[float] = None,
    muon_weight_decay: Optional[float] = None,
    muon_exclude: Sequence[str] = DEFAULT_MUON_EXCLUDE,
    muon_split: Sequence[Tuple[str, int]] = (),
    muon_min_dim: int = 16,
) -> Tuple[List[Dict[str, Any]], str]:
    """(param groups for MuonAdamW, the routing in a few log lines).

    The keyword groups of param_dicts (keyword_groups), each split in two, in order:
    its AdamW part (always, keys as keyword_groups gives them) and, when not empty, its
    matrix part. A parameter goes to the matrix part when it is a trainable 2-D
    ``*.weight``, not an nn.Embedding table, no muon_exclude keyword occurs in its
    name, both sides of its row blocks are >= muon_min_dim, and it is not all zeros at
    build time. The first muon_split (keyword, n) rule whose keyword occurs in its name
    cuts it into n row blocks orthogonalised one by one (a fused qkv weight as q, k, v).

    A matrix part has lr = muon_lr (None: its source group's lr) and weight decay
    muon_weight_decay (None: the source group's decay per step, wd x lr_src / lr_muon).
    Every group carries muon (bool), source (its keyword group) and lr_factor (its lr
    over the source group's: what OneCycleLR's per-keyword-group max_lr is scaled by).
    """
    if isinstance(muon_exclude, str):
        raise ValueError(
            f"muon_exclude is a list of name keywords, got {muon_exclude!r}"
        )
    split = [tuple(rule) for rule in muon_split]
    for rule in split:
        if (
            len(rule) != 2
            or not isinstance(rule[0], str)
            or isinstance(rule[1], bool)
            or not isinstance(rule[1], int)
            or rule[1] < 1
        ):
            raise ValueError(
                f"muon_split entries are (keyword, row blocks >= 1), got {rule}"
            )
    if muon_lr is not None and not muon_lr > 0:
        raise ValueError(f"muon_lr must be > 0 or None, got {muon_lr!r}")
    if muon_weight_decay is not None and not muon_weight_decay >= 0:
        raise ValueError(
            f"muon_weight_decay must be >= 0 or None, got {muon_weight_decay!r}"
        )
    sources = keyword_groups(model, param_dicts or [], lr)
    tables = {
        id(m.weight)
        for m in model.modules()
        if isinstance(m, (nn.Embedding, nn.EmbeddingBag))
    }
    groups, kept = [], {}
    muon_shapes, n_muon, n_adam, t_adam = Counter(), 0, 0, 0
    for i, source in enumerate(sources):
        if "momentum" in source:
            raise ValueError(
                f"param_dicts[{i - 1}] sets momentum: MuonAdamW's AdamW part has betas, "
                "its matrix part muon_momentum"
            )
        adam = {k: v for k, v in source.items() if k not in ("names", "params")}
        adam.update(params=[], muon=False, source=i, lr_factor=1.0)
        muon = dict(params=[], muon=True, source=i, muon_splits=[])
        for n, p in zip(source["names"], source["params"]):
            blocks, reason = _muon_route(
                n, p, tables, list(muon_exclude), split, muon_min_dim
            )
            if blocks:
                muon["params"].append(p)
                muon["muon_splits"].append(blocks)
                muon_shapes[(p.shape[0] // blocks, p.shape[1])] += blocks
                n_muon += p.numel()
            else:
                adam["params"].append(p)
                if reason is not None:
                    kept.setdefault(reason, []).append(f"{n} {tuple(p.shape)}")
                if p.requires_grad:
                    n_adam += p.numel()
                    t_adam += 1
        groups.append(adam)
        if muon["params"]:
            src_lr = source.get("lr", lr)
            src_wd = source.get("weight_decay", weight_decay)
            rate = src_lr if muon_lr is None else float(muon_lr)
            if muon_weight_decay is not None:
                decay = float(muon_weight_decay)
            elif rate == src_lr:
                decay = src_wd
            else:  # the same decay per step as the source group's AdamW
                decay = src_wd * src_lr / rate
            muon.update(lr=rate, weight_decay=decay, lr_factor=rate / src_lr)
            groups.append(muon)
    total = n_muon + n_adam
    lines = [
        f"MuonAdamW: orthogonalised steps on "
        f"{sum(len(g['params']) for g in groups if g['muon'])} "
        f"matrices, {n_muon:,} of {total:,} trainable parameters "
        f"({100 * n_muon / max(total, 1):.1f}%), {sum(muon_shapes.values())} blocks: "
        + ", ".join(f"{r}x{c} x{k}" for (r, c), k in sorted(muon_shapes.items()))
        + f"; AdamW on {t_adam} trainable tensors, {n_adam:,} parameters"
    ]
    for j, g in enumerate(groups):
        if g["muon"]:
            lines.append(
                f"  group {j} (matrices, keyword group {g['source']}): "
                f"{len(g['params'])} matrices, lr {g['lr']:g}, weight decay "
                f"{g['weight_decay']:g}"
            )
    for reason, names in kept.items():
        if reason == "frozen":
            lines.append(f"  kept on AdamW, frozen: {len(names)} 2-D weights")
        else:
            lines.append(f"  kept on AdamW, {reason}: {', '.join(names)}")
    names = [n for n, p in model.named_parameters() if p.ndim == 2]
    unused = [k for k in muon_exclude if not any(k in n for n in names)]
    unused += [k for k, _ in split if not any(k in n for n in names)]
    if unused:
        lines.append(f"  muon_exclude / muon_split keywords matching nothing: {unused}")
    return groups, "\n".join(lines)


def build_muon_adamw(cfg, model, param_dicts=None):
    """MuonAdamW over muon_param_groups (cfg's routing keys); the rest of cfg (lr,
    weight_decay, betas, eps, ..., muon_momentum, nesterov, ns_steps, muon_scale) goes
    to MuonAdamW. Logs the groups and the routing."""
    cfg = copy.deepcopy(cfg)
    routing = {k: cfg.pop(k) for k in ROUTING_KEYS if k in cfg}
    groups, description = muon_param_groups(
        model,
        param_dicts,
        lr=cfg.lr,
        weight_decay=cfg.get("weight_decay", 1e-2),  # torch AdamW's default
        **routing,
    )
    logger = get_root_logger()
    names = {id(p): n for n, p in model.named_parameters()}
    for i, g in enumerate(groups):
        message = "".join(
            f" {k}: {v};" for k, v in g.items() if k not in ("params", "muon_splits")
        )
        logger.info(
            f"Params Group {i+1} -{message} Params: {[names[id(p)] for p in g['params']]}."
        )
    logger.info(description)
    cfg.params = groups
    optimizer = OPTIMIZERS.build(cfg=cfg)
    optimizer.routing = description
    return optimizer


@OPTIMIZERS.register_module()
class MuonAdamW(torch.optim.Optimizer):
    """Orthogonalised momentum on the groups with muon=True (the matrix groups),
    torch's AdamW on the others, in one optimizer.

    The AdamW groups are stepped by an inner torch.optim.AdamW built with this
    optimizer's AdamW keyword arguments over the same group dicts and the same state:
    bit for bit torch.optim.AdamW over those groups.

    A matrix group (muon_splits: the row blocks of each matrix) keeps a
    momentum_buffer per matrix in the parameter's dtype (fp32) and steps as:
    B <- lerp(B, g, 1 - mu); O = newton_schulz(lerp(g, B, mu)) per row block (Nesterov;
    nesterov=False: of B); W <- W (1 - lr wd) - lr s O, s = muon_scale(rows, cols) of the
    block (mu = muon_momentum). Blocks of one shape are stacked: one bf16 Newton-Schulz
    chain per shape.

    Schedules: a matrix group has lr and weight_decay (written by schedules) but no
    betas or momentum: muon_momentum is its own key. ldpt's OneCycleLR leaves it alone
    (it cycles the AdamW groups' beta1 only). DDP: each rank orthogonalises the same
    all-reduced gradients. Gradient clipping does not bound an orthogonalised step (the
    orthogonalisation removes the gradient's scale); a GradScaler with inf / nan skips
    the whole step (host check, as for non-fused AdamW).

    load_state_dict takes MuonAdamW states with the same routing only: an AdamW state
    (or another routing) raises.
    """

    routing: Optional[str] = None

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        muon_momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        muon_scale: str = "match_rms",
        **adamw_kwargs,
    ):
        if not 0.0 <= muon_momentum < 1.0:
            raise ValueError(f"muon_momentum must be in [0, 1), got {muon_momentum!r}")
        if isinstance(ns_steps, bool) or not isinstance(ns_steps, int):
            raise ValueError(f"ns_steps must be an int, got {ns_steps!r}")
        if not 1 <= ns_steps < 100:
            raise ValueError(f"ns_steps must be in [1, 100), got {ns_steps}")
        if muon_scale not in MUON_SCALES:
            raise ValueError(
                f"muon_scale must be one of {MUON_SCALES}, got {muon_scale!r}"
            )
        groups = list(params)
        if not groups or not all(isinstance(g, dict) for g in groups):
            raise ValueError(
                "MuonAdamW takes param groups (muon_param_groups / build_optimizer)"
            )
        self._muon_defaults = dict(
            muon_momentum=float(muon_momentum),
            nesterov=bool(nesterov),
            ns_steps=ns_steps,
            muon_scale=muon_scale,
        )
        adam = [g for g in groups if not g.get("muon", False)]
        self._adamw = (
            torch.optim.AdamW(adam, lr=lr, weight_decay=weight_decay, **adamw_kwargs)
            if adam
            else None
        )
        betas = adamw_kwargs.get("betas", (0.9, 0.999))
        # betas in the defaults: OneCycleLR cycles beta1 (of the AdamW groups)
        super().__init__(
            groups, dict(lr=lr, weight_decay=weight_decay, betas=tuple(betas))
        )
        self._bind()

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if not group.setdefault("muon", False):
            if getattr(self, "_adamw", None) is None:
                raise ValueError("MuonAdamW: AdamW groups are given at construction")
            if not any(group is g for g in self._adamw.param_groups):
                self._adamw.add_param_group(group)
            return
        group.pop("betas", None)
        for key, value in self._muon_defaults.items():
            group.setdefault(key, value)
        splits = group.setdefault("muon_splits", [1] * len(group["params"]))
        if len(splits) != len(group["params"]):
            raise ValueError(
                f"muon_splits has {len(splits)} entries for {len(group['params'])} "
                "parameters"
            )
        for p, k in zip(group["params"], splits):
            if p.ndim != 2 or isinstance(k, bool) or not isinstance(k, int) or k < 1:
                raise ValueError(
                    "a matrix group takes 2-D parameters with an int >= 1 of row "
                    f"blocks, got {tuple(p.shape)} with {k!r}"
                )
            if p.shape[0] % k:
                raise ValueError(f"{tuple(p.shape)} does not cut into {k} row blocks")

    def _bind(self) -> None:
        """Point the inner AdamW at this optimizer's AdamW groups and state (after
        construction, load_state_dict and unpickling replace them)."""
        if self._adamw is not None:
            self._adamw.param_groups = [g for g in self.param_groups if not g["muon"]]
            self._adamw.state = self.state

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state.update(
            _adamw=self._adamw, _muon_defaults=self._muon_defaults, routing=self.routing
        )
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        if hasattr(self, "_adamw"):
            self._bind()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """torch's, for a MuonAdamW state of the same routing only (else ValueError:
        an AdamW state would load into the wrong groups or not at all)."""
        saved = state_dict["param_groups"]
        state = state_dict["state"]
        # per group: muon, row blocks, parameter count; per parameter: the shapes of
        # its state tensors (torch checks neither)
        mine = [
            (g["muon"], list(g.get("muon_splits", [])), len(g["params"]))
            for g in self.param_groups
        ]
        theirs = [
            (g.get("muon"), list(g.get("muon_splits", [])), len(g["params"]))
            for g in saved
        ]
        moved = [
            i
            for g, h in zip(self.param_groups, saved)
            for p, i in zip(g["params"], h["params"])
            if any(
                torch.is_tensor(v) and v.dim() > 0 and v.shape != p.shape
                for v in state.get(i, {}).values()
            )
        ]
        if mine != theirs or moved:
            kind = (
                "an AdamW (not a MuonAdamW) state"
                if not any("muon" in g for g in saved)
                else "a MuonAdamW state of another routing"
            )
            raise ValueError(
                f"MuonAdamW.load_state_dict: the checkpoint holds {kind}; groups "
                f"(muon, row blocks, params) {theirs} vs this optimizer's {mine}"
                + (f", state of other shapes for parameters {moved}" if moved else "")
                + ". Resume with the optimizer the run was started with, or load the "
                "weights only and start the optimizer fresh"
            )
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._adamw is not None:
            self._adamw.step()
        self._muon_step()
        return loss

    def _muon_step(self) -> None:
        # (rows, cols, device, ns steps) -> [(group, param, Nesterov input as bf16
        # [blocks, rows, cols])]
        stacks: Dict[tuple, list] = {}
        for group in self.param_groups:
            if not group["muon"]:
                continue
            mu = group["muon_momentum"]
            for p, k in zip(group["params"], group["muon_splits"]):
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("MuonAdamW does not support sparse gradients")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, memory_format=torch.contiguous_format
                    )
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1.0 - mu)
                u = p.grad.lerp(buf, mu) if group["nesterov"] else buf
                rows, cols = p.shape[0] // k, p.shape[1]
                key = (rows, cols, p.device, group["ns_steps"])
                stacks.setdefault(key, []).append(
                    (group, p, u.reshape(k, rows, cols).to(torch.bfloat16))
                )
        for (rows, cols, _, ns_steps), entries in stacks.items():
            ortho = newton_schulz(torch.cat([e[2] for e in entries]), ns_steps)
            pieces = ortho.split([e[2].shape[0] for e in entries])
            for (group, p, _), o in zip(entries, pieces):
                lr, wd = group["lr"], group["weight_decay"]
                if lr * wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                scale = muon_scale(group["muon_scale"], rows, cols)
                p.add_(o.reshape(p.shape).to(p.dtype), alpha=-lr * scale)
