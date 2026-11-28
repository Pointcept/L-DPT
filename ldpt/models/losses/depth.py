"""
Depth Losses: the L-DPT depth loss (relative_depth_loss, docs/depth_loss.md)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import contextlib
import math
from collections.abc import Mapping
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp.autocast_mode import is_autocast_available

from .builder import LOSSES
from ..utils.depth import nonneg_lsq_scale_and_shift, roe_fit_sample, sample_valid


def _scharr_kernels(device, dtype) -> Tuple[Tensor, Tensor]:
    kx = (
        torch.tensor(
            [[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]],
            device=device,
            dtype=dtype,
        )
        / 32.0
    )
    ky = (
        torch.tensor(
            [[3.0, 10.0, 3.0], [0.0, 0.0, 0.0], [-3.0, -10.0, -3.0]],
            device=device,
            dtype=dtype,
        )
        / 32.0
    )
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def _conv_same(x: Tensor, k: Tensor) -> Tensor:
    return F.conv2d(x, k, padding=1)


def _gaussian_kernel_1d(device, dtype, ksize=5, sigma=1.0) -> Tensor:
    assert ksize % 2 == 1
    r = ksize // 2
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-(xs**2) / (2.0 * sigma**2))
    g = g / (g.sum() + 1e-12)
    return g


def _gaussian_blur(x: Tensor, ksize=5, sigma=1.0) -> Tensor:
    g = _gaussian_kernel_1d(x.device, x.dtype, ksize=ksize, sigma=sigma)
    x = F.conv2d(x, g.view(1, 1, 1, ksize), padding=(0, ksize // 2))
    x = F.conv2d(x, g.view(1, 1, ksize, 1), padding=(ksize // 2, 0))
    return x


def _down2(x: Tensor) -> Tensor:
    x = _gaussian_blur(x, ksize=5, sigma=1.0)
    return F.avg_pool2d(x, kernel_size=2, stride=2)


def _erode(mask: Tensor, ksize: int) -> Tensor:
    """[B, 1, H, W] 0/1 mask -> pixels whose ksize x ksize neighbourhood is valid;
    pixels outside the image count as invalid (the convolutions zero-pad there)."""
    r = ksize // 2
    invalid = F.pad(1.0 - mask, (r, r, r, r), value=1.0)
    return 1.0 - F.max_pool2d(invalid, kernel_size=ksize, stride=1)


def _down2_mask_erode(mask: Tensor) -> Tensor:
    """Mask of the next level of the gradient term's pyramid: every pixel of the 2 x 2
    cell has a fully valid 5 x 5 neighbourhood (the footprint of _down2's blur), so
    a value of the next level is made of valid pixels only."""
    full = _erode(mask, 5)
    return 1.0 - F.max_pool2d(1.0 - full, kernel_size=2, stride=2)


def num_kept(num: int, trim: float) -> int:
    """ceil((1 - trim) * num): pixels a trim of the fraction trim keeps (float noise
    of the product rounded away)."""
    return math.ceil(round((1.0 - trim) * num, 6))


def smallest_k(values: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """(mask of exactly k of the 1-D values, the k-th smallest value theta): every
    value < theta and, of the values equal to theta, the first ones in order (a tie
    at theta never keeps more than k, so a trim always drops len - k values)."""
    theta = values.kthvalue(k).values
    below = values < theta
    tied = values == theta
    return below | (tied & (tied.cumsum(0) <= k - below.sum())), theta


# ------------------------------------------------------- the L-DPT depth loss: helpers
# (docs/depth_loss.md sections 2 and 4)


def pixel_weights(target: Tensor, mask: Tensor, cap: float) -> Tuple[Tensor, Tensor]:
    """(w [B, H, W], h [B]), fp64: w = min(1 / g, cap / h) on the mask, 0 off it;
    h = N / sum_M (1 / g), the harmonic-mean depth of the image (its unit)."""
    g = target.double()
    inv = torch.where(mask, 1.0 / g.clamp_min(1e-30), 0.0)
    num = mask.flatten(1).sum(dim=1).clamp_min(1)
    mean_inv = inv.flatten(1).sum(dim=1) / num
    w = torch.minimum(inv, cap * mean_inv[:, None, None])
    return w, 1.0 / mean_inv.clamp_min(1e-30)


@torch.no_grad()
def refit_weights(p, g, w, keep, s0, t0, trunc: float, floor: float) -> Tensor:
    """[B, P] constant weights of the refit (p, g, w, keep: [B, P], fp64; s0, t0:
    [B]): omega = w^2 / max(e, floor), e = w |s0 p + t0 - g|, 0 where e > trunc or
    the pixel is not kept (or s0, t0 are NaN)."""
    e = w * (s0[:, None] * p + t0[:, None] - g).abs()
    omega = w.square() / e.clamp_min(floor)
    return torch.where(keep & (e <= trunc), omega, 0.0)


def refit_in_graph(
    pred: Tensor,
    target: Tensor,
    w: Tensor,
    keep: Tensor,
    s0: Tensor,
    t0: Tensor,
    trunc: float,
    floor: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """(s, t, ok) [B], fp64: one IRLS step of the truncated weighted L1 objective
    from the ROE solution (s0, t0), as a weighted least-squares fit over the kept
    pixels inside autograd. Weights (constants) omega = w^2 / max(e, floor), e =
    w |s0 p + t0 - g|, and 0 where e > trunc or the pixel is not kept; centred fp64
    moments. ok is False where sum omega = 0, the variance is at most 1e-12 of the
    weighted second moment sum omega p^2 (a rounding residue, e.g. when every
    weighted pixel has the same prediction), s is not finite or s <= 0 (no division
    by a degenerate value, not even in the unselected branch). With constant omega
    the fit is equivariant to an affine map of pred."""
    p = torch.where(keep, pred.double(), 0.0).flatten(1)
    g = torch.where(keep, target.double(), 0.0).flatten(1)
    omega = refit_weights(p, g, w.flatten(1), keep.flatten(1), s0, t0, trunc, floor)
    tot = omega.sum(dim=1)
    safe = tot > 0
    tot_s = torch.where(safe, tot, 1.0)
    mp = (omega * p).sum(dim=1) / tot_s
    mg = (omega * g).sum(dim=1) / tot_s
    dp = p - mp[:, None]
    var = (omega * dp.square()).sum(dim=1)
    cov = (omega * dp * (g - mg[:, None])).sum(dim=1)
    second = (omega * p.square()).sum(dim=1)
    var_d, cov_d = var.detach(), cov.detach()
    ok = safe & (var_d > 1e-12 * second)
    s_d = cov_d / torch.where(ok, var_d, 1.0)
    ok = ok & torch.isfinite(s_d) & (s_d > 0)
    # the decision is taken on detached values; a dropped image enters the graph
    # only as 0 / 1, so every local derivative of it is finite (exactly 0 upstream)
    s = torch.where(ok, cov, 0.0) / torch.where(ok, var, 1.0)
    t = torch.where(ok, mg, 0.0) - s * torch.where(ok, mp, 0.0)
    return s, t, ok


@torch.no_grad()
def local_segments(
    z_ref: Tensor,
    h: Tensor,
    alpha: int,
    cell_px: float,
    phi: Tensor,
    offset: Tensor,
    beta: Tensor,
) -> Tensor:
    """Segment label [B, H, W] (int64, distinct across images) of the local term at
    level alpha: a square grid of cell_px / alpha pixels rotated by phi [B] about
    the image centre and shifted by offset [B, 2] (pixels), times bins of 4 / alpha
    in log2(z_ref / h) with phase beta [B] (a factor 2^(4 / alpha) per bin)."""
    B, H, W = z_ref.shape
    dev = z_ref.device
    cell = cell_px / alpha
    yy = (torch.arange(H, dtype=torch.float64, device=dev) - (H - 1) / 2)[None, :, None]
    xx = (torch.arange(W, dtype=torch.float64, device=dev) - (W - 1) / 2)[None, None]
    cos = phi.double().cos()[:, None, None]
    sin = phi.double().sin()[:, None, None]
    offset = offset.double()
    u1 = (xx * cos + yy * sin + offset[:, 0, None, None]) / cell
    u2 = (-xx * sin + yy * cos + offset[:, 1, None, None]) / cell
    span = int(math.ceil(math.hypot(H, W) / cell)) + 2
    q = 2 * span + 1
    iu = u1.floor().long() + span
    iv = u2.floor().long() + span
    rel = (z_ref.double() / h.double()[:, None, None]).clamp(2.0**-40, 2.0**40)
    phase = beta.double()[:, None, None]
    ib = (alpha * torch.log2(rel) / 4.0 + phase).floor().long() + 10 * alpha + 1
    nb = 20 * alpha + 3
    img = torch.arange(B, device=dev)[:, None, None]
    return ((img * q + iu) * q + iv) * nb + ib


def segment_shift(
    x: Tensor, w: Tensor, seg: Tensor, rank: Optional[Tensor] = None
) -> Tensor:
    """Per-element shift [n]: the weighted median of x over the element's segment
    (the first element, in increasing x, at which the segment's cumulative weight
    reaches half of its weight; a minimiser of sum_i w_i |x_i - c| over the
    segment), taken as x at that element, so it stays in the graph of x through one
    element per segment. x, w, seg: 1-D over the elements that take part; rank:
    distinct integers in [0, 2^31) in the order of x (e.g. the ranks of a superset
    of x: one argsort serves every level), computed here when None. One integer
    sort on segment * 2^31 + rank (unique keys, so no stable sort is needed), a
    fp64 cumulative sum of w."""
    n = x.numel()
    with torch.no_grad():
        if rank is None:
            rank = torch.empty(n, dtype=torch.long, device=x.device)
            rank[torch.argsort(x.detach())] = torch.arange(n, device=x.device)
        span = 1 << 31
        _, inverse = torch.unique(seg, return_inverse=True)  # dense segment ids
        order = torch.argsort(inverse * span + rank)  # by (segment, value)
        inv_s = inverse[order]
        cw = w[order].double().cumsum(0)
        counts = torch.bincount(inv_s)
        ends = counts.cumsum(0) - 1
        starts = ends - counts + 1
        before = torch.where(starts > 0, cw[(starts - 1).clamp_min(0)], 0.0)
        total = cw[ends] - before
        m = torch.searchsorted(cw, before + 0.5 * total, right=False)
        m = torch.minimum(torch.maximum(m, starts), ends)
        median_elem = order[m]  # [segments] index into x
    return x[median_elem][inverse]


def weighted_gradient_pyramid(
    residual: Tensor, target: Tensor, mask: Tensor, cap: Tensor, num_scales: int = 4
) -> Tuple[Tensor, Tensor]:
    """([B, S] sums, [B, S] counts): the gradient-matching error of the residual R
    (aligned prediction - target) over a Gaussian pyramid of S levels: per image and
    level k the sum over the pixels of G_k = erode_3x3(M_k) of
    W_k (|Scharr_x / 32 * R_k| + |Scharr_y / 32 * R_k|), and the count of those
    pixels; M_0 is the mask, M_k+1 = AND_2x2(erode_5x5(M_k)) and R is zeroed off the
    mask, so no invalid pixel (nor the zero padding) ever enters a kernel. W_k =
    min(1 / G_k, cap) (cap [B]); G_k is the target zeroed off the mask and taken
    down the pyramid as R (at a pixel of the eroded mask a mean of valid depths, so
    G_k > 0 there)."""
    m = mask.unsqueeze(1).to(residual.dtype)
    R = torch.where(m > 0, residual.unsqueeze(1), 0.0)
    G = torch.where(m > 0, target.unsqueeze(1).to(residual.dtype), 0.0)
    cap = cap.to(residual.dtype)[:, None, None, None]
    kernels = _scharr_kernels(residual.device, residual.dtype)
    sums, counts = [], []
    for _ in range(num_scales):
        e = _erode(m, 3)
        wk = torch.minimum(1.0 / torch.where(e > 0, G, 1.0).clamp_min(1e-30), cap)
        diff = sum(_conv_same(R, k).abs() for k in kernels)
        sums.append((diff * wk * e).flatten(1).sum(dim=1))
        counts.append(e.flatten(1).sum(dim=1))
        R, G = _down2(R), _down2(G)
        m = _down2_mask_erode(m)
    return torch.stack(sums, dim=1), torch.stack(counts, dim=1)


@LOSSES.register_module("relative_depth_loss")
class RelativeDepthLoss(nn.Module):
    """The L-DPT depth loss, the loss of the released depth models
    (configs/hybrid/*.py); docs/depth_loss.md is its specification. The prediction,
    defined up to a scale and a shift per image, is aligned to the ground truth and
    compared with it, in fp32 with autocast off whatever the caller's precision (fp32
    inputs are used as they are); images with fewer than ``min_valid`` valid pixels
    or a rejected fit are dropped (drop_frac).

    Per image: an exact ROE fit (truncated 1/z-weighted L1, ``roe_fit_sample``) on
    roe_points sampled valid pixels, a trim per label type, one weighted
    least-squares refit inside autograd (fp64), the global term (mean 1/z-weighted
    error of the kept pixels, weights min(1 / g, weight_cap / h), h the harmonic-mean
    depth), local shift terms per label type over rotated image cells x log-depth
    bins, and a 1/z-weighted gradient term on grad_types. Every image weighs the same;
        loss = depth_loss + aux_scale * (local_weight * local_loss
                                         + mage_weight * mage_loss).
    Options: head ("sigmoid" | "exp": the model's head; it only selects the logged
    logit statistics), roe_points (>= 16), roe_trunc (> 0), weight_cap (> 1),
    refit_floor (> 0), label_types (None: every image has one label type and no
    sample names are needed, so the options below that name label types stay at
    their defaults; or dict type -> sample-name prefixes: the model passes the
    batch's names and an unknown prefix raises), type_trim (dict over every type,
    values in [0, 0.5)), local_levels (dict type -> positive integers alpha),
    local_weight, local_cell (pixels of a cell at alpha = 1), local_pred_ratio
    (probability that the prediction is the bin reference), grad_types,
    mage_weight / mage_num_scales (weight and levels of the gradient term),
    aux_scale (set by CriterionWeightWarmup; at 0 the local and gradient terms run
    under no_grad and are logged only). ``align`` has the one value "roe" and
    ``male_weight`` / ``male_num_scales`` are accepted because the release configs
    state them: the loss has no Laplacian term, so male_weight must be 0. Without
    options the loss is the global term alone. The returned keys are ROE_KEYS.

    log_logit_stats=True also logs ``logit_mean`` / ``logit_std`` / ``logit_absmax``
    of the head's logit over the valid pixels (logit(p), or log(p) with head="exp").
    Every key but ``loss`` is detached (the terms and the diagnostics).

    Shapes:
        pred, target, mask: [B, 1, H, W] or [B, H, W] -> dict of scalar losses
    """

    ALIGNS = ("roe",)
    HEADS = ("sigmoid", "exp")
    # logit_* clamp p to [LOGIT_EPS, 1 - LOGIT_EPS]
    LOGIT_EPS = 1e-6
    LOGIT_STATS = ("logit_mean", "logit_std", "logit_absmax")
    # |log d| above this counts in saturation_frac
    LOG_SATURATION = 20.0
    ROE_KEYS = (
        "loss",
        "depth_loss",
        "local_loss",
        "mage_loss",
        "drop_frac",
        "depth_loss_max",
        "scale_median",
        "spread_median",
        "saturation_frac",
        "gm_image_frac",
        "trim_thresh_median",
        "trunc_frac",
        "refit_ratio_median",
        "lsq_loss",
    )

    def __init__(
        self,
        eps: float = 1e-6,
        min_valid: int = 10,
        log_logit_stats: bool = False,
        mage_weight: float = 0.0,
        mage_num_scales: int = 4,
        male_weight: float = 0.0,
        male_num_scales: int = 4,
        align: str = "roe",
        head: str = "sigmoid",
        roe_points: int = 2304,
        roe_trunc: float = 1.0,
        weight_cap: float = 10.0,
        refit_floor: float = 0.01,
        label_types: Optional[Dict[str, Sequence[str]]] = None,
        type_trim: Optional[Dict[str, float]] = None,
        local_levels: Optional[Dict[str, Sequence[int]]] = None,
        local_weight: float = 1.0,
        local_cell: float = 1024.0,
        local_pred_ratio: float = 0.5,
        grad_types: Sequence[str] = (),
        aux_scale: float = 1.0,
    ):
        super().__init__()
        if align not in self.ALIGNS:
            raise ValueError(f"align must be one of {list(self.ALIGNS)}, got {align!r}")
        if head not in self.HEADS:
            raise ValueError(f"head must be one of {list(self.HEADS)}, got {head!r}")
        if male_weight != 0.0:
            raise ValueError(
                f"male_weight must be 0 (the L-DPT depth loss has no Laplacian term), "
                f"got {male_weight!r}"
            )
        self.eps = eps
        self.min_valid = min_valid
        self.log_logit_stats = bool(log_logit_stats)
        self.mage_weight = float(mage_weight)
        self.mage_num_scales = int(mage_num_scales)
        self.align = align
        self.head = head
        if isinstance(roe_points, bool) or int(roe_points) != roe_points:
            raise ValueError(f"roe_points must be an integer, got {roe_points!r}")
        self.roe_points = int(roe_points)
        if self.roe_points < 16:
            raise ValueError(f"roe_points must be >= 16, got {roe_points}")
        self.roe_trunc = float(roe_trunc)
        if not self.roe_trunc > 0.0:
            raise ValueError(f"roe_trunc must be > 0, got {roe_trunc}")
        self.weight_cap = float(weight_cap)
        if not self.weight_cap > 1.0:
            raise ValueError(f"weight_cap must be > 1, got {weight_cap}")
        self.refit_floor = float(refit_floor)
        if not self.refit_floor > 0.0:
            raise ValueError(f"refit_floor must be > 0, got {refit_floor}")
        self.local_weight = float(local_weight)
        if not self.local_weight >= 0.0:
            raise ValueError(f"local_weight must be >= 0, got {local_weight}")
        self.local_cell = float(local_cell)
        if not self.local_cell > 0.0:
            raise ValueError(f"local_cell must be > 0, got {local_cell}")
        self.local_pred_ratio = float(local_pred_ratio)
        if not 0.0 <= self.local_pred_ratio <= 1.0:
            raise ValueError(
                f"local_pred_ratio must be in [0, 1], got {local_pred_ratio}"
            )
        self.aux_scale = float(aux_scale)
        if not self.aux_scale >= 0.0:
            raise ValueError(f"aux_scale must be >= 0, got {aux_scale}")

        # routing: label type -> sample-name prefixes
        self.label_types, self._prefix_type = None, {}
        if label_types is not None:
            if not isinstance(label_types, Mapping):
                raise ValueError(f"label_types must be a dict, got {label_types!r}")
            self.label_types = {}
            for kind, prefixes in label_types.items():
                if isinstance(prefixes, str) or not isinstance(prefixes, Sequence):
                    raise ValueError(
                        f"label_types[{kind!r}] must be a list of sample-name "
                        f"prefixes, got {prefixes!r}"
                    )
                self.label_types[str(kind)] = [str(x) for x in prefixes]
                for prefix in self.label_types[str(kind)]:
                    if not prefix or "/" in prefix:
                        raise ValueError(
                            "label_types entries are sample-name prefixes (the part "
                            f"before '/'), got {prefix!r}"
                        )
                    if prefix in self._prefix_type:
                        raise ValueError(
                            f"prefix {prefix!r} is in label_types "
                            f"{self._prefix_type[prefix]!r} and {kind!r}"
                        )
                    self._prefix_type[prefix] = str(kind)
        kinds = set(self.label_types or {})

        self.type_trim = None
        if type_trim is not None:
            if not isinstance(type_trim, Mapping) or set(map(str, type_trim)) != kinds:
                raise ValueError(
                    f"type_trim must list every label type {sorted(kinds)}, "
                    f"got {type_trim!r}"
                )
            self.type_trim = {str(k): float(v) for k, v in type_trim.items()}
            for kind, value in self.type_trim.items():
                if not 0.0 <= value < 0.5:
                    raise ValueError(
                        f"type_trim[{kind!r}] must be in [0, 0.5), got {value}"
                    )

        self.local_levels = None
        if local_levels is not None:
            if (
                not isinstance(local_levels, Mapping)
                or not set(map(str, local_levels)) <= kinds
            ):
                raise ValueError(
                    f"local_levels must be a dict over the label types {sorted(kinds)}, "
                    f"got {local_levels!r}"
                )
            self.local_levels = {}
            for kind, levels in local_levels.items():
                if isinstance(levels, str) or not isinstance(levels, Sequence):
                    raise ValueError(
                        f"local_levels[{kind!r}] must be a list, got {levels!r}"
                    )
                for a in levels:
                    if isinstance(a, bool) or int(a) != a or int(a) < 1:
                        raise ValueError(
                            f"local_levels[{kind!r}] must hold positive integers, "
                            f"got {levels!r}"
                        )
                self.local_levels[str(kind)] = [int(a) for a in levels]

        if isinstance(grad_types, str):
            raise ValueError(f"grad_types must be a list of types, got {grad_types!r}")
        self.grad_types = tuple(str(x) for x in grad_types)
        if not set(self.grad_types) <= kinds:
            raise ValueError(
                f"grad_types {list(self.grad_types)} must be label types "
                f"{sorted(kinds)}"
            )

    @property
    def uses_names(self) -> bool:
        """The loss needs the batch's sample names (label_types); the model passes
        them as names=data_dict["name"]."""
        return self.label_types is not None

    def label_types_of(self, names, num_images) -> List[Optional[str]]:
        """Label type of every image of the batch, from the sample-name prefix (the
        part before '/', exact and case-sensitive); raises on missing names or a
        prefix that is in no label_types list. Without label_types every image has
        the one type None (names are not needed)."""
        if self.label_types is None:
            return [None] * num_images
        if names is None or len(names) != num_images:
            raise ValueError(
                f"label_types needs the sample names of the batch "
                f"(names=data_dict['name']), got {names!r}"
            )
        types = []
        for name in names:
            prefix = str(name).split("/")[0]
            if prefix not in self._prefix_type:
                raise ValueError(
                    f"sample {name!r}: prefix {prefix!r} is in no label_types list "
                    f"{self.label_types}"
                )
            types.append(self._prefix_type[prefix])
        return types

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor,
        names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Tensor]:
        device_type = pred.device.type
        if is_autocast_available(device_type) and torch.is_autocast_enabled(
            device_type
        ):
            with torch.autocast(device_type, enabled=False):
                return self._forward(pred.float(), target, mask, names)
        return self._forward(pred.float(), target, mask, names)

    def _forward(
        self, pred: Tensor, target: Tensor, mask: Tensor, names=None
    ) -> Dict[str, Tensor]:
        if pred.ndim == 4:
            pred = pred.squeeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        target = target.to(device=pred.device, dtype=pred.dtype)
        mask = mask.to(device=pred.device, dtype=torch.bool)
        return self.roe_loss(pred, target, mask, names)

    @torch.no_grad()
    def logit_stats(self, pred=None, mask=None) -> Dict[str, Tensor]:
        """Detached statistics of the head's logit over the valid pixels (fp32;
        zeros when no image was fitted): z = logit(p), p clamped to [LOGIT_EPS,
        1 - LOGIT_EPS], or z = log(p) with head="exp"; medians over the images of
        the per-image mean and std, max |z| of the batch."""
        if pred is None:
            zero = torch.zeros((), dtype=torch.float32)
            return {k: zero.clone() for k in self.LOGIT_STATS}
        if self.head == "exp":
            p = pred.detach().float()
            z = p.clamp_min(torch.finfo(p.dtype).tiny).log().flatten(1)
        else:
            z = torch.logit(pred.detach().float(), eps=self.LOGIT_EPS).flatten(1)
        m = mask.flatten(1)
        num = m.sum(dim=1).clamp_min(1)
        mean = torch.where(m, z, 0.0).sum(dim=1) / num
        var = torch.where(m, (z - mean[:, None]).square(), 0.0).sum(dim=1) / num
        return dict(
            logit_mean=mean.median(),
            logit_std=var.sqrt().median(),
            logit_absmax=torch.where(m, z.abs(), 0.0).amax(),
        )

    def roe_loss(self, pred, target, mask, names=None) -> Dict[str, Tensor]:
        """The L-DPT depth loss (docs/depth_loss.md section 2) on [B, H, W] fp32 pred,
        target (metres) and bool mask: per image a ROE fit on roe_points sampled valid
        pixels, the trim of its label type, a weighted least-squares refit inside
        autograd (fp64), the 1/z-weighted global L1, the local shift terms of its type
        and (grad_types) the weighted gradient term; images with fewer than min_valid
        valid pixels or a rejected fit are dropped (drop_frac). loss = depth_loss +
        aux_scale * (local_weight * local_loss + mage_weight * mage_loss); with
        aux_scale = 0 the local and gradient terms are computed under no_grad
        (logged only) and loss = depth_loss."""
        B = pred.shape[0]
        dev = pred.device
        types = self.label_types_of(names, B)
        trims = [
            float(self.type_trim.get(kind, 0.0)) if self.type_trim else 0.0
            for kind in types
        ]
        num = mask.flatten(1).sum(dim=1)
        w, h = pixel_weights(target, mask, self.weight_cap)
        p64, g64 = pred.double(), target.double()

        # ROE fit (no grad) on sampled valid pixels, one image at a time; trim
        idx = sample_valid(mask, self.roe_points, jitter=True)
        s0 = torch.full((B,), torch.nan, dtype=torch.float64, device=dev)
        t0 = torch.full_like(s0, torch.nan)
        ok0 = [False] * B
        with torch.no_grad():
            for b, n_b in enumerate(num.tolist()):
                if n_b < self.min_valid:
                    continue
                i = idx[b]
                s_b, t_b, ok_b = roe_fit_sample(
                    p64[b].flatten()[i],
                    g64[b].flatten()[i],
                    w[b].flatten()[i],
                    h[b],
                    self.roe_trunc,
                )
                if ok_b:
                    s0[b], t0[b], ok0[b] = s_b, t_b, True
            fit0 = (s0[:, None, None] * p64 + t0[:, None, None] - g64).abs()
            e = torch.where(mask, w * fit0, 0.0)  # relative error under the ROE fit
            keep = mask.clone()
            thresh = torch.zeros(B, dtype=torch.float64, device=dev)
            for b in range(B):
                if ok0[b] and trims[b] > 0.0:
                    v = e[b][mask[b]]
                    kept, thresh[b] = smallest_k(v, num_kept(v.numel(), trims[b]))
                    keep[b][mask[b]] = kept

        # refit inside autograd (fp64)
        s, t, ok = refit_in_graph(
            pred, target, w, keep, s0, t0, self.roe_trunc, self.refit_floor
        )
        ok = ok & torch.tensor(ok0, dtype=torch.bool, device=dev)
        n_fit = int(ok.sum().item())
        if n_fit == 0:
            return self.roe_no_loss(pred)
        s_use = torch.where(ok, s, 1.0)  # dropped images: finite values, weight 0
        t_use = torch.where(ok, t, 0.0)
        aligned = s_use[:, None, None] * p64 + t_use[:, None, None]
        nkeep = keep.flatten(1).sum(dim=1).clamp_min(1)

        # global term: mean relative error over the kept pixels
        res = torch.where(keep, (aligned - g64).abs() * w, 0.0)
        glob = torch.where(ok, res.flatten(1).sum(dim=1) / nkeep, 0.0)
        depth_loss = glob.sum() / n_fit

        aux = float(self.aux_scale)
        with torch.no_grad() if aux == 0.0 else contextlib.nullcontext():
            local_loss = self.local_term(
                p64, g64, w, h, keep, nkeep, ok, s_use, aligned, types, n_fit
            )
            mage_loss, takes_g = self.weighted_gradient_term(
                aligned, target, mask, h, ok, types, n_fit
            )
        loss = depth_loss
        if aux != 0.0:
            extra = None
            if local_loss is not None:
                extra = self.local_weight * local_loss
            if mage_loss is not None:
                term = self.mage_weight * mage_loss
                extra = term if extra is None else extra + term
            if extra is not None:
                loss = loss + aux * extra
        loss = loss.float()

        with torch.no_grad():
            okl = ok.tolist()
            m = mask.flatten(1)
            num_d = num.clamp_min(1).double()
            pix_f = num[ok].sum().clamp_min(1).double()
            sd = s.detach()
            gain = torch.where(
                mask, sd[:, None, None] * p64.detach() / g64.clamp_min(1e-30), 0.0
            )
            gain = gain.flatten(1).sum(dim=1) / num_d
            logd = torch.where(m, p64.detach().flatten(1).clamp_min(1e-30).log(), 0.0)
            mean = logd.sum(dim=1) / num_d
            var = torch.where(m, (logd - mean[:, None]).square(), 0.0).sum(dim=1)
            spread = (var / num_d).sqrt()
            saturated = m & (logd.abs() > self.LOG_SATURATION)
            trimmed = [okl[b] and trims[b] > 0.0 for b in range(B)]
            trimmed = torch.tensor(trimmed, dtype=torch.bool, device=dev)
            zero = torch.zeros((), dtype=torch.float64, device=dev)
            truncated = (e > self.roe_trunc) & mask
            out = dict(
                loss=loss,
                depth_loss=depth_loss.detach(),
                local_loss=zero if local_loss is None else local_loss.detach(),
                mage_loss=zero if mage_loss is None else mage_loss.detach(),
                drop_frac=torch.tensor(1.0 - n_fit / B, device=dev),
                depth_loss_max=glob.detach()[ok].max(),
                scale_median=gain[ok].median(),
                spread_median=spread[ok].median(),
                saturation_frac=saturated[ok].sum() / pix_f,
                gm_image_frac=takes_g.sum() / float(B),
                trim_thresh_median=(
                    thresh[trimmed].median() if bool(trimmed.any()) else zero
                ),
                trunc_frac=truncated[ok].sum() / pix_f,
                refit_ratio_median=(sd / s0)[ok].median(),
                lsq_loss=self.lsq_term(p64.detach(), g64, w, keep, ok, n_fit),
            )
            if self.log_logit_stats:
                out.update(self.logit_stats(pred[ok], mask[ok]))
        return {
            k: v if k == "loss" else v.detach().float().reshape(())
            for k, v in out.items()
        }

    def roe_no_loss(self, pred) -> Dict[str, Tensor]:
        """No image fitted: zero loss, still connected to pred (DDP sees every
        param); every key of roe_loss, zeros except drop_frac = 1."""
        zero = torch.zeros((), dtype=torch.float32, device=pred.device)
        out = {k: zero.clone() for k in self.ROE_KEYS}
        out["loss"] = pred.mean() * 0.0
        out["drop_frac"] = torch.ones((), dtype=torch.float32, device=pred.device)
        if self.log_logit_stats:
            out.update({k: v.to(pred.device) for k, v in self.logit_stats().items()})
        return out

    def local_term(
        self, p64, g64, w, h, keep, nkeep, ok, s_use, aligned, types, n_fit
    ) -> Optional[Tensor]:
        """Sum over the levels of the local term (docs/depth_loss.md section 2 step 7
        and 4.4), mean over the fitted images, fp64; None when no fitted image has a
        level. Per level and image one draw of the rotation, offset, bin phase and
        reference depth (the prediction with probability local_pred_ratio)."""
        if not self.local_levels:
            return None
        B, dev = len(types), p64.device
        okl = ok.tolist()
        levels = sorted({a for k in set(types) for a in self.local_levels.get(k, ())})
        taking = {
            a: [okl[b] and a in self.local_levels.get(types[b], ()) for b in range(B)]
            for a in levels
        }
        levels = [a for a in levels if any(taking[a])]
        if not levels:
            return None
        sel = keep & ok[:, None, None]
        x_all = (g64 - s_use[:, None, None] * p64)[sel]  # g - s d, in the graph
        w_all = w[sel]
        b_all = torch.arange(B, device=dev)[:, None, None].expand_as(sel)[sel]
        with torch.no_grad():
            # one value rank for every level
            rank = torch.empty(x_all.numel(), dtype=torch.long, device=dev)
            rank[torch.argsort(x_all.detach())] = torch.arange(
                x_all.numel(), device=dev
            )
            z_pred = torch.maximum(aligned.detach(), 1e-3 * h[:, None, None])
        loc_img = torch.zeros(B, dtype=torch.float64, device=dev)
        for alpha in levels:
            with torch.no_grad():
                f64 = dict(dtype=torch.float64, device=dev)
                phi = torch.rand(B, **f64) * (math.pi / 2)
                offset = torch.rand(B, 2, **f64) * (self.local_cell / alpha)
                beta = torch.rand(B, **f64)
                use_pred = torch.rand(B, **f64) < self.local_pred_ratio
                z_ref = torch.where(use_pred[:, None, None], z_pred, g64)
                seg = local_segments(
                    z_ref, h, alpha, self.local_cell, phi, offset, beta
                )[sel]
                takes = torch.tensor(taking[alpha], dtype=torch.bool, device=dev)
                el = takes[b_all]
            x, wl = x_all[el], w_all[el]
            shift = segment_shift(x, wl, seg[el], rank[el])
            loc_img = loc_img.index_add(0, b_all[el], (x - shift).abs() * wl)
        loc_img = torch.where(ok, loc_img / nkeep, 0.0)
        return loc_img.sum() / n_fit

    def weighted_gradient_term(
        self, aligned, target, mask, h, ok, types, n_fit
    ) -> Tuple[Optional[Tensor], Tensor]:
        """(mage_loss or None, takes_part [B]): the 1/z-weighted gradient term
        (section 2 step 8) of the fitted images whose type is in grad_types and whose
        level-0 eroded mask has >= min_valid pixels; mean over its levels per image,
        summed and divided by the number of fitted images. fp32 pyramid on the
        fp64 residual."""
        B, dev = len(types), aligned.device
        takes = torch.zeros(B, dtype=torch.bool, device=dev)
        if not self.grad_types:
            return None, takes
        okl = ok.tolist()
        cand = [b for b in range(B) if okl[b] and types[b] in self.grad_types]
        if not cand:
            return None, takes
        ib = torch.tensor(cand, dtype=torch.long, device=dev)
        tgt = target[ib].float()
        residual = (aligned[ib] - tgt.double()).float()
        cap = (self.weight_cap / h[ib]).float()
        sums, counts = weighted_gradient_pyramid(
            residual, tgt, mask[ib], cap, self.mage_num_scales
        )
        part = counts[:, 0] >= self.min_valid
        per_image = (sums / counts.clamp_min(self.eps)).mean(dim=1)
        value = torch.where(part, per_image, 0.0).double().sum() / n_fit
        takes[ib] = part
        return value, takes

    @torch.no_grad()
    def lsq_term(self, p64, g64, w, keep, ok, n_fit) -> Tensor:
        """The global term under the evaluator's fit (lsq_loss): the evaluator's
        own `nonneg_lsq_scale_and_shift` (unweighted fp64 least squares of g on d
        over the kept pixels; its degenerate-fit rule; s <= 0 or degenerate: s = 0,
        t = mean g), then the mean 1/z-weighted error over the kept pixels, mean
        over the fitted images."""
        k3 = keep & ok[:, None, None]
        s, t, valid, _ = nonneg_lsq_scale_and_shift(p64, g64, k3, min_valid=1)
        s = torch.where(valid, s, 0.0)
        t = torch.where(valid, t, 0.0)
        k = k3.flatten(1)
        n = k.sum(dim=1).clamp_min(1).double()
        err = torch.where(
            k,
            (s[:, None] * p64.flatten(1) + t[:, None] - g64.flatten(1)).abs()
            * w.flatten(1),
            0.0,
        )
        per_image = torch.where(ok, err.sum(dim=1) / n, 0.0)
        return per_image.sum() / n_fit
