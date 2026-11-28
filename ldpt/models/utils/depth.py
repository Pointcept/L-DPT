"""
Depth alignment utilities

Per-image affine alignment (least squares, trimmed, RANSAC, ROE) of a relative depth
prediction to the GT, used by relative_depth_loss (ROE), the depth evaluator and the
visualizer.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

NONPOSITIVE = ("fallback", "drop")


def _clamp(x: Tensor, bounds: Optional[Tuple[float, float]]) -> Tensor:
    return x if bounds is None else x.clamp(bounds[0], bounds[1])


def _check_choice(name: str, value, choices) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")


def _median_mad_fallback(
    pred, target, mask, indices, scale, shift, eps, min_valid, clamp_scale
) -> None:
    """In place: scale / shift of the images ``indices`` by median / MAD matching."""
    # (small loop is OK since only fallback subset)
    for b in indices.tolist():
        mb = mask[b].view(-1)
        if mb.sum().item() < min_valid:
            # shouldn't happen because b is in valid, but keep safe
            s_f, t_f = 1.0, 0.0
        else:
            pb = pred[b].view(-1)[mb]
            gb = target[b].view(-1)[mb]
            med_p = pb.median()
            med_g = gb.median()
            mad_p = (pb - med_p).abs().median()
            mad_g = (gb - med_g).abs().median()

            if mad_p <= eps:
                # keep gradient alive: do NOT set scale=0
                s_f = 1.0
            else:
                s_f = _clamp(mad_g / (mad_p + eps), clamp_scale)
            t_f = med_g - s_f * med_p

        scale[b] = torch.as_tensor(s_f, device=pred.device, dtype=pred.dtype)
        shift[b] = torch.as_tensor(t_f, device=pred.device, dtype=pred.dtype)


@torch.no_grad()
def compute_scale_and_shift(
    pred,
    target,
    mask=None,
    eps: float = 1e-6,
    min_valid: int = 10,
    clamp_scale=(1e-3, 1e3),
    ridge: float = 1e-6,
    nonpositive: str = "fallback",
    return_unfit: bool = False,
):
    """Least-squares (scale, shift, valid) per image, from centred fp64 moments.

    scale = cov(p, g) / (var(p) + ridge), shift = mean(g) - scale * mean(p). A fit is
    degenerate when var(p) <= eps (sum over the valid pixels); a depth-like quantity
    needs scale > 0. The solver of the evaluator's align_solver="centered" and of the
    depth visualizer (the L-DPT depth loss uses roe_fit_sample instead).

    clamp_scale: (lo, hi) clamp of the scale (None: no clamp); the shift is that of
        the *unclamped* scale, so a clamped image is moved off its target by
        (s - s_clamped) * mean(p).
    nonpositive: what a degenerate fit or a scale <= 0 gets: "fallback" (default:
        median / MAD matching) or "drop" (valid=False, scale / shift NaN).
    return_unfit: also return unfit [B] bool: images with >= min_valid pixels whose
        least-squares fit was rejected (fell back or dropped).
    """
    _check_choice("nonpositive", nonpositive, NONPOSITIVE)
    assert pred.shape == target.shape
    if mask is None:
        mask = ~torch.isnan(target)
    mask = mask.to(device=pred.device, dtype=torch.bool)
    target = torch.nan_to_num(target, nan=0.0)

    B = pred.shape[0]
    p = pred.view(B, -1)
    g = target.view(B, -1)
    m = mask.view(B, -1).to(dtype=torch.float32)

    # ---- double-precision accumulation for stability ----
    p64, g64, m64 = p.double(), g.double(), m.double()
    N = m64.sum(dim=1)  # [B] double
    valid = N >= float(min_valid)

    scale = torch.full((B,), torch.nan, device=pred.device, dtype=pred.dtype)
    shift = torch.full((B,), torch.nan, device=pred.device, dtype=pred.dtype)
    if not valid.any():
        unfit = torch.zeros_like(valid)
        return (scale, shift, valid, unfit) if return_unfit else (scale, shift, valid)

    Nv = N[valid].clamp_min(1.0)  # [Bv]
    pv = p64[valid]
    gv = g64[valid]
    mv = m64[valid]

    # masked means
    mean_p = (pv * mv).sum(dim=1) / Nv
    mean_g = (gv * mv).sum(dim=1) / Nv

    # centered var/cov (much less cancellation than N*sum_p2 - sum_p^2)
    dp = pv - mean_p.unsqueeze(1)
    dg = gv - mean_g.unsqueeze(1)
    var_p = (dp * dp * mv).sum(dim=1)  # >= 0
    cov_pg = (dp * dg * mv).sum(dim=1)

    # ridge makes it always solvable in principle
    denom = var_p + ridge
    s = cov_pg / denom
    t = mean_g - s * mean_p

    # a depth-like quantity needs a positive scale: s <= 0 counts as unsolvable
    solvable = (var_p > eps) & (s > 0)
    vidx = torch.where(valid)[0]
    unfit = torch.zeros_like(valid)

    # write primary solutions (cast back)
    good = vidx[solvable]
    s_good = _clamp(s[solvable].to(pred.dtype), clamp_scale)
    scale[good] = s_good
    shift[good] = t[solvable].to(pred.dtype)

    # fallback for the truly bad ones (robust median / MAD), or drop them
    bad = vidx[~solvable]
    unfit[bad] = True
    if nonpositive == "fallback":
        _median_mad_fallback(
            pred, target, mask, bad, scale, shift, eps, min_valid, clamp_scale
        )
    else:  # "drop": no alignment (scale / shift stay NaN)
        valid = valid & ~unfit

    return (scale, shift, valid, unfit) if return_unfit else (scale, shift, valid)


# ---------------------------------------------------------------- evaluation protocols


@torch.no_grad()
def lsq_scale_and_shift(pred, target, mask=None, min_valid: int = 10):
    """Exact least squares in fp64 with no clamp and no fallback: (scale, shift,
    valid) per image; valid is False for < min_valid pixels, a degenerate fit
    (var(p) <= 1e-6, sum of squared deviations) or a scale <= 0 (scale / shift NaN).
    The evaluator's align_solver="lsq"."""
    return compute_scale_and_shift(
        pred,
        target,
        mask,
        min_valid=min_valid,
        clamp_scale=None,
        ridge=0.0,
        nonpositive="drop",
    )


@torch.no_grad()
def nonneg_lsq_scale_and_shift(pred, target, mask, min_valid: int = 10):
    """Exact least squares with a non-negative scale: lsq_scale_and_shift where it
    fits, and where it rejects the fit (a degenerate prediction or a scale <= 0) the
    constrained optimum s = 0, t = mean of the target over the valid pixels (the GT
    mean as a constant prediction), so an anti-correlated or flat prediction scores
    as a constant instead of leaving the evaluation. (scale, shift, valid,
    rejected) [B]; valid is False only for < min_valid pixels. The evaluator's
    align_solver="lsq" and the LSQ fits of its trim5 / disp protocols."""
    scale, shift, fitted = lsq_scale_and_shift(pred, target, mask, min_valid)
    m = mask.flatten(1)
    num = m.sum(dim=1)
    valid = num >= min_valid
    rejected = valid & ~fitted
    if rejected.any():
        g = torch.where(m, target.flatten(1).double(), 0.0)
        mean = (g.sum(dim=1) / num.clamp_min(1)).to(pred.dtype)
        scale = torch.where(rejected, torch.zeros_like(scale), scale)
        shift = torch.where(rejected, mean, shift)
    return scale, shift, valid, rejected


@torch.no_grad()
def free_lsq_scale_and_shift(pred, target, mask, min_valid: int = 10):
    """Exact least squares with no constraint on the scale, Marigold's evaluation
    fit (src/util/alignment.py align_depth_least_square: np.linalg.lstsq of the
    target on [pred, 1] over the valid pixels). (scale, shift, valid, signed) [B]:
    nonneg_lsq_scale_and_shift where it fits (bit for bit), and where it rejects the
    fit the unconstrained optimum in fp64 (centered moments): a scale <= 0 is kept,
    and a flat prediction (var(p) <= 1e-6) gets s = 0, t = mean GT, the aligned
    prediction of lstsq's minimum-norm solution for a constant prediction. valid is
    False only for < min_valid pixels; signed marks the images where this differs
    from the non-negative fit. The evaluator's align_solver="lsq_free"."""
    scale, shift, valid, signed = nonneg_lsq_scale_and_shift(
        pred, target, mask, min_valid
    )
    if signed.any():
        idx = torch.where(signed)[0]
        m = mask[idx].flatten(1)
        w = m.double()
        p = pred[idx].flatten(1).double()
        g = torch.where(m, target[idx].flatten(1).double(), 0.0)
        n = w.sum(dim=1)
        mean_p = (p * w).sum(dim=1) / n
        mean_g = (g * w).sum(dim=1) / n
        dp = p - mean_p.unsqueeze(1)
        dg = g - mean_g.unsqueeze(1)
        var_p = (dp * dp * w).sum(dim=1)
        cov = (dp * dg * w).sum(dim=1)
        s = torch.where(var_p > 1e-6, cov / var_p, torch.zeros_like(cov))
        t = mean_g - s * mean_p
        scale = scale.clone()
        shift = shift.clone()
        scale[idx] = s.to(scale.dtype)
        shift[idx] = t.to(shift.dtype)
    return scale, shift, valid, signed


def _median(x: Tensor) -> Tensor:
    """np.median of a 1-D tensor: the mean of the two middle values for an even count
    (torch.median takes the lower one)."""
    n = x.numel()
    lo = x.kthvalue((n + 1) // 2).values
    return lo if n % 2 else (lo + x.kthvalue(n // 2 + 1).values) / 2


@torch.no_grad()
def trimmed_scale_and_shift(
    pred, target, mask, trim: float = 0.05, min_valid=10, return_rejected=False
):
    """Two-pass trimmed least squares in depth (the evaluator's "trim5"): LSQ with a
    non-negative scale (nonneg_lsq_scale_and_shift) on the valid pixels, drop the
    ceil(trim * N) largest relative residuals |s p + t - g| / g (ties at the
    threshold kept), the same LSQ on the rest. (scale, shift, valid[, rejected]);
    valid is False only for < min_valid pixels, rejected marks a pass-2 fit
    replaced by the GT-mean constant."""
    s1, t1, valid, _ = nonneg_lsq_scale_and_shift(pred, target, mask, min_valid)
    rel = (pred * s1[:, None, None] + t1[:, None, None] - target).abs() / target
    keep = torch.zeros_like(mask)
    for b in torch.where(valid)[0].tolist():
        r = rel[b][mask[b]]
        k = r.numel() - math.ceil(round(trim * r.numel(), 6))
        keep[b] = mask[b] & (rel[b] <= r.kthvalue(k).values)
    s, t, valid2, rejected = nonneg_lsq_scale_and_shift(pred, target, keep, min_valid)
    valid = valid & valid2
    rejected = rejected & valid
    return (s, t, valid, rejected) if return_rejected else (s, t, valid)


@torch.no_grad()
def ransac_scale_and_shift(
    pred,
    target,
    mask,
    seed: int = 0,
    trials: int = 1000,
    chunk: int = 100,
    max_elements: int = 30_000_000,
    return_rejected: bool = False,
):
    """Seeded RANSAC line fit target ~ s * pred + t per image (the evaluator's
    align_solver="ransac" and "ransac" protocol; InfiniDepth's sklearn
    RANSACRegressor protocol, reproducible).

    Per image, on its valid pixels (fp32, on the input's device): a generator seeded
    with `seed` (per image: independent of the batch and the rank) draws `trials`
    index pairs; a pair with equal indices or equal predictions is not a trial. Each
    trial's line through its two points has the inliers |s p + t - g| <= theta,
    theta = median |g - median g| (sklearn's default residual_threshold). The best
    trial has the most inliers, ties broken by the higher R^2 of its line on its
    inliers, then by the later trial (sklearn's rule); no early stop. The result is
    the exact least-squares fit on the best trial's inliers; when it is degenerate
    or has scale <= 0, or when no trial is valid (a constant prediction), the mean
    ratio s = mean g / mean p, t = 0 (InfiniDepth's _ransac_align_depth).
    (scale, shift, valid[, rejected]) [B]; valid is False only for < 2 valid
    pixels, rejected marks the mean-ratio images.

    The residuals of up to `chunk` trials are evaluated at once, fewer when
    chunk * N would exceed max_elements (120 MB of fp32: a full-resolution ETH3D
    map has ~2e7 valid pixels); the result does not depend on the chunking.
    """
    B = pred.shape[0]
    scale = torch.full((B,), torch.nan, device=pred.device, dtype=pred.dtype)
    shift = torch.full_like(scale, torch.nan)
    valid = torch.zeros(B, dtype=torch.bool, device=pred.device)
    rejected = torch.zeros_like(valid)
    for b in range(B):
        p, g = pred[b][mask[b]], target[b][mask[b]]
        n = p.numel()
        if n < 2:
            continue
        valid[b] = True
        gen = torch.Generator(device=p.device).manual_seed(seed)
        idx = torch.randint(0, n, (trials, 2), generator=gen, device=p.device)
        pa, pb, ga, gb = p[idx[:, 0]], p[idx[:, 1]], g[idx[:, 0]], g[idx[:, 1]]
        ok = (idx[:, 0] != idx[:, 1]) & (pa != pb)
        s_t = (gb - ga) / torch.where(ok, pb - pa, 1.0)
        t_t = ga - s_t * pa
        ok &= torch.isfinite(s_t) & torch.isfinite(t_t)
        if not ok.any():  # e.g. a constant prediction
            scale[b], shift[b], rejected[b] = g.mean() / p.mean(), 0.0, True
            continue
        theta = _median((g - _median(g)).abs())
        rows = max(1, min(chunk, max_elements // n))
        best = None  # (inliers, r2, trial)
        for start in range(0, trials, rows):
            sl = slice(start, start + rows)
            res = (s_t[sl, None] * p[None] + t_t[sl, None] - g[None]).abs()
            inl = (res <= theta).to(p.dtype)
            count = inl.sum(dim=1)
            count = torch.where(ok[sl], count, -1.0)  # never picked
            n_in = count.clamp_min(1)
            mean_g = (inl @ g) / n_in
            ss_tot = (inl @ g.square()) - n_in * mean_g.square()
            ss_res = torch.where(inl > 0, res.square(), 0.0).sum(dim=1)
            r2 = torch.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
            # lexicographic (count, r2, trial index): the last of the best
            top = count.max()
            r2_top = torch.where(count == top, r2, -torch.inf)
            last = (r2_top == r2_top.max()).nonzero()[-1, 0]
            cand = (top.item(), r2_top[last].item(), start + last.item())
            if best is None or cand[:2] >= best[:2]:
                best = cand
        i = best[2]  # (ok.any(): the best trial is a valid one)
        inliers = (s_t[i] * pred[b] + t_t[i] - target[b]).abs() <= theta
        inliers &= mask[b]
        s, t, ok_fit = lsq_scale_and_shift(
            pred[b : b + 1], target[b : b + 1], inliers[None], min_valid=2
        )
        if ok_fit.item():
            scale[b], shift[b] = s[0], t[0]
        else:
            scale[b], shift[b], rejected[b] = g.mean() / p.mean(), 0.0, True
    return (scale, shift, valid, rejected) if return_rejected else (scale, shift, valid)


# ---------------------------------------------------- ROE (the L-DPT depth loss)
#
# MoGe's robust optimal estimation (ROE) of an affine alignment: the exact minimiser
# of sum_i min(trunc, w_i |s x_i + t - y_i|) (or of the plain weighted L1 without
# trunc). `_roe_align` / `_roe_support` are ported from microsoft/MoGe,
# moge/utils/alignment.py (`align`, `align_depth_affine`, L56-L150, L167-L218) at
# commit 74fbce054ebed49800de42d0ad0e83495065719a.
#
#   MIT License. Copyright (c) Microsoft Corporation.
#   Permission is hereby granted, free of charge, to any person obtaining a copy of
#   this software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, subject to the following conditions: the above
#   copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
#   WARRANTY OF ANY KIND.


def _scatter_min(size: int, dim: int, index: Tensor, src: Tensor):
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1 :]
    minimum = torch.full(
        shape, float("inf"), dtype=src.dtype, device=src.device
    ).scatter_reduce(dim=dim, index=index, src=src, reduce="amin", include_self=False)
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    indices[(*minimum_where[:dim], index[minimum_where], *minimum_where[dim + 1 :])] = (
        minimum_where[dim]
    )
    return minimum, indices


def _pad_inf(x: Tensor) -> Tensor:
    return torch.cat(
        [
            torch.full_like(x[..., :1], -torch.inf),
            x,
            torch.full_like(x[..., :1], torch.inf),
        ],
        dim=-1,
    )


def _pad_cumsum(cumsum: Tensor) -> Tensor:
    return torch.cat(
        [torch.zeros_like(cumsum[..., :1]), cumsum, cumsum[..., -1:]], dim=-1
    )


def _compute_residual(a: Tensor, xyw: Tensor, trunc: float) -> Tensor:
    return (
        a.mul(xyw[..., 0])
        .sub_(xyw[..., 1])
        .abs_()
        .mul_(xyw[..., 2])
        .clamp_max_(trunc)
        .sum(dim=-1)
    )


@torch.no_grad()
def _roe_align(x: Tensor, y: Tensor, w: Tensor, trunc=None, eps: float = 1e-7):
    """MoGe's `align`: per row of [M, n], min_a sum_i w_i |a x_i - y_i| (or of
    min(trunc, .) per term), a = y[index] / x[index]. (a, loss, index) [M]."""
    if trunc is None:
        sign = torch.sign(x)
        x, y = x * sign, y * sign
        y_div_x = y / x.clamp_min(eps)
        y_div_x, argsort = y_div_x.sort(dim=-1)
        wx = torch.gather(x * w, dim=-1, index=argsort)
        derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
        search = torch.searchsorted(
            derivatives, torch.zeros_like(derivatives[..., :1]), side="left"
        ).clamp_max(derivatives.shape[-1] - 1)
        a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
        index = argsort.gather(dim=-1, index=search).squeeze(-1)
        loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)
        return a, loss, index

    batch_size = x.shape[0]
    sign = torch.sign(x)
    x, y = x * sign, y * sign
    wx, wy = w * x, w * y
    xyw = torch.stack([x, y, w], dim=-1)

    y_div_x = A = y / x.clamp_min(eps)
    B = (wy - trunc) / wx.clamp_min(eps)
    C = (wy + trunc) / wx.clamp_min(eps)
    A, A_argsort = A.sort(dim=-1)
    Q_A = torch.cumsum(torch.gather(wx, dim=-1, index=A_argsort), dim=-1)
    A, Q_A = _pad_inf(A), _pad_cumsum(Q_A)
    B, B_argsort = B.sort(dim=-1)
    Q_B = torch.cumsum(torch.gather(wx, dim=-1, index=B_argsort), dim=-1)
    B, Q_B = _pad_inf(B), _pad_cumsum(Q_B)
    C, C_argsort = C.sort(dim=-1)
    Q_C = torch.cumsum(torch.gather(wx, dim=-1, index=C_argsort), dim=-1)
    C, Q_C = _pad_inf(C), _pad_cumsum(Q_C)

    j_A = torch.searchsorted(A, y_div_x, side="left").sub_(1)
    j_B = torch.searchsorted(B, y_div_x, side="left").sub_(1)
    j_C = torch.searchsorted(C, y_div_x, side="left").sub_(1)
    left_derivative = (
        2 * torch.gather(Q_A, dim=-1, index=j_A)
        - torch.gather(Q_B, dim=-1, index=j_B)
        - torch.gather(Q_C, dim=-1, index=j_C)
    )
    j_A = torch.searchsorted(A, y_div_x, side="right").sub_(1)
    j_B = torch.searchsorted(B, y_div_x, side="right").sub_(1)
    j_C = torch.searchsorted(C, y_div_x, side="right").sub_(1)
    right_derivative = (
        2 * torch.gather(Q_A, dim=-1, index=j_A)
        - torch.gather(Q_B, dim=-1, index=j_B)
        - torch.gather(Q_C, dim=-1, index=j_C)
    )

    is_extrema = (left_derivative < 0) & (right_derivative >= 0)
    is_extrema[..., 0] |= ~is_extrema.any(dim=-1)
    where_extrema_batch, where_extrema_index = torch.where(is_extrema)

    extrema_a = y_div_x[where_extrema_batch, where_extrema_index]
    split_size = max(1, 4096**2 // x.shape[-1])
    extrema_value = torch.cat(
        [
            _compute_residual(a_split[:, None], xyw[i_split, :, :], trunc)
            for a_split, i_split in zip(
                extrema_a.split(split_size), where_extrema_batch.split(split_size)
            )
        ]
    )
    minima, indices = _scatter_min(
        size=batch_size, dim=0, index=where_extrema_batch, src=extrema_value
    )
    index = where_extrema_index[indices]
    a = torch.gather(y, dim=-1, index=index[..., None]) / torch.gather(
        x, dim=-1, index=index[..., None]
    ).clamp_min(eps)
    return a.squeeze(-1), minima, index


@torch.no_grad()
def _roe_support(x: Tensor, y: Tensor, w: Tensor, trunc=None) -> Tuple[int, int]:
    """The anchor loop of MoGe's `align_depth_affine` for one image (1-D x, y, w of
    the same dtype): every point with w > 0 is an anchor (subtracting it removes the
    shift), the 1-D problem of each anchor is solved exactly by _roe_align and the
    best anchor wins. Returns the two support points (i1, i2) of the optimal line
    as 0-d index tensors.

    Two corrections to the port for points that tie with the anchor in x (x_j ==
    x_anchor in the search dtype; `_roe_align` zeroes them through sign(0)): their
    residual w_j |y_j - y_anchor| (truncated), which no slope changes, is added to
    the anchor's loss, so anchors are compared on the full objective; and when the
    chosen candidate is such a point (its slope is a = 0, a horizontal line through
    the anchor), the anchor itself is returned as the second support point, so that
    the slope recomputed from the pair is 0, not (y_j - y_anchor) / tiny."""
    x, y, w = x.reshape(1, -1), y.reshape(1, -1), w.reshape(1, -1)
    _, anchors = torch.where(w > 0)
    x_anchored = x - x[0, anchors][:, None]
    y_anchored = y - y[0, anchors][:, None]
    w_anchored = w.expand(anchors.numel(), -1)
    _, loss, index = _roe_align(x_anchored, y_anchored, w_anchored, trunc)
    tied = x_anchored == 0
    tied_res = (w_anchored * y_anchored.abs()).where(tied, 0.0)
    if trunc is not None:
        tied_res = tied_res.clamp_max(trunc)
    loss = loss + tied_res.sum(dim=-1)
    _, index_anchor = _scatter_min(
        size=1, dim=0, index=torch.zeros_like(anchors), src=loss
    )
    i1, i2 = anchors[index_anchor][0], index[index_anchor][0]
    if bool(tied[index_anchor[0], i2]):
        i2 = i1
    return i1, i2


@torch.no_grad()
def roe_scale_and_shift(
    x: Tensor, y: Tensor, w: Tensor, trunc: Optional[float] = None
) -> Tuple[Tensor, Tensor]:
    """(s, t), 0-d tensors in the inputs' dtype: the exact minimiser of
    sum_i min(trunc, w_i |s x_i + t - y_i|) (trunc=None: sum_i w_i |s x_i + t - y_i|)
    over 1-D inputs of one image (MoGe's ROE solver, `align_depth_affine`, see the
    MIT notice above: "Copyright (c) Microsoft Corporation", commit 74fbce0). The
    search runs in the inputs' dtype on n x n work arrays (O(n^2 log n)); (s, t) are
    recomputed from the two support points as MoGe does. Detached."""
    i1, i2 = _roe_support(x, y, w, trunc)
    x1, x2, y1, y2 = x[i1], x[i2], y[i1], y[i2]
    s = torch.where(x2 != x1, (y2 - y1) / torch.where(x2 != x1, x2 - x1, 1.0), 0.0)
    return s, y1 - s * x1


@torch.no_grad()
def roe_fit_sample(
    x: Tensor,
    y: Tensor,
    w: Tensor,
    h,
    trunc: Optional[float] = None,
    search_dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor, bool]:
    """ROE fit of one image's sampled pixels (prediction x, target y, weights w,
    1-D) on conditioned inputs (docs/depth_loss.md section 4.1): in fp64,
    x' = (x - median x) / spread, spread = median |x - median x| (the mean where the
    median is 0), y' = y / h, w' = w h (h: the image's depth unit, the harmonic-mean
    depth), the search in `search_dtype`, then
    s = s' h / spread, t = h t' - s median. Equal in exact arithmetic (the objective is
    equivariant), and the 1e-7 guards of the search stay meaningful for any offset of
    x.
    Returns (s, t, ok): fp64 0-d tensors, ok False for a constant x (spread <=
    1e-12 max(|median|, 1e-30); s, t NaN), a constant sampled y (s =
    0, t = y: the horizontal optimum, found without the search), a support pair
    closer than 1e-6 in conditioned x (s = 0), a non-finite result or s <= 0."""
    x, y, w = x.double(), y.double(), w.double()
    h = torch.as_tensor(h, dtype=torch.float64, device=x.device)
    med = x.median()
    # robust spread: the median absolute deviation, so one extreme sampled d cannot
    # push every other point under the search's 1e-7 guards; the mean absolute
    # deviation where more than half of x equals the median
    dev = (x - med).abs()
    spread = dev.median()
    if not bool(spread > 0):
        spread = dev.mean()
    nan = torch.full((), torch.nan, dtype=torch.float64, device=x.device)
    if not bool(spread > 1e-12 * med.abs().clamp_min(1e-30)):
        return nan, nan.clone(), False
    xs, ys, ws = (x - med) / spread, y / h, w * h
    if not bool((ys != ys[0]).any()):
        # constant sampled target: the optimum is the horizontal line (s = 0, the
        # image is dropped); skipping the search avoids its n^2 tied candidates
        return torch.zeros_like(nan), h * ys[0], False
    # A sample whose conditioned x spans more than 1e4 (one extreme logit, say)
    # would swamp the search's cumulative sums in fp32; such rare samples are
    # searched in fp64.
    dt = search_dtype if bool(xs.abs().max() <= 1e4) else torch.float64
    i1, i2 = _roe_support(xs.to(dt), ys.to(dt), ws.to(dt), trunc)
    # (s', t') from the two support points, in fp64 on the conditioned values. A
    # pair closer than 1e-6 in x (units of the spread, so far below
    # any real slope's support; an exact tie in the search dtype is returned as
    # i2 = i1 by _roe_support) is a horizontal line: s' = 0, the image is dropped.
    x1, x2, y1, y2 = xs[i1], xs[i2], ys[i1], ys[i2]
    apart = (x2 - x1).abs() >= 1e-6
    s_n = torch.where(apart, (y2 - y1) / torch.where(apart, x2 - x1, 1.0), 0.0)
    t_n = y1 - s_n * x1
    s = s_n * h / spread
    t = h * t_n - s * med
    ok = bool(torch.isfinite(s) & torch.isfinite(t) & (s > 0))
    return s, t, ok


def sample_valid(mask: Tensor, n: int, jitter: bool = True) -> Tensor:
    """[B, n] flat indices (into mask.flatten(1)) of valid pixels: the floor((k +
    xi_k) N / n)-th valid pixel in row-major order, k = 0..n-1, xi_k ~ U[0, 1) from
    the global generator on mask's device (jitter=False: xi = 0.5, deterministic).
    Every valid pixel has the same probability; N < n repeats pixels. An image
    without valid pixels gets index 0 (the caller skips it)."""
    B = mask.shape[0]
    flat = mask.flatten(1).long()
    csum = flat.cumsum(dim=1)
    N = csum[:, -1:].double()
    k = torch.arange(n, dtype=torch.float64, device=mask.device)[None]
    if jitter:
        xi = torch.rand(B, n, dtype=torch.float64, device=mask.device)
    else:
        xi = torch.full((B, n), 0.5, dtype=torch.float64, device=mask.device)
    rank = ((k + xi) * N / n).floor()
    rank = torch.minimum(rank, N - 1).long() + 1  # 1-based
    idx = torch.searchsorted(csum, rank, right=False)
    return idx.clamp_max(flat.shape[1] - 1)


def _weighted_median(values: Tensor, weights: Tensor) -> Tensor:
    """The first value, in increasing order, at which the cumulative weight reaches
    half of the total (a minimiser of sum_i w_i |v_i - c|)."""
    v, order = values.sort()
    cw = weights[order].double().cumsum(0)
    i = torch.searchsorted(cw, 0.5 * cw[-1:]).clamp_max(v.numel() - 1)
    return v[i][0]


@torch.no_grad()
def roe_eval_scale_and_shift(
    pred, target, mask, points: int = 4096, min_valid: int = 10, return_rejected=False
):
    """The evaluator's "roe" protocol (MoGe's evaluation fit): per image, the exact
    1/g-weighted L1 fit (ROE, no truncation) on `points` deterministic samples of
    the valid pixels (sample_valid, jitter=False), conditioned as roe_fit_sample
    (fp32 search). A rejected fit (constant prediction, non-finite, s <= 0) gives
    s = 0 and t = the 1/g-weighted median of the target over the valid pixels (the
    best constant). (scale, shift, valid[, rejected]) [B] in pred's dtype; valid is
    False only for < min_valid pixels (scale / shift NaN). Working memory:
    points x points arrays (measured on CPU: 0.75 GiB peak at 4096)."""
    B = pred.shape[0]
    mask = mask.to(device=pred.device, dtype=torch.bool)
    num = mask.flatten(1).sum(dim=1)
    valid = num >= min_valid
    scale = torch.full((B,), torch.nan, device=pred.device, dtype=pred.dtype)
    shift = torch.full_like(scale, torch.nan)
    rejected = torch.zeros_like(valid)
    if not valid.any():
        return (
            (scale, shift, valid, rejected)
            if return_rejected
            else (scale, shift, valid)
        )
    idx = sample_valid(mask, points, jitter=False)
    for b in torch.where(valid)[0].tolist():
        g = target[b][mask[b]].double()
        h = g.numel() / (1.0 / g).sum()
        d_s = pred[b].flatten()[idx[b]].double()
        g_s = target[b].flatten()[idx[b]].double()
        s, t, ok = roe_fit_sample(d_s, g_s, 1.0 / g_s, h, trunc=None)
        if ok:
            scale[b], shift[b] = s.to(pred.dtype), t.to(pred.dtype)
        else:
            scale[b] = 0.0
            shift[b] = _weighted_median(g, 1.0 / g).to(pred.dtype)
            rejected[b] = True
    return (scale, shift, valid, rejected) if return_rejected else (scale, shift, valid)
