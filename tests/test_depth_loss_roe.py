"""The L-DPT depth loss (relative_depth_loss, docs/depth_loss.md): the ROE solver,
the sampler, the refit, the local and gradient terms, the criterion's invariances,
routing, degenerate cases, keys, option validation, the loss without options and
under autocast. CPU, fixed seeds, fp32 inputs. tests/test_l_dpt_golden.py pins the
loss inside the model bit for bit."""

import math

import pytest
import torch
import torch.nn.functional as F

from ldpt.models.losses import depth as D
from ldpt.models.losses.depth import RelativeDepthLoss
from ldpt.models.utils.depth import (
    roe_eval_scale_and_shift,
    roe_fit_sample,
    roe_scale_and_shift,
    sample_valid,
)

LABELS = dict(A=["hypersim", "tartanair"], B=["blendedmvs"], C=["kitti", "nyuv2"])
NAME = dict(A="hypersim/scene/0001", B="blendedmvs/x/2", C="kitti/drive/3")
ROE = dict(
    align="roe",
    head="exp",
    min_valid=1000,
    roe_points=512,
    label_types=LABELS,
    type_trim=dict(A=0.0, B=0.05, C=0.05),
    local_levels=dict(A=[4, 16, 64], B=[4, 16], C=[4]),
    local_cell=256.0,
    grad_types=["A"],
    mage_weight=4.0,
    mage_num_scales=4,
)
TERMS = ("loss", "depth_loss", "local_loss", "mage_loss")
KEYS = list(RelativeDepthLoss.ROE_KEYS)
LOGIT_KEYS = ["logit_mean", "logit_std", "logit_absmax"]


def crit(**kw):
    return RelativeDepthLoss(**{**ROE, **kw})


# ------------------------------------------------------------------ synthetic scenes
def scene(hw, lo, hi, gen, planes=6):
    """Piecewise-planar depth map in [lo, hi] m (a synthetic scene)."""
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, hw, dtype=torch.float64),
        torch.linspace(0, 1, hw, dtype=torch.float64),
        indexing="ij",
    )
    g = torch.full((hw, hw), float(hi), dtype=torch.float64)
    for _ in range(planes):
        d0 = math.exp(
            math.log(lo) + torch.rand((), generator=gen).item() * math.log(hi / lo)
        )
        gx, gy = (torch.rand(2, generator=gen) - 0.5).tolist()
        cx, cy, r = torch.rand(3, generator=gen).tolist()
        inside = ((xx - cx).abs() < 0.15 + 0.35 * r) & (
            (yy - cy).abs() < 0.15 + 0.35 * r
        )
        plane = d0 * (1 + 0.6 * gx * (xx - cx) + 0.6 * gy * (yy - cy))
        g = torch.where(inside, torch.minimum(g, plane.clamp(lo, hi)), g)
    ground = lo + (hi - lo) * (1 - yy).clamp(0, 1) ** 2
    return torch.minimum(g, ground.clamp(lo, hi))


def make(hw, kind, gen, quality=0.15, outliers=0.0, holes=0.0):
    """(pred d = exp(u), target g, mask), fp32: u = log g + smooth error + noise."""
    lo, hi = (0.5, 10.0) if kind == "indoor" else (2.0, 80.0)
    g = scene(hw, lo, hi, gen)
    k = max(3, (31 * hw // 192) | 1)
    noise = torch.randn(1, 1, hw, hw, generator=gen, dtype=torch.float64)
    smooth = F.avg_pool2d(noise, k, 1, k // 2)[0, 0]
    smooth = smooth / smooth.std()
    rough = torch.randn(hw, hw, generator=gen, dtype=torch.float64)
    u = torch.log(g / g.median()) + quality * smooth + 0.3 * quality * rough
    gt = g.clone()
    if outliers > 0:
        bad = torch.rand(hw, hw, generator=gen) < outliers
        scale = 2 + 8 * torch.rand(hw, hw, generator=gen, dtype=torch.float64)
        gt = torch.where(bad, g * scale, g)
    mask = torch.ones(hw, hw, dtype=torch.bool)
    if holes > 0:
        mask = torch.rand(hw, hw, generator=gen) > holes
    return torch.exp(u).float(), gt.float(), mask


def batch(kinds, hw=96, seed=0, **kw):
    """(pred, target, mask, names) of one image per entry of kinds ("A" / "B" /
    "C"); A and B indoor, C outdoor with 30 % holes."""
    gen = torch.Generator().manual_seed(seed)
    items = []
    for k in kinds:
        if k == "C":
            items.append(make(hw, "outdoor", gen, holes=0.3, **kw))
        elif k == "B":
            items.append(make(hw, "indoor", gen, outliers=0.02, holes=0.02, **kw))
        else:
            items.append(make(hw, "indoor", gen, **kw))
    pred, target, mask = (torch.stack(x) for x in zip(*items))
    return pred, target, mask, [NAME[k] for k in kinds]


def run(criterion, pred, target, mask, names, seed=0):
    torch.manual_seed(seed)
    return criterion(pred, target, mask, names=names)


def grad_of(criterion, pred, target, mask, names, seed=0, key="loss"):
    p = pred.clone().requires_grad_(True)
    out = run(criterion, p, target, mask, names, seed)
    (g,) = torch.autograd.grad(out[key], p)
    return out, g


def objective(s, t, x, y, w, trunc=None):
    r = w * (s * x + t - y).abs()
    if trunc is not None:
        r = r.clamp_max(trunc)
    return r.sum(-1)


def brute_force(x, y, w, trunc=None):
    """Smallest objective over the lines through every pair of points."""
    n = x.shape[0]
    i, j = torch.triu_indices(n, n, offset=1)
    dx = x[j] - x[i]
    ok = dx.abs() > 1e-12
    s = (y[j] - y[i])[ok] / dx[ok]
    t = y[i][ok] - s * x[i][ok]
    return objective(s[:, None], t[:, None], x[None], y[None], w[None], trunc).min()


def rel_close(a, b, tol, atol=1e-12):
    a, b = float(a), float(b)
    return abs(a - b) <= tol * max(abs(a), abs(b)) + atol


# ------------------------------------------------------------------ solver, sampler
@pytest.mark.parametrize("trunc", [None, 1.0, 0.3])
def test_t1_roe_exact_against_brute_force(trunc):
    gen = torch.Generator().manual_seed(1)
    for _ in range(200):
        n = 40
        y = torch.rand(n, generator=gen, dtype=torch.float64) * 9 + 1
        x = (
            (y - 1) / 9 * 0.5
            + 0.2
            + 0.03 * torch.randn(n, generator=gen, dtype=torch.float64)
        )
        out = torch.rand(n, generator=gen) < 0.15  # gross outliers
        x = torch.where(out, torch.rand(n, generator=gen, dtype=torch.float64), x)
        w = 1.0 / y
        s, t = roe_scale_and_shift(x, y, w, trunc)
        assert objective(s, t, x, y, w, trunc) <= brute_force(x, y, w, trunc) + 1e-9


def test_t1b_roe_ties_with_the_anchor():
    """A point with the anchor's x but another y: its residual counts and no slope is
    computed from the tied pair (no dy / 1e-7); the fit is the horizontal line, s = 0,
    in every order of the points."""
    x = torch.tensor([0.0, 1.0, -1.0, 0.0])
    y = torch.tensor([1.0, 1.0, 1.0, 1.5])
    w = torch.ones(4)
    for perm in ([0, 1, 2, 3], [3, 0, 1, 2], [1, 2, 3, 0], [2, 3, 0, 1]):
        p = torch.tensor(perm)
        s, t = roe_scale_and_shift(x[p], y[p], w[p], 1.0)
        assert float(s) == 0.0 and float(t) == 1.0
    s, t, ok = roe_fit_sample(x, y, w, 1.0, 1.0)
    assert not ok and float(s) == 0.0


@pytest.mark.parametrize("trunc", [None, 1.0])
def test_t1c_roe_exact_with_tied_predictions(trunc):
    """Quantised predictions (many exact ties in x): still the exact optimum over
    all two-point lines and horizontal lines, and never a near-vertical pair."""
    gen = torch.Generator().manual_seed(11)
    for _ in range(100):
        n = 40
        y = torch.rand(n, generator=gen, dtype=torch.float64) * 9 + 1
        x = torch.randint(0, 5, (n,), generator=gen).double() / 4
        w = 1.0 / y
        s, t = roe_scale_and_shift(x, y, w, trunc)
        flat = objective(
            torch.zeros(n, 1, dtype=torch.float64),
            y[:, None],
            x[None],
            y[None],
            w[None],
            trunc,
        ).min()
        best = torch.minimum(brute_force(x, y, w, trunc), flat)
        assert objective(s, t, x, y, w, trunc) <= best + 1e-9
        assert float(s.abs()) < 1e3


def test_t1d_roe_tie_residuals_count_without_truncation():
    """Anchors inside a tie group count every residual (x = [0, 0, 0, 1, 2],
    y = [10, 14, 6, 11, 15]): the optimum is the line through points 0 and 4
    (objective 9.5), not s = 4.5, t = 6 (objective 12.5)."""
    x = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([10.0, 14.0, 6.0, 11.0, 15.0], dtype=torch.float64)
    w = torch.ones(5, dtype=torch.float64)
    for trunc in (None, 100.0):
        s, t = roe_scale_and_shift(x, y, w, trunc)
        assert abs(float(objective(s, t, x, y, w, trunc)) - 9.5) < 1e-9


def test_t1e_roe_constant_target_is_dropped_without_search():
    gen = torch.Generator().manual_seed(5)
    x = torch.rand(2304, generator=gen, dtype=torch.float64)
    y = torch.full((2304,), 3.0, dtype=torch.float64)
    s, t, ok = roe_fit_sample(x, y, 1.0 / y, 3.0, 1.0)
    assert not ok and float(s) == 0.0 and abs(float(t) - 3.0) < 1e-12


def test_t5c_refit_dropped_image_has_a_clean_zero_gradient():
    """An image rejected after the division (s <= 0) must give an exactly zero,
    finite gradient, next to a fitted image."""
    g = torch.linspace(1.0, 3.0, 64, dtype=torch.float64).repeat(2, 1)
    p = torch.stack([g[0] * 0.5 + 0.1, -g[1] * 0.5 + 3.0]).requires_grad_(True)
    w = 1.0 / g
    m = torch.ones(2, 64, dtype=torch.bool)
    s0 = torch.tensor([2.0, 2.0], dtype=torch.float64)
    t0 = torch.tensor([-0.2, -3.0], dtype=torch.float64)
    s, t, ok = D.refit_in_graph(p, g, w, m, s0, t0, 100.0, 0.01)
    assert bool(ok[0]) and not bool(ok[1])
    ((s[:, None] * p + t[:, None] - g).abs() * w).where(
        ok[:, None], 0.0
    ).sum().backward()
    assert torch.isfinite(p.grad).all() and bool((p.grad[1] == 0).all())
    assert bool((p.grad[0] != 0).any())


def test_t5b_refit_rejects_a_rounding_variance():
    """Only pixels with one prediction value inside the truncation: the weighted
    variance is a rounding residue, so the image is dropped (no huge, random-sign
    scale); loss and gradient stay finite."""
    p = torch.full((1, 64), 0.1, dtype=torch.float64)
    p[0, 32:] = torch.linspace(1.0, 2.0, 32, dtype=torch.float64)
    p.requires_grad_(True)
    g = torch.linspace(1.0, 3.0, 64, dtype=torch.float64)[None]
    w = 1.0 / g
    m = torch.ones(1, 64, dtype=torch.bool)
    s0 = torch.tensor([5e6], dtype=torch.float64)
    t0 = g[:, :1] - s0 * 0.1  # the line through (0.1, g_0): only p == 0.1 pixels fit
    s, t, ok = D.refit_in_graph(p, g, w, m, s0, t0[:, 0], 1.0, 0.01)
    assert not bool(ok[0])
    (s.where(ok, 0.0).sum() + t.where(ok, 0.0).sum()).backward()
    assert torch.isfinite(p.grad).all()


def _sampled(hw=128, seed=2, n=1024, kind="indoor"):
    gen = torch.Generator().manual_seed(seed)
    p, g, m = make(hw, kind, gen, outliers=0.02)
    torch.manual_seed(seed)
    idx = sample_valid(m[None], n)[0]
    return p.flatten()[idx].double(), g.flatten()[idx].double(), g[m].double()


def _h(g_valid):
    return g_valid.numel() / (1.0 / g_valid).sum()


def test_t2_roe_equivariance():
    d, g, gv = _sampled()
    s, t, ok = roe_fit_sample(d, g, 1.0 / g, _h(gv), 1.0)
    assert ok
    for c in (37.0, 0.01):
        s_c, t_c, ok_c = roe_fit_sample(d, c * g, 1.0 / (c * g), _h(c * gv), 1.0)
        assert ok_c and rel_close(s_c / c, s, 1e-6) and rel_close(t_c / c, t, 1e-6)
    for a, b in ((37.0, 0.0), (3.0, -0.2), (0.5, 0.1)):
        s_a, t_a, ok_a = roe_fit_sample(a * d + b, g, 1.0 / g, _h(gv), 1.0)
        assert (
            ok_a and rel_close(s_a * a, s, 1e-6) and rel_close(t_a + s_a * b, t, 1e-6)
        )


def test_t3b_conditioning_with_one_extreme_prediction():
    """One sampled logit 25 above the rest must not push the other points under the
    search's guards (robust spread); the fp32 fit's objective equals that of the
    fp64 solve on the raw inputs."""
    gen = torch.Generator().manual_seed(7)
    n = 512
    g = torch.rand(n, generator=gen, dtype=torch.float64) * 9 + 1
    u = torch.log(g) + 0.05 * torch.randn(n, generator=gen, dtype=torch.float64)
    u[17] = u.max() + 25.0
    d = torch.exp(u).float()
    w = 1.0 / g
    h = n / (1.0 / g).sum()
    s32, t32, ok = roe_fit_sample(d, g, w, h, 1.0)
    s64, t64 = roe_scale_and_shift(d.double(), g, w, 1.0)
    assert ok
    o32 = objective(s32, t32, d.double(), g, w, 1.0)
    o64 = objective(s64, t64, d.double(), g, w, 1.0)
    assert rel_close(o32, o64, 1e-6, atol=1e-9)


def test_t3_conditioning_large_offset():
    gen = torch.Generator().manual_seed(3)
    p, g, m = make(256, "indoor", gen)
    p = (1000.0 + 0.01 * p.double()).float()  # fp32 values of a 1000-offset map
    torch.manual_seed(3)
    idx = sample_valid(m[None], 1024)[0]
    d, gs = p.flatten()[idx], g.flatten()[idx].double()
    s32, _, ok = roe_fit_sample(d, gs, 1.0 / gs, _h(g[m].double()), 1.0)
    s64, _ = roe_scale_and_shift(d.double(), gs, 1.0 / gs, 1.0)
    assert ok and rel_close(s32, s64, 1e-5)


def test_t4_sample_valid():
    m = torch.zeros(1, 64, 64, dtype=torch.bool)
    m[0, 10:30, 5:60] = True
    m[0, 40:44, 0:8] = True
    torch.manual_seed(4)
    cnt = torch.zeros(64 * 64)
    for _ in range(400):
        idx = sample_valid(m, 256)
        assert idx.shape == (1, 256)
        cnt.index_add_(0, idx[0], torch.ones(256))
    flat = m.flatten()
    assert cnt[~flat].sum() == 0  # only valid pixels
    c = cnt[flat]
    mean = 400 * 256 / flat.sum().item()
    assert c.min() >= 0.6 * mean and c.max() <= 1.4 * mean  # uniform
    # N < n: valid pixels repeated, every one of them drawn
    small = torch.zeros(2, 16, 16, dtype=torch.bool)
    small[0, 3, 3:8] = True
    small[1, 10:12, 1:4] = True
    idx = sample_valid(small, 64)
    for b in range(2):
        assert small[b].flatten()[idx[b]].all()
        assert set(idx[b].tolist()) == set(torch.where(small[b].flatten())[0].tolist())
    # deterministic without jitter, whatever the generator state
    torch.manual_seed(0)
    a = sample_valid(m, 300, jitter=False)
    torch.manual_seed(1)
    assert torch.equal(a, sample_valid(m, 300, jitter=False))
    assert not torch.equal(sample_valid(m, 300), sample_valid(m, 300))


# ------------------------------------------------------------------ refit
def test_t5_refit_closed_form_and_quality():
    gen = torch.Generator().manual_seed(5)
    trunc, floor = 1.0, 0.01
    scenes = [
        make(128, kind, gen, outliers=outl, holes=0.05)
        for kind in ("indoor", "outdoor")
        for outl in (0.0, 0.03)
        for _ in range(2)
    ]
    for p, g, m in scenes:
        p, g, m = p[None], g[None], m[None]
        w, h = D.pixel_weights(g, m, 10.0)
        torch.manual_seed(0)
        idx = sample_valid(m, 2304)[0]
        s0, t0, ok = roe_fit_sample(
            p.flatten()[idx], g.flatten()[idx], w.flatten()[idx], h[0], trunc
        )
        assert ok
        s, t, ok_r = D.refit_in_graph(p, g, w, m, s0[None], t0[None], trunc, floor)
        assert ok_r.all()
        # closed form: weighted least squares with omega = w^2 / max(e, floor)
        pd, gd, wd = p[m].double(), g[m].double(), w[m]
        e = wd * (s0 * pd + t0 - gd).abs()
        om = torch.where(e <= trunc, wd.square() / e.clamp_min(floor), 0.0)
        A = torch.stack([pd, torch.ones_like(pd)], 1) * om.sqrt()[:, None]
        sol = torch.linalg.lstsq(A, gd * om.sqrt()).solution
        assert rel_close(s[0], sol[0], 1e-9) and rel_close(t[0], sol[1], 1e-9, 1e-12)
        # the refit does not lose against the ROE fit on the full image
        l1 = lambda a, b: (wd * (a * pd + b - gd).abs()).sum().item()  # noqa: E731
        assert l1(s[0].item(), t[0].item()) <= l1(s0.item(), t0.item()) * (1 + 1e-3)
    # every pixel truncated: dropped, finite values
    p, g, m = scenes[0]
    p, g, m = p[None], g[None], m[None]
    w, h = D.pixel_weights(g, m, 10.0)
    s0 = torch.ones(1, dtype=torch.float64)
    t0 = 100.0 * h
    s, t, ok = D.refit_in_graph(p, g, w, m, s0, t0, trunc, floor)
    assert not ok.any() and torch.isfinite(s).all() and torch.isfinite(t).all()
    # constant prediction: dropped by the criterion, finite zero loss and gradient
    pc = torch.full_like(p, 0.7).requires_grad_(True)
    out = run(crit(), pc, g, m, [NAME["A"]])
    out["loss"].backward()
    assert out["drop_frac"] == 1.0 and out["loss"] == 0.0
    assert torch.isfinite(pc.grad).all()


# ------------------------------------------------------------------ invariance, gradient
def test_t6_affine_invariance():
    pred, target, mask, names = batch("AB", hw=96, seed=6)
    criterion = crit()
    base = run(criterion, pred, target, mask, names, seed=6)
    for a, b in ((37.0, 0.0), (3.0, -0.2), (0.5, 0.1)):
        q = (a * pred.double() + b).float()
        out = run(criterion, q, target, mask, names, seed=6)
        for k in TERMS:
            assert rel_close(out[k], base[k], 1e-4), (a, b, k, out[k], base[k])


def span_share(gr, p, m):
    """Share of the gradient norm along span{1, p} over the valid pixels."""
    v, x = gr[m].double(), p[m].double()
    one = torch.ones_like(x) / math.sqrt(x.numel())
    d = x - x.mean()
    d = d / d.norm()
    return math.sqrt(((v @ one) ** 2 + (v @ d) ** 2).item()) / max(
        v.norm().item(), 1e-300
    )


def term_grads(pred, target, mask, names, seed, **kw):
    """Gradients of the total loss and of each term (the global term alone; the
    local and gradient terms as differences with the global-only criterion)."""
    _, g_glob = grad_of(
        crit(local_levels=None, grad_types=(), **kw), pred, target, mask, names, seed
    )
    _, g_loc = grad_of(crit(grad_types=(), **kw), pred, target, mask, names, seed)
    _, g_gm = grad_of(crit(local_levels=None, **kw), pred, target, mask, names, seed)
    _, g_tot = grad_of(crit(**kw), pred, target, mask, names, seed)
    return dict(total=g_tot, glob=g_glob, local=g_loc - g_glob, grad=g_gm - g_glob)


def test_t7_gradient_has_no_affine_component():
    pred, target, mask, names = batch("AB", hw=96, seed=7)
    grads = term_grads(pred, target, mask, names, 7)
    for term, gr in grads.items():
        for i in range(2):
            if gr[i].abs().max() == 0:  # the gradient term of the type-B image
                assert term == "grad" and i == 1
                continue
            assert span_share(gr[i], pred[i], mask[i]) < 1e-6, (term, i)


def test_t8_gradient_not_concentrated():
    pred, target, mask, names = batch("A", hw=256, seed=8)
    criterion = crit(roe_points=2304, local_cell=1024.0)
    _, gr = grad_of(criterion, pred, target, mask, names, 8)
    v = gr.flatten().double().square()
    assert v.topk(2).values.sum() / v.sum() < 0.01


def freeze_decisions(monkeypatch):
    """Replay every detached quantity of the first call in the later calls: the ROE
    fits, the refit weights, the trim sets and the segment labels (docs/depth_loss.md
    section 2: constants of the backward). Returns a function that rewinds the
    replay before each call."""
    rec, pos = {}, {}

    def replay(name):
        fn = getattr(D, name)
        rec[name], pos[name] = [], 0

        def wrapped(*a, **k):
            i = pos[name]
            pos[name] += 1
            if i == len(rec[name]):
                rec[name].append(fn(*a, **k))
            return rec[name][i]

        monkeypatch.setattr(D, name, wrapped)

    for name in ("roe_fit_sample", "refit_weights", "smallest_k", "local_segments"):
        replay(name)

    def rewind():
        for name in pos:
            pos[name] = 0

    return rewind


def test_t9_directional_derivative(monkeypatch):
    """Central differences of the loss with the detached quantities held fixed (they
    are constants of the backward by design; the loss itself also moves through
    them: the ROE support points and the refit weights depend on d)."""
    pred, target, mask, names = batch("AB", hw=96, seed=9)
    rewind = freeze_decisions(monkeypatch)
    criterion = crit()
    _, gr = grad_of(criterion, pred, target, mask, names, 9)
    gen = torch.Generator().manual_seed(9)
    h = 3e-4  # L1 kinks: a larger step crosses many

    def f(q):
        rewind()
        return run(criterion, q.float(), target, mask, names, 9)["loss"].double()

    for _ in range(5):
        v = torch.randn(2, 1, 96, 96, generator=gen, dtype=torch.float64)
        v = F.avg_pool2d(v, 15, 1, 7)[:, 0]
        v = v / v.abs().max() * pred.double()  # smooth, relative to d
        fd = ((f(pred.double() + h * v) - f(pred.double() - h * v)) / (2 * h)).item()
        an = (gr.double() * v).sum().item()
        scale = (gr.double() * v).abs().sum().item()  # an without cancellation
        assert abs(fd - an) <= 0.02 * max(abs(an), 0.1 * scale), (fd, an, scale)


# ------------------------------------------------------------------ masks
def test_t10_invalid_pixels_do_not_matter():
    pred, target, mask, names = batch("ABC", hw=96, seed=10)
    criterion = crit(log_logit_stats=True)
    ref, g_ref = None, None
    gen = torch.Generator().manual_seed(10)
    for fill in (0.0, -1.0, 50.0):
        t = torch.where(mask, target, fill)
        junk = torch.rand(pred.shape, generator=gen) * 100 + 0.01
        p = torch.where(mask, pred, junk).requires_grad_(True)
        out = run(criterion, p, t, mask, names, 10)
        out["loss"].backward()
        assert (p.grad[~mask] == 0).all()
        if ref is None:
            ref, g_ref = out, p.grad
            continue
        for k in ref:
            assert torch.equal(out[k], ref[k]), k
        assert torch.equal(p.grad, g_ref)


# ------------------------------------------------------------------ routing
def test_t11_routing(monkeypatch):
    pred, target, mask, _ = batch("ABC", hw=96, seed=11)
    criterion = crit()
    with pytest.raises(ValueError, match="no label_types"):
        run(criterion, pred, target, mask, ["hypersim/a", "eth3d/b", "kitti/c"])
    with pytest.raises(ValueError, match="sample names"):
        run(criterion, pred, target, mask, None)
    with pytest.raises(ValueError, match="sample names"):
        run(criterion, pred, target, mask, ["hypersim/a"])
    with pytest.raises(ValueError, match="no label_types"):  # case-sensitive
        run(criterion, pred[:1], target[:1], mask[:1], ["Hypersim/a"])

    seen = dict(keep=None, alphas=[])
    refit, segments = D.refit_in_graph, D.local_segments

    def spy_refit(pred, target, w, keep, *a):
        seen["keep"] = keep.clone()
        return refit(pred, target, w, keep, *a)

    def spy_segments(z, h, alpha, *a):
        seen["alphas"].append(alpha)
        return segments(z, h, alpha, *a)

    monkeypatch.setattr(D, "refit_in_graph", spy_refit)
    monkeypatch.setattr(D, "local_segments", spy_segments)
    for kind, i in (("A", 0), ("B", 1), ("C", 2)):
        seen["alphas"] = []
        sl = slice(i, i + 1)
        out = run(criterion, pred[sl], target[sl], mask[sl], [NAME[kind]], 11)
        assert out["drop_frac"] == 0.0
        assert seen["alphas"] == ROE["local_levels"][kind]
        n = int(mask[i].sum())
        kept = int(seen["keep"].sum())
        assert (seen["keep"] <= mask[sl]).all()
        if kind == "A":
            assert torch.equal(seen["keep"], mask[sl])
            assert out["gm_image_frac"] == 1.0 and out["mage_loss"] > 0
        else:
            assert kept == math.ceil(round(0.95 * n, 6))
            assert out["trim_thresh_median"] > 0
        if kind == "C":
            assert out["mage_loss"] == 0.0 and out["gm_image_frac"] == 0.0
    # a mixed batch: the gradient term takes the type-A image only
    out = run(criterion, pred, target, mask, [NAME[k] for k in "ABC"], 11)
    assert rel_close(out["gm_image_frac"], 1 / 3, 1e-6)


# ------------------------------------------------------------------ local term
def test_t12_local_term_against_loop(monkeypatch):
    pred, target, mask, names = batch("AB", hw=96, seed=12)
    criterion = crit(local_levels=dict(A=[4, 16], B=[16]), grad_types=())
    calls = []
    segments, shift = D.local_segments, D.segment_shift

    def spy_segments(z, h, alpha, cell, phi, offset, beta):
        lab = segments(z, h, alpha, cell, phi, offset, beta)
        calls.append(
            dict(alpha=alpha, h=h, z=z, phi=phi, offset=offset, beta=beta, lab=lab)
        )
        return lab

    def spy_shift(x, w, seg, rank=None):
        out = shift(x, w, seg, rank)
        calls[-1].update(x=x.detach(), w=w, seg=seg, shift=out.detach())
        return out

    monkeypatch.setattr(D, "local_segments", spy_segments)
    monkeypatch.setattr(D, "segment_shift", spy_shift)
    out = run(criterion, pred, target, mask, names, 12)
    assert [c["alpha"] for c in calls] == [4, 16]
    H = W = 96
    total = 0.0
    for c in calls:
        # the weighted median per segment: a minimiser of sum w |x - c| (loop)
        x, w, seg = c["x"], c["w"], c["seg"]
        r = torch.zeros(())
        for k in seg.unique():
            sel = seg == k
            xs, ws, got = x[sel], w[sel], c["shift"][sel]
            assert (got == got[0]).all()
            obj = lambda v: (ws * (xs - v).abs()).sum()  # noqa: E731
            best = min(obj(v) for v in xs)
            assert obj(got[0]) <= best * (1 + 1e-12) + 1e-15
            o = torch.argsort(xs)
            cw = ws[o].cumsum(0)
            assert got[0] == xs[o][torch.searchsorted(cw, 0.5 * cw[-1])]
            r = r + obj(got[0])
        total += r.item()
        # the partition: cells of the rotated, shifted grid x log-depth bins
        lab, alpha = c["lab"], c["alpha"]
        cell = 256.0 / alpha
        gen = torch.Generator().manual_seed(alpha)
        for b in range(2):
            ii = torch.randint(0, H * W, (300,), generator=gen)
            keys = []
            for j in ii.tolist():
                row, col = divmod(j, W)
                xx, yy = col - (W - 1) / 2, row - (H - 1) / 2
                ph, (o1, o2) = c["phi"][b].item(), c["offset"][b].tolist()
                u1 = (xx * math.cos(ph) + yy * math.sin(ph) + o1) / cell
                u2 = (-xx * math.sin(ph) + yy * math.cos(ph) + o2) / cell
                rel = min(
                    max(c["z"][b, row, col].item() / c["h"][b].item(), 2.0**-40),
                    2.0**40,
                )
                ib = math.floor(alpha * math.log2(rel) / 4 + c["beta"][b].item())
                keys.append((math.floor(u1), math.floor(u2), ib))
            labs = lab[b].flatten()[ii].tolist()
            for p_ in range(0, 300, 7):
                for q_ in range(300):
                    assert (labs[p_] == labs[q_]) == (keys[p_] == keys[q_])
        assert not set(lab[0].unique().tolist()) & set(lab[1].unique().tolist())
    # local_loss = (1 / n_F) sum over images of (sum over levels) / |K|
    nk = [int(mask[0].sum()), math.ceil(round(0.95 * int(mask[1].sum()), 6))]
    ref = 0.0
    for c in calls:  # the labels are distinct across images

        lab0 = set(c["lab"][0].unique().tolist())
        is0 = torch.tensor([s in lab0 for s in c["seg"].tolist()])
        e = (c["x"] - c["shift"]).abs() * c["w"]
        ref += e[is0].sum().item() / nk[0] + e[~is0].sum().item() / nk[1]
    assert rel_close(out["local_loss"], ref / 2, 1e-6)
    assert total > 0


def test_t12_local_term_ignores_constant_per_segment(monkeypatch):
    pred, target, mask, names = batch("A", hw=96, seed=13)
    criterion = crit(local_levels=dict(A=[16]), grad_types=(), local_pred_ratio=0.0)
    labs, fits = [], []
    segments, refit = D.local_segments, D.refit_in_graph

    def spy_segments(*a):
        lab = segments(*a)
        labs.append(lab)
        return lab

    def fixed_refit(*a):
        if not fits:
            s, t, ok = refit(*a)
            fits.append((s.detach(), t.detach(), ok))
        return fits[0]

    monkeypatch.setattr(D, "local_segments", spy_segments)
    monkeypatch.setattr(D, "refit_in_graph", fixed_refit)
    base = run(criterion, pred, target, mask, names, 13)
    s = fits[0][0][0].item()
    lab = labs[0][0]
    uniq, inv = lab.unique(return_inverse=True)
    gen = torch.Generator().manual_seed(13)
    c = (torch.rand(uniq.numel(), generator=gen, dtype=torch.float64) - 0.5) * 0.4
    moved = (pred.double() + (c[inv] / s)[None]).float()  # s d + t + c_segment
    out = run(criterion, moved, target, mask, names, 13)
    assert torch.equal(labs[1], labs[0])
    assert rel_close(out["local_loss"], base["local_loss"], 1e-6)
    assert not rel_close(out["depth_loss"], base["depth_loss"], 1e-3)


# ------------------------------------------------------------------ gradient term
def _ref_pyramid(R, G, M, cap, levels):
    """Explicit loops: per level the sum over the 3x3-eroded mask of min(1 / G, cap)
    (|Scharr_x R| + |Scharr_y R|) / 32, and the mask's count; blur 5x5 Gaussian
    (sigma 1, zero padding) + 2x2 mean between levels, next mask AND_2x2(erode_5x5)."""
    kx = [[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]]
    gk = [math.exp(-(i * i) / 2.0) for i in range(-2, 3)]
    gk = [v / (sum(gk) + 1e-12) for v in gk]
    R = [[R[i][j] if M[i][j] else 0.0 for j in range(len(R[0]))] for i in range(len(R))]
    G = [[G[i][j] if M[i][j] else 0.0 for j in range(len(G[0]))] for i in range(len(G))]
    m = [[bool(M[i][j]) for j in range(len(M[0]))] for i in range(len(M))]

    def full(m, i, j, r):
        H, W = len(m), len(m[0])
        return all(
            0 <= i + a < H and 0 <= j + b < W and m[i + a][j + b]
            for a in range(-r, r + 1)
            for b in range(-r, r + 1)
        )

    def blur_down(X):
        H, W = len(X), len(X[0])
        at = lambda i, j: X[i][j] if 0 <= i < H and 0 <= j < W else 0.0  # noqa: E731
        Y = [
            [sum(gk[b + 2] * at(i, j + b) for b in range(-2, 3)) for j in range(W)]
            for i in range(H)
        ]
        aty = lambda i, j: Y[i][j] if 0 <= i < H else 0.0  # noqa: E731
        Z = [
            [sum(gk[a + 2] * aty(i + a, j) for a in range(-2, 3)) for j in range(W)]
            for i in range(H)
        ]
        return [
            [
                (
                    Z[2 * i][2 * j]
                    + Z[2 * i][2 * j + 1]
                    + Z[2 * i + 1][2 * j]
                    + Z[2 * i + 1][2 * j + 1]
                )
                / 4
                for j in range(W // 2)
            ]
            for i in range(H // 2)
        ]

    sums, counts = [], []
    for _ in range(levels):
        H, W = len(R), len(R[0])
        s, n = 0.0, 0
        for i in range(H):
            for j in range(W):
                if not full(m, i, j, 1):
                    continue
                gx = (
                    sum(
                        kx[a][b] * R[i + a - 1][j + b - 1]
                        for a in range(3)
                        for b in range(3)
                    )
                    / 32
                )
                gy = (
                    sum(
                        kx[b][a] * R[i + a - 1][j + b - 1]
                        for a in range(3)
                        for b in range(3)
                    )
                    / 32
                )
                s += min(1.0 / G[i][j], cap) * (abs(gx) + abs(gy))
                n += 1
        sums.append(s)
        counts.append(n)
        m5 = [[full(m, i, j, 2) for j in range(W)] for i in range(H)]
        m = [
            [
                m5[2 * i][2 * j]
                and m5[2 * i][2 * j + 1]
                and m5[2 * i + 1][2 * j]
                and m5[2 * i + 1][2 * j + 1]
                for j in range(W // 2)
            ]
            for i in range(H // 2)
        ]
        R, G = blur_down(R), blur_down(G)
    return sums, counts


def test_t13_weighted_gradient_term():
    gen = torch.Generator().manual_seed(14)
    B, H = 2, 24
    target = 1.0 + 9.0 * torch.rand(B, H, H, generator=gen, dtype=torch.float64)
    target = F.avg_pool2d(target[:, None], 5, 1, 2)[:, 0]
    residual = 0.3 * torch.randn(B, H, H, generator=gen, dtype=torch.float64)
    mask = torch.rand(B, H, H, generator=gen) > 0.08
    cap = torch.tensor([0.9, 0.4], dtype=torch.float64)
    sums, counts = D.weighted_gradient_pyramid(residual, target, mask, cap, 3)
    for b in range(B):
        rs, rc = _ref_pyramid(
            residual[b].tolist(), target[b].tolist(), mask[b].tolist(), cap[b].item(), 3
        )
        assert counts[b].tolist() == rc
        for k in range(3):
            assert rel_close(sums[b, k], rs[k], 1e-5, 1e-12), (b, k)
    # target x c with the cap / c and the residual x c: unchanged
    c = 37.0
    s2, c2 = D.weighted_gradient_pyramid(c * residual, c * target, mask, cap / c, 3)
    assert torch.equal(c2, counts) and torch.allclose(s2, sums, rtol=1e-5, atol=0)
    # values at invalid pixels never enter
    junk = 1e3 * torch.rand(B, H, H, generator=gen, dtype=torch.float64)
    s3, c3 = D.weighted_gradient_pyramid(
        torch.where(mask, residual, junk),
        torch.where(mask, target, -junk),
        mask,
        cap,
        3,
    )
    assert torch.equal(s3, sums) and torch.equal(c3, counts)


# ------------------------------------------------------------------ degenerate cases, target scale, keys
def test_t14_degenerate():
    pred, target, mask, names = batch("AB", hw=96, seed=15)
    criterion = crit(log_logit_stats=True)

    def check_dropped(p, m):
        p = p.clone().requires_grad_(True)
        out = run(criterion, p, target, m, names, 15)
        assert list(out) == KEYS + LOGIT_KEYS
        assert all(v.shape == () and v.dtype == torch.float32 for v in out.values())
        assert out["loss"] == 0.0 and out["drop_frac"] == 1.0
        out["loss"].backward()
        assert p.grad is not None and (p.grad == 0).all()

    check_dropped(torch.full_like(pred, 2.0), mask)  # constant prediction
    check_dropped((1.0 / target).float(), mask)  # decreasing in depth: s <= 0
    few = torch.zeros_like(mask)
    few[:, :20, :20] = True  # 400 < min_valid
    check_dropped(pred, few)
    # one image under min_valid: dropped, the other fitted
    m = mask.clone()
    m[1] = few[1]
    out = run(criterion, pred, target, m, names, 15)
    assert out["drop_frac"] == 0.5 and out["loss"] > 0


def test_t15_target_scale():
    pred, target, mask, names = batch("ABC", hw=96, seed=16)
    criterion = crit(log_logit_stats=True)
    base = run(criterion, pred, target, mask, names, 16)
    out = run(criterion, pred, 37.0 * target, mask, names, 16)
    for k in base:
        assert rel_close(out[k], base[k], 1e-5, 1e-7), (k, out[k], base[k])


def test_t16_keys():
    pred, target, mask, names = batch("AC", hw=96, seed=17)
    for stats in (False, True):
        criterion = crit(log_logit_stats=stats)
        want = KEYS + (LOGIT_KEYS if stats else [])
        out = run(criterion, pred.clone().requires_grad_(True), target, mask, names, 17)
        assert list(out) == want
        assert out["loss"].requires_grad
        for k, v in out.items():
            assert v.shape == () and v.dtype == torch.float32, k
            assert k == "loss" or not v.requires_grad, k
        out = run(criterion, torch.ones_like(pred), target, mask, names, 17)  # no loss
        assert list(out) == want
        assert all(v.shape == () for v in out.values())
        assert all(v == 0 for k, v in out.items() if k != "drop_frac")
        assert list(criterion.state_dict()) == []
    assert criterion.uses_names and not RelativeDepthLoss().uses_names
    # head="exp": the logit is log d
    out = run(crit(log_logit_stats=True), pred, target, mask, names, 17)
    z = pred.log()
    assert rel_close(out["logit_absmax"], z[mask].abs().max(), 1e-6)


# ------------------------------------------------------------------ options
def test_t17_male_weight():
    """male_weight / male_num_scales are accepted (the release configs state them);
    the loss has no Laplacian term, so male_weight must be 0."""
    assert crit(male_weight=0.0, male_num_scales=4).align == "roe"
    with pytest.raises(ValueError, match="male_weight"):
        crit(male_weight=0.1)


@pytest.mark.parametrize(
    "kw",
    [
        dict(align="ransac"),
        dict(head="linear"),
        dict(label_types=None),
        dict(label_types=["hypersim"]),
        dict(label_types=dict(A="hypersim")),
        dict(label_types=dict(A=["hypersim"], B=["hypersim"])),
        dict(
            label_types=dict(A=["hypersim/x"]),
            type_trim=None,
            local_levels=None,
            grad_types=(),
        ),
        dict(type_trim=dict(A=0.0, B=0.05)),
        dict(type_trim=dict(A=0.0, B=0.5, C=0.05)),
        dict(type_trim=dict(A=-0.1, B=0.05, C=0.05)),
        dict(local_levels=dict(A=[0])),
        dict(local_levels=dict(A=[2.5])),
        dict(local_levels=dict(D=[4])),
        dict(local_levels=dict(A=4)),
        dict(grad_types="A"),
        dict(grad_types=["D"]),
        dict(roe_points=8),
        dict(roe_points=100.5),
        dict(roe_trunc=0.0),
        dict(weight_cap=1.0),
        dict(refit_floor=0.0),
        dict(local_weight=-1.0),
        dict(local_cell=0.0),
        dict(local_pred_ratio=1.5),
        dict(aux_scale=-1.0),
    ],
)
def test_t17_invalid_values(kw):
    with pytest.raises(ValueError):
        crit(**kw)


def test_t17_valid_presets():
    c = crit(type_trim=None, local_levels=None, grad_types=())
    assert c.uses_names and c.align == "roe"
    c = crit(
        label_types=dict(A=["hypersim"], B=["blendedmvs"], C=[]),
        type_trim=dict(A=0.0, B=0.05, C=0.05),
    )
    assert c.label_types["C"] == []
    d = RelativeDepthLoss()
    assert d.align == "roe" and d.head == "sigmoid" and not d.uses_names


def test_t17_without_label_types():
    """label_types=None: every image has one type and needs no name; the loss is
    the one of a single label type without trim, local or gradient term (bit for
    bit); options that name a label type raise."""
    pred, target, mask, names = batch("ABC", hw=96, seed=21)
    plain = RelativeDepthLoss(head="exp", min_valid=1000, roe_points=512)
    one_type = crit(
        label_types=dict(A=["hypersim", "blendedmvs", "kitti"]),
        type_trim=None,
        local_levels=None,
        grad_types=(),
    )
    out, g = grad_of(plain, pred, target, mask, None, 21)
    ref, g_ref = grad_of(one_type, pred, target, mask, names, 21)
    assert list(out) == KEYS and out["depth_loss"] > 0
    for k in KEYS:
        assert torch.equal(out[k], ref[k]), k
    assert torch.equal(g, g_ref)
    for kw in (
        dict(type_trim=dict(A=0.05)),
        dict(local_levels=dict(A=[4])),
        dict(grad_types=["A"]),
    ):
        with pytest.raises(ValueError):
            RelativeDepthLoss(**kw)


def test_t17_under_autocast():
    """A bf16 prediction under CPU autocast: the loss runs in fp32, as for the same
    values in fp32, with a finite gradient."""
    pred, target, mask, names = batch("AB", hw=96, seed=22)
    criterion = crit(log_logit_stats=True)
    ref = run(criterion, pred.bfloat16().float(), target, mask, names, 22)
    p = pred.bfloat16().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = run(criterion, p, target, mask, names, 22)
    out["loss"].backward()
    assert torch.isfinite(p.grad.float()).all() and p.grad.abs().sum() > 0
    assert list(out) == list(ref)
    for k, v in ref.items():
        assert out[k].dtype == torch.float32 and torch.equal(out[k], v), k


# ------------------------------------------------------------------ aux_scale = 0, exact cases
def test_t18_aux_scale_zero():
    pred, target, mask, names = batch("AB", hw=96, seed=18)
    out0, g0 = grad_of(crit(aux_scale=0.0), pred, target, mask, names, 18)
    out1, g1 = grad_of(
        crit(local_levels=None, grad_types=()), pred, target, mask, names, 18
    )
    assert out0["loss"] == out0["depth_loss"]
    assert torch.equal(g0, g1)
    assert out0["local_loss"] > 0 and out0["mage_loss"] > 0
    assert out1["local_loss"] == 0 and out1["mage_loss"] == 0
    full = run(crit(), pred, target, mask, names, 18)
    for k in ("depth_loss", "local_loss", "mage_loss"):
        assert out0[k] == full[k]
    assert full["loss"] > full["depth_loss"]


def test_t19_exact_cases():
    _, target, mask, names = batch("A", hw=96, seed=19)
    criterion = crit()
    out = run(criterion, target.clone(), target, mask, names, 19)
    assert out["depth_loss"] < 1e-6 and rel_close(out["scale_median"], 1.0, 1e-4)
    g = target[mask].double()
    h = g.numel() / (1.0 / g).sum().item()
    out = run(criterion, target + 5.0, target, mask, names, 19)
    assert out["depth_loss"] < 1e-6
    assert rel_close(out["scale_median"], 1.0 + 5.0 / h, 1e-4)


# ------------------------------------------------------------------ evaluator fit (8.1)
def test_roe_eval_scale_and_shift():
    gen = torch.Generator().manual_seed(20)
    g = 1.0 + 9.0 * torch.rand(3, 8, 8, generator=gen)
    mask = torch.ones(3, 8, 8, dtype=torch.bool)
    mask[2] = False
    mask[2, 0, :5] = True  # 5 < min_valid
    pred = (g - 0.5) / 2.0  # exact affine map of the target
    pred[1] = 3.0  # constant: rejected
    s, t, valid, rejected = roe_eval_scale_and_shift(
        pred, g, mask, points=64, return_rejected=True
    )
    assert valid.tolist() == [True, True, False] and rejected.tolist() == [
        False,
        True,
        False,
    ]
    assert rel_close(s[0], 2.0, 1e-5) and rel_close(t[0], 0.5, 1e-5, 1e-6)
    assert s[1] == 0.0
    v = g[1].flatten().double()
    best = min(((1.0 / v) * (v - c).abs()).sum().item() for c in v.tolist())
    assert rel_close(((1.0 / v) * (v - t[1].double()).abs()).sum(), best, 1e-6)
    assert torch.isnan(s[2]) and torch.isnan(t[2])
    # 64 pixels, 64 points: the exact 1/g-weighted L1 fit (brute force)
    p = pred[0].flatten().double() + 0.3 * torch.randn(
        64, generator=gen, dtype=torch.float64
    )
    s, t, _ = roe_eval_scale_and_shift(
        p.float().view(1, 8, 8), g[:1], mask[:1], points=64
    )
    gg = g[0].flatten().double()
    got = objective(s[0].double(), t[0].double(), p.float().double(), gg, 1.0 / gg)
    assert got <= brute_force(p.float().double(), gg, 1.0 / gg) * (1 + 1e-6)
