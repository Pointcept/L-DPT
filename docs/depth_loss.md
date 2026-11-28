# The L-DPT depth loss

The specification of the loss that trains the released depth models, for someone who
reimplements it or checks an implementation against it. In the code it is
`relative_depth_loss` (`ldpt/models/losses/depth.py`), configured in
`configs/hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py`,
`depth-l-dpt-v1m1-1-large-mixed.py` and `depth-l-dpt-v1m1-1a-large-mixed-anneal.py`.
Without options it is the global term alone, with every image of one label type; the
options below add the rest.

Sections 1 and 2 define the objective. Section 4 gives the algorithms an implementation
must match, section 5 the warm-up, section 7 the numeric precision and section 8 the
released values and the tests. Section 3 gives the reason for each choice; section 6
lists the logged keys.

## 0. Overview

The loss carries the objective of MoGe over to a model that predicts depth only:

1. an exponential depth head, computed in fp32: `d = exp(u)`;
2. a global term: the mean relative error of the aligned depth. The scale and shift come
   from MoGe's exact ROE solver (a truncated, 1/z-weighted L1 fit) on 2304 sampled
   pixels, followed by one weighted least-squares refit inside autograd;
3. local terms: the same error after a shift per region (a weighted median; the global
   scale is shared). A region is an image cell combined with a log-depth bin, at up to
   three cell sizes;
4. a multi-scale gradient-matching term, on synthetic labels only;
5. routing by label type: A (synthetic renders), B (renders of multi-view meshes) and C
   (sensor depth). Labels of types B and C are trimmed by 5 %, and training depth is
   resampled without staircases (section 4.6).

A *rank* below is one GPU's process with its share of the batch. The option and key
names `mage_weight` and `mage_loss` belong to the gradient term.

## 1. Setting and routing

Notation, per image: `u` the readout logit, `d = exp(u)` the prediction (fp32), `g` the
ground truth in metres, `M = {g > 0}` the valid pixels, `N = |M|`,
`h = N / sum_M (1/g)` the harmonic-mean depth. The label type `T` comes from the
sample-name prefix (`name.split("/")[0]`, exact) through `label_types`; a prefix in no
list raises at the first batch.

| Set | Prefix | Labels | Type | Local levels | Gradient term | Trim |
|---|---|---|---|---|---|---|
| Hypersim | `hypersim` | rendered | A | 4, 16, 64 | yes | 0 |
| VKITTI2 | `VKITTI2` | rendered | A | 4, 16, 64 | yes | 0 |
| Structured3D | `structure3d` | rendered | A | 4, 16, 64 | yes | 0 |
| TartanAir | `tartanair` | rendered | A | 4, 16, 64 | yes | 0 |
| IRS | `irs` | rendered | A | 4, 16, 64 | yes | 0 |
| BlendedMVS / + / ++ | `blendedmvs`, `blendedmvs+`, `blendedmvs++` | rendered from MVS meshes | B | 4, 16 | no | 5 % |
| NYUv2 (mixed only) | `nyuv2` | Kinect | C | 4 | no | 5 % |
| KITTI (mixed only) | `kitti` | LiDAR | C | 4 | no | 5 % |
| DIODE indoor / outdoor (mixed only) | `DIODE-INDOOR`, `DIODE-OUTDOOR` | laser scan | C | 4 | no | 5 % |

Section 3.4 explains the levels per type, section 3.5 the gradient term on synthetic
labels only.

## 2. The objective

Per image, in this order; `c = 10` (weight cap, `weight_cap`), `tau = 1` (truncation,
`roe_trunc`), `delta = 0.01` (refit floor, `refit_floor`), `n = 2304` (`roe_points`).

```
0  skip      N < 1000 (min_valid): the image is dropped (no loss, counted in drop_frac)

1  weights   w_j = min(1 / g_j, c / h)  on M,  0 elsewhere

2  ROE fit   S = n valid pixels: the floor((k + xi_k) N / n)-th valid pixel in row-major
   (no grad) order, k = 0..n-1, xi_k ~ U[0, 1) drawn every step
             (s0, t0) = argmin_{s, t} sum_{j in S} min(tau, w_j |s d_j + t - g_j|)   exact (4.1)
             dropped if d is constant on S, the result is not finite, or s0 <= 0

3  trim      e_j = w_j |s0 d_j + t0 - g_j|,  j in M         (relative error under the ROE fit)
   (no grad) K = the ceil((1 - rho_T) N) pixels of M with the smallest e
             rho_A = 0 (K = M),  rho_B = rho_C = 0.05                        (type_trim)

4  refit     omega_j = w_j^2 / max(e_j, delta)  for j in K with e_j <= tau,  else 0  (constants)
   (in graph, pbar = sum omega d / sum omega,   gbar = sum omega g / sum omega
    fp64)    s = sum omega (d - pbar)(g - gbar) / sum omega (d - pbar)^2,   t = gbar - s pbar
             dropped if sum omega = 0, the denominator is 0, s is not finite, or s <= 0

5  aligned   a_j = s d_j + t

6  global    l_g = (1 / |K|) sum_{j in K} w_j |a_j - g_j|

7  local     for alpha in levels(T):                                          (local_levels)
               z = max(a.detach(), 1e-3 h)  with probability 0.5 (local_pred_ratio),  else  g
               segment(j) = (cell of pixel j in a square grid of local_cell / alpha px,
                             rotated by phi ~ U[0, pi/2), shifted by o ~ U[0, cell)^2;
                             floor(alpha log2(z_j / h) / 4 + beta),  beta ~ U[0, 1))
               t_G = weighted median over j in G and K of (g_j - s d_j), weights w_j
               l_alpha = (1 / |K|) sum_{j in K} w_j |s d_j + t_G(j) - g_j|

8  gradient  (types in grad_types) R_0 = a - g,  G_0 = g,  M_0 = M;  for k = 0..3:
               E_k = erode_3x3(M_k);   W_k = min(1 / G_k, c / h)
               m_k = (1 / |E_k|) sum_{E_k} W_k (|Scharr_x R_k| + |Scharr_y R_k|)   (Scharr / 32)
               R_k+1 = down2(R_k),  G_k+1 = down2(G_k),  M_k+1 = AND_2x2(erode_5x5(M_k))
             l_gm = (1 / 4) sum_k m_k;   the image takes part only if |E_0| >= 1000

9  batch     F = the rank's images that were not dropped,  n_F = |F|
             depth_loss = (1 / n_F) sum_F l_g
             local_loss = (1 / n_F) sum_F sum_{alpha in levels(T)} l_alpha
             mage_loss  = (1 / n_F) sum_{F, gradient types, taking part} l_gm
             loss = depth_loss + aux_scale * (local_weight * local_loss + mage_weight * mage_loss)
             aux_scale = 0: steps 7 and 8 run under no_grad (logged, no graph)
             n_F = 0: loss = pred.mean() * 0 (connected to the prediction)
```

The release recipes train at short side 768 with `local_cell=768` (a nominal focal length
in pixels, scaled with the resize: cells of 192 / 48 / 12 px at alpha 4 / 16 / 64),
`local_weight=1`, `mage_weight=4` (1 per pyramid level), `aux_scale=1` after the warm-up.

Properties (each has a unit test in `tests/test_depth_loss_roe.py`):
- **Units.** Every term is a mean relative error; `depth_loss` is the AbsRel of the kept
  training pixels under the L1-optimal fit. Multiplying `g` by a constant changes nothing.
- **Invariance.** For fixed discrete sets (subsample, kept set, truncated set, segments,
  median pixels) the loss and its gradient are exactly invariant to `d -> a d + b`: the
  refit is a weighted least-squares fit with constant weights.
- **Gradient shape.** With `s d + t = g - r` and small `r`, `dL/du_j = +-(s d_j w_j) / |K|`,
  about `+-1 / |K|` for every pixel when the shift is small: no pixel class dominates, and
  there is no `p (1 - p)` factor that vanishes at saturation (as with a sigmoid head).

Differences from MoGe, each deliberate:

| MoGe | Here | Why |
|---|---|---|
| fit weights `1 / max(z, 1e-2)`, loss weights capped | one capped weight for both | the fit and the loss are then the same objective, which the refit's weights assume |
| subsample = masked nearest resize to 48 x 48 | 2304 pixels uniform over the valid set, redrawn every step | unbiased for the mean over valid pixels; one rule for dense and LiDAR masks |
| `(s, t)` in the graph through two support points | weighted least-squares refit (step 4) | section 3.3 |
| mean over all H x W pixels | mean over the kept valid pixels of each image | section 3.8 |
| local regions from 3-D points and rays | image cells x log-depth bins | no intrinsics in the training samples (section 3.4) |
| edge-angle term on the point map | 1/z-weighted gradient term on the aligned depth | section 3.5 |

## 3. Design choices

One paragraph per choice: what an implementation must keep, and why.

### 3.1 Output: exponential head, fp32 readout

`d = exp(clamp(u, -30, 30))`, with the readout convolution and the pixel shuffle outside
the bf16 autocast. In bf16 the logit would keep 8 significant bits, and the sign-based
terms of the loss see that rounding. The evaluator aligns `s * pred + t` to metric depth,
so the output is used like that of a sigmoid head. An affine-invariant loss leaves one
direction free (`u = log(a (g - t))` for any shift `t`); it is not regularised, and
`scale_median` (section 6) shows it.

### 3.2 Global alignment: ROE

MoGe's solver in its depth form, with the truncation at 1 in the fit only. With 1/z
weights the fit is equivariant to the scale of the target, so the target needs no median
or MAD normalisation. The 2304 points are MoGe's 48 x 48.

### 3.3 Gradient through the fit: weighted least-squares refit

A detached L1 fit leaves a gradient along the prediction's offset and stretch, and
MoGe's gradient through two support points puts the correction on those two pixels.
Step 4 is one IRLS step of the same objective, written as a weighted least-squares fit
inside autograd. It gives exact affine invariance and spreads the correction over the
kept pixels.

### 3.4 Local alignment without rays

One shift per region, with the global scale shared; every kept pixel belongs to one
region per level, and each level has weight 1. The levels per type follow MoGe's
training configurations (synthetic 4 / 16 / 64, multi-view 4 / 16, sensor 4). MoGe-3
groups 3-D points by ray direction and log distance. The training samples carry no
intrinsics, so the regions here are square image cells of `local_cell / alpha` pixels
(a nominal focal length) combined with log2 bins of a reference depth, with a random
rotation, offset and bin phase at every step. A wrong focal length changes the shape of
the cells, not what the term measures. The reference depth is the ground truth or the
detached aligned prediction, with probability 0.5 each.

### 3.5 Detail term: weighted gradient matching, synthetic only

The eroded multi-scale gradient pyramid (`weighted_gradient_pyramid`) on the aligned
residual:
4 levels, each weighted by `min(1 / G_k, c / h)`, total weight 4 (1 per level; MiDaS's
alpha = 0.5 in this loss's units). MoGe's edge-angle term and its normal terms need the
focal length or rays, which the training samples lack; the depth-only methods (Depth
Pro, Depth Anything V2, the Depth Anything 3 teacher, MiDaS) match image gradients
instead. The label gradients of LiDAR and laser scans are not reliable, so only
synthetic labels take part.

### 3.6 Training fit and evaluation fit

The benchmarks align by a RANSAC fit in depth, an unweighted least-squares fit on the
inliers of the best of 1000 two-point lines, which far pixels dominate; the loss fits by
a 1/z-weighted L1. The loss has no term for the evaluation fit. The gap is logged
instead: `lsq_loss` (the global term under an unweighted least-squares fit) in the
training logs; `hooks.4.align_protocols=[roe]` adds `<group>@roe` (MoGe's evaluation
fit, never counted in the score) to every evaluation.

### 3.7 Trim and truncation

The truncation at relative error 1 acts in the fit only: a pixel with an error above
100 % does not steer the fit and still receives a gradient, bounded by the weight cap.
The trim is 0 on synthetic labels and 5 % on multi-view and sensor labels (MoGe's
value), ranked by the weighted residual under the ROE fit. Trimmed pixels are left out
of the refit, the global term and the local terms.

### 3.8 Per-image weighting

Each fitted image contributes the mean over its own kept pixels, and the loss of a rank
is the mean over its fitted images, so a sparse LiDAR crop weighs as much as a dense
one. A term that an image does not have counts as 0 for that image.

### 3.9 Weights

| Term | Weight |
|---|---|
| global | 1 |
| local, each level | `local_weight` = 1 |
| gradient | `mage_weight` = 4 on the level mean (1 per level) |
| `aux_scale` | 0 -> 1 by the warm-up (section 5), then 1 |

Constants: weight cap 10, truncation 1, 2304 points, refit floor 0.01.

## 4. Algorithms

### 4.1 ROE solver

`ldpt/models/utils/depth.py`: a port of MoGe's `align` and the anchor loop of
`align_depth_affine` (MIT licence, Microsoft Corporation). Every sampled pixel is an
anchor (subtracting it removes the shift); per anchor the 1-D problem
`min_a sum_i min(tau, w_i |a x_i - y_i|)` is solved exactly from sorted arrays and prefix
sums, and the best anchor wins. Before the call, in fp64: `x = (d_S - median d_S) /
spread` with `spread = median |d_S - median d_S|` (the mean of `|d_S - median d_S|` when
that median is 0), `y = g_S / h`, `w' = w_S h`; the search runs in fp32 (in fp64 when a
conditioned `x` exceeds 1e4 in magnitude); then `s0 = sigma h / spread`,
`t0 = h theta - s0 median`. This changes nothing in exact
arithmetic and keeps the solver's guards meaningful for any offset of `d`. One image at
a time: O(n^2 log n) time and 1 GiB of working memory at n = 2304.

### 4.2 Subsample

`sample_valid(mask, n, jitter)`: the cumulative sum of the mask, ranks
`floor((k + xi) N / n) + 1`, `searchsorted`. Only valid pixels, each with the same
probability; `N < n` repeats pixels. Random numbers come from the global generator on
the prediction's device.

### 4.3 Trim and refit

The trim keeps exactly `ceil((1 - rho) N)` pixels by `e` (ties by pixel order). The refit
uses centred fp64 moments with the weights of step 4 and divides by no degenerate value,
not even in an unselected branch; dropped images get `s = 1, t = 0` and weight 0 in
every term.

### 4.4 Local term

Per level, for all the rank's images at once: integer segment labels from the rotated,
shifted cell grid and the log-depth bin (no grad); values `x_j = g_j - s d_j` (fp64, in
the graph) on the kept pixels; the weighted median per segment from one global rank of
the values and one integer sort per level (the first element whose cumulative weight
reaches half of the segment's weight); `l_alpha` per image by `index_add`. The result
equals a per-segment loop exactly, ties included.

### 4.5 Gradient term

`weighted_gradient_pyramid`: the residual and the target, both zeroed off the mask, go
down one Gaussian pyramid each (5 x 5 blur, 2 x 2 average); level k sums
`W_k (|Scharr_x R_k| + |Scharr_y R_k|)` (Scharr / 32), `W_k = min(1 / G_k, c / h)`, over
`E_k = erode_3x3(M_k)`, and `M_k+1 = AND_2x2(erode_5x5(M_k))`, so every value that counts
is made of valid pixels only (no invalid pixel and no zero padding in its kernels or
blurs).

### 4.6 Depth resampling in the train transform

`Resize(depth_resample="edge_aware")` (`ldpt/datasets/transforms.py`): nearest-neighbour
upsampling of a depth map leaves staircases (runs of equal neighbouring values), which
the gradient term would learn. MoGe's rule: `ok` = pixels whose 5 x 5 window is fully
valid and has no edge. With `q = 1 / depth` and `L = (q - mean_5x5 q) / q`, a window has
an edge when its maximum of `L` exceeds 0.01 and its minimum is below -0.01. The output
is `1 / bilinear(q)` where all four source neighbours are `ok`, else the nearest source
pixel (`nearest-exact`); invalid stays 0.
LiDAR maps fall back to nearest everywhere by themselves, so one transform list serves
every dataset. The validation transform is unchanged.

## 5. Warm-up

With the scale inside the graph, the local and gradient terms can be lowered by
shrinking `s` while the prediction is still weakly correlated with the target, which
works against the global term. So they start at 0:

```python
dict(type="CriterionWeightWarmup", key="aux_scale", gate_key="drop_frac",
     min_hold=1000, ramp=1000, gate_depth_loss=0.20, gate_clamp_frac=0.20, max_hold=3000)
```

`aux_scale` stays 0 for at least 1000 iterations and until the all-rank EMAs of
`depth_loss <= 0.20` and `drop_frac <= 0.20`, then rises linearly to 1 over 1000 gated
iterations (a warning after `max_hold`). While it is 0 the local and gradient terms are
computed under `no_grad` and logged. 0.20 is 0.75 of the `depth_loss` of a flat
prediction on the synthetic sets. The hook's state is saved in `model_last.pth`, so a
resumed run continues the ramp. The anneal recipe
(`depth-l-dpt-v1m1-1a-large-mixed-anneal.py`) starts from a trained decoder and has no
warm-up.

## 6. Logged keys

All 0-d fp32, detached except `loss`; the same keys every step (zeros when no image is
fitted):

| Key | Definition |
|---|---|
| `loss` | step 9 |
| `depth_loss` | global term: mean relative error of the kept pixels under the fit |
| `local_loss` | sum of the levels (before `aux_scale`) |
| `mage_loss` | gradient term (before its weight) |
| `drop_frac` | share of the rank's images dropped (under 1000 valid pixels, non-positive or degenerate fit) |
| `depth_loss_max` | largest per-image global term |
| `scale_median` | median gain `(1 / N) sum_M s d / g` (1 = no shift) |
| `spread_median` | median standard deviation of `log d` over the valid pixels |
| `saturation_frac` | share of valid pixels with `abs(log d) > 20` |
| `gm_image_frac` | share of images in the gradient term |
| `trim_thresh_median` | median trim threshold of `e` over the trimmed images |
| `trunc_frac` | share of valid pixels with `e > 1` |
| `refit_ratio_median` | median `s / s0` |
| `lsq_loss` | the global term under an unweighted least-squares fit with a non-negative scale (section 3.6) |
| `logit_mean`, `logit_std`, `logit_absmax` | with `log_logit_stats=True`: statistics of `u = log d` |

The trainer adds `grad_norm` (before clipping).

## 7. Numerics under the bf16 decoder

| Part | Precision |
|---|---|
| frozen encoder; decoder blocks | bf16 autocast |
| attention of the last decoder stage | fp32 (`attn_fp32="last_stage"`), every logit bounded by 1000 (`dec_qk_bound`) |
| readout convolution, pixel shuffle, `exp` | fp32 (`readout_fp32=True`) |
| loss | fp32 with autocast off; ROE search fp32 on normalised inputs; refit, aligned depth, local values and medians fp64 |

The fp32 attention and the logit bound are model options (both act in training and at
inference, without new parameters); they are part of the released architecture.

## 8. Configuration reference

The loss block of `configs/hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py`; the mixed
recipes differ only in `C=["nyuv2", "kitti", "DIODE-INDOOR", "DIODE-OUTDOOR"]`. On the
model, `head="exp"` must match the loss's `head`, with `readout_fp32=True`; the released
models also use `attn_fp32="last_stage"` and `dec_qk_bound=1000` (section 7).

```python
criterion=dict(
    type="relative_depth_loss",
    mage_weight=4.0,             # gradient term (1 per level)
    mage_num_scales=4,
    male_weight=0.0,             # no Laplacian term (must be 0)
    male_num_scales=4,           # not used
    align="roe",                 # the only value (the default)
    head="exp",                  # must match model.head
    min_valid=1000,              # step 0
    roe_points=2304,             # step 2
    roe_trunc=1.0,               # tau
    weight_cap=10.0,             # c
    refit_floor=0.01,            # delta
    label_types=dict(            # sample-name prefixes per label type (section 1)
        A=["hypersim", "VKITTI2", "structure3d", "tartanair", "irs"],
        B=["blendedmvs", "blendedmvs+", "blendedmvs++"],
        C=[],
    ),
    type_trim=dict(A=0.0, B=0.05, C=0.05),
    local_levels=dict(A=[4, 16, 64], B=[4, 16], C=[4]),
    local_weight=1.0,
    local_cell=768.0,            # nominal focal length in px at the training resolution
    local_pred_ratio=0.5,        # probability of the prediction as the bin reference
    grad_types=["A"],
    aux_scale=1.0,               # set by CriterionWeightWarmup during the warm-up
    log_logit_stats=True,
)
```

Tests: `tests/test_depth_loss_roe.py` (the objective, its invariances, routing, trim,
local medians, gradient term, keys), `tests/test_depth_resample.py` (section 4.6),
`tests/test_criterion_warmup.py` (section 5), `tests/test_structural_options.py` (the
exp head and the fp32 readout).

## References

- MoGe: Wang et al., *MoGe: Unlocking Accurate Monocular Geometry Estimation for
  Open-Domain Images with Optimal Training Supervision*, CVPR 2025;
  <https://github.com/microsoft/MoGe> (MIT).
- MoGe-2 / MoGe-3 training configurations and losses in the same repository.
- Depth Anything 3: Lin et al., 2025. Depth Pro: Bochkovskii et al., 2024. Depth
  Anything V2: Yang et al., 2024. MiDaS: Ranftl et al., TPAMI 2022.
