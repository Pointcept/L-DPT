# Evaluation

How the numbers of the README's Results table are measured, and the command that
reproduces them. The evaluation sets live under `data/<name>` like the training sets
([data.md](data.md)); the DINOv3 file under `checkpoints/dinov3/` ([install.md](install.md)).

All four sets are scored under one protocol: the images, masks and ranges of
Marigold's public evaluation code ([prs-eth/Marigold](https://github.com/prs-eth/Marigold),
`script/depth/eval.py`), and a seeded RANSAC scale and shift per image (InfiniDepth's
protocol with scikit-learn's `RANSACRegressor`, made reproducible). No image is dropped.

## Sets and masks

| Set | Split | Images | Scored ground truth | Range (m) |
|---|---|---|---|---|
| NYUv2 | `data/nyuv2/splits/test.json` (the BTS test split, raw depth; the RGB as BTS writes it, a JPEG black outside rows 7:474, cols 7:632) | 654 | rows 45:471, cols 41:601 of the 480 x 640 frame (the standard Eigen crop) | 1e-3 < d < 10 |
| KITTI | `data/kitti/splits/test.json` (the Eigen test images with ground truth, BTS lists) | 652 | the 352 x 1216 benchmark crop (image and ground truth), then its rows 117:321, cols 43:1172 (the Eigen crop) | 1e-5 < d < 80 |
| ETH3D | `data/eth3d/splits/train.json` (the split with depth) | 454 | the whole frame | d > 1e-5 |
| DIODE | `data/diode/splits/val_indoors.json`, `val_outdoor.json` | 325 + 446 | the whole frame, where DIODE's depth mask is set | 0.6 - 350 (prediction only) |

The crops only mask the ground truth: the image itself is never cropped, except for
KITTI's benchmark crop. Ground truth outside the range is not scored, except on DIODE,
where the depth mask alone decides (Marigold's DIODE loader) and the range only clips the
prediction.

## Protocol

1. Resize the short side of the image to 768 (bicubic, antialiased) and normalise it
   with the ImageNet mean and standard deviation. KITTI enters at 768 x 2653.
2. Predict, average with the prediction of the mirrored image, and resize the result
   bilinearly to the ground-truth size.
3. Per image, fit a scale s and a shift t to the ground truth in depth by RANSAC over
   the scored pixels:
   - a random generator seeded with 0 draws 1000 pairs of scored pixels; the line
     g = a p + b through each pair (equal pixels or equal predictions are skipped) is
     a trial;
   - a pixel is an inlier of a trial when |a p + b - g| <= median |g - median g|;
   - the trial with the most inliers wins (ties: the higher R² of its line on its
     inliers, then the later trial); s and t are the least-squares fit on its inliers;
   - when no trial exists (a constant prediction) or that fit is degenerate or has
     s <= 0, s = mean g / mean p and t = 0.

   Clip the aligned depth to the set's range.
4. AbsRel and δ1 (the fraction of pixels with max(pred / gt, gt / pred) < 1.25) per image,
   then averaged over the images of a set. DIODE counts as one set of 771 images; its
   indoor and outdoor halves are logged as `DIODE-INDOOR` and `DIODE-OUTDOOR`.

The decoder's position encoding normalises token coordinates by the short side of the
image (`model.dec_pos_embed_rope_normalize_coords="min"`, set by the depth configs, the
presets and `api.load`; the class default is `"separate"`). The attention runs in
PyTorch's default kernel order (FlashAttention first), in `scripts/evaluate.py` and in
the evaluation during training, while `api.predict` tries cuDNN attention first on
GPUs of compute capability 10.x such as the GB200 ([inference.md](inference.md)); the
kernels compute the same attention, and at 768 the depth of the two orders differs at
the rounding level of bf16.

In the configs: `depth_range=(min, max)` keeps the ground truth with min < d < max and
clips the aligned prediction to [min, max] (max None: no upper bound); DIODE's
`mask_only=True` scores every masked pixel; `hooks.4.align_solver="ransac"` with
`hooks.4.ransac_seed=0` is the fit (`ransac_scale_and_shift` in
`ldpt/models/utils/depth.py`). Only the evaluation follows this protocol: the mixed
model's training splits have their own ground-truth masks (NYUv2 rows 159:438, cols
23:616; KITTI the benchmark and Garg crops; DIODE indoor the Eigen crop and 10 m,
outdoor the Garg crop and 80 m).

DIODE split by scene type (AbsRel / δ1, %):

| Model | DIODE indoor | DIODE outdoor |
|---|---|---|
| L-DPT Large (zero-shot) | 4.73 / 95.98 | 33.66 / 78.27 |
| L-DPT Large (mixed) | 4.45 / 96.48 | 21.70 / 81.19 |

The mean-ratio fit of step 3 scores 2 of the 446 DIODE outdoor images of the zero-shot
model and 1 of the 325 DIODE indoor images of the mixed model; every other image has
its RANSAC fit (`num_rejected` in the JSON).

## Reproduce the table

The training configs carry the evaluation. On 8 GPUs, for the mixed model:

```bash
# the released weights file, downloaded once into the Hugging Face cache (sha256 checked)
CKPT=$(python -c "from ldpt.models.l_dpt.weights import fetch; print(fetch('l-dpt_large_depth_mixed'))")
python scripts/evaluate.py \
    --config-file configs/hybrid/depth-l-dpt-v1m1-1-large-mixed.py \
    --checkpoints $CKPT --num-gpus 8 \
    --output eval_large_depth_mixed.json
```

For the zero-shot model, fetch `l-dpt_large_depth_zeroshot` and use
`configs/hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py`; the evaluation part of both
configs is the same. The JSON holds every metric of every set, per checkpoint, and for
each set the protocol it was scored with (`evaluator.protocol`: alignment, crop, range,
TTA).

- `--checkpoints` also takes training checkpoints (`exp/<folder>/<run>/model/epoch_<N>.pth`,
  `model_best.pth`), several at once.
- `--average` evaluates the uniform average of the given checkpoints' weights.
- `--per-image` adds every image's metrics.
- `--encoder <file>` reads DINOv3 from another path than the config's `checkpoint`.
- `--options` overrides any config key.
