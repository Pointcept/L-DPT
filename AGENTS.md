# L-DPT: guide for contributors and coding agents

L-DPT: dense prediction (relative depth, semantic segmentation) from a frozen DINOv3
encoder with a latent DPT decoder. The training framework follows Pointcept: Python
configs, registries (`MODELS`, `DATASETS`, `HOOKS`, `LOSSES`, `OPTIMIZERS`,
`SCHEDULERS`, `TRAINERS`) and trainer hooks. User-facing documentation is
[`README.md`](README.md) (the front page) and `docs/`: installation, inference,
evaluation, training, data, and the specification of the L-DPT depth loss
([`docs/depth_loss.md`](docs/depth_loss.md)). Keep commands in them in step with the
code.

## Layout

```
train.py, train.sh             entry point; launcher (code snapshot, auto-resume, -o overrides;
                               under srun one train.sh per node, rendezvous from SLURM_*)
configs/_base_/default_runtime.py
configs/hybrid/                depth recipes: depth-l-dpt-v1m1-0-large-zeroshot.py,
                               depth-l-dpt-v1m1-1-large-mixed.py and its anneal
                               depth-l-dpt-v1m1-1a-large-mixed-anneal.py
configs/cocostuff/             segmentation recipe: semseg-l-dpt-v1m1-1-large.py and
                               its anneal semseg-l-dpt-v1m1-1a-large-anneal.py
configs/voc2012/               semseg-l-dpt-v1m1-1-large.py (from the COCO-Stuff
                               recipe-1 average)
ldpt/models/l_dpt/             LatentDPT (l_dpt_v1m1_base.py, registry "L-DPT-v1m1"),
                               modules.py (attention, block, readout), encoder_tiles.py
                               (the encoder on overlapping windows of a large input),
                               presets.py, weights.py (pinned files), convert.py (release
                               files), api.py
ldpt/models/dinov3/            DINOv3 backbones, inference copy (dinov3_vit7b; vitb / vitl
                               / vith; a tiny test variant lives in tests/helpers.py);
                               hf_format.py reads Meta's Hugging Face (transformers-format)
                               weights into the original key layout
ldpt/models/losses/            relative_depth_loss (depth.py), ce_loss, lovasz_loss
ldpt/models/utils/depth.py     least-squares / RANSAC / ROE alignment, shared by loss
                               and evaluator
ldpt/datasets/                 one module per dataset, transforms, ConcatDataset
ldpt/hooks/                    CheckpointLoader / Saver, CriterionWeightWarmup,
                               Depth / SemSegEvaluator, DepthVisualizer
ldpt/engines/                  config parser, launcher, DefaultTrainer, evaluate.py
ldpt/utils/depth_image.py      depth colours (tone curve per image, near blue, far red)
                               and the 16-bit PNG, shared by the scripts and the demo
ldpt/utils/seg_image.py        COCO-Stuff class names (train-id order; datasets/meta.py
                               imports them), class colours (PASCAL VOC map), top classes
ldpt/utils/sdpa.py             SDPA backend order (sdpa_context, SDPA_ORDER, auto_order),
                               shared by the trainer and api.predict
scripts/demo_depth.py          one image -> colour PNG + float32 .npy (native resolution)
scripts/demo_segment.py        one image -> class colour PNG + uint8 class ids .npy (768)
scripts/highres_infer.py       presets or release files x large photos at native resolution
scripts/evaluate.py            checkpoints / release files -> JSON table on a config's val sets
scripts/average_weights.py     weighted average of decoder checkpoints
scripts/datasets/              split JSONs (process_<name>_json.py), deduplicate.py
demo/hf_space/                 Gradio demo for a Hugging Face Space (ZeroGPU): app.py (UI),
                               space_utils.py (everything else, testable without gradio),
                               models.json (the release files, pinned by commit and
                               sha256); sync_ldpt.sh copies the inference subset of ldpt
                               into it
docs/                          user guides: install.md, inference.md, evaluation.md,
                               training.md, data.md; depth_loss.md (the L-DPT depth loss,
                               specification); assets/ (the README figures)
tests/                         pytest, CPU
```

## Model: `L-DPT-v1m1` (`LatentDPT`)

A frozen ViT encoder is tapped at one layer per decoder stage; a zero-initialised latent
grid (one token per patch) is decoded in stages: stage s adds a linear selection of the
tapped features and runs `dec_num_layers[s]` RoPE self-attention blocks; a k x k
convolution + pixel shuffle reads out to pixels. One class serves every variant; the
state-dict keys are those of the trained checkpoints, so they load strictly.

Behaviours the trained checkpoints depend on (keep them):
- the attention residual is added to the *normalised* tensor
  (`x = norm1(x); x = x + attn(x)`), not to the block input;
- in training the decoder RoPE coordinate augmentation (jitter / rescale) is re-sampled
  for every block;
- by default the frozen encoder follows `model.train()`, so its RoPE augmentation is on
  in training (`encoder_train_mode=True`);
- `tests/test_l_dpt_golden.py` pins outputs, losses and gradients bitwise (fp32 CPU,
  recorded on x86_64; skipped on other CPUs); regenerate it only for a deliberate change
  of the math, and say so in the commit.

Options besides the decoder sizes (`dec_dim`, `dec_num_heads`, `dec_num_layers`,
`dec_attn_downscale`, `dec_ffn_ratio`, RoPE base / normalisation / augmentation):
- `task`: `"relative_depth"` or `"semseg"` (`out_dim` features + 1x1 `seg_head`, needs
  `num_classes`).
- `criterion`: None (the default loss of the task), a loss cfg, or False: inference
  only (no loss built; an eval forward with a target skips the loss, a training forward
  raises; same state dict). `api.load` and release files use False.
- `head`: `"sigmoid"` (default) or `"exp"` (`exp(clamp(logit, -30, 30))`, the release
  models; the criterion's `head` must match). `readout_fp32`: the readout in fp32
  outside the autocast.
- `precision`: `"fp32"` (default) or `"bf16"` (selection, decoder blocks and readout
  under bf16 autocast; heads and loss in fp32). `attn_dtype`: `"fp32"` or `"bf16"`
  (legacy `attn_impl` `"auto"` / `"flash"`).
- `attn_fp32`: None, `"last"`, `"last_stage"` (the release models) or `"all"`: the
  attention of those blocks (projections, RoPE, SDPA, out_proj) in fp32 with autocast
  off, eager also under `compile_blocks` (an inlined island would get bf16 gradient
  matmuls). `block_fp32`: None, `"last"` or `"last_stage"`: whole blocks outside
  autocast. Both act in eval too.
- `dec_qk_bound`: None or B > 0 (the release models: 1000): q and k are clipped per
  head and token to an L2 norm of at most sqrt(B sqrt(head_dim)) before RoPE, so every
  attention logit is at most B; no parameters, in training and eval (a model trained
  with it needs it at inference). The strings "None" / "null" turn it off from
  `--options`.
- `encoder_weight_dtype`: None or `"bf16"`: frozen encoder Linear / Conv weights stored
  in bf16 (bit-identical under the encoder's bf16 autocast; half the memory of fp32).
- Inference memory (attributes, no effect on the weights): `selection_impl` `"concat"`
  (default) or `"stream"` (each tapped layer projected as the encoder yields it);
  `chunk_tokens` None or an int (encoder blocks and decoder MLPs on token chunks, eval
  without autograd only).
- Encoder windows (inference, eval mode; `encoder_tiles.py`):
  `infer(image, encoder_tiles=EncoderTiles(tile=192, overlap=64, rope_scale=2.0))`.
  When a side of the token grid exceeds `tile`, `encode_select_tiled` runs the frozen
  encoder and the selection on overlapping `tile` x `tile` windows (per axis of n
  tokens k = ceil((n - overlap) / (tile - overlap)) windows at
  round(i (n - tile) / (k - 1))), with the encoder RoPE coordinates times `rope_scale`
  inside a window (`encoder.rope_embed.eval_scale`, restored afterwards), sums the
  per-stage selections in fp32 on the whole grid with a separable linear ramp over
  `overlap` tokens at inner window edges, divides by the weight sum, and the decoder
  decodes the whole grid in one pass, unchanged. A smaller grid runs the plain
  forward, bit for bit (`tests/test_encoder_tiles.py`); `encoder_tiles=None` (the
  default of `infer`, and every training and evaluation forward) is the plain forward.
- `selection`: `"stage"` (one Linear per stage, default) or `"shared"`;
  `readout_kernel_size`: 3 (default, all trained models) or 1.
- Off by default and bitwise the model without them when off (tests in
  `tests/test_qk_norm.py`, `test_structural_options.py`, `test_seg_options.py`,
  `test_l_dpt_model.py`); `dec_qk_norm` and `tap_bn` are part of the COCO-Stuff
  recipe 1 and the segmentation preset, the depth presets use none of them:
  - `dec_qk_norm` (+ `dec_qk_tau_max`): cosine attention with a learned, capped
    temperature per head (one `attn.logit_scale` key per block; excludes
    `dec_qk_bound`); the COCO-Stuff recipe 1;
  - `tap_bn`: a BatchNorm per tapped encoder layer before the selection; the
    COCO-Stuff recipe 1;
  - experimental, in no recipe: `dec_conv_mixer` (None, `"last"` or `"last_stage"`:
    those blocks mix tokens with a 3x3 depthwise convolution, `LocalMix`, instead of
    attention); `dec_pos_embed_rope_eval_scale` (in eval, decoder RoPE coordinates
    multiplied by r).

RoPE coordinates: `api.load` (depth models) and the depth recipes use
`dec_pos_embed_rope_normalize_coords="min"` (by the short side); `"separate"` normalises
each axis to [-1, 1] (the segmentation recipes and preset: `api.load` keeps a
segmentation model's value). They are identical on square crops.

### Presets, release files, API

- `presets.py`: `"l-dpt_large_depth_zeroshot"` / `"l-dpt_large_depth_mixed"`
  (`l-dpt_<decoder size>_<task>_<training data>`; the weight files are `<name>.pth`) =
  one dict `RELEASE_L` (the depth recipes' model with `criterion=False`), and
  `"l-dpt_large_semseg_cocostuff"` = `SEMSEG_L` (COCO-Stuff, 171 classes: decoder
  2048 / (2, 2, 2, 2), attention at 1024, `dec_qk_norm`, `tap_bn`, `out_dim` 64, RoPE
  `"separate"`; `CLASS_NAMES` per segmentation preset); `INPUT_SIZE` 768, `WEIGHTS`
  slots (Hugging Face repo, revision, sha256: `pointcept-bot/L-DPT`; env
  `LDPT_CKPT_LARGE_DEPTH_ZEROSHOT` / `LDPT_CKPT_LARGE_DEPTH_MIXED` /
  `LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF` for a local copy; a slot of `TBD` values is not
  published: `WeightSource.published`). `LEGACY_PRESETS` maps the short names stored
  inside the published weights files ("l-zs", "l-mix") to these; `canonical_preset`
  reads them. `tests/test_configs.py` checks that the depth recipes' model is the
  preset.
- `convert.py`: training checkpoint -> release file (`l-dpt-release/2`: decoder-only
  fp32 state, the model kwargs without the criterion, preset name, provenance with the
  source sha256 and epoch; for a `scripts/average_weights.py` output also its inputs,
  from `<file>.avg.json`). `/1` files still load.
- `api.load(preset | path, encoder_weights, ...)` builds on the meta device and assigns
  the memory-mapped tensors (no fp32 copy of the encoder); `precision=None` keeps the
  file's / preset's value; a preset with `weights=<file>` must match the file on
  `api._ARCH_KEYS` (every option that changes the output), except the per-load options
  the caller passes (`api._LOAD_KEYS`); a training checkpoint is checked against
  `<run>/config.py` when it sits in `<run>/model/`; `model.class_names` from
  `CLASS_NAMES`. `api.predict(model, image, size, flip, sdpa_backends,
  encoder_tiles)`: validation preprocessing, depth at the input size, or for a
  segmentation model the int64 class map of `SemSegEvaluator` (logits resized
  bilinearly to the image, argmax; `flip`: the mean of the softmaxes; `api.class_map`
  resizes a few classes at a time on large images, same values); `size=None`
  (native resolution) runs with `api.NATIVE_MEMORY` (`selection_impl="stream"`,
  `chunk_tokens=32768`) for the call unless the caller chose them
  (`model.native_memory`, set by `api.load`); the resized path runs the model as loaded.
  Both run inside `sdpa_context(sdpa_backends, device)`, default `"auto"`
  (`sdpa.auto_order`): `api.SDPA_ORDER` (cuDNN attention first, the trainer's speed
  option) on compute capability 10.x (the fastest on a GB200), else PyTorch's order
  (cuDNN first is slower on an RTX PRO 6000, sm_120); `sdpa_backends=None` keeps
  PyTorch's order, which evaluation runs. `encoder_tiles="auto"` (default) runs a depth
  model's `model.infer` with `api.ENCODER_TILES` (`EncoderTiles(192, 64, 2.0)`): an
  input with a side above 3072 px runs the encoder in overlapping 3072 px windows and
  the decoder on the whole latent grid in one pass, a smaller one is one forward pass
  (the 768 px benchmarks are unchanged); a segmentation model gets None (one encoder
  pass: in the windows the classes of large regions fragment); None or False: one
  encoder pass at any size; or an `EncoderTiles`.
- `encoder_weights` (`weights.encoder_weights` / `encoder_state`): the official DINOv3
  `.pth` (sha256 prefix of its name checked) or a folder in the transformers format (a
  download of Meta's gated Hugging Face repo; `config.json` checked against the
  backbone). `ldpt/models/dinov3/hf_format.py` is the inverse of transformers'
  `convert_dinov3_vit_to_hf.py`: q / k / v joined into `qkv`, the dropped key bias and
  `bias_mask` rebuilt, `rope_embed.periods` recomputed (the 7B file's two GPU-rounded
  values pinned), so the state equals the original file's tensor for tensor
  (`tests/test_dinov3_hf_format.py`). Training reads the `.pth` only.

## Loss `relative_depth_loss` (`ldpt/models/losses/depth.py`)

- The L-DPT depth loss of the released depth models, specified in
  `docs/depth_loss.md` (tests: `tests/test_depth_loss_roe.py`): options `min_valid`,
  `roe_points`, `roe_trunc`, `weight_cap`, `refit_floor`, `label_types` (sample-name
  prefix -> label type; an unknown prefix raises; None, the default: every image of
  one type, no names needed), `type_trim`, `local_levels`, `local_weight`,
  `local_cell`, `local_pred_ratio`, `grad_types`, `mage_weight` / `mage_num_scales`
  (the gradient term), `aux_scale` (set by `CriterionWeightWarmup`). Without options
  it is the global term alone. `align` (`"roe"`, its only value) and `male_weight`
  (must be 0) / `male_num_scales` remain because the depth recipes state them.
- `log_logit_stats=True` adds detached logit statistics to the logged keys.

The model passes `names=data_dict["name"]` when the loss needs sample names
(`uses_names`); every loss dict entry is logged by `InformationWriter`.

## Configs

`configs/<folder>/<name>.py` inherit `configs/_base_/default_runtime.py` and are
flattened: each states every key of its recipe. Names follow Pointcept: the folder is
the training data (`hybrid`: the mixed depth sets, `cocostuff`, `voc2012`), the name
`<task>-<model>-v<X>m<Y>-<id>-<desc>` (`depth-l-dpt-v1m1-0-large-zeroshot`).
`train.sh -d <folder> -c <name> -n <run>` (`-d` defaults to `hybrid`, Pointcept's flag;
`-e` replaces the experiment root `exp`) writes to
`exp/<folder>/<run>/` (`config.py`, `code/`, `model/`, `metrics.jsonl`, `train.log`);
the run is named after its config, and another run of the same config (another seed)
adds a suffix (`-seed1`) instead of a new config. `--options` / `-o` override any key,
list items by index (`hooks.4.align_solver=lsq`); `batch_size` / `num_workers` are
totals over all GPUs.

- Depth hooks (indices are stable: overrides use them): 0 `CheckpointLoader`, 1
  `ModelHook`, 2 `IterationTimer`, 3 `InformationWriter`, 4 `DepthEvaluator`, 5
  `DepthVisualizer`, 6 `CheckpointSaver`, 7 `CriterionWeightWarmup` (absent from the
  anneal config). Segmentation hooks: 0 `CheckpointLoader`, 1 `ModelHook`, 2
  `IterationTimer`, 3 `InformationWriter`, 4 `SemSegEvaluator`, 5 `CheckpointSaver`.
- `checkpoint`: the DINOv3 file, loaded into `model.encoder` by
  `CheckpointLoader(keywords="module.", replacement="module.encoder.")`. A second stage
  loads `[DINOv3, <earlier stage>/model/<file>]` with `keywords=["module.", "module."]`
  and `replacement=["module.encoder.", "module."]` (VOC: the COCO-Stuff decoder without
  its head, `seg_head_rows` initialising the VOC head from it); the COCO-Stuff anneals
  and the VOC recipe add `allow_missing=False` (a model key in none of the files
  raises).
- Checkpoints are decoder-only (`CheckpointSaver(skip_keywords=["encoder"])`); a resume
  reloads the skipped encoder keys from the `checkpoint` of `<run>/config.py`.
  `keep_epoch_weights=True` writes `model/epoch_<N>.pth` per epoch;
  `save_freq_iter=N` writes `model_last.pth` every N iterations and resumes mid-epoch
  on the same batches (`ResumableSampler`).
- `DepthEvaluator(relative=True, tta=[["flip"]], align_solver="centered" |
  "lsq" | "lsq_free" | "ransac", ransac_seed=0, best_metric="abs_rel",
  best_metric_mean="geometric", align_protocols=..., align_max_res=None | {group: N})`:
  per-image alignment in depth space (the depth recipes: "ransac", the seeded depth
  RANSAC of `ransac_scale_and_shift` on every valid pixel, seed 0; "lsq_free":
  Marigold's unconstrained least squares; "lsq": scale >= 0; `align_max_res`: a group's
  fit on a nearest subsample of the columns), clip to the set's range; groups =
  sample-name prefixes (the combined DIODE group is logged; `score_groups` selects what
  counts); `align_protocols` (`trim5`, `disp`, `ransac`, `roe`) add logged-only groups
  `<group>@<protocol>` (`@ransac` = the "ransac" solver bit for bit).
  `SemSegEvaluator(tta=None | [["flip"], ["scale@r"], ...])`, `miou` over all
  classes. Every evaluation appends a line to `metrics.jsonl`;
  `model_best.pth` follows the score.
- Datasets: `NYUv2Dataset(eigen_crop="legacy" | True)` (KITTI's Eigen ratios on the
  NYUv2 frame, the crop of the mixed model's NYUv2 training labels / the standard
  Eigen crop), `KITTIDataset(kb_crop, eigen_crop | garg_crop)`,
  `DIODE*Dataset(eigen_crop | garg_crop, mask_only)`; the four evaluation sets take
  `depth_range=(min, max)` (GT kept where min < d < max, max None: no bound; the
  prediction clipped to it), default None: the class range, closed, as trained. The
  depth recipes' evaluation is `docs/evaluation.md` (`tests/test_configs.py` pins it);
  `COCOStuffDataset(label_mapping="official" | "legacy")` (the official 171 train ids /
  the thing ids read one off with person ignored, only for checkpoints trained with
  it); `Resize(depth_resample="nearest" | "edge_aware")`.
- `param_dicts`: keyword groups (a name substring or a list of them, first match wins)
  with their own lr / weight decay; OneCycleLR takes one `max_lr` per group.
- `optimizer`: `AdamW`, or `MuonAdamW` (`ldpt/utils/optimizer.py`, the segmentation
  recipes): the decoder's hidden 2-D weights step with Nesterov momentum
  orthogonalised by Newton-Schulz iterations, scaled to the RMS of an AdamW step at the
  group's lr (`muon_scale="match_rms"`); every other parameter with torch's AdamW. Each
  keyword group splits into its AdamW part and its matrix part (`muon=True`);
  OneCycleLR still takes one `max_lr` per keyword group and cycles beta1 of the AdamW
  parts only. Options (`muon_lr`, `muon_weight_decay`, `muon_exclude`, `muon_split`,
  `muon_min_dim`, `muon_momentum`, `nesterov`, `ns_steps`) are at their defaults in the
  recipes (`tests/test_muon_adamw.py`).
- Speed (not part of a recipe): `compile_blocks=True` (torch.compile of the blocks,
  training steps only), `sdpa_backends=[cudnn,flash,efficient,math]` (training steps;
  evaluation keeps PyTorch's order), `allow_tf32`.

`tests/test_configs.py` pins every recipe (decoder parameter count, SHA-256 of the
resolved config) and the relations between them; a changed recipe must update its pin
deliberately. Keep the recipes flattened and their hook order.

## Data and weights

- `data/<name>` relative to the repo root (`docs/data.md`, `ldpt/datasets/README.md`);
  split JSONs in `data/<name>/splits/`, from `scripts/datasets/process_<name>_json.py
  --root data/<name>`; the `*_dino_95.json` training splits from `deduplicate.py`. The
  processed depth sets are on Hugging Face (`pointcept-bot/DepthHybrid`).
- DINOv3: `checkpoints/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` (official
  release; never redistributed; the sha256 prefix in the name is checked). Inference
  also takes a download of `facebook/dinov3-vit7b16-pretrain-lvd1689m` (see above).
- `data/`, `exp/`, `checkpoints/` are git-ignored.

## Tests

```bash
pytest -m "not weights and not data"      # CPU (conftest hides the GPUs)
LDPT_CKPT_LARGE_DEPTH_ZEROSHOT=<file> LDPT_CKPT_LARGE_DEPTH_MIXED=<file> \
    LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF=<file> \
    LDPT_DINOV3=<DINOv3 file> pytest -m weights   # docs/install.md fetches the files
pytest -m data                            # needs the datasets under data/
```

Markers: `weights` (release files via `LDPT_CKPT_LARGE_DEPTH_ZEROSHOT` /
`LDPT_CKPT_LARGE_DEPTH_MIXED` / `LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF`, the DINOv3 file via
`LDPT_DINOV3`; skipped when unset),
`data` (datasets on disk). Tests build tiny models (`tests/helpers.py`: a tiny
DINOv3-shaped encoder, a parameter-free feature stub) and run real training /
evaluation / resume loops on synthetic data on the CPU.
`tests/test_release_hygiene.py` keeps private paths, e-mail addresses, run links and
commit ids out of the tree.

## Conventions

- Python 3.10, PyTorch 2.8; format with black (line length 88).
- A new model or loss option is off by default and bitwise the existing behaviour when
  off; add a test that shows both.
- Never change the math behind a released checkpoint silently: the golden, preset and
  recipe tests are the guard.
- Docstrings state shapes (`[B, C, H, W]`), units and what a checkpoint depends on.
- Keep the inference path (`ldpt.models.l_dpt.api`, `ldpt.utils.depth_image`,
  `ldpt.utils.seg_image` and their imports) free of training dependencies; `tests/test_hf_space.py` checks the import
  closure.
