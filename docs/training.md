# Training

The released depth models are reproduced by the configs in `configs/hybrid/`, on 8 GPUs
with 80 GB each (the released runs used NVIDIA GB200s). Install the `train` extra
([install.md](install.md)), put the DINOv3 file at
`checkpoints/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` and the data under
`data/` ([data.md](data.md)).

## The model

Both depth models share one architecture and differ only in training data:

- **Encoder:** DINOv3 ViT-7B/16, frozen (stored in bf16), tapped after blocks 10, 20, 30
  and 40 of 40 (`encoder_inter_layers=[9, 19, 29, 39]`, 0-based in code).
- **Decoder:** a latent DPT on the encoder's patch grid: 1024 channels, 16 heads, 4
  stages of 4 transformer blocks with 2D RoPE (270,995,712 parameters, run in bf16); a 3x3
  convolution and a pixel shuffle read out to pixels. The attention of the last stage
  runs in fp32, and every decoder attention logit is bounded by 1000.
- **Output:** an exponential depth head with an fp32 readout: positive relative depth,
  larger = farther, defined up to a scale and a shift per image.

One class, `LatentDPT` in `ldpt/models/l_dpt/l_dpt_v1m1_base.py` (registered as
`L-DPT-v1m1`), serves every variant; [AGENTS.md](../AGENTS.md) lists its options.

## Config names

Configs follow Pointcept's naming:
`configs/<training data>/<task>-<model>-v<X>m<Y>-<id>-<description>.py`.

| Part | Here | Meaning |
|---|---|---|
| training data | `hybrid`, `cocostuff`, `voc2012` | the folder; `hybrid` is the mix of depth sets |
| task | `depth`, `semseg` | relative depth or semantic segmentation |
| model | `l-dpt` | |
| `v<X>m<Y>` | `v1m1` | the model version: architecture 1, modification 1, the registered `L-DPT-v1m1` |
| id | `0`, `1`, `1a` | the recipe; `<id>a` is a second stage that starts from `<id>` |
| description | `large-zeroshot`, `large-mixed`, ... | decoder size and data |

"Large" is the largest decoder of each task: 1024 channels and (4, 4, 4, 4) blocks for
depth, 2048 channels and (2, 2, 2, 2) blocks for segmentation. A run is named after its
config; another seed of the same config adds a suffix (`-seed1`) rather than a new
config. Each config states every key and resolves key by key to the configuration of
the run it reproduces (for the depth configs, the runs of the released weights);
`tests/test_configs.py` pins them.

## `train.sh`

```bash
./train.sh -g <GPUs> -d <config folder> -c <config> -n <run name> [-r auto] [-o "key=value ..."]
```

It copies the code into the experiment folder `exp/<config folder>/<run name>/` and runs
`train.py` there with one process per GPU. The flags are Pointcept's:

| Flag | |
|---|---|
| `-g` | GPUs per node (default: all visible) |
| `-d` | the config folder under `configs/` (default `hybrid`) |
| `-c` | the config name, without `.py` |
| `-n` | the run name |
| `-r auto` | resume from `model/model_last.pth` when it exists |
| `-o "key=value ..."` | override config keys, list items by index (`hooks.6.save_freq_iter=2000`) |
| `-e` | the experiment root (default `exp`) |
| `-m` | the number of nodes |

The run folder holds `config.py`, `code/`, `model/`, `metrics.jsonl`, `train.log` and the
TensorBoard logs.

## Depth recipes

| Config | Training data | Samples | Released weights |
|---|---|---|---|
| `depth-l-dpt-v1m1-0-large-zeroshot` | Hypersim, Virtual KITTI 2, Structured3D, TartanAir, BlendedMVS, BlendedMVS+, BlendedMVS++, IRS | 506,790 | epoch 7 is `l-dpt_large_depth_zeroshot` |
| `depth-l-dpt-v1m1-1-large-mixed` | the same + NYUv2, KITTI, DIODE indoor, DIODE outdoor (training splits) | 579,637 | epoch 4 (stage 1) |
| `depth-l-dpt-v1m1-1a-large-mixed-anneal` | as `1-large-mixed` | 579,637 | epoch 1 (stage 2); the average of both stages is `l-dpt_large_depth_mixed` |

```bash
# zero-shot: 25-epoch schedule; the released weights are epoch 7
./train.sh -g 8 -r auto -d hybrid -c depth-l-dpt-v1m1-0-large-zeroshot \
    -n depth-l-dpt-v1m1-0-large-zeroshot
python -m ldpt.models.l_dpt.convert \
    exp/hybrid/depth-l-dpt-v1m1-0-large-zeroshot/model/epoch_7.pth \
    l-dpt_large_depth_zeroshot.pth --preset l-dpt_large_depth_zeroshot

# mixed: stage 1 to epoch 4, a one-epoch anneal from it, the average of both
./train.sh -g 8 -r auto -d hybrid -c depth-l-dpt-v1m1-1-large-mixed \
    -n depth-l-dpt-v1m1-1-large-mixed
./train.sh -g 8 -r auto -d hybrid -c depth-l-dpt-v1m1-1a-large-mixed-anneal \
    -n depth-l-dpt-v1m1-1a-large-mixed-anneal
python scripts/average_weights.py \
    --inputs exp/hybrid/depth-l-dpt-v1m1-1-large-mixed/model/epoch_4.pth \
             exp/hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal/model/epoch_1.pth \
    --output exp/hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal/model/average.pth
python -m ldpt.models.l_dpt.convert \
    exp/hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal/model/average.pth \
    l-dpt_large_depth_mixed.pth --preset l-dpt_large_depth_mixed
```

`ldpt.models.l_dpt.convert` turns a training checkpoint into a weights file. Load your
own file by its path: `api.load("l-dpt_large_depth_zeroshot.pth")`,
`scripts/demo_depth.py --weights <file>` or `scripts/highres_infer.py --model
NAME=<file>`. Keep the run names of this page: the anneal config reads stage 1's
`exp/hybrid/depth-l-dpt-v1m1-1-large-mixed/model/epoch_4.pth`, the segmentation anneals
their recipe's epoch file, and the VOC config
`exp/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal/model/average.pth`.

- **Optimisation.** Global batch 64 (8 per GPU on 8 GPUs), AdamW with lr 1e-4 and
  weight decay 0.1, OneCycleLR (cosine, 5 % warm-up), gradient clipping at 1.0, bf16
  decoder. Training crops are 768 x 768, from images rescaled by 0.5-2.0 of 768.
- **Schedule.** 25 epochs, kept although the released weights stop early, because
  OneCycleLR depends on the total number of steps. Every epoch's decoder weights are
  kept (`model/epoch_<N>.pth`, 1.1 GB each). A run may be stopped once its released
  epoch is written (`epoch_7.pth` for zero-shot, `epoch_4.pth` for stage 1 of mixed):
  later epochs are not used.
- **Loss.** The depth loss is a robust scale-and-shift-invariant loss in the MoGe family:
  a 1/z-weighted relative L1 under a robust fit, local shift terms and gradient matching
  on synthetic labels, with a gated warm-up. Its specification:
  [depth_loss.md](depth_loss.md).
- **Expected numbers.** The README's Results table, measured as in
  [evaluation.md](evaluation.md). `model_best.pth` follows the geometric mean of AbsRel over the
  evaluation sets; the released weights are fixed epochs instead.
- **Speed** (not part of the recipe). `-o "compile_blocks=True
  sdpa_backends=[cudnn,flash,efficient,math]"`. `-o "hooks.6.save_freq_iter=2000"` also
  writes `model_last.pth` every 2000 iterations, and a resume then continues mid-epoch on
  the same batches.
- **Logging.** TensorBoard logs go to the run folder; W&B is off unless
  `-o "enable_wandb=True"`.

The hooks of the depth configs keep fixed indices, which overrides use: 0
`CheckpointLoader`, 1 `ModelHook`, 2 `IterationTimer`, 3 `InformationWriter`, 4
`DepthEvaluator`, 5 `DepthVisualizer`, 6 `CheckpointSaver`, 7 `CriterionWeightWarmup`
(absent from the anneal config).

Debug on one GPU:

```bash
python train.py --config-file configs/hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py \
    --num-gpus 1 --options save_path=exp/debug batch_size=2 num_workers=0
```

On several nodes, start one `train.sh` per node with `-m <nodes>` (for example under
`srun --ntasks-per-node=1`): it takes the rendezvous address and the node rank from
`SLURM_NODELIST` and `SLURM_NODEID`.

## Segmentation

The same design also does semantic segmentation; the COCO-Stuff model's weights are
`l-dpt_large_semseg_cocostuff` ([inference.md](inference.md#segmentation)).
The segmentation decoder is larger: 2048 channels, 16 heads and (2, 2, 2, 2) blocks
(attention at 1024 channels; 772,017,131 parameters) on the same frozen DINOv3
ViT-7B/16, with QK-norm attention, a BatchNorm on each tapped encoder layer, 64-channel
pixel features, a 1x1 classifier and a cross-entropy + Lovász loss, trained with
MuonAdamW. COCO-Stuff has one recipe, with a one-epoch anneal as its second stage:

| Config | Recipe | Starts from |
|---|---|---|
| `semseg-l-dpt-v1m1-1-large` | the decoder above, MuonAdamW, no weight decay on norms, biases and LayerScale | DINOv3 |
| `semseg-l-dpt-v1m1-1a-large-anneal` | one epoch of `1` at a lower peak lr, seeds 0, 1 and 2 | `1`'s `epoch_9.pth` |

The COCO-Stuff model is the uniform average of the start of the anneal `1a` and the ends
of its three runs (seeds 0, 1 and 2):

```bash
# stage 1: 25-epoch schedule; the anneal starts from epoch 9
./train.sh -g 8 -r auto -d cocostuff -c semseg-l-dpt-v1m1-1-large \
    -n semseg-l-dpt-v1m1-1-large
# stage 2: one epoch from stage 1's epoch_9.pth; seed 0, then seeds 1 and 2
./train.sh -g 8 -r auto -d cocostuff -c semseg-l-dpt-v1m1-1a-large-anneal \
    -n semseg-l-dpt-v1m1-1a-large-anneal
for seed in 1 2; do
  ./train.sh -g 8 -r auto -d cocostuff -c semseg-l-dpt-v1m1-1a-large-anneal \
      -n semseg-l-dpt-v1m1-1a-large-anneal-seed$seed -o "seed=$seed"
done
# the average of the start and the three ends of stage 2, then its mIoU
python scripts/average_weights.py \
    --inputs exp/cocostuff/semseg-l-dpt-v1m1-1-large/model/epoch_9.pth \
             exp/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal/model/epoch_1.pth \
             exp/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal-seed1/model/epoch_1.pth \
             exp/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal-seed2/model/epoch_1.pth \
    --output exp/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal/model/average.pth
python scripts/evaluate.py \
    --config-file configs/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal.py \
    --checkpoints exp/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal/model/average.pth \
    --num-gpus 8 --output exp/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal/eval.json
python -m ldpt.models.l_dpt.convert \
    exp/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal/model/average.pth \
    l-dpt_large_semseg_cocostuff.pth --preset l-dpt_large_semseg_cocostuff
```

`--options 'hooks.4.tta=[["flip"]]'` adds the flip TTA to the evaluation (hook 4 is the
`SemSegEvaluator`). `scripts/evaluate.py --checkpoints <the four files> --average`
evaluates the same average without writing it (equal up to float rounding: it sums in
float32, `average_weights.py` in float64): both average every weight and the
BatchNorm running statistics, and take the BatchNorm batch counters (not read in
evaluation) from the last file.

PASCAL VOC (`semseg-l-dpt-v1m1-1-large`) fine-tunes the COCO-Stuff average:
COCO-Stuff recipe `1` on VOC for 20 epochs, with `allow_missing=False`:

```bash
# after the COCO-Stuff average above; seed 0, then seeds 1 and 2
./train.sh -g 8 -r auto -d voc2012 -c semseg-l-dpt-v1m1-1-large \
    -n semseg-l-dpt-v1m1-1-large
for seed in 1 2; do
  ./train.sh -g 8 -r auto -d voc2012 -c semseg-l-dpt-v1m1-1-large \
      -n semseg-l-dpt-v1m1-1-large-seed$seed -o "seed=$seed"
done
```

- **COCO-Stuff 164k** (`data/cocostuff`): the official label mapping (all 171 classes),
  batch 32 as 4 per GPU on 8 GPUs (the Lovász term is computed per GPU), 768 x 768 crops
  from images rescaled by 0.5-2.0 of 768, OneCycleLR (cosine, 5 % warm-up) with peak lr
  5e-4 and the frozen encoder's group at lr / 10; mIoU at short side 768, whole image,
  single scale; `model_best.pth` is the epoch with the highest val mIoU. Stage 1 keeps
  the 25-epoch schedule (OneCycleLR depends on the total number of steps) and every
  epoch's decoder weights (`model/epoch_<N>.pth`, 3.1 GB each); it may be stopped once
  the anneal's start file is written. The anneal: one epoch with a 3 % warm-up to 4e-4;
  three runs, with seeds 0, 1 and 2.
- **MuonAdamW**: Nesterov momentum orthogonalised by Newton-Schulz iterations on the
  decoder's attention, MLP and selection matrices, scaled to the RMS of an AdamW step at
  the same lr; AdamW on everything else (`ldpt/utils/optimizer.py`).
- **PASCAL VOC 2012** (`data/voc2012`): the decoder starts from
  `exp/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal/model/average.pth` and the 21-class
  head from the matching rows of that COCO-Stuff head
  (`CheckpointLoader(seg_head_rows=...)`); 1,464 training images, 20 epochs at batch
  32; three runs, with seeds 0, 1 and 2.

The data of both sets: [`ldpt/datasets/README.md`](../ldpt/datasets/README.md).
