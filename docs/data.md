# Data

Where the training and evaluation data goes, how to download the processed depth sets,
and how to rebuild them from the original sources.

Every dataset lives in `data/<name>` relative to the repository root (the configs'
`data_root`), with its split JSONs in `data/<name>/splits/`. The split JSONs list the
files of each sample relative to `data/<name>`.

## Layout

| Set | Folder | Training splits | Samples | Used by |
|---|---|---|---|---|
| Hypersim | `data/hypersim` | `train_dino_95.json`, `val_dino_95.json` | 63,865 | both |
| Virtual KITTI 2 | `data/vkitti2` | `train_dino_95.json`, `val_dino_95.json` | 7,011 | both |
| Structured3D | `data/structured3d` (files in `Structured3D/`) | `train.json` | 71,005 | both |
| TartanAir | `data/tartanair` (files in `data/`) | `easy_dino_95.json`, `hard_dino_95.json` | 207,255 | both |
| BlendedMVS | `data/blendedmvs` (files in `files/`) | `train_dino_95.json` | 17,314 | both |
| BlendedMVS+ | `data/blendedmvs+` (files in `files/`) | `train_dino_95.json` | 41,133 | both |
| BlendedMVS++ | `data/blendedmvs++` (files in `files/`) | `train_dino_95.json` | 40,332 | both |
| IRS | `data/irs` (files in `IRS_small/`) | `train_dino_95.json` | 58,875 | both |
| NYUv2 | `data/nyuv2` | `train.json` | 24,231 | mixed |
| KITTI | `data/kitti` | `train.json` | 23,158 | mixed |
| DIODE | `data/diode` | `train_indoors.json`, `train_outdoor.json` | 8,574 + 16,884 | mixed |

"Both" is the zero-shot and the mixed model: 506,790 samples for zero-shot, 579,637 for
mixed. NYUv2, KITTI, DIODE and ETH3D also provide the evaluation splits
([evaluation.md](evaluation.md)).

The training splits named `*_dino_95.json` drop near-duplicate neighbouring frames:
within a scene, a frame is kept only when the cosine similarity of its DINOv2 feature to
the last kept frame is below 0.95 (`scripts/datasets/deduplicate.py`).

## Download the processed sets

[DepthHybrid](https://huggingface.co/datasets/pointcept-bot/DepthHybrid) on Hugging Face
holds the sets above and ETH3D in this layout, with the split JSONs the released runs
used. We do not own these datasets: each keeps the terms of its source, which you must
accept before you use it.

Per set, the repository holds `<set>/splits/` (the split JSONs), `<set>/data/<set>-NNN.tar.gz`
(the files the splits reference, in shards of at most 10 GB before compression) and
`<set>/MANIFEST.json` (the shards with their sha256 and the sample count of every split).
To download one set and unpack it:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "pointcept-bot/DepthHybrid",
    repo_type="dataset",
    allow_patterns=["hypersim/*"],  # leave out for every set
    local_dir="data",
)
```

```bash
for f in data/hypersim/data/hypersim-*.tar.gz; do tar -xzf "$f" -C data/hypersim; done
```

The shards can be deleted once unpacked. For every set at once:
`for f in data/*/data/*.tar.gz; do tar -xzf "$f" -C "${f%/data/*}"; done`.

## Rebuild from the original sources

Download each set from its official source and write its split JSONs as described in
[`ldpt/datasets/README.md`](../ldpt/datasets/README.md):

```bash
python scripts/datasets/process_<name>_json.py --root data/<name>
# the training splits *_dino_95.json, once per input split; the names per set are in
# ldpt/datasets/README.md (needs CUDA and the data extra: pip install -e ".[data]")
torchrun --nproc_per_node=8 scripts/datasets/deduplicate.py \
  --input_json data/<name>/splits/train.json \
  --output_json data/<name>/splits/train_dino_95.json --data_root data/<name>
```

These two commands are the general form. Several sets need other arguments or file
names: Hypersim needs `--metadata-csv`, KITTI and DIODE need `--splits` for their
training splits, TartanAir and IRS write `train_easy.json`, `train_hard.json` and
`train_small.json`, and the three BlendedMVS folders share `process_blendedmvs_json.py`.
[`ldpt/datasets/README.md`](../ldpt/datasets/README.md) has the exact commands for every
set; the header of each script gives its download source. The folder column above names
the sub-folder some archives must be extracted into.

## Check the splits

```bash
pytest -m data -k split_counts
```

compares the sample counts of your training splits with those of the released runs
(pinned in `tests/test_configs.py` and in the headers of `configs/hybrid/*.py`). The
deduplication can move a few borderline frames between rebuilds (another GPU or
transformers version), which changes the steps per epoch and with them the learning-rate
schedule slightly.
