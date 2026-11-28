# Datasets

What each dataset class in this folder reads from disk, and the script that writes its
split files. Read it to rebuild a set from its original source or to add a dataset class.
To download the processed depth sets instead, see [docs/data.md](../../docs/data.md).

## How a set is read

- A set lives in `data/<name>` (the config's `data_root`), its split files in
  `data/<name>/splits/`. A split file is a JSON list with one entry per sample; an entry
  maps `image`, `depth`, `segment` or `depth_mask` to a path relative to `data/<name>`.
- A config's `split` names one split file or a list of them, read in order; a missing
  file raises (`DefaultDataset`, `default.py`).
- `get_data(idx)` returns the keys of `keys.py`: `image` (HxWx3 float32 RGB in [0, 1]);
  `depth` (HxW, metres; 0 where the ground truth is invalid, outside the class range or
  outside an evaluation crop) with `depth_min` and `depth_max`, or `segment` (HxW train
  ids, -1 ignored); and `name`, from `get_data_name(idx)`.
- A name is `<prefix>/<id>`. The depth loss picks a sample's label type by the prefix
  (`label_types` in `configs/hybrid/*.py`; an unknown prefix raises), and the evaluator
  groups its metrics by it.

## Depth sets

| Folder | Class | Files of one sample, under the folder | Depth on disk | Class range (m) | Prefix |
|---|---|---|---|---|---|
| `hypersim` | `HypersimDataset` | `scenes/<scene>/images/scene_cam_XX_final_preview/frame.NNNN.tonemap.jpg`; `scene_cam_XX_geometry_hdf5/frame.NNNN.depth_meters.hdf5` and `.semantic.hdf5` (read for depth training too) | distance along the ray (HDF5), converted to depth | 0.01-100 | `hypersim` |
| `vkitti2` | `VKITTI2Dataset` | `Scene*/<condition>/frames/{rgb,depth,classSegmentation}/Camera_0/` | 16-bit PNG, cm | 0.01-80 | `VKITTI2` |
| `structured3d` | `Structured3DDataset` | `Structured3D/` (the "Perspective (full)" archives) | 16-bit PNG, mm | 1e-6-65 | `structure3d` |
| `tartanair` | `TartanAirDataset` | `data/` (left camera): `.png` image, `.npy` depth | `.npy`, m | 0.01-500 | `tartanair` |
| `blendedmvs`, `blendedmvs+`, `blendedmvs++` | `BlendedMVSDataset`, `BlendedMVSPDataset`, `BlendedMVSPPDataset` | `files/*/blended_images/*.jpg`, `files/*/rendered_depth_maps/*.pfm` | PFM, as stored | 0.01-500 | `blendedmvs`, `blendedmvs+`, `blendedmvs++` |
| `irs` | `IRSDataset` | `IRS_small/`: `l_*.png` image, `d_*.exr` disparity | EXR disparity; depth = 48 / disparity | 0.01-20 | `irs` |
| `nyuv2` | `NYUv2Dataset` | `train/`, `test/` in the BTS layout; the BTS file lists in `splits/` | 16-bit PNG, mm | 1e-6-10 | `nyuv2` |
| `kitti` | `KITTIDataset` | `image/` (raw data), `depth/` (BTS ground truth); the BTS Eigen file lists in `splits/` | 16-bit PNG, 1/256 m | 1e-6-80 | `kitti` |
| `diode` | `DIODEIndoorDataset`, `DIODEOutdoorDataset` | the DIODE release with its file lists `data_list/<split>.csv`; `*_depth.npy` and `*_depth_mask.npy` | `.npy`, m, with a validity mask | 1e-6-10 / 1e-6-80 | `DIODE-INDOOR`, `DIODE-OUTDOOR` |
| `eth3d` | `ETH3DDataset` | `<scene>/images/` and `<scene>/ground_truth_depth/` of the two archives below | raw float32 at the image size | 0.1-200 | `eth3d` |

The four evaluation sets (NYUv2, KITTI, ETH3D, DIODE) take `depth_range` in place of the
class range ([docs/evaluation.md](../../docs/evaluation.md)).

## Split files

Run each script as `python scripts/datasets/<script> --root data/<name>`, with the extra
arguments shown. The header of each script gives the download source.

| Folder | Script and extra arguments | Writes to `splits/` | Trains on |
|---|---|---|---|
| `hypersim` | `process_hypersim_json.py --metadata-csv <csv>` | `train.json`, `val.json`, `test.json` (59,543 / 7,386 / 7,690 frames) | `train_dino_95.json`, `val_dino_95.json` |
| `vkitti2` | `process_vkitti2_json.py` | `train.json`, `val.json` (Scene06) | `train_dino_95.json`, `val_dino_95.json` |
| `structured3d` | `process_structured3d_json.py` (drops mostly white renders) | `train.json` | `train.json` |
| `tartanair` | `process_tartanair_json.py` | `train_easy.json`, `train_hard.json` | `easy_dino_95.json`, `hard_dino_95.json` |
| `blendedmvs`, `blendedmvs+`, `blendedmvs++` | `process_blendedmvs_json.py`, once per folder | `train.json` | `train_dino_95.json` |
| `irs` | `process_irs_json.py` | `train_small.json` | `train_dino_95.json` |
| `nyuv2` | `process_nyuv2_json.py` | `train.json`, `test.json` | `train.json` (mixed model) |
| `kitti` | `process_kitti_json.py --splits train test` | `train.json`, `test.json` | `train.json` (mixed model) |
| `diode` | `process_diode_json.py --splits train_indoors train_outdoor val_indoors val_outdoor` | one JSON per split | `train_indoors.json`, `train_outdoor.json` (mixed model) |
| `eth3d` | `process_eth3d_json.py` | `train.json` (the split with depth) | evaluation only |

The evaluation splits are NYUv2's and KITTI's `test.json`, DIODE's `val_indoors.json`
and `val_outdoor.json`, and ETH3D's `train.json`.

The `*_dino_95.json` splits drop near-duplicate neighbouring frames: within a scene, a
frame is kept only when the cosine similarity of its DINOv2 feature to the last kept
frame is below 0.95. `deduplicate.py` needs CUDA and the `data` extra
(`pip install -e ".[data]"`):

```bash
dedup() {  # dedup <name> <input split> <output split>
  torchrun --nproc_per_node=8 scripts/datasets/deduplicate.py \
    --input_json data/$1/splits/$2 --output_json data/$1/splits/$3 --data_root data/$1
}
dedup hypersim train.json train_dino_95.json
dedup hypersim val.json val_dino_95.json
dedup vkitti2 train.json train_dino_95.json
dedup vkitti2 val.json val_dino_95.json
dedup tartanair train_easy.json easy_dino_95.json
dedup tartanair train_hard.json hard_dino_95.json
dedup irs train_small.json train_dino_95.json
for name in blendedmvs blendedmvs+ blendedmvs++; do
  dedup $name train.json train_dino_95.json
done
```

The threshold can move a few borderline frames between rebuilds. The sample counts the
released depth models were trained on are in the headers of `configs/hybrid/*.py` and in
`tests/test_configs.py` (`TRAIN_SAMPLES`); `pytest -m data -k split_counts` compares your
training splits with them.

## Sources

**hypersim.** The ml-hypersim contrib downloader pulls only the three files of every frame
(every scene, test included) out of the remote per-scene zips:

```bash
git clone https://github.com/apple/ml-hypersim
for f in tonemap.jpg depth_meters.hdf5 semantic.hdf5; do
  python ml-hypersim/contrib/99991/download.py -d data/hypersim/scenes -c $f -s
done
python scripts/datasets/process_hypersim_json.py --root data/hypersim \
  --metadata-csv ml-hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv
```

The full zips of ml-hypersim's `dataset_download_images.py`, decompressed into
`data/hypersim/scenes`, give the same layout. The downloader skips existing files: after
an interrupted run, redo the scene it was on with `-e <scene> -o`. The split script
follows the official scene split and skips frames with a missing file.
`filter_hypersim_invalid_segments.py` writes a copy of a split without the frames whose
segment labels are unusable; `process_hypersim_segment_json.py` adds `segment` to a split
file that lacks it.

**vkitti2.**

```bash
mkdir -p data/vkitti2
for part in rgb depth classSegmentation; do
  wget -P data/vkitti2 https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_$part.tar
  tar -xf data/vkitti2/vkitti_2.0.3_$part.tar -C data/vkitti2
done
```

**tartanair**, with the download tool of <https://github.com/castacks/tartanair_tools>:

```bash
python download_training.py --output-dir data/tartanair/data --rgb --unzip --huggingface
python download_training.py --output-dir data/tartanair/data --depth --unzip --huggingface
```

**structured3d**: every "Perspective (full)" archive listed in
<https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/README.txt>, into
`data/structured3d/Structured3D/`. **blendedmvs, blendedmvs+, blendedmvs++**: the
low-res sets, e.g. from <https://aistudio.baidu.com/datasetdetail/130299/0>, into
`data/<folder>/files/`. **irs**: <https://github.com/HKBU-HPML/IRS>, its `IRS_small`
(`Home/`, `Office/`, `Restaurant/`, `Store/`) into `data/irs/IRS_small/`. **nyuv2,
kitti**: the BTS preparation (links in the headers of `process_nyuv2_json.py` and
`process_kitti_json.py`). **diode**: <https://diode-dataset.org/>. **eth3d**:
<https://www.eth3d.net/data/multi_view_training_dslr_jpg.7z> and
<https://www.eth3d.net/data/multi_view_training_dslr_scan_eval.7z>, into `data/eth3d`.

## Segmentation sets

| Folder | Class | Files | Prefix |
|---|---|---|---|
| `cocostuff` | `COCOStuffDataset` (`label_mapping="official"`: 171 train ids) | `train2017/`, `val2017/`: the `.jpg` images and the stuffthingmaps `.png` label maps together | `cocostuff` |
| `voc2012` | `VOC2012Dataset` (21 classes, 255 ignored) | `JPEGImages/<id>.jpg`, `SegmentationClass/<id>.png`, the lists in `ImageSets/Segmentation/` | `voc2012` |

```bash
mkdir -p data/cocostuff
wget -P data/cocostuff http://images.cocodataset.org/zips/train2017.zip
wget -P data/cocostuff http://images.cocodataset.org/zips/val2017.zip
wget -P data/cocostuff http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
for f in data/cocostuff/*.zip; do unzip -q "$f" -d data/cocostuff; done
python scripts/datasets/process_cocostuff_json.py --root data/cocostuff   # train.json, val.json

git clone https://www.modelscope.cn/datasets/anyforge/pascal-voc-2012.git
# data/voc2012 is the clone's VOC2012 folder (move it there or link it)
python scripts/datasets/process_voc2012_json.py --root data/voc2012       # train.json, val.json
```

## Adding a dataset class

1. A module `ldpt/datasets/<name>.py` with a class derived from `DefaultDataset`,
   registered with `@DATASETS.register_module()` and imported in `__init__.py`; set
   `DEPTH_MIN` / `DEPTH_MAX` and implement `get_data` and `get_data_name` as above.
2. A script `scripts/datasets/process_<name>_json.py --root data/<name>` that writes
   `splits/*.json`, with the download source in its header.
3. For depth training, add the class to the config's `data` and its prefix to
   `label_types` (the type sets its levels and trim, section 1 of
   [docs/depth_loss.md](../../docs/depth_loss.md)); a changed recipe updates its pin in
   `tests/test_configs.py` ([AGENTS.md](../../AGENTS.md)).
