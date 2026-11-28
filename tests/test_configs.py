"""
The release recipes (configs/hybrid/*.py, configs/cocostuff/*.py, configs/voc2012/*.py)
vs the training runs behind the released weights and the reported numbers (CPU).

For every recipe:
- the model builds on the meta device with the run's trainable (= decoder) parameter
  count;
- the resolved config hashes to the recorded SHA-256 (first 16 hex digits). A depth
  recipe evaluates with the protocol of docs/evaluation.md; without its evaluation
  (data.val and hook 4, which do not affect training) it hashes to the SHA-256
  recorded when that part was compared key by key with the run's config.py. With
  LDPT_RELEASE_RUN_CONFIGS=<folder of the runs' config.py, each at its recipe's path
  under configs/, e.g. hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py> they are compared
  key by key: only LAUNCH_KEYS (and the saver's mid-epoch interval) may differ, a
  depth recipe's evaluation is left out, and a start checkpoint names the earlier
  stage's own save path (train.sh: exp/<folder>/<name>) instead of the run's;
- the relations between the recipes (mixed = zero-shot + the real-domain sets; the
  anneal = the mixed recipe from its epoch 4; COCO-Stuff recipe 1: QK-norm, tap
  BatchNorms, 64 pixel features and MuonAdamW; its anneal = the recipe for one epoch
  from its epoch 9) and the depth model = presets "l-dpt_large_depth_zeroshot" /
  "l-dpt_large_depth_mixed";
- VOC recipe 1 = COCO-Stuff recipe 1 on VOC (20 epochs, 21 classes, the head from the
  COCO-Stuff rows), from that recipe's anneal average.
"""

import glob
import hashlib
import inspect
import json
import os
from dataclasses import dataclass

import pytest
import torch

from helpers import REPO

from ldpt.models.builder import build_model
from ldpt.utils.config import Config


@dataclass(frozen=True)
class Recipe:
    params: int  # trainable (decoder) parameters
    sha256: str  # of the resolved config (first 16 hex digits)
    start: str = None  # the earlier stage whose save path checkpoint[1] names
    start_file: str = None  # that stage's file in its model/ folder
    # depth: of the config without its evaluation (without_evaluation), recorded when
    # that part was compared key by key with the run's config.py
    train_sha256: str = None


ZS = "hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py"
MIX = "hybrid/depth-l-dpt-v1m1-1-large-mixed.py"
ANNEAL = "hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal.py"
COCO_1 = "cocostuff/semseg-l-dpt-v1m1-1-large.py"
COCO_1A = "cocostuff/semseg-l-dpt-v1m1-1a-large-anneal.py"
# VOC recipe 1, from the recipe-1 average
VOC_1 = "voc2012/semseg-l-dpt-v1m1-1-large.py"
RECIPE = {
    ZS: Recipe(270_995_712, "acd37d364ed0c288", train_sha256="39d7f83e80a9cb8b"),
    MIX: Recipe(270_995_712, "4d29f1c8678e2c76", train_sha256="dd7007e96b1cf162"),
    ANNEAL: Recipe(
        270_995_712, "f8d2072646fcce82", MIX, "epoch_4.pth", "0e8d6f1adf19dc31"
    ),
    COCO_1: Recipe(772_017_131, "8caf88b1d5892638"),
    COCO_1A: Recipe(772_017_131, "e4313e0ce67e61db", COCO_1, "epoch_9.pth"),
    VOC_1: Recipe(772_007_381, "916822da74e72d86", COCO_1A, "average.pth"),
}
DEPTH = (ZS, MIX, ANNEAL)
# the paper's variants table: decoder-size keys and the parameter count without the
# 3x3 readout convolution (in millions; the release decoder with it has 271M)
DECODER_SIZES = {
    "large": (dict(dec_dim=1024, dec_attn_downscale=1, dec_num_heads=16), 269),
}
DECODER_LAYERS = {"large": [4, 4, 4, 4]}
# VOC recipe 1 -> (its size, the COCO-Stuff recipe 1 it fine-tunes)
VOC_1_RECIPES = {
    VOC_1: ("large", COCO_1),
}
# recipes without a run's config.py to compare with (VOC recipe 1: none in the set)
NO_RUN_CONFIG = {*VOC_1_RECIPES}
RELEASE_RUN_CONFIGS = os.environ.get("LDPT_RELEASE_RUN_CONFIGS")
DINOV3 = "checkpoints/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
# training samples per set of each depth run, in the order of data.train.datasets (as
# train.log reports them: "Totally N samples in <set> / <split> set"), and the total.
# The *_dino_95.json splits come from scripts/datasets/deduplicate.py (DINOv2 features,
# cosine 0.95 per frame), so a rebuild can move borderline frames (another GPU or
# transformers version) and with them the steps per epoch and the OneCycleLR schedule;
# test_recipe_split_counts (marker data) compares the splits under data/ with these.
ZS_SAMPLES = (
    ("HypersimDataset", 63_865),
    ("VKITTI2Dataset", 7_011),
    ("Structured3DDataset", 71_005),
    ("TartanAirDataset", 207_255),
    ("BlendedMVSDataset", 17_314),
    ("BlendedMVSPDataset", 41_133),
    ("BlendedMVSPPDataset", 40_332),
    ("IRSDataset", 58_875),
)
REAL_DOMAIN_SAMPLES = (
    ("NYUv2Dataset", 24_231),
    ("KITTIDataset", 23_158),
    ("DIODEIndoorDataset", 8_574),
    ("DIODEOutdoorDataset", 16_884),
)
REAL_DOMAIN = tuple(name for name, _ in REAL_DOMAIN_SAMPLES)
TRAIN_SAMPLES = {
    ZS: (ZS_SAMPLES, 506_790),
    MIX: (ZS_SAMPLES + REAL_DOMAIN_SAMPLES, 579_637),
    ANNEAL: (ZS_SAMPLES + REAL_DOMAIN_SAMPLES, 579_637),
}
# settings of a launch, not of the recipe (speed, logging, workers, output folder);
# some (log_group_norms, log_module_norms) occur only in the runs' config.py
LAUNCH_KEYS = (
    "compile_blocks",
    "sdpa_backends",
    "log_group_norms",
    "log_module_norms",
    "num_workers",
    "save_path",
    "enable_wandb",
    "wandb_project",
    "wandb_key",
)
# CheckpointSaver keys of a launch (mid-epoch saves, kept iteration states;
# keep_iter_weights occurs only in the runs' config.py)
LAUNCH_SAVER_KEYS = ("save_freq_iter", "keep_iter_weights")


# ---------------------------------------------------------------- helpers


def plain(cfg):
    """Resolved Config -> JSON types (tuples become lists)."""
    return json.loads(json.dumps(dict(cfg)))


def digest(cfg):
    text = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def diff(a, b, path="cfg"):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in b:
                out.append(f"{path}.{k}: only in the recipe")
            elif k not in a:
                out.append(f"{path}.{k}: only in the run config")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [f"{path}: {a!r} (recipe) != {b!r} (run)"]


def load_recipe(rel):
    return Config.fromfile(os.path.join(REPO, "configs", rel))


def without_evaluation(cfg):
    """plain() cfg of a depth recipe or run without its evaluation: data.val and the
    DepthEvaluator (hook 4), which do not affect training."""
    cfg = dict(cfg, data={k: v for k, v in cfg["data"].items() if k != "val"})
    assert cfg["hooks"][4]["type"] == "DepthEvaluator"
    cfg["hooks"] = cfg["hooks"][:4] + cfg["hooks"][5:]
    return cfg


def save_path(rel):
    """train.sh's experiment folder of a run of recipe ``rel`` named after it."""
    folder, name = rel[: -len(".py")].split("/")
    return f"exp/{folder}/{name}"


def without_launch(cfg):
    cfg = {k: v for k, v in cfg.items() if k not in LAUNCH_KEYS}
    cfg["hooks"] = [
        {k: v for k, v in h.items() if k not in LAUNCH_SAVER_KEYS} for h in cfg["hooks"]
    ]
    return cfg


def merged(changes, cfg):
    cfg = dict(cfg)
    for k, v in changes.items():
        # a dict merges into the base's dict; a key new to the base is set as is
        cfg[k] = merged(v, cfg[k]) if isinstance(v, dict) and k in cfg else v
    return cfg


# ---------------------------------------------------------------- every recipe


def test_recipes_listed():
    found = glob.glob(os.path.join(REPO, "configs", "*", "*.py"))
    found = {os.path.relpath(f, os.path.join(REPO, "configs")) for f in found}
    found = {f for f in found if not f.startswith("_base_" + os.sep)}
    assert found == set(RECIPE)


@pytest.mark.parametrize("rel", list(RECIPE))
def test_recipe_param_count(rel):
    with torch.device("meta"):
        model = build_model(load_recipe(rel).model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decoder = sum(
        p.numel() for n, p in model.named_parameters() if not n.startswith("encoder.")
    )
    assert params == decoder == RECIPE[rel].params


@pytest.mark.parametrize("rel", list(RECIPE))
def test_recipe_digest(rel):
    assert digest(plain(load_recipe(rel))) == RECIPE[rel].sha256, (
        f"{rel} no longer resolves to the checked recipe of its run; run with "
        "LDPT_RELEASE_RUN_CONFIGS=<the runs' config.py> to see the keys"
    )


@pytest.mark.parametrize("rel", list(RECIPE))
def test_recipe_start_checkpoint(rel):
    """A second stage starts from the earlier stage's own save path (train.sh:
    exp/<folder>/<name>/model/<file>); a first stage from the DINOv3 file only."""
    checkpoint = plain(load_recipe(rel))["checkpoint"]
    recipe = RECIPE[rel]
    if recipe.start is None:
        assert checkpoint == DINOV3
    else:
        assert checkpoint == [
            DINOV3,
            f"{save_path(recipe.start)}/model/{recipe.start_file}",
        ]


@pytest.mark.skipif(not RELEASE_RUN_CONFIGS, reason="LDPT_RELEASE_RUN_CONFIGS not set")
@pytest.mark.parametrize("rel", [r for r in RECIPE if r not in NO_RUN_CONFIG])
def test_recipe_matches_run(rel):
    """Key by key against the run's config.py: only LAUNCH_KEYS (and the saver's
    LAUNCH_SAVER_KEYS) may differ, a depth recipe's evaluation is left out
    (without_evaluation); a start checkpoint is the same file of the earlier stage
    under the recipe's save path."""
    path = os.path.join(RELEASE_RUN_CONFIGS, rel)
    run = plain(Config.fromfile(path))
    recipe = plain(load_recipe(rel))
    if rel in DEPTH:
        run, recipe = without_evaluation(run), without_evaluation(recipe)
    run, recipe = without_launch(run), without_launch(recipe)
    if isinstance(recipe["checkpoint"], list):
        mine, theirs = recipe["checkpoint"][1], run["checkpoint"][1]
        assert theirs.split("/")[-2:] == mine.split("/")[-2:]
        run["checkpoint"][1] = mine
    assert diff(recipe, run) == []


def test_every_config_parses():
    paths = glob.glob(os.path.join(REPO, "configs", "*", "*.py"))
    paths = [p for p in paths if os.sep + "_base_" + os.sep not in p]
    assert len(paths) == len(RECIPE)
    for path in paths:
        cfg = Config.fromfile(path)
        assert cfg.model.type == "L-DPT-v1m1" and cfg.data.val, path


def test_public_runtime_defaults():
    """W&B is opt-in (TensorBoard always logs); the project is l-dpt."""
    cfg = Config.fromfile(os.path.join(REPO, "configs", "_base_", "default_runtime.py"))
    assert cfg.enable_wandb is False and cfg.wandb_project == "l-dpt"


# ---------------------------------------------------------------- depth recipes


@pytest.mark.parametrize("rel", DEPTH)
def test_recipe_train_samples(rel):
    """TRAIN_SAMPLES names the recipe's training sets in order, and the per-set counts
    of the run add up to its total."""
    samples, total = TRAIN_SAMPLES[rel]
    datasets = load_recipe(rel).data.train.datasets
    assert [d.type for d in datasets] == [name for name, _ in samples]
    assert sum(n for _, n in samples) == total


@pytest.mark.data
@pytest.mark.parametrize("rel", (ZS, MIX))
def test_recipe_split_counts(rel):
    """The split JSONs under data/ hold as many samples per training set as the run
    trained on (TRAIN_SAMPLES); sets whose splits are not on disk are left out."""
    from pathlib import Path

    import ldpt.datasets  # noqa: F401  (registers the datasets)
    from ldpt.datasets.builder import DATASETS

    samples, _ = TRAIN_SAMPLES[rel]
    checked, wrong = 0, []
    for d, (_, expected) in zip(load_recipe(rel).data.train.datasets, samples):
        root = Path(REPO) / d.data_root
        splits = [d.split] if isinstance(d.split, str) else list(d.split)
        if not all((root / split).is_file() for split in splits):
            continue
        dataset = object.__new__(DATASETS.get(d.type))  # reads the split JSONs only
        dataset.data_root, dataset.split = root, d.split
        found = len(dataset.get_data_list())
        checked += 1
        if found != expected:
            wrong.append(f"{d.type} {splits}: {found} (the run: {expected})")
    if not checked:
        pytest.skip(f"no training split of {rel} under data/")
    assert not wrong, "split JSONs differ from the released run: " + "; ".join(wrong)


def test_recipe_mix_is_zs_plus_real_domain():
    """The mixed recipe = the zero-shot recipe + the NYUv2 / KITTI / DIODE training sets
    (label type C of the loss; NYUv2 with eigen_crop="legacy")."""
    zs, mix = plain(load_recipe(ZS)), plain(load_recipe(MIX))
    train = mix["data"]["train"]["datasets"]
    assert train[:8] == zs["data"]["train"]["datasets"]
    assert [d["type"] for d in train[8:]] == list(REAL_DOMAIN)
    assert all(d["transform"] == zs["train_transform"] for d in train)
    assert train[8]["eigen_crop"] == "legacy"
    assert mix["model"]["criterion"]["label_types"]["C"] == [
        "nyuv2",
        "kitti",
        "DIODE-INDOOR",
        "DIODE-OUTDOOR",
    ]
    zs["data"]["train"]["datasets"] = train
    zs["model"]["criterion"]["label_types"]["C"] = mix["model"]["criterion"][
        "label_types"
    ]["C"]
    assert diff(mix, zs) == []  # the same evaluation


def test_recipe_anneal_is_mix_from_epoch_4():
    """The anneal = the mixed recipe for one epoch from its epoch_4.pth (hook 0 loads
    DINOv3 + the decoder), OneCycleLR pct_start 0.03, without the weight warm-up."""
    mix = plain(load_recipe(MIX))
    hooks = mix["hooks"]
    assert len(hooks) == 8 and hooks[7]["type"] == "CriterionWeightWarmup"
    loader = dict(
        type="CheckpointLoader",
        keywords=["module.", "module."],
        replacement=["module.encoder.", "module."],
    )
    changes = dict(
        checkpoint=[DINOV3, f"{save_path(MIX)}/model/epoch_4.pth"],
        epoch=1,
        scheduler=dict(pct_start=0.03),
        hooks=[loader] + hooks[1:7],
    )
    assert diff(plain(load_recipe(ANNEAL)), merged(changes, mix)) == []
    assert (
        hooks[4]["type"] == "DepthEvaluator" and hooks[6]["type"] == "CheckpointSaver"
    )


# Marigold's masks and ranges (prs-eth/Marigold: config/dataset_depth/data_*.yaml and
# the min_depth / max_depth of src/dataset/*_dataset.py): the val datasets' keys
MARIGOLD_VAL = [
    ("NYUv2Dataset", dict(eigen_crop=True, depth_range=[1e-3, 10.0])),
    ("KITTIDataset", dict(kb_crop=True, eigen_crop=True, depth_range=[1e-5, 80.0])),
    ("ETH3DDataset", dict(depth_range=[1e-5, None])),
    ("DIODEIndoorDataset", dict(depth_range=[0.6, 350.0], mask_only=True)),
    ("DIODEOutdoorDataset", dict(depth_range=[0.6, 350.0], mask_only=True)),
]
VAL_SPLITS = [
    "data/nyuv2/splits/test.json",
    "data/kitti/splits/test.json",
    "data/eth3d/splits/train.json",
    "data/diode/splits/val_indoors.json",
    "data/diode/splits/val_outdoor.json",
]


@pytest.mark.parametrize("rel", DEPTH)
def test_recipe_evaluation_protocol(rel):
    """Every depth recipe evaluates NYUv2 / KITTI / ETH3D / DIODE with Marigold's
    masks and ranges, no other crop, flip TTA and the seeded depth RANSAC fit (seed 0)
    on every valid pixel (docs/evaluation.md)."""
    cfg = plain(load_recipe(rel))
    val = cfg["data"]["val"]["datasets"]
    assert [d["type"] for d in val] == [t for t, _ in MARIGOLD_VAL]
    assert [f"{d['data_root']}/{d['split']}" for d in val] == VAL_SPLITS
    for d, (_, keys) in zip(val, MARIGOLD_VAL):
        extra = {k: v for k, v in d.items() if k not in ("type", "data_root", "split")}
        assert extra == dict(keys, transform=cfg["val_transform"]), d["type"]
    evaluator = cfg["hooks"][4]
    assert evaluator == dict(
        type="DepthEvaluator",
        relative=True,
        tta=[["flip"]],
        align_solver="ransac",
        ransac_seed=0,
    )


@pytest.mark.parametrize("rel", DEPTH)
def test_recipe_training_matches_run(rel):
    """Without its evaluation (data.val, hook 4) a depth recipe hashes to the SHA-256
    recorded when that part was compared key by key with the run's config.py
    (test_recipe_matches_run): the evaluation changes nothing of the training."""
    assert digest(without_evaluation(plain(load_recipe(rel)))) == (
        RECIPE[rel].train_sha256
    )


@pytest.mark.parametrize("rel", DEPTH)
def test_recipe_model_is_the_release_preset(rel):
    """The model of every depth recipe is presets.RELEASE_L (the preset of both
    released depth models) with its training criterion (defaults filled on both
    sides)."""
    from ldpt.models.l_dpt.l_dpt_v1m1_base import LatentDPT
    from ldpt.models.l_dpt.presets import RELEASE_L, get_preset

    defaults = {
        k: p.default
        for k, p in inspect.signature(LatentDPT.__init__).parameters.items()
        if p.default is not inspect.Parameter.empty and k != "criterion"
    }

    def full(model):
        model = {k: v for k, v in model.items() if k not in ("type", "criterion")}
        return json.loads(json.dumps(dict(defaults, **model)))

    model = plain(load_recipe(rel))["model"]
    assert model["criterion"]["head"] == "exp"
    assert full(model) == full(RELEASE_L)
    for name in ("l-dpt_large_depth_zeroshot", "l-dpt_large_depth_mixed"):
        assert full(get_preset(name)) == full(model)


@pytest.mark.parametrize("rel, size", [(ZS, "large")])
def test_recipe_decoder_is_the_variants_table(rel, size):
    """The depth decoder is the large row of the paper's variants table; the table's
    parameter count (millions) is the decoder without the 3x3 readout convolution
    (unpatchfy: 2,359,552)."""
    keys, paper_millions = DECODER_SIZES[size]
    cfg = load_recipe(rel)
    model_cfg = plain(cfg)["model"]
    assert {k: model_cfg[k] for k in keys} == keys
    assert model_cfg["dec_num_layers"] == DECODER_LAYERS[size]
    with torch.device("meta"):
        model = build_model(cfg.model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    readout = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith("unpatchfy.")
    )
    assert params == RECIPE[rel].params
    assert round((params - readout) / 1e6) == paper_millions


# ---------------------------------------------------------------- segmentation recipes


@pytest.mark.parametrize(
    "voc_rel, coco_rel", [(voc, coco) for voc, (_, coco) in VOC_1_RECIPES.items()]
)
def test_recipe_voc_is_cocostuff_on_voc(voc_rel, coco_rel):
    """VOC = the COCO-Stuff recipe with a 21-class head initialised
    from the COCO-Stuff head, the VOC data, 20 epochs; the COCO-Stuff recipe keeps every
    epoch's decoder weights and reads COCO-Stuff with the official label mapping."""
    coco, voc = plain(load_recipe(coco_rel)), plain(load_recipe(voc_rel))
    for split in ("train", "val"):
        assert coco["data"][split]["type"] == "COCOStuffDataset"
        assert coco["data"][split]["label_mapping"] == "official"
        assert voc["data"][split]["type"] == "VOC2012Dataset"
    assert coco["num_classes"] == coco["model"]["num_classes"] == 171
    assert voc["num_classes"] == voc["model"]["num_classes"] == 21
    for key in ("num_classes", "data", "epoch", "checkpoint", "hooks"):
        coco.pop(key), voc.pop(key)
    coco["model"].pop("num_classes"), voc["model"].pop("num_classes")
    assert diff(voc, coco) == []
    loader = plain(load_recipe(voc_rel))["hooks"][0]
    assert loader["seg_head_rows"] == dict(
        classes="voc2012_from_cocostuff", source="dummy"
    )
    (saver,) = [
        h for h in plain(load_recipe(coco_rel))["hooks"] if "Saver" in h["type"]
    ]
    assert saver["keep_epoch_weights"] is True


def test_recipe_cocostuff_1_keys():
    """COCO-Stuff recipe 1: the large segmentation decoder (2048 / 16 heads /
    (2, 2, 2, 2) blocks, attention at 1024, RoPE "separate") with QK-norm (temperature
    cap 200), tap BatchNorms and 64 pixel features; cross-entropy + Lovasz; MuonAdamW
    (lr 5e-4, weight decay 0.1, every other option at its default) with the frozen
    encoder's group at lr / 10 and a no-weight-decay group (one OneCycleLR peak per
    group); 25 epochs at batch 32 on COCO-Stuff with the official label mapping."""
    one = plain(load_recipe(COCO_1))
    model = one["model"]
    assert {
        k: model[k]
        for k in (
            "task",
            "dec_dim",
            "dec_num_heads",
            "dec_num_layers",
            "dec_attn_downscale",
            "dec_ffn_ratio",
            "dec_pos_embed_rope_normalize_coords",
            "out_dim",
            "dec_qk_norm",
            "dec_qk_tau_max",
            "tap_bn",
            "num_classes",
            "precision",
        )
    } == dict(
        task="semseg",
        dec_dim=2048,
        dec_num_heads=16,
        dec_num_layers=[2, 2, 2, 2],
        dec_attn_downscale=2,
        dec_ffn_ratio=4.0,
        dec_pos_embed_rope_normalize_coords="separate",
        out_dim=64,
        dec_qk_norm=True,
        dec_qk_tau_max=200.0,
        tap_bn=True,
        num_classes=171,
        precision="bf16",
    )
    assert [c["type"] for c in model["criterion"]] == ["ce_loss", "lovasz_loss"]
    assert one["optimizer"] == dict(type="MuonAdamW", lr=5e-4, weight_decay=0.1)
    assert one["param_dicts"] == [
        dict(keyword="encoder", lr=5e-5),
        dict(keyword=["norm", "tap_bn", "bias", "gamma"], weight_decay=0.0),
    ]
    assert one["scheduler"] == dict(
        type="OneCycleLR",
        max_lr=[5e-4, 5e-5, 5e-4],
        pct_start=0.05,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1000.0,
    )
    assert (one["epoch"], one["batch_size"], one["clip_grad"]) == (25, 32, 1.0)
    for split in ("train", "val"):
        assert one["data"][split]["type"] == "COCOStuffDataset"
        assert one["data"][split]["label_mapping"] == "official"


@pytest.mark.parametrize(
    "anneal, recipe, start, peaks",
    [(COCO_1A, COCO_1, "epoch_9.pth", [4e-4, 4e-5, 4e-4])],
)
def test_recipe_cocostuff_anneal(anneal, recipe, start, peaks):
    """A COCO-Stuff anneal = its recipe for one epoch from the recipe's epoch file
    (hook 0 loads DINOv3 + the decoder, every model key from them), the same groups at
    lower peaks, OneCycleLR pct_start 0.03."""
    base = plain(load_recipe(recipe))
    loader = dict(
        type="CheckpointLoader",
        keywords=["module.", "module."],
        replacement=["module.encoder.", "module."],
        allow_missing=False,
    )
    param_dicts = [dict(g) for g in base["param_dicts"]]
    param_dicts[0]["lr"] = peaks[1]
    changes = dict(
        checkpoint=[DINOV3, f"{save_path(recipe)}/model/{start}"],
        epoch=1,
        base_lr=peaks[0],
        optimizer=dict(lr=peaks[0]),
        param_dicts=param_dicts,
        scheduler=dict(max_lr=peaks, pct_start=0.03),
        hooks=[loader] + base["hooks"][1:],
    )
    assert diff(plain(load_recipe(anneal)), merged(changes, base)) == []
    assert base["hooks"][5]["keep_epoch_weights"] is True


# ---------------------------------------------------------------- VOC recipe 1


@pytest.mark.parametrize("rel", list(VOC_1_RECIPES))
def test_recipe_voc_1_is_cocostuff_1_on_voc(rel):
    """VOC recipe 1 = COCO-Stuff recipe 1 (model, MuonAdamW and its groups,
    lr, schedule shape, batch, transforms, loss) with only these keys changed: the VOC
    data (data/voc2012, no label mapping), 21 classes (model, evaluator), 20 epochs,
    the recipe-1 anneal average as the start, a loader that reads the
    COCO-Stuff head into the VOC rows with allow_missing=False (every model key from the
    two files), and a saver without the per-epoch files."""
    _, coco = VOC_1_RECIPES[rel]
    anneal = {COCO_1: COCO_1A}[coco]
    assert RECIPE[rel].start == anneal and RECIPE[anneal].start == coco
    coco1, voc1 = plain(load_recipe(coco)), plain(load_recipe(rel))
    expected = dict(coco1, num_classes=21, epoch=20)
    expected["checkpoint"] = [DINOV3, f"{save_path(anneal)}/model/average.pth"]
    expected["model"] = dict(coco1["model"], num_classes=21)
    expected["data"] = dict(coco1["data"])
    for split in ("train", "val"):
        d = {k: v for k, v in coco1["data"][split].items() if k != "label_mapping"}
        expected["data"][split] = dict(
            d, type="VOC2012Dataset", data_root="data/voc2012"
        )
    hooks = [dict(h) for h in coco1["hooks"]]
    assert hooks[0] == dict(
        type="CheckpointLoader", keywords="module.", replacement="module.encoder."
    )
    hooks[0] = dict(
        type="CheckpointLoader",
        keywords=["module.", "module.seg_head"],
        replacement=["module.encoder.", "module.dummy"],
        seg_head_rows=dict(classes="voc2012_from_cocostuff", source="dummy"),
        allow_missing=False,
    )
    assert hooks[4] == dict(type="SemSegEvaluator", num_classes=171, ignore_index=-1)
    hooks[4] = dict(hooks[4], num_classes=21)
    assert hooks[5].pop("keep_epoch_weights") is True
    expected["hooks"] = hooks
    assert diff(voc1, expected) == []
