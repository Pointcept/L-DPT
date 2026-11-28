"""
Semantic segmentation through the inference API on CPU: the COCO-Stuff preset and its
weight slot, convert (also of an averaged checkpoint) and load of a tiny segmentation
model, api.predict against SemSegEvaluator (the same class map, with and without the
flip TTA) and its one encoder pass by default (encoder_tiles="auto"), the class-chunked
resize, ldpt.utils.seg_image and scripts/demo_segment.py.
"""

import importlib.util
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from helpers import REPO, DINOv3TinyTest, init_encoder_, randomize_decoder_

from ldpt.models.builder import build_model
from ldpt.models.l_dpt import api, convert, presets, weights
from ldpt.models.l_dpt.presets import (
    CLASS_NAMES,
    INPUT_SIZE,
    PRESETS,
    SEMSEG_L,
    WEIGHTS,
    get_preset,
)
from ldpt.utils import seg_image

SEG = "l-dpt_large_semseg_cocostuff"
NUM_CLASSES = 7
TINY_SEG = dict(
    encoder=dict(type="dinov3_tiny_test"),
    encoder_inter_layers=[0, 1, 2, 3],
    task="semseg",
    num_classes=NUM_CLASSES,
    out_dim=8,
    dec_dim=32,
    dec_num_heads=2,
    dec_num_layers=(1, 2, 1, 1),
    dec_attn_downscale=1,
    dec_pos_embed_rope_normalize_coords="separate",
    dec_pos_embed_rope_rescale_coords=2,
    precision="bf16",
    attn_dtype="fp32",
    dec_qk_norm=True,
    tap_bn=True,
)
NAMES = tuple(f"class{i}" for i in range(NUM_CLASSES))
HW = (50, 70)


def load_script(name):
    path = os.path.join(REPO, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LDPT_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def tiny_seg(monkeypatch):
    """A preset of the tiny segmentation model (registered for the test): its name."""
    name = "tiny-seg-test"
    monkeypatch.setitem(presets.PRESETS, name, dict(TINY_SEG, criterion=False))
    monkeypatch.setitem(presets.INPUT_SIZE, name, 64)
    monkeypatch.setitem(presets.CLASS_NAMES, name, NAMES)
    return name


def seg_checkpoint(path, seed=0, epoch=3):
    """A trainer-style checkpoint of the tiny segmentation model (random decoder and
    tap BatchNorm statistics, the encoder kept out): (model, path)."""
    model = build_model(dict(type="L-DPT-v1m1", **TINY_SEG, criterion=False))
    init_encoder_(model.encoder, seed=0)
    randomize_decoder_(model, seed=seed)
    g = torch.Generator().manual_seed(100 + seed)
    for bn in model.tap_bn:
        bn.running_mean.copy_(0.1 * torch.randn(bn.running_mean.shape, generator=g))
        bn.running_var.copy_(0.5 + torch.rand(bn.running_var.shape, generator=g))
        bn.num_batches_tracked.fill_(10 * epoch)
    state = {
        f"module.{k}": v
        for k, v in model.state_dict().items()
        if not k.startswith("encoder.")
    }
    torch.save(
        dict(epoch=epoch, state_dict=state, optimizer=dict(state={}, param_groups=[])),
        path,
    )
    return model.eval(), path


@pytest.fixture
def encoder_file(tmp_path):
    path = tmp_path / "dinov3_tiny.pth"
    torch.save(init_encoder_(DINOv3TinyTest(), seed=0).state_dict(), path)
    return path


@pytest.fixture
def seg_release(tiny_seg, encoder_file, tmp_path):
    """The tiny model's training checkpoint and its release file (with its preset):
    (model, checkpoint, release file)."""
    model, src = seg_checkpoint(tmp_path / "epoch_3.pth")
    dst = tmp_path / f"{tiny_seg}.pth"
    convert.convert(src, dst, preset=tiny_seg)
    return model, src, dst


def rgb_image(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (*HW, 3), dtype=np.uint8)


# ---- the preset


def test_semseg_preset():
    """The COCO-Stuff preset: the segmentation model of DINOv3 ViT-7B taps, decoder
    2048 / (2, 2, 2, 2) with cosine attention and tap BatchNorm, 64 features, 171
    classes, RoPE per axis; inference only; short side 768; the dataset's class
    names in train-id order; its weight slot in the release repo."""
    assert PRESETS[SEG] is SEMSEG_L
    kwargs = get_preset(SEG)
    assert kwargs == SEMSEG_L and kwargs is not SEMSEG_L
    with torch.device("meta"):
        model = build_model(dict(type="L-DPT-v1m1", **kwargs))
    assert model.task == "semseg" and model.criterion is None
    assert (model.num_classes, model.out_dim, model.dec_dim) == (171, 64, 2048)
    assert model.rope.normalize_coords == "separate"
    assert model.tap_bn is not None and len(model.tap_bn) == 4
    assert all(blk.attn.qk_norm for stage in model.dec_blocks for blk in stage)
    assert model.seg_head.weight.shape == (171, 64, 1, 1)
    assert INPUT_SIZE[SEG] == 768

    from ldpt.datasets.cocostuff import COCOStuffDataset
    from ldpt.datasets.meta import COCOSTUFF_CLASS_NAMES

    assert COCOSTUFF_CLASS_NAMES is seg_image.COCOSTUFF_CLASS_NAMES
    assert list(CLASS_NAMES[SEG]) == COCOStuffDataset.CLASS_NAMES
    assert len(CLASS_NAMES[SEG]) == 171 == len(set(CLASS_NAMES[SEG]))
    assert CLASS_NAMES[SEG][0] == "person" and CLASS_NAMES[SEG][80] == "banner"

    source = WEIGHTS[SEG]
    assert source.repo == "pointcept-bot/L-DPT" and source.file == f"{SEG}.pth"
    assert source.env == "LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF"
    assert source.format == "release"


def test_semseg_weights_pinned(monkeypatch, tmp_path):
    """Release gate: the segmentation file is pinned as the depth files are (a commit
    of the release repo, sha256, size), since the README and docs/inference.md describe
    it as released. Fails until presets.WEIGHTS[SEG] is filled (PENDING there); a local
    copy named by the variable must then have the pinned sha256."""
    source = WEIGHTS[SEG]
    assert source.published, (
        f"PENDING: pin the revision, sha256 and size of {source.file} in "
        "presets.WEIGHTS (the docs describe the file as released)"
    )
    assert len(source.revision) == 40 and len(source.sha256) == 64
    assert source.size > 3_000_000_000
    local = tmp_path / source.file
    local.write_bytes(b"x")
    monkeypatch.setenv(source.env, str(local))
    with pytest.raises(RuntimeError, match="not the pinned"):
        weights.fetch(SEG)
    assert weights.fetch(SEG, verify=False) == local


# ---- convert and load


def test_convert_and_load(seg_release, tiny_seg, encoder_file, tmp_path):
    """convert --preset: decoder-only state (tap BatchNorm statistics included), the
    preset's kwargs without the loss; load_release gives them back bitwise; api.load by
    path and with the preset (weights= the release file or the training checkpoint)
    builds the same model: RoPE per axis as in the file, the preset's class names,
    bitwise the trained model on a non-square image."""
    model, src, dst = seg_release
    config, state, info = convert.load_release(dst)
    assert info["preset"] == tiny_seg and "averaged" not in info["provenance"]
    assert config == convert.release_kwargs(PRESETS[tiny_seg])
    decoder = {k: v for k, v in model.state_dict().items() if not k.startswith("enc")}
    assert state.keys() == decoder.keys()
    assert all(torch.equal(state[k], v) for k, v in decoder.items())
    assert "tap_bn.0.running_var" in state and "seg_head.weight" in state

    image = torch.randn(1, 3, 48, 80, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = model.infer(image)
    loads = [
        api.load(dst, encoder_file, device="cpu"),
        api.load(tiny_seg, encoder_file, device="cpu", weights=dst),
        api.load(tiny_seg, encoder_file, device="cpu", weights=src),
    ]
    for loaded in loads:
        assert loaded.preset == tiny_seg and loaded.class_names == list(NAMES)
        assert loaded.rope.normalize_coords == "separate"
        assert loaded.criterion is None and not loaded.training
        with torch.no_grad():
            assert torch.equal(loaded.infer(image), ref)
    chosen = api.load(
        dst, encoder_file, device="cpu", dec_pos_embed_rope_normalize_coords="min"
    )
    assert chosen.rope.normalize_coords == "min"
    # another architecture is refused: the decoder keys and shapes are checked
    with pytest.raises(convert.ConvertError, match="missing"):
        convert.convert(src, tmp_path / "x.pth", preset=SEG)
    # a file without a preset name has no class names
    bare = tmp_path / "bare.pth"
    convert.convert(src, bare, config=dict(type="L-DPT-v1m1", **TINY_SEG))
    assert api.load(bare, encoder_file, device="cpu").class_names is None


def test_convert_averaged_checkpoint(tiny_seg, tmp_path):
    """scripts/average_weights.py of two checkpoints, then convert: the release state
    is the average evaluate.py --average evaluates (floats averaged, the last file's
    num_batches_tracked), and the provenance lists the inputs (names, sha256) and
    weights of OUTPUT.avg.json."""
    from ldpt.engines.evaluate import average_states

    _, a = seg_checkpoint(tmp_path / "start.pth", seed=1, epoch=9)
    _, b = seg_checkpoint(tmp_path / "end.pth", seed=2, epoch=1)
    avg = tmp_path / "average.pth"
    assert (
        load_script("average_weights").main(
            ["--inputs", str(a), str(b), "--output", str(avg)]
        )
        == 0
    )
    dst = tmp_path / "release.pth"
    provenance = convert.convert(avg, dst, preset=tiny_seg)
    assert provenance["averaged"] == dict(
        sources=[
            dict(source="start.pth", sha256=weights.sha256_of(a)),
            dict(source="end.pth", sha256=weights.sha256_of(b)),
        ],
        weights=[0.5, 0.5],
    )
    assert provenance["source"] == "average.pth" and provenance["epoch"] == 1
    _, state, info = convert.load_release(dst)
    assert info["provenance"]["averaged"] == provenance["averaged"]
    expected = average_states([str(a), str(b)])
    assert state.keys() == expected.keys()
    assert all(torch.equal(state[k], v) for k, v in expected.items())
    assert int(state["tap_bn.0.num_batches_tracked"]) == 10


# ---- predict


def evaluator_hist(model, rgb, labels, tta):
    """SemSegEvaluator.eval_step on ``rgb`` through the COCO-Stuff val pipeline, with
    ``labels`` as the ground truth: its confusion matrix [C, C]."""
    from ldpt.datasets.transforms import Compose
    from ldpt.hooks.evaluator import SemSegEvaluator
    from ldpt.utils.config import Config

    cfg = Config.fromfile(
        os.path.join(REPO, "configs/cocostuff/semseg-l-dpt-v1m1-1-large.py")
    )
    data = Compose(cfg.data.val.transform)(
        dict(
            image=rgb.astype(np.float32) / 255.0,
            segment=labels.astype(np.int32),
            name="cocostuff/x",
        )
    )
    batch = dict(
        image=data["image"].as_subclass(torch.Tensor)[None],
        segment=data["segment"].as_subclass(torch.Tensor)[None],
        name=[data["name"]],
    )
    assert batch["image"].shape[-2:] == (768, 1075)  # short side 768
    assert batch["segment"].shape == (1, 1, *HW)  # labels at the image size
    evaluator = SemSegEvaluator(num_classes=NUM_CLASSES, tta=tta, tta_batched=False)
    evaluator.trainer = SimpleNamespace(
        model=model, comm_info={}, cfg=SimpleNamespace(log_interval=10)
    )
    evaluator.eval_before()
    with torch.inference_mode():
        evaluator.eval_step(batch, step=1)
    ((_, hist),) = evaluator.trainer.comm_info["semseg_results"]
    return hist


@pytest.mark.parametrize("flip", [False, True])
def test_predict_is_the_evaluators_class_map(seg_release, encoder_file, flip):
    """api.predict (short side 768) gives the class map SemSegEvaluator scores on the
    COCO-Stuff val pipeline, pixel for pixel: with the predicted map as the ground
    truth, the evaluator's confusion matrix is diagonal over every pixel (single
    scale, and the flip TTA: the mean of the softmaxes)."""
    _, _, dst = seg_release
    model = api.load(dst, encoder_file, device="cpu")
    rgb = rgb_image()
    classes = api.predict(model, rgb, size=768, flip=flip)
    assert classes.shape == HW and classes.dtype == np.int64
    assert 0 <= classes.min() and classes.max() < NUM_CLASSES
    assert len(np.unique(classes)) >= 3  # a non-trivial map
    hist = evaluator_hist(model, rgb, classes, [["flip"]] if flip else None)
    assert int(hist.sum()) == HW[0] * HW[1]
    assert torch.equal(hist, torch.diag(torch.diag(hist)))
    # the input types: a float image, a tensor in -> a tensor out
    as_float = api.predict(model, rgb.astype(np.float32) / 255.0, size=768, flip=flip)
    assert np.array_equal(as_float, classes)
    tensor = api.predict(model, torch.from_numpy(rgb), size=768, flip=flip)
    assert tensor.dtype == torch.int64 and np.array_equal(tensor.numpy(), classes)


def test_predict_sizes(seg_release, encoder_file):
    """size="auto" is the preset's INPUT_SIZE (the tiny preset's 64 here); an int
    resizes the short side; size=None runs at the native resolution with the high-res
    memory options for the call (restored after it); the class map is at the image's
    size either way."""
    _, _, dst = seg_release
    model = api.load(dst, encoder_file, device="cpu")
    rgb = rgb_image(1)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
    mean = torch.tensor(presets.IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(presets.IMAGE_STD).view(1, 3, 1, 1)
    with torch.no_grad():
        small = model.infer((api.resize_short_side(x, 64) - mean) / std)
        model.selection_impl, model.chunk_tokens = "stream", 32768
        native = model.infer((x - mean) / std)
        model.selection_impl, model.chunk_tokens = "concat", None
    assert small.shape[-2:] == (64, 89) and native.shape[-2:] == HW
    want_small = api.class_map(small, HW)[0].numpy()
    assert np.array_equal(api.predict(model, rgb), want_small)
    assert np.array_equal(api.predict(model, rgb, size=64), want_small)
    native_map = api.predict(model, rgb, size=None)
    assert np.array_equal(native_map, native.argmax(dim=1)[0].numpy())
    assert (model.selection_impl, model.chunk_tokens) == ("concat", None)


def test_predict_encoder_tiles_auto_is_one_pass(seg_release, encoder_file, monkeypatch):
    """encoder_tiles="auto" (the default) runs a segmentation model with one encoder
    pass at any size (model.infer gets None; in the windows the classes of large
    regions fragment), on both size paths and both passes of flip; an EncoderTiles,
    api.ENCODER_TILES too, reaches model.infer as for depth, and a small tiling tiles
    this small image."""
    _, _, dst = seg_release
    model = api.load(dst, encoder_file, device="cpu")
    rgb = rgb_image(2)
    small = api.EncoderTiles(tile=2, overlap=1, rope_scale=1.0)
    infer, seen = model.infer, []

    def recording(x, encoder_tiles=None):
        seen.append(encoder_tiles)
        return infer(x, encoder_tiles=encoder_tiles)

    monkeypatch.setattr(model, "infer", recording)
    for size in ("auto", None):
        plain = api.predict(model, rgb, size=size, flip=True, encoder_tiles=None)
        for value, want in (
            ("default", None),
            ("auto", None),
            (False, None),
            (api.ENCODER_TILES, api.ENCODER_TILES),
            (small, small),
        ):
            seen.clear()
            kwargs = {} if value == "default" else dict(encoder_tiles=value)
            got = api.predict(model, rgb, size=size, flip=True, **kwargs)
            assert seen == [want, want], (size, value)
            if want is not small:  # one encoder pass (ENCODER_TILES: below its tile)
                assert np.array_equal(got, plain), (size, value)
    tiled = api.predict(model, rgb, size=None, encoder_tiles=small)
    assert tiled.shape == HW and not np.array_equal(tiled, plain)
    # the rule: "auto" is the windows for depth only
    assert api._encoder_tiles("auto", segment=True) is None
    assert api._encoder_tiles("auto") is api.ENCODER_TILES
    assert api._encoder_tiles(small, segment=True) is small


@pytest.mark.parametrize("batch", [1, 2])
def test_class_map_chunks(batch):
    """class_map: the evaluator's bilinear resize + argmax, also a few classes at a
    time (each class resizes on its own: the same values; ties to the lowest id) and
    at the size of the scores."""
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(batch, 9, 24, 32, generator=g)
    scores[:, 5] = scores[:, 2]  # ties: the lower id wins
    scores[:, 8] = scores[:, 3]
    size = (240, 333)
    ref = F.interpolate(scores, size=size, mode="bilinear", align_corners=False)
    ref = ref.argmax(dim=1)
    per_class = batch * size[0] * size[1]
    for max_elements in (1, 3 * per_class, 8 * per_class, api.CLASS_MAP_ELEMENTS):
        got = api.class_map(scores, size, max_elements=max_elements)
        assert got.dtype == torch.int64 and torch.equal(got, ref), max_elements
    assert not (ref == 5).any() and not (ref == 8).any() and (ref == 2).any()
    assert torch.equal(api.class_map(scores, (24, 32)), scores.argmax(dim=1))


# ---- seg_image


def test_palette_and_colorize():
    colors = seg_image.palette(256)
    assert colors.shape == (256, 3) and colors.dtype == np.uint8
    assert len({tuple(c) for c in colors}) == 256 and not (colors == 0).all(1).any()
    # PASCAL VOC colour map entries 1, 2, 3, 15 (class c: entry c + 1)
    assert colors[0].tolist() == [128, 0, 0] and colors[1].tolist() == [0, 128, 0]
    assert colors[2].tolist() == [128, 128, 0] and colors[14].tolist() == [
        192,
        128,
        128,
    ]
    assert np.array_equal(seg_image.palette(171), colors[:171])
    classes = np.array([[0, 1, -1], [2, 170, 0]], np.int64)
    rgb = seg_image.colorize(classes, 171)
    assert rgb.shape == (2, 3, 3) and rgb.dtype == np.uint8
    assert rgb[0, 0].tolist() == [128, 0, 0] and rgb[0, 2].tolist() == [0, 0, 0]
    assert np.array_equal(rgb[1, 1], colors[170])
    assert np.array_equal(seg_image.colorize(classes, 171, bgr=True), rgb[..., ::-1])
    assert np.array_equal(seg_image.colorize(classes), rgb)  # the same fixed colours
    with pytest.raises(ValueError, match="no colour"):
        seg_image.colorize(classes, colors=colors[:100])
    with pytest.raises(ValueError, match="integer"):
        seg_image.colorize(classes.astype(np.float32))


def test_top_classes():
    classes = np.array([[3, 3, 3, 1], [1, 0, -1, 2], [2, 3, 1, -1]])
    top = seg_image.top_classes(classes, ["a", "b", "c", "d"], k=3)
    assert top == [(3, "d", 0.4), (1, "b", 0.3), (2, "c", 0.2)]
    assert seg_image.top_classes(classes, k=None)[-1] == (0, None, 0.1)
    assert seg_image.top_classes(np.full((2, 2), -1)) == []


# ---- scripts


def test_demo_segment(seg_release, encoder_file, tmp_path, monkeypatch, capsys):
    """scripts/demo_segment.py: --size auto by default (the preset's INPUT_SIZE), the
    colour PNG of seg_image.colorize, uint8 class ids (.npy), the paths and the
    largest classes printed; demo_depth.py refuses a segmentation model."""
    demo = load_script("demo_segment")
    _, _, dst = seg_release
    loads = []
    real_load = api.load

    def tiny_load(name, **kwargs):  # the preset's weights stand-in: the tiny file
        loads.append((name, kwargs))
        return real_load(dst, encoder_file, device="cpu")

    monkeypatch.setattr(api, "load", tiny_load)
    image = tmp_path / "photo.png"
    rgb = rgb_image(2)
    cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    demo.main([str(image)])
    assert loads[0][0] == SEG and loads[0][1]["weights"] is None
    model = real_load(dst, encoder_file, device="cpu")
    classes = np.load(tmp_path / "photo_seg.npy")
    assert classes.dtype == np.uint8 and classes.shape == HW
    want = api.predict(model, rgb)
    assert np.array_equal(classes, want)
    color = cv2.imread(str(tmp_path / "photo_seg.png"))
    assert np.array_equal(color, seg_image.colorize(want, NUM_CLASSES, bgr=True))
    out = capsys.readouterr().out.strip().split("\n")
    assert out[0] == f"{tmp_path / 'photo_seg.png'}  colour, 70x50"
    assert out[1] == f"{tmp_path / 'photo_seg.npy'}  uint8 class ids, 70x50"
    top = seg_image.top_classes(want, NAMES, 5)
    assert len(out) == 2 + len(top)
    assert out[2] == f"{100 * top[0][2]:5.1f} %  {top[0][1]} ({top[0][0]})"

    options = ["--size", "native", "--flip", "--top", "0", "-o"]
    demo.main([str(image), *options, str(tmp_path / "n.png")])
    native = np.load(tmp_path / "n.npy")
    assert np.array_equal(native, api.predict(model, rgb, size=None, flip=True))
    for bad in (["--size", "0"], ["--top", "-1"]):
        with pytest.raises(SystemExit):
            demo.parse_args([str(image), *bad])

    depth_demo = load_script("demo_depth")
    with pytest.raises(SystemExit, match="demo_segment.py"):
        depth_demo.main([str(image)])


def test_highres_infer_refuses_segmentation(seg_release, encoder_file, tmp_path):
    """scripts/highres_infer.py writes depth: a segmentation preset is no --model
    choice, a segmentation release file stops the run before any load."""
    hi_res = load_script("highres_infer")
    _, _, dst = seg_release
    image = tmp_path / "photo.png"
    cv2.imwrite(str(image), rgb_image())
    common = ["--image", str(image), "--out", str(tmp_path / "out"), "--device", "cpu"]
    with pytest.raises(SystemExit):
        hi_res.parse_args(["--model", SEG, *common])
    with pytest.raises(SystemExit, match="semseg model"):
        hi_res.main([f"--model=seg={dst}", *common])
    assert not (tmp_path / "out").exists()
