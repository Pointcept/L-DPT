"""
scripts/highres_infer.py (batch native-resolution inference) and scripts/demo_depth.py
on CPU: two tiny release files (tiny DINOv3 encoder from its own file; a sigmoid and an
exp head) on a small synthetic image. Both scripts write the colours and the 16-bit PNG
of ldpt/utils/depth_image.py (the demo's).
"""

import importlib.util
import json
import logging
import os
import shutil

import cv2
import numpy as np
import pytest
import torch

from helpers import REPO, DINOv3TinyTest, init_encoder_, randomize_decoder_

from ldpt.models.builder import build_model
from ldpt.models.l_dpt import api, convert, weights
from ldpt.models.l_dpt.weights import read_checkpoint
from ldpt.utils.depth_image import ToneCurve, colorize, read_depth16, tone_curve

TINY = dict(
    encoder=dict(type="dinov3_tiny_test"),
    encoder_inter_layers=[0, 1, 2, 3],
    dec_dim=32,
    dec_num_heads=2,
    dec_num_layers=(1, 2, 1, 1),
    dec_pos_embed_rope_jitter_coords=1.2,
    dec_pos_embed_rope_rescale_coords=8,
)
HEADS = {"sig": "sigmoid", "exp": "exp"}
# the L-DPT depth loss (head "exp"): the criterion's head must match the model's
CONFIGS = dict(
    sig=dict(TINY, head="sigmoid"),
    exp=dict(
        TINY,
        head="exp",
        criterion=dict(
            type="relative_depth_loss",
            align="roe",
            head="exp",
            label_types=dict(A=["scene"]),
        ),
    ),
)
HW = (75, 110)  # padded to 80 x 112 inside the model: 35 tokens, 3 chunks of 16
LOAD = dict(precision="bf16", selection_impl="stream", chunk_tokens=16)
CROP = ("0.25,0.5,32", (28, 38, 32))  # spec, (x0, y0, side) on HW
META_KEYS = {
    "output_format",
    "model",
    "options",
    "image",
    "load_seconds",
    "inference_seconds",
    "peak_gpu_allocated_gb",
    "process_max_rss_gb",
    "depth_stats",
    "depth16",
    "color",
    "crops",
}


def load_script(name):
    path = os.path.join(REPO, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hi_res = load_script("highres_infer")
demo = load_script("demo_depth")


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    """Encoder file, per head a run dir (config.py, model/epoch_3.pth) and its release
    file, and a synthetic PNG with distinct channels."""
    root = tmp_path_factory.mktemp("highres")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LDPT_CACHE", str(root / "cache"))
        encoder = init_encoder_(DINOv3TinyTest(), seed=0)
        encoder_file = root / "dinov3_tiny.pth"
        torch.save(encoder.state_dict(), encoder_file)
        runs, releases = {}, {}
        for seed, (name, head) in enumerate(HEADS.items()):
            cfg = CONFIGS[name]
            model = build_model(dict(type="L-DPT-v1m1", **cfg))
            model.encoder.load_state_dict(encoder.state_dict())
            randomize_decoder_(model, seed=seed)
            run = root / f"run_{name}"
            (run / "model").mkdir(parents=True)
            (run / "config.py").write_text(
                f"model = dict(type='L-DPT-v1m1', **{cfg!r})\n"
            )
            decoder = {
                f"module.{k}": v
                for k, v in model.state_dict().items()
                if not k.startswith("encoder.")
            }
            torch.save(dict(epoch=3, state_dict=decoder), run / "model" / "epoch_3.pth")
            releases[name] = root / "release" / f"{name}.pth"
            convert.convert(
                run / "model" / "epoch_3.pth",
                releases[name],
                config=dict(type="L-DPT-v1m1", **cfg),
            )
            runs[name] = run
    h, w = HW
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[..., 0] = np.linspace(0, 255, w)[None, :]  # red: left -> right
    rgb[..., 1] = np.linspace(0, 255, h)[:, None]  # green: top -> bottom
    rgb[..., 2] = 40  # flat blue (JPEG subsamples chroma noise)
    images = root / "images"
    images.mkdir()
    cv2.imwrite(str(images / "scene.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return dict(
        root=root,
        encoder=encoder_file,
        runs=runs,
        releases=releases,
        image=images / "scene.png",
        images=images,
        rgb=rgb,
    )


@pytest.fixture(autouse=True)
def cache(assets, monkeypatch):
    monkeypatch.setenv("LDPT_CACHE", str(assets["root"] / "cache"))


def argv(assets, out, *extra, releases=None):
    releases = assets["releases"] if releases is None else releases
    models = [f"--model={n}={p}" for n, p in releases.items()]
    return models + [
        "--image",
        str(assets["image"]),
        "--out",
        str(out),
        "--encoder-weights",
        str(assets["encoder"]),
        "--device",
        "cpu",
        "--chunk-tokens",
        "16",
        "--crop",
        CROP[0],
        *extra,
    ]


def reference(assets, name):
    model = api.load(assets["releases"][name], assets["encoder"], device="cpu", **LOAD)
    return api.predict(model, assets["rgb"], size=None)


def jpeg(bgr):
    """``bgr`` through the JPEG codec at the script's quality (deterministic): what
    reading back a file written from exactly these pixels gives."""
    params = [cv2.IMWRITE_JPEG_QUALITY, hi_res.JPEG_QUALITY]
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(bgr), params)
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def read(path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


@pytest.fixture(scope="module")
def result(assets, tmp_path_factory):
    """One main() run: the output folder."""
    out = tmp_path_factory.mktemp("out")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LDPT_CACHE", str(assets["root"] / "cache"))
        assert hi_res.main(argv(assets, out)) == 0
    return out


def test_outputs(assets, result):
    h, w = HW
    x0, y0, s = CROP[1]
    folder = result / "scene"
    bgr = cv2.cvtColor(assets["rgb"], cv2.COLOR_RGB2BGR)
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        ["rgb_preview.jpg", "rgb_crop_0.jpg", *HEADS]
    )
    # the input pixels in OpenCV's BGR order (a channel swap would differ)
    assert np.array_equal(read(folder / "rgb_preview.jpg"), jpeg(bgr))  # small: native
    assert not np.array_equal(jpeg(bgr), jpeg(bgr[..., ::-1]))
    crop = read(folder / "rgb_crop_0.jpg")
    assert np.array_equal(crop, jpeg(bgr[y0 : y0 + s, x0 : x0 + s]))

    for name, head in HEADS.items():
        out = folder / name
        assert sorted(p.name for p in out.iterdir()) == sorted(
            ["depth16.png", "depth_color.jpg", "preview.jpg", "crop_0.jpg", "meta.json"]
        )
        meta = json.loads((out / "meta.json").read_text())
        assert META_KEYS <= meta.keys()
        assert meta["image"] == dict(
            path=str(assets["image"]), width=w, height=h, megapixels=0.008
        )
        assert meta["model"]["name"] == name and meta["model"]["head"] == head
        assert meta["model"]["weights_info"]["provenance"]["epoch"] == 3
        assert meta["model"]["weights_info"]["encoder"] == str(assets["encoder"])
        assert meta["model"]["release_config"]["head"] == head
        assert meta["options"] == dict(
            size="native",
            flip=False,
            precision="bf16",
            selection_impl="stream",
            chunk_tokens=16,
            encoder_tiles=dict(tile=192, overlap=64, rope_scale=2.0),  # the default
            rope_normalize_coords="min",
            sdpa_backends=None,  # "auto" on the CPU: PyTorch's order
            device="cpu",
        )
        assert meta["peak_gpu_allocated_gb"] is None  # CPU
        assert meta["inference_seconds"] > 0 and meta["load_seconds"] > 0
        assert [(c["x"], c["y"], c["side"]) for c in meta["crops"]] == [CROP[1]]

        # the demo's 16-bit PNG (minimum to maximum) against api.predict, within half
        # a level; lo / hi in its text chunk and in meta.json
        depth = reference(assets, name)
        assert depth.std() > 0 and depth.min() > 0
        back, info = read_depth16(out / "depth16.png")
        assert meta["depth16"] == dict(info, file="depth16.png")
        lo, hi = info["lo"], info["hi"]
        assert (lo, hi) == (float(depth.min()), float(depth.max())) and hi > lo
        assert info["mapping"] == "minmax"
        v = read(out / "depth16.png")
        assert v.dtype == np.uint16 and v.shape == (h, w)
        assert v.min() == 0 and v.max() == 65535
        err = np.abs(back - depth).max()
        assert err <= (hi - lo) / 65535 / 2 + 1e-6 * hi  # + the float32 decode

        # the demo's colours (the tone curve, Spectral, near blue), written in BGR;
        # the curve in meta.json re-colours the depth
        ref = colorize(depth, bgr=True)
        color = meta["color"]
        assert color["curve"] == tone_curve(depth).as_dict()
        assert np.array_equal(
            colorize(depth, bgr=True, curve=ToneCurve(**color["curve"])), ref
        )
        assert (color["scale"], color["percentiles"], color["near"]) == (
            "tone curve",
            [1.0, 99.0],
            "blue",
        )
        assert (meta["color"]["file"], meta["color"]["format"]) == (
            "depth_color.jpg",
            "jpeg",
        )
        assert np.array_equal(read(out / "depth_color.jpg"), jpeg(ref))
        assert np.array_equal(read(out / "preview.jpg"), jpeg(ref))  # small: native
        crop = read(out / "crop_0.jpg")
        assert np.array_equal(crop, jpeg(ref[y0 : y0 + s, x0 : x0 + s]))


def test_demo_depth(assets, tmp_path, monkeypatch, capsys):
    """scripts/demo_depth.py: native resolution by default, the colour PNG of the
    demo's colorize and the float32 depth (.npy), both paths printed."""
    loads = []
    real_load = api.load

    def tiny_load(name, **kwargs):  # the preset's weights stand-in: a tiny file
        loads.append((name, kwargs))
        return real_load(assets["releases"]["exp"], assets["encoder"], device="cpu")

    monkeypatch.setattr(api, "load", tiny_load)
    image = tmp_path / "photo.png"
    shutil.copy(assets["image"], image)
    demo.main([str(image)])
    assert loads[0][0] == "l-dpt_large_depth_zeroshot"
    depth = np.load(tmp_path / "photo_depth.npy")
    model = real_load(assets["releases"]["exp"], assets["encoder"], device="cpu")
    assert depth.dtype == np.float32 and depth.shape == HW
    assert np.array_equal(depth, api.predict(model, assets["rgb"], size=None))
    color = read(tmp_path / "photo_depth.png")
    assert np.array_equal(color, colorize(depth, bgr=True))
    out = capsys.readouterr().out.split("\n")
    assert out[0].startswith(f"{tmp_path / 'photo_depth.png'}  colour, 110x75")
    assert out[1].startswith(f"{tmp_path / 'photo_depth.npy'}  float32 depth")

    demo.main([str(image), "--size", "auto", "-o", str(tmp_path / "d.png")])
    depth = np.load(tmp_path / "d.npy")
    assert np.array_equal(depth, api.predict(model, assets["rgb"]))  # short side 768
    for bad in (["--size", "0"], ["--cmap", "jet"]):
        with pytest.raises(SystemExit):
            demo.parse_args([str(image), *bad])

    # api.predict's encoder tiling by default, --no-encoder-tiles: one encoder pass
    tiles, real_predict = [], api.predict

    def recording(*args, **kwargs):
        tiles.append(kwargs["encoder_tiles"])
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(api, "predict", recording)
    demo.main([str(image), "-o", str(tmp_path / "t.png")])
    demo.main([str(image), "--no-encoder-tiles", "-o", str(tmp_path / "n.png")])
    assert tiles == ["auto", None]
    assert np.array_equal(np.load(tmp_path / "n.npy"), np.load(tmp_path / "t.npy"))

    def out_of_memory(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(api, "predict", out_of_memory)  # native: a hint, no traceback
    with pytest.raises(
        SystemExit, match=r"native resolution \(110 x 75\).*--size auto"
    ):
        demo.main([str(image)])
    with pytest.raises(torch.cuda.OutOfMemoryError):  # resized: as raised
        demo.main([str(image), "--size", "auto"])


def test_geometry():
    assert hi_res.preview(np.zeros((1000, 3000, 3), np.uint8)).shape == (683, 2048, 3)
    assert hi_res.preview(np.zeros((4096, 100), np.uint16)).shape == (2048, 50)
    # default crops: 1024 px, centred at the middle and the thirds lines
    boxes = [c.box(12000, 8000) for c in hi_res.DEFAULT_CROPS]
    assert boxes == [(5488, 3488, 1024), (5488, 2155, 1024), (5488, 4821, 1024)]
    # shrunk / shifted into a small image
    assert [c.box(110, 75) for c in hi_res.DEFAULT_CROPS] == [(18, 0, 75)] * 3
    assert hi_res.Crop(0.9, 0.9, 32).box(110, 75) == (78, 43, 32)


def test_colour_png_above_the_jpeg_limit(assets, tmp_path, monkeypatch):
    """A side above JPEG's limit (65,500 px in libjpeg / OpenCV) writes the colorized
    depth as a lossless PNG, named in meta.json; previews and crops stay JPEG."""
    bgr = np.zeros((2, hi_res.JPEG_MAX_SIDE, 3), np.uint8)
    assert cv2.imencode(".jpg", bgr)[0]
    assert not cv2.imencode(".jpg", np.zeros((2, hi_res.JPEG_MAX_SIDE + 1, 3)))[0]

    monkeypatch.setattr(hi_res, "JPEG_MAX_SIDE", HW[1] - 1)  # 110 px wide: too wide
    releases = {"sig": assets["releases"]["sig"]}
    assert hi_res.main(argv(assets, tmp_path, releases=releases)) == 0
    out = tmp_path / "scene" / "sig"
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["depth16.png", "depth_color.png", "preview.jpg", "crop_0.jpg", "meta.json"]
    )
    color = json.loads((out / "meta.json").read_text())["color"]
    assert (color["file"], color["format"], color["jpeg_quality"]) == (
        "depth_color.png",
        "png",
        None,
    )
    ref = colorize(reference(assets, "sig"), bgr=True)
    assert np.array_equal(read(out / "depth_color.png"), ref)  # lossless
    assert np.array_equal(read(out / "preview.jpg"), jpeg(ref))


def test_nonfinite_depth_fails_the_pair(assets, tmp_path, monkeypatch, caplog):
    """Policy: a non-finite depth value fails the pair (exit 1) before any depth file
    is written and without a meta.json, so a rerun retries it."""
    real_predict = api.predict

    def nan_predict(model, image, **kwargs):
        depth = real_predict(model, image, **kwargs)
        depth[3, 4] = np.nan
        return depth

    monkeypatch.setattr(api, "predict", nan_predict)
    caplog.set_level(logging.INFO, logger="highres_infer")
    releases = {"sig": assets["releases"]["sig"]}
    assert hi_res.main(argv(assets, tmp_path, releases=releases)) == 1
    assert "scene.png / sig: 1 non-finite depth values" in caplog.text
    assert list((tmp_path / "scene" / "sig").iterdir()) == []
    monkeypatch.setattr(api, "predict", real_predict)
    assert hi_res.main(argv(assets, tmp_path, releases=releases)) == 0
    assert (tmp_path / "scene" / "sig" / "meta.json").is_file()


def test_reset_peak_initialises_cuda_first(monkeypatch):
    """torch.cuda.reset_peak_memory_stats(cuda:N) fails in a fresh process ("did you
    call init?"): CUDA is initialised first, and both call sites go through it."""
    import inspect

    calls = []
    monkeypatch.setattr(torch.cuda, "init", lambda: calls.append("init"))
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device=None: calls.append(("reset", str(device))),
    )
    hi_res._reset_peak(torch.device("cuda:1"))
    assert calls == ["init", ("reset", "cuda:1")]
    hi_res._reset_peak(torch.device("cpu"))
    assert len(calls) == 2
    for fn in (hi_res.run_model, hi_res.run_image):
        source = inspect.getsource(fn)
        assert "_reset_peak(device)" in source and "reset_peak_memory" not in source


def test_skip_existing(assets, result, monkeypatch, caplog):
    """A pair with a matching meta.json is skipped (a model with nothing to do is not
    loaded); --overwrite redoes it."""
    tree = {p: p.stat().st_mtime_ns for p in result.rglob("*") if p.is_file()}
    loads = []
    real_load = api.load

    def counting_load(path, *args, **kwargs):
        loads.append(os.path.basename(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(api, "load", counting_load)
    caplog.set_level(logging.INFO, logger="highres_infer")
    assert hi_res.main(argv(assets, result)) == 0
    assert loads == [] and "skip scene.png / sig" in caplog.text
    assert {p: p.stat().st_mtime_ns for p in tree} == tree

    (result / "scene" / "exp" / "meta.json").unlink()
    assert hi_res.main(argv(assets, result)) == 0
    assert loads == ["exp.pth"]
    meta = result / "scene" / "sig" / "meta.json"
    assert meta.stat().st_mtime_ns == tree[meta]

    assert hi_res.main(argv(assets, result, "--overwrite")) == 0
    assert loads == ["exp.pth", "sig.pth", "exp.pth"]
    assert meta.stat().st_mtime_ns != tree[meta]
    assert not list(result.rglob(".*tmp*"))  # no temporary file left


def test_skip_needs_a_matching_meta(assets, result, tmp_path, monkeypatch, caplog):
    """A pair is redone when its meta.json differs from the release file (provenance
    without the conversion time, config), the depth options or the crops."""
    out = tmp_path / "out"
    shutil.copytree(result, out)
    loads = []
    real_load = api.load

    def counting_load(path, *args, **kwargs):
        loads.append(os.path.basename(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(api, "load", counting_load)
    caplog.set_level(logging.INFO, logger="highres_infer")
    metas = {n: out / "scene" / n / "meta.json" for n in HEADS}
    for path in metas.values():  # fields outside the signature do not count
        meta = json.loads(path.read_text())
        meta["peak_host_rss_gb"] = meta.pop("process_max_rss_gb")
        del meta["depth16"]["mapping"], meta["color"]["format"]
        path.write_text(json.dumps(meta))
    assert hi_res.main(argv(assets, out)) == 0 and loads == []

    assert hi_res.main(argv(assets, out, "--flip")) == 0
    assert loads == ["sig.pth", "exp.pth"]
    assert "redo scene.png / sig: meta.json differs in options" in caplog.text
    assert json.loads(metas["sig"].read_text())["options"]["flip"] is True
    assert hi_res.main(argv(assets, out, "--flip")) == 0 and len(loads) == 2

    # the same checkpoint converted again (a new conversion time): up to date; with
    # another config under the same name: redone
    src = {n: assets["runs"][n] / "model" / "epoch_3.pth" for n in HEADS}
    releases = {n: tmp_path / "again" / f"{n}.pth" for n in HEADS}
    convert.convert(src["sig"], releases["sig"], config=CONFIGS["sig"])
    exp_bound = dict(CONFIGS["exp"], dec_qk_bound=1000.0)
    convert.convert(src["exp"], releases["exp"], config=exp_bound)
    assert hi_res.main(argv(assets, out, "--flip", releases=releases)) == 0
    assert loads[2:] == ["exp.pth"]
    assert "redo scene.png / exp: meta.json differs in release_config" in caplog.text
    meta = json.loads(metas["exp"].read_text())
    assert meta["model"]["release_config"]["dec_qk_bound"] == 1000.0

    # every other option that changes the outputs, and a broken meta.json
    args = hi_res.parse_args(argv(assets, out, "--flip", releases=releases))
    signature = hi_res.pair_signature(read_checkpoint(releases["sig"]), args)
    assert hi_res.stale_reason(metas["sig"], signature) is None
    for extra, key in (
        ("--size=512", "options"),
        ("--precision=fp32", "options"),
        ("--selection-impl=concat", "options"),
        ("--chunk-tokens=none", "options"),
        ("--no-encoder-tiles", "options"),
        ("--crop=0.1,0.1,8", "crops"),
    ):
        args = hi_res.parse_args(argv(assets, out, "--flip", extra, releases=releases))
        other = hi_res.pair_signature(read_checkpoint(releases["sig"]), args)
        assert hi_res.stale_reason(metas["sig"], other) == f"meta.json differs in {key}"
    # a meta.json without output_format is redone
    meta = json.loads(metas["sig"].read_text())
    del meta["output_format"]
    metas["sig"].write_text(json.dumps(meta))
    reason = hi_res.stale_reason(metas["sig"], signature)
    assert reason == "meta.json differs in output_format"
    metas["sig"].write_text("{")
    assert hi_res.stale_reason(metas["sig"], signature).startswith("unreadable")
    assert hi_res.stale_reason(out / "nope.json", signature) == "no meta.json"


def test_presets_and_positional_images(assets, tmp_path, monkeypatch):
    """--model takes a preset (its pinned file, fetched: here a tiny stand-in),
    NAME=PRESET or a file named after its stem; without --model the default preset
    runs; images may be positional."""
    fetched = []

    def fake_fetch(name, **kwargs):
        fetched.append(name)
        return assets["releases"]["sig"]

    monkeypatch.setattr(weights, "fetch", fake_fetch)
    common = [
        "--out",
        str(tmp_path),
        "--encoder-weights",
        str(assets["encoder"]),
        "--device",
        "cpu",
        "--chunk-tokens",
        "16",
        "--crop",
        CROP[0],
    ]
    assert hi_res.main([str(assets["image"]), *common]) == 0
    assert fetched == ["l-dpt_large_depth_zeroshot"]
    meta = tmp_path / "scene" / "l-dpt_large_depth_zeroshot" / "meta.json"
    assert json.loads(meta.read_text())["model"]["release_file"] == str(
        assets["releases"]["sig"]
    )
    exp = assets["releases"]["exp"]
    args = hi_res.parse_args(
        [str(assets["image"]), "--model", "zs=l-dpt_large_depth_zeroshot"]
        + ["--model", "l-dpt_large_depth_mixed", "--model", str(exp), *common]
    )
    assert args.models == [
        ("zs", "l-dpt_large_depth_zeroshot"),
        ("l-dpt_large_depth_mixed", "l-dpt_large_depth_mixed"),
        ("exp", exp),
    ]
    assert args.images == [assets["image"]]
    for bad in (["--model", "nope"], ["--model", "a=nope"], [str(tmp_path / "x.jpg")]):
        with pytest.raises(SystemExit) as exc:
            hi_res.parse_args([str(assets["image"]), *common, *bad])
        assert exc.value.code == 2


def test_fails_loudly(assets, tmp_path):
    good = argv(assets, tmp_path)
    dots = tmp_path / "..png"  # stem "."
    shutil.copy(assets["image"], dots)
    for bad in (
        ["--model", "x=" + str(tmp_path / "missing.pth")],
        ["--image", str(tmp_path / "missing.jpg")],
        ["--encoder-weights", str(tmp_path / "missing.pth")],
        ["--crop", "0.5,0.5"],
        ["--model", "a/b=" + str(assets["releases"]["sig"])],
        ["--model", "..=" + str(assets["releases"]["sig"])],  # would write into OUT/
        ["--model", ".=" + str(assets["releases"]["sig"])],
        ["--image", str(dots)],
    ):
        with pytest.raises(SystemExit) as exc:
            hi_res.main(good + bad)
        assert exc.value.code == 2
    dev = assets["runs"]["sig"] / "model" / "epoch_3.pth"
    with pytest.raises(SystemExit, match="not an L-DPT release file"):
        hi_res.main(good + ["--model", f"dev={dev}"])
    assert not (tmp_path / "scene").exists()
