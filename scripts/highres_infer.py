"""
Native-resolution relative depth of large photos with one or more L-DPT models.

    python scripts/highres_infer.py photo1.jpg photo2.jpg --out out/
    python scripts/highres_infer.py --model l-dpt_large_depth_zeroshot \
        --model l-dpt_large_depth_mixed --images-dir photos/ --out out/

--model takes a preset (its weights downloaded from Hugging Face on first use, or read
from $LDPT_CKPT_LARGE_DEPTH_ZEROSHOT / $LDPT_CKPT_LARGE_DEPTH_MIXED, sha256 checked;
default l-dpt_large_depth_zeroshot), NAME=PRESET, or NAME=RELEASE_FILE (a file written
by python -m ldpt.models.l_dpt.convert); NAME (for a preset, its name) is the output
folder. The DINOv3 weights come from --encoder-weights or $LDPT_DINOV3. Each model is
loaded once (api.load with the high-res memory options: bf16 decoder, selection_impl
"stream", chunk_tokens 32768; a 102 MP image peaks at 33 GiB (36 GB in meta.json) of
GPU memory), runs every image, and is freed before the next. An image with a side above
3072 px runs the frozen encoder on overlapping 3072 px windows and the decoder on the
whole latent grid in one pass (api.predict's default encoder_tiles, recorded in
meta.json); --no-encoder-tiles runs the encoder once on the whole image. Per image and
model, OUT/<image stem>/<model>/ gets
    depth16.png      uint16, depth mapped linearly from its minimum to its maximum (the
                     demo's file, ldpt/utils/depth_image.py):
                     depth = lo + value / 65535 * (hi - lo), lo / hi in its "l-dpt"
                     text chunk and in meta.json
    depth_color.jpg  full resolution, the demo's colours (near blue, far red, on the
                     tone curve fitted to the image: ToneCurve);
                     depth_color.png when a side exceeds JPEG's 65,500 pixels
    preview.jpg      the colorized depth, long side 2048 (smaller images: native)
    crop_<k>.jpg     native-resolution crops of the colorized depth
    meta.json        image size, model and release-file provenance, options, timings,
                     peak GPU memory, the 16-bit range and the tone curve; written last
and OUT/<image stem>/ gets rgb_preview.jpg and rgb_crop_<k>.jpg once. Depth is relative
(larger = farther, up to an affine map; an "exp" head gives positive unbounded values).
A non-finite depth value fails the pair before any depth file is written (exit 1, no
meta.json: a rerun retries it). meta.json's process_max_rss_gb is the process's peak
resident set so far (ru_maxrss), the memory-mapped checkpoint pages included.

An (image, model) pair is skipped unless --overwrite when its meta.json matches the
release file (provenance without the conversion time, config), --size, --flip,
--precision, --selection-impl, --chunk-tokens, the encoder tiling, the crops and the
output files' format (OUTPUT_FORMAT); a model whose images are
all done is not loaded. Files are written under a temporary name and
renamed, so parallel processes (one model each) can share OUT. Images are read with
OpenCV's cv2.IMREAD_COLOR, which applies the EXIF orientation (as a photo viewer shows
it), and converted to RGB in place. Host memory per 100 MP image stays at a few GB: the
float32 depth (400 MB), its uint16 copy and the uint8 colorized image are the only
full-resolution arrays; the colormap and the uint16 mapping run in row blocks.
"""

import argparse
import datetime
import gc
import json
import logging
import os
import re
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# PyTorch's allocator with expandable segments reserves less GPU memory beyond the peak it
# allocates at native resolution (set before torch initialises CUDA; a value already in
# the environment wins)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

log = logging.getLogger("highres_infer")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp")
DEFAULT_MODEL = "l-dpt_large_depth_zeroshot"
# format of the output files (depth16.png, the colours of ldpt/utils/depth_image.py);
# a pair whose meta.json states another output_format, or none, is redone
OUTPUT_FORMAT = "l-dpt-highres/4"
PREVIEW_SIDE = 2048
JPEG_QUALITY = 90
JPEG_MAX_SIDE = 65500  # libjpeg's largest side; a longer colorized depth is a PNG
# default crops: squares of DEFAULT_CROP_SIDE native pixels centred on the horizontal
# middle at the vertical centre and the upper / lower thirds lines (different depths
# in most photos); --crop replaces them
DEFAULT_CROP_SIDE = 1024
DEFAULT_CROP_CENTRES = ((0.5, 0.5), (0.5, 1 / 3), (0.5, 2 / 3))
NAME = re.compile(r"^[A-Za-z0-9._-]+$")  # a folder name; "." and ".." are refused
# meta.json options a skipped pair must match (device, the RoPE default and the
# attention order do not count)
SKIP_OPTIONS = (
    "size",
    "flip",
    "precision",
    "selection_impl",
    "chunk_tokens",
    "encoder_tiles",
)


class Crop(NamedTuple):
    """A square crop: (fx, fy) is its top-left corner (or its centre) as fractions of
    the image width / height, ``side`` its side in native pixels."""

    fx: float
    fy: float
    side: int
    centred: bool = False

    def box(self, width: int, height: int) -> Tuple[int, int, int]:
        """(x0, y0, side) in pixels, shrunk and shifted to lie inside the image."""
        side = max(1, min(self.side, width, height))
        x0 = self.fx * width - (side / 2 if self.centred else 0)
        y0 = self.fy * height - (side / 2 if self.centred else 0)
        x0 = min(max(int(round(x0)), 0), width - side)
        y0 = min(max(int(round(y0)), 0), height - side)
        return x0, y0, side


DEFAULT_CROPS = tuple(
    Crop(fx, fy, DEFAULT_CROP_SIDE, centred=True) for fx, fy in DEFAULT_CROP_CENTRES
)


# ---- arguments


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "images",
        nargs="*",
        metavar="IMAGE",
        help="input images (or --image, --images-dir)",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="PRESET | NAME=PRESET | NAME=RELEASE_FILE",
        help=f"a released model (default {DEFAULT_MODEL}) or a release file (python "
        f"-m ldpt.models.l_dpt.convert), NAME its output folder (default: the "
        f"preset's name, the file's stem); repeatable",
    )
    parser.add_argument(
        "--image", action="append", default=[], metavar="PATH", help="repeatable"
    )
    parser.add_argument(
        "--images-dir", help=f"every {'/'.join(IMAGE_EXTENSIONS)} file in it (sorted)"
    )
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument(
        "--encoder-weights",
        default=os.environ.get("LDPT_DINOV3"),
        help="official DINOv3 file or Hugging Face (transformers-format) folder "
        "(default: $LDPT_DINOV3)",
    )
    parser.add_argument("--device", default="cuda", help="cuda (default), cuda:N, cpu")
    parser.add_argument(
        "--size",
        default="native",
        help="'native' (default) or the short side the image is resized to (the "
        "depth is resized back to the native size)",
    )
    parser.add_argument("--flip", action="store_true", help="flip test-time average")
    parser.add_argument(
        "--no-encoder-tiles",
        dest="encoder_tiles",
        action="store_const",
        const=None,
        default="auto",
        help="run the encoder once on the whole image (default: overlapping 3072 px "
        "windows when a side exceeds 3072 px)",
    )
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument(
        "--selection-impl", default="stream", choices=("stream", "concat")
    )
    parser.add_argument(
        "--chunk-tokens",
        default="32768",
        help="token chunk of the encoder blocks / decoder FFN (default 32768; "
        "'none' or 0: off)",
    )
    parser.add_argument(
        "--crop",
        action="append",
        default=[],
        metavar="X,Y,W",
        help="top-left corner (fractions of width / height) and side (native pixels) "
        "of a square crop; repeatable (default: three 1024 px crops centred at the "
        "middle and the upper / lower thirds lines)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="redo pairs with a meta.json"
    )
    args = parser.parse_args(argv)

    from ldpt.models.l_dpt.presets import PRESETS, WEIGHTS

    # the depth presets with a weight slot
    depth_presets = [n for n in WEIGHTS if PRESETS[n]["task"] == "relative_depth"]
    args.models = []  # (output folder name, preset name or release file Path)
    for spec in args.model or [DEFAULT_MODEL]:
        name, sep, value = spec.partition("=")
        if not sep:  # a preset (named after itself) or a file (after its stem)
            value = spec
            name = spec if spec in depth_presets else Path(spec).stem
        if not NAME.match(name) or name in (".", "..") or not value:
            parser.error(
                f"--model {spec!r}: expected PRESET "
                f"({', '.join(sorted(depth_presets))}), "
                f"NAME=PRESET or NAME=RELEASE_FILE, NAME [A-Za-z0-9._-] (not . or ..)"
            )
        if name in [n for n, _ in args.models]:
            parser.error(f"--model: name {name!r} given twice")
        if value in depth_presets:
            args.models.append((name, value))
            continue
        path = Path(value).expanduser()
        if not path.is_file():
            what = "no such file" if sep else "not a preset or a file"
            parser.error(f"--model {name}: {what}: {path}")
        args.models.append((name, path))

    images = [Path(p).expanduser() for p in args.images + args.image]
    if args.images_dir:
        folder = Path(args.images_dir).expanduser()
        if not folder.is_dir():
            parser.error(f"--images-dir: no such folder: {folder}")
        images += sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not images:
        parser.error("no images: pass IMAGE, --image or --images-dir")
    missing = [str(p) for p in images if not p.is_file()]
    if missing:
        parser.error(f"no such image: {', '.join(missing)}")
    unique, stems = [], {}
    for p in images:
        if p.resolve() in [q.resolve() for q in unique]:
            continue
        if p.stem in (".", ".."):
            parser.error(f"{p}: the stem {p.stem!r} cannot be an output folder")
        if p.stem in stems:
            parser.error(f"two images with the stem {p.stem!r}: {stems[p.stem]}, {p}")
        stems[p.stem] = p
        unique.append(p)
    args.images = unique

    if args.encoder_weights and not Path(args.encoder_weights).expanduser().exists():
        parser.error(
            f"--encoder-weights: no such file or folder: {args.encoder_weights}"
        )

    if args.size == "native":
        args.size = None
    else:
        try:
            args.size = int(args.size)
            assert args.size > 0
        except (ValueError, AssertionError):
            parser.error(
                f"--size: expected native or a positive int, got {args.size!r}"
            )
    if args.chunk_tokens.lower() in ("none", "0"):
        args.chunk_tokens = None
    else:
        try:
            args.chunk_tokens = int(args.chunk_tokens)
            assert args.chunk_tokens > 0
        except (ValueError, AssertionError):
            parser.error(
                f"--chunk-tokens: expected an int or none, got {args.chunk_tokens!r}"
            )

    crops = []
    for spec in args.crop:
        try:
            fx, fy, side = spec.split(",")
            crop = Crop(float(fx), float(fy), int(side))
            assert 0 <= crop.fx <= 1 and 0 <= crop.fy <= 1 and crop.side > 0
        except (ValueError, AssertionError):
            parser.error(f"--crop {spec!r}: expected X,Y,W (X, Y in [0, 1], W > 0 px)")
        crops.append(crop)
    args.crops = tuple(crops) or DEFAULT_CROPS
    return args


# ---- depth -> images


def preview(image: np.ndarray, side: int = PREVIEW_SIDE) -> np.ndarray:
    """``image`` with its long side reduced to ``side`` (area interpolation); a
    smaller image is returned as is."""
    import cv2

    h, w = image.shape[:2]
    s = side / max(h, w)
    if s >= 1:
        return image
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def imwrite(path: Path, image: np.ndarray, params: Sequence[int] = ()) -> None:
    """cv2.imwrite (BGR or gray) under a temporary name, then renamed: a reader or a
    parallel writer never sees a partial file. Raises when OpenCV cannot write."""
    import cv2

    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    if not cv2.imwrite(str(tmp), image, list(params)):
        raise OSError(f"cv2.imwrite failed: {path}")
    os.replace(tmp, path)


def write_bytes(path: Path, data: bytes) -> None:
    """``data`` into ``path`` under a temporary name, then renamed (as imwrite)."""
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def read_rgb(path: Path) -> np.ndarray:
    """HxWx3 uint8 RGB; cv2.IMREAD_COLOR applies the EXIF orientation."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"OpenCV cannot read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)  # in place


# ---- run


def _jpeg() -> List[int]:
    import cv2

    return [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]


def _sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device) -> None:
    """Peak GPU memory counted from here. CUDA is initialised first: with an explicit
    index (cuda:N) nothing has done it yet, and the reset fails in a fresh process."""
    import torch

    if device.type == "cuda":
        torch.cuda.init()  # no-op once initialised
        torch.cuda.reset_peak_memory_stats(device)


def _max_rss_gb() -> float:
    """The process's peak resident set so far (Linux ru_maxrss, KB): file-backed pages
    of the memory-mapped checkpoints count, so it is not the per-image host memory."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9


def _code_commit() -> Optional[str]:
    path = Path(REPO) / "COMMIT"  # optional: a code snapshot may record its commit here
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def write_rgb(image: np.ndarray, folder: Path, crops, overwrite: bool) -> None:
    """rgb_preview.jpg and rgb_crop_<k>.jpg of the RGB ``image`` (when missing)."""
    import cv2

    h, w = image.shape[:2]
    path = folder / "rgb_preview.jpg"
    if overwrite or not path.exists():
        imwrite(path, cv2.cvtColor(preview(image), cv2.COLOR_RGB2BGR), _jpeg())
    for k, crop in enumerate(crops):
        path = folder / f"rgb_crop_{k}.jpg"
        if overwrite or not path.exists():
            x0, y0, s = crop.box(w, h)
            patch = cv2.cvtColor(image[y0 : y0 + s, x0 : x0 + s], cv2.COLOR_RGB2BGR)
            imwrite(path, patch, _jpeg())


def run_image(model, name, path: Path, args, base_meta: Dict[str, Any]) -> None:
    """One (image, model) pair: depth -> files -> meta.json (last)."""
    import cv2
    import torch

    from ldpt.models.l_dpt import api
    from ldpt.utils.depth_image import (
        DISPLAY_CMAP,
        DISPLAY_PERCENTILES,
        colorize,
        depth16_png,
        tone_curve,
    )

    folder = Path(args.out) / path.stem
    out = folder / name
    out.mkdir(parents=True, exist_ok=True)
    device = api._model_device(model)

    t0 = time.perf_counter()
    image = read_rgb(path)
    h, w = image.shape[:2]
    write_rgb(image, folder, args.crops, args.overwrite)
    read_s = time.perf_counter() - t0
    log.info("%s / %s: %d x %d (%.1f MP)", path.name, name, w, h, w * h / 1e6)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    _reset_peak(device)
    _sync(device)
    t0 = time.perf_counter()
    depth = api.predict(
        model,
        image,
        size=args.size,
        flip=args.flip,
        encoder_tiles=args.encoder_tiles,
    )
    _sync(device)
    infer_s = time.perf_counter() - t0
    peak = reserved = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        reserved = torch.cuda.max_memory_reserved(device) / 1e9
    del image
    log.info("%s / %s: inference %.1f s, peak %s GB", path.name, name, infer_s, peak)

    t0 = time.perf_counter()
    nonfinite = int(depth.size - np.count_nonzero(np.isfinite(depth)))
    if nonfinite:
        # fail the pair (no meta.json is written, so a rerun retries it)
        raise RuntimeError(f"{path.name} / {name}: {nonfinite} non-finite depth values")
    stats = dict(min=float(depth.min()), max=float(depth.max()))
    stats["mean"] = float(depth.mean(dtype=np.float64))

    png, depth16 = depth16_png(depth)
    write_bytes(out / "depth16.png", png)
    del png
    curve = tone_curve(depth)  # anchored at DISPLAY_PERCENTILES
    color = colorize(depth, DISPLAY_CMAP, bgr=True, curve=curve)
    del depth
    if max(h, w) <= JPEG_MAX_SIDE:
        color_file, color_format = "depth_color.jpg", "jpeg"
        imwrite(out / color_file, color, _jpeg())
    else:
        color_file, color_format = "depth_color.png", "png"
        imwrite(out / color_file, color, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    imwrite(out / "preview.jpg", preview(color), _jpeg())
    crops = []
    for k, crop in enumerate(args.crops):
        x0, y0, s = crop.box(w, h)
        imwrite(out / f"crop_{k}.jpg", color[y0 : y0 + s, x0 : x0 + s], _jpeg())
        crops.append(
            dict(file=f"crop_{k}.jpg", x=x0, y=y0, side=s, spec=crop._asdict())
        )
    del color
    write_s = time.perf_counter() - t0

    meta = dict(
        base_meta,
        image=dict(path=str(path), width=w, height=h, megapixels=round(w * h / 1e6, 3)),
        inference_seconds=round(infer_s, 3),
        read_seconds=round(read_s, 3),
        write_seconds=round(write_s, 3),
        peak_gpu_allocated_gb=None if peak is None else round(peak, 3),
        peak_gpu_reserved_gb=None if reserved is None else round(reserved, 3),
        process_max_rss_gb=round(_max_rss_gb(), 3),
        depth_stats=dict(stats, nonfinite=nonfinite),
        depth16=dict(depth16, file="depth16.png"),
        color=dict(
            file=color_file,
            format=color_format,
            cmap=DISPLAY_CMAP,
            scale="tone curve",
            curve=curve.as_dict(),  # depths lo / hi, blend w: ToneCurve(**curve)
            percentiles=list(DISPLAY_PERCENTILES),
            near="blue",
            jpeg_quality=JPEG_QUALITY if color_format == "jpeg" else None,
        ),
        preview=dict(file="preview.jpg", long_side=PREVIEW_SIDE),
        crops=crops,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    )
    write_json(out / "meta.json", meta)
    log.info("%s / %s: written in %.1f s", path.name, name, write_s)


def _identity(provenance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A release file's provenance without its conversion time (the same weights
    converted again count as the same)."""
    return {k: v for k, v in (provenance or {}).items() if k != "converted"}


def encoder_tiles_option(args) -> Optional[Dict[str, Any]]:
    """The encoder tiling of the run as meta.json stores it: the EncoderTiles fields of
    api.predict's default, or None (--no-encoder-tiles: one encoder pass)."""
    import dataclasses

    from ldpt.models.l_dpt import api

    if args.encoder_tiles is None:
        return None
    return dataclasses.asdict(api.ENCODER_TILES)


def pair_signature(ckpt: Dict[str, Any], args) -> Dict[str, Any]:
    """What a pair's meta.json must hold to be skipped, as meta.json stores it: the
    release file's identity and config, the SKIP_OPTIONS, the crop specs, the format
    of the output files."""
    signature = dict(
        output_format=OUTPUT_FORMAT,
        provenance=_identity(ckpt.get("provenance")),
        release_config=ckpt.get("config"),
        options=dict(
            size="native" if args.size is None else args.size,
            flip=args.flip,
            precision=args.precision,
            selection_impl=args.selection_impl,
            chunk_tokens=args.chunk_tokens,
            encoder_tiles=encoder_tiles_option(args),
        ),
        crops=[crop._asdict() for crop in args.crops],
    )
    return json.loads(json.dumps(signature, default=str))  # write_json's conversion


def stale_reason(meta_path: Path, signature: Dict[str, Any]) -> Optional[str]:
    """None when ``meta_path`` exists and matches ``signature``, else why the pair runs."""
    try:
        meta = json.loads(meta_path.read_text())
        model, options = meta["model"], meta["options"]
        found = dict(
            output_format=meta.get("output_format"),
            provenance=_identity(model["weights_info"].get("provenance")),
            release_config=model.get("release_config"),
            options={k: options.get(k) for k in SKIP_OPTIONS},
            crops=[crop.get("spec") for crop in meta.get("crops", [])],
        )
    except FileNotFoundError:
        return "no meta.json"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        return f"unreadable meta.json ({type(e).__name__})"
    differ = [k for k in signature if found.get(k) != signature[k]]
    return f"meta.json differs in {', '.join(differ)}" if differ else None


def run_model(name: str, release: Path, images: List[Path], args) -> List[str]:
    """Load one model, run ``images``, free it; returns the failed image names."""
    import torch

    from ldpt.models.l_dpt import api
    from ldpt.models.l_dpt.weights import read_checkpoint

    device = torch.device(args.device)
    _reset_peak(device)
    _sync(device)
    t0 = time.perf_counter()
    model = api.load(
        release,
        encoder_weights=args.encoder_weights,
        device=device,
        precision=args.precision,
        selection_impl=args.selection_impl,
        chunk_tokens=args.chunk_tokens,
    )
    _sync(device)
    load_s = time.perf_counter() - t0
    model_gb = (
        torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else None
    )
    log.info(
        "%s: loaded %s in %.1f s (%s GB on %s)", name, release, load_s, model_gb, device
    )
    order = api.auto_order(device)  # api.predict's attention order (None: PyTorch's)
    base_meta = dict(
        output_format=OUTPUT_FORMAT,
        model=dict(
            name=name,
            release_file=str(release),
            weights_info=model.weights_info,
            release_config=read_checkpoint(release).get("config"),
            head=getattr(model, "head", None),
        ),
        options=dict(
            size="native" if args.size is None else args.size,
            flip=args.flip,
            precision=args.precision,
            selection_impl=args.selection_impl,
            chunk_tokens=args.chunk_tokens,
            encoder_tiles=encoder_tiles_option(args),
            rope_normalize_coords=model.rope.normalize_coords,
            sdpa_backends=None if order is None else list(order),
            device=str(device),
        ),
        load_seconds=round(load_s, 3),
        gpu_model_gb=None if model_gb is None else round(model_gb, 3),
        device_name=(
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        torch=torch.__version__,
        code_commit=_code_commit(),
    )
    failed = []
    for path in images:
        try:
            run_image(model, name, path, args, base_meta)
        except Exception:  # noqa: BLE001  (logged; the other images still run)
            log.error("%s / %s FAILED:\n%s", path.name, name, traceback.format_exc())
            failed.append(f"{path.name} / {name}")
        gc.collect()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return failed


def main(argv=None) -> int:
    args = parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            stream=sys.stdout,
        )
    log.setLevel(logging.INFO)
    from ldpt.models.l_dpt.convert import FORMAT, FORMATS
    from ldpt.models.l_dpt.weights import fetch, read_checkpoint

    models = []  # (name, release file)
    for name, source in args.models:  # a preset: its pinned, sha256-checked file
        try:
            models.append((name, fetch(source) if isinstance(source, str) else source))
        except (OSError, ValueError, RuntimeError) as e:
            raise SystemExit(f"--model {name}: {type(e).__name__}: {e}") from None
    signatures = {}
    for name, release in models:  # every file checked before the first load
        ckpt = read_checkpoint(release)
        if ckpt.get("format") not in FORMATS:
            raise SystemExit(
                f"--model {name}: {release} is not an L-DPT release file ({FORMAT}): "
                f"python -m ldpt.models.l_dpt.convert SRC DST --config <run>/config.py"
            )
        task = (ckpt.get("config") or {}).get("task", "relative_depth")
        if task != "relative_depth":
            raise SystemExit(
                f"--model {name}: {release} is a {task} model; this script writes "
                f"depth (a class map of one photo: scripts/demo_segment.py)"
            )
        signatures[name] = pair_signature(ckpt, args)
        del ckpt
    Path(args.out).mkdir(parents=True, exist_ok=True)
    log.info(
        "%d model(s) x %d image(s) -> %s; crops %s",
        len(models),
        len(args.images),
        args.out,
        [tuple(c) for c in args.crops],
    )
    failed = []
    for name, release in models:
        todo = []
        for p in args.images:
            meta = Path(args.out) / p.stem / name / "meta.json"
            reason = None if args.overwrite else stale_reason(meta, signatures[name])
            if args.overwrite or reason is not None:
                todo.append(p)
                if reason not in (None, "no meta.json"):
                    log.info("redo %s / %s: %s", p.name, name, reason)
            else:
                log.info(
                    "skip %s / %s: meta.json matches (--overwrite redoes it)",
                    p.name,
                    name,
                )
        if not todo:
            log.info("%s: every image done, not loaded", name)
            continue
        try:
            failed += run_model(name, release, todo, args)
        except Exception:  # noqa: BLE001  (a failed load: the other models still run)
            log.error("%s FAILED:\n%s", name, traceback.format_exc())
            failed += [f"{p.name} / {name}" for p in todo]
            gc.collect()
    if failed:
        log.error("%d failed: %s", len(failed), "; ".join(failed))
        return 1
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
