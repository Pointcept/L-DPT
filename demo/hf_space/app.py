"""
L-DPT relative depth: the Gradio demo, for a Hugging Face Space or a local GPU.

    python demo/hf_space/app.py                 # locally: http://127.0.0.1:7860
    python demo/hf_space/app.py --host 0.0.0.0 --port 8080 --examples photos/
    python app.py                               # in the Space

Locally it runs on the installed ldpt package (pip install -e ".[demo,app]"; from a
checkout without it, the repo's ldpt next to demo/); the Space runs on the inference
subset of the package copied next to this file (sync_ldpt.sh). It needs the release
files of models.json (downloaded from Hugging Face, or local copies:
$LDPT_CKPT_LARGE_DEPTH_ZEROSHOT, as in the API; see space_utils) and the DINOv3 ViT-7B
weights ($LDPT_DINOV3, a local copy; else Meta's gated transformers-format repo,
downloaded with $HF_TOKEN or the token of `hf auth login`: see
space_utils.resolve_dinov3). The default model is loaded before the server starts
(--lazy, or LDPT_PRELOAD=0: at the first request instead), so a missing file stops the
start with a message. On ZeroGPU (``spaces`` imports and $SPACES_ZERO_GPU is set)
every model is loaded when this module is imported, before demo.launch (they share the
DINOv3 encoder; ZeroGPU cannot load one after startup), and the inference runs in
spaces.GPU with a duration sized by the photo's megapixels: on the default slot (half a
GPU), or on the full GPU (size="xlarge") when the photo's estimated peak GPU memory is
more than the half slot holds (space_utils.zerogpu_size; no photo up to the default
105 MP). The network always runs at the photo's native resolution (up to LDPT_MAX_MP,
105 MP; api.predict's encoder windows above 3072 px), no flip; the depth is shown near
blue, far red, on the tone curve fitted to the photo. Everything but the UI is in
space_utils.py.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

# PyTorch's allocator with expandable segments reserves less GPU memory beyond the peak it
# allocates at native resolution (set before torch initialises CUDA; a value already in
# the environment wins)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if not (HERE / "ldpt").is_dir() and (HERE.parent.parent / "ldpt").is_dir():
    sys.path.insert(1, str(HERE.parent.parent))  # repo checkout without sync_ldpt.sh

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

import space_utils as su  # noqa: E402

# space_utils.read_rgb checks the size from the header before decoding; PIL's own
# limit (89 MP warning, 179 MP error in gradio's upload handling) would come first
Image.MAX_IMAGE_PIXELS = None

try:
    import spaces  # ZeroGPU
except ImportError:
    spaces = None

SERVICE: Optional[su.DepthService] = None
MAX_FILE_SIZE = "400mb"  # a 100 MP PNG upload is 150-300 MB
# a free account's daily ZeroGPU quota (Hugging Face's published quotas; PRO: 40 min)
ZEROGPU_FREE_QUOTA_S = 5 * 60
ZERO_GPU = spaces is not None and su.zero_gpu_active()
if ZERO_GPU:  # ZeroGPU: weights are loaded at startup, outside any spaces.GPU call
    SERVICE = su.DepthService.from_env()
    SERVICE.ensure_all(keep_going=True)  # a model that fails is logged and refused
    SERVICE.freeze()


def _gpu_duration(model_id: str, rgb, size: Optional[int], flip: bool) -> int:
    return su.gpu_seconds(rgb.shape[0], rgb.shape[1], size, flip)


def _full_gpu_duration(model_id: str, rgb, size: Optional[int], flip: bool) -> int:
    return su.gpu_seconds(rgb.shape[0], rgb.shape[1], size, flip, su.GPU_XLARGE)


def _predict(model_id: str, rgb, size: Optional[int], flip: bool):
    return SERVICE.predict(model_id, rgb, size, flip)


def _predict_full_gpu(model_id: str, rgb, size: Optional[int], flip: bool):
    """_predict again: spaces.GPU keeps one wrapper per function, so the full-GPU entry
    point needs a function of its own."""
    return SERVICE.predict(model_id, rgb, size, flip)


def _on_gpu(fn, duration, **size):
    """``fn`` in spaces.GPU on ZeroGPU (``size``: the slot, none for the default one),
    else ``fn`` itself: a local run never calls spaces."""
    if not ZERO_GPU:
        return fn
    try:
        return spaces.GPU(duration=duration, **size)(fn)
    except TypeError:  # a spaces version without dynamic durations
        return spaces.GPU(duration=su.GPU_SECONDS_MAX, **size)(fn)


predict_on_gpu = _on_gpu(_predict, _gpu_duration)  # the default slot, half a GPU
predict_on_full_gpu = _on_gpu(_predict_full_gpu, _full_gpu_duration, size=su.GPU_XLARGE)


def gpu_entry(height: int, width: int, size: Optional[int]):
    """The GPU entry point of a request, chosen before the GPU is requested: on ZeroGPU
    the full GPU's when su.zerogpu_size routes the photo there, else (and off ZeroGPU
    always) predict_on_gpu."""
    if ZERO_GPU and su.zerogpu_size(height, width, size) == su.GPU_XLARGE:
        return predict_on_full_gpu
    return predict_on_gpu


# The code repository, shown as the Code link of the hero
CODE_URL = "https://github.com/Pointcept/L-DPT"
LINKS = (
    ("Project", "https://xywu.me/l-dpt"),
    ("Code", CODE_URL),
    ("Weights", "https://huggingface.co/pointcept-bot/L-DPT"),
    ("Training data", "https://huggingface.co/datasets/pointcept-bot/DepthHybrid"),
)

HERO = """<div id="hero">
<h1>L-DPT</h1>
<p class="tag">Relative depth at your photo's <b>native resolution</b>,
up to {max_mp:g} MP, from a frozen DINOv3 ViT-7B/16 encoder and a latent DPT decoder
trained on 768 x 768 crops only.</p>
<p class="links">{links}</p>
</div>"""

FOOTER = """<div class="foot">Depth is relative (larger = farther, up to a scale and
a shift): near is blue, far is red, on a tone curve fitted to each photo. Downloads: a
16-bit PNG at full resolution (<code>depth = lo + value / 65535 * (hi - lo)</code>,
lo / hi in its text chunk and in the JSON) and a colour JPEG. L-DPT code and weights:
CC BY-NC 4.0.</div>"""


def hero_html(max_mp: float) -> str:
    links = " ".join(
        f'<a href="{url}" target="_blank">{text}</a>'
        for text, url in LINKS
        if url.startswith("https://")
    )
    return HERO.format(max_mp=max_mp, links=links)


def timing_html(space: bool, precomputed: bool = False) -> str:
    """The GPU-time line under the button, from the timing guide of space_utils: this
    Space's times (ZeroGPU, each size on the slot it runs on, at the size it stands
    for) and what a 100 MP photo reserves of the quota on a Space, one RTX PRO 6000's
    times and peak memory off it.
    """
    gpu = su.ZEROGPU if space else su.RTX_PRO_6000
    rows = su.zerogpu_timing_rows() if space else su.timing_rows(gpu)
    times = " · ".join(
        f"{su.format_duration(s, est)} at {mp:g} MP" + (" (estimate)" if est else "")
        for mp, s, est in rows
    )
    if space:
        hw = (10000, 10000)  # 100 MP
        reserved = su.quota_seconds(*hw, None, False)
        full = su.zerogpu_size(*hw) == su.GPU_XLARGE
        runs = "runs on a full GPU, which counts twice, and " if full else ""
        free = (
            ", more than a free account has" if reserved > ZEROGPU_FREE_QUOTA_S else ""
        )
        # a rough number, as the times (two significant digits from 100 s up)
        rough = su.format_duration(reserved, estimate=True)
        text = (
            f"<b>GPU time on this Space</b> (native resolution, no resizing): {times}. "
            f"A 100 MP photo {runs}reserves {rough} of your daily ZeroGPU quota "
            f"before it starts{free}."
        )
        if precomputed:
            text += " The examples show a precomputed result at once."
    else:
        peaks = [su.measured_mp(gpu.peak_gib, mp) for mp in su.TIMING_ROWS_MP]
        peaks = ", ".join(
            f"{gpu.peak_gib[k]:.0f} GiB at {k:g} MP" for k in peaks if k is not None
        )
        text = (
            f"<b>GPU time</b> (native resolution, no resizing) on one {gpu.name}: "
            f"{times}. Peak GPU memory allocated: {peaks}."
        )
    return f'<div class="timing">{text}</div>'


def examples_title(examples: Sequence[su.Example], space: bool) -> str:
    """The text above the gallery: what a click shows, and what a new run costs."""
    runs = [r for e in examples if e.rgb_preview for r in e.results.values()]
    if not runs:
        return "### Examples\nClick a photo to load it, then press **Estimate depth**."
    devices = {r.get("device") for r in runs}
    on = (
        f"one {devices.pop()}" if len(devices) == 1 and None not in devices else "a GPU"
    )
    text = (
        "### Examples\nOriginal photos at full resolution. Click one to load it: its "
        f"depth from our run on {on} shows at once. **Estimate depth** runs it again "
    )
    if not space:
        return text + "on your GPU."
    big = max(e.megapixels or 0 for e in examples)
    side = round(math.sqrt(big * 1e6))  # a square photo of that many pixels (the
    hw = (side, side)  # encoder windows depend on the shape, not only the size)
    if su.quota_seconds(*hw, None, False) <= ZEROGPU_FREE_QUOTA_S:
        return text + "here."
    gpu = su.zerogpu_timing(su.zerogpu_size(*hw))
    seconds, est = su.quoted_seconds(*hw, gpu)  # as the GPU-time line
    t = ("an estimated " if est else "") + su.format_duration(seconds, estimate=True)
    return (
        text + f"here; the {big:.0f} MP photos take {t} and need a Hugging Face PRO "
        "account's ZeroGPU quota."
    )


CSS = """
#hero { text-align: center; padding: 22px 8px 4px; }
#hero h1 { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; }
#hero .tag { max-width: 760px; margin: 8px auto 10px; font-size: 1.05rem;
             color: var(--body-text-color-subdued); }
#hero .links a { display: inline-block; padding: 4px 14px; border-radius: 999px;
                 border: 1px solid var(--border-color-primary); text-decoration: none; }
.timing, .foot { font-size: 0.88rem; color: var(--body-text-color-subdued); }
.foot { text-align: center; margin: 8px auto 16px; max-width: 820px; }
#run-info { min-height: 1.6em; font-size: 0.9rem; }
#examples-title h3 { margin-bottom: 0; }
"""


# Gradio 6 takes the theme and the CSS in launch(), Gradio 5 (the Space) in Blocks()
STYLE_IN_LAUNCH = int(gr.__version__.split(".")[0]) >= 6


def on_space(env=os.environ) -> bool:
    """Running in a Hugging Face Space (which sets SPACE_ID), not on a local GPU."""
    return bool(env.get("SPACE_ID"))


def style() -> Dict[str, Any]:
    """The theme and the CSS: Blocks() keyword arguments on Gradio 5, launch() ones on
    Gradio 6."""
    theme = gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate",
        font=[
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
    )
    return dict(theme=theme, css=CSS)


def build_demo(service: Optional[su.DepthService] = None) -> gr.Blocks:
    """The Blocks app (not launched); ``service`` defaults to the one loaded at import
    (ZeroGPU), else models.json + env."""
    global SERVICE
    SERVICE = service or SERVICE or su.DepthService.from_env()
    registry, settings = SERVICE.registry, SERVICE.settings
    slider = hasattr(gr, "ImageSlider")
    examples = su.resolve_examples()
    space = on_space()
    gpu = su.ZEROGPU if space else su.RTX_PRO_6000  # what the progress text expects
    precomputed_any = any(e.rgb_preview and e.results for e in examples)

    with gr.Blocks(
        title="L-DPT · relative depth at native resolution",
        delete_cache=(1800, 3600),
        **({} if STYLE_IN_LAUNCH else style()),
    ) as demo:
        gr.HTML(hero_html(settings.max_infer_mp))
        picked = gr.State(None)  # index of the loaded example (None: an upload)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320):
                image = gr.Image(
                    label="Photo",
                    type="filepath",
                    image_mode=None,  # the file as uploaded, decoded by read_rgb
                    sources=["upload", "clipboard"],
                    height=460,
                )
                single = len(registry.models) == 1
                if single:  # no choice to show: the model id rides along as state
                    model = gr.State(registry.default.id)
                else:
                    model = gr.Radio(
                        registry.choices(), value=registry.default.id, label="Model"
                    )
                about = gr.Markdown(registry.default.description)
                run = gr.Button("Estimate depth", variant="primary", size="lg")
                gr.HTML(timing_html(space, precomputed_any))
            with gr.Column(scale=7, min_width=360):
                with gr.Tabs():
                    if slider:
                        with gr.Tab("Compare"):
                            compare = [
                                gr.ImageSlider(
                                    label="Photo / depth", height=560, show_label=False
                                )
                            ]
                    with gr.Tab("Depth"):
                        depth = gr.Image(label="Depth", interactive=False, height=560)
                    if not slider:
                        with gr.Tab("Photo / depth"):
                            with gr.Row():
                                compare = [
                                    gr.Image(label="Photo", interactive=False),
                                    gr.Image(label="Depth", interactive=False),
                                ]
                info = gr.Markdown(elem_id="run-info")
                files = gr.File(
                    label="Downloads (full resolution)",
                    file_count="multiple",
                    height=120,
                )
        gallery = None
        if examples:
            gr.Markdown(examples_title(examples, space), elem_id="examples-title")
            gallery = gr.Gallery(
                [(e.thumb, e.caption) for e in examples],
                show_label=False,
                columns=7,
                height="auto",
                allow_preview=False,
                object_fit="cover",
                elem_id="examples",
            )
        gr.HTML(FOOTER)

        outputs = [depth, *compare, info, files]

        def shown(pair, depth_img, text, downloads) -> Dict[Any, Any]:
            out = {depth: depth_img, info: text, files: downloads}
            if slider:
                out[compare[0]] = pair
            else:
                out.update(zip(compare, pair if pair else (None, None)))
            return out

        def credit(ex: su.Example) -> str:
            return f" · Photo: {ex.credit}" if ex.credit else ""

        def precomputed(index, model_id) -> Optional[Dict[Any, Any]]:
            """The example's precomputed result for model_id as outputs, else None."""
            if index is None:
                return None
            ex = examples[index]
            r = ex.result(model_id)
            if r is None:
                return None
            label = registry.get(model_id).label
            mp = f"{ex.megapixels:g} MP · " if ex.megapixels else ""
            text = (
                f"Precomputed: {ex.title} · {mp}{label} · computed on "
                f"{r.get('device', 'a GPU')} (native resolution){credit(ex)}. Press "
                f"**Estimate depth** to run it here."
            )
            return shown((ex.rgb_preview, r["preview"]), r["preview"], text, None)

        def pick(model_id, evt: gr.SelectData):
            index = evt.index
            ex = examples[index]
            out = {image: ex.path, picked: index}
            named = f"{ex.title}{credit(ex)}" if ex.credit else ""
            out.update(precomputed(index, model_id) or shown(None, None, named, None))
            return out

        def on_model(model_id, index):
            out = {about: registry.get(model_id).description}
            out.update(precomputed(index, model_id) or {})
            return out

        def run_depth(image_path, model_id, progress=gr.Progress()) -> Dict[Any, Any]:
            """The photo's native resolution, no flip."""
            if not image_path:
                raise gr.Error("Upload an image first.")
            size, use_flip = None, False
            try:
                rgb = su.read_rgb(image_path, settings.max_mp)
                h, w = rgb.shape[:2]
                su.check_inference(
                    h,
                    w,
                    size,
                    settings.max_infer_mp,
                    max_gpu_seconds=settings.max_gpu_seconds,
                )
                entry = registry.get(model_id)
                if not SERVICE.is_loaded(model_id):
                    progress(0.05, desc=f"Loading {entry.label}")
                load_s = SERVICE.ensure(model_id)  # outside the GPU call (ZeroGPU)
                on_gpu = gpu_entry(h, w, size)  # the slot, before the GPU is requested
                full = on_gpu is predict_on_full_gpu
                timing = su.ZEROGPU_XLARGE if full else gpu
                # the card's and the GPU-time line's rough time near a measured size
                expect = su.format_duration(su.quoted_seconds(h, w, timing)[0], True)
                where = " on a full GPU" if full else ""
                progress(0.3, desc=f"Inference at {w}x{h}{where} (estimate {expect})")
                pred, seconds, peak = on_gpu(model_id, rgb, size, use_flip)
                progress(0.8, desc="Writing the outputs")
                result = su.render_result(
                    rgb,
                    pred,
                    entry=entry,
                    size=size,
                    flip=use_flip,
                    cmap=su.DISPLAY_CMAP,
                    seconds=seconds,
                    peak=peak,
                    settings=settings,
                    name=image_path,
                    load_seconds=load_s,
                )
            except Exception as e:  # also a ZeroGPU task's InputError (su.user_error)
                err = su.user_error(e)
                if err is None:
                    raise
                raise gr.Error(str(err)) from None
            return shown(
                (result.input_preview, result.depth_preview),
                result.depth_preview,
                result.info,
                result.files,
            )

        if gallery is not None:
            gallery.select(
                pick, [model], [image, picked, *outputs], show_progress="minimal"
            )
        image.upload(lambda: None, None, picked)
        image.clear(lambda: None, None, picked)
        if not single:
            model.change(on_model, [model, picked], [about, *outputs])
        run.click(
            run_depth,
            [image, model],
            outputs,
            api_name="depth",
        )
    demo.queue(default_concurrency_limit=1, max_size=16)
    return demo


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="The L-DPT demo (Gradio): relative depth at a photo's native "
        "resolution. Needs the DINOv3 ViT-7B weights in $LDPT_DINOV3 (docs/install.md).",
    )
    parser.add_argument(
        "--host",
        help="address to listen on (default $GRADIO_SERVER_NAME, else 127.0.0.1; "
        "0.0.0.0: every network interface)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="port (default $GRADIO_SERVER_PORT, else the first free one from 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="also open a temporary public gradio.live link",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        metavar="DIR",
        help="a folder of photos (or of make_examples.py output) for the example "
        "gallery (default $LDPT_EXAMPLES_DIR, else demo/hf_space/examples)",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="start the server at once and load the model at the first request "
        "(default: load it first; LDPT_PRELOAD=0 does the same)",
    )
    args = parser.parse_args(argv)
    if args.examples is not None and not args.examples.is_dir():
        parser.error(f"--examples: {args.examples} is not a folder")
    return args


def launch_kwargs(
    args: argparse.Namespace, settings: su.Settings, examples_dir: Path
) -> Dict[str, Any]:
    """demo.launch() arguments: the host, port and share link only when given (Gradio
    reads $GRADIO_SERVER_NAME / $GRADIO_SERVER_PORT / $GRADIO_SHARE otherwise, as a
    Space sets them), the folders whose files the app returns, and on Gradio 6 the
    theme and CSS."""
    kwargs: Dict[str, Any] = dict(
        max_file_size=MAX_FILE_SIZE,
        allowed_paths=[
            str(settings.output_dir),
            str(su.EXAMPLES_DIR),
            str(examples_dir.expanduser().resolve()),
        ],
    )
    if args.host:
        kwargs["server_name"] = args.host
    if args.port is not None:
        kwargs["server_port"] = args.port
    if args.share:
        kwargs["share"] = True
    if STYLE_IN_LAUNCH:
        kwargs.update(style())
    return kwargs


def preload(service: su.DepthService) -> None:
    """Load the default model before the server starts, saying what takes the time; a
    missing weights file or DINOv3 ends the program with the reason (SystemExit)."""
    entry = service.registry.default
    if service.is_loaded(entry.id):  # ZeroGPU: loaded at import
        return
    print(
        f"Loading {entry.label}: the weights file (1.08 GB, downloaded once) and "
        f"DINOv3 ViT-7B (27 GB; a few minutes on the first start) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        service.ensure(entry.id)
    except su.SetupError as e:
        raise SystemExit(f"app.py: {e}") from None
    print(f"Loaded {entry.label} in {time.perf_counter() - t0:.0f} s", flush=True)
    torch = sys.modules.get("torch")
    if torch is not None and not torch.cuda.is_available():
        print(
            "Warning: no CUDA GPU, the model runs on the CPU or Apple MPS: slow, and "
            "photos above a few megapixels need a CUDA GPU (docs/inference.md).",
            flush=True,
        )


def main(argv: Optional[Sequence[str]] = None) -> gr.Blocks:
    """Parse the command line, load the default model (unless --lazy) and launch."""
    args = parse_args(argv)
    if args.examples is not None:  # read by su.resolve_examples in build_demo
        os.environ[su.EXAMPLES_DIR_ENV] = str(args.examples)
    demo = build_demo()
    if SERVICE.settings.preload and not args.lazy:  # (ZeroGPU: loaded at import)
        preload(SERVICE)
    demo.launch(**launch_kwargs(args, SERVICE.settings, su.local_examples_dir()))
    return demo


if __name__ == "__main__":
    main()
