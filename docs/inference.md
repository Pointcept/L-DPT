# Inference

Relative depth of a photo with a released L-DPT model: the Python API, native resolution
and the encoder windows, time and memory, the two scripts, how depth is coloured and
stored, segmentation, and the demo. Installation and the DINOv3 weights:
[install.md](install.md).

## Python API

```python
import cv2
from ldpt.models.l_dpt import api

model = api.load("l-dpt_large_depth_zeroshot")  # or "l-dpt_large_depth_mixed"
image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)  # HxWx3 uint8 RGB
depth = api.predict(model, image)               # HxW float32, short side resized to 768
depth = api.predict(model, image, size=None)    # at the photo's native resolution
depth = api.predict(model, image, flip=True)    # averaged with the mirrored image (as evaluated)
depth = api.predict(model, image, size=None, encoder_tiles=None)  # one encoder pass at any size
```

The output is relative depth: positive, larger = farther, defined up to a positive scale
and a shift per image. To compare it with metric depth, fit `s * depth + t` with `s > 0`
per image, as the evaluation does.

`api.load(name_or_path, encoder_weights=None, device="auto", precision=None, *, weights=None, ...)`:

- `name_or_path`: a preset (`"l-dpt_large_depth_zeroshot"`, `"l-dpt_large_depth_mixed"`;
  `"l-dpt_large_semseg_cocostuff"`: [Segmentation](#segmentation)) or the path of a
  weights file. A preset's file is downloaded from Hugging Face on first
  use, at a pinned revision, and its sha256 is checked; `weights=<file>` or
  `$LDPT_CKPT_LARGE_DEPTH_ZEROSHOT` / `$LDPT_CKPT_LARGE_DEPTH_MIXED` /
  `$LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF` use a local copy.
  With a preset name, the file must have that preset's architecture.
- `encoder_weights`: the DINOv3 file or Hugging Face folder (default `$LDPT_DINOV3`).
- `device`: `"auto"` (CUDA, else Apple MPS, else CPU) or a device.
- `precision`: `None` keeps the decoder precision the model was trained with (bf16);
  `"fp32"` or `"bf16"` to choose.
- `share_encoder=other.encoder`: a second model reuses the frozen encoder of the first,
  so it costs only its decoder.

`load` builds the model on the meta device and reads the files memory-mapped, with the 7B
encoder stored in bf16 (bit-identical under its bf16 autocast), so no fp32 copy of the
encoder is ever made: the weights take 13.5 GiB of GPU memory.

`api.predict(model, image, *, size="auto", flip=False, sdpa_backends="auto", encoder_tiles="auto")`:

- `image`: HxWx3 RGB, uint8 or float in [0, 1], numpy or torch.
- `size`: the short side the image is resized to before the network: `"auto"` (768, the
  training resolution), an int, or `None` for the native resolution. The depth is always
  returned at the input size (resized bilinearly).
- `flip`: also run the mirrored image and average the two, as in the evaluation; the
  network runs twice.
- `sdpa_backends`: the order in which PyTorch's attention kernels are tried for the call.
  The default `"auto"` picks it per GPU: on compute capability 10.x (B200, GB200, GB300)
  cuDNN attention first, then FlashAttention, memory-efficient attention and the plain
  math fallback (`api.SDPA_ORDER`, the order of the training option `sdpa_backends`,
  [training.md](training.md)), which took less than half the time of PyTorch's order
  on a GB200 from 32 MP up with one encoder pass; on every other GPU PyTorch's own
  order, because cuDNN first is slower on an RTX PRO 6000 Blackwell
  ([Time and memory](#time-and-memory)). `None`
  keeps PyTorch's order (FlashAttention first), which the evaluation uses
  ([evaluation.md](evaluation.md)); a list of names sets one. A kernel the GPU or the
  data type cannot use is skipped (on the CPU and Apple MPS the order changes nothing).
  The kernels compute the same attention: the depth differs at the rounding level of
  bf16.
- `encoder_tiles`: `"auto"` (default) runs the frozen encoder of a depth model in
  overlapping 3072 px windows when a side of the network's input exceeds 3072 px, and
  the decoder on the whole latent grid in one pass
  ([Encoder windows](#encoder-windows)); a smaller input, and a segmentation model at
  any size ([Segmentation](#segmentation)), is one forward pass. `None` (or `False`)
  runs the encoder once on the whole image at any size; an
  `api.EncoderTiles(tile, overlap, rope_scale)` sets a tiling (`api.ENCODER_TILES`:
  the windows above).

The preprocessing is that of the validation pipeline: bicubic, antialiased resize and
ImageNet normalisation.

## Native resolution and memory

`size=None` runs the network on every pixel of the photo. For that call `predict` turns
on two memory options of the model (they change no weights; same math, and in bf16
the result differs at the rounding level):

- `selection_impl="stream"` projects each tapped encoder layer as the encoder yields it,
  instead of keeping all four;
- `chunk_tokens=32768` runs the encoder blocks and the decoder MLPs on chunks of tokens.

To choose other values, pass them to `api.load` (for example
`api.load(..., selection_impl="concat")`) or set `model.native_memory = {}` to run the
model as loaded. The default `size="auto"` path never uses these options.

At tens of megapixels a CUDA GPU is required: the fp32 attention of the last decoder
stage needs PyTorch's memory-efficient attention kernel, and the fallback on the CPU or
Apple MPS materialises the full attention matrix.

### Encoder windows

When a side of a depth model's input exceeds 3072 px (192 patches of 16 px), the frozen
DINOv3 encoder runs in overlapping windows of 3072 x 3072 px, and the latent decoder
decodes the whole latent grid in one pass. Below 3072 px the network runs one forward
pass, so the benchmark numbers (short side 768) are those of the plain forward.

- Windows: along a side of n patches, k = ceil((n - 64) / 128) windows of 192 patches
  at round(i (n - 192) / (k - 1)), so neighbours overlap by 64 patches or more: 4 x 6
  windows for a 102 MP photo of 8750 x 11664 px (width x height), 5 x 3 for 60 MP
  (9504 x 6336 px), 1 x 2 for 8 MP (2448 x 3264 px).
- Inside a window the encoder's RoPE coordinates are multiplied by 2. DINOv3
  normalises its patch coordinates to [-1, 1] per axis whatever the number of
  patches, and L-DPT was trained on 768 px crops (48 patches) with the encoder's
  coordinate rescale of 0.5 to 2: patch spacings from 1/48 to 1/12. A window of 192
  patches with the factor 2 has a spacing of 1/48; a 100 MP photo in one pass has
  2/547 and 2/729, 5.7 and 7.6 times denser.
- Each window's per-stage selections (the decoder's inputs) are added to the whole
  grid with a linear ramp over the 64 patches next to an inner window edge, and
  divided by the summed weights. The decoder is unchanged.

`encoder_tiles=None` runs the encoder in one pass at any size (`--no-encoder-tiles` in
the scripts).

## Time and memory

Released zero-shot model, native resolution, no mirrored pass, bf16 decoder, the memory
options and the encoder windows above; one model loaded and warm, PyTorch 2.8, on an
NVIDIA RTX PRO 6000 Blackwell. Each time is one timed call, except the medians of three
calls at 2, 8 and 32 MP.
Peak memory is `torch.cuda.max_memory_allocated` in GiB (2^30 bytes); it does not
depend on the GPU. The scripts and the demo set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (a value already set wins; with the
API, set it before importing torch), with which PyTorch reserves 34 GiB at 100 MP on
an RTX PRO 6000 (44 GiB without). The fp32 attention of the last decoder stage runs
PyTorch's memory-efficient kernel in every order.

| Photo | Peak GPU memory | RTX PRO 6000 Blackwell |
|---|---|---|
| 2 MP | 14.8 GiB | 0.64 s |
| 8 MP | 18.6 GiB | 7.6 s |
| 32 MP | 21.0 GiB | 39 s |
| 60 MP | 24.5 GiB | 106 s |
| 100 MP | 33.1 GiB | 205 s |

The 100 MP row is a 102 MP photo. The 2, 8 and 32 MP photos are that photo resized
(1232 x 1632, 2448 x 3264 and 4896 x 6528 px), the 60 MP photo is 9504 x 6336 px
(width x height); from 8 MP up the encoder runs in 1 x 2, 2 x 3, 5 x 3 and 4 x 6
windows. The RTX PRO 6000 runs the attention order `predict` picks for it
(`sdpa_backends="auto"`), PyTorch's; cuDNN attention first is slower:

| Photo | PyTorch's order | cuDNN first |
|---|---|---|
| 2 MP | 0.64 s | 0.69 s |
| 8 MP | 7.6 s | 8.5 s |
| 32 MP | 39 s | 41 s |
| 60 MP | 106 s | 110 s |
| 100 MP | 205 s | 212 s |

The windows add encoder work where they overlap most, just above 3072 px. On the RTX
PRO 6000, two windows of 2448 x 3072 px cover the 8 MP photo in 7.6 s, against 4.4 s
in one encoder pass (`encoder_tiles=None`), and four windows cover a 3088 x 3088 px
photo (9.5 MP) in 22 s, against 6.1 s. At 32 MP the windows are faster than one pass:
39 s against 48 s. From 32 MP to 100 MP (a 102 MP photo, 3.2 times the
pixels) the time grows 5.3 times on the RTX PRO 6000: the encoder windows grow with the
pixels, and the decoder's attention spans the whole photo.

## Command line

### One photo: `scripts/demo_depth.py`

```bash
python scripts/demo_depth.py photo.jpg
python scripts/demo_depth.py photo.jpg --preset l-dpt_large_depth_mixed --size auto
```

It writes `photo_depth.png` (the depth in colour, see below) and `photo_depth.npy` (the
float32 depth) next to the photo and prints both paths. Options:

| Option | Default | |
|---|---|---|
| `--preset` | `l-dpt_large_depth_zeroshot` | or `l-dpt_large_depth_mixed` |
| `--weights` | the download | a local weights file of the preset |
| `--encoder-weights` | `$LDPT_DINOV3` | the DINOv3 file or folder |
| `--size` | `native` | `native`, `auto` (short side 768) or a short side in pixels |
| `--flip` | off | also average with the mirrored image |
| `--no-encoder-tiles` | off | run the encoder in one pass at any size (default: 3072 px windows above 3072 px) |
| `--cmap` | `Spectral` | `Spectral`, `Turbo`, `Inferno` or `Gray` |
| `--device` | `auto` | `cuda`, `mps` or `cpu` |
| `-o`, `--output` | `<photo>_depth.png` | the colour PNG; the `.npy` takes its stem |

Note that the script runs at the native resolution by default, while `api.predict`
resizes to 768 unless you pass `size=None`.

### Many large photos: `scripts/highres_infer.py`

```bash
python scripts/highres_infer.py photo1.jpg photo2.jpg --out out/
python scripts/highres_infer.py --model l-dpt_large_depth_zeroshot \
    --model l-dpt_large_depth_mixed --images-dir photos/ --out out/
```

It runs one or more models over the photos at native resolution, one model loaded at a
time. `--model` takes a preset (default `l-dpt_large_depth_zeroshot`), `NAME=PRESET`, or
`NAME=FILE` for a weights file; `NAME` is the output folder. Photos come as arguments,
`--image` or `--images-dir`. For each photo and model, `out/<photo>/<model>/` gets:

| File | |
|---|---|
| `depth16.png` | 16-bit depth at full resolution (the format below) |
| `depth_color.jpg` | the depth in colour at full resolution (`.png` when a side exceeds 65,500 pixels) |
| `preview.jpg` | the colour depth, long side 2048 |
| `crop_<k>.jpg` | crops of the native-resolution colour depth (three 1024 px squares by default; `--crop X,Y,W` chooses) |
| `meta.json` | image size, model and weights provenance, options, timings, peak GPU memory, the 16-bit range and the colours' tone curve |

and `out/<photo>/` gets `rgb_preview.jpg` and `rgb_crop_<k>.jpg` once. A photo and model
whose `meta.json` matches the current options is skipped (`--overwrite` redoes it), so a
rerun resumes; parallel processes with one model each can share `out/`. The memory
fields of `meta.json` (`peak_gpu_allocated_gb`, `peak_gpu_reserved_gb`) are in GB
(10^9 bytes): 33 GiB reads as 36 GB there. Other options: `--size`, `--flip`,
`--no-encoder-tiles`, `--precision`, `--selection-impl`, `--chunk-tokens`, `--device`
(`python scripts/highres_infer.py --help`).

## How depth is coloured and stored

Depth images follow one convention in the scripts and the demo: near is blue, far is
red (matplotlib's "Spectral", reversed, without its deep-purple and deep-red ends),
through a tone curve fitted to each image. The curve spans the image's 1-99 % depth
range, with soft knees beyond it, and blends linear depth with at most 40 % log depth
so that the median depth lands in blue-teal: the near layers of a near-heavy photo
spread out, a far-heavy photo stays even, and no part of the range is stretched more
than 2.6 times a linear map. Only relative depth counts: a scale and a shift of the depth give the
same image. `--cmap` of `demo_depth.py` also takes Turbo (near blue, far red), Inferno
and Gray (near bright, far dark).

The 16-bit PNG is grayscale, mapped linearly from the depth's minimum `lo` to its
maximum `hi`: `depth = lo + value / 65535 * (hi - lo)`, with `lo` and `hi` stored as JSON
in the PNG's text chunk `l-dpt` (and in `meta.json` or the demo's JSON). Both live in
`ldpt.utils.depth_image`:

```python
from ldpt.utils.depth_image import colorize, read_depth16, tone_curve, write_depth16

rgb = colorize(depth)                       # HxW depth -> HxWx3 uint8 RGB
curve = tone_curve(depth)                   # its curve (lo, hi, w) ...
small_rgb = colorize(small, curve=curve)    # ... colours a downsampled copy alike
info = write_depth16("depth16.png", depth)  # {"lo": ..., "hi": ..., ...}
depth, info = read_depth16("depth16.png")   # back to float32 depth
```

## Segmentation

`"l-dpt_large_semseg_cocostuff"` labels every pixel with one of the 171 COCO-Stuff
classes (80 things, 91 stuff). It runs through the same API:

```python
model = api.load("l-dpt_large_semseg_cocostuff")  # downloads the 3.09 GB weights file once
classes = api.predict(model, image)             # HxW int64 class ids, short side resized to 768
classes = api.predict(model, image, flip=True)  # averaged with the mirrored image
model.class_names[classes[0, 0]]                # the class names, in id order
```

`predict` does what the COCO-Stuff evaluation does: the short side resized to 768
(bicubic), ImageNet normalisation, the whole image through the network, the logits
resized bilinearly to the photo, the argmax per pixel. `flip=True` averages the softmax
of the photo with that of its mirror image, as the evaluation's flip TTA. `size` and
`sdpa_backends` work as for depth; at `size=None` the logits take 171 x H x W x 4 bytes.
The encoder runs in one pass at any size (`encoder_tiles="auto"`): in 3072 px windows
the classes of large regions fragment. `encoder_tiles=api.ENCODER_TILES` runs the
windows all the same.
`weights=<file>` or `$LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF` loads a local copy of the
weights file.

### One photo: `scripts/demo_segment.py`

```bash
python scripts/demo_segment.py photo.jpg
python scripts/demo_segment.py photo.jpg --flip --size native
```

It writes `photo_seg.png` (the classes in colour) and `photo_seg.npy` (the class ids,
uint8) next to the photo and prints both paths and the five largest classes with their
share of the pixels. Options:

| Option | Default | |
|---|---|---|
| `--preset` | `l-dpt_large_semseg_cocostuff` | |
| `--weights` | the download | a local weights file of the preset |
| `--encoder-weights` | `$LDPT_DINOV3` | the DINOv3 file or folder |
| `--size` | `auto` | `auto` (short side 768), `native` or a short side in pixels |
| `--flip` | off | also average with the mirrored image |
| `--top` | `5` | the number of classes listed |
| `--device` | `auto` | `cuda`, `mps` or `cpu` |
| `-o`, `--output` | `<photo>_seg.png` | the colour PNG; the `.npy` takes its stem |

Every class has a fixed colour: class `c` is entry `c + 1` of the PASCAL VOC colour map
(entry 0, black, is left for unlabelled pixels). Colours, class names and the largest
classes come from `ldpt.utils.seg_image`:

```python
from ldpt.utils.seg_image import COCOSTUFF_CLASS_NAMES, colorize, top_classes

rgb = colorize(classes, 171)                              # HxW ids -> HxWx3 uint8 RGB
for c, name, share in top_classes(classes, COCOSTUFF_CLASS_NAMES):
    print(f"{100 * share:.1f} % {name}")
```

## The demo

The [online demo](https://huggingface.co/spaces/pointcept-bot/L-DPT) is the Gradio app in
`demo/hf_space`. It runs the same way on your own GPU, in a clone of this repository:

```bash
pip install -e ".[demo,app]"           # Gradio 5.49.1, the version of the online demo
export LDPT_DINOV3=/path/to/dinov3     # the DINOv3 file or Hugging Face folder (install.md)
python demo/hf_space/app.py            # then open http://127.0.0.1:7860
```

At start the app loads the zero-shot model, then serves the page. The first start
downloads the weights file pinned in `demo/hf_space/models.json` (1.08 GB, into the
Hugging Face cache; `LDPT_CKPT_LARGE_DEPTH_ZEROSHOT` uses a local copy, which the demo
does not check against the pinned sha256)
and reads DINOv3 (27 GB), which takes a few minutes. Without `LDPT_DINOV3` it downloads
DINOv3 from Meta's gated Hugging Face repository, which needs an account with access
(`HF_TOKEN` or `hf auth login`). A missing file stops the start with the reason.

Upload a photo and press **Estimate depth**. The network runs at the photo's native
resolution, without the mirrored pass. The page shows the depth (coloured as above) and
offers the 16-bit PNG, the colour JPEG and a JSON of the run, all at full resolution.
Time and peak GPU memory are those of the tables above: 21 GiB at 32 MP, 33 GiB at
100 MP. A photo that does not fit in GPU memory ends with an out-of-memory message:
downscale it, or set `LDPT_MAX_INFER_MP` (megapixels) to refuse larger photos at once.

| Option | |
|---|---|
| `--host ADDR` | the address to listen on (default `127.0.0.1`, this machine only; `0.0.0.0`: every network interface) |
| `--port N` | the port (default the first free one from 7860) |
| `--share` | also a temporary public `gradio.live` link |
| `--examples DIR` | a gallery: a folder with an `examples.json`, such as a download of [L-DPT-examples](https://huggingface.co/datasets/pointcept-bot/L-DPT-examples) (the online demo's photos with their thumbnails and precomputed depth, below), or any folder of photos (each is its own thumbnail; `demo/hf_space/make_examples.py` writes smaller copies) |
| `--lazy` | start the server at once and load the model at the first request |

The 21 example photos of the online demo, with their precomputed depth, are the dataset
[L-DPT-examples](https://huggingface.co/datasets/pointcept-bot/L-DPT-examples):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('pointcept-bot/L-DPT-examples', repo_type='dataset', local_dir='l-dpt-examples')"
python demo/hf_space/app.py --examples l-dpt-examples
```

On a remote GPU server, open a tunnel from your computer
(`ssh -L 7860:127.0.0.1:7860 <server>`, then http://127.0.0.1:7860), or listen on
`--host 0.0.0.0` where the network is trusted. Scripts can call the app with
`gradio_client` (`api_name="/depth"`), though `api.predict` and
`scripts/demo_depth.py` need no server. The other settings (device, precision, limits)
are environment variables: [demo/hf_space/README.md](../demo/hf_space/README.md).
