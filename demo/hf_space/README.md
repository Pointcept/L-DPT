---
title: L-DPT
emoji: 🌄
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: "3.10"
app_file: app.py
license: cc-by-nc-4.0
short_description: Relative depth at a photo's native resolution, up to 100 MP
suggested_hardware: zero-a10g
startup_duration_timeout: 1h
pinned: false
---

# L-DPT: relative depth at a photo's native resolution

[ [Project](https://xywu.me/l-dpt) ] [ [Code](https://github.com/Pointcept/L-DPT) ] [ [Weights](https://huggingface.co/pointcept-bot/L-DPT) ] [ [Training data](https://huggingface.co/datasets/pointcept-bot/DepthHybrid) ] [ [Examples](https://huggingface.co/datasets/pointcept-bot/L-DPT-examples) ]

Upload a photo and get its relative depth from L-DPT Large (zero-shot), computed on every
pixel of the photo. The Space accepts photos of up to 105 MP. L-DPT pairs a frozen DINOv3
ViT-7B/16 encoder with a latent DPT decoder trained on 768 x 768 crops only, with no
high-resolution training data. The model
([`l-dpt_large_depth_zeroshot.pth`](https://huggingface.co/pointcept-bot/L-DPT)) is
trained on synthetic and multi-view data only.

**GPU time** on this Space, at native resolution: 1.2 s at 2 MP, 14 s at 8 MP, 70 s at
32 MP, 160 s at 60 MP and 370 s at 100 MP. Every photo runs on half of the GPU. Above
3072 px the frozen encoder runs in overlapping 3072 px windows and the decoder decodes
the whole latent grid in one pass; below it is one forward pass.

**Quota.** Before it starts, a request reserves GPU time from your daily ZeroGPU quota:
its estimated time with a margin, counted 1.5 times on half of the GPU. That is 170 s
for a 32 MP photo, 350 s for a 60 MP photo and 720 s for a 100 MP photo;
a free account has 300 s, enough for photos up to 47 MP, so larger photos need a
Hugging Face PRO account here. The app on your own GPU (below) has no such limit.
Each example shows a precomputed result at once.

**Output.** Relative depth: larger = farther, defined up to a scale and a shift. It is
shown as by the L-DPT scripts: near is blue, far is red, through a tone curve fitted to
each photo (its 1-99 % depth range, a blend of linear and at most 40 % log depth that
puts the median depth in blue-teal). The downloads hold it at full resolution as
a 16-bit PNG (`depth = lo + value / 65535 * (hi - lo)`, with `lo` / `hi` in its `l-dpt`
text chunk and in the sidecar JSON) and as a colour JPEG.

## Run it on your own GPU

The same app runs from the L-DPT code on an NVIDIA GPU: its peak allocation is 15 GiB at
2 MP and 33 GiB at 100 MP, where PyTorch reserves 34 GiB on an RTX PRO 6000. It needs
Meta's DINOv3 ViT-7B/16 weights, which we cannot
redistribute: request access on the
[release page](https://github.com/facebookresearch/dinov3) or the gated
[Hugging Face repo](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m)
(27 GB).

```bash
git clone https://github.com/Pointcept/L-DPT.git && cd L-DPT
pip install -e ".[demo,app]"
export LDPT_DINOV3=/path/to/dinov3   # the DINOv3 file or Hugging Face folder
python demo/hf_space/app.py          # loads the model, then serves http://127.0.0.1:7860
```

Without `LDPT_DINOV3` the app downloads the gated repository itself, with the `HF_TOKEN`
of an account that has access. `--host 0.0.0.0`, `--port` or `--share` serve it from a
remote machine, and `--examples DIR` adds a gallery: your photos, or a download of
[pointcept-bot/L-DPT-examples](https://huggingface.co/datasets/pointcept-bot/L-DPT-examples)
(`docs/inference.md` of the code).

## Duplicate this Space

At startup the Space downloads Meta's gated repo
[facebook/dinov3-vit7b16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m)
(transformers format, 27 GB), which needs the **`HF_TOKEN` secret**: a read token of an
account that Meta has granted access. The L-DPT weights file is public and pinned in
`models.json`, which can list more weights files (e.g. `l-dpt_large_depth_mixed.pth`);
with more than one the UI offers a choice. The example photos and their precomputed
depth are the dataset [pointcept-bot/L-DPT-examples](https://huggingface.co/datasets/pointcept-bot/L-DPT-examples):
set the variable `LDPT_EXAMPLES_REPO=pointcept-bot/L-DPT-examples` to give a copy the
same gallery. Without it, a copy starts without a gallery.

Optional variables:

| Variable | |
|---|---|
| `LDPT_DINOV3` | a local DINOv3 ViT-7B/16: the official `.pth` or a transformers-format folder |
| `LDPT_DINOV3_REPO`, `LDPT_DINOV3_FILE` (+ `LDPT_DINOV3_REVISION`) | a repo holding the official `.pth` instead |
| `LDPT_DINOV3_HF_REPO`, `LDPT_DINOV3_HF_REVISION` | another transformers-format repo / commit |
| `LDPT_CKPT_LARGE_DEPTH_ZEROSHOT` | a local copy of the weights file (the same variable as in the L-DPT API and scripts) |
| `LDPT_MAX_MP` | largest input (default 105) |
| `LDPT_MAX_INFER_MP` | largest inference resolution (default 105) |
| `LDPT_PRELOAD=0` | load the model at the first request instead of at startup (`app.py --lazy`; on ZeroGPU every model is loaded at startup) |
| `LDPT_DEVICE`, `LDPT_PRECISION` (default: each model's own), `LDPT_CHUNK_TOKENS` (32768) | inference options |
| `LDPT_EXAMPLES_REPO` (+ `LDPT_EXAMPLES_REVISION`) | a dataset repo with the example photos (`examples.json`, originals, thumbnails) instead of `examples/` |
| `LDPT_EXAMPLES_DIR` | a local folder of example photos instead of `examples/` (`app.py --examples`) |
| `LDPT_VERIFY=0` | skip the weight checks (sha256 of downloaded files, DINOv3 config) |

## License

The L-DPT code, the weights and this Space are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The DINOv3 encoder is
Meta's, under the DINOv3 License, and is not redistributed. The example photos are not
covered: they keep their owners' rights
([pointcept-bot/L-DPT-examples](https://huggingface.co/datasets/pointcept-bot/L-DPT-examples)).

## Citation

```bibtex
@inproceedings{wu2026latent,
  title     = {Scaling Dense Prediction with Latent Decoding},
  author    = {Wu, Xiaoyang and Lao, Yixing and Wang, Chengyao and Yang, Senqiao and Zhang, Yujia and Zhao, Hengshuang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
