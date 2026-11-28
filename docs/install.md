# Installation

How to install L-DPT for inference only, or for training, evaluation and the tests, and
how to get the DINOv3 weights it needs.

L-DPT runs on Linux with Python 3.10 or newer and PyTorch 2.8 or newer; it is tested with
Python 3.10 and PyTorch 2.8.0. The test suite runs on x86_64 with
the CUDA 12.8 wheels (`cu128`, pinned in `requirements.txt`); the released models were
trained and evaluated on NVIDIA GB200 nodes (aarch64) with the CUDA 12.9 wheels
(`cu129`: PyTorch 2.8.0 has no aarch64 `cu128` wheel).

## Inference only

In a clone of this repository (install PyTorch 2.8 or newer with CUDA first to choose
its build; otherwise pip installs PyPI's; on aarch64 see the next section):

```bash
pip install -e ".[demo]"
```

This installs the `ldpt` package with the inference API (torch, numpy, huggingface_hub,
safetensors) and the `demo` extra (OpenCV and matplotlib, for the quick start and the
scripts). Add `app` for the Gradio demo: `pip install -e ".[demo,app]"`, then
`python demo/hf_space/app.py` ([inference.md](inference.md#the-demo)).

## Training, evaluation and tests

With conda (the CUDA toolkit and compilers come from `environment.yml`):

```bash
conda env create -f environment.yml
conda activate l-dpt
pip install -r requirements.txt          # torch 2.8.0 + CUDA 12.8 wheels, training dependencies
pip install -e ".[demo,dev]"             # the ldpt package, OpenCV / matplotlib, pytest
```

Or with pip only, on an existing PyTorch 2.8 or newer:

```bash
pip install -e ".[train,demo,dev]"
```

On aarch64, first install `torch==2.8.0 torchvision==0.23.0` from
`https://download.pytorch.org/whl/cu129`, then `pip install -e ".[train,demo,dev]"`.

| Extra | For |
|---|---|
| `demo` | OpenCV and matplotlib: the quick start, `scripts/demo_depth.py`, `scripts/highres_infer.py` |
| `app` | Gradio 5.49.1 (the version of the online demo): the demo in `demo/hf_space` |
| `train` | training and evaluation (`train.py`, `train.sh`, `scripts/evaluate.py`) |
| `data` | transformers: rebuilding the deduplicated training splits ([data.md](data.md)) |
| `dev` | pytest and black |

For the conda route, `requirements-extra.txt` lists the optional packages: transformers
(as `data`), pytest and black (as `dev`).

## DINOv3 weights

The weights files of L-DPT hold the decoder only. The frozen encoder is Meta's DINOv3
ViT-7B/16 (LVD-1689M), under the DINOv3 License, and is not redistributed here. Get it
from Meta in either form:

- **The original file** of the DINOv3 release
  ([facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)),
  `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth`. Its name ends in the first 8 hex
  digits of its sha256; the first load checks the file against them, whatever it is
  called. Training reads it under this name (below).
- **The gated Hugging Face repository**
  [facebook/dinov3-vit7b16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m)
  (request access on its page). The inference API reads the downloaded folder
  (transformers format) into the same tensors as the original file and checks its
  `config.json`:

  ```python
  from huggingface_hub import snapshot_download

  # after Meta granted access: export HF_TOKEN=... (or hf auth login)
  dinov3 = snapshot_download(  # 27 GB
      "facebook/dinov3-vit7b16-pretrain-lvd1689m",
      allow_patterns=["config.json", "model*.safetensors*"],
  )
  print(dinov3)  # the folder for LDPT_DINOV3
  ```

Where each part of the code looks for it:

- The inference API and everything built on it (`api.load`, `scripts/demo_depth.py`,
  `scripts/highres_infer.py`, the demo) take the file or the folder as
  `encoder_weights=` / `--encoder-weights`, or from `$LDPT_DINOV3`.
- Training and evaluation read the original file only, from
  `checkpoints/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` (the configs'
  `checkpoint`); `scripts/evaluate.py --encoder <file>` takes another path.

## Environment variables

| Variable | Meaning |
|---|---|
| `LDPT_DINOV3` | the DINOv3 ViT-7B/16 file or Hugging Face folder (inference) |
| `LDPT_CKPT_LARGE_DEPTH_ZEROSHOT` | a local copy of `l-dpt_large_depth_zeroshot.pth`, instead of the download (the API, the scripts and the demo) |
| `LDPT_CKPT_LARGE_DEPTH_MIXED` | a local copy of `l-dpt_large_depth_mixed.pth` |
| `LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF` | a local copy of `l-dpt_large_semseg_cocostuff.pth` |
| `HF_TOKEN` | a Hugging Face token of an account with access to the DINOv3 repository, for its gated download (`snapshot_download` or the demo) |

Downloaded weights files go to the Hugging Face cache (`HF_HOME`). In the API and the
scripts, a local copy named by one of the `LDPT_CKPT_*` variables must match the pinned
sha256, as a download does; the demo does not check it.

## Tests

```bash
pytest -m "not weights and not data"   # CPU only, a few minutes
```

The tests build tiny models and run real training, evaluation and resume loops on
synthetic data. `tests/test_l_dpt_golden.py` pins the model's outputs bitwise as recorded
on an x86_64 CPU and is skipped on other CPUs. Two markers need files on disk and are
skipped without them:

```bash
# the released weights files and DINOv3 (fetch prints the path of a checked download)
export LDPT_CKPT_LARGE_DEPTH_ZEROSHOT=$(python -c "from ldpt.models.l_dpt.weights import fetch; print(fetch('l-dpt_large_depth_zeroshot'))")
export LDPT_CKPT_LARGE_DEPTH_MIXED=$(python -c "from ldpt.models.l_dpt.weights import fetch; print(fetch('l-dpt_large_depth_mixed'))")
export LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF=$(python -c "from ldpt.models.l_dpt.weights import fetch; print(fetch('l-dpt_large_semseg_cocostuff'))")
pytest -m weights   # also needs LDPT_DINOV3

pytest -m data   # the datasets under data/ (data.md)
```
