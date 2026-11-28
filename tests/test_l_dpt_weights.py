"""
The release files (convert.py output: the two depth models and the COCO-Stuff
segmentation model) load strictly with their preset and give sane outputs (CPU, random
7B-width features from a parameter-free stub encoder, so the 7B encoder is never built).

File paths come from environment variables; a test is skipped when its variable is
unset or the file is missing:
    LDPT_CKPT_LARGE_DEPTH_ZEROSHOT    l-dpt_large_depth_zeroshot.pth
    LDPT_CKPT_LARGE_DEPTH_MIXED       l-dpt_large_depth_mixed.pth
    LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF  l-dpt_large_semseg_cocostuff.pth
    LDPT_DINOV3                       the official DINOv3 ViT-7B/16 file (memory-mapped,
                                      keys only)
A file stores its preset's name (convert --preset), a short name of
presets.LEGACY_PRESETS (read as the current one) or none (convert --config); loaded by
path, a file without a preset name reports None.
"""

import os

import pytest
import torch

import helpers  # noqa: F401  (registers the stub encoder)
from ldpt.models.l_dpt import api, convert, weights
from ldpt.models.l_dpt.presets import (
    CLASS_NAMES,
    RELEASE_L,
    SEMSEG_L,
    WEIGHTS,
    canonical_preset,
    get_preset,
)

pytestmark = pytest.mark.weights

STUB = dict(type="feature_stub_test")


def env_file(name):
    path = os.environ.get(name)
    if not path or not os.path.isfile(path):
        pytest.skip(f"{name} not set or missing")
    return path


@pytest.mark.parametrize(
    "preset", ["l-dpt_large_depth_zeroshot", "l-dpt_large_depth_mixed"]
)
def test_release_file(preset, monkeypatch):
    """The file: format l-dpt-release/2, its preset's name (or its LEGACY_PRESETS
    name) and architecture (every output-changing option equal to RELEASE_L),
    decoder-only fp32 state; it loads by path and by preset name (weights=) to the
    same, finite, non-constant prediction, both reporting the current preset name."""
    path = env_file(WEIGHTS[preset].env)
    ckpt = weights.read_checkpoint(path)
    assert ckpt["format"] == convert.FORMAT
    assert canonical_preset(ckpt["preset"]) in (preset, None)
    config, state, info = convert.load_release(path)
    api._check_same_architecture(config, RELEASE_L, "file", preset, path)
    assert all(not k.startswith("encoder.") for k in state)
    assert {v.dtype for v in state.values()} == {torch.float32}
    assert info["provenance"]["source_sha256"]
    image = torch.randn(1, 3, 64, 96, generator=torch.Generator().manual_seed(0))
    by_path = api.load(path, device="cpu", encoder=STUB)
    # the stub replaces the file's encoder: leave "encoder" out of the architecture
    # check of the load by name (every other key is compared, as checked above)
    monkeypatch.setattr(
        api, "_ARCH_KEYS", tuple(k for k in api._ARCH_KEYS if k != "encoder")
    )
    by_name = api.load(preset, device="cpu", encoder=STUB, weights=path)
    assert by_name.preset == preset
    assert by_path.preset == canonical_preset(ckpt["preset"])
    with torch.no_grad():
        a, b = by_path.infer(image), by_name.infer(image)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all() and a.min() > 0 and a.std() > 0


def test_semseg_release_file(monkeypatch):
    """The segmentation file: format l-dpt-release/2, its preset's name and
    architecture (SEMSEG_L), decoder-only fp32 state with the tap BatchNorm statistics;
    by path and by preset name (weights=) the same finite logits over 171 classes, the
    COCO-Stuff class names, RoPE per axis as evaluated, and a class map from
    api.predict at the image's size."""
    preset = "l-dpt_large_semseg_cocostuff"
    path = env_file(WEIGHTS[preset].env)
    ckpt = weights.read_checkpoint(path)
    assert ckpt["format"] == convert.FORMAT and ckpt["preset"] == preset
    config, state, info = convert.load_release(path)
    api._check_same_architecture(config, SEMSEG_L, "file", preset, path)
    assert all(not k.startswith("encoder.") for k in state)
    assert {v.dtype for k, v in state.items() if "num_batches" not in k} == {
        torch.float32
    }
    assert state["tap_bn.0.running_var"].min() > 0
    assert info["provenance"]["source_sha256"]
    by_path = api.load(path, device="cpu", encoder=STUB)
    monkeypatch.setattr(
        api, "_ARCH_KEYS", tuple(k for k in api._ARCH_KEYS if k != "encoder")
    )
    by_name = api.load(preset, device="cpu", encoder=STUB, weights=path)
    for model in (by_path, by_name):
        assert model.preset == preset and model.class_names == list(CLASS_NAMES[preset])
        assert model.rope.normalize_coords == "separate" and model.num_classes == 171
    image = torch.randn(1, 3, 64, 96, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a, b = by_path.infer(image), by_name.infer(image)
    assert torch.equal(a, b) and a.shape == (1, 171, 64, 96)
    assert torch.isfinite(a).all() and a.std() > 0
    rgb = torch.randint(0, 256, (40, 56, 3), dtype=torch.uint8)
    classes = api.predict(by_path, rgb, size=64)
    assert classes.shape == (40, 56) and classes.dtype == torch.int64


def test_dinov3_7b_keys():
    """The official 7B file fills the encoder of the release preset exactly
    (memory-mapped; shapes only, nothing is read)."""
    path = env_file(weights.ENCODER_ENV)
    state = weights.encoder_state(path)
    model = convert.build_meta(get_preset("l-dpt_large_depth_zeroshot"))
    expected = {
        k: v.shape for k, v in model.state_dict().items() if k.startswith("encoder.")
    }
    assert state.keys() == expected.keys()
    assert all(state[k].shape == shape for k, shape in expected.items())
    # keys api.load stores in bf16: the Linear / Conv2d weights and biases of the file
    stored = api._storage_dtypes(model)
    assert stored and {f"encoder.{k}" for k in stored} <= state.keys()
    assert all(model.encoder.get_parameter(k).dtype == torch.bfloat16 for k in stored)
