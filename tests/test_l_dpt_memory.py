"""
Inference memory options of L-DPT-v1m1 (CPU): selection_impl="stream" (no concatenated
encoder features) and chunk_tokens (token-chunked encoder blocks and decoder FFN).
Both keep the math and only change summation order: close to the default path, not
bit-identical. The default path is pinned by test_l_dpt_golden.py.

The native-resolution entry points (scripts/demo_depth.py, scripts/highres_infer.py,
scripts/demo_segment.py, demo/hf_space/app.py) set
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (os.environ.setdefault) before torch is
imported; the ldpt package never sets it.
"""

import ast
import copy
import glob
import json
import os
import subprocess
import sys
import weakref

import pytest
import torch

from helpers import REPO, randomize_decoder_
from test_l_dpt_golden import BASE, DEPTH_LOSS, SEG_LOSS, make_data, randomize_

from ldpt.models.builder import build_model

DEPTH = dict(BASE, criterion=DEPTH_LOSS)
SEMSEG = dict(BASE, task="semseg", out_dim=8, num_classes=5, criterion=SEG_LOSS)
STUB = dict(
    encoder=dict(type="feature_stub_test", embed_dim=32),
    encoder_inter_layers=[0, 1, 2, 3],
)
HW = (72, 88)  # padded to 80 x 96


def pair(cfg, **opts):
    ref = build_model(cfg)
    ref = randomize_(ref) if hasattr(ref.encoder, "blocks") else randomize_decoder_(ref)
    model = build_model(dict(cfg, **opts))
    model.load_state_dict(ref.state_dict(), strict=True)
    return ref, model


def infer(model, data):
    with torch.no_grad():
        return model.eval().infer(data["image"])


@pytest.mark.parametrize("cfg", [DEPTH, SEMSEG], ids=["depth", "semseg"])
@pytest.mark.parametrize("selection", ["stage", "shared"])
@pytest.mark.parametrize(
    "opts",
    [
        dict(selection_impl="stream"),
        dict(chunk_tokens=7),
        dict(selection_impl="stream", chunk_tokens=16),
    ],
    ids=["stream", "chunk", "stream+chunk"],
)
def test_options_match_default_fp32(cfg, selection, opts):
    ref, model = pair(dict(cfg, selection=selection), **opts)
    data = make_data(None, HW)
    a, b = infer(ref, data), infer(model, data)
    assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
    torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)


def test_stream_bf16_close():
    ref, model = pair(dict(DEPTH, precision="bf16"), selection_impl="stream")
    image, _ = model.pad(make_data(None, HW)["image"])
    with torch.no_grad():
        a = ref.eval().select(ref.encode(image))
        b = model.eval().encode_select(image)
    assert [t.dtype for t in b] == [t.dtype for t in a] == [torch.bfloat16] * 4
    for x, y in zip(a, b):
        # one bf16 rounding (concat) vs four rounded partial products summed in fp32
        torch.testing.assert_close(y.float(), x.float(), rtol=2e-2, atol=2e-2)


def test_stream_trains_like_concat():
    """encode_select also works with autograd: same loss and decoder gradients."""
    ref, model = pair(DEPTH, selection_impl="stream", chunk_tokens=8)
    data = make_data("depth", HW)
    grads = []
    for m in (ref, model):
        m.train()
        torch.manual_seed(5)
        loss = m(copy.deepcopy(data))["loss"]
        loss.backward()
        grads.append({n: p.grad for n, p in m.named_parameters() if p.requires_grad})
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        torch.testing.assert_close(grads[1][n], grads[0][n], rtol=1e-4, atol=1e-6)


def test_chunk_tokens_off_in_training():
    """chunk_tokens is an eval option: the training forward stays bit-identical."""
    ref, model = pair(DEPTH, chunk_tokens=7)
    data = make_data("depth", HW)
    losses = []
    for m in (ref, model):
        m.train()
        torch.manual_seed(3)
        losses.append(m(copy.deepcopy(data))["loss"])
    assert torch.equal(losses[0], losses[1])


def test_stream_with_encoder_without_iterator():
    """An encoder with only get_intermediate_layers falls back to computing all layers."""
    cfg = dict(type="L-DPT-v1m1", **STUB, dec_dim=64, dec_num_heads=4)
    ref, model = pair(cfg, selection_impl="stream")
    data = make_data(None, HW)
    torch.testing.assert_close(
        infer(model, data), infer(ref, data), rtol=1e-5, atol=1e-5
    )


def test_stream_frees_each_layer():
    """Layer i (and the normalised tensor it views) is gone before the encoder computes
    the blocks of layer i + 1."""
    _, model = pair(DEPTH, selection_impl="stream")
    seen = []
    select_layer = model._select_layer

    def spy(feat, i, neck):
        if not any(r() is feat for r in seen):
            seen.extend([weakref.ref(feat), weakref.ref(feat._base)])
        return select_layer(feat, i, neck)

    def check(module, args):
        assert all(r() is None for r in seen), "a consumed layer is still alive"

    model._select_layer = spy
    for blk in model.encoder.blocks:
        blk.register_forward_pre_hook(check)
    infer(model, make_data(None, HW))
    assert len(seen) == 8


def test_options_keep_state_dict_and_validate():
    ref, model = pair(DEPTH, selection_impl="stream", chunk_tokens=64)
    assert list(model.state_dict()) == list(ref.state_dict())
    assert "selection_impl='stream', chunk_tokens=64" in model.extra_repr()
    with pytest.raises(ValueError, match="selection_impl"):
        build_model(dict(DEPTH, selection_impl="cat"))
    with pytest.raises(ValueError, match="chunk_tokens"):
        build_model(dict(DEPTH, chunk_tokens=0))
    # runtime switch (as api users would do on a loaded model)
    model.selection_impl, model.chunk_tokens = "concat", None
    data = make_data(None, HW)
    assert torch.equal(infer(model, data), infer(ref, data))


# ---------------------------------------------------------------- entry points

ALLOC_CONF = ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ENTRY_POINTS = (
    "scripts/demo_depth.py",
    "scripts/highres_infer.py",
    "scripts/demo_segment.py",
    "demo/hf_space/app.py",
)
# what a script may import before it sets the variable: modules that cannot import torch
BEFORE_ALLOC_CONF = set(sys.stdlib_module_names) | {"__future__", "numpy"}


def is_alloc_conf_setdefault(node):
    """``os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")``"""
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return False
    call = node.value
    return (
        ast.unparse(call.func) == "os.environ.setdefault"
        and [ast.literal_eval(a) for a in call.args] == list(ALLOC_CONF)
        and not call.keywords
    )


@pytest.mark.parametrize("rel", ENTRY_POINTS)
def test_entry_point_sets_alloc_conf_before_torch(rel):
    """At module level the entry point sets the variable before any import that could
    import torch (only the standard library and numpy come first)."""
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        body = ast.parse(f.read()).body
    sets = [i for i, node in enumerate(body) if is_alloc_conf_setdefault(node)]
    assert len(sets) == 1, rel
    for node in body[: sets[0]]:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert name.split(".")[0] in BEFORE_ALLOC_CONF, (rel, name)


# runs an entry point's module code (not main) in a fresh interpreter and prints the
# variable as it was when torch was first imported (during that code, or right after)
PROBE = """
import importlib.abc, json, os, runpy, sys
seen = {}
class Probe(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" and "torch" not in seen:
            seen["torch"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        return None
sys.meta_path.insert(0, Probe())
runpy.run_path(sys.argv[1], run_name="probe")
import torch
print(json.dumps(seen))
"""


@pytest.mark.parametrize("preset", [None, "max_split_size_mb:128"])
@pytest.mark.parametrize("rel", ENTRY_POINTS)
def test_entry_point_alloc_conf_at_torch_import(rel, preset):
    """CPU only: when torch is imported the variable is expandable_segments:True, or the
    value the environment already had."""
    if rel.startswith("demo/"):
        pytest.importorskip("gradio")
    env = {k: v for k, v in os.environ.items() if k != ALLOC_CONF[0]}
    env["CUDA_VISIBLE_DEVICES"] = ""
    if preset is not None:
        env[ALLOC_CONF[0]] = preset
    out = subprocess.run(
        [sys.executable, "-c", PROBE, os.path.join(REPO, rel)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    seen = json.loads(out.strip().splitlines()[-1])
    assert seen == {"torch": preset or ALLOC_CONF[1]}, rel


def test_library_leaves_alloc_conf_alone():
    """The variable is the entry points' choice: no module of the ldpt package sets it."""
    for path in glob.glob(os.path.join(REPO, "ldpt", "**", "*.py"), recursive=True):
        with open(path, encoding="utf-8") as f:
            assert ALLOC_CONF[0] not in f.read(), path
