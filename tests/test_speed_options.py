"""
Training-speed options (CPU): model.compile_blocks and the SDPA backend order of the
trainer (ldpt/utils/sdpa.py, also api.predict's).
"""

import copy

import pytest
import torch

from test_l_dpt_golden import BASE, DEPTH_LOSS, make_data, randomize_

from ldpt.utils import sdpa
from ldpt.utils.sdpa import sdpa_context
from ldpt.models.builder import build_model

DEPTH = dict(BASE, criterion=DEPTH_LOSS)
HW = (64, 80)


def counted(model):
    """Count the calls of each block's compiled forward."""
    calls = []
    blocks = list(model.encoder.blocks) + [b for s in model.dec_blocks for b in s]
    for blk in blocks:
        fwd = blk.forward
        compiled = fwd.compiled

        def wrapper(*args, _c=compiled, **kwargs):
            calls.append(1)
            return _c(*args, **kwargs)

        fwd.compiled = wrapper
    return calls


def test_compile_blocks_train_only():
    torch._dynamo.reset()
    ref = randomize_(build_model(DEPTH))
    model = copy.deepcopy(ref)
    keys = list(model.state_dict())
    model.compile_blocks(backend="eager")
    assert list(model.state_dict()) == keys  # checkpoints load unchanged
    calls = counted(model)
    data = make_data("depth", HW)

    for m in (ref, model):
        m.train()
    torch.manual_seed(1)
    want = ref(copy.deepcopy(data))["loss"]
    torch.manual_seed(1)
    got = model(copy.deepcopy(data))["loss"]
    n_blocks = len(model.encoder.blocks) + sum(len(s) for s in model.dec_blocks)
    assert len(calls) == n_blocks
    torch.testing.assert_close(got, want)
    got.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    for m in (ref, model):
        m.eval()
    with torch.no_grad():
        want = ref(copy.deepcopy(data))["pred_depth"]
        got = model(copy.deepcopy(data))["pred_depth"]
    assert len(calls) == n_blocks  # eval: eager forward
    torch.testing.assert_close(got, want)


def test_compile_blocks_one_graph_per_block_kind():
    """Blocks of a kind share a graph: no recompilation per block (which would hit
    torch._dynamo's cache-size limit on the 40 ViT-7B blocks and fall back to eager)."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    model = randomize_(build_model(DEPTH)).train()
    model.compile_blocks(backend="eager")
    torch.manual_seed(1)
    model(make_data("depth", HW))
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    n_blocks = len(model.encoder.blocks) + sum(len(s) for s in model.dec_blocks)
    assert 2 <= graphs < n_blocks


def test_compile_blocks_options():
    model = build_model(DEPTH)
    model.compile_blocks(encoder=False, backend="eager")
    assert "forward" not in vars(model.encoder.blocks[0])
    assert "forward" in vars(model.dec_blocks[0][0])
    with pytest.raises(RuntimeError, match="compiled already"):
        model.compile_blocks(backend="eager")


def test_sdpa_context():
    q = torch.randn(1, 2, 8, 16)
    with sdpa_context(None):
        a = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    with sdpa_context(["math"]):
        b = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.testing.assert_close(a, b)
    with sdpa_context(["cudnn", "flash", "efficient", "math"]):  # CPU: cuDNN skipped
        c = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    assert torch.equal(a, c)
    with sdpa_context("math"):  # one name
        assert not torch.backends.cuda.flash_sdp_enabled()
    assert torch.backends.cuda.flash_sdp_enabled()  # restored
    with pytest.raises(ValueError, match="unknown"):
        sdpa_context(["cudnn", "fast"])


def _priority():
    from torch.nn.attention import _cur_sdpa_kernel_backends

    return [b.name for b in _cur_sdpa_kernel_backends(with_priority=True)]


@pytest.mark.parametrize(
    "capability, want",
    [((10, 0), sdpa.SDPA_ORDER), ((10, 3), sdpa.SDPA_ORDER), ((12, 0), None)]
    + [((9, 0), None), ((8, 0), None)],
)
def test_auto_order(monkeypatch, capability, want):
    """ "auto": cuDNN attention first on compute capability 10.x (B200 / GB200 / GB300),
    PyTorch's order (None) on any other GPU (12.0: RTX PRO 6000 Blackwell; 9.0, 8.0),
    on the CPU, Apple MPS and without CUDA; sdpa_context("auto") is that order."""
    torch_order = _priority()
    if not torch.cuda.is_available():  # this machine: no CUDA, "auto" is torch's order
        assert sdpa.auto_order() is None and sdpa.auto_order("cuda") is None
        with sdpa_context("auto"):
            assert _priority() == torch_order
    seen = []

    def capability_of(device=None):
        seen.append(device)
        return capability

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability_of)
    assert sdpa.auto_order() == want and seen == [None]  # the current CUDA device
    assert sdpa.auto_order("cuda:1") == want and seen[-1] == "cuda:1"
    for device in ("cpu", torch.device("cpu"), "mps"):  # not a CUDA device
        assert sdpa.auto_order(device) is None
    assert len(seen) == 2
    with sdpa_context("auto"):
        state = _priority()
    with sdpa_context(want):
        assert state == _priority()
    assert _priority() == torch_order  # restored
    if want is not None:
        assert state[0] == "CUDNN_ATTENTION"
