"""
The order in which PyTorch tries its scaled-dot-product attention backends: the context
the trainer runs each step in (``cfg.sdpa_backends``, unset: PyTorch's order) and
``api.predict`` runs the model in (``sdpa_backends=``, default "auto": auto_order).

    from ldpt.utils.sdpa import SDPA_ORDER, sdpa_context

    with sdpa_context(SDPA_ORDER):   # cuDNN attention first
        y = model(x)

SDPA_ORDER puts cuDNN attention first. "auto" (auto_order) runs it on GPUs of compute
capability 10.x (B200, GB200, GB300): on an NVIDIA GB200 it took less than half the
time of PyTorch's order at 32 MP and up with one encoder pass. Elsewhere "auto" keeps
PyTorch's order: on an RTX PRO 6000 Blackwell (12.0) cuDNN first is slower, and other
GPUs were not measured.
PyTorch's own order (torch 2.8: FlashAttention, memory-efficient, math, cuDNN) is also
what evaluation runs (scripts/evaluate.py, the evaluator hooks). The kernels compute
the same attention, so the order changes the output only at the rounding level. A
backend the device, the dtype or the shapes cannot use is skipped (fp32 attention has
neither a cuDNN nor a FlashAttention kernel; on the CPU and Apple MPS the order changes
nothing). Needs torch only.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import contextlib
from typing import Any, ContextManager, Optional, Sequence, Tuple, Union

__all__ = ["SDPA_BACKENDS", "SDPA_ORDER", "auto_order", "sdpa_context"]

# name -> torch.nn.attention.SDPBackend member
SDPA_BACKENDS = dict(
    cudnn="CUDNN_ATTENTION",
    flash="FLASH_ATTENTION",
    efficient="EFFICIENT_ATTENTION",
    math="MATH",
)
SDPA_ORDER = ("cudnn", "flash", "efficient", "math")
AUTO = "auto"
# compute capability majors on which "auto" runs SDPA_ORDER: 10 = B200 / GB200 / GB300
CUDNN_FIRST_MAJORS = (10,)


def auto_order(device: Any = None) -> Optional[Tuple[str, ...]]:
    """The order "auto" stands for on ``device`` (None: the current CUDA device):
    SDPA_ORDER on a CUDA GPU of compute capability 10.x (cuDNN attention first is the
    fastest on an NVIDIA GB200), else None, PyTorch's order (cuDNN first is slower on
    an RTX PRO 6000 Blackwell, 12.0; other GPUs were not measured). The CPU, Apple MPS
    and a machine without CUDA get None."""
    import torch

    if device is not None and torch.device(device).type != "cuda":
        return None
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return SDPA_ORDER if major in CUDNN_FIRST_MAJORS else None


def sdpa_context(
    backends: Union[Sequence[str], str, None] = None,
    device: Any = None,
) -> ContextManager:
    """A context that tries the scaled-dot-product attention backends in the order of
    ``backends`` (names of SDPA_BACKENDS; one name or a list), the others disabled, and
    restores the previous state on exit (torch's process-wide SDPA flags). None or an
    empty list keeps PyTorch's choice (flash, mem-efficient, math, cuDNN in torch
    2.8); "auto" is auto_order(``device``). An unknown name raises ValueError here,
    before the context is entered."""
    if backends == AUTO:
        backends = auto_order(device)
    if isinstance(backends, str):
        backends = [backends]
    if not backends:
        return contextlib.nullcontext()
    unknown = [b for b in backends if b not in SDPA_BACKENDS]
    if unknown:
        raise ValueError(
            f"sdpa_backends: unknown {unknown}, choose from {list(SDPA_BACKENDS)}"
        )
    from torch.nn.attention import SDPBackend, sdpa_kernel

    order = [getattr(SDPBackend, SDPA_BACKENDS[b]) for b in backends]
    return sdpa_kernel(order, set_priority=True)
