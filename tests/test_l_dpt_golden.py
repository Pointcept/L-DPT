"""
Golden outputs of L-DPT-v1m1 (tiny DINOv3-7B-shaped encoder, CPU fp32, fixed seeds).

tests/data/l_dpt_golden.pth pins outputs, losses and gradients of the implementation
that the checkpoints were trained with; every case must stay bit-identical. The depth
cases train with the L-DPT depth loss as the depth recipes configure it (label types,
trims, local levels, gradient term on type A), at the toy size.

Regenerate only on purpose (a deliberate change of the model's math), and say so in
the commit:
    CUDA_VISIBLE_DEVICES="" python tests/test_l_dpt_golden.py

The pins were recorded on an Intel x86_64 CPU and are compared bit for bit on Intel
CPUs. Other x86_64 CPUs (AMD EPYC) take other kernel paths (differences up to 3e-6 in
fp32, 4e-4 in the bf16 attention case), so there the cases are compared with
tolerances; aarch64 rounds the bf16 autocast kernels differently again (~2e-3), so the
cases are skipped there.
"""

import os
import platform
import sys

import pytest
import torch
import torch.nn.functional as F

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [TESTS, os.path.dirname(TESTS)]  # also run as a script

from helpers import init_encoder_  # noqa: E402

from ldpt.models.builder import build_model  # noqa: E402

pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64"),
    reason="bitwise pins recorded on an x86_64 CPU",
)

PATH = os.path.join(TESTS, "data", "l_dpt_golden.pth")


def _cpu_vendor() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("vendor_id"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return (
        platform.processor()
    )  # Windows: "Intel64 Family ..." / "AMD64 ... AuthenticAMD"


# bit for bit on the CPU family the pins were recorded on, else within tolerances
EXACT = "intel" in _cpu_vendor().lower()
THREADS = 4  # fixed intra-op threads: fp32 reduction order depends on it

ENC = dict(encoder=dict(type="dinov3_tiny_test"), encoder_inter_layers=[0, 1, 2, 3])
DEC = dict(
    dec_dim=64,
    dec_num_heads=4,
    dec_num_layers=(2, 1, 2, 1),
    dec_attn_downscale=2,
    dec_pos_embed_rope_jitter_coords=1.2,
    dec_pos_embed_rope_rescale_coords=8,
)
# the loss of configs/hybrid/*.py; min_valid, roe_points and local_cell scaled to the
# toy images
DEPTH_LOSS = dict(
    type="relative_depth_loss",
    mage_weight=4.0,
    mage_num_scales=4,
    align="roe",
    min_valid=100,
    roe_points=256,
    label_types=dict(A=["hypersim"], B=["blendedmvs"], C=["kitti"]),
    type_trim=dict(A=0.0, B=0.05, C=0.05),
    local_levels=dict(A=[4, 16, 64], B=[4, 16], C=[4]),
    local_cell=64.0,
    grad_types=["A"],
    log_logit_stats=True,
)
# sample names of make_data's depth batches (label types A, B, C, A, ...)
DEPTH_NAMES = ("hypersim/scene/0", "blendedmvs/scene/1", "kitti/scene/2")
SEG_LOSS = [
    dict(type="ce_loss", loss_weight=1.0, ignore_index=-1),
    dict(type="lovasz_loss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
]
BASE = dict(type="L-DPT-v1m1", **ENC, **DEC)
# name -> (model kwargs, task, what to record)
CASES = {
    "depth": (dict(BASE, criterion=DEPTH_LOSS), "depth", ("eval", "loss", "train")),
    "depth_shared_k1": (
        dict(BASE, criterion=DEPTH_LOSS, selection="shared", readout_kernel_size=1),
        "depth",
        ("eval", "train"),
    ),
    "depth_encoder_eval": (
        dict(BASE, criterion=DEPTH_LOSS, encoder_train_mode=False),
        "depth",
        ("train",),
    ),
    "depth_attn_bf16": (dict(BASE, attn_dtype="bf16"), "depth", ("eval",)),
    "semseg": (
        dict(BASE, task="semseg", out_dim=8, num_classes=5, criterion=SEG_LOSS),
        "segment",
        ("eval", "loss", "train"),
    ),
}
EVAL_HW = [(40, 56), (32, 48)]  # padded, unpadded
GT_HW = (44, 60)  # depth GT at another size: bilinear resize in the head


@torch.no_grad()
def randomize_(model, seed=0):
    """Encoder: DINOv3 init; decoder: random by sorted parameter name (no dependence on
    module registration order, so a refactor that keeps the keys keeps the weights)."""
    init_encoder_(model.encoder, seed)
    g = torch.Generator().manual_seed(seed + 1)
    params = dict(model.named_parameters())
    for name in sorted(params):
        if name.startswith("encoder."):
            continue
        p = params[name]
        if name.endswith("gamma"):
            p.copy_(torch.rand(p.shape, generator=g) * 0.45 + 0.05)
        elif "norm" in name and name.endswith("weight"):
            p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
        elif name.endswith("bias"):
            p.copy_(0.05 * torch.randn(p.shape, generator=g))
        else:
            fan_in = p[0].numel() if p.ndim > 1 else p.numel()
            p.copy_(torch.randn(p.shape, generator=g) / fan_in**0.5)
    return model


def make_data(task, hw, gt_hw=None, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = dict(image=torch.randn(batch, 3, *hw, generator=g))
    if task == "depth":
        depth = torch.rand(batch, 1, *(gt_hw or hw), generator=g) * 10 + 0.5
        depth[torch.rand(depth.shape, generator=g) < 0.2] = 0.0
        data["depth"] = depth
        data["name"] = [DEPTH_NAMES[i % len(DEPTH_NAMES)] for i in range(batch)]
    elif task == "segment":
        segment = torch.randint(0, 5, (batch, 1, *hw), generator=g)
        segment[torch.rand(segment.shape, generator=g) < 0.1] = -1
        data["segment"] = segment
    return data


def losses(out):
    return {k: v.detach().clone() for k, v in out.items() if k.endswith("loss")}


@torch.no_grad()
def correlated_depth(model, data, seed):
    """data with its valid depth replaced by an affine map of the model's prediction
    in eval (resized bilinearly) with 5 % noise: the loss's fits are accepted (half
    of the fits of a random model on random depth have a negative scale and drop the
    image). Leaves the model in eval."""
    model.eval()
    pred = model(dict(image=data["image"]))["pred_depth"]
    depth = data["depth"]
    pred = F.interpolate(
        pred, size=depth.shape[-2:], mode="bilinear", align_corners=False
    )
    g = torch.Generator().manual_seed(seed)
    noise = 1.0 + 0.05 * torch.randn(depth.shape, generator=g)
    depth = torch.where(depth > 0, (0.5 + 10.0 * pred) * noise, 0.0)
    return dict(data, depth=depth)


def outputs():
    """Every recorded tensor of every case, under THREADS intra-op threads."""
    threads = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    try:
        result = {}
        for name, (cfg, task, kinds) in CASES.items():
            model = randomize_(build_model(cfg))
            key = "pred_segment" if task == "segment" else "pred_depth"
            rec = {}
            if "eval" in kinds:
                model.eval()
                for hw in EVAL_HW:
                    with torch.no_grad():
                        out = model(make_data(None, hw))
                    rec[f"eval_{hw[0]}x{hw[1]}"] = out[key]
            if "loss" in kinds:
                model.eval()
                if task == "depth":  # label types A, B, C
                    data = make_data(task, EVAL_HW[0], GT_HW, batch=3, seed=1)
                    data = correlated_depth(model, data, seed=11)
                else:
                    data = make_data(task, EVAL_HW[0], seed=1)
                torch.manual_seed(7)  # the depth loss samples pixels
                with torch.no_grad():
                    out = model(data)
                rec.update({f"eval_{k}": v for k, v in losses(out).items()})
            if "train" in kinds:
                if task == "depth":
                    data = make_data(task, EVAL_HW[0], batch=3, seed=2)
                    data = correlated_depth(model, data, seed=12)
                else:
                    data = make_data(task, EVAL_HW[0], seed=2)
                model.train()
                model.zero_grad(set_to_none=True)
                torch.manual_seed(123)
                out = model(data)
                out["loss"].backward()
                rec.update({f"train_{k}": v for k, v in losses(out).items()})
                # per decoder parameter (sorted names): sum and sum of |grad| (fp64)
                grads = {
                    n: p.grad for n, p in model.named_parameters() if p.requires_grad
                }
                rec["train_grad_names"] = sorted(grads)
                rec["train_grad_sums"] = torch.stack(
                    [
                        torch.stack(
                            [grads[n].double().sum(), grads[n].double().abs().sum()]
                        )
                        for n in sorted(grads)
                    ]
                )
            result[name] = rec
        return result
    finally:
        torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def current():
    return outputs()


@pytest.mark.parametrize("name", list(CASES))
def test_golden(name, current):
    golden = torch.load(PATH, weights_only=True)[name]
    mine = current[name]
    assert mine.keys() == golden.keys()
    assert mine.pop("train_grad_names", None) == golden.pop("train_grad_names", None)
    for k, ref in golden.items():
        assert mine[k].dtype == ref.dtype and mine[k].shape == ref.shape, k
        if EXACT:
            diff = (mine[k] - ref).abs().max().item()
            assert torch.equal(mine[k], ref), (name, k, diff)
        else:  # one bf16 rounding step in the bf16 case, fp32 noise elsewhere
            tol = 1e-2 if name.endswith("bf16") else 1e-5
            torch.testing.assert_close(mine[k], ref, rtol=tol, atol=tol)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    data = outputs()
    torch.save(data, PATH)
    n = sum(len(v) for v in data.values())
    print(f"wrote {PATH}: {len(data)} cases, {n} tensors")
