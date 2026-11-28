"""
scripts/average_weights.py on CPU: the weighted average (float64 sum, stored dtype), what
comes from --keep-from, the input checks, an averaged tiny-run checkpoint through
convert.py into a release file, and the uniform average of two files equal to
scripts/evaluate.py --average (BatchNorm statistics included).
"""

import importlib.util
import json
import os

import pytest
import torch

from helpers import REPO, randomize_decoder_

from ldpt.models.builder import build_model
from ldpt.models.l_dpt import convert

TINY = dict(
    encoder=dict(type="feature_stub_test", embed_dim=32),
    encoder_inter_layers=[0, 1, 2, 3],
    dec_dim=32,
    dec_num_heads=2,
    dec_num_layers=(1, 1, 1, 1),
)


def load_script():
    path = os.path.join(REPO, "scripts", "average_weights.py")
    spec = importlib.util.spec_from_file_location("average_weights_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


avg = load_script()


def test_average(tmp_path, capsys):
    """Float tensors averaged (float64 sum, stored dtype), non-float tensors,
    top-level entries and key order from --keep-from (default: the last input); the
    sidecar records the sources, weights and counts."""
    a = dict(
        epoch=4,
        state_dict={
            "module.w": torch.tensor([1.0, 2.0]),
            "module.h": torch.tensor([1.0, 3.0], dtype=torch.float16),
            "module.n": torch.tensor([3]),
        },
    )
    b = dict(
        epoch=1,
        state_dict={  # another key order: the output keeps --keep-from's
            "module.n": torch.tensor([4]),
            "module.w": torch.tensor([2.0, 5.0]),
            "module.h": torch.tensor([3.0, 5.0], dtype=torch.float16),
        },
    )
    pa, pb = tmp_path / "a.pth", tmp_path / "b.pth"
    torch.save(a, pa)
    torch.save(b, pb)
    out = tmp_path / "stage" / "model" / "epoch_1.pth"
    assert avg.main(["--inputs", str(pa), str(pb), "--output", str(out)]) == 0
    assert "wrote" in capsys.readouterr().out
    got = torch.load(out, weights_only=True)
    assert list(got) == ["epoch", "state_dict"] and got["epoch"] == 1
    sd = got["state_dict"]
    assert list(sd) == ["module.n", "module.w", "module.h"]
    assert torch.equal(sd["module.w"], torch.tensor([1.5, 3.5]))
    assert sd["module.h"].dtype == torch.float16
    assert torch.equal(sd["module.h"], torch.tensor([2.0, 4.0], dtype=torch.float16))
    assert torch.equal(sd["module.n"], torch.tensor([4]))
    rec = json.loads(out.with_name(out.name + ".avg.json").read_text())
    assert rec["weights"] == [0.5, 0.5] and rec["keep_from"] == 1
    assert rec["float_tensors"] == 2 and rec["nonfloat_differ"] == ["module.n"]
    assert rec["entries"] == {"epoch": 1}
    assert [s["path"] for s in rec["sources"]] == [str(pa), str(pb)]
    assert all(len(s["sha256"]) == 64 for s in rec["sources"])
    assert rec["rel_dist_1_vs_0"] > 0

    # weights and --keep-from
    out2 = tmp_path / "w.pth"
    argv = ["--inputs", str(pa), str(pb), "--output", str(out2)]
    assert avg.main(argv + ["--weights", "0.25", "0.75", "--keep-from", "0"]) == 0
    got = torch.load(out2, weights_only=True)
    assert got["epoch"] == 4 and list(got["state_dict"]) == [
        "module.w",
        "module.h",
        "module.n",
    ]
    assert torch.equal(got["state_dict"]["module.w"], torch.tensor([1.75, 4.25]))
    assert torch.equal(got["state_dict"]["module.n"], torch.tensor([3]))


def test_checks_write_nothing(tmp_path):
    """Different keys, shapes or dtypes, encoder keys, bad weights: exit, no file."""
    good = dict(epoch=1, state_dict={"module.w": torch.zeros(2)})
    variants = dict(
        keys=dict(epoch=1, state_dict={"module.v": torch.zeros(2)}),
        shape=dict(epoch=1, state_dict={"module.w": torch.zeros(3)}),
        dtype=dict(epoch=1, state_dict={"module.w": torch.zeros(2).half()}),
        encoder=dict(
            epoch=1,
            state_dict={"module.w": torch.zeros(2), "module.encoder.x": torch.zeros(1)},
        ),
    )
    pg = tmp_path / "good.pth"
    torch.save(good, pg)
    out = tmp_path / "out.pth"
    for name, bad in variants.items():
        pb = tmp_path / f"{name}.pth"
        torch.save(bad, pb)
        with pytest.raises(SystemExit, match="nothing written"):
            avg.main(["--inputs", str(pg), str(pb), "--output", str(out)])
        assert not out.exists(), name
    for argv in (
        ["--inputs", str(pg)],
        ["--inputs", str(pg), str(pg), "--weights", "0.6", "0.6"],
        ["--inputs", str(pg), str(pg), "--weights", "1.0"],
        ["--inputs", str(pg), str(tmp_path / "missing.pth")],
    ):
        with pytest.raises(SystemExit):
            avg.main(argv + ["--output", str(out)])
        assert not out.exists()


def test_average_converts_to_a_release_file(tmp_path):
    """An averaged tiny-run checkpoint converts into a release file (strict) holding
    the mean weights, which api.load builds."""
    from ldpt.models.l_dpt import api

    model = randomize_decoder_(build_model(dict(type="L-DPT-v1m1", **TINY)), seed=0)
    other = randomize_decoder_(build_model(dict(type="L-DPT-v1m1", **TINY)), seed=1)
    paths = []
    for i, (m, epoch) in enumerate(((model, 4), (other, 1))):
        state = {f"module.{k}": v for k, v in m.state_dict().items()}
        paths.append(tmp_path / f"epoch_{i}.pth")
        torch.save(dict(epoch=epoch, state_dict=state), paths[-1])
    out = tmp_path / "avg" / "model" / "epoch_1.pth"
    assert avg.main(["--inputs", *map(str, paths), "--output", str(out)]) == 0
    release = tmp_path / "release.pth"
    convert.convert(out, release, config=dict(type="L-DPT-v1m1", **TINY))
    _, state, info = convert.load_release(release)
    for k, v in model.state_dict().items():
        want = ((v.double() + other.state_dict()[k].double()) / 2).to(v.dtype)
        assert torch.equal(state[k], want), k
    assert info["provenance"]["epoch"] == 1
    loaded = api.load(release, device="cpu")
    image = torch.randn(1, 3, 32, 48)
    with torch.no_grad():
        assert torch.isfinite(loaded.infer(image)).all()


def test_uniform_average_equals_evaluate_average(tmp_path):
    """Two decoder files of a segmentation model with tap BatchNorms (the COCO-Stuff
    recipe 1): the script's uniform average holds the same tensors, bit for bit, as
    scripts/evaluate.py --average (ldpt.engines.evaluate.average_states) of the same
    files: weights and BatchNorm running statistics averaged (round((x0 + x1) / 2) in
    fp32 both ways, also for values far apart in magnitude), num_batches_tracked from
    the last file. The input order changes num_batches_tracked only."""
    from ldpt.engines.evaluate import average_states

    seg = dict(
        TINY,
        type="L-DPT-v1m1",
        task="semseg",
        out_dim=8,
        num_classes=5,
        criterion=[dict(type="ce_loss", loss_weight=1.0, ignore_index=-1)],
        tap_bn=True,
        dec_qk_norm=True,
    )
    paths = []
    for i, (seed, epoch) in enumerate(((0, 9), (1, 1))):
        model = randomize_decoder_(build_model(seg), seed=seed)
        g = torch.Generator().manual_seed(10 + seed)
        state = {}
        for k, v in model.state_dict().items():
            if k.startswith("encoder."):
                continue
            if k.endswith(("running_mean", "running_var")):
                v = torch.rand(v.shape, generator=g) * 10.0 ** (2 * seed - 1)
            elif k.endswith("num_batches_tracked"):
                v = torch.tensor(1000 * (i + 1))
            elif v.is_floating_point() and k.endswith("seg_head.weight"):
                # magnitudes 1e-30 ... 1e30: rounding of the sum in both ways
                v = torch.randn(v.shape, generator=g) * 10.0 ** torch.randint(
                    -30, 31, v.shape, generator=g
                )
            state[f"module.{k}"] = v
        paths.append(tmp_path / f"epoch_{i}.pth")
        torch.save(dict(epoch=epoch, state_dict=state), paths[-1])
    out = tmp_path / "average.pth"
    assert avg.main(["--inputs", *map(str, paths), "--output", str(out)]) == 0
    script = {
        k[len("module.") :]: v
        for k, v in torch.load(out, weights_only=True)["state_dict"].items()
    }
    evaluate = average_states(paths)
    assert script.keys() == evaluate.keys()
    assert any(k.startswith("tap_bn.") and "running" in k for k in script)
    for k, v in script.items():
        assert v.dtype == evaluate[k].dtype and torch.equal(v, evaluate[k]), k
    assert int(script["tap_bn.0.num_batches_tracked"]) == 2000  # the last file's
    swapped = average_states(paths[::-1])
    for k, v in script.items():
        if k.endswith("num_batches_tracked"):
            assert int(swapped[k]) == 1000
        else:
            assert torch.equal(v, swapped[k]), k
