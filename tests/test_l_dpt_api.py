"""
convert (training checkpoint -> release file), weights (pinned files) and api (load,
predict) on CPU with a parameter-free stub encoder (the released files: see
test_l_dpt_weights.py).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from helpers import init_encoder_, randomize_decoder_

from ldpt.models.builder import build_model
from ldpt.models.l_dpt import api, convert, presets, weights
from ldpt.models.l_dpt.presets import IMAGE_MEAN, IMAGE_STD, WEIGHTS, get_preset

STUB = dict(type="feature_stub_test", embed_dim=32)
TINY = dict(
    encoder=STUB,
    encoder_inter_layers=[0, 1, 2, 3],
    dec_dim=32,
    dec_num_heads=2,
    dec_num_layers=(1, 2, 1, 1),
    dec_pos_embed_rope_jitter_coords=1.2,
    dec_pos_embed_rope_rescale_coords=8,
)


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LDPT_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def dev_ckpt(tmp_path):
    """A trainer-style checkpoint of a tiny model: (model, path)."""
    model = randomize_decoder_(build_model(dict(type="L-DPT-v1m1", **TINY)))
    path = tmp_path / "model_best.pth"
    torch.save(
        dict(
            epoch=3,
            state_dict={f"module.{k}": v for k, v in model.state_dict().items()},
            optimizer=dict(state={}, param_groups=[]),
            scheduler=dict(last_epoch=7),
            best_metric_value=0.5,
        ),
        path,
    )
    return model.eval(), path


@pytest.fixture
def release(dev_ckpt, tmp_path):
    model, src = dev_ckpt
    dst = tmp_path / "release.pth"
    convert.convert(src, dst, config=dict(type="L-DPT-v1m1", **TINY))
    return model, dst


def test_convert_round_trip(dev_ckpt, tmp_path):
    model, src = dev_ckpt
    dst = tmp_path / "out" / "release.pth"
    info = convert.convert(src, dst, config=dict(type="L-DPT-v1m1", **TINY))
    assert info["epoch"] == 3 and info["best_metric_value"] == 0.5
    assert info["dropped"] == ["optimizer", "scheduler"]
    assert info["source_sha256"] == weights.sha256_of(src)
    config, state, rel = convert.load_release(dst)
    assert rel["preset"] is None and config == convert.model_kwargs(TINY)
    assert state.keys() == model.state_dict().keys()
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    with pytest.raises(convert.ConvertError, match="already a release file"):
        convert.convert(dst, tmp_path / "again.pth", config=TINY)


def test_convert_is_strict(dev_ckpt, tmp_path):
    _, src = dev_ckpt
    with pytest.raises(convert.ConvertError, match="missing .*shape mismatch"):
        convert.convert(src, tmp_path / "x.pth", preset="l-dpt_large_depth_zeroshot")
    with pytest.raises(ValueError, match="exactly one"):
        convert.convert(src, tmp_path / "x.pth")
    assert not (tmp_path / "x.pth").exists()


def test_convert_cli(dev_ckpt, tmp_path, capsys):
    _, src = dev_ckpt
    cfg = tmp_path / "cfg.py"
    cfg.write_text(f"model = dict(type='L-DPT-v1m1', **{TINY!r})\n")
    convert.main([str(src), str(tmp_path / "cli.pth"), "--config", str(cfg)])
    assert "wrote" in capsys.readouterr().out
    assert convert.load_release(tmp_path / "cli.pth")[0] == convert.model_kwargs(TINY)


def test_load_release(release):
    model, path = release
    loaded = api.load(path, device="cpu")
    assert not loaded.training and loaded.precision == "fp32" and loaded.preset is None
    assert loaded.weights_info["provenance"]["epoch"] == 3
    assert loaded.rope.normalize_coords == "min"
    trained = api.load(
        path, device="cpu", dec_pos_embed_rope_normalize_coords="separate"
    )
    image = torch.randn(2, 3, 40, 56)
    square = torch.randn(2, 3, 48, 48)
    with torch.no_grad():
        assert torch.equal(model.infer(image), trained.infer(image))
        # RoPE "min" (api default) == "separate" (presets) on square inputs only
        assert torch.equal(model.infer(square), loaded.infer(square))
        assert not torch.equal(model.infer(image), loaded.infer(image))
    bf16 = api.load(path, device="cpu", precision="bf16")
    assert bf16.precision == "bf16"
    with torch.no_grad():
        assert (bf16.infer(image) - loaded.infer(image)).abs().max() < 2e-2


# an exp-head model with the L-DPT depth loss: its criterion must name the head (the
# default loss names "sigmoid") and routes by sample name
EXP_LOSS = dict(
    type="relative_depth_loss",
    align="roe",
    head="exp",
    label_types=dict(A=["hypersim"], B=[], C=[]),
    grad_types=["A"],
    mage_weight=4.0,
)
TINY_EXP = dict(TINY, head="exp", readout_fp32=True)


def test_release_file_has_no_criterion(tmp_path, monkeypatch):
    """convert stores the config without the criterion (FORMAT /2); api.load builds
    the exp-head model inference-only, without running any loss code, bitwise the
    model built with its training criterion."""
    cfg = dict(type="L-DPT-v1m1", **TINY_EXP, criterion=EXP_LOSS)
    model = randomize_decoder_(build_model(cfg)).eval()
    torch.save(dict(epoch=7, state_dict=model.state_dict()), tmp_path / "dev.pth")
    dst = tmp_path / "rel.pth"
    convert.convert(tmp_path / "dev.pth", dst, config=cfg)
    raw = weights.read_checkpoint(dst)
    assert raw["format"] == convert.FORMAT == "l-dpt-release/2"
    assert "criterion" not in raw["config"] and raw["config"]["head"] == "exp"
    assert raw["config"] == convert.release_kwargs(cfg)
    assert not any(k.startswith("criterion.") for k in raw["state_dict"])

    from ldpt.models.l_dpt import l_dpt_v1m1_base

    def no_loss(cfg):
        raise AssertionError(f"a loss was built: {cfg}")

    monkeypatch.setattr(l_dpt_v1m1_base.LOSSES, "build", no_loss)
    config, _, info = convert.load_release(dst)
    assert info["format"] == convert.FORMAT and "criterion" not in config
    loaded = api.load(dst, device="cpu", dec_pos_embed_rope_normalize_coords="separate")
    assert loaded.criterion is None and loaded.head == "exp"
    image = torch.randn(2, 3, 40, 56)
    with torch.no_grad():
        assert torch.equal(loaded.infer(image), model.infer(image))
        out = loaded(dict(image=image, depth=torch.rand(2, 1, 40, 56) + 0.5))
    assert set(out) == {"pred_depth"}  # target given in eval: no loss
    with pytest.raises(ValueError, match="criterion=False"):
        loaded.train()(dict(image=image, depth=torch.rand(2, 1, 40, 56) + 0.5))


def test_old_release_format_loads(dev_ckpt, tmp_path):
    """An l-dpt-release/1 file (the config holds the training criterion)
    still loads, inference-only, bitwise; ensure_release converts it again as /2."""
    model, src = dev_ckpt
    old_loss = dict(type="relative_depth_loss", mage_weight=0.1, male_weight=0.01)
    old = tmp_path / "old.pth"
    torch.save(
        dict(
            format="l-dpt-release/1",
            model="L-DPT-v1m1",
            preset=None,
            config=dict(convert.model_kwargs(TINY), criterion=old_loss),
            state_dict=dict(model.state_dict()),
            provenance=dict(
                source="model_best.pth", source_sha256=weights.sha256_of(src), epoch=3
            ),
        ),
        old,
    )
    config, state, info = convert.load_release(old)
    assert info["format"] == "l-dpt-release/1" and info["provenance"]["epoch"] == 3
    assert config == convert.model_kwargs(TINY)  # the criterion is dropped
    assert state.keys() == model.state_dict().keys()
    loaded = api.load(old, device="cpu", dec_pos_embed_rope_normalize_coords="separate")
    assert loaded.criterion is None
    image = torch.randn(2, 3, 40, 56)
    with torch.no_grad():
        assert torch.equal(loaded.infer(image), model.infer(image))
    # a preset load of a /1 file compares the architecture too
    with pytest.raises(ValueError, match="another architecture"):
        api.load("l-dpt_large_depth_zeroshot", device="cpu", weights=old)
    status, _ = convert.ensure_release(src, old, config=TINY)
    assert status == "stale: format l-dpt-release/1, not l-dpt-release/2"
    assert weights.read_checkpoint(old)["format"] == convert.FORMAT
    assert convert.ensure_release(src, old, config=TINY)[0] == "up to date"


@pytest.mark.parametrize("old, new", list(presets.LEGACY_PRESETS.items()))
def test_legacy_preset_name_in_a_release_file(
    old, new, dev_ckpt, tmp_path, monkeypatch
):
    """A release file that stores the short preset name "l-zs" / "l-mix" (as the files
    on Hugging Face do): load_release and api.load (by path, or with the preset and
    weights=) read it as the current name, ensure_release finds it up to date for
    that name, and the file is not rewritten. The preset is a tiny stand-in here."""
    model, src = dev_ckpt
    monkeypatch.setitem(presets.PRESETS, new, dict(TINY))
    fresh, dst = tmp_path / "fresh.pth", tmp_path / f"{new}.pth"
    convert.convert(src, fresh, preset=new)
    # another file: read_checkpoint memory-maps fresh, overwriting it would fault
    torch.save(dict(weights.read_checkpoint(fresh), preset=old), dst)
    written = dst.read_bytes()
    assert convert.load_release(dst)[2]["preset"] == new
    loaded = api.load(dst, device="cpu")
    assert loaded.preset == new
    assert api.load(new, device="cpu", weights=dst).preset == new
    assert convert.ensure_release(src, dst, preset=new)[0] == "up to date"
    assert dst.read_bytes() == written
    assert weights.read_checkpoint(dst)["preset"] == old
    square = torch.randn(1, 3, 48, 48)
    with torch.no_grad():
        assert torch.equal(loaded.infer(square), model.infer(square))


@pytest.fixture
def tiny_preset(monkeypatch):
    """A preset of the tiny sigmoid model (registered for the test): its name."""
    monkeypatch.setitem(presets.PRESETS, "tiny-test", dict(TINY))
    monkeypatch.setitem(presets.INPUT_SIZE, "tiny-test", 64)
    return "tiny-test"


EXP_RUN = dict(TINY_EXP, attn_fp32="last_stage", dec_qk_bound=1000.0, precision="bf16")


@pytest.fixture
def exp_run(tmp_path):
    """A training run of the tiny exp-head model in the trainer's layout:
    (model, <run>/model/epoch_3.pth, <run>/config.py)."""
    cfg = dict(type="L-DPT-v1m1", **EXP_RUN, criterion=EXP_LOSS)
    model = randomize_decoder_(build_model(cfg)).eval()
    run = tmp_path / "run"
    (run / "model").mkdir(parents=True)
    ckpt = run / "model" / "epoch_3.pth"
    state = {f"module.{k}": v for k, v in model.state_dict().items()}
    torch.save(dict(epoch=3, state_dict=state), ckpt)
    (run / "config.py").write_text(f"model = {cfg!r}\n")
    return model, ckpt, run / "config.py"


def test_precision_default_is_the_files(exp_run, tmp_path):
    """precision=None (default): the release file's / preset's precision, as trained;
    an explicit value wins."""
    model, ckpt, config_file = exp_run
    dst = tmp_path / "rel.pth"
    convert.main([str(ckpt), str(dst), "--config", str(config_file)])
    assert api.load(dst, device="cpu").precision == "bf16"
    assert api.load(dst, device="cpu", precision="fp32").precision == "fp32"
    with pytest.raises(ValueError, match="precision"):
        api.load(dst, device="cpu", precision="fp16")
    loaded = api.load(dst, device="cpu", dec_pos_embed_rope_normalize_coords="separate")
    image = torch.randn(1, 3, 40, 56)
    with torch.no_grad():
        assert torch.equal(loaded.infer(image), model.infer(image))


def test_preset_release_file_architecture(exp_run, tiny_preset, tmp_path):
    """A release file loaded with a preset must have the preset's architecture after
    the overrides: the exp head, fp32 island and bound of the file are not dropped
    silently; with them passed as overrides it loads, bitwise the load by path."""
    model, ckpt, config_file = exp_run
    dst = tmp_path / "rel.pth"
    convert.main([str(ckpt), str(dst), "--config", str(config_file)])
    with pytest.raises(ValueError, match="another architecture") as err:
        api.load(tiny_preset, device="cpu", weights=dst)
    for key in ("head", "readout_fp32", "attn_fp32", "dec_qk_bound"):
        assert f"{key}: file" in str(err.value)
    match = {k: EXP_RUN[k] for k in ("head", "readout_fp32", "attn_fp32")}
    with pytest.raises(ValueError, match="dec_qk_bound: file 1000.0 vs preset None"):
        api.load(tiny_preset, device="cpu", weights=dst, **match)
    loaded = api.load(
        tiny_preset, device="cpu", weights=dst, dec_qk_bound=1000, **match
    )
    assert loaded.preset == tiny_preset and loaded.precision == "fp32"  # the preset's
    by_path = api.load(dst, device="cpu", precision="fp32")
    image = torch.randn(1, 3, 40, 56)
    with torch.no_grad():
        assert torch.equal(by_path.infer(image), loaded.infer(image))


def test_preset_dev_checkpoint_checked_against_run_config(
    exp_run, tiny_preset, tmp_path
):
    """weights=<training checkpoint>: compared with <run>/config.py (found beside the
    model/ folder, or run_config=); run_config=False skips the check (the preset and
    overrides must then be the run's model)."""
    model, ckpt, config_file = exp_run
    with pytest.raises(
        ValueError, match=r"was trained \(per .*config.py\) with another"
    ):
        api.load(tiny_preset, device="cpu", weights=ckpt)
    opts = {
        k: EXP_RUN[k] for k in ("head", "readout_fp32", "attn_fp32", "dec_qk_bound")
    }
    loaded = api.load(
        tiny_preset,
        device="cpu",
        precision="bf16",
        weights=ckpt,
        dec_pos_embed_rope_normalize_coords="separate",
        **opts,
    )
    assert loaded.weights_info["run_config"] == str(config_file)
    image = torch.randn(1, 3, 40, 56)
    with torch.no_grad():
        assert torch.equal(loaded.infer(image), model.infer(image))
    # the silent path, by request only
    silent = api.load(tiny_preset, device="cpu", weights=ckpt, run_config=False)
    assert silent.head == "sigmoid" and "run_config" not in silent.weights_info
    # a checkpoint outside the trainer's layout: checked only with run_config=
    moved = tmp_path / "moved.pth"
    moved.write_bytes(ckpt.read_bytes())
    assert api.load(tiny_preset, device="cpu", weights=moved).head == "sigmoid"
    with pytest.raises(ValueError, match="another architecture"):
        api.load(tiny_preset, device="cpu", weights=moved, run_config=config_file)
    with pytest.raises(ValueError, match="go with a preset name"):
        api.load(moved, device="cpu", run_config=config_file)


@pytest.mark.parametrize(
    "key, value",
    [
        ("head", "exp"),
        ("readout_fp32", True),
        ("attn_fp32", "last"),
        ("block_fp32", "last_stage"),
        ("dec_qk_bound", 200),
        ("dec_qk_norm", True),
        ("dec_qk_tau_max", 100.0),
        ("dec_conv_mixer", "last"),
        ("attn_dtype", "bf16"),
        ("dec_pos_embed_rope_base", 10),
        ("dec_pos_embed_rope_eval_scale", 2.0),
        ("freeze_encoder", False),
        ("tap_bn", True),
    ],
)
def test_arch_keys(key, value):
    """Every option that changes the output is compared (a missing key = the model
    default); the legacy attn_impl and the "None" bound string resolve first. A
    per-load option (_LOAD_KEYS) the caller passes is the caller's choice."""
    assert key in api._ARCH_KEYS
    base = convert.model_kwargs(TINY)
    api._check_same_architecture(base, dict(base), "f", "p", "f")
    with pytest.raises(ValueError, match=f"{key}: file"):
        api._check_same_architecture(dict(base, **{key: value}), base, "f", "p", "f")
    api._check_same_architecture(
        dict(base, attn_impl="flash", dec_qk_bound="None"),
        dict(base, attn_dtype="bf16", dec_num_layers=[1, 2, 1, 1]),
        "f",
        "p",
        "f",
    )
    chosen = {key} if key in api._LOAD_KEYS else set()
    if chosen:
        api._check_same_architecture(
            dict(base, **{key: value}), base, "f", "p", "f", chosen=chosen
        )
    else:  # a recipe key is compared even when passed
        with pytest.raises(ValueError, match=f"{key}: file"):
            api._check_same_architecture(
                dict(base, **{key: value}), base, "f", "p", "f", chosen={key}
            )


def test_encoder_weight_dtype_not_compared():
    """encoder_weight_dtype is bit-identical under the frozen encoder's bf16 autocast
    and api.load picks it per load (bf16 unless overridden): not an _ARCH_KEYS key, so
    a run config and a preset that differ only in encoder_weight_dtype pass the
    check, and an override restating load's default does too."""
    assert "encoder_weight_dtype" not in api._ARCH_KEYS
    base = convert.model_kwargs(TINY)
    api._check_same_architecture(
        dict(base, encoder_weight_dtype="bf16"), base, "f", "p", "f"
    )
    api._check_same_architecture(
        base, dict(base, encoder_weight_dtype="bf16"), "f", "p", "f"
    )


REPO = Path(__file__).resolve().parents[1]
RECIPE_PRESETS = {  # depth recipe -> the preset of its released model
    "depth-l-dpt-v1m1-0-large-zeroshot.py": "l-dpt_large_depth_zeroshot",
    "depth-l-dpt-v1m1-1-large-mixed.py": "l-dpt_large_depth_mixed",
    "depth-l-dpt-v1m1-1a-large-mixed-anneal.py": "l-dpt_large_depth_mixed",
}


@pytest.mark.parametrize("config, preset", list(RECIPE_PRESETS.items()))
def test_repo_configs_pass_their_preset_check(config, preset):
    """The model of every depth recipe passes its preset's check, so a run trained
    from the recipe loads with its preset name (the run_config check reads exactly
    this dict from <run>/config.py)."""
    model = api._model_of(REPO / "configs" / "hybrid" / config)
    api._check_same_architecture(model, get_preset(preset), "f", preset, "f")


@pytest.mark.parametrize(
    "config",
    ["semseg-l-dpt-v1m1-1-large.py", "semseg-l-dpt-v1m1-1a-large-anneal.py"],
)
def test_cocostuff_recipe_passes_the_semseg_preset_check(config):
    """The COCO-Stuff recipe of the segmentation preset (1 and its anneal 1a) passes
    the preset's check, and keeps the "separate" RoPE coordinates it was trained and
    validated with."""
    model = api._model_of(REPO / "configs" / "cocostuff" / config)
    preset = "l-dpt_large_semseg_cocostuff"
    api._check_same_architecture(model, get_preset(preset), "f", preset, "f")
    assert model["dec_pos_embed_rope_normalize_coords"] == "separate"


def test_recipe_run_loads_with_its_preset(tmp_path):
    """A run of configs/hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py (decoder shrunk to
    64 / 4 heads, in the run's config.py and in the load's overrides alike) in the
    trainer's layout (<run>/config.py, <run>/model/epoch_7.pth) loads with preset
    "l-dpt_large_depth_zeroshot", its run_config checked (exp head, fp32 islands, logit
    bound), bitwise the trained model (stub encoder in both)."""
    from ldpt.utils.config import Config

    stub = dict(type="feature_stub_test")
    small = dict(dec_dim=64, dec_num_heads=4)
    cfg = Config.fromfile(
        str(REPO / "configs/hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py")
    )
    model_cfg = dict(cfg.model.to_dict(), encoder=stub, **small)
    assert model_cfg["encoder_weight_dtype"] == "bf16"
    model = randomize_decoder_(build_model(model_cfg)).eval()
    run = tmp_path / "run"
    (run / "model").mkdir(parents=True)
    (run / "config.py").write_text(f"model = {model_cfg!r}\n")
    ckpt = run / "model" / "epoch_7.pth"
    state = {f"module.{k}": v for k, v in model.state_dict().items()}
    torch.save(dict(epoch=7, state_dict=state), ckpt)
    loaded = api.load(
        "l-dpt_large_depth_zeroshot", device="cpu", weights=ckpt, encoder=stub, **small
    )
    assert loaded.precision == "bf16" and loaded.head == "exp"
    assert loaded.weights_info["run_config"] == str(run / "config.py")
    image = torch.randn(1, 3, 48, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(loaded.infer(image), model.infer(image))


@pytest.fixture
def exp_preset(monkeypatch):
    """A preset of the tiny exp-head run's architecture (EXP_RUN): its name."""
    monkeypatch.setitem(presets.PRESETS, "tiny-exp-test", dict(EXP_RUN))
    monkeypatch.setitem(presets.INPUT_SIZE, "tiny-exp-test", 64)
    return "tiny-exp-test"


@pytest.mark.parametrize(
    "option, changes_output",
    [
        (dict(attn_fp32="all"), True),
        # under the bf16 autocast q / k / v are bf16 already: same output
        (dict(attn_dtype="fp32"), False),
        (dict(attn_impl="flash"), False),  # = attn_dtype "bf16", the file's
        (dict(block_fp32="last_stage"), True),
        (dict(readout_fp32=False), True),
        (dict(dec_pos_embed_rope_eval_scale=2.0), True),
    ],
)
def test_preset_load_options_are_the_callers(
    exp_run, exp_preset, tmp_path, option, changes_output
):
    """A per-load option (_LOAD_KEYS: attention dtype, fp32 islands, RoPE eval scale)
    passed with a preset is the caller's choice, as with a load by path: the preset
    load of a release file and of a training checkpoint (run_config checked) take it,
    bitwise the load by path; a recipe key passed against the file still raises."""
    model, ckpt, config_file = exp_run
    dst = tmp_path / "rel.pth"
    convert.main([str(ckpt), str(dst), "--config", str(config_file)])
    by_path = api.load(dst, device="cpu", **option)
    image = torch.randn(1, 3, 40, 56)
    with torch.no_grad():
        expected = by_path.infer(image)
        default = api.load(dst, device="cpu").infer(image)
        assert torch.equal(default, expected) != changes_output
        for weights_file in (dst, ckpt):
            loaded = api.load(exp_preset, device="cpu", weights=weights_file, **option)
            assert torch.equal(loaded.infer(image), expected)
    assert api.load(exp_preset, device="cpu", weights=ckpt).weights_info[
        "run_config"
    ] == str(config_file)
    with pytest.raises(ValueError, match="head: file 'exp' vs preset 'sigmoid'"):
        api.load(exp_preset, device="cpu", weights=dst, head="sigmoid", **option)
    with pytest.raises(ValueError, match="dec_qk_bound: file 1000.0 vs preset 200"):
        api.load(exp_preset, device="cpu", weights=ckpt, dec_qk_bound=200, **option)


def test_preset_load_option_against_another_architecture(exp_run, tiny_preset):
    """A per-load option passed to a preset that does not describe the file does not
    hide the other differences."""
    _, ckpt, _ = exp_run
    with pytest.raises(ValueError, match="another architecture") as err:
        api.load(tiny_preset, device="cpu", weights=ckpt, attn_fp32="last_stage")
    assert "attn_fp32" not in str(err.value)
    for key in ("head", "readout_fp32", "dec_qk_bound"):
        assert f"{key}: file" in str(err.value)


def test_run_config_without_the_config_parser(
    exp_run, tiny_preset, monkeypatch, caplog
):
    """An inference-only install (no addict / yapf): the implicit run_config check of a
    checkpoint in the trainer's layout is skipped with a warning; an explicit
    run_config= still needs the parser."""
    import sys

    _, ckpt, config_file = exp_run
    monkeypatch.setitem(sys.modules, "addict", None)
    monkeypatch.delitem(sys.modules, "ldpt.utils.config", raising=False)
    with pytest.raises(ImportError):
        api._model_of(config_file)
    with caplog.at_level("WARNING", logger=api.__name__):
        loaded = api.load(tiny_preset, device="cpu", weights=ckpt)
    assert loaded.head == "sigmoid" and "run_config" not in loaded.weights_info
    assert "not checked against" in caplog.text and str(config_file) in caplog.text
    with pytest.raises(ImportError):
        api.load(tiny_preset, device="cpu", weights=ckpt, run_config=config_file)


def test_training_loss_keys_are_dropped(dev_ckpt, tiny_preset, tmp_path):
    """A training checkpoint whose loss has state (criterion.*): convert writes a
    release file without criterion.* keys and api.load(preset, weights=<that
    checkpoint>) loads strictly, bitwise the model."""
    model, src = dev_ckpt
    ckpt = torch.load(src, weights_only=True)
    ckpt["state_dict"]["module.criterion.net.w"] = torch.zeros(3)
    ckpt["state_dict"]["module.criterion.scaling_layer.shift"] = torch.ones(3)
    stateful = tmp_path / "stateful" / "model_best.pth"
    stateful.parent.mkdir()
    torch.save(ckpt, stateful)
    assert not any(k.startswith("criterion.") for k in weights.decoder_state(ckpt))
    dst = tmp_path / "rel.pth"
    convert.convert(stateful, dst, config=dict(type="L-DPT-v1m1", **TINY))
    state = weights.read_checkpoint(dst)["state_dict"]
    assert state.keys() == model.state_dict().keys()
    loaded = api.load(tiny_preset, device="cpu", weights=stateful)
    square = torch.randn(1, 3, 48, 48)  # RoPE "min" == "separate" on square inputs
    with torch.no_grad():
        assert torch.equal(loaded.infer(square), model.infer(square))
    assert torch.equal(api.load(dst, device="cpu").infer(square), model.infer(square))


def test_unpublished_weight_slot(monkeypatch, tmp_path):
    """A slot whose repo / revision / sha256 is TBD: no download, no check; its
    variable works with verify=False, and weights= always does."""
    tbd = presets.WeightSource(
        repo=presets.TBD,
        file="l-dpt_large_depth_zeroshot.pth",
        revision=presets.TBD,
        sha256=presets.TBD,
        size=0,
        env="LDPT_CKPT_LARGE_DEPTH_ZEROSHOT",
        format="release",
    )
    assert not tbd.published
    name = "l-dpt_large_depth_zeroshot"
    monkeypatch.setitem(presets.WEIGHTS, name, tbd)
    monkeypatch.delenv(tbd.env, raising=False)
    with pytest.raises(ValueError, match="not published yet"):
        weights.fetch(name)
    with pytest.raises(ValueError, match="not published yet"):
        api.load(name, device="cpu")
    local = tmp_path / f"{name}.pth"
    local.write_bytes(b"x")
    monkeypatch.setenv(tbd.env, str(local))
    with pytest.raises(ValueError, match="verify=False"):
        weights.fetch(name)
    assert weights.fetch(name, verify=False) == local


def test_release_weights_published(monkeypatch, tmp_path):
    """The release slots are pinned: a local file named by the variable must have the
    pinned sha256 (verify=False skips the check)."""
    for name in ("l-dpt_large_depth_zeroshot", "l-dpt_large_depth_mixed"):
        assert WEIGHTS[name].published
        local = tmp_path / f"{name}.pth"
        local.write_bytes(b"x")
        monkeypatch.setenv(WEIGHTS[name].env, str(local))
        with pytest.raises(RuntimeError, match="not the pinned"):
            weights.fetch(name)
        assert weights.fetch(name, verify=False) == local


def test_load_with_encoder_weights(tmp_path, monkeypatch):
    """Encoder from its own file (bare keys, as DINOv3 ships), stored in bf16 on load:
    same output as the eager fp32 model."""
    cfg = dict(TINY, encoder=dict(type="dinov3_tiny_test"))
    model = build_model(dict(type="L-DPT-v1m1", **cfg))
    init_encoder_(model.encoder)
    randomize_decoder_(model)
    model.eval()
    encoder_file = tmp_path / "dinov3_tiny.pth"
    torch.save(model.encoder.state_dict(), encoder_file)
    decoder = {
        k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")
    }
    torch.save(dict(state_dict=decoder), tmp_path / "dev.pth")
    convert.convert(tmp_path / "dev.pth", tmp_path / "rel.pth", config=cfg)

    monkeypatch.delenv(weights.ENCODER_ENV, raising=False)
    with pytest.raises(ValueError, match="not redistributed"):
        api.load(tmp_path / "rel.pth", device="cpu")
    loaded = api.load(
        tmp_path / "rel.pth",
        encoder_file,
        device="cpu",
        dec_pos_embed_rope_normalize_coords="separate",
    )
    assert loaded.encoder.blocks[0].attn.qkv.weight.dtype == torch.bfloat16
    assert loaded.encoder.norm.weight.dtype == torch.float32
    assert not any(p.requires_grad for p in loaded.encoder.parameters())
    image = torch.randn(1, 3, 40, 56)
    with torch.no_grad():
        assert torch.equal(model.infer(image), loaded.infer(image))


def test_load_shares_an_encoder(tmp_path, monkeypatch):
    """share_encoder=: a second model uses the first one's encoder module (no copy, no
    encoder file), with the outputs of its own eager model; a mismatch raises."""
    cfg = dict(TINY, encoder=dict(type="dinov3_tiny_test"))
    torch.manual_seed(0)
    first = build_model(dict(type="L-DPT-v1m1", **cfg))
    init_encoder_(first.encoder)
    second = build_model(dict(type="L-DPT-v1m1", **cfg))
    second.encoder.load_state_dict(first.encoder.state_dict())
    encoder_file = tmp_path / "dinov3_tiny.pth"
    torch.save(first.encoder.state_dict(), encoder_file)
    for i, model in enumerate((first, second)):
        randomize_decoder_(model, seed=i)
        model.eval()
        decoder = {
            k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")
        }
        torch.save(dict(state_dict=decoder), tmp_path / f"dev{i}.pth")
        convert.convert(tmp_path / f"dev{i}.pth", tmp_path / f"rel{i}.pth", config=cfg)

    monkeypatch.delenv(weights.ENCODER_ENV, raising=False)
    same = dict(device="cpu", dec_pos_embed_rope_normalize_coords="separate")
    a = api.load(tmp_path / "rel0.pth", encoder_file, **same)
    b = api.load(tmp_path / "rel1.pth", share_encoder=a.encoder, **same)
    assert b.encoder is a.encoder
    assert b.weights_info["encoder"] == f"{encoder_file} (shared)"
    assert not any(t.is_meta for t in b.state_dict().values())
    image = torch.randn(1, 3, 40, 56)
    with torch.no_grad():
        assert torch.equal(first.infer(image), a.infer(image))
        assert torch.equal(second.infer(image), b.infer(image))
        assert not torch.equal(a.infer(image), b.infer(image))

    with pytest.raises(ValueError, match="share_encoder= or encoder_weights="):
        api.load(
            tmp_path / "rel1.pth", encoder_file, device="cpu", share_encoder=a.encoder
        )
    stub = build_model(dict(type="L-DPT-v1m1", **TINY)).encoder
    with pytest.raises(ValueError, match="this model's encoder is a"):
        api.load(tmp_path / "rel1.pth", device="cpu", share_encoder=stub)
    other = build_model(
        dict(type="L-DPT-v1m1", **dict(cfg, encoder=dict(type="dinov3_tiny_test")))
    ).encoder
    other.norm = torch.nn.LayerNorm(other.norm.normalized_shape[0] + 1)
    with pytest.raises(ValueError, match="names or shapes differ"):
        api.load(tmp_path / "rel1.pth", device="cpu", share_encoder=other)
    with pytest.raises(ValueError, match="meta device"):
        with torch.device("meta"):
            meta = build_model(dict(type="L-DPT-v1m1", **cfg)).encoder
        api.load(tmp_path / "rel1.pth", device="cpu", share_encoder=meta)
    # an encoder on another device than this model's decoder
    built = convert.build_meta(dict(cfg, encoder_weight_dtype="bf16")).encoder
    with pytest.raises(ValueError, match=r"the encoder is on \['cpu'\]"):
        api._check_shared_encoder(built, a.encoder, cfg, torch.device("meta"))
    # another encoder config than the one the encoder was read with
    source = a.encoder._ldpt_source
    a.encoder._ldpt_source = dict(source, config=dict(type="dinov3_vit7b"))
    with pytest.raises(ValueError, match="was built from"):
        api.load(tmp_path / "rel1.pth", share_encoder=a.encoder, **same)
    a.encoder._ldpt_source = source
    # another storage dtype: refused, and the shared module is left as it is (no
    # load hook of the second model runs on it)
    a.encoder.float()
    with pytest.raises(ValueError, match="another dtype"):
        api.load(tmp_path / "rel1.pth", share_encoder=a.encoder, **same)
    assert a.encoder.blocks[0].attn.qkv.weight.dtype == torch.float32
    b32 = api.load(
        tmp_path / "rel1.pth",
        share_encoder=a.encoder,
        encoder_weight_dtype=None,
        **same,
    )
    assert b32.encoder.blocks[0].attn.qkv.weight.dtype == torch.float32


def test_load_errors(dev_ckpt, monkeypatch):
    _, src = dev_ckpt
    with pytest.raises(convert.ConvertError, match="convert the training checkpoint"):
        api.load(src, device="cpu")
    with pytest.raises(ValueError, match="did you mean 'l-dpt_large_depth_mixed'"):
        api.load("l-dpt-large-depth-mixed")
    with pytest.raises(ValueError, match="renamed 'l-dpt_large_depth_mixed'"):
        api.load("l-mix")  # a name of the first release files: not a preset
    monkeypatch.setitem(presets.PRESETS, "tiny-test", dict(TINY))
    with pytest.raises(ValueError, match="no released weights"):
        api.load("tiny-test", device="cpu")


def test_predict(release):
    _, path = release
    model = api.load(path, device="cpu")
    img = np.random.default_rng(0).integers(0, 256, (50, 70, 3), dtype=np.uint8)
    depth = api.predict(model, img)  # resized to short side 768 and back
    assert depth.shape == (50, 70) and depth.dtype == np.float32
    assert 0 < depth.min() and depth.max() < 1
    as_float = api.predict(model, torch.from_numpy(img.astype(np.float32) / 255.0))
    assert torch.equal(as_float, torch.from_numpy(depth))

    # the val normalisation; the resized path runs the model as loaded
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
    x = (x - torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGE_STD).view(
        1, 3, 1, 1
    )
    x768 = api.resize_short_side(
        torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0, 768
    )
    x768 = (x768 - torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1)) / torch.tensor(
        IMAGE_STD
    ).view(1, 3, 1, 1)
    with torch.no_grad():
        ref768 = torch.nn.functional.interpolate(
            model.infer(x768), size=(50, 70), mode="bilinear", align_corners=False
        )
    assert torch.equal(torch.from_numpy(depth), ref768[0, 0])

    # native resolution: padding inside the model, no resize, with the high-res memory
    # options (NATIVE_MEMORY) for the call only
    assert model.native_memory == api.NATIVE_MEMORY
    assert (model.selection_impl, model.chunk_tokens) == ("concat", None)
    model.selection_impl, model.chunk_tokens = "stream", 32768
    with torch.no_grad():
        ref = model.infer(x)
        ref_flip = torch.stack([ref, model.infer(x.flip(-1)).flip(-1)], 1).mean(1)
    model.selection_impl, model.chunk_tokens = "concat", None
    assert torch.equal(torch.from_numpy(api.predict(model, img, size=None)), ref[0, 0])
    flipped = api.predict(model, img, size=None, flip=True)
    assert torch.equal(torch.from_numpy(flipped), ref_flip[0, 0])
    assert (model.selection_impl, model.chunk_tokens) == ("concat", None)  # restored

    # the caller's choice wins: an option set on the model, native_memory = {}, or
    # the options passed to load
    with torch.no_grad():
        plain = model.infer(x)
    model.selection_impl, model.chunk_tokens = "stream", 8
    with torch.no_grad():
        chunked = model.infer(x)  # stream (from native_memory) + the caller's chunk
    model.selection_impl = "concat"  # the default: native_memory's applies
    assert torch.equal(
        torch.from_numpy(api.predict(model, img, size=None)), chunked[0, 0]
    )
    assert model.chunk_tokens == 8
    model.chunk_tokens = None
    model.native_memory = {}
    assert torch.equal(
        torch.from_numpy(api.predict(model, img, size=None)), plain[0, 0]
    )
    chosen = api.load(path, device="cpu", selection_impl="concat", chunk_tokens=None)
    assert chosen.native_memory == {}
    assert torch.equal(
        torch.from_numpy(api.predict(chosen, img, size=None)), plain[0, 0]
    )
    half = api.load(path, device="cpu", selection_impl="concat")
    assert half.native_memory == dict(chunk_tokens=32768)

    with pytest.raises(ValueError, match="divide by 255"):
        api.predict(model, img.astype(np.float32))
    with pytest.raises(ValueError, match="HxWx3"):
        api.predict(model, x[0])


def _sdpa_state():
    """The SDPA backends torch tries, in its priority order (the enabled ones)."""
    from torch.nn.attention import _cur_sdpa_kernel_backends

    return [b.name for b in _cur_sdpa_kernel_backends(with_priority=True)]


def test_predict_attention_order(release, monkeypatch):
    """predict runs the model inside sdpa_context(sdpa_backends, its device): by
    default "auto", the order sdpa.auto_order picks for the model's device (on the CPU
    torch's; cuDNN attention first where it returns SDPA_ORDER), for both size paths
    and the mirrored pass, and restores torch's flags after the call, also when the
    model raises; None runs torch's order, a list or a name replaces it, an unknown
    name raises before the model runs. On the CPU the order changes no bit of the
    depth."""
    from ldpt.utils import sdpa

    _, path = release
    model = api.load(path, device="cpu")
    img = np.random.default_rng(1).integers(0, 256, (40, 56, 3), dtype=np.uint8)
    torch_order = _sdpa_state()
    assert torch_order[0] == "FLASH_ATTENTION"  # torch 2.8's choice
    assert api.SDPA_ORDER == ("cudnn", "flash", "efficient", "math")
    cudnn_first = ["CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"]
    infer, seen = model.infer, []

    def recording(x, **kwargs):
        seen.append(_sdpa_state())
        return infer(x, **kwargs)

    auto_order, asked = sdpa.auto_order, []

    def compute_10(device=None):  # auto_order on a GPU of compute capability 10.x
        asked.append(device)
        return api.SDPA_ORDER

    monkeypatch.setattr(model, "infer", recording)
    for size in ("auto", None):
        ref = api.predict(model, img, size=size, flip=True, sdpa_backends=None)
        assert auto_order(torch.device("cpu")) is None
        monkeypatch.setattr(sdpa, "auto_order", compute_10)
        for kwargs in ({}, dict(sdpa_backends="auto")):
            seen.clear()
            got = api.predict(model, img, size=size, flip=True, **kwargs)
            assert seen == [cudnn_first, cudnn_first], (size, kwargs, seen)
            assert asked[-1] == torch.device("cpu")  # the model's device
            assert np.array_equal(got, ref), (size, kwargs)
        monkeypatch.setattr(sdpa, "auto_order", auto_order)
        for backends, want in (
            ("default", torch_order),  # "auto" on the CPU
            ("auto", torch_order),
            (api.SDPA_ORDER, cudnn_first),
            (None, torch_order),
            ([], torch_order),
            (["math"], ["MATH"]),
            ("flash", ["FLASH_ATTENTION"]),  # one name
        ):
            seen.clear()
            kwargs = {} if backends == "default" else dict(sdpa_backends=backends)
            got = api.predict(model, img, size=size, flip=True, **kwargs)
            assert seen == [want, want], (size, backends, seen)  # image, mirrored
            assert _sdpa_state() == torch_order  # restored
            if want != ["MATH"]:  # the CPU's flash kernel in each of these orders
                assert np.array_equal(got, ref), (size, backends)

    seen.clear()
    with pytest.raises(ValueError, match="unknown"):
        api.predict(model, img, sdpa_backends=["cudnn", "fast"])
    assert not seen

    def failing(x, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(model, "infer", failing)
    with pytest.raises(RuntimeError, match="out of memory"):
        api.predict(model, img)
    assert _sdpa_state() == torch_order


def test_predict_encoder_tiles(release, monkeypatch):
    """encoder_tiles: "auto" (the default) runs model.infer with ENCODER_TILES, None and
    False with none, an EncoderTiles with itself, on both size paths and both passes
    of flip; another value raises before the model runs. A small tiling tiles this
    small image (the stub encoder has no RoPE: rope_scale 1.0); an image of at most
    3072 px gives the plain forward bit for bit."""
    _, path = release
    model = api.load(path, device="cpu")
    img = np.random.default_rng(2).integers(0, 256, (40, 56, 3), dtype=np.uint8)
    small = api.EncoderTiles(tile=2, overlap=1, rope_scale=1.0)  # 3 x 4 tokens
    infer, seen = model.infer, []

    def recording(x, encoder_tiles=None):
        seen.append(encoder_tiles)
        return infer(x, encoder_tiles=encoder_tiles)

    monkeypatch.setattr(model, "infer", recording)
    for size in ("auto", None):
        plain = api.predict(model, img, size=size, flip=True, encoder_tiles=None)
        for value, want in (
            ("default", api.ENCODER_TILES),
            ("auto", api.ENCODER_TILES),
            (None, None),
            (False, None),
            (small, small),
        ):
            seen.clear()
            kwargs = {} if value == "default" else dict(encoder_tiles=value)
            got = api.predict(model, img, size=size, flip=True, **kwargs)
            assert seen == [want, want], (size, value)
            if want is not small:  # the plain forward, bit for bit
                assert np.array_equal(got, plain), (size, value)
    seen.clear()
    for bad in ("on", True, 192, dict(tile=192)):
        with pytest.raises(ValueError, match="encoder_tiles"):
            api.predict(model, img, encoder_tiles=bad)
    assert not seen
    monkeypatch.setattr(model, "infer", infer)

    # the tiled path through predict: model.infer with the tiling under NATIVE_MEMORY
    tiled = api.predict(model, img, size=None, encoder_tiles=small)
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
    x = (x - torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGE_STD).view(
        1, 3, 1, 1
    )
    model.selection_impl, model.chunk_tokens = "stream", 32768
    with torch.no_grad():
        ref = model.infer(x, encoder_tiles=small)[0, 0]
    model.selection_impl, model.chunk_tokens = "concat", None
    assert torch.equal(torch.from_numpy(tiled), ref)
    assert not np.array_equal(tiled, api.predict(model, img, size=None))


@pytest.mark.parametrize("hw", [(50, 70), (90, 40)])
def test_resize_matches_torchvision(hw):
    T = pytest.importorskip("torchvision.transforms.v2")
    x = torch.rand(3, *hw, generator=torch.Generator().manual_seed(0))
    ref = T.Resize(768, interpolation=T.InterpolationMode.BICUBIC)(x)
    assert torch.equal(api.resize_short_side(x[None], 768)[0], ref)


def test_weights_checks(tmp_path, monkeypatch):
    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"not a checkpoint")
    pinned = presets.WeightSource(
        repo="org/l-dpt",
        file="l-dpt_large_depth_zeroshot.pth",
        revision="0" * 40,
        sha256="0" * 64,
        size=16,
        env="LDPT_CKPT_LARGE_DEPTH_ZEROSHOT",
        format="release",
    )
    assert pinned.published
    monkeypatch.setitem(presets.WEIGHTS, "l-dpt_large_depth_zeroshot", pinned)
    monkeypatch.setenv(pinned.env, str(fake))
    with pytest.raises(RuntimeError, match="not the pinned"):
        weights.fetch("l-dpt_large_depth_zeroshot")
    assert weights.fetch("l-dpt_large_depth_zeroshot", verify=False) == fake

    monkeypatch.delenv(weights.ENCODER_ENV, raising=False)
    with pytest.raises(ValueError, match="not redistributed"):
        weights.encoder_weights("dinov3_vit7b")
    with pytest.raises(RuntimeError, match="expected a955f4ea"):
        weights.encoder_weights("dinov3_vit7b", fake)
    assert weights.encoder_weights("dinov3_vit7b", fake, verify=False) == fake

    # hashed once per (path, size, mtime)
    digest = weights.verified_sha256(fake)
    monkeypatch.setattr(weights, "sha256_of", lambda path: pytest.fail("re-hashed"))
    assert weights.verified_sha256(fake) == digest
