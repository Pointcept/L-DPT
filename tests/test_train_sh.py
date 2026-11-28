"""train.sh with a stand-in python that records its argv (config folder: the default
of -d, hybrid): a fresh run passes the -o options as given; a resume passes
checkpoint=<run>/model/model_last.pth and drops a checkpoint= option (the start
weights of a fine-tune), which would otherwise replace model_last.pth on a requeue;
every other option is kept in order. -d names the config folder as in Pointcept
(configs/<folder>/, exp/<folder>/), -e the experiment root."""

import os
import shutil
import stat
import subprocess

from helpers import REPO

OPTS = (
    "num_workers=8 checkpoint=[ckpt/dinov3.pth,exp/hybrid/src/model/epoch_25.pth] "
    "sdpa_backends=[cudnn,flash,efficient,math] hooks.6.save_freq_iter=1000"
)


def launch(root, opts, *flags, check=True):
    """Run root/train.sh -r auto with the stand-in python (+ ``flags``); returns its
    argv and output."""
    record = root / "argv.txt"
    if record.exists():
        record.unlink()
    cmd = [
        "bash",
        str(root / "train.sh"),
        "-p",
        str(root / "fakepy"),
        "-g",
        "1",
        "-e",
        "exp",
        *flags,
        "-c",
        "some-config",
        "-n",
        "ft-run",
        "-r",
        "auto",
        "-o",
        opts,
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    p = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    if not check:
        return p
    assert p.returncode == 0, p.stdout + p.stderr
    return record.read_text().split("\n")[:-1], p.stdout


def setup(tmp_path):
    root = tmp_path / "code"
    (root / "ldpt").mkdir(parents=True)
    shutil.copy(os.path.join(REPO, "train.sh"), root / "train.sh")
    fake = root / "fakepy"
    fake.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {root / "argv.txt"}\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return root


def test_fresh_run_passes_the_options(tmp_path):
    root = setup(tmp_path)
    argv, _ = launch(root, OPTS)
    i = argv.index("--options")
    assert argv[i + 1 :] == ["save_path=exp/hybrid/ft-run"] + OPTS.split()
    assert argv[argv.index("--config-file") + 1] == "configs/hybrid/some-config.py"


def test_resume_drops_a_checkpoint_option(tmp_path):
    root = setup(tmp_path)
    launch(root, OPTS)  # creates the run dir
    run = root / "exp" / "hybrid" / "ft-run"
    (run / "model" / "model_last.pth").write_bytes(b"")
    (run / "config.py").write_text("")
    argv, out = launch(root, OPTS)
    i = argv.index("--options")
    kept = [o for o in OPTS.split() if not o.startswith("checkpoint=")]
    assert (
        argv[i + 1 :]
        == [
            "save_path=exp/hybrid/ft-run",
            "resume=true",
            "checkpoint=exp/hybrid/ft-run/model/model_last.pth",
        ]
        + kept
    )
    assert argv[argv.index("--config-file") + 1] == "exp/hybrid/ft-run/config.py"
    assert "option checkpoint=[ckpt/dinov3.pth" in out
    # without a checkpoint= option the options pass unchanged
    argv, _ = launch(root, "num_workers=8 hooks.6.save_freq_iter=1000")
    assert argv[argv.index("--options") + 1 :][3:] == [
        "num_workers=8",
        "hooks.6.save_freq_iter=1000",
    ]


def test_config_folder_and_experiment_root(tmp_path):
    root = setup(tmp_path)
    argv, _ = launch(root, "num_workers=8", "-d", "cocostuff", "-e", "outputs")
    assert argv[argv.index("--config-file") + 1] == "configs/cocostuff/some-config.py"
    assert argv[argv.index("--options") + 1] == "save_path=outputs/cocostuff/ft-run"
    assert (root / "outputs" / "cocostuff" / "ft-run" / "code").is_dir()
    p = launch(root, "num_workers=8", "-s", "hybrid", check=False)  # not a flag
    assert p.returncode != 0 and "Invalid option: -s" in p.stderr
