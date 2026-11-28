"""
Release hygiene (CPU, no data): no text file of the repository holds a private path,
a private e-mail address, a link to a training run or the commit id of an unpublished
history. The patterns of this file are generic; a hit names the file, the line and
the pattern.

Project-specific names (internal tools, hosts, runs, ...) are not listed here. They
come from a pattern file outside the repository named by $LDPT_HYGIENE_PATTERNS: one
``name<TAB>regex<TAB>example`` per line (``#`` starts a comment line), each regex must
match its example and no text of PUBLIC_TEXT. test_private_patterns checks every file
against them, this one included, and is skipped when the variable is unset.
$LDPT_HYGIENE_EXEMPT (comma-separated repository paths) leaves files out of that check.

The files are the git-tracked ones (``git ls-files``), or every file under the repo
root without a git checkout; binary files are skipped.
"""

import os
import re
import subprocess

import pytest

from helpers import REPO

SELF = "tests/test_release_hygiene.py"  # holds the examples of the generic patterns
SKIP_DIRS = {".git", "__pycache__", "data", "exp", "checkpoints", "weights", "outputs"}

# short hex words that are not commits of this repository: MoGe's upstream commit
# (THIRD_PARTY_LICENSES.md) and the sha256 prefixes in Meta's DINOv3 file names
HEX_ALLOWED = ("74fbce0", "a955f4ea", "7c1da9a5", "8aa4cbdd")

# (name, regex): generic shapes of what must not appear anywhere
PATTERNS = [
    (
        "home directories",
        r"/home/[A-Za-z_][\w.-]*/|/Users/[^/\s]+/|\b[A-Za-z]:\\+Users\\",
    ),
    ("mounted drives", r"/mnt/[a-z]/"),
    ("shared file systems", r"(?<![\w.])/(?:projects|scratch|lustre|gpfs|fsx)/"),
    ("run links", r"wandb\.ai/"),
    ("job ids", r"(?i)\bjobs?(?: id)?[ #:=]+\d{5,}\b"),
    (
        "commit ids",
        r"\b(?!(?:" + "|".join(HEX_ALLOWED) + r")\b)"
        r"(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,12}\b",
    ),
]
EXAMPLES = {
    "home directories": "/home/someone/projects/x.py",
    "mounted drives": "/mnt/d/datasets",
    "shared file systems": "/scratch/someone/runs",
    "run links": "https://wandb.ai/team/project/runs/abcd1234",
    "job ids": "job 1234567",
    "commit ids": "fixed in 3f2c1ab",
}
# text the patterns (generic and private) must leave alone
PUBLIC_TEXT = [
    "configs/hybrid/depth-l-dpt-v1m1-0-large-zeroshot.py",
    "configs/hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal.py",
    "exp/hybrid/depth-l-dpt-v1m1-1-large-mixed/model/epoch_4.pth",
    "exp/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal/model/average.pth",
    "configs/voc2012/semseg-l-dpt-v1m1-1-large.py -n semseg-l-dpt-v1m1-1-large-seed1",
    "l-dpt_large_depth_zeroshot.pth, l-dpt_large_depth_mixed.pth",
    "d1 = torch.sum(thresh < 1.25)",
    "delta1 98.30, 270,995,712 parameters, 579,637 samples",
    "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "moge/utils/alignment.py at commit 74fbce0",
    "sha256 b4e7e95301fda7557760f787e7992f4ccca3c03cb70e46eadeb48e227b1b31ca",
    "relative_depth_loss align='roe' (the L-DPT depth loss), dec_qk_norm=True, QK-norm",
    "Pointcept",
    "NVIDIA GB200 (aarch64)",
    "tests/data/l_dpt_golden.pth",
]
# e-mail addresses allowed in the tree (the authors' public ones)
EMAILS = {"xiaoyang.wu.cs@gmail.com", "xiaoyang.wu@connect.hku.hk"}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


def repo_files():
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout
        files = [f for f in out.decode().split("\0") if f]
        if files:
            return files
    except (OSError, subprocess.CalledProcessError):
        pass
    files = []
    for root, dirs, names in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            files.append(os.path.relpath(os.path.join(root, name), REPO))
    return files


def text_lines(rel):
    """(line number, line) of a text file; nothing for a binary or missing file."""
    path = os.path.join(REPO, rel)
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:  # e.g. deleted in the working tree
        return []
    if b"\0" in data:
        return []
    return list(enumerate(data.decode("utf-8", errors="replace").splitlines(), 1))


@pytest.fixture(scope="module")
def corpus():
    files = [f.replace(os.sep, "/") for f in repo_files()]
    assert len(files) > 50, "no repository files found"
    return {f: text_lines(f) for f in files}


def find(regex, corpus, skip=()):
    return [
        f"{rel}:{n}: {line.strip()[:120]}"
        for rel, lines in corpus.items()
        if rel not in skip
        for n, line in lines
        if regex.search(line) or regex.search(rel)
    ]


@pytest.mark.parametrize("name, pattern", PATTERNS, ids=[n for n, _ in PATTERNS])
def test_no_private_paths_or_ids(corpus, name, pattern):
    hits = find(re.compile(pattern), corpus, skip={SELF})
    assert not hits, f"{name} ({pattern}):\n" + "\n".join(hits[:40])


def test_only_public_emails(corpus):
    hits = [
        f"{rel}:{n}: {email}"
        for rel, lines in corpus.items()
        for n, line in lines
        for email in EMAIL.findall(line)
        if email not in EMAILS and not email.endswith("@users.noreply.github.com")
    ]
    assert not hits, "\n".join(hits[:40])


def test_patterns_catch_examples():
    """Each generic pattern matches an example of what it guards against, and none
    matches the public text."""
    assert set(EXAMPLES) == {n for n, _ in PATTERNS}
    for name, pattern in PATTERNS:
        assert re.search(pattern, EXAMPLES[name]), name
        for text in PUBLIC_TEXT:
            assert not re.search(pattern, text), (name, text)


def private_patterns():
    path = os.environ.get("LDPT_HYGIENE_PATTERNS")
    if not path:
        pytest.skip("LDPT_HYGIENE_PATTERNS not set (project-specific patterns)")
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            assert len(parts) == 3, f"{path}:{n}: expected name<TAB>regex<TAB>example"
            rows.append(parts)
    assert rows, f"{path}: no patterns"
    return rows


def test_private_patterns(corpus):
    rows = private_patterns()
    exempt = os.environ.get("LDPT_HYGIENE_EXEMPT", "")
    exempt = {p.strip() for p in exempt.split(",") if p.strip()}
    failed = []
    for name, pattern, example in rows:
        regex = re.compile(pattern)
        assert regex.search(example), f"private pattern {name!r} misses its example"
        for text in PUBLIC_TEXT:
            assert not regex.search(text), (name, text)
        hits = find(regex, corpus, skip=exempt)
        if hits:
            failed.append(f"{name}:\n" + "\n".join(hits[:40]))
    assert not failed, "\n".join(failed)
