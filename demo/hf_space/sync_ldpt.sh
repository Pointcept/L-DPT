#!/usr/bin/env bash
# Copy the inference subset of the ldpt package into the Space (demo/hf_space/ldpt/):
# every ldpt module that `from ldpt.models.l_dpt import api` and
# `import ldpt.utils.depth_image` (the colour and 16-bit PNG of space_utils.py) import,
# traced by importing them (not listed by hand), so api.load / api.predict run without
# the training code and its dependencies (tests/test_hf_space.py checks that).
# SUBSET.txt lists the copied files and the commit they come from.
#
#   demo/hf_space/sync_ldpt.sh [DEST]     DEST default demo/hf_space/ldpt; PY=python
#
# Then upload the folder as the Space, without its .gitignore (it ignores ldpt/ and
# examples/ for this repo only) and without compiled files ("*.pyc" also matches the
# top-level __pycache__/, which "**/__pycache__/**" does not), e.g.
#   hf upload <user>/<space> demo/hf_space . --repo-type space \
#     --exclude ".gitignore" --exclude "*.pyc"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DEST="${1:-$HERE/ldpt}"
PY="${PY:-python}"

if [[ "$(basename "$DEST")" != "ldpt" ]]; then
  echo "sync_ldpt.sh: DEST must be a folder named ldpt, got $DEST" >&2
  exit 2
fi
mkdir -p "$(dirname "$DEST")"
DEST="$(cd "$(dirname "$DEST")" && pwd)/ldpt"
if [[ "$DEST" == "$REPO/ldpt" ]]; then
  echo "sync_ldpt.sh: DEST is the repo's own ldpt package" >&2
  exit 2
fi

# the ldpt files `import ldpt.models.l_dpt.api, ldpt.utils.depth_image` load from the
# package under ROOT
# (-I: no user site, no PYTHONPATH, no cwd on sys.path, only ROOT added; -B: no .pyc)
trace() {
  "$PY" -I -B - "$1" <<'EOF'
import os
import sys

root = os.path.abspath(sys.argv[1])
sys.path.insert(0, root)
import ldpt.models.l_dpt.api  # noqa: F401,E402  (load / predict and all they import)
import ldpt.utils.depth_image  # noqa: F401,E402  (colour, 16-bit PNG: space_utils.py)

files = set()
for name, module in list(sys.modules.items()):
    if name != "ldpt" and not name.startswith("ldpt."):
        continue
    path = getattr(module, "__file__", None)  # None: the namespace package ldpt
    if path is None:
        continue
    rel = os.path.relpath(os.path.abspath(path), root)
    if not rel.startswith("ldpt" + os.sep) or not rel.endswith(".py"):
        sys.exit(f"{name} was imported from {path}, outside {root}/ldpt")
    files.add(rel)
print("\n".join(sorted(files)))
EOF
}

FILES="$(trace "$REPO")"
rm -rf "$DEST"
mkdir -p "$DEST"
count=0
while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue
  mkdir -p "$DEST/$(dirname "${rel#ldpt/}")"
  cp "$REPO/$rel" "$DEST/${rel#ldpt/}"
  count=$((count + 1))
done <<< "$FILES"
# the copy must import from itself alone, file for file
if [[ "$(trace "$(dirname "$DEST")")" != "$FILES" ]]; then
  echo "sync_ldpt.sh: the copy in $DEST imports other files than the repo" >&2
  exit 1
fi

# the commit of the repo the files come from (a folder that is not its own git
# checkout, e.g. an exported tree inside another repository, has none)
commit="not a git checkout"
top="$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -n "$top" && "$top" == "$(cd "$REPO" && pwd -P)" ]]; then
  commit="$(git -C "$REPO" rev-parse --short HEAD)"
  if [[ -n "$(git -C "$REPO" status --porcelain -- ldpt)" ]]; then
    commit="$commit (ldpt/ has local changes)"
  fi
fi
{
  echo "# ldpt inference subset for the Space: written by demo/hf_space/sync_ldpt.sh"
  echo "# source: $commit"
  echo "$FILES"
} > "$DEST/SUBSET.txt"
echo "sync_ldpt.sh: $count files from $commit -> $DEST"
