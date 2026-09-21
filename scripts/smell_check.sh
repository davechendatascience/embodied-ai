#!/usr/bin/env bash
# The code-smell gate: docs/code_smell_guide.md, as tools.
#
#   scripts/smell_check.sh            # all of our code; exit 1 on any finding
#   scripts/smell_check.sh FILE...    # just these files
#
# Four checks, each pinned so a clean run means the same thing next month:
#   ruff       unused code, bug patterns, complexity, argument counts, unnamed thresholds,
#              blind excepts (ruff.toml)
#   vulture    functions, methods, attributes nothing uses
#   pylint     blocks of 6+ lines duplicated across files
#   radon      the five most complex functions left, for information
#
# Runs through uvx, so neither .venv nor .venv-libero is touched (the two must not
# reference each other). third_party/ is not ours and is not checked.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ $# -gt 0 ]; then FILES=("$@"); else mapfile -t FILES < <(git ls-files '*.py' | grep -v '^third_party/'); fi

# Names vulture cannot see being used, each with the reason it is not dead:
IGNORE=(
  # robosuite builds robots and grippers from these class attributes by name
  default_mount default_gripper default_controller_config init_qpos top_offset arm_type
  # read by torch.nn / dataclass machinery or by callers outside this repo's call graph
  forward
)
IGNORE_CSV=$(IFS=,; echo "${IGNORE[*]}")

fail=0
echo "== ruff"
uvx ruff@0.16.8 check "${FILES[@]}" || fail=1

echo "== vulture (unused code)"
out=$(uvx vulture@2.16 --min-confidence 60 --ignore-names "$IGNORE_CSV" "${FILES[@]}" 2>&1)
if [ -n "$out" ]; then echo "$out"; fail=1; else echo "clean"; fi

echo "== pylint (duplicated blocks across files)"
out=$(uvx pylint@4.0.8 --disable=all --enable=duplicate-code --min-similarity-lines=6 \
      --ignore-imports=yes --ignore-signatures=yes --score=n "${FILES[@]}" 2>&1 | grep -v '^\*\*\*' || true)
if [ -n "$out" ]; then echo "$out"; fail=1; else echo "clean"; fi

echo "== radon (most complex functions, for information)"
uvx radon@6.0.1 cc -s -n C -o SCORE "${FILES[@]}" 2>/dev/null | head -12 || true

if [ $fail -eq 0 ]; then echo; echo "smell check: clean"; else echo; echo "smell check: FINDINGS"; fi
exit $fail
