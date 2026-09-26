#!/usr/bin/env bash
# The code-smell gate: docs/code_smell_guide.md, as tools.
#
#   scripts/smell_check.sh            # all of our code; exit 1 on any finding
#   scripts/smell_check.sh FILE...    # just these files
#
# Five checks, each pinned so a clean run means the same thing next month:
#   ruff       unused code, bug patterns, complexity, argument counts, unnamed thresholds,
#              blind excepts (ruff.toml)
#   vulture    functions, methods, attributes nothing uses
#   pylint     blocks of 6+ lines duplicated across files
#   size       (whole-repo runs) no file git tracks or would add over MAX_FILE_MB, and the tracked
#              total within TRACKED_BUDGET_MB: data, media and run traces belong in ignored folders
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
  # called on sim/joint_ramp.JointRamp by robosuite's JointPositionController (its interpolator API)
  set_goal get_interpolated_goal
  # SimArm's lean step refreshes robosuite's controller cache and env clock, which robosuite reads
  joint_pos joint_vel mass_matrix cur_time
  # task_loss's stub domain: LIBERO's predicate code reads these attributes of the env it is given
  objects_dict fixtures_dict
  # SimArm._set_render_samples: MuJoCo reads offsamples when a render context is made; robosuite reads the context
  offsamples _render_context_offscreen
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

if [ $# -eq 0 ]; then
  echo "== repository size (tracked files, and untracked files not ignored)"
  MAX_FILE_MB=5          # the largest tracked file on 2026-09-25 was a 2.4 MB run record
  TRACKED_BUDGET_MB=100  # 26 MB tracked on 2026-09-25, after a run's raw read traces were ignored
  big=$( { git ls-files -z; git ls-files -z --others --exclude-standard; } | xargs -0 -r stat -c '%s %n' 2>/dev/null \
        | awk -v m=$((MAX_FILE_MB * 1000000)) '$1 > m {printf "  %.1f MB  %s\n", $1 / 1e6, $2}')
  total=$(git ls-files -z | xargs -0 -r stat -c '%s' 2>/dev/null | awk '{s += $1} END {printf "%.0f", s / 1e6}')
  if [ -n "$big" ]; then echo "files over ${MAX_FILE_MB} MB (keep them out of git):"; echo "$big"; fail=1; fi
  if [ "$total" -gt $TRACKED_BUDGET_MB ]; then echo "tracked total ${total} MB, over the ${TRACKED_BUDGET_MB} MB budget"; fail=1; fi
  if [ -z "$big" ] && [ "$total" -le $TRACKED_BUDGET_MB ]; then echo "clean (${total} MB tracked)"; fi
fi

echo "== radon (most complex functions, for information)"
uvx radon@6.0.1 cc -s -n C -o SCORE "${FILES[@]}" 2>/dev/null | head -12 || true

if [ $fail -eq 0 ]; then echo; echo "smell check: clean"; else echo; echo "smell check: FINDINGS"; fi
exit $fail
