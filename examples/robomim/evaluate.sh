#!/bin/bash
#
# evaluate.sh — Run a robomimic Lift CAPOF evaluation repeatedly, each trial
# with a fresh random seed (drawn inside the entry script), then report the
# mean success rate with its std.
#
# Usage:
#   ./evaluate.sh [task] [N]
#
#   task: force_conditioned | object_centric   (default: force_conditioned)
#   N:    number of trials                      (default: 20)
#
# Run with the project environment active (or set PY to the python binary):
#   source /path/to/env/bin/activate
#   ./evaluate.sh force_conditioned 20
#
# Each trial runs one full evaluation (GP fit + control loop) and prints a
# "Score: <final reward>" line. A trial counts as a success when the final
# reward is 1.0 (the Lift reward is 1.0 on success, 0.0 otherwise).
#
set -uo pipefail

TASK=${1:-force_conditioned}
N=${2:-20}
PY=${PY:-python3}

case "$TASK" in
    force_conditioned|object_centric) ;;
    *) echo "Unknown task: $TASK (expected force_conditioned or object_centric)"; exit 1 ;;
esac

# Resolve the repository root from this script's location (examples/robomim/evaluate.sh)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

scores=()
for i in $(seq 1 "$N"); do
    echo "=== [$TASK] Trial $i / $N ==="
    # Run one evaluation; capture the full output to extract the
    # "Score: <reward>" line and the seed from the "Video saved to .../<seed>.mp4" line.
    output=$(cd "$REPO" && "$PY" "examples/robomim/${TASK}.py" 2>&1)
    score=$(printf '%s\n' "$output" | grep 'Score:' | grep -oE '[-+0-9.eE]+' | tail -1)
    seed=$(printf '%s\n' "$output" | grep 'Video saved to' | grep -oE '[0-9]+\.mp4' | head -1)
    seed=${seed%.mp4}
    if [[ -z "$score" ]]; then
        echo "  WARNING: no Score found (run may have failed)"
        score=0
    fi
    echo "  Seed:  ${seed:-?}"
    echo "  Score: $score"
    scores+=("$score")
done

echo ""
echo "================ Results [$TASK] ================"
printf '%s\n' "${scores[@]}" | "$PY" -c "
import sys
import numpy as np
s = np.array([float(x) for x in sys.stdin.read().split()])
success = (s == 1.0).astype(float)
print(f'Trials:        {len(s)}')
print(f'Mean reward:   {s.mean():.3f} +/- {s.std(ddof=1):.3f}')
print(f'Success rate:  {success.mean():.3f} +/- {success.std(ddof=1):.3f}   ({int(success.sum())}/{len(s)} successes)')
"
