#!/bin/bash
# Run all 14 OPD method experiments sequentially.
# Each method trains for 200 steps, evaluates every 50 steps,
# then auto-generates comparison figures.
#
# Usage:
#   bash experiments/scripts/run_all.sh
#   bash experiments/scripts/run_all.sh --steps 50   # quick test

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

EXTRA_ARGS="$@"

METHODS=(
    "standard_opd"
    "gopd_interp"
    "gopd_extrap"
    "gopd_correction"
    "veto_logit"
    "veto_token"
    "entropy_aware"
    "aopd"
    "aopd_clipped"
    "extrapolation_cliff"
    "caopd"
    "uni_opd"
    "aligndistil"
    "coverage_opd"
)

TOTAL=${#METHODS[@]}
START_TIME=$(date +%s)

echo "========================================================"
echo "OPD Method Comparison: Running all $TOTAL experiments"
echo "  Start: $(date)"
echo "  Args: $EXTRA_ARGS"
echo "========================================================"

COMPLETED=0
FAILED=0

for i in "${!METHODS[@]}"; do
    METHOD="${METHODS[$i]}"
    NUM=$((i + 1))

    echo ""
    echo "--------------------------------------------------------"
    echo "[$NUM/$TOTAL] Starting: $METHOD"
    echo "--------------------------------------------------------"

    if bash "$SCRIPT_DIR/run_method.sh" "$METHOD" $EXTRA_ARGS; then
        COMPLETED=$((COMPLETED + 1))
        echo "[OK] $METHOD completed successfully"
    else
        FAILED=$((FAILED + 1))
        echo "[FAIL] $METHOD failed (continuing with next method)"
    fi

    # Generate intermediate plots after each method
    echo "  Updating comparison figures..."
    cd "$PROJECT_ROOT"
    python -m experiments.analysis.plot_curves 2>/dev/null || true
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "========================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED / $TOTAL"
echo "  Total time: ${ELAPSED} minutes"
echo "========================================================"

# Final comparison report
echo ""
echo "Generating final comparison report..."
cd "$PROJECT_ROOT"
python -m experiments.analysis.generate_comparison

echo ""
echo "Done! Results at: experiments/results/summary/"
