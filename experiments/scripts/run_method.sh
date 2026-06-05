#!/bin/bash
# Run a single OPD method experiment.
#
# Usage:
#   bash experiments/scripts/run_method.sh standard_opd
#   bash experiments/scripts/run_method.sh gopd_extrap --steps 100

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

METHOD="${1:?Usage: $0 <method_name> [--steps N]}"
shift

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps) EXTRA_ARGS="$EXTRA_ARGS --steps $2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

RESULTS_DIR="$PROJECT_ROOT/experiments/results/$METHOD"
mkdir -p "$RESULTS_DIR"

echo "================================================"
echo "Running OPD Experiment: $METHOD"
echo "  Results: $RESULTS_DIR"
echo "  Time: $(date)"
echo "================================================"

cd "$PROJECT_ROOT"

python -m experiments.src.opd_trainer \
    --method "$METHOD" \
    $EXTRA_ARGS \
    2>&1 | tee "$RESULTS_DIR/training_stdout.log"

echo ""
echo "Training complete: $METHOD"
echo "  Logs: $RESULTS_DIR/training_log.jsonl"
echo "  Stdout: $RESULTS_DIR/training_stdout.log"

# Auto-evaluate at each checkpoint
echo ""
echo "Running evaluation..."
python -m experiments.eval.evaluate \
    --method "$METHOD" \
    --all-steps \
    2>&1 | tee "$RESULTS_DIR/eval_stdout.log"

echo ""
echo "Experiment complete: $METHOD at $(date)"
