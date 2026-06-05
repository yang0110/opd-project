"""
Collect results from all experiments into a unified comparison table.

Reads eval.json and training logs from experiments/results/<method>/
and produces:
- experiments/results/summary/comparison_table.json
- experiments/results/summary/comparison_table.md (markdown)
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")

sys.path.insert(0, PROJECT_ROOT)
from experiments.src.method_registry import METHOD_ORDER


def collect_eval_results() -> list[dict]:
    """Collect final eval results for all methods."""
    rows = []

    for method in METHOD_ORDER:
        method_dir = os.path.join(RESULTS_DIR, method)
        eval_summary = os.path.join(method_dir, "eval_summary.json")

        if not os.path.exists(eval_summary):
            rows.append({"method": method, "status": "not_run"})
            continue

        with open(eval_summary) as f:
            results = json.load(f)

        if not results:
            rows.append({"method": method, "status": "no_results"})
            continue

        final = results[-1]
        row = {
            "method": method,
            "step": final["step"],
            "status": "complete",
        }

        for bench_name, bench_data in final.get("benchmarks", {}).items():
            for metric_name, metric_data in bench_data.items():
                key = f"{bench_name}_{metric_name}"
                if isinstance(metric_data, dict):
                    row[key] = metric_data.get("pass_at_1") or metric_data.get("pass_at_8")
                else:
                    row[key] = metric_data

        rows.append(row)

    return rows


def collect_training_metrics() -> dict:
    """Collect training loss trajectories for all methods."""
    metrics = {}

    for method in METHOD_ORDER:
        log_path = os.path.join(RESULTS_DIR, method, "training_log.jsonl")
        if not os.path.exists(log_path):
            continue

        steps, losses, kls, entropies = [], [], [], []
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                steps.append(entry.get("step", 0))
                losses.append(entry.get("loss", 0))
                kls.append(entry.get("kl_to_teacher", 0))
                entropies.append(entry.get("entropy", 0))

        metrics[method] = {
            "steps": steps,
            "losses": losses,
            "kl_to_teacher": kls,
            "entropy": entropies,
        }

    return metrics


def generate_markdown_table(rows: list[dict]) -> str:
    """Generate a markdown comparison table."""
    columns = [
        ("method", "Method", 25),
        ("humaneval_plus_pass_at_1", "HE+ P@1", 8),
        ("humaneval_plus_pass_at_8", "HE+ P@8", 8),
        ("mbpp_plus_pass_at_1", "MBPP+ P@1", 9),
        ("mbpp_plus_pass_at_8", "MBPP+ P@8", 9),
    ]

    header = "| " + " | ".join(f"{name:<{w}}" for _, name, w in columns) + " | Delta vs Base |"
    sep = "|" + "|".join("-" * (w + 2) for _, _, w in columns) + "|---------------|"

    lines = [header, sep]

    baseline_score = None
    for row in rows:
        if row["method"] == "standard_opd" and row.get("humaneval_plus_pass_at_1"):
            baseline_score = row["humaneval_plus_pass_at_1"]

    for row in rows:
        values = []
        for key, _, w in columns:
            val = row.get(key, "—")
            if isinstance(val, float):
                val = f"{val:.1f}%"
            values.append(f"{str(val):<{w}}")

        delta = "—"
        he_score = row.get("humaneval_plus_pass_at_1")
        if baseline_score and he_score and isinstance(he_score, (int, float)):
            diff = he_score - baseline_score
            delta = f"{diff:+.1f}%"

        line = "| " + " | ".join(values) + f" | {delta:<13} |"
        lines.append(line)

    return "\n".join(lines)


def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    rows = collect_eval_results()
    metrics = collect_training_metrics()

    with open(os.path.join(SUMMARY_DIR, "comparison_table.json"), "w") as f:
        json.dump({"eval_results": rows, "training_metrics": metrics}, f, indent=2)

    md_table = generate_markdown_table(rows)
    with open(os.path.join(SUMMARY_DIR, "comparison_table.md"), "w") as f:
        f.write("# OPD Method Comparison: Code Generation\n\n")
        f.write("**Setup**: Teacher=Qwen3-14B, Student=Qwen3-4B, Dataset=Eurus-RL-Code, 200 steps\n\n")
        f.write(md_table)
        f.write("\n")

    print(md_table)
    print(f"\nResults saved to: {SUMMARY_DIR}/")


if __name__ == "__main__":
    main()
