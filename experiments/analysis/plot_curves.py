"""
Plot training curves for all methods.

Reads training_log.jsonl from each method's results directory
and generates comparison figures automatically.

Outputs:
- experiments/results/summary/figures/loss_curves.png
- experiments/results/summary/figures/kl_curves.png
- experiments/results/summary/figures/entropy_curves.png
- experiments/results/summary/figures/eval_progress.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "summary", "figures")

sys.path.insert(0, PROJECT_ROOT)
from experiments.src.method_registry import METHOD_ORDER

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
]

METHOD_LABELS = {
    "standard_opd": "Standard OPD (baseline)",
    "gopd_interp": "G-OPD (λ=0.75)",
    "gopd_extrap": "G-OPD/ExOPD (λ=1.25)",
    "gopd_correction": "G-OPD + Correction",
    "veto_logit": "Veto (logit)",
    "veto_token": "Veto (token)",
    "entropy_aware": "Entropy-Aware OPD",
    "aopd": "AOPD",
    "aopd_clipped": "AOPD (clipped)",
    "extrapolation_cliff": "Extrapolation Cliff",
    "caopd": "CaOPD",
    "uni_opd": "Uni-OPD",
    "aligndistil": "AlignDistil",
    "coverage_opd": "Coverage OPD",
}


def load_training_logs() -> dict:
    """Load training logs for all methods."""
    data = {}
    for method in METHOD_ORDER:
        log_path = os.path.join(RESULTS_DIR, method, "training_log.jsonl")
        if not os.path.exists(log_path):
            continue

        entries = []
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))

        if entries:
            data[method] = entries

    return data


def load_eval_results() -> dict:
    """Load eval results at each step for all methods."""
    data = {}
    for method in METHOD_ORDER:
        eval_path = os.path.join(RESULTS_DIR, method, "eval_summary.json")
        if not os.path.exists(eval_path):
            continue
        with open(eval_path) as f:
            data[method] = json.load(f)
    return data


def plot_metric(
    data: dict,
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: str,
    figsize: tuple = (12, 7),
):
    """Plot a single metric across all methods."""
    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(METHOD_ORDER):
        if method not in data:
            continue
        entries = data[method]
        pairs = [(e.get("step", j), e.get(metric_key, None)) for j, e in enumerate(entries)]
        pairs = [(s, v) for s, v in pairs if v is not None]

        if not pairs:
            continue

        steps, values = zip(*pairs)

        label = METHOD_LABELS.get(method, method)
        color = COLORS[i % len(COLORS)]
        linewidth = 2.5 if method == "standard_opd" else 1.5
        linestyle = "--" if method == "standard_opd" else "-"

        ax.plot(steps, values, label=label, color=color, linewidth=linewidth, linestyle=linestyle)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_eval_progress(eval_data: dict, output_path: str):
    """Plot eval scores (pass@1) over training steps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for bench_idx, (bench, bench_label) in enumerate([
        ("humaneval_plus", "HumanEval+"),
        ("mbpp_plus", "MBPP+"),
    ]):
        ax = axes[bench_idx]
        for i, method in enumerate(METHOD_ORDER):
            if method not in eval_data:
                continue
            results = eval_data[method]
            steps = [r["step"] for r in results]
            scores = []
            for r in results:
                bench_data = r.get("benchmarks", {}).get(bench, {})
                p1 = bench_data.get("pass_at_1", {})
                score = p1.get("pass_at_1") if isinstance(p1, dict) else p1
                scores.append(score)

            valid = [(s, v) for s, v in zip(steps, scores) if v is not None]
            if not valid:
                continue
            steps_v, scores_v = zip(*valid)

            label = METHOD_LABELS.get(method, method)
            color = COLORS[i % len(COLORS)]
            linewidth = 2.5 if method == "standard_opd" else 1.5

            ax.plot(steps_v, scores_v, "o-", label=label, color=color, linewidth=linewidth, markersize=5)

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Pass@1 (%)", fontsize=12)
        ax.set_title(f"{bench_label} Pass@1 over Training", fontsize=13)
        ax.grid(True, alpha=0.3)

    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_comparison(eval_data: dict, output_path: str):
    """Bar chart comparing final pass@1 and pass@8 across all methods."""
    fig, ax = plt.subplots(figsize=(14, 7))

    methods_with_data = [m for m in METHOD_ORDER if m in eval_data and eval_data[m]]
    if not methods_with_data:
        print("No eval data to plot final comparison")
        return

    x = np.arange(len(methods_with_data))
    width = 0.35

    he_scores = []
    mbpp_scores = []
    for method in methods_with_data:
        final = eval_data[method][-1]
        he = final.get("benchmarks", {}).get("humaneval_plus", {}).get("pass_at_1", {})
        mbpp = final.get("benchmarks", {}).get("mbpp_plus", {}).get("pass_at_1", {})
        he_scores.append(he.get("pass_at_1", 0) if isinstance(he, dict) else (he or 0))
        mbpp_scores.append(mbpp.get("pass_at_1", 0) if isinstance(mbpp, dict) else (mbpp or 0))

    ax.bar(x - width / 2, he_scores, width, label="HumanEval+ P@1", color="#1f77b4")
    ax.bar(x + width / 2, mbpp_scores, width, label="MBPP+ P@1", color="#ff7f0e")

    labels = [METHOD_LABELS.get(m, m) for m in methods_with_data]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Pass@1 (%)", fontsize=12)
    ax.set_title("Final Evaluation: All Methods Compared", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading training logs...")
    train_data = load_training_logs()

    if train_data:
        print(f"Found training logs for {len(train_data)} methods")
        plot_metric(train_data, "loss", "Training Loss", "Loss", os.path.join(FIGURES_DIR, "loss_curves.png"))
        plot_metric(train_data, "kl_to_teacher", "KL Divergence to Teacher", "KL", os.path.join(FIGURES_DIR, "kl_curves.png"))
        plot_metric(train_data, "entropy", "Policy Entropy", "Entropy", os.path.join(FIGURES_DIR, "entropy_curves.png"))
    else:
        print("No training logs found yet")

    print("\nLoading eval results...")
    eval_data = load_eval_results()

    if eval_data:
        print(f"Found eval results for {len(eval_data)} methods")
        plot_eval_progress(eval_data, os.path.join(FIGURES_DIR, "eval_progress.png"))
        plot_final_comparison(eval_data, os.path.join(FIGURES_DIR, "final_comparison.png"))
    else:
        print("No eval results found yet")

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
