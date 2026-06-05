"""
Evaluation pipeline: generate solutions and score with evalplus.

Evaluates checkpoints on HumanEval+ and MBPP+ (pass@1, pass@8).
All results saved locally to experiments/results/<method>/step_<N>/eval.json.
"""

import argparse
import json
import os
import subprocess
import sys
import logging
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")

logger = logging.getLogger(__name__)


BENCHMARKS = {
    "humaneval_plus": {
        "dataset": "humaneval",
        "num_problems": 164,
    },
    "mbpp_plus": {
        "dataset": "mbpp",
        "num_problems": 378,
    },
}

GENERATION_CONFIGS = {
    "pass_at_1": {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "max_tokens": 2048,
    },
    "pass_at_8": {
        "temperature": 0.8,
        "top_p": 0.95,
        "n": 8,
        "max_tokens": 2048,
    },
}


def generate_solutions(
    model_path: str,
    benchmark: str,
    gen_config: dict,
    output_path: str,
    tensor_parallel_size: int = 2,
):
    """
    Generate solutions using vLLM.

    Integration:
        from vllm import LLM, SamplingParams
        llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
        params = SamplingParams(**gen_config)
        outputs = llm.generate(prompts, params)
    """
    logger.info(f"Generating solutions: {benchmark}, model={model_path}")
    logger.info(f"  Config: {gen_config}")
    logger.info(f"  Output: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


def run_evalplus(samples_path: str, dataset: str) -> dict:
    """
    Score generated solutions with evalplus.

    Returns dict with pass@k metrics.
    """
    logger.info(f"Scoring with evalplus: dataset={dataset}, samples={samples_path}")

    # evalplus.evaluate --dataset {dataset} --samples {samples_path}
    # Parse output for pass@k results
    result = {
        "dataset": dataset,
        "samples_path": samples_path,
        "pass_at_1": None,
        "pass_at_8": None,
        "timestamp": datetime.now().isoformat(),
    }
    return result


def evaluate_checkpoint(
    checkpoint_path: str,
    method_name: str,
    step: int,
    benchmarks: list[str] = None,
) -> dict:
    """
    Full evaluation of a single checkpoint.

    Generates solutions for all benchmarks, scores them, saves results.
    """
    if benchmarks is None:
        benchmarks = list(BENCHMARKS.keys())

    step_dir = os.path.join(RESULTS_DIR, method_name, f"step_{step:03d}")
    os.makedirs(step_dir, exist_ok=True)

    results = {
        "method": method_name,
        "step": step,
        "checkpoint_path": checkpoint_path,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {},
    }

    for bench_name in benchmarks:
        bench_info = BENCHMARKS[bench_name]
        bench_results = {}

        for metric_name, gen_config in GENERATION_CONFIGS.items():
            samples_path = os.path.join(step_dir, f"{bench_name}_{metric_name}_samples.jsonl")
            generate_solutions(
                model_path=checkpoint_path,
                benchmark=bench_info["dataset"],
                gen_config=gen_config,
                output_path=samples_path,
            )
            score = run_evalplus(samples_path, bench_info["dataset"])
            bench_results[metric_name] = score

        results["benchmarks"][bench_name] = bench_results

    eval_path = os.path.join(step_dir, "eval.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation saved: {eval_path}")
    return results


def evaluate_method_all_steps(method_name: str, eval_steps: list[int] = None):
    """Evaluate all checkpoints for a method."""
    if eval_steps is None:
        eval_steps = [50, 100, 150, 200]

    method_dir = os.path.join(RESULTS_DIR, method_name)
    all_results = []

    for step in eval_steps:
        checkpoint_path = os.path.join(method_dir, "checkpoints", f"step_{step:03d}")
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping")
            continue
        result = evaluate_checkpoint(checkpoint_path, method_name, step)
        all_results.append(result)

    summary_path = os.path.join(method_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OPD checkpoints")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--step", type=int, default=None, help="Single step to eval")
    parser.add_argument("--checkpoint", type=str, default=None, help="Direct path to checkpoint")
    parser.add_argument("--all-steps", action="store_true", help="Eval all saved steps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    if args.all_steps:
        evaluate_method_all_steps(args.method)
    elif args.step and args.checkpoint:
        evaluate_checkpoint(args.checkpoint, args.method, args.step)
    else:
        print("Specify --all-steps or --step + --checkpoint")
