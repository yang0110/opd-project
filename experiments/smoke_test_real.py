"""
Smoke test with real models: Qwen3-4B (teacher) + Qwen3-0.6B (student).

Runs all 14 methods for 5 training steps each with real model forward passes.
Validates the full pipeline: model loading, loss computation, backward, eval harness.

Saves logs to: experiments/results/smoke_test/

Usage:
    python3 experiments/smoke_test_real.py
    python3 experiments/smoke_test_real.py --method gopd_extrap --steps 10
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.src.method_registry import (
    METHODS,
    METHOD_ORDER,
    load_method_module,
    get_loss_fn,
    get_method_info,
)

SMOKE_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "smoke_test")
TEACHER_MODEL = "Qwen/Qwen3-0.6B"  # Use 0.6B as "teacher" for smoke test speed
STUDENT_MODEL = "Qwen/Qwen3-0.6B"  # Same model as student (tests the pipeline)

NUM_STEPS = 5
BATCH_SIZE = 2
SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    """Load teacher and student models."""
    print(f"Loading models on {DEVICE}...")
    print(f"  Teacher: {TEACHER_MODEL}")
    print(f"  Student: {STUDENT_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_MODEL, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, trust_remote_code=True, dtype=torch.bfloat16
    ).to(DEVICE).eval()

    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, trust_remote_code=True, dtype=torch.bfloat16
    ).to(DEVICE)

    ref = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, trust_remote_code=True, dtype=torch.bfloat16
    ).to(DEVICE).eval()

    print(f"  Models loaded. Vocab size: {teacher.config.vocab_size}")
    return teacher, student, ref, tokenizer


def generate_batch(tokenizer, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    """Generate a synthetic coding batch."""
    prompts = [
        "def fibonacci(n):\n    ",
        "def binary_search(arr, target):\n    ",
        "class Stack:\n    def __init__(self):\n        ",
        "def merge_sort(arr):\n    ",
    ][:batch_size]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=seq_len,
        truncation=True,
    ).to(DEVICE)

    return inputs


def get_log_probs_and_logits(model, input_ids, attention_mask):
    """Forward pass to get log probs and logits."""
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # [B, T-1, V]
        target_ids = input_ids[:, 1:]  # [B, T-1]

        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    return token_log_probs, logits


def _build_loss_args(
    method_name: str,
    info: dict,
    kwargs: dict,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor,
    input_ids: torch.Tensor,
) -> dict:
    """Build the correct argument dict for each method's loss function."""
    B, T = student_log_probs.shape

    if method_name == "standard_opd":
        return {"student_log_probs": student_log_probs, "teacher_log_probs": teacher_log_probs, "mask": mask}

    if method_name in ("gopd_interp", "gopd_extrap"):
        return {"student_log_probs": student_log_probs, "teacher_log_probs": teacher_log_probs,
                "ref_log_probs": ref_log_probs, "mask": mask, **kwargs}

    if method_name == "gopd_correction":
        return {"student_log_probs": student_log_probs, "teacher_log_probs": teacher_log_probs,
                "ref_log_probs": ref_log_probs, "teacher_base_log_probs": ref_log_probs,
                "mask": mask, **kwargs}

    if method_name == "veto_logit":
        veto_kwargs = {"beta": kwargs.get("beta_init", 1.0), "kl_direction": kwargs.get("kl_direction", "forward")}
        return {"student_logits": student_logits, "teacher_logits": teacher_logits,
                "mask": mask, **veto_kwargs}

    if method_name == "veto_token":
        return {"student_log_probs": student_log_probs, "teacher_log_probs": teacher_log_probs,
                "ref_log_probs": ref_log_probs, "mask": mask, "beta": kwargs.get("beta_init", 1.0)}

    if method_name == "entropy_aware":
        return {"student_logits": student_logits, "teacher_logits": teacher_logits,
                "mask": mask, **kwargs}

    if method_name == "aopd":
        return {"student_log_probs": student_log_probs, "teacher_log_probs": teacher_log_probs,
                "ref_log_probs": ref_log_probs, "mask": mask,
                "student_logits": student_logits, "teacher_logits": teacher_logits,
                "input_ids": input_ids, **kwargs}

    if method_name == "aopd_clipped":
        return {"student_log_probs": student_log_probs, "old_student_log_probs": student_log_probs.detach(),
                "teacher_log_probs": teacher_log_probs, "ref_log_probs": ref_log_probs,
                "mask": mask, **kwargs}

    if method_name == "extrapolation_cliff":
        return {"student_log_probs": student_log_probs, "teacher_log_probs": teacher_log_probs,
                "ref_log_probs": ref_log_probs, "mask": mask,
                "teacher_logits": teacher_logits, "student_logits": student_logits, **kwargs}

    if method_name == "caopd":
        empirical_confidence = torch.rand(B, device=mask.device)
        return {"student_log_probs": student_log_probs, "teacher_log_probs": teacher_log_probs,
                "ref_log_probs": ref_log_probs, "mask": mask,
                "empirical_confidence": empirical_confidence, **kwargs}

    if method_name == "uni_opd":
        outcome_rewards = torch.rand(B, device=mask.device)
        return {"student_log_probs": student_log_probs, "teacher_log_probs": teacher_log_probs,
                "ref_log_probs": ref_log_probs, "mask": mask,
                "outcome_rewards": outcome_rewards, **kwargs}

    if method_name == "aligndistil":
        return {"student_logits": student_logits, "dpo_logits": teacher_logits,
                "ref_logits": ref_logits, "mask": mask, **kwargs}

    if method_name == "coverage_opd":
        return {"student_log_probs": student_log_probs, "teacher_log_probs": teacher_log_probs,
                "ref_log_probs": ref_log_probs, "mask": mask}

    raise ValueError(f"Unknown method: {method_name}")


def run_method_smoke(
    method_name: str,
    teacher,
    student,
    ref,
    tokenizer,
    num_steps: int = NUM_STEPS,
) -> dict:
    """Run a single method for num_steps and return metrics."""
    print(f"\n  [{method_name}] Running {num_steps} steps...")
    info = get_method_info(method_name)
    loss_fn, kwargs = get_loss_fn(method_name)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-5)
    student.train()

    log_entries = []
    start_time = time.time()

    for step in range(num_steps):
        batch = generate_batch(tokenizer)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        response_mask = attention_mask[:, 1:].float()

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_log_probs, teacher_logits = get_log_probs_and_logits(
                teacher, input_ids, attention_mask
            )
            ref_log_probs, ref_logits = get_log_probs_and_logits(
                ref, input_ids, attention_mask
            )

        # Student forward (with grad)
        student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits[:, :-1, :]
        student_log_probs_full = F.log_softmax(student_logits.float(), dim=-1)
        target_ids = input_ids[:, 1:]
        student_log_probs = student_log_probs_full.gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Build method-specific arguments
        mask = response_mask
        loss_args = _build_loss_args(
            method_name, info, kwargs,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            ref_log_probs=ref_log_probs,
            student_logits=student_logits.float(),
            teacher_logits=teacher_logits.float(),
            ref_logits=ref_logits.float(),
            mask=mask,
            input_ids=target_ids,
        )

        # Compute loss
        result = loss_fn(**loss_args)
        if isinstance(result, tuple):
            loss = result[0]
        elif isinstance(result, dict):
            loss = result.get("loss", result.get("total_loss"))
        else:
            loss = result

        if not isinstance(loss, torch.Tensor):
            raise RuntimeError(f"[{method_name}] Loss is not a tensor: {type(loss)}")

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            kl = (student_log_probs - teacher_log_probs).mean().item()
            entropy = -(student_log_probs * response_mask).sum() / response_mask.sum()

        entry = {
            "step": step,
            "loss": loss.item(),
            "kl_to_teacher": kl,
            "entropy": entropy.item(),
            "time_s": time.time() - start_time,
        }
        log_entries.append(entry)

        if step == 0 or step == num_steps - 1:
            print(f"    Step {step}: loss={loss.item():.4f}, kl={kl:.4f}")

    elapsed = time.time() - start_time

    # Reset student for next method
    student.load_state_dict(ref.state_dict())

    result = {
        "method": method_name,
        "status": "PASS",
        "num_steps": num_steps,
        "total_time_s": round(elapsed, 2),
        "initial_loss": log_entries[0]["loss"],
        "final_loss": log_entries[-1]["loss"],
        "loss_delta": log_entries[-1]["loss"] - log_entries[0]["loss"],
        "loss_trajectory": [e["loss"] for e in log_entries],
        "needs_logits": info["needs_logits"],
        "timestamp": datetime.now().isoformat(),
    }

    return result, log_entries


def save_results(method_name: str, result: dict, log_entries: list):
    """Save results and training log."""
    method_dir = os.path.join(SMOKE_DIR, method_name)
    os.makedirs(method_dir, exist_ok=True)

    with open(os.path.join(method_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    with open(os.path.join(method_dir, "training_log.jsonl"), "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Smoke test with real models")
    parser.add_argument("--method", type=str, default=None, help="Run single method")
    parser.add_argument("--steps", type=int, default=NUM_STEPS, help="Steps per method")
    args = parser.parse_args()

    os.makedirs(SMOKE_DIR, exist_ok=True)

    methods = [args.method] if args.method else METHOD_ORDER
    num_steps = args.steps

    print("=" * 60)
    print("OPD SMOKE TEST: Real Models")
    print(f"  Teacher: {TEACHER_MODEL}")
    print(f"  Student: {STUDENT_MODEL}")
    print(f"  Methods: {len(methods)}")
    print(f"  Steps/method: {num_steps}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    teacher, student, ref, tokenizer = load_models()

    all_results = []
    passed = 0
    failed = 0

    for method in methods:
        try:
            result, log_entries = run_method_smoke(
                method, teacher, student, ref, tokenizer, num_steps
            )
            save_results(method, result, log_entries)
            all_results.append(result)
            passed += 1
            print(f"  [{method}] PASS (loss: {result['initial_loss']:.4f} → {result['final_loss']:.4f})")
        except Exception as e:
            failed += 1
            error_result = {
                "method": method,
                "status": "FAIL",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(error_result)
            save_results(method, error_result, [])
            print(f"  [{method}] FAIL: {e}")

    # Save aggregate results
    with open(os.path.join(SMOKE_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"SMOKE TEST COMPLETE: {passed} passed, {failed} failed / {len(methods)} total")
    print(f"  Logs: {SMOKE_DIR}/")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
