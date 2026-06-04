"""
Run each OPD method individually with 10 training steps and save stepwise logs.

Usage:
    python3 run_individual_tests.py              # Run all
    python3 run_individual_tests.py --method gopd  # Run one
    python3 run_individual_tests.py --steps 20     # Custom steps

Logs saved to: logs/<method_name>/training_log.jsonl
"""

import argparse
import os
import sys
import json
import time
import importlib.util
import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def load_module(subdir: str, module_name: str = "core_algos"):
    module_path = os.path.join(PROJECT_ROOT, subdir, "src", f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(f"{subdir}.{module_name}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class TrainConfig:
    batch_size: int = 8
    seq_len: int = 64
    vocab_size: int = 2000
    hidden_dim: int = 256
    num_steps: int = 10
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class StepLogger:
    """Logs each training step to JSONL file."""

    def __init__(self, method_name: str, log_dir: str):
        self.method_name = method_name
        self.log_dir = os.path.join(log_dir, method_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "training_log.jsonl")
        self.summary_path = os.path.join(self.log_dir, "summary.json")
        self.entries = []

        # Clear previous log
        with open(self.log_path, "w") as f:
            pass

    def log_step(self, step: int, loss: float, metrics: dict, elapsed_ms: float):
        entry = {
            "step": step,
            "loss": loss,
            "elapsed_ms": round(elapsed_ms, 2),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        self.entries.append(entry)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_summary(self, config: TrainConfig, total_time: float, status: str, error: str = None):
        losses = [e["loss"] for e in self.entries]
        summary = {
            "method": self.method_name,
            "status": status,
            "total_time_s": round(total_time, 3),
            "num_steps": len(self.entries),
            "config": asdict(config),
            "loss_trajectory": losses,
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "loss_delta": round(losses[-1] - losses[0], 6) if len(losses) >= 2 else None,
            "min_loss": min(losses) if losses else None,
            "max_loss": max(losses) if losses else None,
            "timestamp": datetime.now().isoformat(),
        }
        if error:
            summary["error"] = error

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary


def create_batch(config: TrainConfig, model: torch.nn.Module):
    """Create a synthetic training batch."""
    device = config.device
    input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=device)
    mask = torch.ones(config.batch_size, config.seq_len, device=device)

    # Student logits from model
    dummy_input = torch.randn(config.batch_size, config.seq_len, config.hidden_dim, device=device)
    student_logits = model(dummy_input)

    # Teacher and ref logits (fixed random, consistent across steps via seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(config.seed + 1000)
    teacher_logits = torch.randn(config.batch_size, config.seq_len, config.vocab_size, device=device, generator=gen)
    gen.manual_seed(config.seed + 2000)
    ref_logits = torch.randn(config.batch_size, config.seq_len, config.vocab_size, device=device, generator=gen)

    # Derive per-token log probs
    student_lp = F.log_softmax(student_logits, dim=-1)
    student_log_probs = student_lp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1).gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1).gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    return {
        "input_ids": input_ids,
        "mask": mask,
        "student_logits": student_logits,
        "student_log_probs": student_log_probs,
        "teacher_logits": teacher_logits,
        "teacher_log_probs": teacher_log_probs,
        "ref_logits": ref_logits,
        "ref_log_probs": ref_log_probs,
    }


def run_method(method_key: str, loss_fn_factory, config: TrainConfig) -> dict:
    """Run a single method for N steps with logging."""
    logger = StepLogger(method_key, LOG_DIR)

    torch.manual_seed(config.seed)
    model = torch.nn.Linear(config.hidden_dim, config.vocab_size).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loss_fn = loss_fn_factory(config)

    start_time = time.time()
    try:
        for step in range(config.num_steps):
            step_start = time.time()
            optimizer.zero_grad()

            batch = create_batch(config, model)
            loss, metrics = loss_fn(batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            elapsed_ms = (time.time() - step_start) * 1000
            logger.log_step(step, loss.item(), metrics, elapsed_ms)

        total_time = time.time() - start_time
        summary = logger.save_summary(config, total_time, status="PASS")
        return summary

    except Exception as e:
        total_time = time.time() - start_time
        summary = logger.save_summary(config, total_time, status="FAIL", error=str(e))
        return summary


# ============================================================
# Loss function factories for each method
# ============================================================

def factory_standard_opd(config):
    mod = load_module("standard-opd")
    def loss_fn(batch):
        return mod.compute_opd_loss(batch["student_log_probs"], batch["teacher_log_probs"], batch["mask"])
    return loss_fn


def factory_gopd_extrapolation(config):
    mod = load_module("g-opd")
    def loss_fn(batch):
        return mod.compute_gopd_loss(
            batch["student_log_probs"], batch["teacher_log_probs"],
            batch["ref_log_probs"], batch["mask"], lambda_=1.25
        )
    return loss_fn


def factory_gopd_interpolation(config):
    mod = load_module("g-opd")
    def loss_fn(batch):
        return mod.compute_gopd_loss(
            batch["student_log_probs"], batch["teacher_log_probs"],
            batch["ref_log_probs"], batch["mask"], lambda_=0.5
        )
    return loss_fn


def factory_gopd_reward_correction(config):
    mod = load_module("g-opd")
    def loss_fn(batch):
        return mod.compute_gopd_loss_with_reward_correction(
            batch["student_log_probs"], batch["teacher_log_probs"],
            batch["ref_log_probs"], batch["ref_log_probs"], batch["mask"], lambda_=1.25
        )
    return loss_fn


def factory_veto(config):
    mod = load_module("veto")
    def loss_fn(batch):
        return mod.compute_veto_loss(
            batch["student_logits"], batch["teacher_logits"], batch["mask"], beta=1.0
        )
    return loss_fn


def factory_veto_token(config):
    mod = load_module("veto")
    def loss_fn(batch):
        return mod.compute_veto_loss_token_level(
            batch["student_log_probs"], batch["teacher_log_probs"],
            batch["ref_log_probs"], batch["mask"]
        )
    return loss_fn


def factory_entropy_aware(config):
    mod = load_module("entropy-aware-opd")
    def loss_fn(batch):
        return mod.compute_entropy_aware_opd_loss(
            batch["student_logits"], batch["teacher_logits"], batch["mask"]
        )
    return loss_fn


def factory_aopd(config):
    mod = load_module("aopd")
    def loss_fn(batch):
        return mod.compute_aopd_loss(
            batch["student_log_probs"], batch["teacher_log_probs"],
            batch["ref_log_probs"], batch["mask"],
            student_logits=batch["student_logits"], teacher_logits=batch["teacher_logits"],
        )
    return loss_fn


def factory_aopd_clipped(config):
    mod = load_module("aopd")
    def loss_fn(batch):
        return mod.compute_aopd_loss_with_clipping(
            batch["student_log_probs"], batch["student_log_probs"].detach(),
            batch["teacher_log_probs"], batch["ref_log_probs"], batch["mask"]
        )
    return loss_fn


def factory_extrapolation_cliff(config):
    mod = load_module("extrapolation-cliff")
    def loss_fn(batch):
        return mod.compute_safe_exopd_loss(
            batch["student_log_probs"], batch["teacher_log_probs"],
            batch["ref_log_probs"], batch["mask"],
            teacher_logits=batch["teacher_logits"], student_logits=batch["student_logits"],
            base_lambda=1.25, use_adaptive_lambda=True,
        )
    return loss_fn


def factory_caopd(config):
    mod = load_module("caopd")
    def loss_fn(batch):
        empirical_conf = torch.rand(config.batch_size, device=config.device)
        return mod.compute_caopd_loss(
            batch["student_log_probs"], batch["teacher_log_probs"],
            batch["ref_log_probs"], batch["mask"], empirical_conf
        )
    return loss_fn


def factory_uni_opd(config):
    mod = load_module("uni-opd")
    def loss_fn(batch):
        outcome_rewards = torch.randint(0, 2, (config.batch_size,), device=config.device).float()
        return mod.compute_uni_opd_loss(
            batch["student_log_probs"], batch["teacher_log_probs"],
            batch["ref_log_probs"], batch["mask"], outcome_rewards
        )
    return loss_fn


def factory_aligndistil(config):
    mod = load_module("aligndistil")
    def loss_fn(batch):
        return mod.compute_aligndistil_loss(
            batch["student_logits"], batch["teacher_logits"],
            batch["ref_logits"], batch["mask"], beta=0.1, adaptive=True
        )
    return loss_fn


def factory_coverage_opd(config):
    mod = load_module("on-policy-preference")
    def loss_fn(batch):
        return mod.compute_opd_with_coverage_aware_sampling(
            batch["student_log_probs"], batch["teacher_log_probs"],
            batch["ref_log_probs"], batch["mask"]
        )
    return loss_fn


# ============================================================
# Method registry
# ============================================================

METHODS = {
    "standard_opd":       ("Standard OPD",              factory_standard_opd),
    "gopd_extrap":        ("G-OPD ExOPD (λ=1.25)",     factory_gopd_extrapolation),
    "gopd_interp":        ("G-OPD Interpolation (λ=0.5)", factory_gopd_interpolation),
    "gopd_correction":    ("G-OPD Reward Correction",   factory_gopd_reward_correction),
    "veto":               ("Veto (Adaptive Bridge)",    factory_veto),
    "veto_token":         ("Veto (Token-Level)",        factory_veto_token),
    "entropy_aware":      ("Entropy-Aware OPD",         factory_entropy_aware),
    "aopd":               ("AOPD (Asymmetric)",         factory_aopd),
    "aopd_clipped":       ("AOPD (Clipped)",            factory_aopd_clipped),
    "extrapolation_cliff":("Extrapolation Cliff",       factory_extrapolation_cliff),
    "caopd":              ("CaOPD (Calibration)",       factory_caopd),
    "uni_opd":            ("Uni-OPD (Outcome-Guided)",  factory_uni_opd),
    "aligndistil":        ("AlignDistil",               factory_aligndistil),
    "coverage_opd":       ("Coverage-Aware OPD",        factory_coverage_opd),
}


def main():
    parser = argparse.ArgumentParser(description="Run individual OPD method training with stepwise logs")
    parser.add_argument("--method", type=str, default=None, help="Run single method (or 'all')")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list", action="store_true", help="List methods")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable methods:")
        for key, (name, _) in METHODS.items():
            print(f"  {key:22s} → {name}")
        sys.exit(0)

    config = TrainConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.steps,
        lr=args.lr,
        device=args.device,
    )

    methods_to_run = [args.method] if args.method and args.method != "all" else list(METHODS.keys())

    print(f"\n{'='*70}")
    print(f"  OPD Individual Training | steps={config.num_steps} | batch={config.batch_size} | seq_len={config.seq_len}")
    print(f"  device={config.device} | lr={config.lr}")
    print(f"  Logs → {LOG_DIR}/")
    print(f"{'='*70}\n")

    all_summaries = {}
    passed = 0
    failed = 0

    for method_key in methods_to_run:
        if method_key not in METHODS:
            print(f"  [SKIP] Unknown: {method_key}")
            continue

        name, factory = METHODS[method_key]
        print(f"  Running: {name} ({method_key})...")

        summary = run_method(method_key, factory, config)
        all_summaries[method_key] = summary

        status = summary["status"]
        if status == "PASS":
            passed += 1
            losses = summary["loss_trajectory"]
            print(f"    [{status}] loss: {losses[0]:.4f} → {losses[-1]:.4f} "
                  f"(Δ={summary['loss_delta']:+.4f}) | {summary['total_time_s']:.2f}s")
        else:
            failed += 1
            print(f"    [{status}] error: {summary.get('error', 'unknown')}")
        print()

    # Save global summary
    global_summary_path = os.path.join(LOG_DIR, "all_results.json")
    with open(global_summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"{'='*70}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"  Logs saved to: {LOG_DIR}/")
    print(f"  Global summary: {global_summary_path}")
    print(f"{'='*70}\n")

    # Print comparison table
    print(f"  {'Method':<24} {'Init Loss':>10} {'Final Loss':>10} {'Delta':>10} {'Time':>8}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for key, s in all_summaries.items():
        if s["status"] == "PASS":
            print(f"  {key:<24} {s['initial_loss']:>10.4f} {s['final_loss']:>10.4f} "
                  f"{s['loss_delta']:>+10.4f} {s['total_time_s']:>7.2f}s")
        else:
            print(f"  {key:<24} {'FAILED':>10}")
    print()

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
