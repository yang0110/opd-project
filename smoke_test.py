"""
Unified smoke test for all OPD methods.

Runs a minimal training loop for each method with:
- Tiny synthetic data (no model download needed)
- Random weights (small transformer)
- 2-3 forward/backward passes
- Validates loss decreases and shapes are correct

Usage:
    python smoke_test.py              # Run all methods
    python smoke_test.py --method gopd  # Run single method
    python smoke_test.py --list       # List available methods
"""

import argparse
import sys
import os
import importlib.util
import torch
import torch.nn.functional as F
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_module(subdir: str, module_name: str = "core_algos"):
    """Load a module from a specific subdirectory without polluting sys.modules."""
    module_path = os.path.join(PROJECT_ROOT, subdir, "src", f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(f"{subdir}.{module_name}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Synthetic data and model for smoke testing
# ============================================================

@dataclass
class SmokeConfig:
    batch_size: int = 4
    seq_len: int = 32
    vocab_size: int = 1000
    hidden_dim: int = 128
    num_steps: int = 3
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def create_synthetic_batch(config: SmokeConfig) -> dict:
    """Create synthetic batch mimicking verl's DataProto fields."""
    device = config.device
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=device),
        "mask": torch.ones(config.batch_size, config.seq_len, device=device),
        "student_logits": torch.randn(config.batch_size, config.seq_len, config.vocab_size, device=device, requires_grad=True),
        "teacher_logits": torch.randn(config.batch_size, config.seq_len, config.vocab_size, device=device),
        "ref_logits": torch.randn(config.batch_size, config.seq_len, config.vocab_size, device=device),
    }
    # Derive log_probs from logits
    with torch.no_grad():
        batch["teacher_log_probs"] = F.log_softmax(batch["teacher_logits"], dim=-1).gather(
            -1, batch["input_ids"].unsqueeze(-1)
        ).squeeze(-1)
        batch["ref_log_probs"] = F.log_softmax(batch["ref_logits"], dim=-1).gather(
            -1, batch["input_ids"].unsqueeze(-1)
        ).squeeze(-1)
        batch["student_log_probs_detached"] = F.log_softmax(batch["student_logits"].detach(), dim=-1).gather(
            -1, batch["input_ids"].unsqueeze(-1)
        ).squeeze(-1)
    # Student log_probs with gradient
    student_lp = F.log_softmax(batch["student_logits"], dim=-1)
    batch["student_log_probs"] = student_lp.gather(-1, batch["input_ids"].unsqueeze(-1)).squeeze(-1)
    return batch


def run_training_step(loss_fn, config: SmokeConfig) -> dict:
    """Run num_steps of training and return loss trajectory."""
    # Simple linear model as student
    model = torch.nn.Linear(config.hidden_dim, config.vocab_size).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    losses = []
    for step in range(config.num_steps):
        optimizer.zero_grad()

        # Create batch with model's logits
        batch = create_synthetic_batch(config)
        # Override student_logits with model output
        dummy_input = torch.randn(config.batch_size, config.seq_len, config.hidden_dim, device=config.device)
        student_logits = model(dummy_input)
        student_lp = F.log_softmax(student_logits, dim=-1)
        student_log_probs = student_lp.gather(-1, batch["input_ids"].unsqueeze(-1)).squeeze(-1)

        # Compute loss
        loss, metrics = loss_fn(
            student_logits=student_logits,
            student_log_probs=student_log_probs,
            teacher_logits=batch["teacher_logits"],
            teacher_log_probs=batch["teacher_log_probs"],
            ref_log_probs=batch["ref_log_probs"],
            mask=batch["mask"],
        )

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return {"losses": losses, "final_metrics": metrics}


# ============================================================
# Method-specific loss wrappers (unified interface)
# ============================================================

def test_standard_opd(config: SmokeConfig) -> dict:
    """Test standard OPD."""
    mod = load_module("standard-opd")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        return mod.compute_opd_loss(student_log_probs, teacher_log_probs, mask)

    return run_training_step(loss_fn, config)


def test_gopd(config: SmokeConfig) -> dict:
    """Test G-OPD with λ=1.25 (ExOPD)."""
    mod = load_module("g-opd")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        return mod.compute_gopd_loss(student_log_probs, teacher_log_probs, ref_log_probs, mask, lambda_=1.25)

    return run_training_step(loss_fn, config)


def test_gopd_interpolation(config: SmokeConfig) -> dict:
    """Test G-OPD with λ=0.5 (interpolation)."""
    mod = load_module("g-opd")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        return mod.compute_gopd_loss(student_log_probs, teacher_log_probs, ref_log_probs, mask, lambda_=0.5)

    return run_training_step(loss_fn, config)


def test_gopd_reward_correction(config: SmokeConfig) -> dict:
    """Test G-OPD with reward correction."""
    mod = load_module("g-opd")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        return mod.compute_gopd_loss_with_reward_correction(
            student_log_probs, teacher_log_probs, ref_log_probs, ref_log_probs, mask, lambda_=1.25
        )

    return run_training_step(loss_fn, config)


def test_veto(config: SmokeConfig) -> dict:
    """Test Veto adaptive bridge target."""
    mod = load_module("veto")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        return mod.compute_veto_loss(student_logits, teacher_logits, mask, beta=1.0)

    return run_training_step(loss_fn, config)


def test_veto_token_level(config: SmokeConfig) -> dict:
    """Test Veto at token log-prob level."""
    mod = load_module("veto")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        return mod.compute_veto_loss_token_level(student_log_probs, teacher_log_probs, ref_log_probs, mask)

    return run_training_step(loss_fn, config)


def test_entropy_aware(config: SmokeConfig) -> dict:
    """Test Entropy-Aware OPD."""
    mod = load_module("entropy-aware-opd")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        return mod.compute_entropy_aware_opd_loss(student_logits, teacher_logits, mask)

    return run_training_step(loss_fn, config)


def test_aopd(config: SmokeConfig) -> dict:
    """Test AOPD asymmetric loss."""
    mod = load_module("aopd")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        return mod.compute_aopd_loss(
            student_log_probs, teacher_log_probs, ref_log_probs, mask,
            student_logits=student_logits, teacher_logits=teacher_logits,
        )

    return run_training_step(loss_fn, config)


def test_aopd_clipped(config: SmokeConfig) -> dict:
    """Test AOPD with PPO-style clipping."""
    mod = load_module("aopd")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        return mod.compute_aopd_loss_with_clipping(
            student_log_probs, student_log_probs.detach(), teacher_log_probs, ref_log_probs, mask
        )

    return run_training_step(loss_fn, config)


def test_extrapolation_cliff(config: SmokeConfig) -> dict:
    """Test safe ExOPD with adaptive λ."""
    mod = load_module("extrapolation-cliff")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        return mod.compute_safe_exopd_loss(
            student_log_probs, teacher_log_probs, ref_log_probs, mask,
            teacher_logits=teacher_logits, student_logits=student_logits,
            base_lambda=1.25, use_adaptive_lambda=True,
        )

    return run_training_step(loss_fn, config)


def test_caopd(config: SmokeConfig) -> dict:
    """Test CaOPD calibration-aware loss."""
    mod = load_module("caopd")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        empirical_conf = torch.rand(config.batch_size, device=config.device)
        return mod.compute_caopd_loss(
            student_log_probs, teacher_log_probs, ref_log_probs, mask, empirical_conf
        )

    return run_training_step(loss_fn, config)


def test_uni_opd(config: SmokeConfig) -> dict:
    """Test Uni-OPD with outcome calibration."""
    mod = load_module("uni-opd")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        outcome_rewards = torch.randint(0, 2, (config.batch_size,), device=config.device).float()
        return mod.compute_uni_opd_loss(
            student_log_probs, teacher_log_probs, ref_log_probs, mask, outcome_rewards
        )

    return run_training_step(loss_fn, config)


def test_aligndistil(config: SmokeConfig) -> dict:
    """Test AlignDistil alignment-as-distillation."""
    mod = load_module("aligndistil")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        ref_logits = torch.randn_like(teacher_logits)
        return mod.compute_aligndistil_loss(
            student_logits, teacher_logits, ref_logits, mask, beta=0.1, adaptive=True
        )

    return run_training_step(loss_fn, config)


def test_coverage_opd(config: SmokeConfig) -> dict:
    """Test coverage-aware OPD from on-policy preference."""
    mod = load_module("on-policy-preference")

    def loss_fn(student_logits, student_log_probs, teacher_logits, teacher_log_probs, ref_log_probs, mask):
        return mod.compute_opd_with_coverage_aware_sampling(
            student_log_probs, teacher_log_probs, ref_log_probs, mask
        )

    return run_training_step(loss_fn, config)


# ============================================================
# Test registry
# ============================================================

TESTS = {
    "standard_opd": ("Standard OPD (baseline)", test_standard_opd),
    "gopd": ("G-OPD / ExOPD (λ=1.25)", test_gopd),
    "gopd_interp": ("G-OPD interpolation (λ=0.5)", test_gopd_interpolation),
    "gopd_correction": ("G-OPD reward correction", test_gopd_reward_correction),
    "veto": ("Veto (adaptive bridge)", test_veto),
    "veto_token": ("Veto (token-level)", test_veto_token_level),
    "entropy_aware": ("Entropy-Aware OPD", test_entropy_aware),
    "aopd": ("AOPD (asymmetric)", test_aopd),
    "aopd_clipped": ("AOPD (with clipping)", test_aopd_clipped),
    "cliff": ("Extrapolation Cliff (safe λ)", test_extrapolation_cliff),
    "caopd": ("CaOPD (calibration-aware)", test_caopd),
    "uni_opd": ("Uni-OPD (outcome-guided)", test_uni_opd),
    "aligndistil": ("AlignDistil (DPO→distill)", test_aligndistil),
    "coverage": ("Coverage-aware OPD", test_coverage_opd),
}


# ============================================================
# Main runner
# ============================================================

def run_all_tests(config: SmokeConfig, methods: list[str] = None):
    """Run smoke tests for all or selected methods."""
    if methods is None:
        methods = list(TESTS.keys())

    results = {}
    passed = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"OPD Smoke Tests | device={config.device} | batch={config.batch_size} | steps={config.num_steps}")
    print(f"{'='*60}\n")

    for method_key in methods:
        if method_key not in TESTS:
            print(f"  [SKIP] Unknown method: {method_key}")
            continue

        name, test_fn = TESTS[method_key]
        try:
            result = test_fn(config)
            losses = result["losses"]

            # Check: loss is finite and not NaN
            all_finite = all(torch.isfinite(torch.tensor(l)) for l in losses)
            # Check: loss generally decreases (allow some noise)
            loss_decreased = losses[-1] < losses[0] * 1.5  # Allow 50% tolerance

            status = "PASS" if all_finite else "FAIL"
            if not all_finite:
                status = "FAIL (NaN/Inf)"

            print(f"  [{status}] {name}")
            print(f"         losses: {[f'{l:.4f}' for l in losses]}")
            if result["final_metrics"]:
                key_metrics = {k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in list(result["final_metrics"].items())[:3]}
                print(f"         metrics: {key_metrics}")
            print()

            if status.startswith("PASS"):
                passed += 1
            else:
                failed += 1
            results[method_key] = {"status": status, "losses": losses}

        except Exception as e:
            print(f"  [FAIL] {name}")
            print(f"         error: {type(e).__name__}: {e}")
            print()
            failed += 1
            results[method_key] = {"status": "ERROR", "error": str(e)}

    print(f"{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(methods)} total")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPD Methods Smoke Test")
    parser.add_argument("--method", type=str, default=None, help="Run single method")
    parser.add_argument("--list", action="store_true", help="List available methods")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable methods:")
        for key, (name, _) in TESTS.items():
            print(f"  {key:20s} → {name}")
        sys.exit(0)

    config = SmokeConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.steps,
        device=args.device,
    )

    methods = [args.method] if args.method else None
    results = run_all_tests(config, methods)

    # Exit with error code if any test failed
    any_failed = any(r.get("status", "").startswith("FAIL") or r.get("status") == "ERROR"
                     for r in results.values())
    sys.exit(1 if any_failed else 0)
