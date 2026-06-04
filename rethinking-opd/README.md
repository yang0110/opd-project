# Rethinking OPD: Phenomenology, Mechanism, and Recipe

**Paper:** [arXiv:2604.13016](https://arxiv.org/abs/2604.13016) — "Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe"

## What Problem It Solves

All other methods in this project improve the OPD **objective function**. But what if the objective is fine and the real problem is that the teacher and student are fundamentally incompatible? This paper asks: **when does OPD work, and when does it fail?**

Key finding: No amount of loss engineering helps if teacher and student have incompatible thinking patterns. This module provides diagnostic tools to predict OPD success before investing compute.

## How It Works

Two conditions for OPD success:

1. **Compatible thinking patterns**: Teacher and student need to reason in similar ways at student-visited states
2. **Novel capability**: Teacher must provide something the student hasn't already learned from other training

Observable mechanism of successful OPD:
- A **small shared token set** concentrates 97-99% of probability mass
- Success manifests as **progressive alignment** of high-probability tokens in this shared set
- Failure manifests as alignment stalling or oscillating

This module provides:
- `compute_shared_token_set()`: measure overlap between teacher and student top-K tokens
- `compute_thinking_pattern_similarity()`: correlation of log-prob profiles
- `predict_opd_success()`: pre-training diagnostic
- `monitor_progressive_alignment()`: during-training health check

## When to Use

- **Before starting OPD**: Run compatibility diagnostics to predict if it will work
- **During OPD training**: Monitor alignment progress to detect stalling early
- **Model selection**: Choose which teacher to distill from based on compatibility scores
- **Debugging**: When OPD underperforms expectations, diagnose whether it's the loss or the pairing

## When NOT to Use

- As a standalone training method (this is diagnostics, not a loss function)
- When you're certain teacher-student compatibility is high (same family, similar size)

## Files

```
src/
├── __init__.py
└── core_algos.py   # compute_shared_token_set, compute_thinking_pattern_similarity,
                    # monitor_progressive_alignment, predict_opd_success
```

## Usage

```bash
# No standalone training test — this is a diagnostic module.
# Use within other training loops:

from rethinking_opd.src.core_algos import (
    compute_shared_token_set,
    compute_thinking_pattern_similarity,
    predict_opd_success,
    monitor_progressive_alignment,
)

# Pre-training diagnostic:
overlap, metrics = compute_shared_token_set(teacher_logits, student_logits, mask)
pattern_metrics = compute_thinking_pattern_similarity(teacher_lp, student_lp, mask)
all_metrics = {**metrics, **pattern_metrics}
success, explanation = predict_opd_success(all_metrics)
print(f"OPD likely to succeed: {success}")
print(f"Reason: {explanation}")

# During training:
alignment_metrics = monitor_progressive_alignment(student_lp, teacher_lp, mask, step=i)
if alignment_metrics.get("alignment/improving") == 0.0:
    print("WARNING: alignment stalling, consider switching teacher or method")
```

## Key Metrics

| Metric | Good Range | Meaning |
|--------|-----------|---------|
| `compat/top_k_overlap` | > 0.3 | Fraction of teacher's top tokens that student also ranks highly |
| `compat/logprob_correlation` | > 0.3 | Pearson correlation of log-prob profiles |
| `alignment/kl_delta` | < 0 | KL decreasing = alignment progressing |
| `alignment/improving` | 1.0 | Binary: is alignment getting better? |

## Key Findings from Paper

1. **97-99% mass concentration**: Success requires teacher and student to share a tiny set of high-probability tokens. If distributions are disjoint, no loss will bridge the gap.

2. **Thinking pattern compatibility**: Even if token overlap exists, the temporal pattern of reasoning must be compatible. A chain-of-thought teacher paired with a direct-answer student may fail.

3. **Practical implication**: Before running expensive OPD, do a cheap diagnostic pass on a small batch.

## Design Principle

```
The prerequisite for OPD is NOT just a good loss function.
It's that teacher and student operate in overlapping regions of token space.

If they don't overlap, no λ, no α, no KL direction change will save you.
First verify compatibility. Then optimize the objective.
```

## Relationship to Other Methods

- **Prerequisites check for ALL other methods**: Run diagnostics before choosing G-OPD, Veto, etc.
- **Explains failures**: If AOPD or Entropy-Aware OPD underperforms, check compatibility first
- **Guides method selection**: Low overlap → use Veto (bridge). High overlap → use G-OPD (extrapolate)
