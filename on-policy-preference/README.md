# On-Policy Preference Learning: Coverage Improvement and Fast Convergence

**Paper:** [arXiv:2601.08421](https://arxiv.org/abs/2601.08421) — "Coverage Improvement and Fast Convergence of On-policy Preference Learning"

## What Problem It Solves

This paper provides the **theoretical foundation** for why on-policy methods (including OPD) outperform offline methods. It proves that on-policy updates create a virtuous cycle:

```
Better policy → visits better states → more informative data → even better policy → ...
```

This is the **coverage improvement principle**: each update moves the policy to a region where the target has better coverage, making subsequent supervision increasingly informative. Under appropriate conditions, this leads to **exponential convergence** rather than the slower rates of offline learning.

## How It Works

### Theory

Under contextual bandit + Bradley-Terry preferences + linear softmax policy:
- **On-policy DPO** converges exponentially when a generalized coverage threshold is met
- **Offline learner** (fixed dataset) has strictly slower minimax convergence rate
- The gap comes from coverage improvement: on-policy data becomes more informative over time

### Practical Implementation

This module provides:

1. **Coverage metrics**: measure how well student covers teacher's support
2. **Coverage-aware sampling**: weight samples by informativeness (inverse coverage)
3. **On-policy DPO**: DPO variant using student-generated preference pairs
4. **Batch size estimation**: estimate sufficient batch size for coverage improvement guarantee

## When to Use

- **Understanding OPD dynamics**: Monitor coverage improvement across training steps
- **Sample weighting**: Prioritize samples where student has poor coverage (frontier learning)
- **Batch size decisions**: Estimate how large batches need to be for stable improvement
- **On-policy DPO**: When you have outcome reward and want preference learning (not KD)
- **Diagnosing OPD stalls**: If coverage stops improving, OPD will stall

## When NOT to Use

- As a replacement for OPD (this is complementary theory + monitoring)
- When you're doing offline/off-policy training (the theory doesn't apply)
- As the primary training method (combine with other OPD losses)

## Files

```
src/
├── __init__.py
└── core_algos.py   # compute_coverage_score, estimate_coverage_improvement,
                    # compute_online_dpo_loss, compute_opd_with_coverage_aware_sampling,
                    # estimate_sufficient_batch_size
```

## Usage

```bash
# Smoke test:
python3 run_individual_tests.py --method coverage_opd

# Coverage-aware OPD (prioritize low-coverage samples):
# compute_opd_with_coverage_aware_sampling(student_lp, teacher_lp, ref_lp, mask)

# Coverage monitoring:
from on_policy_preference.src.core_algos import (
    compute_coverage_score,
    estimate_coverage_improvement,
)
coverage = compute_coverage_score(student_lp, teacher_lp, mask)
history.append(coverage.mean().item())
improvement = estimate_coverage_improvement(history)
print(f"Coverage trend: {improvement['trend']}")

# On-policy DPO (when you have chosen/rejected pairs):
# compute_online_dpo_loss(student_lp_chosen, student_lp_rejected,
#     ref_lp_chosen, ref_lp_rejected, mask_chosen, mask_rejected, beta=0.1)
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Coverage weighting | auto | Inverse coverage: low coverage → high weight |
| `beta` (DPO) | 0.1 | DPO temperature for on-policy preference learning |
| `window` | 5 | Steps for trend estimation |

## Monitored Metrics

- `coverage_opd/mean_coverage_weight`: average sample weight (higher = more reweighting)
- `coverage_opd/max_coverage_weight`: highest weight (potential outlier)
- Coverage improvement `trend`: "improving", "plateau", or "degrading"
- `improvement_rate`: slope of coverage over recent steps

## Key Theoretical Insights

1. **On-policy is a dynamic system**: OPD reward/divergence design doesn't just affect the current batch — it changes what states the student visits next round, which changes the informativeness of future supervision.

2. **Coverage compounds**: Each round of good coverage improvement makes the next round easier. This is why OPD can converge much faster than offline KD.

3. **Batch size matters**: Below a threshold, coverage improvement principle doesn't hold. The module provides `estimate_sufficient_batch_size()` as a rough guide.

## Design Principle

```
On-policy is NOT just "more fresh data."
It's a COMPOUNDING process where each update makes future data more informative.

Monitor coverage. If it stops improving, OPD will stall regardless of the loss function.
```

## Relationship to Other Methods

- **Theoretical basis for all OPD methods**: Explains WHY on-policy works better
- **Monitoring tool**: Use alongside any other method to track health
- **Sample efficiency**: Coverage-aware weighting can improve any OPD variant
- **Connects to Uni-OPD**: Both care about data informativeness, but Uni-OPD calibrates by outcome while this calibrates by coverage
