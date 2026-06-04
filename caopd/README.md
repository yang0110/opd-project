# CaOPD: Calibration-Aware On-Policy Distillation

**Paper:** [arXiv:2604.16830](https://arxiv.org/abs/2604.16830) — "The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation"

## What Problem It Solves

OPD improves task accuracy but creates a hidden danger: **systematic overconfidence**. The paper calls this the "Scaling Law of Miscalibration" — as OPD improves capability, the model's confidence increasingly overshoots its actual reliability.

Root cause: During training, the student learns from teacher supervision (which may use privileged context). At deployment, the student must express confidence based only on deployment-time information. Teacher-conditioned success ≠ deployment-time reliability.

Example: Teacher helps student solve 90% of problems correctly during training. Student then reports 90% confidence. But without teacher at deployment, student actually solves only 60%. The gap is miscalibration.

## How It Works

Two-phase approach:

**Phase 1: Estimate empirical confidence**
- Generate multiple rollouts per prompt from the student
- Compute fraction that produce correct answers → this is the true deployment confidence

**Phase 2: Calibration-aware loss**
```
L_CaOPD = L_capability + α * L_calibration
```
- `L_capability`: Standard OPD (distill accuracy from teacher)
- `L_calibration`: Align student's expressed confidence with empirical confidence

The calibration loss prevents the student from becoming overconfident as it gets better at the task.

## When to Use

- **Safety-critical applications** where calibrated confidence matters (medical, legal, financial)
- **Selective generation / abstention**: when the system should say "I don't know" appropriately
- **Retrieval-augmented generation**: when confidence scores gate whether to retrieve more context
- **When you observe your OPD student is overconfident** on held-out evaluation
- **Continual learning**: CaOPD is more robust to OOD settings than standard OPD

## When NOT to Use

- When you only care about accuracy and not confidence
- When the task has no notion of "confidence" (e.g., creative writing)
- When you don't have the compute budget for multiple rollouts per prompt (needed to estimate empirical confidence)

## Files

```
src/
├── __init__.py
└── core_algos.py   # compute_caopd_loss, estimate_empirical_confidence,
                    # compute_calibration_target, compute_ece,
                    # caopd_self_distillation_step
```

## Usage

```bash
# Smoke test:
python3 run_individual_tests.py --method caopd

# Key function:
# compute_caopd_loss(student_lp, teacher_lp, ref_lp, mask,
#     empirical_confidence,
#     capability_coef=1.0, calibration_coef=0.5)
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `capability_coef` | 1.0 | Weight for standard OPD loss |
| `calibration_coef` | 0.5 | Weight for calibration loss |
| `num_samples` | 8 | Rollouts per prompt for empirical confidence |
| `calibration_method` | `empirical` | Options: `empirical`, `smoothed`, `temperature_scaled` |

## Monitored Metrics

- `caopd/ece`: Expected Calibration Error (lower is better)
- `caopd/overconfidence_rate`: fraction of samples where student confidence > empirical
- `caopd/mean_confidence_gap`: average (student_confidence - empirical_confidence)
- `caopd/capability_loss`: standard OPD component
- `caopd/calibration_loss`: calibration alignment component

## Design Principle

```
Capability and calibration are DIFFERENT properties:
- Capability: "Can you solve this?" → distill from teacher
- Calibration: "How sure are you?" → ground in student's own deployment behavior

OPD can distill capability. Confidence must reflect the student's own reliability.
```

## Relationship to Other Methods

- **vs. Standard OPD**: OPD distills capability only; CaOPD adds calibration alignment
- **vs. Entropy-Aware OPD**: Entropy-Aware adapts based on teacher uncertainty; CaOPD calibrates student confidence
- **Complementary**: Can combine CaOPD's calibration loss with any other OPD variant
