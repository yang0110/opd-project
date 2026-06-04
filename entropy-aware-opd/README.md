# Entropy-Aware On-Policy Distillation

**Paper:** [arXiv:2603.07079](https://arxiv.org/abs/2603.07079) — "Entropy-Aware On-Policy Distillation of Language Models"

## What Problem It Solves

Standard OPD uses **reverse KL everywhere**. Reverse KL is mode-seeking — it chases the teacher's highest-probability mode. This works well when the teacher is confident (low entropy), but when the teacher is uncertain (high entropy, multiple valid continuations), reverse KL:

- Forces the student to pick just one of several valid options
- Loses diversity in the student's outputs
- Creates unstable gradients at high-entropy positions

This method adaptively switches KL direction based on teacher confidence at each position.

## How It Works

Per-position KL direction switching:

```
L = Σ_t [(1 - w_t) * D_KL_reverse(π_θ || π*) + w_t * D_KL_forward(π* || π_θ)]
```

where `w_t = σ(sharpness * (H(π*_t) - threshold))`:
- **Low teacher entropy** → w_t ≈ 0 → reverse KL (precise mode matching)
- **High teacher entropy** → w_t ≈ 1 → forward KL (cover all plausible outputs)

## When to Use

- **Math reasoning tasks** where some steps have a unique correct next token (low entropy) and others allow multiple valid approaches (high entropy)
- **Creative or open-ended generation** where teacher assigns significant probability to multiple continuations
- **When Pass@K matters** — forward KL preserves diversity, improving Pass@8 even if Pass@1 is similar
- **When standard OPD produces overly deterministic students** that always take the same path

## When NOT to Use

- When teacher is always confident (all low entropy) — the forward KL branch never activates
- When you need pure accuracy (Pass@1) and don't care about diversity
- Without full teacher logits — entropy computation requires the full vocabulary distribution

## Key Results (from paper)

Pass@8 improvements over baseline OPD on math benchmarks:

| Student Model | Improvement |
|---------------|-------------|
| Qwen3-0.6B-Base | +1.37 |
| Qwen3-1.7B-Base | +2.39 |
| Qwen3-4B-Base | +5.05 |

## Files

```
src/
├── __init__.py
└── core_algos.py   # compute_entropy_aware_opd_loss, compute_teacher_entropy,
                    # compute_entropy_threshold, compute_entropy_aware_mixing_weight
```

## Usage

```bash
# Smoke test:
python3 run_individual_tests.py --method entropy_aware

# Key function:
# compute_entropy_aware_opd_loss(student_logits, teacher_logits, mask,
#     threshold_method="percentile", threshold_percentile=0.5, sharpness=5.0)
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `threshold_method` | `percentile` | How to compute entropy threshold: `percentile`, `fixed`, `adaptive` |
| `threshold_percentile` | 0.5 | Median split: 50% tokens use reverse KL, 50% add forward KL |
| `sharpness` | 5.0 | Sigmoid sharpness for weight transition |
| `forward_kl_weight` | 1.0 | Scaling for forward KL term |

## Design Principle

```
Teacher signal semantics change with entropy:
- Low entropy: "Use THIS token" → mode-seeking (reverse KL) is appropriate
- High entropy: "Any of THESE tokens work" → mode-covering (forward KL) is appropriate

KL direction should follow the signal type, not be globally fixed.
```

## Relationship to Other Methods

- **vs. Standard OPD**: OPD uses reverse KL everywhere; this adapts per-position
- **vs. Veto**: Veto modifies the target distribution; Entropy-Aware modifies the loss objective
- **vs. AOPD**: Both are per-position adaptive, but AOPD splits by advantage sign while this splits by teacher entropy
- **Complementary with CaOPD**: Entropy-Aware handles teacher uncertainty; CaOPD handles student confidence calibration
