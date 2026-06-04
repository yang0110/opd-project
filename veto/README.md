# Veto: Stable On-Policy Distillation through Adaptive Target Reformulation

**Paper:** [arXiv:2601.07155](https://arxiv.org/abs/2601.07155) — "Stable On-Policy Distillation through Adaptive Target Reformulation"

## What Problem It Solves

When the teacher-student gap is large (e.g., early training or strong-to-weak distillation), directly matching teacher logits creates **pathological gradients**:

- **Forward KL** forces the student to cover all teacher modes → harmful gradients at positions where teacher has mass the student can't reach
- **Reverse KL** collapses diversity → student only captures one mode of teacher

Veto solves this by constructing an **intermediate bridge target** in logit space that the student can actually reach, then progressively moving toward the full teacher.

## How It Works

Instead of directly optimizing KL(student || teacher), Veto creates a geometric bridge:

```
π_bridge(v|h_t) ∝ π_teacher(v|h_t)^α * π_student(v|h_t)^(1-α)
```

In logit space:
```
bridge_logits = α * teacher_logits + (1-α) * student_logits
```

The mixing weight α is **adaptive per position** based on teacher-student agreement:
- High agreement (low JSD) → α → 1 (trust teacher, push harder)
- Low agreement (high JSD) → α → 0 (stay near student, avoid harmful gradients)

## When to Use

- **Large teacher-student gap**: when the student is much weaker than teacher (e.g., 4B from 70B)
- **Early training instability**: when standard OPD shows loss spikes or divergence in first steps
- **Novice students**: when student hasn't seen the domain before and teacher distribution is far
- **Complementary to G-OPD**: Veto stabilizes, G-OPD extrapolates. Can combine them

## When NOT to Use

- When teacher and student are similar in capability (the bridge adds no value, just overhead)
- When teacher is only slightly better — the adaptive α will just be ~1 everywhere
- When you need extrapolation beyond teacher (Veto only bridges toward teacher, never past it)

## Files

```
src/
├── __init__.py
└── core_algos.py   # compute_veto_loss, compute_veto_loss_token_level,
                    # compute_adaptive_alpha, compute_veto_target_logits
```

## Usage

```bash
# Smoke test:
python3 run_individual_tests.py --method veto
python3 run_individual_tests.py --method veto_token

# With full logits (recommended):
# Uses compute_veto_loss(student_logits, teacher_logits, mask, adaptive=True)

# Token-level only (when teacher logits unavailable):
# Uses compute_veto_loss_token_level(student_lp, teacher_lp, ref_lp, mask)
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `base_alpha` | 0.5 | Base mixing weight (before adaptation) |
| `adaptive` | True | Enable per-position adaptive α |
| `kl_direction` | `reverse` | KL between student and bridge: `reverse` or `forward` |
| `temperature` | 1.0 | Softmax temperature for KL computation |

## How α Adaptation Works

```
1. Compute JSD(teacher_t, student_t) at each position
2. agreement = exp(-temperature * JSD)
3. α_t = base_alpha + (1 - base_alpha) * agreement
4. High agreement → α close to 1 → bridge ≈ teacher (aggressive)
5. Low agreement → α close to base_alpha → bridge near midpoint (conservative)
```

## Relationship to Other Methods

- **vs. Standard OPD**: Veto modifies the target, OPD uses raw teacher
- **vs. G-OPD**: G-OPD controls reward magnitude (λ), Veto controls target proximity (α)
- **Complementary**: Use Veto for stability + G-OPD for extrapolation on different training phases
