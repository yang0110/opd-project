# AOPD: Asymmetric On-Policy Distillation

**Paper:** [arXiv:2605.06387](https://arxiv.org/abs/2605.06387) — "Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation at the Token Level"

## What Problem It Solves

Standard advantage-weighted OPD applies the same update rule to all tokens. This causes three structural problems:

1. **High variance**: tokens with large advantage magnitudes dominate the gradient
2. **Zero-advantage gradient vanishing**: tokens where teacher and reference agree get no learning signal
3. **Exploration bottleneck**: insufficient corrective signals when student makes errors

AOPD recognizes that positive-advantage and non-positive-advantage tokens should serve **different training purposes**.

## How It Works

Split each token into two asymmetric update branches:

```
if A_t > 0:   # Token is better than reference (worth reinforcing)
    L_t = -A_t * log π_θ(y_t)              → RL-style exploitation
else:          # Token is not better (but still informative)
    L_t = D_local(π_θ || π*)_t              → Imitation / distribution matching
```

- **Exploitation branch**: Positive advantage tokens are reinforced via policy gradient (like standard RL). These are tokens where the teacher improves over reference.
- **Imitation branch**: Non-positive advantage tokens are corrected via local divergence matching. Instead of useless negative reinforcement, the student locally aligns with the teacher's full distribution.

## When to Use

- **Weak student initialization**: when starting from a base model far from the teacher (+8.34 avg improvement)
- **Strong student initialization**: even with pre-trained student (+4.09 avg improvement)
- **Sequential tool-use adaptation**: better capability retention when adapting to new tools
- **When student entropy drops too fast** in standard OPD — AOPD maintains higher policy entropy
- **When you observe gradient vanishing** at positions where teacher and reference agree

## When NOT to Use

- When teacher and reference are identical (no meaningful advantage signal exists)
- When you want pure imitation without any RL-style reinforcement
- When computational budget is very tight (AOPD with full logits requires computing local KL)

## Key Results (from paper)

Average improvement over standard OPD on math reasoning:

| Setting | Improvement |
|---------|-------------|
| Strong initialization | +4.09 |
| Weak initialization | +8.34 |

Plus: higher policy entropy and better capability retention in tool-use tasks.

## Files

```
src/
├── __init__.py
└── core_algos.py   # compute_aopd_loss, compute_aopd_loss_with_clipping,
                    # compute_token_advantage
```

## Usage

```bash
# Smoke test:
python3 run_individual_tests.py --method aopd
python3 run_individual_tests.py --method aopd_clipped

# With full logits (best quality):
# compute_aopd_loss(student_lp, teacher_lp, ref_lp, mask,
#     student_logits=s_logits, teacher_logits=t_logits)

# With clipping (PPO-style stability):
# compute_aopd_loss_with_clipping(student_lp, old_student_lp, teacher_lp, ref_lp, mask)
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `exploit_coef` | 1.0 | Weight for exploitation (positive advantage) branch |
| `imitate_coef` | 1.0 | Weight for imitation (non-positive advantage) branch |
| `clip_range` | 0.2 | PPO clip range for clipped variant |
| `temperature` | 1.0 | Temperature for distribution matching in imitation branch |

## Design Principle

```
Not all tokens serve the same learning purpose:
- Positive advantage tokens: "This token is good, reinforce it" → RL
- Non-positive advantage tokens: "This token isn't great, but learn
  what the teacher would do here" → Imitation

Same loss, different semantics per token position.
```

## Monitored Metrics

- `aopd/frac_positive`: fraction of tokens with positive advantage (typically 40-60%)
- `aopd/exploit_loss`: loss from reinforcement branch
- `aopd/imitate_loss`: loss from imitation branch
- `aopd/student_entropy`: student's policy entropy (should stay high)
