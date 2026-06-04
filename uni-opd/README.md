# Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe

**Paper:** [arXiv:2605.03677](https://arxiv.org/abs/2605.03677) — "Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe"

## What Problem It Solves

OPD gives dense token-level supervision, but the real task objective is sequence-level (correct answer, passing test). If the accumulated token-level signal **disagrees** with the outcome reward ranking, dense supervision can actively mislead the student.

Example: Token-level KL says trajectory A is closer to teacher than B. But outcome reward says B gets the right answer and A doesn't. The token-level signal is sending the student the wrong way.

Uni-OPD ensures token-level guidance maintains **order consistency** with outcome reward.

## How It Works

Dual-perspective recipe:

1. **Token perspective**: standard dense teacher guidance (log π* - log π_ref)
2. **Outcome perspective**: sequence-level outcome reward (correct/incorrect)

**Outcome-guided margin calibration**:
```
calibrated_reward_t = token_reward_t * calibration_factor

calibration_factor:
  - > 1 for correct trajectories (amplify token signal)
  - < 1 for incorrect trajectories (dampen token signal)
```

This ensures that when a trajectory gets the right answer, its token-level guidance is amplified. When it gets the wrong answer, guidance is dampened — even if the token-level KL says it's "close" to teacher.

## When to Use

- **Math reasoning** with binary outcome verification (correct/incorrect final answer)
- **Code generation** with unit test pass/fail
- **Any task with verifiable outcomes** where you can compute sequence-level reward
- **Multi-generation setting** (n>1 per prompt): needed for relative ranking across generations
- **When you observe OPD training diverges from eval performance** (token loss decreases but accuracy doesn't improve)

## When NOT to Use

- Open-ended tasks without verifiable outcomes (creative writing, summarization)
- When you can't compute outcome reward during training
- Single-generation per prompt (n=1) — no relative ranking possible within prompt groups

## Key Results

Covers 5 domains, 16 benchmarks:
- LLM and MLLM (multi-modal)
- Single-teacher and multi-teacher
- Strong-to-weak and cross-modal distillation

## Files

```
src/
├── __init__.py
└── core_algos.py   # compute_uni_opd_loss, compute_outcome_guided_margin,
                    # check_order_consistency, compute_token_guidance_score
```

## Usage

```bash
# Smoke test:
python3 run_individual_tests.py --method uni_opd

# Key function:
# compute_uni_opd_loss(student_lp, teacher_lp, ref_lp, mask,
#     outcome_rewards,    # [batch] 0/1 or continuous
#     num_generations=1, margin_scale=1.0, kl_coef=0.1)
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `margin_scale` | 1.0 | Strength of outcome calibration |
| `num_generations` | 1 | Generations per prompt for relative scoring |
| `kl_coef` | 0.1 | KL regularization coefficient |

## Monitored Metrics

- `uni_opd/order_consistency`: fraction of pairs where token guidance agrees with outcome reward ordering
- `uni_opd/mean_outcome_reward`: average outcome reward (task accuracy proxy)
- `uni_opd/mean_calibrated_reward`: average calibrated token reward
- `uni_opd/frac_correct`: fraction of correct trajectories

## Design Principle

```
Fundamental tension in OPD:
  Token-level supervision is dense but locally defined.
  Task objective is sequence-level.
  
Good OPD must maintain CONSISTENCY between these two levels.
If token signal contradicts outcome, the student gets confused.
```

## Relationship to Other Methods

- **vs. Standard OPD**: OPD trusts token-level signal blindly; Uni-OPD calibrates with outcome
- **vs. G-OPD**: G-OPD scales reward uniformly (λ); Uni-OPD scales per-trajectory based on correctness
- **Complementary with AOPD**: AOPD splits by advantage sign per-token; Uni-OPD weights per-trajectory by outcome
