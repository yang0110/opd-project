# AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation

**Paper:** [arXiv:2503.02832](https://arxiv.org/abs/2503.02832) — "AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation"

## What Problem It Solves

RLHF/DPO produces aligned models, but the alignment is encoded implicitly in model weights. AlignDistil shows that the alignment reward can be **explicitly extracted** and rewritten as a token-level distillation target. This bridges the gap between alignment (reward optimization) and distillation (teacher matching).

Key insight: if you have a DPO-aligned model and its reference base, you can construct a target distribution that captures exactly what DPO learned — then distill that into any student.

## How It Works

The alignment target distribution is derived from DPO model and reference:

```
log π_target(v|h_t) = α * log π_DPO(v|h_t) + (1-α) * log π_ref(v|h_t)
```

Default α = 1/β (DPO temperature), but AlignDistil makes it **token-adaptive**:
- Under-optimized tokens (student far from target): increase α slightly
- Over-optimized tokens (student overshoots): decrease α to prevent reward hacking

This is the inverse of G-OPD's direction:
- G-OPD: starts from OPD, derives that it's dense RL
- AlignDistil: starts from RLHF/DPO reward, derives that it's token-level distillation

## When to Use

- **Distilling alignment** from a DPO/RLHF-tuned model into a smaller student
- **When you have both** the aligned model AND its pre-alignment base model
- **Avoiding reward hacking** that pure RL-based alignment causes at high training steps
- **Token-level control** over alignment transfer (some positions need more, others less)
- **Multi-stage training**: first align a large model with DPO, then distill alignment into small model

## When NOT to Use

- When you don't have the reference/base model that the DPO model was trained from
- When you only have teacher log-probs (not full logits) — need vocabulary-level distributions
- When the alignment was done with methods other than DPO/RLHF that don't have a clean β interpretation

## Files

```
src/
├── __init__.py
└── core_algos.py   # compute_aligndistil_loss, compute_aligndistil_target,
                    # compute_token_adaptive_alpha, compute_aligndistil_token_level_loss
```

## Usage

```bash
# Smoke test:
python3 run_individual_tests.py --method aligndistil

# Full logits version (recommended):
# compute_aligndistil_loss(student_logits, dpo_logits, ref_logits, mask,
#     beta=0.1, adaptive=True)

# Token-level version (when only log-probs available):
# compute_aligndistil_token_level_loss(student_lp, dpo_lp, ref_lp, mask, beta=0.1)
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `beta` | 0.1 | DPO temperature (determines base α = 1/β) |
| `adaptive` | True | Enable token-adaptive extrapolation |
| `kl_direction` | `forward` | Forward KL from target to student |
| `under_opt_threshold` | 0.1 | KL threshold for detecting under-optimization |
| `over_opt_threshold` | 5.0 | KL threshold for detecting over-optimization |

## Conceptual Connection to G-OPD

```
G-OPD path:    OPD → "it's actually dense RL" → reward scaling (λ)
AlignDistil:   RLHF/DPO → "it's actually distillation" → adaptive α

Both show: distillation and reward optimization are two views of the same thing.
The boundary between them is dissolving.
```

## Design Principle

```
Alignment reward is not uniform across tokens:
- Some tokens carry heavy alignment signal (safety-critical, preference-loaded)
- Others are neutral (formatting, connectives)

Token-adaptive α prevents over-optimizing neutral tokens while
ensuring alignment-critical tokens get sufficient signal.
```
