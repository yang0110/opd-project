# Extrapolation Cliff in OPD of Near-Deterministic Structured Outputs

**Paper:** [arXiv:2605.08737](https://arxiv.org/abs/2605.08737) — "The Extrapolation Cliff in On-Policy Distillation of Near-Deterministic Structured Outputs"

## What Problem It Solves

G-OPD/ExOPD shows that λ>1 improves task performance. But this paper discovers a **sharp failure mode**: on structured output tasks (JSON, rankings, function calls), there exists a critical threshold λ* beyond which the model's **format compliance suddenly collapses**.

The collapse is NOT gradual degradation — it's a cliff:
- Below λ*: format is valid, semantic quality is good
- Above λ*: format breaks completely (invalid JSON, missing fields, broken IDs)

Critically: the semantic quality (e.g., ranking NDCG) stays flat on successfully parsed outputs. The model doesn't forget how to rank — it forgets how to output valid JSON.

## How It Works

The paper derives a **closed-form safety threshold** from three measurable quantities:

```
λ* = f(p_T, m, c)
```

where:
- `p_T` = teacher's probability on the modal (top-1) token at a position
- `m` = student's warm-start mass on that same token
- `c` = importance-sampling clip strength

The implementation provides **position-adaptive λ**:
- Contract-critical tokens (brackets, quotes, field names): use conservative λ ≤ 1
- Semantic tokens (content, reasoning): use full extrapolation λ = base_lambda

## When to Use

- **Any structured output task**: JSON generation, function calling, API output, XML
- **Listwise ranking** tasks where output must parse correctly
- **When combining ExOPD with format-sensitive tasks** — prevents format collapse
- **As a safety layer on top of G-OPD** — automatically clips λ at dangerous positions

## When NOT to Use

- Free-form text generation (math reasoning, creative writing) — no format constraints to violate
- When λ ≤ 1 (interpolation regime has no extrapolation cliff)
- When using standard OPD without extrapolation

## Key Findings

On Amazon Fashion listwise ranking task:
- NDCG@1 on **parsed** outputs: stays flat across all λ values
- Parse validity: **sharp cliff** at predicted λ* boundary
- Collapse is in format tokens, not semantic understanding

## Files

```
src/
├── __init__.py
└── core_algos.py   # compute_safe_exopd_loss, compute_safety_threshold,
                    # identify_contract_critical_tokens, compute_adaptive_lambda
```

## Usage

```bash
# Smoke test:
python3 run_individual_tests.py --method extrapolation_cliff

# Key function:
# compute_safe_exopd_loss(student_lp, teacher_lp, ref_lp, mask,
#     teacher_logits, student_logits,
#     base_lambda=1.25, use_adaptive_lambda=True)
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `base_lambda` | 1.25 | Target extrapolation factor |
| `clip_strength` | 0.2 | IS clip range for threshold computation |
| `use_adaptive_lambda` | True | Enable per-position λ clipping |
| `entropy_threshold` | 0.5 | Entropy below this → contract-critical |
| `top1_prob_threshold` | 0.9 | Top-1 prob above this → contract-critical |

## Monitored Metrics

- `cliff/mean_adaptive_lambda`: average effective λ across positions
- `cliff/frac_critical_tokens`: fraction identified as format-critical
- `cliff/lambda_at_critical`: λ used at critical positions (should be ≤ 1)
- `cliff/lambda_at_semantic`: λ used at semantic positions (can be > 1)

## Design Principle

```
Extrapolation risk is NOT uniform across tokens:
- Format tokens (near-deterministic): single correct option, extrapolation is dangerous
- Semantic tokens (high entropy): multiple valid options, extrapolation is safe

Apply λ > 1 selectively, not blindly.
```
