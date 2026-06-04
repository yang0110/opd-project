# G-OPD / ExOPD: Generalized On-Policy Distillation with Reward Extrapolation

**Paper:** [arXiv:2602.12125](https://arxiv.org/abs/2602.12125) — "Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation"

## What Problem It Solves

Standard OPD can only produce a student that **matches** the teacher at best. G-OPD enables the student to **surpass** the teacher by extrapolating along the teacher's improvement direction. It also provides a unified framework where the reward weight (λ) and reference model choice are explicit design knobs rather than hidden defaults.

## How It Works

G-OPD introduces a **reward scaling factor** λ and a **flexible reference model** π_ref:

```
J_G-OPD(θ) = max E_{y~π_θ} [λ * log(π*(y|x) / π_ref(y|x)) - D_KL(π_θ || π_ref)]
```

The optimal student satisfies:
```
log π_θ*(y|x) = λ * log π*(y|x) + (1-λ) * log π_ref(y|x)
```

Three regimes:
- **λ = 1**: Standard OPD (student matches teacher)
- **0 < λ < 1**: Reward interpolation (student between reference and teacher)
- **λ > 1**: Reward extrapolation / **ExOPD** (student can exceed teacher)

**Reward correction** (for strong-to-weak): Using teacher's pre-RL base as reference gives a cleaner reward signal:
```
r_t = log π_teacher(y_t) - log π_teacher_base(y_t)
```

## When to Use

- **λ > 1 (ExOPD):** When you want the student to surpass the teacher. Best setting: λ=1.25
- **Multi-teacher merging:** When merging domain experts (math, code) back into one model. ExOPD is the only method that consistently surpasses ALL domain teachers
- **Strong-to-weak distillation:** When distilling 30B → 4B. Add reward correction for best results
- **Budget-controlled reasoning:** Use λ ∈ (0,1) to produce students with shorter responses than teacher but still better than base

## When NOT to Use

- If λ > 1.5 — instability and reward hacking appear (see Extrapolation Cliff)
- On structured output tasks (JSON, function calling) — format compliance collapses at high λ
- Without monitoring response length — ExOPD tends to produce longer outputs

## Key Results (from paper)

| Setting | Standard OPD | ExOPD (λ=1.25) | Gain |
|---------|-------------|----------------|------|
| Same-size single-teacher (math) | 32.4 avg | **33.9 avg** | +1.5 |
| Multi-teacher (math+code) | 38.3 / 60.6 | **39.2 / 62.1** | +0.9/+1.5 |
| Strong-to-weak 30B→1.7B | 23.1 avg | **25.4 avg** | +2.3 |
| Strong-to-weak 30B→4B | 42.6 avg | **45.3 avg** | +2.7 |

## Files

```
src/
├── __init__.py
└── core_algos.py   # compute_gopd_loss, compute_gopd_loss_with_reward_correction
configs/
└── gopd_exopd.yaml
```

## Usage

```bash
# Smoke test:
python3 run_individual_tests.py --method gopd_extrap
python3 run_individual_tests.py --method gopd_interp
python3 run_individual_tests.py --method gopd_correction

# Full training:
python -m verl.trainer.main_ppo \
    --config-path configs --config-name gopd_exopd \
    algorithm.lambda=1.25
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lambda_` | 1.25 | Reward scaling. Paper sweeps: 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5 |
| `ref_model` | student_base | For strong-to-weak: use teacher_base for reward correction |
| `loss_agg` | `token-mean` | Aggregation mode |

## Gradient Formula

```
A_t^G-OPD = (log π_θ(y_t|h_t) - log π*(y_t|h_t)) + (λ-1)(log π_ref(y_t|h_t) - log π*(y_t|h_t))
∇J = E[Σ_t A_t^G-OPD * ∇log π_θ(y_t|h_t)]
```
