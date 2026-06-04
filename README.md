# OPD Research Project

On-Policy Distillation (OPD) objective function design - a unified implementation of 10 recent OPD methods built on the [verl](https://github.com/volcengine/verl) framework.

## Overview

This project implements the latest research on OPD objective functions, covering reward/divergence variants that treat OPD not as a fixed loss but as a modular design space:

| # | Method | Paper | Key Idea |
|---|--------|-------|----------|
| 1 | **Standard OPD** | Baseline | Reverse KL on student trajectories with teacher supervision |
| 2 | **G-OPD / ExOPD** | [arXiv:2602.12125](https://arxiv.org/abs/2602.12125) | Reward scaling λ: interpolation (λ<1) and extrapolation (λ>1) |
| 3 | **Veto** | [arXiv:2601.07155](https://arxiv.org/abs/2601.07155) | Logit-space bridge target when teacher-student gap is large |
| 4 | **Entropy-Aware OPD** | [arXiv:2603.07079](https://arxiv.org/abs/2603.07079) | Adaptive KL direction based on teacher entropy |
| 5 | **AOPD** | [arXiv:2605.06387](https://arxiv.org/abs/2605.06387) | Asymmetric: RL for positive advantage, imitation for non-positive |
| 6 | **Extrapolation Cliff** | [arXiv:2605.08737](https://arxiv.org/abs/2605.08737) | Safety threshold for λ>1 on structured outputs |
| 7 | **CaOPD** | [arXiv:2604.16830](https://arxiv.org/abs/2604.16830) | Calibration-aware: decouple capability from confidence |
| 8 | **Uni-OPD** | [arXiv:2605.03677](https://arxiv.org/abs/2605.03677) | Outcome-guided margin calibration for token guidance |
| 9 | **AlignDistil** | [arXiv:2503.02832](https://arxiv.org/abs/2503.02832) | RLHF reward as token-level distillation target |
| 10 | **Rethinking OPD** | [arXiv:2604.13016](https://arxiv.org/abs/2604.13016) | Compatibility diagnostics and progressive alignment monitoring |
| 11 | **On-Policy Preference** | [arXiv:2601.08421](https://arxiv.org/abs/2601.08421) | Coverage improvement theory + coverage-aware OPD |

## Project Structure

```
opd-project/
├── README.md
├── setup_env.sh              # Environment setup script
├── requirements.txt          # Python dependencies
├── papers/                   # Downloaded PDFs (10 papers)
│
├── standard-opd/             # Base OPD implementation
│   ├── src/
│   │   ├── core_algos.py    # OPD loss functions (reverse KL, forward KL, PG)
│   │   ├── opd_trainer.py   # Trainer class with loss computation
│   │   ├── ray_opd_trainer.py  # Distributed trainer (Ray + verl)
│   │   ├── workers.py       # Worker definitions (rollout, teacher, ref)
│   │   └── verl_integration.py  # verl registry integration
│   ├── configs/
│   └── scripts/
│
├── datasets/                 # Shared data loading
│   └── src/
│       ├── data_loader.py   # Math, code, ranking datasets
│       └── reward_functions.py  # Verification (math, code, JSON)
│
├── g-opd/                    # G-OPD with reward extrapolation
│   ├── src/core_algos.py    # λ-scaled reward, reward correction
│   └── configs/
│
├── veto/                     # Adaptive target reformulation
│   └── src/core_algos.py    # Geometric bridge, adaptive α
│
├── entropy-aware-opd/        # Entropy-adaptive KL direction
│   └── src/core_algos.py    # Per-position KL switching
│
├── aopd/                     # Asymmetric OPD
│   └── src/core_algos.py    # Split exploit/imitate branches
│
├── extrapolation-cliff/      # Safety boundaries for ExOPD
│   └── src/core_algos.py    # Safety threshold, adaptive λ per token
│
├── caopd/                    # Calibration-aware OPD
│   └── src/core_algos.py    # Empirical confidence, ECE, self-distillation
│
├── uni-opd/                  # Outcome-guided calibration
│   └── src/core_algos.py    # Order consistency, margin calibration
│
├── aligndistil/              # Alignment as distillation
│   └── src/core_algos.py    # DPO→distillation target, adaptive α
│
├── rethinking-opd/           # Compatibility diagnostics
│   └── src/core_algos.py    # Shared token set, thinking patterns
│
└── on-policy-preference/     # Coverage improvement theory
    └── src/core_algos.py    # Coverage metrics, on-policy DPO
```

## Setup

### Step 1: Clone and enter project

```bash
cd /path/to/opd-project
```

### Step 2: Run environment setup

```bash
chmod +x setup_env.sh
bash setup_env.sh
```

This installs:
- PyTorch 2.4+ with CUDA 12.1
- verl framework
- vLLM for inference
- Ray for distributed training
- Flash Attention 2
- Evaluation tools (math-verify, lm-eval)

### Step 3: Activate environment

```bash
# If using conda:
conda activate opd

# If using venv:
source .venv/bin/activate
```

### Step 4: Download models

```bash
# Student model
huggingface-cli download Qwen/Qwen3-4B --local-dir models/qwen3-4b

# Teacher model (domain-specific RL variant)
# You need to train this with GRPO first, or use a pre-trained teacher
```

### Step 5: Prepare data

```bash
# Download DeepMath (math reasoning, filter difficulty >= 6)
# Download Eurus-RL-Code (code generation, 25K samples)
# Place in datasets/data/ directory
```

## Running Methods

### Standard OPD

```bash
cd standard-opd
bash scripts/run_opd.sh configs/opd_qwen3_4b.yaml
```

### G-OPD / ExOPD (λ=1.25)

```bash
cd g-opd
python -m verl.trainer.main_ppo \
    --config-path configs \
    --config-name gopd_exopd \
    algorithm.lambda=1.25
```

### Entropy-Aware OPD

```bash
# Requires full teacher logits (not just log_probs)
cd entropy-aware-opd
python -m verl.trainer.main_ppo \
    --config-path configs \
    --config-name entropy_opd \
    algorithm.threshold_percentile=0.5 \
    algorithm.sharpness=5.0
```

### AOPD (Asymmetric)

```bash
cd aopd
python -m verl.trainer.main_ppo \
    --config-path configs \
    --config-name aopd \
    algorithm.exploit_coef=1.0 \
    algorithm.imitate_coef=1.0
```

## Key Concepts

### OPD as Dense KL-Constrained RL

Standard OPD:
```
J_OPD(θ) = max E[log π*(y|x)/π_ref(y|x) - D_KL(π_θ || π_ref)]
```

The implicit token-level reward is:
```
r_t = log π*(y_t|h_t) - log π_ref(y_t|h_t)
```

### G-OPD Generalization

```
J_G-OPD(θ) = max E[λ * log π*(y|x)/π_ref(y|x) - D_KL(π_θ || π_ref)]
```

- λ=1: Standard OPD
- λ<1: Reward interpolation (conservative)
- λ>1: Reward extrapolation (ExOPD, can surpass teacher)

### Design Space

| Module | Default OPD | Variants |
|--------|-------------|----------|
| Reward scale | λ=1 | G-OPD: λ∈(0,∞) |
| Target distribution | Teacher logits | Veto: bridge target |
| KL direction | Reverse KL | Entropy-Aware: adaptive |
| Advantage region | Uniform update | AOPD: split exploit/imitate |
| Extrapolation boundary | Manual | Cliff: derived threshold |
| Confidence target | Teacher-conditioned | CaOPD: empirical |
| Token vs outcome | Token KL only | Uni-OPD: outcome calibration |

## Hardware Requirements

- **Minimum**: 4× A100 80GB (standard OPD with Qwen3-4B)
- **Recommended**: 8× A100 80GB (G-OPD with teacher inference)
- **Strong-to-weak**: 16× A100 80GB (30B teacher → 4B student)

## References

1. Yang et al. "Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation" (2026)
2. "Stable On-Policy Distillation through Adaptive Target Reformulation" (2025)
3. "Entropy-Aware On-Policy Distillation of Language Models" (2025)
4. "Asymmetric On-Policy Distillation" (2025)
5. "The Extrapolation Cliff in On-Policy Distillation of Near-Deterministic Structured Outputs" (2025)
6. "The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation" (2025)
7. "Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe" (2025)
8. "Rethinking On-Policy Distillation of Large Language Models" (2025)
9. "AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation" (2025)
10. "Coverage Improvement and Fast Convergence of On-policy Preference Learning" (2025)
