# Standard On-Policy Distillation (OPD)

## What Problem It Solves

Traditional knowledge distillation (KD) trains the student on **teacher-generated** trajectories. This creates a train-test mismatch: at inference time the student generates its own tokens, visiting states it never saw during training. Standard OPD fixes this by having the student generate its own trajectories and then receiving teacher supervision on those student-visited states.

## How It Works

The student generates responses on-policy, then minimizes the reverse KL divergence between itself and the teacher on those same trajectories:

```
J_OPD(θ) = min E_{y~π_θ} [D_KL(π_θ(y|x) || π*(y|x))]
```

This is equivalent to a token-level policy gradient where the advantage at each token is:

```
A_t = log π*(y_t|h_t) - log π_θ(y_t|h_t)
```

The key insight (shown in G-OPD paper) is that this can be reinterpreted as dense KL-constrained RL, where the implicit token-level reward is:

```
r_t = log π*(y_t|h_t) - log π_ref(y_t|h_t)
```

## When to Use

- **Default choice** for distilling a larger/stronger teacher into a smaller student
- When you have a teacher model that is an RL-finetuned variant of the student's base
- When off-policy KD (SFT on teacher outputs) hits a ceiling due to distribution mismatch
- For merging domain-specific RL experts back into one unified model

## When NOT to Use

- If teacher and student have incompatible thinking patterns (see `rethinking-opd/`)
- If you need the student to **exceed** the teacher (use G-OPD with λ>1 instead)
- If the teacher-student gap is very large and causes unstable gradients (use Veto instead)
- If structured output format compliance is critical (add Extrapolation Cliff safeguards)

## Files

```
src/
├── core_algos.py        # Loss functions: reverse KL, forward KL, token-level PG
├── opd_trainer.py       # OPDTrainer class with configurable loss types
├── ray_opd_trainer.py   # Distributed trainer (Ray + verl architecture)
├── workers.py           # Worker definitions (rollout, teacher, reference)
└── verl_integration.py  # Register OPD as verl algorithm (advantage + policy loss)
```

## Usage

```bash
# Smoke test (no GPU needed):
cd /path/to/opd-project
python3 run_individual_tests.py --method standard_opd

# Full training with verl:
cd standard-opd
bash scripts/run_opd.sh configs/opd_qwen3_4b.yaml
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `loss_type` | `token_pg` | Options: `token_pg`, `reverse_kl`, `forward_kl` |
| `kl_coef` | 0.0 | KL penalty against reference (0 = pure OPD) |
| `temperature` | 1.0 | Softmax temperature for distribution matching |
| `loss_agg` | `token-mean` | Aggregation: `token-mean` or `seq-mean-token-sum` |
| `num_opd_steps` | 50 | Training iterations |
| `generation_temperature` | 1.0 | Student sampling temperature |

## Training Loop (verl integration)

```
for step in range(num_opd_steps):
    prompts = sample_batch()
    trajectories = student.generate(prompts)           # on-policy rollout
    teacher_lp = teacher.log_probs(trajectories)       # dense supervision
    ref_lp = reference.log_probs(trajectories)         # KL baseline
    loss = OPD_loss(student_lp, teacher_lp, ref_lp)    # token-level PG
    student.update(loss)                                # gradient step
    sync_weights_to_rollout_engine()                    # update vLLM/SGLang
```

## Relationship to Other Methods

Standard OPD is the **λ=1** special case of G-OPD. All other methods in this project modify one or more aspects of this baseline:

| Module | Standard OPD | What others change |
|--------|-------------|-------------------|
| Reward scale | λ=1 fixed | G-OPD: variable λ |
| Target | Raw teacher logits | Veto: bridge target |
| KL direction | Reverse KL everywhere | Entropy-Aware: adaptive |
| Update rule | Uniform across tokens | AOPD: split by advantage sign |
| Safety | No boundary | Extrapolation Cliff: threshold |
