"""
G-OPD / ExOPD core algorithm implementation.

G-OPD objective (Eq. 11):
    J_G-OPD(θ) = max_θ E[λ * log(π*(y|x) / π_ref(y|x)) - D_KL(π_θ(y|x) || π_ref(y|x))]

G-OPD gradient (Eq. 14):
    ∇J = E[Σ_t A_t^G-OPD * ∇log π_θ(y_t|h_t)]

where:
    A_t^G-OPD = (log π_θ(y_t|h_t) - log π*(y_t|h_t)) + (λ-1)(log π_ref(y_t|h_t) - log π*(y_t|h_t))

Optimal solution (Eq. 12):
    log π_θ*(y|x) = λ * log π*(y|x) + (1-λ) * log π_ref(y|x)

Key settings:
    λ = 1: Standard OPD
    0 < λ < 1: Reward interpolation (student between ref and teacher)
    λ > 1: Reward extrapolation (ExOPD, student can surpass teacher)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_gopd_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    lambda_: float = 1.25,
) -> torch.Tensor:
    """
    Compute G-OPD token-level advantage.

    A_t^G-OPD = (log π_θ(y_t|h_t) - log π*(y_t|h_t))
              + (λ-1) * (log π_ref(y_t|h_t) - log π*(y_t|h_t))

    Simplified (from paper Eq. 14):
    A_t^G-OPD = log π_θ - log π* + (λ-1)(log π_ref - log π*)
              = log π_θ - λ*log π* + (λ-1)*log π_ref - log π_ref + log π_ref
              = log π_θ - [λ*log π* + (1-λ)*log π_ref]

    So the advantage is the gap between student and the G-OPD target distribution.

    Args:
        student_log_probs: [batch, seq_len]
        teacher_log_probs: [batch, seq_len]
        ref_log_probs: [batch, seq_len]
        mask: [batch, seq_len]
        lambda_: reward scaling factor

    Returns:
        advantages: [batch, seq_len] (negated for minimization)
    """
    # G-OPD target: λ*log π* + (1-λ)*log π_ref
    target_log_probs = lambda_ * teacher_log_probs + (1 - lambda_) * ref_log_probs

    # Advantage: target - student (positive = student should increase prob)
    advantages = target_log_probs - student_log_probs

    return advantages * mask


def compute_gopd_token_reward(
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    lambda_: float = 1.25,
) -> torch.Tensor:
    """
    Compute scaled token-level reward for G-OPD.

    r_t^G-OPD = λ * log(π*(y_t|h_t) / π_ref(y_t|h_t))
              = λ * (log π* - log π_ref)

    When λ > 1 (ExOPD), the reward is amplified beyond standard OPD.
    """
    token_rewards = lambda_ * (teacher_log_probs - ref_log_probs)
    return token_rewards * mask


def compute_gopd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    lambda_: float = 1.25,
    loss_agg: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    Compute G-OPD policy gradient loss.

    Loss = -E[Σ_t A_t^G-OPD * log π_θ(y_t|h_t)]

    where A_t^G-OPD uses the scaled reward and KL terms.

    Args:
        student_log_probs: [batch, seq_len] (with grad)
        teacher_log_probs: [batch, seq_len] (detached)
        ref_log_probs: [batch, seq_len] (detached)
        mask: [batch, seq_len]
        lambda_: reward scaling factor
        loss_agg: aggregation mode
    """
    advantages = compute_gopd_advantage(
        student_log_probs.detach(),
        teacher_log_probs,
        ref_log_probs,
        mask,
        lambda_=lambda_,
    )

    # Policy gradient
    token_losses = -advantages * student_log_probs
    token_losses = token_losses * mask

    if loss_agg == "token-mean":
        loss = token_losses.sum() / mask.sum().clamp(min=1)
    elif loss_agg == "seq-mean-token-sum":
        seq_losses = token_losses.sum(dim=-1)
        seq_counts = mask.sum(dim=-1).clamp(min=1)
        loss = (seq_losses / seq_counts).mean()
    else:
        loss = token_losses.sum() / mask.sum().clamp(min=1)

    # Metrics
    with torch.no_grad():
        target_lp = lambda_ * teacher_log_probs + (1 - lambda_) * ref_log_probs
        kl_to_target = (student_log_probs.detach() - target_lp) * mask
        mean_kl = kl_to_target.sum() / mask.sum().clamp(min=1)

        rewards = compute_gopd_token_reward(teacher_log_probs, ref_log_probs, mask, lambda_)
        mean_reward = rewards.sum() / mask.sum().clamp(min=1)

        # Track extrapolation magnitude
        extrap = (lambda_ - 1) * (teacher_log_probs - ref_log_probs) * mask
        mean_extrap = extrap.sum() / mask.sum().clamp(min=1)

    metrics = {
        "gopd/loss": loss.item(),
        "gopd/lambda": lambda_,
        "gopd/mean_kl_to_target": mean_kl.item(),
        "gopd/mean_token_reward": mean_reward.item(),
        "gopd/mean_extrapolation": mean_extrap.item(),
        "gopd/mean_advantage": (advantages * mask).sum().item() / mask.sum().clamp(min=1).item(),
    }

    return loss, metrics


def compute_gopd_loss_with_reward_correction(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    teacher_base_log_probs: torch.Tensor,
    mask: torch.Tensor,
    lambda_: float = 1.25,
    loss_agg: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    G-OPD with reward correction for strong-to-weak distillation.

    In strong-to-weak setting, using teacher's pre-RL base as reference
    gives a cleaner reward signal:

    r_t = log π*(y_t|h_t) - log π_teacher_base(y_t|h_t)

    This captures the teacher's RL improvement direction more accurately
    than using student base as reference.

    Args:
        student_log_probs: current student [batch, seq_len]
        teacher_log_probs: teacher (post-RL) [batch, seq_len]
        ref_log_probs: reference for KL (student base) [batch, seq_len]
        teacher_base_log_probs: teacher's pre-RL base [batch, seq_len]
        mask: [batch, seq_len]
        lambda_: reward scaling factor
    """
    # Corrected reward uses teacher_base as the denominator
    corrected_reward = lambda_ * (teacher_log_probs - teacher_base_log_probs)

    # KL regularization still uses student's ref
    # Loss = -(corrected_reward - KL(student || ref))
    # Advantage: corrected_reward_t - (log π_θ - log π_ref)
    advantages = corrected_reward - (student_log_probs.detach() - ref_log_probs)
    advantages = advantages * mask

    token_losses = -advantages * student_log_probs
    token_losses = token_losses * mask

    if loss_agg == "token-mean":
        loss = token_losses.sum() / mask.sum().clamp(min=1)
    else:
        loss = token_losses.sum() / mask.sum().clamp(min=1)

    with torch.no_grad():
        mean_corrected_reward = (corrected_reward * mask).sum() / mask.sum().clamp(min=1)
        uncorrected_reward = lambda_ * (teacher_log_probs - ref_log_probs) * mask
        mean_uncorrected = uncorrected_reward.sum() / mask.sum().clamp(min=1)

    metrics = {
        "gopd/loss": loss.item(),
        "gopd/lambda": lambda_,
        "gopd/mean_corrected_reward": mean_corrected_reward.item(),
        "gopd/mean_uncorrected_reward": mean_uncorrected.item(),
        "gopd/reward_correction_delta": (mean_corrected_reward - mean_uncorrected).item(),
    }

    return loss, metrics
