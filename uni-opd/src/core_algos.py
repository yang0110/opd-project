"""
Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe.

Core insight: Token-level teacher guidance must maintain order consistency
with sequence-level outcome reward. If aggregated token-level KL signal
contradicts outcome reward ranking, dense supervision can mislead student.

Key contributions:
1. Dual perspective: token-level guidance + outcome-level calibration
2. Outcome-guided margin calibration: adjust token-level margin using
   global outcome reward to maintain signal consistency
3. Coverage: 5 domains, 16 benchmarks, LLM/MLLM, single/multi-teacher,
   strong-to-weak, cross-modal settings

Fundamental tension in OPD:
    Token-level supervision is dense but locally defined.
    Task objective is sequence-level (correct answer, passing test).
    Good OPD must calibrate between these two levels.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_outcome_reward(
    responses: list[str],
    ground_truths: list[str],
    reward_fn: callable,
) -> torch.Tensor:
    """
    Compute sequence-level outcome reward.

    r(x, y) = 1 if answer correct, 0 otherwise (for math/code)
    or continuous score from reward model.

    Args:
        responses: list of generated response strings
        ground_truths: list of ground truth answers
        reward_fn: function mapping (response, ground_truth) → float

    Returns:
        outcome_rewards: [batch] sequence-level rewards
    """
    rewards = []
    for resp, gt in zip(responses, ground_truths):
        rewards.append(reward_fn(resp, gt))
    return torch.tensor(rewards)


def compute_token_guidance_score(
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute aggregated token-level guidance score per sequence.

    Token guidance = Σ_t (log π* - log π_ref)
    This is the cumulative dense signal the student receives.

    If this disagrees with outcome reward ordering, calibration is needed.
    """
    token_scores = (teacher_log_probs - ref_log_probs) * mask
    seq_scores = token_scores.sum(dim=-1)  # [batch]
    return seq_scores


def check_order_consistency(
    token_guidance_scores: torch.Tensor,
    outcome_rewards: torch.Tensor,
    num_generations: int = 1,
) -> tuple[float, torch.Tensor]:
    """
    Check order consistency between token-level guidance and outcome reward.

    For each pair of responses to the same prompt:
    - If outcome_reward(y1) > outcome_reward(y2),
      then token_guidance(y1) should also > token_guidance(y2).

    Returns fraction of consistent pairs and per-pair consistency mask.
    """
    batch_size = token_guidance_scores.shape[0]
    if num_generations <= 1:
        return 1.0, torch.ones(batch_size, device=token_guidance_scores.device)

    # Reshape to [num_prompts, num_generations]
    num_prompts = batch_size // num_generations
    tg = token_guidance_scores.view(num_prompts, num_generations)
    or_ = outcome_rewards.view(num_prompts, num_generations)

    consistent = 0
    total = 0

    for i in range(num_generations):
        for j in range(i + 1, num_generations):
            reward_order = (or_[:, i] > or_[:, j]).float() - (or_[:, j] > or_[:, i]).float()
            token_order = (tg[:, i] > tg[:, j]).float() - (tg[:, j] > tg[:, i]).float()
            agreement = (reward_order * token_order >= 0).float()
            consistent += agreement.sum().item()
            total += agreement.shape[0]

    consistency_rate = consistent / max(total, 1)
    return consistency_rate, torch.tensor(consistency_rate)


def compute_outcome_guided_margin(
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    outcome_rewards: torch.Tensor,
    mask: torch.Tensor,
    num_generations: int = 1,
    margin_scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute outcome-guided margin calibration for token-level guidance.

    Adjusts token-level reward margin so that:
    - Correct trajectories (high outcome reward) get amplified token guidance
    - Incorrect trajectories get dampened or reversed token guidance

    This ensures token-level signal is consistent with outcome-level signal.

    margin_t = token_reward_t * calibration_factor

    calibration_factor:
    - > 1 for correct responses (amplify guidance)
    - < 1 for incorrect responses (dampen guidance)

    Args:
        teacher_log_probs: [batch, seq_len]
        ref_log_probs: [batch, seq_len]
        outcome_rewards: [batch] (e.g., 0 or 1 for binary correctness)
        mask: [batch, seq_len]
        num_generations: generations per prompt (for relative scoring)
        margin_scale: overall scaling of the margin adjustment
    """
    # Base token-level reward
    token_rewards = (teacher_log_probs - ref_log_probs) * mask

    # Compute calibration factor from outcome reward
    if num_generations > 1:
        # Relative: normalize within group
        batch_size = outcome_rewards.shape[0]
        num_prompts = batch_size // num_generations
        grouped_rewards = outcome_rewards.view(num_prompts, num_generations)
        group_mean = grouped_rewards.mean(dim=-1, keepdim=True)
        group_std = grouped_rewards.std(dim=-1, keepdim=True).clamp(min=0.1)
        normalized = (grouped_rewards - group_mean) / group_std
        calibration_factor = (1.0 + margin_scale * normalized).view(batch_size)
    else:
        # Absolute: use reward directly (center around mean)
        mean_reward = outcome_rewards.mean()
        std_reward = outcome_rewards.std().clamp(min=0.1)
        normalized = (outcome_rewards - mean_reward) / std_reward
        calibration_factor = 1.0 + margin_scale * normalized

    # Apply calibration to token rewards
    calibrated_rewards = token_rewards * calibration_factor.unsqueeze(-1)
    return calibrated_rewards


def compute_uni_opd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    outcome_rewards: torch.Tensor,
    num_generations: int = 1,
    margin_scale: float = 1.0,
    kl_coef: float = 0.1,
    loss_agg: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    Uni-OPD loss with outcome-guided margin calibration.

    L = -E[Σ_t calibrated_advantage_t * log π_θ(y_t|h_t)]

    The calibrated advantage uses outcome-guided margins to ensure
    token-level signal is consistent with sequence-level outcome.

    Args:
        student_log_probs: [batch, seq_len]
        teacher_log_probs: [batch, seq_len]
        ref_log_probs: [batch, seq_len]
        mask: [batch, seq_len]
        outcome_rewards: [batch] sequence-level outcome reward
        num_generations: generations per prompt for relative scoring
        margin_scale: strength of outcome calibration
        kl_coef: KL regularization coefficient
    """
    # Outcome-calibrated token rewards
    calibrated_rewards = compute_outcome_guided_margin(
        teacher_log_probs, ref_log_probs, outcome_rewards,
        mask, num_generations, margin_scale,
    )

    # Advantage: calibrated reward - KL cost
    kl_cost = (student_log_probs.detach() - ref_log_probs) * mask
    advantages = calibrated_rewards - kl_coef * kl_cost

    # Policy gradient
    token_losses = -advantages * student_log_probs
    token_losses = token_losses * mask

    if loss_agg == "token-mean":
        loss = token_losses.sum() / mask.sum().clamp(min=1)
    else:
        loss = token_losses.sum() / mask.sum().clamp(min=1)

    # Metrics
    with torch.no_grad():
        # Order consistency check
        token_guidance = compute_token_guidance_score(teacher_log_probs, ref_log_probs, mask)
        consistency, _ = check_order_consistency(token_guidance, outcome_rewards, num_generations)

        # Outcome reward stats
        mean_reward = outcome_rewards.mean()
        mean_calibrated = (calibrated_rewards * mask).sum() / mask.sum().clamp(min=1)

        # Per correct/incorrect
        correct_mask = (outcome_rewards > 0.5).float()
        incorrect_mask = 1.0 - correct_mask
        n_correct = correct_mask.sum().clamp(min=1)
        n_incorrect = incorrect_mask.sum().clamp(min=1)

    metrics = {
        "uni_opd/loss": loss.item(),
        "uni_opd/order_consistency": consistency,
        "uni_opd/mean_outcome_reward": mean_reward.item(),
        "uni_opd/mean_calibrated_reward": mean_calibrated.item(),
        "uni_opd/frac_correct": (n_correct / (n_correct + n_incorrect)).item(),
        "uni_opd/margin_scale": margin_scale,
    }

    return loss, metrics
