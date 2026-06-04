"""
Coverage Improvement and Fast Convergence of On-Policy Preference Learning.

Theoretical foundation for why on-policy methods outperform offline:

Core principle - Coverage Improvement:
    If batch size is sufficient, each on-policy update moves the policy
    to a region with better coverage of the target, making subsequent
    data increasingly informative → exponential convergence.

Key result:
    Under contextual bandit + Bradley-Terry + linear softmax policy:
    - On-policy DPO converges exponentially when generalized coverage threshold met
    - Offline learner has slower minimax rate

Implication for OPD:
    OPD is inherently an on-policy system. After each update, the student
    visits new states, and teacher supervision on those states becomes
    more informative. This dynamic compounds across steps.

This module implements:
1. Coverage metrics for monitoring OPD dynamics
2. On-policy DPO variant compatible with OPD framework
3. Batch size sufficiency estimation
"""

import torch
import torch.nn.functional as F
from typing import Optional
import math


def compute_coverage_score(
    student_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute coverage score measuring how well student's distribution
    covers the target distribution's support.

    Higher coverage → student visits more states where target has mass
    → more informative supervision in next round.

    C(π_θ, π*) = E_{y~π*}[π_θ(y) / π*(y)]
               ≈ exp(-KL(π* || π_θ))  (forward KL as proxy)

    Args:
        student_log_probs: [batch, seq_len]
        target_log_probs: [batch, seq_len]
        mask: [batch, seq_len]

    Returns:
        coverage: [batch] per-sequence coverage score
    """
    # Forward KL approximation at token level
    fwd_kl_tokens = (target_log_probs - student_log_probs) * mask
    seq_fwd_kl = fwd_kl_tokens.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    # Coverage ∝ exp(-forward_KL)
    coverage = torch.exp(-seq_fwd_kl.clamp(max=10))
    return coverage


def estimate_coverage_improvement(
    coverage_history: list[float],
    window: int = 5,
) -> dict:
    """
    Track coverage improvement rate across OPD steps.

    Healthy OPD: coverage improves monotonically (coverage improvement principle).
    Stalling OPD: coverage plateaus → student stuck, teacher signal less useful.

    Args:
        coverage_history: list of mean coverage scores per step
        window: smoothing window

    Returns:
        dict with improvement rate, trend, and diagnosis
    """
    if len(coverage_history) < 2:
        return {"improvement_rate": 0.0, "trend": "insufficient_data"}

    recent = coverage_history[-window:]
    if len(recent) < 2:
        return {"improvement_rate": 0.0, "trend": "insufficient_data"}

    # Linear regression on recent coverage
    x = torch.arange(len(recent), dtype=torch.float32)
    y = torch.tensor(recent, dtype=torch.float32)
    x_mean = x.mean()
    y_mean = y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum().clamp(min=1e-8)

    improvement_rate = slope.item()

    if improvement_rate > 0.01:
        trend = "improving"
    elif improvement_rate > -0.01:
        trend = "plateau"
    else:
        trend = "degrading"

    return {
        "improvement_rate": improvement_rate,
        "trend": trend,
        "current_coverage": recent[-1],
        "initial_coverage": coverage_history[0] if coverage_history else 0,
    }


def estimate_sufficient_batch_size(
    state_space_estimate: int,
    coverage_target: float = 0.9,
    confidence: float = 0.95,
) -> int:
    """
    Estimate sufficient batch size for coverage improvement guarantee.

    From the theory: batch size B must satisfy B ≥ f(|S|, δ, ε)
    for coverage improvement principle to hold.

    Rough heuristic: B ≈ |S| * log(1/δ) / ε²
    where S is effective state space, δ is failure prob, ε is coverage gap.

    For LLM context: effective state space ≈ prompt diversity × response diversity.
    """
    delta = 1 - confidence
    epsilon = 1 - coverage_target
    sufficient_B = int(state_space_estimate * math.log(1 / delta) / max(epsilon ** 2, 0.01))
    return max(sufficient_B, 16)


def compute_online_dpo_loss(
    student_log_probs_chosen: torch.Tensor,
    student_log_probs_rejected: torch.Tensor,
    ref_log_probs_chosen: torch.Tensor,
    ref_log_probs_rejected: torch.Tensor,
    mask_chosen: torch.Tensor,
    mask_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    On-policy DPO loss for preference learning.

    Unlike standard DPO which uses offline chosen/rejected pairs,
    on-policy DPO generates pairs from current policy:
    - Chosen: student responses with high outcome reward
    - Rejected: student responses with low outcome reward

    L_DPO = -log σ(β * (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))

    On-policy variant converges exponentially (coverage improvement).

    Args:
        student_log_probs_chosen: [batch, seq_len] log probs for preferred responses
        student_log_probs_rejected: [batch, seq_len] log probs for rejected
        ref_log_probs_chosen: [batch, seq_len] reference log probs for preferred
        ref_log_probs_rejected: [batch, seq_len] reference log probs for rejected
        mask_chosen: [batch, seq_len]
        mask_rejected: [batch, seq_len]
        beta: DPO temperature
    """
    # Sequence-level log ratios
    chosen_ratio = ((student_log_probs_chosen - ref_log_probs_chosen) * mask_chosen).sum(dim=-1)
    rejected_ratio = ((student_log_probs_rejected - ref_log_probs_rejected) * mask_rejected).sum(dim=-1)

    # DPO loss
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -F.logsigmoid(logits).mean()

    # Metrics
    with torch.no_grad():
        reward_margin = (chosen_ratio - rejected_ratio).mean()
        accuracy = (logits > 0).float().mean()

    metrics = {
        "online_dpo/loss": loss.item(),
        "online_dpo/reward_margin": reward_margin.item(),
        "online_dpo/accuracy": accuracy.item(),
        "online_dpo/beta": beta,
    }

    return loss, metrics


def compute_opd_with_coverage_aware_sampling(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    coverage_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """
    OPD loss with coverage-aware sample weighting.

    Weight samples by their informativeness (inverse coverage):
    - Low coverage samples → more informative → higher weight
    - High coverage samples → less novel information → lower weight

    This implements the coverage improvement principle by focusing
    learning on the frontier of student's capabilities.

    Args:
        student_log_probs: [batch, seq_len]
        teacher_log_probs: [batch, seq_len]
        ref_log_probs: [batch, seq_len]
        mask: [batch, seq_len]
        coverage_weights: [batch] pre-computed per-sample weights
    """
    if coverage_weights is None:
        coverage = compute_coverage_score(student_log_probs.detach(), teacher_log_probs, mask)
        # Inverse coverage weighting: low coverage → high weight
        coverage_weights = (1.0 / coverage.clamp(min=0.1)).detach()
        # Normalize
        coverage_weights = coverage_weights / coverage_weights.mean()

    # Standard OPD advantage
    advantages = (teacher_log_probs - student_log_probs.detach()) * mask

    # Weighted policy gradient
    token_losses = -advantages * student_log_probs * mask
    seq_losses = token_losses.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    weighted_losses = seq_losses * coverage_weights

    loss = weighted_losses.mean()

    metrics = {
        "coverage_opd/loss": loss.item(),
        "coverage_opd/mean_coverage_weight": coverage_weights.mean().item(),
        "coverage_opd/max_coverage_weight": coverage_weights.max().item(),
    }

    return loss, metrics
