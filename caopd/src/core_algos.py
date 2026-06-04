"""
CaOPD: Calibration-Aware On-Policy Distillation (Algorithm 1 from paper).

From paper Section 4 + Algorithm 1:

The key insight: Standard OPD entangles capability and confidence because
the teacher operates "open-book" (with privileged context z) while the
student deploys "closed-book" (only input x). This forces the student
to imitate the teacher's unjustified certainty.

CaOPD pipeline (Algorithm 1):
    1. Student-Grounded Confidence Estimation:
       - Sample K rollouts from student: (a_k, c_k) ~ π_θ(·|x)
       - Compute empirical success: μ̂(x) = 1/K * Σ R(x, a_k)

    2. Target Replacement:
       - Sample trajectory y = (a, c) from student
       - Revise completion: ỹ = (a, μ̂(x))  [replace confidence tokens]
       - Revise teacher context: z̃ = z with confidence overwritten by μ̂(x)

    3. Distillation:
       - Standard per-token reverse KL on the REVISED trajectory:
         L_CaOPD = Σ_{t ∈ I_a} D_KL(π_θ(·|ỹ<t, x) || π_θ(·|ỹ<t, x, z̃))  [capability]
                 + Σ_{t ∈ I_c} D_KL(π_θ(·|ỹ<t, x) || π_θ(·|ỹ<t, x, z̃))  [calibration]

The CaOPD loss (Eq. 7) is still standard reverse KL — the innovation is
in the DATA (target replacement), not the loss function itself.

Key: No additional loss term. No reward shaping. No calibration penalty.
Just replace what the model is trained to predict at confidence positions.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def estimate_empirical_confidence(
    outcome_rewards: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Estimate student's empirical confidence μ̂(x) from K rollouts (Eq. 6).

    μ̂(x) = 1/K * Σ_{k=1}^K R(x, a_k)

    where R(x, a_k) ∈ {0, 1} indicates if rollout k is correct.

    Args:
        outcome_rewards: [num_prompts, K] binary rewards per rollout
        num_samples: K (number of rollouts per prompt)

    Returns:
        empirical_conf: [num_prompts] per-prompt empirical confidence
    """
    return outcome_rewards.float().mean(dim=-1)


def create_revised_trajectory(
    reasoning_tokens: torch.Tensor,
    confidence_tokens: torch.Tensor,
    empirical_confidence: torch.Tensor,
    confidence_token_ids: dict,
    tokenizer=None,
) -> torch.Tensor:
    """
    Target Replacement: replace confidence segment c with μ̂(x).

    From Algorithm 1 lines 9-12:
        - Revise completion: ỹ = (a, μ̂(x))
        - Revise teacher context: z̃ with confidence = μ̂(x)

    In practice: the confidence segment is a verbalized statement
    like "Confidence: 0.85". We replace the numeric part with μ̂(x).

    Args:
        reasoning_tokens: [batch, reasoning_len] token ids for reasoning part
        confidence_tokens: [batch, confidence_len] original confidence tokens
        empirical_confidence: [batch] μ̂(x) values
        confidence_token_ids: mapping for verbalized confidence
        tokenizer: for encoding μ̂(x) as tokens

    Returns:
        revised_tokens: [batch, total_len] with confidence segment replaced
    """
    # Implementation depends on tokenizer and confidence format
    # This is a data-pipeline operation, not a loss computation
    raise NotImplementedError(
        "Target replacement is a data-pipeline operation. "
        "Implement with your specific tokenizer and confidence format."
    )


def compute_caopd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    empirical_confidence: torch.Tensor,
    confidence_mask: Optional[torch.Tensor] = None,
    capability_coef: float = 1.0,
    calibration_coef: float = 1.0,
    loss_agg: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    CaOPD loss after target replacement (Eq. 7).

    After target replacement, this is standard per-token reverse KL,
    but the teacher's log_probs at confidence positions now reflect
    μ̂(x) instead of the original privileged confidence.

    L_CaOPD = Σ_{t ∈ I_a} D_KL(student_t || teacher_t)  [capability, unchanged]
            + Σ_{t ∈ I_c} D_KL(student_t || teacher_t)  [calibration, revised target]

    In our implementation, teacher_log_probs should already be computed
    on the REVISED trajectory. The loss itself is standard OPD.

    For smoke testing / simplified usage, we approximate CaOPD by:
    - Reasoning positions: standard OPD toward teacher
    - Confidence positions: OPD toward a target derived from empirical_confidence

    Args:
        student_log_probs: [batch, seq_len] (with grad)
        teacher_log_probs: [batch, seq_len] (from revised teacher context)
        ref_log_probs: [batch, seq_len] unused (kept for API compatibility)
        mask: [batch, seq_len]
        empirical_confidence: [batch] μ̂(x) per prompt
        confidence_mask: [batch, seq_len] 1 at confidence positions, 0 at reasoning
        capability_coef: weight for reasoning positions
        calibration_coef: weight for confidence positions
    """
    if confidence_mask is None:
        # Without explicit confidence mask: treat all as reasoning (standard OPD)
        # + add calibration signal via sequence-level confidence alignment
        reasoning_mask = mask
        conf_mask = torch.zeros_like(mask)
    else:
        reasoning_mask = (1 - confidence_mask) * mask
        conf_mask = confidence_mask * mask

    # Capability loss (reasoning positions): standard OPD
    adv_reasoning = (teacher_log_probs - student_log_probs.detach()) * reasoning_mask
    cap_loss = -(adv_reasoning * student_log_probs * reasoning_mask)
    cap_loss = cap_loss.sum() / reasoning_mask.sum().clamp(min=1)

    # Calibration loss (confidence positions): OPD toward revised target
    # After target replacement, teacher_log_probs at conf positions already
    # encode μ̂(x). So it's still the same OPD formula.
    adv_conf = (teacher_log_probs - student_log_probs.detach()) * conf_mask
    cal_loss = -(adv_conf * student_log_probs * conf_mask)
    cal_loss = cal_loss.sum() / conf_mask.sum().clamp(min=1) if conf_mask.sum() > 0 else torch.tensor(0.0, device=mask.device)

    # When no confidence_mask is provided, add a lightweight calibration proxy:
    # Penalize student's mean sequence log-prob deviating from empirical confidence
    if confidence_mask is None:
        seq_lengths = mask.sum(dim=-1).clamp(min=1)
        student_mean_lp = (student_log_probs * mask).sum(dim=-1) / seq_lengths
        student_conf_proxy = torch.exp(student_mean_lp)
        cal_proxy = F.mse_loss(student_conf_proxy, empirical_confidence, reduction='mean')
        total_loss = capability_coef * cap_loss + calibration_coef * cal_proxy
        cal_loss_val = cal_proxy.item()
    else:
        total_loss = capability_coef * cap_loss + calibration_coef * cal_loss
        cal_loss_val = cal_loss.item()

    # Metrics
    with torch.no_grad():
        # Confidence statistics
        seq_lengths = mask.sum(dim=-1).clamp(min=1)
        student_mean_lp = (student_log_probs.detach() * mask).sum(dim=-1) / seq_lengths
        student_confidence = torch.exp(student_mean_lp)

        overconf_rate = (student_confidence > empirical_confidence + 0.1).float().mean()
        conf_gap = (student_confidence - empirical_confidence).mean()
        ece = compute_ece(student_confidence, empirical_confidence)

    metrics = {
        "caopd/total_loss": total_loss.item(),
        "caopd/capability_loss": cap_loss.item(),
        "caopd/calibration_loss": cal_loss_val,
        "caopd/ece": ece.item(),
        "caopd/overconfidence_rate": overconf_rate.item(),
        "caopd/mean_confidence_gap": conf_gap.item(),
        "caopd/mean_empirical_confidence": empirical_confidence.mean().item(),
        "caopd/mean_student_confidence": student_confidence.mean().item(),
    }

    return total_loss, metrics


def compute_ece(
    predicted_confidence: torch.Tensor,
    actual_accuracy: torch.Tensor,
    n_bins: int = 10,
) -> torch.Tensor:
    """
    Expected Calibration Error.
    ECE = Σ_b |B_b|/N * |acc(B_b) - conf(B_b)|
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=predicted_confidence.device)
    ece = torch.tensor(0.0, device=predicted_confidence.device)
    n_total = predicted_confidence.shape[0]

    for i in range(n_bins):
        in_bin = (predicted_confidence > bin_boundaries[i]) & (predicted_confidence <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            avg_conf = predicted_confidence[in_bin].mean()
            avg_acc = actual_accuracy[in_bin].mean()
            ece += (in_bin.sum().float() / n_total) * torch.abs(avg_acc - avg_conf)

    return ece
