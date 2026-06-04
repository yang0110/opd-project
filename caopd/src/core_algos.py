"""
CaOPD: The Illusion of Certainty - Decoupling Capability and Calibration in OPD.

Core finding: OPD improves accuracy but causes systematic overconfidence
(Scaling Law of Miscalibration). Root cause: teacher-conditioned success
≠ deployment-time confidence.

Solution: Calibration-aware OPD pipeline:
1. Generate student rollouts
2. Estimate empirical confidence from multiple rollouts
3. Replace teacher-conditioned confidence target with student-grounded target
4. Self-distillation with calibrated confidence

Key insight: Capability can be distilled, but confidence should reflect
the student's actual deployment-time reliability, not teacher's privileged view.
"""

import torch
import torch.nn.functional as F
from typing import Optional
import math


def estimate_empirical_confidence(
    student_log_probs_multi: torch.Tensor,
    outcome_rewards: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int = 8,
) -> torch.Tensor:
    """
    Estimate student's empirical confidence from multiple rollouts.

    Empirical confidence = fraction of rollouts that produce correct answer.

    This represents the student's actual deployment-time reliability,
    as opposed to teacher-conditioned success probability.

    Args:
        student_log_probs_multi: [batch, num_samples, seq_len]
        outcome_rewards: [batch, num_samples] binary correct/incorrect
        mask: [batch, seq_len]
        num_samples: number of rollouts per prompt

    Returns:
        empirical_conf: [batch] per-prompt empirical confidence
    """
    # Empirical confidence = mean reward across samples
    empirical_conf = outcome_rewards.float().mean(dim=-1)  # [batch]
    return empirical_conf


def compute_calibration_target(
    empirical_confidence: torch.Tensor,
    teacher_confidence: Optional[torch.Tensor] = None,
    calibration_method: str = "empirical",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute calibration-aware confidence target.

    Methods:
    - "empirical": directly use estimated empirical confidence
    - "smoothed": smooth between teacher and empirical confidence
    - "temperature_scaled": apply temperature scaling to teacher confidence

    Args:
        empirical_confidence: [batch] from multiple rollouts
        teacher_confidence: [batch] from teacher model (optional)
        calibration_method: target construction method
        temperature: temperature for scaling

    Returns:
        calibration_target: [batch] target confidence level
    """
    if calibration_method == "empirical":
        return empirical_confidence

    elif calibration_method == "smoothed" and teacher_confidence is not None:
        # Weighted average: trust empirical more when teacher overconfident
        overconf_mask = (teacher_confidence > empirical_confidence).float()
        alpha = 0.7 * overconf_mask + 0.3 * (1 - overconf_mask)
        return alpha * empirical_confidence + (1 - alpha) * teacher_confidence

    elif calibration_method == "temperature_scaled" and teacher_confidence is not None:
        # Temperature scale teacher confidence
        logit = torch.log(teacher_confidence / (1 - teacher_confidence + 1e-8))
        scaled_logit = logit / temperature
        return torch.sigmoid(scaled_logit)

    else:
        return empirical_confidence


def compute_caopd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    empirical_confidence: torch.Tensor,
    confidence_log_probs: Optional[torch.Tensor] = None,
    capability_coef: float = 1.0,
    calibration_coef: float = 0.5,
    loss_agg: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    CaOPD loss combining capability distillation and calibration alignment.

    L_CaOPD = L_capability + α * L_calibration

    L_capability: Standard OPD (distill task accuracy from teacher)
    L_calibration: Align student's expressed confidence with empirical confidence

    The calibration loss prevents the student from becoming overconfident
    despite improving task accuracy.

    Args:
        student_log_probs: [batch, seq_len] current student log probs
        teacher_log_probs: [batch, seq_len] teacher log probs
        ref_log_probs: [batch, seq_len] reference log probs
        mask: [batch, seq_len]
        empirical_confidence: [batch] estimated true confidence
        confidence_log_probs: [batch] student's self-reported confidence (optional)
        capability_coef: weight for capability loss
        calibration_coef: weight for calibration loss
    """
    # === Capability loss (standard OPD) ===
    advantages = teacher_log_probs - student_log_probs.detach()
    capability_loss = -(advantages * student_log_probs * mask)

    if loss_agg == "token-mean":
        cap_loss = capability_loss.sum() / mask.sum().clamp(min=1)
    else:
        cap_loss = capability_loss.sum() / mask.sum().clamp(min=1)

    # === Calibration loss ===
    # Align student's sequence-level confidence with empirical confidence
    # Using student's mean log-prob as confidence proxy
    with torch.no_grad():
        seq_lengths = mask.sum(dim=-1).clamp(min=1)

    student_mean_lp = (student_log_probs * mask).sum(dim=-1) / seq_lengths
    student_confidence = torch.exp(student_mean_lp)  # Proxy for confidence

    # Calibration: MSE between student confidence proxy and empirical confidence
    cal_loss = F.mse_loss(student_confidence, empirical_confidence, reduction='mean')

    # Total loss
    total_loss = capability_coef * cap_loss + calibration_coef * cal_loss

    # Metrics
    with torch.no_grad():
        # Expected Calibration Error (ECE)
        ece = compute_ece(student_confidence, empirical_confidence, n_bins=10)

        # Overconfidence rate
        overconf_rate = (student_confidence > empirical_confidence + 0.1).float().mean()

        # Mean confidence gap
        conf_gap = (student_confidence - empirical_confidence).mean()

    metrics = {
        "caopd/total_loss": total_loss.item(),
        "caopd/capability_loss": cap_loss.item(),
        "caopd/calibration_loss": cal_loss.item(),
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
    Compute Expected Calibration Error (ECE).

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


def caopd_self_distillation_step(
    student_log_probs: torch.Tensor,
    calibrated_response_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Self-distillation step: distill calibrated response back into student.

    After computing the calibration target, generate a revised response
    that reflects appropriate confidence, then distill it back.

    Args:
        student_log_probs: [batch, seq_len] current student
        calibrated_response_log_probs: [batch, seq_len] from calibrated generation
        mask: [batch, seq_len]
    """
    advantages = calibrated_response_log_probs - student_log_probs.detach()
    loss = -(advantages * student_log_probs * mask).sum() / mask.sum().clamp(min=1)

    metrics = {
        "caopd/self_distill_loss": loss.item(),
    }
    return loss, metrics
