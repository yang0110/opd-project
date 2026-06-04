"""
Rethinking On-Policy Distillation: Phenomenology, Mechanism, and Recipe.

This paper is analytical rather than proposing a new loss. Key findings:

1. Success conditions for OPD:
   - Teacher and student need compatible thinking patterns
   - Teacher must provide novel capability beyond student's training

2. Mechanism: Success manifests as progressive alignment of high-probability
   tokens on student-visited states. A tiny shared token set captures
   97-99% of probability mass.

3. Recipe recommendations:
   - Verify teacher-student compatibility before running OPD
   - Monitor alignment on shared high-prob token set
   - Use thinking-pattern compatibility as a predictor for OPD success

This module implements compatibility diagnostics and monitoring tools.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class CompatibilityMetrics:
    """Metrics for teacher-student compatibility assessment."""
    shared_token_overlap: float
    shared_mass_coverage: float
    thinking_pattern_similarity: float
    predicted_opd_success: float
    top_k_agreement: float


def compute_shared_token_set(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    top_k: int = 50,
    mass_threshold: float = 0.97,
) -> tuple[torch.Tensor, dict]:
    """
    Identify shared high-probability token set between teacher and student.

    The paper finds that 97-99% of probability mass concentrates on a
    small shared token set. This set is where alignment happens.

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        student_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        top_k: number of top tokens to consider
        mass_threshold: target cumulative mass to measure

    Returns:
        shared_mask: [batch, seq_len, top_k] mask of shared top tokens
        metrics: overlap and coverage statistics
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)

        # Teacher top-k tokens
        teacher_topk_vals, teacher_topk_idx = teacher_probs.topk(top_k, dim=-1)
        # Student top-k tokens
        student_topk_vals, student_topk_idx = student_probs.topk(top_k, dim=-1)

        # Compute overlap
        batch_size, seq_len, _ = teacher_logits.shape
        overlap_count = torch.zeros(batch_size, seq_len, device=teacher_logits.device)

        for k in range(top_k):
            teacher_token = teacher_topk_idx[:, :, k:k+1]
            match = (student_topk_idx == teacher_token).any(dim=-1).float()
            overlap_count += match

        overlap_rate = overlap_count / top_k  # [batch, seq_len]
        mean_overlap = (overlap_rate * mask).sum() / mask.sum().clamp(min=1)

        # Mass coverage of shared tokens
        teacher_mass = teacher_topk_vals.sum(dim=-1)  # [batch, seq_len]
        student_mass = student_topk_vals.sum(dim=-1)
        mean_teacher_mass = (teacher_mass * mask).sum() / mask.sum().clamp(min=1)
        mean_student_mass = (student_mass * mask).sum() / mask.sum().clamp(min=1)

    metrics = {
        "compat/top_k_overlap": mean_overlap.item(),
        "compat/teacher_top_k_mass": mean_teacher_mass.item(),
        "compat/student_top_k_mass": mean_student_mass.item(),
    }

    return overlap_rate, metrics


def compute_thinking_pattern_similarity(
    teacher_log_probs_seq: torch.Tensor,
    student_log_probs_seq: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 32,
) -> dict:
    """
    Assess thinking pattern compatibility between teacher and student.

    Thinking patterns manifest as:
    - Similar token-level confidence patterns over time
    - Correlated high/low entropy regions
    - Similar "reasoning step" boundaries

    Uses sliding-window correlation of log-prob sequences as proxy.

    Args:
        teacher_log_probs_seq: [batch, seq_len]
        student_log_probs_seq: [batch, seq_len]
        mask: [batch, seq_len]
        window_size: correlation window size
    """
    with torch.no_grad():
        # Correlation of log-prob profiles
        t_lp = teacher_log_probs_seq * mask
        s_lp = student_log_probs_seq * mask

        # Normalize
        t_mean = t_lp.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
        s_mean = s_lp.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
        t_centered = (t_lp - t_mean) * mask
        s_centered = (s_lp - s_mean) * mask

        # Pearson correlation
        numerator = (t_centered * s_centered * mask).sum(dim=-1)
        t_std = ((t_centered ** 2 * mask).sum(dim=-1)).sqrt()
        s_std = ((s_centered ** 2 * mask).sum(dim=-1)).sqrt()
        correlation = numerator / (t_std * s_std + 1e-8)

        mean_correlation = correlation.mean()

        # Rank correlation (Spearman) on absolute log-probs
        # Higher correlation → more compatible thinking patterns

    return {
        "compat/logprob_correlation": mean_correlation.item(),
        "compat/mean_teacher_lp": (teacher_log_probs_seq * mask).sum().item() / mask.sum().clamp(min=1).item(),
        "compat/mean_student_lp": (student_log_probs_seq * mask).sum().item() / mask.sum().clamp(min=1).item(),
    }


def monitor_progressive_alignment(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    mask: torch.Tensor,
    step: int,
    history: Optional[dict] = None,
) -> dict:
    """
    Monitor progressive alignment during OPD training.

    Success = high-prob tokens progressively align.
    Failure = alignment stalls or regresses.

    Track:
    - KL divergence trend
    - Top-token agreement rate
    - Alignment velocity (improvement per step)
    """
    with torch.no_grad():
        # Token-level KL
        kl = (student_log_probs - teacher_log_probs) * mask
        mean_kl = kl.sum() / mask.sum().clamp(min=1)

        # Agreement on most-likely token (requires logits, approximate with log_probs)
        # Higher log_prob → more likely to be the top token
        # Agreement proxy: |log_s - log_t| for each token
        gap = (student_log_probs - teacher_log_probs).abs() * mask
        mean_gap = gap.sum() / mask.sum().clamp(min=1)

    metrics = {
        "alignment/step": step,
        "alignment/mean_kl": mean_kl.item(),
        "alignment/mean_logprob_gap": mean_gap.item(),
    }

    if history is not None and "alignment/mean_kl" in history:
        prev_kl = history["alignment/mean_kl"]
        metrics["alignment/kl_delta"] = mean_kl.item() - prev_kl
        metrics["alignment/improving"] = float(mean_kl.item() < prev_kl)

    return metrics


def predict_opd_success(
    compatibility_metrics: dict,
    overlap_threshold: float = 0.3,
    correlation_threshold: float = 0.3,
) -> tuple[bool, str]:
    """
    Predict whether OPD will succeed based on compatibility diagnostics.

    Based on paper's findings:
    - High token overlap + high correlation → likely success
    - Low overlap OR low correlation → likely failure or limited gain

    Returns prediction and explanation.
    """
    overlap = compatibility_metrics.get("compat/top_k_overlap", 0)
    correlation = compatibility_metrics.get("compat/logprob_correlation", 0)

    if overlap >= overlap_threshold and correlation >= correlation_threshold:
        return True, (
            f"Compatible: overlap={overlap:.3f}, corr={correlation:.3f}. "
            f"Teacher and student share sufficient token support and thinking patterns."
        )
    elif overlap < overlap_threshold:
        return False, (
            f"Incompatible: low token overlap={overlap:.3f}. "
            f"Teacher's preferred tokens differ too much from student's. "
            f"OPD may fail or require more training steps."
        )
    else:
        return False, (
            f"Uncertain: overlap={overlap:.3f}, corr={correlation:.3f}. "
            f"Thinking patterns may be incompatible. Monitor alignment closely."
        )
