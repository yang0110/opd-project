"""
AOPD: Asymmetric On-Policy Distillation.

Core idea: Split token-level updates into two asymmetric regimes:
1. Positive advantage tokens → RL-style exploitation (policy gradient)
2. Non-positive advantage tokens → Imitation (localized divergence matching)

Motivation:
- Standard advantage-weighted OPD has: high variance updates, zero-advantage
  gradient vanishing, and exploration bottleneck from insufficient corrective signals.
- AOPD treats positive and non-positive regions differently instead of one unified loss.

Results: +4.09 (strong init) / +8.34 (weak init) avg improvement over standard OPD
on math reasoning, with better policy entropy and capability retention.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_token_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token-level advantage for AOPD.

    A_t = log π*(y_t|h_t) - log π_ref(y_t|h_t)
        = implicit reward from teacher relative to reference

    Positive advantage: token is better than reference baseline → exploit
    Non-positive advantage: token is not better → use imitation to correct
    """
    advantages = teacher_log_probs - ref_log_probs
    return advantages * mask


def compute_aopd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    student_logits: Optional[torch.Tensor] = None,
    teacher_logits: Optional[torch.Tensor] = None,
    exploit_coef: float = 1.0,
    imitate_coef: float = 1.0,
    temperature: float = 1.0,
    loss_agg: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    AOPD asymmetric loss.

    For each token t:
        if A_t > 0:  L_t = -A_t * log π_θ(y_t|h_t)  [exploitation / RL-style]
        else:        L_t = D_local(π_θ || π*)_t        [imitation / divergence matching]

    The exploitation branch reinforces tokens where teacher improves over reference.
    The imitation branch locally aligns student with teacher where the token
    isn't worth reinforcing but still provides corrective signal.

    Args:
        student_log_probs: [batch, seq_len] current student log probs (with grad)
        teacher_log_probs: [batch, seq_len] teacher log probs
        ref_log_probs: [batch, seq_len] reference model log probs
        mask: [batch, seq_len]
        student_logits: [batch, seq_len, vocab] (optional, for full distribution matching)
        teacher_logits: [batch, seq_len, vocab] (optional)
        exploit_coef: weight for exploitation loss
        imitate_coef: weight for imitation loss
        temperature: softmax temperature for distribution matching
        loss_agg: aggregation mode
    """
    # Compute advantage
    advantages = compute_token_advantage(
        student_log_probs.detach(), teacher_log_probs, ref_log_probs, mask
    )

    # Split into positive and non-positive regions
    pos_mask = (advantages > 0).float() * mask
    neg_mask = (advantages <= 0).float() * mask

    # ===== Exploitation branch (positive advantage) =====
    # Standard policy gradient: -A_t * log π_θ(y_t)
    exploit_loss = -advantages * student_log_probs * pos_mask

    # ===== Imitation branch (non-positive advantage) =====
    if student_logits is not None and teacher_logits is not None:
        # Full distribution matching via reverse KL at non-positive positions
        student_lp = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_lp = F.log_softmax(teacher_logits / temperature, dim=-1)
        student_p = F.softmax(student_logits / temperature, dim=-1)

        # Local reverse KL: Σ_v π_θ(v) * (log π_θ(v) - log π*(v))
        local_kl = (student_p * (student_lp - teacher_lp)).sum(dim=-1)
        imitate_loss = local_kl * neg_mask
    else:
        # Token-level approximation: move toward teacher log prob
        # L_imitate = (log π_θ(y_t) - log π*(y_t))^2 / 2 (squared error on log-probs)
        log_diff = student_log_probs - teacher_log_probs
        imitate_loss = 0.5 * log_diff.pow(2) * neg_mask

    # Combined loss
    total_token_loss = exploit_coef * exploit_loss + imitate_coef * imitate_loss

    if loss_agg == "token-mean":
        loss = total_token_loss.sum() / mask.sum().clamp(min=1)
    elif loss_agg == "seq-mean-token-sum":
        seq_losses = total_token_loss.sum(dim=-1)
        loss = seq_losses.mean()
    else:
        loss = total_token_loss.sum() / mask.sum().clamp(min=1)

    # Metrics
    with torch.no_grad():
        n_pos = pos_mask.sum().clamp(min=1)
        n_neg = neg_mask.sum().clamp(min=1)
        n_total = mask.sum().clamp(min=1)

        mean_pos_adv = (advantages * pos_mask).sum() / n_pos
        mean_neg_adv = (advantages * neg_mask).sum() / n_neg
        frac_positive = n_pos / n_total

        exploit_loss_val = (exploit_coef * exploit_loss).sum() / n_pos
        imitate_loss_val = (imitate_coef * imitate_loss).sum() / n_neg

        # Entropy of student (proxy for diversity)
        if student_logits is not None:
            student_entropy = -(F.softmax(student_logits, dim=-1) *
                               F.log_softmax(student_logits, dim=-1)).sum(dim=-1)
            mean_entropy = (student_entropy * mask).sum() / n_total
        else:
            mean_entropy = torch.tensor(0.0)

    metrics = {
        "aopd/loss": loss.item(),
        "aopd/exploit_loss": exploit_loss_val.item(),
        "aopd/imitate_loss": imitate_loss_val.item(),
        "aopd/frac_positive": frac_positive.item(),
        "aopd/mean_pos_advantage": mean_pos_adv.item(),
        "aopd/mean_neg_advantage": mean_neg_adv.item(),
        "aopd/student_entropy": mean_entropy.item(),
    }

    return loss, metrics


def compute_aopd_loss_with_clipping(
    student_log_probs: torch.Tensor,
    old_student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float = 0.2,
    exploit_coef: float = 1.0,
    imitate_coef: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    AOPD with PPO-style importance ratio clipping for the exploitation branch.

    Exploitation branch uses clipped IS ratio for stability:
        ratio = π_θ(y_t) / π_θ_old(y_t)
        L_exploit = -min(ratio * A_t, clip(ratio, 1-ε, 1+ε) * A_t)

    Imitation branch remains unclipped (local divergence matching).
    """
    advantages = compute_token_advantage(
        student_log_probs.detach(), teacher_log_probs, ref_log_probs, mask
    )

    pos_mask = (advantages > 0).float() * mask
    neg_mask = (advantages <= 0).float() * mask

    # Exploitation with clipping
    log_ratio = student_log_probs - old_student_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    exploit_loss = torch.max(pg_loss1, pg_loss2) * pos_mask

    # Imitation (token-level squared error)
    log_diff = student_log_probs - teacher_log_probs
    imitate_loss = 0.5 * log_diff.pow(2) * neg_mask

    total_loss = exploit_coef * exploit_loss + imitate_coef * imitate_loss
    loss = total_loss.sum() / mask.sum().clamp(min=1)

    with torch.no_grad():
        clip_frac = ((ratio - 1).abs() > clip_range).float()
        clip_frac = (clip_frac * pos_mask).sum() / pos_mask.sum().clamp(min=1)

    metrics = {
        "aopd/loss": loss.item(),
        "aopd/clip_fraction": clip_frac.item(),
        "aopd/frac_positive": pos_mask.sum().item() / mask.sum().clamp(min=1).item(),
    }

    return loss, metrics
