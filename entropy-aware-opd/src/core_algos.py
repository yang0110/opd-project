"""
Entropy-Aware On-Policy Distillation (EOPD).

From paper (Eq. 9, Algorithm 1):
    L_EOPD(θ; c_t) = L_OPD(θ; c_t) + α * I[H_t^te > τ] * L_FKL(θ; c_t)

where:
    L_OPD = clipped reverse KL (standard OPD, Eq. 7-8)
    L_FKL = forward KL over teacher's top-k tokens (Eq. 10)
    H_t^te = teacher entropy at position t
    τ = entropy threshold
    α = forward KL weight (default 1.0)
    I[·] = indicator function (hard threshold, NOT sigmoid)

Key: Forward KL is ADDED to (not replaces) reverse KL, only at high-entropy positions.
Forward KL uses teacher's top-k tokens only (k=16 in paper) for efficiency.

Results: +1.37/+2.39/+5.05 Pass@8 on Qwen3-0.6B/1.7B/4B.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_teacher_entropy(
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-position teacher entropy.

    H_t^te = -Σ_v π_te(v|c_t) * log π_te(v|c_t)

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]

    Returns:
        entropy: [batch, seq_len]
    """
    probs = F.softmax(teacher_logits, dim=-1)
    log_probs = F.log_softmax(teacher_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy * mask


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    top_k: int = 16,
) -> torch.Tensor:
    """
    Forward KL approximated over teacher's top-k tokens (Eq. 10).

    L_FKL(θ; c_t) ≈ Σ_{x ∈ S_k} π̃_te(x|c_t) * log(π̃_te(x|c_t) / π_θ(x|c_t))

    where π̃_te is teacher distribution renormalized over top-k:
        π̃_te(x|c_t) = π_te(x|c_t) / Σ_{x' ∈ S_k} π_te(x'|c_t)

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        top_k: number of teacher tokens to use (paper uses k=16)

    Returns:
        fkl: [batch, seq_len] per-position forward KL
    """
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    # Get teacher's top-k tokens
    topk_probs, topk_indices = teacher_probs.topk(top_k, dim=-1)

    # Renormalize teacher over top-k
    topk_probs_normalized = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    # Get student log-probs at those same positions
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_lp = student_log_probs.gather(dim=-1, index=topk_indices)

    # Teacher log-probs (renormalized)
    teacher_topk_lp = topk_probs_normalized.log()

    # Forward KL: Σ π̃_T(v) * (log π̃_T(v) - log π_S(v))
    fkl = (topk_probs_normalized * (teacher_topk_lp - student_topk_lp)).sum(dim=-1)

    return fkl * mask


def compute_reverse_kl_opd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Standard reverse KL for OPD (per-position).

    L_RKL(θ; c_t) = D_KL(π_θ(·|c_t) || π_te(·|c_t))
                   = Σ_v π_θ(v) * (log π_θ(v) - log π_te(v))

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]

    Returns:
        rkl: [batch, seq_len] per-position reverse KL
    """
    student_probs = F.softmax(student_logits, dim=-1)
    student_lp = F.log_softmax(student_logits, dim=-1)
    teacher_lp = F.log_softmax(teacher_logits, dim=-1)

    rkl = (student_probs * (student_lp - teacher_lp)).sum(dim=-1)
    return rkl * mask


def compute_entropy_aware_opd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 0.8,
    alpha: float = 1.0,
    top_k: int = 16,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Entropy-Aware OPD loss (Eq. 9 from paper).

    L_EOPD = L_OPD + α * I[H_t^te > τ] * L_FKL

    Forward KL is ADDED (not replaces) reverse KL at high-entropy positions.
    Uses hard indicator I[H > τ], not soft sigmoid.
    Forward KL computed over teacher's top-k tokens for efficiency.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        tau: entropy threshold τ (paper uses τ=0.8)
        alpha: forward KL weight α (paper uses α=1.0)
        top_k: teacher top-k for forward KL (paper uses k=16)
        temperature: softmax temperature
    """
    # Apply temperature
    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits / temperature

    # Compute teacher entropy
    teacher_entropy = compute_teacher_entropy(teacher_scaled, mask)

    # Hard indicator: I[H_t > τ]
    high_entropy_mask = (teacher_entropy > tau).float() * mask

    # Reverse KL (standard OPD component)
    rkl = compute_reverse_kl_opd(student_scaled, teacher_scaled, mask)

    # Forward KL on top-k (only at high-entropy positions)
    fkl = compute_forward_kl_topk(student_scaled, teacher_scaled, mask, top_k=top_k)

    # Combined loss: L_OPD + α * I[H > τ] * L_FKL
    per_token_loss = rkl + alpha * high_entropy_mask * fkl
    per_token_loss = per_token_loss * mask

    loss = per_token_loss.sum() / mask.sum().clamp(min=1)

    # Metrics
    with torch.no_grad():
        mean_entropy = (teacher_entropy * mask).sum() / mask.sum().clamp(min=1)
        frac_high_entropy = high_entropy_mask.sum() / mask.sum().clamp(min=1)
        mean_rkl = (rkl * mask).sum() / mask.sum().clamp(min=1)
        mean_fkl = (fkl * high_entropy_mask).sum() / high_entropy_mask.sum().clamp(min=1)
        # Forward KL contribution to total loss
        fkl_contribution = (alpha * high_entropy_mask * fkl).sum() / mask.sum().clamp(min=1)

    metrics = {
        "entropy_opd/loss": loss.item(),
        "entropy_opd/mean_teacher_entropy": mean_entropy.item(),
        "entropy_opd/frac_high_entropy": frac_high_entropy.item(),
        "entropy_opd/threshold_tau": tau,
        "entropy_opd/mean_reverse_kl": mean_rkl.item(),
        "entropy_opd/mean_forward_kl_at_high_H": mean_fkl.item(),
        "entropy_opd/fkl_contribution": fkl_contribution.item(),
    }

    return loss, metrics
