"""
AOPD: Asymmetric On-Policy Distillation.

From paper (Eq. 8-10, Section 5.1-5.2):

    Advantage: A_t = sg[log P_T(y_t|c_t) - log P_S(y_t|c_t)]  (Eq. 1)

    Intervention mask G_t: G_t = I(P_T(y_t|c_t) - P_S(y_t|c_t) ≤ τ)  (Eq. 9)
    Default τ=0 → G_t activates on non-positive advantage (probability domain)

    AOPD objective (Eq. 10):
        L_AOPD = E[1/|y| * Σ_t (G_t * L_FKL_t + (1-G_t) * L_OPD_t)]

    where:
        L_OPD_t = standard advantage-weighted policy gradient (exploitation)
        L_FKL_t = forward KL on teacher's top-K support (imitation/guidance)

    Truncated Forward KL (Eq. 7):
        L_FKL_t = 1/K * Σ_{v ∈ TopK(P_T)} P_T(v|c_t) * (log P_T(v|c_t) - log P_S(v|c_t))

Key design choices from paper:
- Intervention uses PROBABILITY difference (P_T - P_S), not log-prob
- Imitation branch uses FORWARD KL (teacher→student) on teacher's top-K tokens
- τ=0 by default (intervene on all non-positive advantage tokens)
- K (top-K) truncation for computational efficiency
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_advantage(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token-level advantage (Eq. 1).

    A_t = sg[log P_T(y_t|c_t) - log P_S(y_t|c_t)]

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        student_logits: [batch, seq_len, vocab_size]
        input_ids: [batch, seq_len] sampled token ids
        mask: [batch, seq_len]

    Returns:
        advantages: [batch, seq_len]
    """
    with torch.no_grad():
        teacher_lp = F.log_softmax(teacher_logits, dim=-1)
        student_lp = F.log_softmax(student_logits, dim=-1)

        teacher_token_lp = teacher_lp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        student_token_lp = student_lp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

        advantages = teacher_token_lp - student_token_lp

    return advantages * mask


def compute_intervention_mask(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 0.0,
) -> torch.Tensor:
    """
    Compute intervention mask G_t (Eq. 9).

    G_t = I(P_T(y_t|c_t) - P_S(y_t|c_t) ≤ τ)

    NOTE: Uses PROBABILITY difference, not log-prob.
    τ=0 → intervene when teacher prob ≤ student prob for sampled token
         (i.e., non-positive advantage in probability space)

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        student_logits: [batch, seq_len, vocab_size]
        input_ids: [batch, seq_len]
        mask: [batch, seq_len]
        tau: threshold (default 0, paper default)

    Returns:
        G: [batch, seq_len] binary intervention mask
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)

        teacher_token_prob = teacher_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        student_token_prob = student_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

        prob_diff = teacher_token_prob - student_token_prob
        G = (prob_diff <= tau).float() * mask

    return G


def compute_truncated_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    top_k: int = 10,
) -> torch.Tensor:
    """
    Truncated forward KL on teacher's top-K support (Eq. 7).

    L_FKL_t = 1/K * Σ_{v ∈ S_t} P_T(v|c_t) * (log P_T(v|c_t) - log P_S(v|c_t))

    where S_t = TopK(P_T(·|c_t), K)

    Forward KL is the natural choice because:
    - It preserves the teacher-conditioned measure on the support
    - It specifies both candidate tokens AND reference distribution
    - Reverse KL would reweight by student, defeating the purpose

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        top_k: number of teacher top tokens

    Returns:
        fkl: [batch, seq_len] per-position truncated forward KL
    """
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Teacher's top-K tokens
    topk_probs, topk_indices = teacher_probs.topk(top_k, dim=-1)

    # Teacher log-probs at top-K
    teacher_topk_lp = teacher_log_probs.gather(-1, topk_indices)

    # Student log-probs at teacher's top-K positions
    student_topk_lp = student_log_probs.gather(-1, topk_indices)

    # Forward KL: 1/K * Σ P_T(v) * (log P_T(v) - log P_S(v))
    fkl = (1.0 / top_k) * (topk_probs * (teacher_topk_lp - student_topk_lp)).sum(dim=-1)

    return fkl * mask


def compute_aopd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    student_logits: Optional[torch.Tensor] = None,
    teacher_logits: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    tau: float = 0.0,
    top_k: int = 10,
    loss_agg: str = "token-mean",
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    AOPD objective (Eq. 8 + 10).

    L_AOPD = E[1/|y| * Σ_t (G_t * L_FKL_t + (1-G_t) * L_OPD_t)]

    Exploitation (1-G_t): Standard OPD advantage-weighted policy gradient
    Imitation (G_t): Truncated forward KL on teacher's top-K

    Args:
        student_log_probs: [batch, seq_len] per-token log probs (with grad)
        teacher_log_probs: [batch, seq_len] per-token teacher log probs
        ref_log_probs: [batch, seq_len] reference log probs (for advantage)
        mask: [batch, seq_len]
        student_logits: [batch, seq_len, vocab] needed for forward KL
        teacher_logits: [batch, seq_len, vocab] needed for forward KL + mask
        input_ids: [batch, seq_len] token ids (for probability-based mask)
        tau: intervention threshold (default 0)
        top_k: top-K for truncated forward KL
    """
    # Compute advantage A_t = log P_T(y_t) - log P_S(y_t)
    advantages = (teacher_log_probs - student_log_probs.detach()) * mask

    if student_logits is not None and teacher_logits is not None:
        # Full AOPD with proper intervention mask and forward KL

        if input_ids is None:
            # Infer: use advantage sign as proxy for probability-based mask
            G = (advantages <= tau).float() * mask
        else:
            G = compute_intervention_mask(
                teacher_logits, student_logits.detach(), input_ids, mask, tau
            )

        # Exploitation branch: standard OPD loss (1-G_t positions)
        exploit_loss = -advantages * student_log_probs * (1 - G)

        # Imitation branch: truncated forward KL (G_t positions)
        fkl = compute_truncated_forward_kl(student_logits, teacher_logits, mask, top_k)
        imitate_loss = fkl * G

        # Combined
        total_token_loss = exploit_loss + imitate_loss
    else:
        # Fallback without full logits: use log-prob advantage for splitting
        G = (advantages <= 0).float() * mask

        # Exploitation
        exploit_loss = -advantages * student_log_probs * (1 - G)

        # Imitation approximation (squared error on log-probs)
        log_diff = student_log_probs - teacher_log_probs
        imitate_loss = 0.5 * log_diff.pow(2) * G

        total_token_loss = exploit_loss + imitate_loss

    if loss_agg == "token-mean":
        loss = total_token_loss.sum() / mask.sum().clamp(min=1)
    elif loss_agg == "seq-mean-token-mean":
        seq_lengths = mask.sum(dim=-1).clamp(min=1)
        loss = (total_token_loss.sum(dim=-1) / seq_lengths).mean()
    else:
        loss = total_token_loss.sum() / mask.sum().clamp(min=1)

    # Metrics
    with torch.no_grad():
        n_intervened = G.sum().clamp(min=1)
        n_exploit = ((1 - G) * mask).sum().clamp(min=1)
        n_total = mask.sum().clamp(min=1)

        frac_intervened = n_intervened / n_total
        mean_exploit = (exploit_loss).sum() / n_exploit
        mean_imitate = (imitate_loss).sum() / n_intervened if student_logits is not None else torch.tensor(0.0)
        mean_advantage = (advantages * mask).sum() / n_total

        # Student entropy
        if student_logits is not None:
            s_entropy = -(F.softmax(student_logits, dim=-1) *
                         F.log_softmax(student_logits, dim=-1)).sum(dim=-1)
            mean_entropy = (s_entropy * mask).sum() / n_total
        else:
            mean_entropy = torch.tensor(0.0)

    metrics = {
        "aopd/loss": loss.item(),
        "aopd/exploit_loss": mean_exploit.item(),
        "aopd/imitate_loss": mean_imitate.item(),
        "aopd/frac_intervened": frac_intervened.item(),
        "aopd/mean_advantage": mean_advantage.item(),
        "aopd/student_entropy": mean_entropy.item(),
        "aopd/tau": tau,
    }

    return loss, metrics


def compute_aopd_loss_with_clipping(
    student_log_probs: torch.Tensor,
    old_student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float = 0.2,
    tau: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    AOPD with PPO-style clipping on exploitation branch.

    Exploitation uses clipped IS ratio (standard PPO):
        ratio = π_θ(y_t) / π_θ_old(y_t)
        L_exploit = -min(ratio * A_t, clip(ratio, 1-ε, 1+ε) * A_t)

    Imitation branch uses simple squared log-prob error (fallback).
    """
    advantages = (teacher_log_probs - old_student_log_probs) * mask
    G = (advantages <= tau).float() * mask

    # Exploitation with clipping
    log_ratio = student_log_probs - old_student_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    exploit_loss = torch.max(pg_loss1, pg_loss2) * (1 - G)

    # Imitation (token-level)
    log_diff = student_log_probs - teacher_log_probs
    imitate_loss = 0.5 * log_diff.pow(2) * G

    total_loss = exploit_loss + imitate_loss
    loss = total_loss.sum() / mask.sum().clamp(min=1)

    with torch.no_grad():
        clip_frac = ((ratio - 1).abs() > clip_range).float()
        clip_frac = (clip_frac * (1 - G) * mask).sum() / ((1 - G) * mask).sum().clamp(min=1)
        frac_intervened = G.sum() / mask.sum().clamp(min=1)

    metrics = {
        "aopd/loss": loss.item(),
        "aopd/clip_fraction": clip_frac.item(),
        "aopd/frac_intervened": frac_intervened.item(),
    }

    return loss, metrics
