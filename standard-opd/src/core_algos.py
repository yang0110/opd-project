"""
Standard OPD core algorithm implementation for verl framework.

OPD objective (Eq. 4 from G-OPD paper):
    J_OPD(θ) = min_θ E[D_KL(π_θ(y|x) || π*(y|x))]

Gradient (Eq. 6, with discount=0 approximation):
    ∇J_OPD(θ) = E[Σ_t (log π_θ(y_t|x,y<t) - log π*(y_t|x,y<t)) ∇log π_θ(y_t|x,y<t)]

Token-level advantage in OPD:
    A_t^OPD = -(log π_θ(y_t|x,y<t) - log π*(y_t|x,y<t))
            = log π*(y_t|x,y<t) - log π_θ(y_t|x,y<t)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_opd_token_reward(
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token-level implicit reward for standard OPD.

    r_t^OPD = log π*(y_t|x,y<t) / π_ref(y_t|x,y<t)

    For standard OPD, ref = student_init, so:
    r_t = log π*(y_t|h_t) - log π_ref(y_t|h_t)

    Args:
        teacher_log_probs: [batch, seq_len] teacher log probs on student trajectories
        ref_log_probs: [batch, seq_len] reference model log probs
        mask: [batch, seq_len] valid token mask

    Returns:
        token_rewards: [batch, seq_len] dense token-level rewards
    """
    token_rewards = teacher_log_probs - ref_log_probs
    return token_rewards * mask


def compute_opd_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token-level advantage for standard OPD.

    A_t = log π*(y_t|h_t) - log π_θ(y_t|h_t)

    This is the negative of the per-token reverse KL contribution.

    Args:
        student_log_probs: [batch, seq_len] current student log probs
        teacher_log_probs: [batch, seq_len] teacher log probs on same tokens
        mask: [batch, seq_len]

    Returns:
        advantages: [batch, seq_len]
    """
    advantages = teacher_log_probs - student_log_probs
    return advantages * mask


def compute_opd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    mask: torch.Tensor,
    loss_agg: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    Compute the standard OPD policy gradient loss.

    Loss = -E[Σ_t A_t * log π_θ(y_t|h_t)]
         = -E[Σ_t (log π*(y_t|h_t) - log π_θ(y_t|h_t)) * log π_θ(y_t|h_t)]

    Equivalently, this is the reverse KL: D_KL(π_θ || π*) on student trajectories.

    Args:
        student_log_probs: [batch, seq_len] log probs from student (with grad)
        teacher_log_probs: [batch, seq_len] log probs from teacher (detached)
        mask: [batch, seq_len]
        loss_agg: aggregation mode

    Returns:
        loss: scalar loss
        metrics: dict of training metrics
    """
    advantages = compute_opd_advantage(
        student_log_probs.detach(), teacher_log_probs, mask
    )

    # Policy gradient: -A_t * log π_θ(y_t)
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
        kl_per_token = student_log_probs.detach() - teacher_log_probs
        mean_kl = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)
        mean_advantage = (advantages * mask).sum() / mask.sum().clamp(min=1)
        mean_reward = compute_opd_token_reward(
            teacher_log_probs, student_log_probs.detach(), mask
        ).sum() / mask.sum().clamp(min=1)

    metrics = {
        "opd/loss": loss.item(),
        "opd/mean_kl": mean_kl.item(),
        "opd/mean_advantage": mean_advantage.item(),
        "opd/mean_token_reward": mean_reward.item(),
    }

    return loss, metrics


def compute_opd_loss_with_kl_penalty(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    kl_coef: float = 0.1,
    loss_agg: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    OPD loss with explicit KL penalty against reference model.

    Loss = reverse_KL(π_θ || π*) + kl_coef * KL(π_θ || π_ref)

    This is the standard OPD formulation where the implicit reward and KL
    regularization are weighted equally (λ=1 in G-OPD terms).

    Args:
        student_log_probs: [batch, seq_len]
        teacher_log_probs: [batch, seq_len]
        ref_log_probs: [batch, seq_len]
        mask: [batch, seq_len]
        kl_coef: KL penalty coefficient
        loss_agg: aggregation mode
    """
    # Main OPD loss (reverse KL on student trajectories)
    opd_loss, metrics = compute_opd_loss(
        student_log_probs, teacher_log_probs, mask, loss_agg
    )

    # KL penalty against reference
    kl_penalty = (student_log_probs - ref_log_probs) * mask
    if loss_agg == "token-mean":
        kl_term = kl_penalty.sum() / mask.sum().clamp(min=1)
    else:
        kl_term = kl_penalty.sum(dim=-1).mean()

    total_loss = opd_loss + kl_coef * kl_term
    metrics["opd/kl_penalty"] = kl_term.item()
    metrics["opd/total_loss"] = total_loss.item()

    return total_loss, metrics


def compute_reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Direct reverse KL divergence on full distributions (not just sampled tokens).

    D_KL(π_θ || π*) = Σ_v π_θ(v) * (log π_θ(v) - log π*(v))

    Used when full teacher logits are available.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        temperature: softmax temperature
    """
    student_logprobs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_logprobs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.softmax(student_logits / temperature, dim=-1)

    # Reverse KL: Σ π_θ(v) * (log π_θ(v) - log π*(v))
    kl = (student_probs * (student_logprobs - teacher_logprobs)).sum(dim=-1)
    kl = kl * mask

    loss = kl.sum() / mask.sum().clamp(min=1)

    metrics = {
        "opd/reverse_kl": loss.item(),
    }
    return loss, metrics


def compute_forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Forward KL divergence (mode-covering).

    D_KL(π* || π_θ) = Σ_v π*(v) * (log π*(v) - log π_θ(v))

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        temperature: softmax temperature
    """
    student_logprobs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_logprobs = F.log_softmax(teacher_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Forward KL: Σ π*(v) * (log π*(v) - log π_θ(v))
    kl = (teacher_probs * (teacher_logprobs - student_logprobs)).sum(dim=-1)
    kl = kl * mask

    loss = kl.sum() / mask.sum().clamp(min=1)

    metrics = {
        "opd/forward_kl": loss.item(),
    }
    return loss, metrics
