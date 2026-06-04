"""
Veto: Stable On-Policy Distillation through Adaptive Target Reformulation.

From paper (Algorithm 1 + Section 4.2):
    Target Q is a Product of Experts (geometric bridge):
        Q(y|x) ∝ exp(z_T(y|x) + β * z_S(y|x))
                = P_T(y|x) * P_S(y|x)^β

    In logit space:
        target_logits = teacher_logits + β * student_logits

    β decays linearly over training: β_i = β_init * (1 - i/N)

    KL direction per Algorithm 1:
        Forward KL regime: L_t = D_KL(Q || P_θ)
        Reverse KL regime: L_t = D_KL(P_θ || Q)

Key properties (Theorem 1 + 2):
- Adaptive Gradient Veto: when P_S(y) → 0 for teacher-preferred tokens,
  Q incorporates student uncertainty, preventing gradient explosion
- Sharpening Effect: optimal student converges to P_T^(1/(1-β)),
  naturally more decisive than teacher when 0 < β < 1
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_veto_target_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Compute Veto target distribution in logit space.

    Q ∝ exp(z_T + β * z_S)  [Product of Experts]

    In logit space (before softmax):
        target_logits = teacher_logits + β * student_logits

    NOT a convex combination. β controls how much the student's
    current beliefs influence the target.

    Args:
        teacher_logits: [batch, seq_len, vocab_size] z_T
        student_logits: [batch, seq_len, vocab_size] z_S
        beta: scalar ≥ 0, decays over training

    Returns:
        target_logits: [batch, seq_len, vocab_size]
    """
    return teacher_logits + beta * student_logits


def compute_beta_schedule(
    step: int,
    total_steps: int,
    beta_init: float = 1.0,
) -> float:
    """
    Linear β decay schedule from Algorithm 1.

    β_i = β_init * (1 - i/N)

    β starts at β_init and decays to 0 over training.
    As β → 0, target Q → P_T (pure teacher).
    Early training (high β): target is a compromise.
    Late training (low β): target approaches teacher directly.

    Args:
        step: current training step i
        total_steps: total training steps N
        beta_init: initial β value

    Returns:
        beta: current β value
    """
    return beta_init * (1.0 - step / total_steps)


def compute_veto_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
    kl_direction: str = "forward",
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute Veto loss: KL between student and Veto target Q.

    From Algorithm 1:
        Forward KL regime: L_t = D_KL(Q || P_θ) — mode-covering, stable
        Reverse KL regime: L_t = D_KL(P_θ || Q) — mode-seeking, decisive

    The paper shows β acts as:
    - In forward KL: Adaptive Gradient Veto (suppresses gradient explosion)
    - In reverse KL: Decisiveness Knob (controls mode-seeking intensity)

    Args:
        student_logits: [batch, seq_len, vocab_size] P_θ
        teacher_logits: [batch, seq_len, vocab_size] P_T
        mask: [batch, seq_len]
        beta: current β value (from decay schedule)
        kl_direction: "forward" (D_KL(Q||P_θ)) or "reverse" (D_KL(P_θ||Q))
        temperature: softmax temperature
    """
    # Compute Veto target: Q ∝ exp(z_T + β * z_S)
    target_logits = compute_veto_target_logits(
        teacher_logits, student_logits.detach(), beta
    )

    # Distributions
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    target_log_probs = F.log_softmax(target_logits / temperature, dim=-1)

    if kl_direction == "forward":
        # D_KL(Q || P_θ) = Σ Q(v) * (log Q(v) - log P_θ(v))
        target_probs = F.softmax(target_logits / temperature, dim=-1)
        kl = (target_probs * (target_log_probs - student_log_probs)).sum(dim=-1)
    else:
        # D_KL(P_θ || Q) = Σ P_θ(v) * (log P_θ(v) - log Q(v))
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        kl = (student_probs * (student_log_probs - target_log_probs)).sum(dim=-1)

    kl = kl * mask
    loss = kl.sum() / mask.sum().clamp(min=1)

    # Metrics
    with torch.no_grad():
        # Direct teacher-student KL for comparison
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        direct_fwd_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        direct_fwd_kl = (direct_fwd_kl * mask).sum() / mask.sum().clamp(min=1)

        # Target entropy (higher β → closer to student → higher entropy target)
        target_probs_m = F.softmax(target_logits, dim=-1)
        target_entropy = -(target_probs_m * target_log_probs).sum(dim=-1)
        mean_target_entropy = (target_entropy * mask).sum() / mask.sum().clamp(min=1)

    metrics = {
        "veto/loss": loss.item(),
        "veto/beta": beta,
        "veto/kl_direction": kl_direction,
        "veto/direct_teacher_fwd_kl": direct_fwd_kl.item(),
        "veto/target_entropy": mean_target_entropy.item(),
    }

    return loss, metrics


def compute_veto_loss_token_level(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Veto-style bridge at token log-prob level (approximation without full logits).

    When only per-token log-probs are available, approximate the Veto effect:
    The advantage is computed toward a softened target that incorporates
    student's current state via β weighting.

    This is a heuristic approximation — the full-logits version is preferred.

    Args:
        student_log_probs: [batch, seq_len]
        teacher_log_probs: [batch, seq_len]
        ref_log_probs: [batch, seq_len] (unused, kept for interface compatibility)
        mask: [batch, seq_len]
        beta: Veto β parameter
    """
    # At token level, Veto's effect is to dampen the teacher signal
    # proportional to how much the student already agrees
    # Approximate: target_lp ≈ (teacher_lp + β * student_lp) / (1 + β)
    # (normalization is approximate at token level)
    target_lp = (teacher_log_probs + beta * student_log_probs.detach()) / (1 + beta)

    # Standard OPD-style loss toward softened target
    advantages = target_lp - student_log_probs.detach()
    token_losses = -advantages * student_log_probs
    token_losses = token_losses * mask

    loss = token_losses.sum() / mask.sum().clamp(min=1)

    metrics = {
        "veto/loss": loss.item(),
        "veto/beta": beta,
    }

    return loss, metrics
