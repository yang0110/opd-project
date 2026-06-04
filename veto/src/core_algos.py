"""
Veto: Stable On-Policy Distillation through Adaptive Target Reformulation.

Core idea: When teacher-student gap is large, directly matching teacher logits
causes pathological gradients. Veto constructs an intermediate target distribution
in logit space that bridges teacher and student.

Key mechanism:
- Forward KL → pathological gradients when teacher has modes student can't cover
- Reverse KL → diversity collapse
- Veto → geometric bridge in logit space, emphasizing consensus regions

The intermediate target:
    π_bridge(v|h_t) ∝ π_teacher(v|h_t)^α * π_student(v|h_t)^(1-α)

where α is adaptive based on teacher-student agreement.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_veto_target_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    base_alpha: float = 0.5,
    adaptive: bool = True,
) -> torch.Tensor:
    """
    Compute Veto's intermediate bridge target in logit space.

    The bridge distribution is a geometric mixture:
        log π_bridge = α * log π_teacher + (1-α) * log π_student + const

    In logit space (before softmax):
        bridge_logits = α * teacher_logits + (1-α) * student_logits

    When adaptive=True, α is modulated by teacher-student agreement:
    - High agreement → α closer to 1 (trust teacher more)
    - Low agreement → α closer to 0 (stay close to student)

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        student_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        alpha: [batch, seq_len] pre-computed mixing weight (optional)
        base_alpha: default mixing weight
        adaptive: whether to compute adaptive alpha

    Returns:
        bridge_logits: [batch, seq_len, vocab_size]
    """
    if alpha is None:
        if adaptive:
            alpha = compute_adaptive_alpha(teacher_logits, student_logits, mask, base_alpha)
        else:
            alpha = torch.full_like(mask, base_alpha).unsqueeze(-1)
            alpha = alpha.expand_as(teacher_logits)

    if alpha.dim() == 2:
        alpha = alpha.unsqueeze(-1)  # [batch, seq_len, 1]

    bridge_logits = alpha * teacher_logits + (1 - alpha) * student_logits
    return bridge_logits


def compute_adaptive_alpha(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    base_alpha: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute position-adaptive mixing weight α based on teacher-student agreement.

    Agreement measure: cosine similarity between teacher and student probability
    distributions, or negative KL divergence.

    High agreement → α → 1 (push toward teacher)
    Low agreement → α → 0 (stay near student, avoid harmful gradients)

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        student_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        base_alpha: baseline mixing weight
        temperature: controls sharpness of adaptation

    Returns:
        alpha: [batch, seq_len] per-position mixing weights in [0, 1]
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)

        # Agreement via Jensen-Shannon divergence (symmetric, bounded [0, log2])
        m = 0.5 * (teacher_probs + student_probs)
        kl_tm = (teacher_probs * (teacher_probs.log() - m.log())).sum(dim=-1)
        kl_sm = (student_probs * (student_probs.log() - m.log())).sum(dim=-1)
        jsd = 0.5 * (kl_tm + kl_sm)  # [batch, seq_len]

        # Convert JSD to agreement score: higher JSD → lower agreement → lower α
        # Use sigmoid mapping: α = base_alpha * sigmoid(-temperature * (JSD - threshold))
        agreement = torch.exp(-temperature * jsd)  # [0, 1], 1 = perfect agreement

        # Scale by base_alpha
        alpha = base_alpha + (1 - base_alpha) * agreement
        alpha = alpha.clamp(0.0, 1.0)

        # Apply mask
        alpha = alpha * mask

    return alpha


def compute_veto_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    base_alpha: float = 0.5,
    adaptive: bool = True,
    kl_direction: str = "reverse",
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute Veto loss: KL between student and bridge target.

    The bridge target is computed adaptively, then we minimize:
        D_KL(π_student || π_bridge)  [reverse KL]
    or:
        D_KL(π_bridge || π_student)  [forward KL]

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        alpha: pre-computed mixing weights (optional)
        base_alpha: base mixing weight for bridge
        adaptive: use adaptive alpha
        kl_direction: "reverse" or "forward"
        temperature: softmax temperature
    """
    # Compute bridge target
    bridge_logits = compute_veto_target_logits(
        teacher_logits, student_logits.detach(), mask,
        alpha=alpha, base_alpha=base_alpha, adaptive=adaptive,
    )

    # Compute KL divergence
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    bridge_log_probs = F.log_softmax(bridge_logits / temperature, dim=-1)

    if kl_direction == "reverse":
        # D_KL(π_student || π_bridge) = Σ π_student * (log π_student - log π_bridge)
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        kl = (student_probs * (student_log_probs - bridge_log_probs)).sum(dim=-1)
    else:
        # D_KL(π_bridge || π_student) = Σ π_bridge * (log π_bridge - log π_student)
        bridge_probs = F.softmax(bridge_logits / temperature, dim=-1)
        kl = (bridge_probs * (bridge_log_probs - student_log_probs)).sum(dim=-1)

    kl = kl * mask
    loss = kl.sum() / mask.sum().clamp(min=1)

    # Metrics
    with torch.no_grad():
        if alpha is None:
            alpha_vals = compute_adaptive_alpha(teacher_logits, student_logits.detach(), mask, base_alpha)
        else:
            alpha_vals = alpha

        # Direct teacher-student KL for comparison
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        direct_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        direct_kl = (direct_kl * mask).sum() / mask.sum().clamp(min=1)

    metrics = {
        "veto/loss": loss.item(),
        "veto/mean_alpha": (alpha_vals * mask).sum().item() / mask.sum().clamp(min=1).item(),
        "veto/direct_teacher_kl": direct_kl.item(),
        "veto/kl_direction": kl_direction,
    }

    return loss, metrics


def compute_veto_loss_token_level(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    base_alpha: float = 0.5,
    adaptive: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Veto-style bridge applied at token log-prob level (without full logits).

    Bridge target:
        log π_bridge(y_t|h_t) = α_t * log π_teacher(y_t|h_t) + (1-α_t) * log π_student(y_t|h_t)

    This approximation works when only per-token log-probs are available
    (no full vocabulary logits from teacher).
    """
    with torch.no_grad():
        # Agreement based on log-prob gap
        gap = torch.abs(teacher_log_probs - student_log_probs)
        agreement = torch.exp(-gap)  # High gap → low agreement → stay near student
        alpha = base_alpha + (1 - base_alpha) * agreement if adaptive else torch.full_like(mask, base_alpha)
        alpha = alpha.clamp(0.0, 1.0) * mask

    # Bridge target at token level
    bridge_log_probs = alpha * teacher_log_probs + (1 - alpha) * student_log_probs.detach()

    # Loss: student moves toward bridge
    advantages = bridge_log_probs - student_log_probs.detach()
    token_losses = -advantages * student_log_probs
    token_losses = token_losses * mask

    loss = token_losses.sum() / mask.sum().clamp(min=1)

    metrics = {
        "veto/loss": loss.item(),
        "veto/mean_alpha": (alpha * mask).sum().item() / mask.sum().clamp(min=1).item(),
    }

    return loss, metrics
