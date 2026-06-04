"""
AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation.

This implements the BASIC version (Theorem 1, Eq. 11):
    z_t* = (β_0/β) * z_t^dpo + (1 - β_0/β) * z_t^ref

The FULL version (Eq. 14-17) additionally uses:
- A reverse DPO model (trained with chosen/rejected swapped) for contrastive reward
- Token-adaptive α based on TVD between DPO and reverse DPO (Eq. 16-17):
    z_t* = z_t^dpo + α_t * (z_t^dpo - z_t^dpo^-)
    α_t = D_TVD(t) * r + ε
- Per-token β_t weighting in the loss: β_t = β_0/α_t

The full version requires an additional reverse-DPO model not available here.
Our adaptive α uses a KL-based heuristic as a simplified proxy.

Paper: arXiv:2503.02832
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_aligndistil_target(
    dpo_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    beta_0: float = 0.1,
    beta: float = 0.1,
    extrapolation_factor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute AlignDistil target distribution from DPO model and reference.

    Basic version (Eq. 11):
        z_t* = (β_0/β) * z_t^dpo + (1 - β_0/β) * z_t^ref

    where β_0 = DPO training temperature, β = RLHF optimization temperature.
    When β_0 = β: z_t* = z_t^dpo (no extrapolation, just use DPO model).
    When β > β_0: α < 1 (interpolation toward ref, more conservative).
    When β < β_0: α > 1 (extrapolation beyond DPO, more aggressive).

    With adaptive extrapolation (Eq. 17):
        z_t* = z_t^dpo + α_t * (z_t^dpo - z_t^dpo^-)

    Args:
        dpo_logits: [batch, seq_len, vocab] DPO-aligned model logits
        ref_logits: [batch, seq_len, vocab] reference (base/DPO_ref) model logits
        beta_0: DPO training temperature (β_0 in paper)
        beta: RLHF optimization temperature (β in paper)
        extrapolation_factor: [batch, seq_len] or scalar, adaptive α_t override

    Returns:
        target_logits: [batch, seq_len, vocab]
    """
    if extrapolation_factor is None:
        # Default from Eq. 11: α = β_0 / β
        alpha = beta_0 / beta
    else:
        alpha = extrapolation_factor

    if isinstance(alpha, (int, float)):
        target_logits = alpha * dpo_logits + (1 - alpha) * ref_logits
    else:
        # alpha is per-position: [batch, seq_len] → [batch, seq_len, 1]
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(-1)
        target_logits = alpha * dpo_logits + (1 - alpha) * ref_logits

    return target_logits


def compute_token_adaptive_alpha(
    dpo_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    base_alpha: float = 1.0,
    under_opt_threshold: float = 0.1,
    over_opt_threshold: float = 5.0,
) -> torch.Tensor:
    """
    Compute token-adaptive extrapolation factor to avoid over/under-optimization.

    For each token position:
    - If student is far from target (under-optimized): increase α
    - If student overshoots target (over-optimized): decrease α

    Detection:
    - Under-optimization: KL(student || target) > threshold_high
    - Over-optimization: student logit exceeds target in reward direction

    Args:
        dpo_logits: [batch, seq_len, vocab]
        ref_logits: [batch, seq_len, vocab]
        student_logits: [batch, seq_len, vocab]
        mask: [batch, seq_len]
        base_alpha: base extrapolation factor
        under_opt_threshold: KL threshold for under-optimization
        over_opt_threshold: KL threshold for over-optimization

    Returns:
        alpha: [batch, seq_len] adaptive extrapolation factor
    """
    with torch.no_grad():
        # Target distribution (at base_alpha)
        target_logits = base_alpha * dpo_logits + (1 - base_alpha) * ref_logits
        target_probs = F.softmax(target_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)

        # KL divergence student → target
        student_lp = F.log_softmax(student_logits, dim=-1)
        target_lp = F.log_softmax(target_logits, dim=-1)
        kl = (student_probs * (student_lp - target_lp)).sum(dim=-1)  # [batch, seq_len]

        # Reward direction: how much DPO differs from ref
        reward_magnitude = (dpo_logits - ref_logits).norm(dim=-1)  # [batch, seq_len]

        # Adaptive alpha
        alpha = torch.full_like(kl, base_alpha)

        # Under-optimized: student far from target → increase alpha slightly
        under_mask = (kl > under_opt_threshold) & (kl < over_opt_threshold)
        alpha[under_mask] = base_alpha * 1.1

        # Over-optimized: student overshoots → decrease alpha
        over_mask = kl > over_opt_threshold
        alpha[over_mask] = base_alpha * 0.8

        # Clip alpha to safe range
        alpha = alpha.clamp(0.5, 2.0)
        alpha = alpha * mask

    return alpha


def compute_aligndistil_loss(
    student_logits: torch.Tensor,
    dpo_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.1,
    adaptive: bool = True,
    kl_direction: str = "forward",
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    AlignDistil loss: distill from alignment-derived target distribution.

    The target is constructed from DPO model and reference model logits,
    with optional token-adaptive extrapolation.

    Args:
        student_logits: [batch, seq_len, vocab] student model (with grad)
        dpo_logits: [batch, seq_len, vocab] DPO-aligned teacher
        ref_logits: [batch, seq_len, vocab] reference (base) model
        mask: [batch, seq_len]
        beta: DPO temperature
        adaptive: use token-adaptive extrapolation
        kl_direction: "forward" or "reverse"
        temperature: softmax temperature
    """
    # Compute adaptive alpha if needed
    if adaptive:
        alpha = compute_token_adaptive_alpha(
            dpo_logits, ref_logits, student_logits.detach(), mask
        )
    else:
        alpha = None

    # Compute target
    target_logits = compute_aligndistil_target(
        dpo_logits, ref_logits, beta_0=beta, beta=beta, extrapolation_factor=alpha
    )

    # Compute KL loss
    student_lp = F.log_softmax(student_logits / temperature, dim=-1)
    target_lp = F.log_softmax(target_logits / temperature, dim=-1)

    if kl_direction == "forward":
        # D_KL(target || student)
        target_probs = F.softmax(target_logits / temperature, dim=-1)
        kl = (target_probs * (target_lp - student_lp)).sum(dim=-1)
    else:
        # D_KL(student || target)
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        kl = (student_probs * (student_lp - target_lp)).sum(dim=-1)

    kl = kl * mask
    loss = kl.sum() / mask.sum().clamp(min=1)

    # Metrics
    with torch.no_grad():
        mean_alpha = (alpha * mask).sum() / mask.sum().clamp(min=1) if alpha is not None else torch.tensor(1.0 / beta)
        # Implicit reward magnitude
        reward = (F.log_softmax(dpo_logits, dim=-1) - F.log_softmax(ref_logits, dim=-1))
        reward_entropy = -(F.softmax(dpo_logits, dim=-1) * reward).sum(dim=-1)
        mean_reward_mag = (reward_entropy.abs() * mask).sum() / mask.sum().clamp(min=1)

    metrics = {
        "aligndistil/loss": loss.item(),
        "aligndistil/mean_alpha": mean_alpha.item() if torch.is_tensor(mean_alpha) else mean_alpha,
        "aligndistil/beta": beta,
        "aligndistil/mean_reward_magnitude": mean_reward_mag.item(),
    }

    return loss, metrics


def compute_aligndistil_token_level_loss(
    student_log_probs: torch.Tensor,
    dpo_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.1,
    adaptive_alpha: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """
    AlignDistil at token log-prob level (no full logits needed).

    Target log-prob at token level:
        log π_target(y_t) = α * log π_DPO(y_t) + (1-α) * log π_ref(y_t)

    Then standard OPD-style loss toward this target.
    """
    if adaptive_alpha is None:
        alpha = 1.0  # β_0/β = 1 when β_0 = β (default: no extrapolation)
    else:
        alpha = adaptive_alpha

    if isinstance(alpha, (int, float)):
        target_log_probs = alpha * dpo_log_probs + (1 - alpha) * ref_log_probs
    else:
        target_log_probs = alpha * dpo_log_probs + (1 - alpha) * ref_log_probs

    # OPD-style advantage toward target
    advantages = target_log_probs - student_log_probs.detach()
    token_losses = -advantages * student_log_probs * mask

    loss = token_losses.sum() / mask.sum().clamp(min=1)

    metrics = {
        "aligndistil/loss": loss.item(),
    }
    return loss, metrics
