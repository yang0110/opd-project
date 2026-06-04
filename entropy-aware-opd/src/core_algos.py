"""
Entropy-Aware On-Policy Distillation.

Core idea: KL direction should not be globally fixed. Teacher entropy
at each position determines whether to use:
- Reverse KL (mode-seeking): when teacher is confident (low entropy)
- Forward KL (mode-covering): when teacher is uncertain (high entropy)

Key principle:
    low teacher entropy → reverse KL (precise imitation of peaked mode)
    high teacher entropy → forward KL (cover all plausible outputs)

Results: +1.37, +2.39, +5.05 Pass@8 improvement on Qwen3-0.6B/1.7B/4B
across 6 math reasoning benchmarks.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_teacher_entropy(
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute per-position teacher entropy.

    H(π*_t) = -Σ_v π*(v|h_t) * log π*(v|h_t)

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        top_k: if set, compute entropy only over top-k tokens

    Returns:
        entropy: [batch, seq_len]
    """
    if top_k is not None:
        topk_logits, _ = teacher_logits.topk(top_k, dim=-1)
        probs = F.softmax(topk_logits, dim=-1)
        log_probs = F.log_softmax(topk_logits, dim=-1)
    else:
        probs = F.softmax(teacher_logits, dim=-1)
        log_probs = F.log_softmax(teacher_logits, dim=-1)

    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy * mask


def compute_entropy_threshold(
    teacher_entropy: torch.Tensor,
    mask: torch.Tensor,
    method: str = "percentile",
    percentile: float = 0.5,
    fixed_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute threshold for switching between reverse and forward KL.

    Methods:
    - "percentile": use median or specified percentile of batch entropy
    - "fixed": use a fixed threshold value
    - "adaptive": use running mean + std

    Args:
        teacher_entropy: [batch, seq_len]
        mask: [batch, seq_len]
        method: thresholding method
        percentile: percentile for percentile method
        fixed_threshold: fixed threshold value

    Returns:
        threshold: scalar or [batch] threshold
    """
    if method == "fixed" and fixed_threshold is not None:
        return torch.tensor(fixed_threshold, device=teacher_entropy.device)

    # Gather valid entropy values
    valid_entropy = teacher_entropy[mask.bool()]

    if method == "percentile":
        threshold = torch.quantile(valid_entropy, percentile)
    elif method == "adaptive":
        mean = valid_entropy.mean()
        std = valid_entropy.std()
        threshold = mean + 0.5 * std  # Positions above mean+0.5σ → high entropy
    else:
        threshold = valid_entropy.median()

    return threshold


def compute_entropy_aware_mixing_weight(
    teacher_entropy: torch.Tensor,
    mask: torch.Tensor,
    threshold: Optional[torch.Tensor] = None,
    sharpness: float = 5.0,
) -> torch.Tensor:
    """
    Compute per-position mixing weight between reverse and forward KL.

    w_t ∈ [0, 1]:
        w_t → 0: use reverse KL (teacher confident)
        w_t → 1: use forward KL (teacher uncertain)

    Uses sigmoid for smooth transition:
        w_t = σ(sharpness * (H(π*_t) - threshold))

    Args:
        teacher_entropy: [batch, seq_len]
        mask: [batch, seq_len]
        threshold: entropy threshold for switching
        sharpness: sigmoid sharpness (higher = sharper transition)

    Returns:
        weights: [batch, seq_len] forward KL mixing weight
    """
    if threshold is None:
        threshold = compute_entropy_threshold(teacher_entropy, mask)

    weights = torch.sigmoid(sharpness * (teacher_entropy - threshold))
    return weights * mask


def compute_entropy_aware_opd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    threshold_method: str = "percentile",
    threshold_percentile: float = 0.5,
    sharpness: float = 5.0,
    forward_kl_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Entropy-Aware OPD loss with adaptive KL direction.

    L = Σ_t [(1 - w_t) * D_KL^rev(π_θ || π*) + w_t * D_KL^fwd(π* || π_θ)]

    where w_t is determined by teacher entropy at position t.

    Low entropy → mostly reverse KL (precise mode matching)
    High entropy → add forward KL (cover multiple valid options)

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        temperature: softmax temperature
        threshold_method: how to determine entropy threshold
        threshold_percentile: percentile for threshold
        sharpness: sigmoid sharpness for weight computation
        forward_kl_weight: scaling for forward KL term
    """
    # Compute teacher entropy
    teacher_entropy = compute_teacher_entropy(teacher_logits, mask)

    # Compute mixing weights
    threshold = compute_entropy_threshold(
        teacher_entropy, mask, method=threshold_method, percentile=threshold_percentile
    )
    weights = compute_entropy_aware_mixing_weight(
        teacher_entropy, mask, threshold=threshold, sharpness=sharpness
    )

    # Compute both KL directions
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Reverse KL: D_KL(π_θ || π*) = Σ π_θ * (log π_θ - log π*)
    reverse_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

    # Forward KL: D_KL(π* || π_θ) = Σ π* * (log π* - log π_θ)
    forward_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)

    # Mixed loss
    mixed_kl = (1 - weights) * reverse_kl + weights * forward_kl_weight * forward_kl
    mixed_kl = mixed_kl * mask

    loss = mixed_kl.sum() / mask.sum().clamp(min=1)

    # Metrics
    with torch.no_grad():
        mean_entropy = (teacher_entropy * mask).sum() / mask.sum().clamp(min=1)
        mean_weight = (weights * mask).sum() / mask.sum().clamp(min=1)
        frac_forward = (weights > 0.5).float()
        frac_forward = (frac_forward * mask).sum() / mask.sum().clamp(min=1)
        mean_rev_kl = (reverse_kl * mask).sum() / mask.sum().clamp(min=1)
        mean_fwd_kl = (forward_kl * mask).sum() / mask.sum().clamp(min=1)

    metrics = {
        "entropy_opd/loss": loss.item(),
        "entropy_opd/mean_teacher_entropy": mean_entropy.item(),
        "entropy_opd/mean_forward_weight": mean_weight.item(),
        "entropy_opd/frac_forward_kl": frac_forward.item(),
        "entropy_opd/threshold": threshold.item(),
        "entropy_opd/mean_reverse_kl": mean_rev_kl.item(),
        "entropy_opd/mean_forward_kl": mean_fwd_kl.item(),
    }

    return loss, metrics


def compute_entropy_aware_opd_loss_token_level(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_entropy: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 2.0,
    sharpness: float = 5.0,
) -> tuple[torch.Tensor, dict]:
    """
    Entropy-aware OPD using only token-level log-probs (no full logits).

    Approximation when full vocabulary logits aren't available:
    - Low entropy positions: use reverse KL via advantage (standard OPD)
    - High entropy positions: use symmetric loss (less mode-seeking)

    Args:
        student_log_probs: [batch, seq_len] per-token
        teacher_log_probs: [batch, seq_len] per-token
        teacher_entropy: [batch, seq_len] pre-computed teacher entropy
        mask: [batch, seq_len]
    """
    weights = torch.sigmoid(sharpness * (teacher_entropy - threshold))
    weights = weights * mask

    # Standard OPD advantage (reverse KL style)
    adv_reverse = teacher_log_probs - student_log_probs.detach()

    # Symmetric penalty (approximation of forward KL at token level)
    # For the sampled token, forward KL contribution ≈ -(log π_θ - log π*)
    adv_symmetric = 0.5 * adv_reverse + 0.5 * (-adv_reverse)  # → 0 (uninformative)
    # Better approximation: use absolute gap with sign
    adv_forward = -(student_log_probs.detach() - teacher_log_probs)  # same as adv_reverse

    # In practice, at token level, the difference manifests as weighting:
    # High entropy → reduce the gradient magnitude (teacher signal unreliable)
    # Low entropy → full gradient (teacher signal reliable)
    effective_weight = 1.0 - 0.5 * weights  # Reduce weight at high-entropy positions

    token_losses = -effective_weight * adv_reverse * student_log_probs
    token_losses = token_losses * mask

    loss = token_losses.sum() / mask.sum().clamp(min=1)

    metrics = {
        "entropy_opd/loss": loss.item(),
        "entropy_opd/mean_weight": (weights * mask).sum().item() / mask.sum().clamp(min=1).item(),
    }

    return loss, metrics
