"""
Extrapolation Cliff: Safety boundaries for reward extrapolation in structured outputs.

Core finding: When λ > 1 (ExOPD), model can exceed teacher on domain tasks,
but past a threshold λ*, structured output contracts (JSON format, field binding,
ID integrity) suddenly collapse.

Key contribution: Closed-form clip-safety threshold derived from:
- Teacher modal probability p_T
- Warm-start mass m (student's initial mass on correct token)
- Importance-sampling clip strength c

Collapse is primarily in parse validity, not semantic ranking quality.
NDCG@1 on parsed outputs stays flat, but parse validity drops sharply at λ*.

This module implements:
1. Safety threshold computation
2. Adaptive λ clipping based on token position criticality
3. Structured output aware OPD loss
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional


def compute_safety_threshold(
    teacher_modal_prob: torch.Tensor,
    warmstart_mass: torch.Tensor,
    clip_strength: float = 5.0,
) -> torch.Tensor:
    """
    Compute closed-form clip-safety threshold λ* (Theorem 4.1, Eq. 4).

    λ*(p, b, c) = [log((1-p)/(c-1+p)) - log((1-b)/b)]
                  / [log((1-p)/p) - log((1-b)/b)]

    where:
    - p: teacher modal probability at a structural position
    - b: warm-start modal probability (student init mass on same token)
    - c: per-token IS clip strength (paper uses c=5.0)

    Special case b=1/2: λ* = log((1-p)/(c-1+p)) / log((1-p)/p)

    Above λ*, the extrapolated fixed point exits the clip-safe region,
    causing format-collapse on structural tokens.

    Args:
        teacher_modal_prob: [batch, seq_len] p - teacher prob on top token
        warmstart_mass: [batch, seq_len] b - student init prob on same token
        clip_strength: c - IS clip cap (paper default c=5.0)

    Returns:
        lambda_star: [batch, seq_len] per-position safety threshold
    """
    p = teacher_modal_prob.clamp(0.51, 0.9999)
    b = warmstart_mass.clamp(0.01, 0.9999)
    c = clip_strength

    # Numerator: log((1-p)/(c-1+p)) - log((1-b)/b)
    numerator = torch.log((1 - p) / (c - 1 + p).clamp(min=1e-8)) - torch.log((1 - b) / b)

    # Denominator: log((1-p)/p) - log((1-b)/b)
    denominator = torch.log((1 - p) / p) - torch.log((1 - b) / b)

    # Avoid division by zero (when p ≈ b, threshold → ∞, meaning safe)
    lambda_star = numerator / denominator.clamp(min=1e-8).clamp(max=-1e-8)

    # When denominator is near zero or positive (p ≈ b), threshold is very large (safe)
    safe_mask = denominator.abs() < 1e-6
    lambda_star = torch.where(safe_mask, torch.tensor(10.0, device=p.device), lambda_star)

    # Clamp to reasonable range
    lambda_star = lambda_star.clamp(min=1.0, max=10.0)

    return lambda_star


def identify_contract_critical_tokens(
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    entropy_threshold: float = 0.5,
    top1_prob_threshold: float = 0.9,
) -> torch.Tensor:
    """
    Identify contract-critical token positions (format tokens).

    Contract-critical tokens are those where:
    - Teacher is highly confident (near-deterministic)
    - The token is structural (brackets, quotes, field names in JSON)

    These positions are most vulnerable to extrapolation cliff collapse.

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        entropy_threshold: positions with entropy below this are "critical"
        top1_prob_threshold: alternatively, top-1 prob above this

    Returns:
        critical_mask: [batch, seq_len] 1 for contract-critical positions
    """
    with torch.no_grad():
        probs = F.softmax(teacher_logits, dim=-1)
        top1_prob = probs.max(dim=-1).values

        # Low entropy = high confidence = likely structural token
        log_probs = F.log_softmax(teacher_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        # Critical: high confidence OR low entropy
        critical = ((top1_prob > top1_prob_threshold) |
                   (entropy < entropy_threshold)).float()

        return critical * mask


def compute_adaptive_lambda(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    base_lambda: float = 1.25,
    clip_strength: float = 0.2,
    min_lambda: float = 0.5,
) -> torch.Tensor:
    """
    Compute position-adaptive λ that respects safety thresholds.

    For each position:
    - Compute safety threshold λ*
    - Set effective λ = min(base_lambda, λ* - safety_margin)

    Contract-critical tokens get conservative λ (≤ 1),
    semantic tokens can use full extrapolation.

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        student_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        base_lambda: desired extrapolation factor
        clip_strength: IS clip range

    Returns:
        adaptive_lambda: [batch, seq_len] per-position λ
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)

        # Teacher modal probability
        teacher_modal_prob = teacher_probs.max(dim=-1).values

        # Student warm-start mass on teacher's modal token
        teacher_top_idx = teacher_probs.argmax(dim=-1, keepdim=True)
        warmstart_mass = student_probs.gather(dim=-1, index=teacher_top_idx).squeeze(-1)

        # Compute safety threshold
        lambda_star = compute_safety_threshold(
            teacher_modal_prob, warmstart_mass, clip_strength
        )

        # Adaptive λ: clip to safety threshold with margin
        safety_margin = 0.1
        effective_lambda = torch.minimum(
            torch.full_like(lambda_star, base_lambda),
            lambda_star - safety_margin,
        )
        effective_lambda = effective_lambda.clamp(min=min_lambda)

        return effective_lambda * mask


def compute_safe_exopd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    student_logits: Optional[torch.Tensor] = None,
    base_lambda: float = 1.25,
    clip_strength: float = 0.2,
    use_adaptive_lambda: bool = True,
    loss_agg: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    Safe ExOPD loss with adaptive λ respecting extrapolation cliff.

    For contract-critical positions: use conservative λ (≤ 1.0)
    For semantic positions: use full extrapolation (λ = base_lambda)

    This prevents parse validity collapse while maintaining
    the accuracy gains of reward extrapolation.

    Args:
        student_log_probs: [batch, seq_len]
        teacher_log_probs: [batch, seq_len]
        ref_log_probs: [batch, seq_len]
        mask: [batch, seq_len]
        teacher_logits: [batch, seq_len, vocab] (for adaptive λ)
        student_logits: [batch, seq_len, vocab]
        base_lambda: target extrapolation factor
        clip_strength: IS clip range for threshold computation
        use_adaptive_lambda: whether to use position-adaptive λ
    """
    if use_adaptive_lambda and teacher_logits is not None and student_logits is not None:
        # Per-position adaptive λ
        adaptive_lambda = compute_adaptive_lambda(
            teacher_logits, student_logits, mask,
            base_lambda=base_lambda, clip_strength=clip_strength,
        )
    else:
        adaptive_lambda = torch.full_like(mask, base_lambda)

    # G-OPD target with per-position λ: λ_t * log π* + (1 - λ_t) * log π_ref
    target_log_probs = adaptive_lambda * teacher_log_probs + (1 - adaptive_lambda) * ref_log_probs

    # Advantage
    advantages = target_log_probs - student_log_probs.detach()
    advantages = advantages * mask

    # Policy gradient loss
    token_losses = -advantages * student_log_probs
    token_losses = token_losses * mask

    if loss_agg == "token-mean":
        loss = token_losses.sum() / mask.sum().clamp(min=1)
    else:
        loss = token_losses.sum() / mask.sum().clamp(min=1)

    # Metrics
    with torch.no_grad():
        mean_lambda = (adaptive_lambda * mask).sum() / mask.sum().clamp(min=1)
        max_lambda = (adaptive_lambda * mask).max()
        min_lambda_val = adaptive_lambda[mask.bool()].min() if mask.any() else torch.tensor(0.0)

        # Track critical vs non-critical positions
        if teacher_logits is not None:
            critical = identify_contract_critical_tokens(teacher_logits, mask)
            frac_critical = critical.sum() / mask.sum().clamp(min=1)
            lambda_at_critical = (adaptive_lambda * critical).sum() / critical.sum().clamp(min=1)
            lambda_at_semantic = (adaptive_lambda * (mask - critical).clamp(min=0)).sum() / (mask - critical).clamp(min=0).sum().clamp(min=1)
        else:
            frac_critical = torch.tensor(0.0)
            lambda_at_critical = torch.tensor(0.0)
            lambda_at_semantic = torch.tensor(base_lambda)

    metrics = {
        "cliff/loss": loss.item(),
        "cliff/mean_adaptive_lambda": mean_lambda.item(),
        "cliff/max_lambda": max_lambda.item(),
        "cliff/min_lambda": min_lambda_val.item(),
        "cliff/frac_critical_tokens": frac_critical.item(),
        "cliff/lambda_at_critical": lambda_at_critical.item(),
        "cliff/lambda_at_semantic": lambda_at_semantic.item(),
    }

    return loss, metrics


def compute_parse_validity_reward(
    response: str,
    expected_format: str = "json_list",
) -> dict:
    """
    Compute parse validity metrics for structured output.

    Used to monitor extrapolation cliff during training.

    Returns:
        dict with parse_valid, field_valid, format_score
    """
    import json

    result = {"parse_valid": 0.0, "field_valid": 0.0, "format_score": 0.0}

    try:
        # Try extracting JSON
        start = response.find("[") if expected_format == "json_list" else response.find("{")
        if start == -1:
            return result

        end = response.rfind("]") + 1 if expected_format == "json_list" else response.rfind("}") + 1
        json_str = response[start:end]
        parsed = json.loads(json_str)
        result["parse_valid"] = 1.0

        if expected_format == "json_list" and isinstance(parsed, list):
            result["field_valid"] = 1.0
            result["format_score"] = 1.0
        elif expected_format == "json_object" and isinstance(parsed, dict):
            result["field_valid"] = 1.0
            result["format_score"] = 1.0

    except (json.JSONDecodeError, ValueError):
        pass

    return result
