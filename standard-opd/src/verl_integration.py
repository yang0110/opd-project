"""
Integration with verl framework - registers OPD as an algorithm in verl's registry.

This module provides:
1. OPD advantage estimator (registered via @register_adv_est)
2. OPD policy loss function (registered via @register_policy_loss)
3. OPD-specific reward computation

These plug directly into verl's existing PPO trainer infrastructure,
allowing OPD to run with minimal modification to verl's training loop.
"""

import torch
from typing import Optional


# ============================================================
# Advantage Estimator Registration
# ============================================================

def register_opd_advantage():
    """
    Register OPD advantage estimator with verl's registry.

    In verl, advantage estimators are registered via @register_adv_est decorator
    and called from compute_advantage() in ray_trainer.py.

    OPD advantage:
        A_t = log π*(y_t|h_t) - log π_θ(y_t|h_t)

    This requires teacher_log_probs to be present in the batch.
    """
    # @register_adv_est("opd")
    def compute_opd_advantage_verl(
        token_level_rewards: torch.Tensor,
        values: Optional[torch.Tensor],
        eos_mask: torch.Tensor,
        gamma: float,
        lam: float,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        OPD advantage computation for verl.

        In OPD, the token_level_rewards already contain the teacher-student
        log ratio, so the advantage IS the reward (no GAE needed).

        Returns:
            advantages: [batch, seq_len]
            returns: [batch, seq_len] (same as advantages for OPD)
        """
        advantages = token_level_rewards * eos_mask
        returns = advantages  # No value function in standard OPD
        return advantages, returns

    return compute_opd_advantage_verl


# ============================================================
# Policy Loss Registration
# ============================================================

def register_opd_policy_loss():
    """
    Register OPD policy loss with verl's registry.

    In verl, policy losses are registered via @register_policy_loss decorator.

    OPD loss: -Σ_t A_t * log π_θ(y_t|h_t)
    where A_t = log π*(y_t|h_t) - log π_θ_old(y_t|h_t)
    """
    # @register_policy_loss("opd")
    def compute_opd_policy_loss_verl(
        log_probs: torch.Tensor,       # current student log probs
        old_log_probs: torch.Tensor,    # old student log probs (from rollout)
        advantages: torch.Tensor,       # OPD advantages
        eos_mask: torch.Tensor,
        cliprange: float = 0.2,         # not used in standard OPD
        **kwargs,
    ) -> torch.Tensor:
        """
        OPD policy loss for verl.

        Unlike PPO which uses clipped importance ratios, standard OPD
        directly uses the advantage-weighted policy gradient.
        """
        # Policy gradient loss
        pg_loss = -advantages * log_probs
        pg_loss = (pg_loss * eos_mask).sum() / eos_mask.sum().clamp(min=1)
        return pg_loss

    return compute_opd_policy_loss_verl


# ============================================================
# Token-level Reward Computation
# ============================================================

def compute_opd_token_rewards_for_verl(
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token-level OPD rewards to feed into verl's reward pipeline.

    r_t = log π*(y_t|h_t) - log π_ref(y_t|h_t)

    This integrates with verl's apply_kl_penalty() which adds KL-based
    adjustments to token_level_rewards.

    Args:
        teacher_log_probs: [batch, seq_len] from teacher model
        ref_log_probs: [batch, seq_len] from reference model
        response_mask: [batch, seq_len]

    Returns:
        token_rewards: [batch, seq_len]
    """
    token_rewards = (teacher_log_probs - ref_log_probs) * response_mask
    return token_rewards


# ============================================================
# Main Integration Entry Point
# ============================================================

def get_opd_config_overrides() -> dict:
    """
    Configuration overrides to run OPD within verl's PPO trainer.

    These settings configure verl's RayPPOTrainer to operate in OPD mode:
    - Use teacher model for reward computation
    - Use OPD advantage estimator
    - Use OPD policy loss
    - Disable critic/value function
    """
    return {
        "algorithm": {
            "adv_estimator": "opd",
            "policy_loss": "opd",
            "use_critic": False,        # OPD doesn't need value function
            "use_kl_penalty": True,     # KL against reference
            "kl_ctrl": {"type": "fixed", "kl_coef": 0.0},
        },
        "actor_rollout_ref": {
            "rollout": {
                "temperature": 1.0,
                "top_p": 1.0,
                "n": 1,
            },
        },
        "teacher_model": {
            "enable": True,
            "return_log_probs": True,
            "return_logits": False,
        },
        "reward_model": {
            "enable": False,  # OPD uses teacher as implicit reward
        },
    }
