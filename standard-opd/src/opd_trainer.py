"""
Standard OPD Trainer built on verl's RayPPOTrainer architecture.

Training loop:
1. Student generates trajectories (on-policy rollout)
2. Teacher computes logits/log_probs on student trajectories
3. Reference model computes log_probs (optional, for KL penalty)
4. Compute OPD loss and update student
"""

import torch
from dataclasses import dataclass, field
from typing import Optional

from .core_algos import (
    compute_opd_loss,
    compute_opd_loss_with_kl_penalty,
    compute_opd_token_reward,
    compute_reverse_kl_loss,
)


@dataclass
class OPDConfig:
    # Model
    student_model_path: str = ""
    teacher_model_path: str = ""
    ref_model_path: Optional[str] = None

    # Training
    learning_rate: float = 5e-7
    num_train_epochs: int = 1
    num_opd_steps: int = 50
    batch_size: int = 128
    mini_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # OPD specific
    kl_coef: float = 0.0  # KL penalty against reference (0 = pure OPD)
    temperature: float = 1.0
    loss_type: str = "token_pg"  # "token_pg" | "reverse_kl" | "forward_kl"
    loss_agg: str = "token-mean"

    # Generation
    max_prompt_length: int = 1024
    max_response_length: int = 16384
    num_generations: int = 1
    generation_temperature: float = 1.0
    top_p: float = 1.0

    # Distributed
    num_gpus: int = 8
    tensor_parallel_size: int = 1
    rollout_tp_size: int = 1

    # Logging
    log_interval: int = 1
    save_interval: int = 10
    eval_interval: int = 10
    output_dir: str = "./output/standard_opd"
    wandb_project: str = "opd"
    wandb_run_name: str = "standard_opd"


class OPDTrainer:
    """
    Standard OPD trainer following verl's architecture.

    Implements the core OPD loop:
        for step in range(num_opd_steps):
            1. Sample batch of prompts
            2. Student rollout (generate trajectories)
            3. Teacher inference (get log_probs on student tokens)
            4. [Optional] Reference inference (get ref log_probs)
            5. Compute OPD loss and update student
    """

    def __init__(self, config: OPDConfig):
        self.config = config

    def compute_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor],
        mask: torch.Tensor,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute OPD loss based on config.loss_type.
        """
        if self.config.loss_type == "token_pg":
            if ref_log_probs is not None and self.config.kl_coef > 0:
                return compute_opd_loss_with_kl_penalty(
                    student_log_probs=student_log_probs,
                    teacher_log_probs=teacher_log_probs,
                    ref_log_probs=ref_log_probs,
                    mask=mask,
                    kl_coef=self.config.kl_coef,
                    loss_agg=self.config.loss_agg,
                )
            else:
                return compute_opd_loss(
                    student_log_probs=student_log_probs,
                    teacher_log_probs=teacher_log_probs,
                    mask=mask,
                    loss_agg=self.config.loss_agg,
                )

        elif self.config.loss_type == "reverse_kl":
            assert student_logits is not None and teacher_logits is not None
            return compute_reverse_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=mask,
                temperature=self.config.temperature,
            )

        elif self.config.loss_type == "forward_kl":
            assert student_logits is not None and teacher_logits is not None
            from .core_algos import compute_forward_kl_loss
            return compute_forward_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=mask,
                temperature=self.config.temperature,
            )

        else:
            raise ValueError(f"Unknown loss_type: {self.config.loss_type}")

    def compute_metrics(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> dict:
        """Compute training metrics for logging."""
        with torch.no_grad():
            metrics = {}
            # Token-level KL
            kl = (student_log_probs - teacher_log_probs) * mask
            metrics["train/kl_student_teacher"] = kl.sum() / mask.sum().clamp(min=1)

            if ref_log_probs is not None:
                kl_ref = (student_log_probs - ref_log_probs) * mask
                metrics["train/kl_student_ref"] = kl_ref.sum() / mask.sum().clamp(min=1)

            # Implicit reward
            if ref_log_probs is not None:
                rewards = compute_opd_token_reward(teacher_log_probs, ref_log_probs, mask)
                metrics["train/mean_implicit_reward"] = rewards.sum() / mask.sum().clamp(min=1)

            # Response length
            metrics["train/mean_response_length"] = mask.sum(dim=-1).float().mean()

            return {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
