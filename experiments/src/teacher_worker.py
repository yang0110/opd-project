"""
Teacher inference worker: wraps teacher model for OPD training.

Supports two modes:
- log_probs only (standard OPD, G-OPD, veto-token, aligndistil, coverage)
- full logits (veto-logit, entropy-aware, AOPD, extrapolation cliff)

Configured via teacher_inference section of the config YAML.
"""

import torch
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    model_path: str = "Qwen/Qwen3-14B"
    tensor_parallel_size: int = 4
    micro_batch_size_per_gpu: int = 4
    return_logits: bool = False
    return_entropy: bool = False
    gpu_memory_utilization: float = 0.6
    enable_thinking: bool = False


class TeacherWorker:
    """
    Manages teacher model inference for OPD training.

    In verl, this maps to the teacher model worker that runs on a
    separate set of GPUs (or colocated with time-sharing).
    """

    def __init__(self, config: TeacherConfig):
        self.config = config
        self.model = None
        logger.info(
            f"TeacherWorker initialized: TP={config.tensor_parallel_size}, "
            f"logits={config.return_logits}"
        )

    def setup(self):
        """
        Initialize the teacher model with vLLM or HF.

        Integration with verl:
            from vllm import LLM, SamplingParams
            self.model = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )
        """
        logger.info(f"Loading teacher model: {self.config.model_path}")

    def get_teacher_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_positions: torch.Tensor,
    ) -> dict:
        """
        Get teacher model outputs for a batch of sequences.

        Args:
            input_ids: [B, T] full sequences (prompt + response)
            attention_mask: [B, T]
            response_start_positions: [B] where response tokens begin

        Returns:
            dict with:
                - teacher_log_probs: [B, T] log probs at response positions
                - teacher_logits: [B, T, V] (if return_logits=True)
                - teacher_entropy: [B, T] (if return_entropy=True)
        """
        outputs = {}

        # teacher_log_probs is always computed
        # outputs["teacher_log_probs"] = ...

        if self.config.return_logits:
            pass
            # outputs["teacher_logits"] = ...

        if self.config.return_entropy:
            pass
            # outputs["teacher_entropy"] = ...

        return outputs

    @staticmethod
    def estimate_memory_usage(model_path: str, tp_size: int, seq_len: int) -> float:
        """Estimate GPU memory per shard in GB."""
        param_counts = {
            "Qwen/Qwen3-4B": 4e9,
            "Qwen/Qwen3-8B": 8e9,
            "Qwen/Qwen3-14B": 14e9,
        }
        params = param_counts.get(model_path, 14e9)
        params_per_shard = params / tp_size
        param_memory_gb = params_per_shard * 2 / 1e9  # fp16
        kv_cache_gb = seq_len * 128 * 2 * 2 / 1e9  # estimate
        return param_memory_gb + kv_cache_gb
