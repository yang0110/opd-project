"""
Worker definitions for OPD training on verl.

Following verl's worker architecture:
- ActorRolloutWorker: generates student trajectories
- TeacherWorker: computes teacher logits/log_probs on student tokens
- RefWorker: computes reference model log_probs

These map to verl's existing worker types:
- ActorRolloutRefWorker → student generation + ref log_probs
- TeacherModel → teacher inference on student trajectories
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class RolloutOutput:
    """Output from student rollout generation."""
    input_ids: torch.Tensor          # [batch, prompt_len + response_len]
    attention_mask: torch.Tensor     # [batch, prompt_len + response_len]
    response_mask: torch.Tensor      # [batch, response_len] - 1 for response tokens
    prompt_length: torch.Tensor      # [batch]
    response_length: torch.Tensor    # [batch]


@dataclass
class TeacherOutput:
    """Output from teacher inference on student trajectories."""
    log_probs: torch.Tensor          # [batch, response_len] per-token log probs
    logits: Optional[torch.Tensor] = None  # [batch, response_len, vocab] (optional)
    entropy: Optional[torch.Tensor] = None  # [batch, response_len] (optional)


@dataclass
class OPDBatch:
    """Complete batch for OPD training step."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    student_log_probs: torch.Tensor
    teacher_log_probs: torch.Tensor
    ref_log_probs: Optional[torch.Tensor] = None
    student_logits: Optional[torch.Tensor] = None
    teacher_logits: Optional[torch.Tensor] = None
    teacher_entropy: Optional[torch.Tensor] = None


def extract_log_probs_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_start_idx: int,
) -> torch.Tensor:
    """
    Extract per-token log probs for the actual generated tokens.

    Args:
        logits: [batch, seq_len, vocab_size] model output logits
        input_ids: [batch, seq_len] token ids
        response_start_idx: where the response starts in input_ids

    Returns:
        log_probs: [batch, response_len] log prob of each response token
    """
    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, response_start_idx - 1:-1, :]
    shift_labels = input_ids[:, response_start_idx:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-position entropy from logits.

    H(p) = -Σ p(v) log p(v)

    Args:
        logits: [batch, seq_len, vocab_size]

    Returns:
        entropy: [batch, seq_len]
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


class StudentRolloutWorker:
    """
    Wraps verl's ActorRolloutWorker for student generation.

    In verl, this corresponds to the ActorRollout role using vLLM/SGLang
    for efficient batched generation.
    """

    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def generate(self, prompts: list[str]) -> RolloutOutput:
        """Generate trajectories from student model."""
        raise NotImplementedError("Use verl's rollout infrastructure")

    def compute_log_probs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute student log probs on existing sequences (for loss computation)."""
        raise NotImplementedError("Use verl's actor worker")


class TeacherInferenceWorker:
    """
    Wraps verl's TeacherModel worker for teacher inference.

    The teacher evaluates student-generated trajectories to provide
    token-level supervision (log_probs and optionally full logits).
    """

    def __init__(self, model, tokenizer, return_logits: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.return_logits = return_logits

    def compute_teacher_output(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor
    ) -> TeacherOutput:
        """Compute teacher log probs (and optionally logits) on student tokens."""
        raise NotImplementedError("Use verl's teacher model worker")


class ReferenceModelWorker:
    """
    Wraps verl's RefPolicy worker for reference model inference.

    The reference model provides the baseline for KL regularization.
    In standard OPD, this is typically the student's initial checkpoint.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compute_ref_log_probs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute reference model log probs."""
        raise NotImplementedError("Use verl's ref policy worker")
