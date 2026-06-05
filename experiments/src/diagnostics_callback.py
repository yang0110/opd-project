"""
Diagnostics callback: hooks rethinking-opd monitoring into every training run.

Computes compatibility metrics at eval intervals to track:
- Token overlap between teacher and student
- Thinking pattern similarity (logprob correlation)
- Progressive alignment score
"""

import os
import sys
import torch
import json
import logging
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


def load_diagnostics_module():
    """Load rethinking-opd core_algos for diagnostic functions."""
    import importlib.util

    module_path = os.path.join(PROJECT_ROOT, "rethinking-opd", "src", "core_algos.py")
    spec = importlib.util.spec_from_file_location("rethinking_opd", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DiagnosticsCallback:
    """
    Callback that runs compatibility diagnostics during training.

    Attach to the trainer and call at eval intervals.
    """

    def __init__(self, output_dir: str, method_name: str):
        self.output_dir = output_dir
        self.method_name = method_name
        self.diagnostics_mod = load_diagnostics_module()
        self.history = []

    def on_eval_step(
        self,
        step: int,
        teacher_log_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        student_logits: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute diagnostics at an evaluation step.

        Returns dict of metrics to log (e.g., to WandB).
        """
        metrics = {}

        if teacher_logits is not None and student_logits is not None:
            overlap = self.diagnostics_mod.compute_shared_token_set(
                teacher_logits, student_logits
            )
            metrics["compat/top_k_overlap"] = overlap

            similarity = self.diagnostics_mod.compute_thinking_pattern_similarity(
                teacher_log_probs, student_log_probs
            )
            metrics["compat/logprob_correlation"] = similarity

        alignment = self.diagnostics_mod.monitor_progressive_alignment(
            teacher_log_probs, student_log_probs, response_mask
        )
        metrics["alignment/mean_kl"] = alignment

        record = {"step": step, **metrics}
        self.history.append(record)
        self._save()

        logger.info(f"[{self.method_name}] Step {step} diagnostics: {metrics}")
        return metrics

    def predict_success(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> dict:
        """Run pre-training compatibility prediction."""
        prediction = self.diagnostics_mod.predict_opd_success(
            teacher_logits, student_logits
        )
        logger.info(f"[{self.method_name}] Compatibility prediction: {prediction}")
        return prediction

    def _save(self):
        """Save diagnostics history to file."""
        path = os.path.join(self.output_dir, "diagnostics.json")
        with open(path, "w") as f:
            json.dump(
                {"method": self.method_name, "history": self.history},
                f,
                indent=2,
            )
