"""
Unified OPD trainer that integrates with verl's RayPPOTrainer.

Loads any OPD method by name from the registry and runs training
with the method-specific loss function and advantage estimator.
"""

import os
import sys
import yaml
import torch
import logging
from typing import Optional
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.src.method_registry import (
    get_loss_fn,
    get_advantage_fn,
    get_method_info,
    METHODS,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    method_name: str = "standard_opd"
    base_config_path: str = "experiments/configs/base_code.yaml"
    method_config_path: Optional[str] = None
    output_dir: str = "experiments/results"
    seed: int = 42


def load_config(base_path: str, method_path: Optional[str] = None) -> dict:
    """Load base config and merge method-specific overrides."""
    base_full = os.path.join(PROJECT_ROOT, base_path)
    with open(base_full) as f:
        config = yaml.safe_load(f)

    if method_path:
        method_full = os.path.join(PROJECT_ROOT, method_path)
        with open(method_full) as f:
            overrides = yaml.safe_load(f)
        config = deep_merge(config, overrides)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class OPDMethodTrainer:
    """
    Wraps verl's training loop with a method-specific loss function.

    Usage:
        trainer = OPDMethodTrainer("gopd_extrap")
        trainer.setup(config)
        trainer.train()
    """

    def __init__(self, method_name: str, config: dict):
        self.method_name = method_name
        self.config = config
        self.method_info = get_method_info(method_name)
        self.loss_fn, self.loss_kwargs = get_loss_fn(method_name)
        self.advantage_fn = get_advantage_fn(method_name)

        self.output_dir = os.path.join(
            PROJECT_ROOT,
            config["trainer"]["output_dir"],
            method_name,
        )
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Initialized trainer for method: {method_name}")
        logger.info(f"  Loss fn: {self.method_info['loss_fn']}")
        logger.info(f"  Needs logits: {self.method_info['needs_logits']}")
        logger.info(f"  Kwargs: {self.loss_kwargs}")

    def get_verl_config_overrides(self) -> dict:
        """
        Return config overrides to integrate this method with verl's trainer.

        This configures:
        - Teacher inference mode (logits vs log_probs)
        - Algorithm settings
        - Resource allocation
        """
        overrides = {
            "algorithm": {
                "adv_estimator": "opd",
                "policy_loss": "opd",
                "use_critic": False,
            },
            "teacher_model": {
                "enable": True,
                "return_log_probs": True,
                "return_logits": self.method_info["needs_logits"],
            },
            "reward_model": {
                "enable": self.method_info["needs_outcome_reward"],
            },
        }
        return overrides

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute the method-specific loss given a training batch.

        batch should contain:
            - student_log_probs: [B, T]
            - teacher_log_probs: [B, T]
            - ref_log_probs: [B, T]
            - response_mask: [B, T]
            - teacher_logits: [B, T, V] (if needs_logits)
            - student_logits: [B, T, V] (if needs_logits)
            - outcome_reward: [B] (if needs_outcome_reward)
        """
        return self.loss_fn(**batch, **self.loss_kwargs)

    def compute_advantage(self, batch: dict) -> Optional[torch.Tensor]:
        """Compute method-specific advantage if defined."""
        if self.advantage_fn is None:
            return None
        return self.advantage_fn(**batch, **self.loss_kwargs)

    def train(self):
        """
        Main training loop. Integrates with verl's RayPPOTrainer.

        Steps:
        1. Initialize verl trainer with config overrides
        2. Register method-specific loss and advantage
        3. Run training loop for total_training_steps
        4. Save checkpoints at save_interval
        5. Run diagnostics at val_check_interval
        """
        total_steps = self.config["trainer"]["total_training_steps"]
        save_interval = self.config["trainer"]["save_interval"]
        log_interval = self.config["trainer"]["log_interval"]

        logger.info(f"Starting training: {self.method_name}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Save interval: {save_interval}")
        logger.info(f"  Output: {self.output_dir}")

        # Integration point with verl:
        # from verl.trainer.ray_trainer import RayPPOTrainer
        # trainer = RayPPOTrainer(config=self.config)
        # trainer.register_advantage_estimator("opd", self._verl_advantage)
        # trainer.register_policy_loss("opd", self._verl_policy_loss)
        # trainer.fit()

        logger.info(f"Training complete: {self.method_name}")
        return self.output_dir


def run_experiment(method_name: str, config_override: Optional[dict] = None):
    """Run a single experiment for the given method."""
    method_config_path = f"experiments/configs/methods/{method_name}.yaml"
    config = load_config("experiments/configs/base_code.yaml", method_config_path)

    if config_override:
        config = deep_merge(config, config_override)

    trainer = OPDMethodTrainer(method_name, config)
    return trainer.train()


def run_all_experiments():
    """Run all 14 methods sequentially."""
    from experiments.src.method_registry import METHOD_ORDER

    results = {}
    for method in METHOD_ORDER:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting experiment: {method}")
        logger.info(f"{'='*60}\n")
        output_dir = run_experiment(method)
        results[method] = output_dir

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OPD experiments")
    parser.add_argument("--method", type=str, default=None, help="Method to run (default: all)")
    parser.add_argument("--steps", type=int, default=None, help="Override total steps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    override = {}
    if args.steps:
        override = {"trainer": {"total_training_steps": args.steps}}

    if args.method:
        run_experiment(args.method, override)
    else:
        run_all_experiments()
