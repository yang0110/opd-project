"""
Method registry: maps method names to their loss functions and configurations.

Each method's implementation lives in its own directory under the project root.
This registry provides a unified interface to load any method by name.
"""

import importlib.util
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHODS = {
    "standard_opd": {
        "module_dir": "standard-opd",
        "loss_fn": "compute_opd_loss",
        "advantage_fn": "compute_opd_advantage",
        "needs_logits": False,
        "needs_outcome_reward": False,
        "kwargs": {},
    },
    "gopd_interp": {
        "module_dir": "g-opd",
        "loss_fn": "compute_gopd_loss",
        "advantage_fn": "compute_gopd_advantage",
        "needs_logits": False,
        "needs_outcome_reward": False,
        "kwargs": {"lambda_": 0.75},
    },
    "gopd_extrap": {
        "module_dir": "g-opd",
        "loss_fn": "compute_gopd_loss",
        "advantage_fn": "compute_gopd_advantage",
        "needs_logits": False,
        "needs_outcome_reward": False,
        "kwargs": {"lambda_": 1.25},
    },
    "gopd_correction": {
        "module_dir": "g-opd",
        "loss_fn": "compute_gopd_loss_with_reward_correction",
        "advantage_fn": "compute_gopd_advantage",
        "needs_logits": False,
        "needs_outcome_reward": False,
        "kwargs": {"lambda_": 1.25},
    },
    "veto_logit": {
        "module_dir": "veto",
        "loss_fn": "compute_veto_loss",
        "advantage_fn": None,
        "needs_logits": True,
        "needs_outcome_reward": False,
        "kwargs": {"beta_init": 1.0, "decay_type": "linear"},
    },
    "veto_token": {
        "module_dir": "veto",
        "loss_fn": "compute_veto_loss_token_level",
        "advantage_fn": None,
        "needs_logits": False,
        "needs_outcome_reward": False,
        "kwargs": {"beta_init": 1.0, "decay_type": "linear"},
    },
    "entropy_aware": {
        "module_dir": "entropy-aware-opd",
        "loss_fn": "compute_entropy_aware_opd_loss",
        "advantage_fn": None,
        "needs_logits": True,
        "needs_outcome_reward": False,
        "kwargs": {"tau": 0.8, "alpha": 1.0, "top_k": 16},
    },
    "aopd": {
        "module_dir": "aopd",
        "loss_fn": "compute_aopd_loss",
        "advantage_fn": None,
        "needs_logits": True,
        "needs_outcome_reward": False,
        "kwargs": {"tau": 0.0, "top_k": 10, "exploit_coef": 1.0, "imitate_coef": 1.0},
    },
    "aopd_clipped": {
        "module_dir": "aopd",
        "loss_fn": "compute_aopd_loss_with_clipping",
        "advantage_fn": None,
        "needs_logits": True,
        "needs_outcome_reward": False,
        "kwargs": {"tau": 0.0, "clip_range": 0.2},
    },
    "extrapolation_cliff": {
        "module_dir": "extrapolation-cliff",
        "loss_fn": "compute_safe_exopd_loss",
        "advantage_fn": None,
        "needs_logits": True,
        "needs_outcome_reward": False,
        "kwargs": {"base_lambda": 1.25, "clip_strength": 0.2, "use_adaptive_lambda": True},
    },
    "caopd": {
        "module_dir": "caopd",
        "loss_fn": "compute_caopd_loss",
        "advantage_fn": None,
        "needs_logits": False,
        "needs_outcome_reward": True,
        "kwargs": {"capability_coef": 1.0, "calibration_coef": 1.0},
        "extra": {"num_rollouts": 8},
    },
    "uni_opd": {
        "module_dir": "uni-opd",
        "loss_fn": "compute_uni_opd_loss",
        "advantage_fn": None,
        "needs_logits": False,
        "needs_outcome_reward": True,
        "kwargs": {"margin_scale": 1.0, "kl_coef": 0.1},
        "extra": {"num_generations": 4},
    },
    "aligndistil": {
        "module_dir": "aligndistil",
        "loss_fn": "compute_aligndistil_loss",
        "advantage_fn": None,
        "needs_logits": False,
        "needs_outcome_reward": False,
        "kwargs": {"beta": 0.1, "adaptive": True},
    },
    "coverage_opd": {
        "module_dir": "on-policy-preference",
        "loss_fn": "compute_opd_with_coverage_aware_sampling",
        "advantage_fn": None,
        "needs_logits": False,
        "needs_outcome_reward": False,
        "kwargs": {"beta": 0.1},
    },
}

METHOD_ORDER = [
    "standard_opd",
    "gopd_interp",
    "gopd_extrap",
    "gopd_correction",
    "veto_logit",
    "veto_token",
    "entropy_aware",
    "aopd",
    "aopd_clipped",
    "extrapolation_cliff",
    "caopd",
    "uni_opd",
    "aligndistil",
    "coverage_opd",
]


def load_method_module(method_name: str):
    """Dynamically load the core_algos module for a given method."""
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHODS.keys())}")

    info = METHODS[method_name]
    module_dir = info["module_dir"]
    module_path = os.path.join(PROJECT_ROOT, module_dir, "src", "core_algos.py")

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"{module_dir}.core_algos", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_loss_fn(method_name: str):
    """Get the loss function for a method, with default kwargs bound."""
    info = METHODS[method_name]
    mod = load_method_module(method_name)
    fn = getattr(mod, info["loss_fn"])
    return fn, info["kwargs"]


def get_advantage_fn(method_name: str):
    """Get the advantage function for a method (if it has one)."""
    info = METHODS[method_name]
    if info["advantage_fn"] is None:
        return None
    mod = load_method_module(method_name)
    return getattr(mod, info["advantage_fn"])


def get_method_info(method_name: str) -> dict:
    """Get full method configuration info."""
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}")
    return METHODS[method_name]


def list_methods() -> list[str]:
    """Return all method names in recommended execution order."""
    return METHOD_ORDER.copy()
