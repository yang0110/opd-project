"""
Dataset loading and preprocessing for OPD training.

Supports:
- Math reasoning datasets (DeepMath, GSM8K, MATH, AIME)
- Code generation datasets (Eurus-RL-Code, HumanEval, MBPP)
- General instruction following datasets

Following verl's dataset format:
- Each sample has a 'prompt' field (list of messages or string)
- Optional 'reward_model' field for outcome verification
- Supports parquet/jsonl formats
"""

import json
import os
from typing import Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataConfig:
    train_files: list[str] = field(default_factory=list)
    val_files: list[str] = field(default_factory=list)
    max_prompt_length: int = 1024
    max_response_length: int = 16384
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    difficulty_key: str = "difficulty"
    min_difficulty: int = 0
    max_samples: Optional[int] = None
    seed: int = 42


class OPDDataset(Dataset):
    """
    Dataset for OPD training.

    Each item returns a prompt dict compatible with verl's DataProto format.

    Format: {"prompt": [{"role": "user", "content": "..."}], "extra": {...}}
    """

    def __init__(self, data_path: str, config: DataConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> list[dict]:
        if data_path.endswith(".jsonl"):
            return self._load_jsonl(data_path)
        elif data_path.endswith(".json"):
            with open(data_path) as f:
                return json.load(f)
        elif data_path.endswith(".parquet"):
            return self._load_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

    def _load_jsonl(self, path: str) -> list[dict]:
        data = []
        with open(path) as f:
            for line in f:
                item = json.loads(line.strip())
                if self._filter_item(item):
                    data.append(self._format_item(item))
        if self.config.max_samples:
            data = data[:self.config.max_samples]
        return data

    def _load_parquet(self, path: str) -> list[dict]:
        import pandas as pd
        df = pd.read_parquet(path)
        data = []
        for _, row in df.iterrows():
            item = row.to_dict()
            if self._filter_item(item):
                data.append(self._format_item(item))
        if self.config.max_samples:
            data = data[:self.config.max_samples]
        return data

    def _filter_item(self, item: dict) -> bool:
        if self.config.difficulty_key in item:
            diff = item[self.config.difficulty_key]
            if diff < self.config.min_difficulty:
                return False
        return True

    def _format_item(self, item: dict) -> dict:
        """Format item into verl-compatible prompt format."""
        prompt = item[self.config.prompt_key]

        # If prompt is a string, wrap in chat format
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        result = {
            "prompt": prompt,
        }

        # Add answer for verification
        if self.config.answer_key in item:
            result["reward_model"] = {
                "ground_truth": item[self.config.answer_key]
            }

        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MathReasoningDataset(OPDDataset):
    """
    Math reasoning dataset following G-OPD paper settings.

    Uses DeepMath with difficulty >= 6 (57K samples).
    Evaluation: AIME24, AIME25, HMMT25(Feb), HMMT25(Nov)
    """

    def __init__(self, data_path: str, config: DataConfig, tokenizer=None):
        config.min_difficulty = 6
        super().__init__(data_path, config, tokenizer)

    def _format_item(self, item: dict) -> dict:
        problem = item.get("problem", item.get(self.config.prompt_key, ""))
        prompt = [{"role": "user", "content": problem}]

        result = {"prompt": prompt}
        if "answer" in item:
            result["reward_model"] = {"ground_truth": str(item["answer"])}
        return result


class CodeGenerationDataset(OPDDataset):
    """
    Code generation dataset following G-OPD paper settings.

    Uses Eurus-RL-Code (25K samples).
    Evaluation: HumanEval+, MBPP+, LiveCodeBench
    """

    def _format_item(self, item: dict) -> dict:
        problem = item.get("problem", item.get(self.config.prompt_key, ""))
        prompt = [{"role": "user", "content": problem}]

        result = {"prompt": prompt}
        if "test_cases" in item:
            result["reward_model"] = {"test_cases": item["test_cases"]}
        return result


class ListwiseRankingDataset(OPDDataset):
    """
    Listwise ranking dataset for structured output tasks.

    Used by Extrapolation Cliff paper (Amazon Fashion ranking).
    Output format: JSON with ranked item IDs.
    """

    def _format_item(self, item: dict) -> dict:
        query = item.get("query", "")
        items = item.get("items", [])

        instruction = (
            f"Rank the following items by relevance to the query.\n"
            f"Query: {query}\n"
            f"Items: {json.dumps(items)}\n"
            f"Return a JSON array of item IDs in order of relevance."
        )
        prompt = [{"role": "user", "content": instruction}]

        result = {"prompt": prompt}
        if "gold_ranking" in item:
            result["reward_model"] = {"gold_ranking": item["gold_ranking"]}
        return result


def create_opd_dataloader(
    dataset: OPDDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader compatible with verl's training loop."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_opd_batch,
        generator=generator,
    )


def collate_opd_batch(batch: list[dict]) -> dict:
    """
    Collate function for OPD batches.

    Returns a dict with:
    - prompts: list of prompt dicts
    - reward_model: list of verification info (if available)
    """
    prompts = [item["prompt"] for item in batch]
    result = {"prompts": prompts}

    if "reward_model" in batch[0]:
        result["reward_model"] = [item.get("reward_model") for item in batch]

    return result
