# OPD Datasets Module

## What This Is

Shared data loading, preprocessing, and reward verification for all OPD methods. Provides unified dataset classes compatible with verl's DataProto format.

## Supported Task Types

### Math Reasoning
- **Training data**: DeepMath (filtered difficulty ≥ 6, ~57K samples)
- **Evaluation**: AIME24, AIME25, HMMT25 (Feb), HMMT25 (Nov)
- **Verification**: Rule-based answer extraction from `\boxed{}` format, via Math-Verify
- **Sampling**: 32 solutions per problem, temperature=1.0, top-p=1.0, max 16384 tokens

### Code Generation
- **Training data**: Eurus-RL-Code (~25K samples)
- **Evaluation**: HumanEval+, MBPP+, LiveCodeBench
- **Verification**: Unit test execution (4 solutions per problem)

### Structured Output / Ranking
- **Training data**: Listwise ranking datasets (e.g., Amazon Fashion)
- **Verification**: JSON parse validity + NDCG@K

## Files

```
src/
├── __init__.py
├── data_loader.py      # OPDDataset, MathReasoningDataset, CodeGenerationDataset,
│                       # ListwiseRankingDataset, create_opd_dataloader
└── reward_functions.py # verify_math_answer, verify_code_execution,
                        # verify_json_output, compute_ndcg
```

## Usage

```python
from datasets.src.data_loader import MathReasoningDataset, DataConfig, create_opd_dataloader

config = DataConfig(
    train_files=["path/to/deepmath.jsonl"],
    max_prompt_length=1024,
    max_response_length=16384,
    min_difficulty=6,
)

dataset = MathReasoningDataset("path/to/deepmath.jsonl", config)
dataloader = create_opd_dataloader(dataset, batch_size=128, shuffle=True)

for batch in dataloader:
    prompts = batch["prompts"]  # List of chat-format prompt dicts
    # Feed to verl's rollout worker
```

## Data Format

Each dataset item follows verl's expected format:
```json
{
    "prompt": [{"role": "user", "content": "Solve: ..."}],
    "reward_model": {"ground_truth": "42"}
}
```

## Reward Functions

```python
from datasets.src.reward_functions import verify_math_answer, verify_code_execution

# Math verification
score = verify_math_answer(response="... \\boxed{42}", ground_truth="42")
# Returns 1.0 or 0.0

# Code verification
score = verify_code_execution(response="```python\ndef f(x): ...\n```",
                              test_cases=[{"test_code": "assert f(1) == 2"}])
# Returns fraction of tests passed

# Structured output
result = verify_json_output(response, required_fields=["id", "rank"])
# Returns {"parse_valid": 1.0, "field_valid": 1.0, "semantic_score": 0.85}
```
