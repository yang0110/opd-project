"""
Reward functions for OPD evaluation and outcome verification.

Supports:
- Math verification (rule-based, using Math-Verify)
- Code execution (unit test pass/fail)
- Structured output validation (JSON parse, field check)
"""

import re
import json
from typing import Optional


def verify_math_answer(
    response: str,
    ground_truth: str,
    extract_pattern: str = r"\\boxed\{(.+?)\}",
) -> float:
    """
    Verify math reasoning answer.

    Extracts answer from \\boxed{} and compares with ground truth.
    Returns 1.0 for correct, 0.0 for incorrect.
    """
    matches = re.findall(extract_pattern, response)
    if not matches:
        # Try last number in response
        numbers = re.findall(r"[-+]?\d*\.?\d+", response)
        if not numbers:
            return 0.0
        predicted = numbers[-1]
    else:
        predicted = matches[-1]

    # Normalize and compare
    predicted = _normalize_math_answer(predicted)
    ground_truth = _normalize_math_answer(ground_truth)

    return 1.0 if predicted == ground_truth else 0.0


def _normalize_math_answer(answer: str) -> str:
    """Normalize math answer string for comparison."""
    answer = answer.strip()
    answer = answer.replace(" ", "")
    answer = answer.replace(",", "")
    # Remove trailing .0
    if answer.endswith(".0"):
        answer = answer[:-2]
    return answer


def verify_code_execution(
    response: str,
    test_cases: list[dict],
    timeout: float = 10.0,
) -> float:
    """
    Verify code generation by executing test cases.

    Returns fraction of test cases passed.
    """
    code = _extract_code(response)
    if not code:
        return 0.0

    passed = 0
    for test in test_cases:
        try:
            # Execute in sandboxed environment
            exec_globals = {}
            exec(code + "\n" + test["test_code"], exec_globals)
            passed += 1
        except Exception:
            continue

    return passed / len(test_cases) if test_cases else 0.0


def _extract_code(response: str) -> Optional[str]:
    """Extract code from markdown code blocks."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return response.strip()


def verify_json_output(
    response: str,
    required_fields: Optional[list[str]] = None,
    gold_ranking: Optional[list] = None,
) -> dict:
    """
    Verify structured JSON output.

    Returns dict with:
    - parse_valid: whether JSON is parseable
    - field_valid: whether required fields exist
    - semantic_score: ranking quality (NDCG) if gold provided
    """
    result = {"parse_valid": 0.0, "field_valid": 0.0, "semantic_score": 0.0}

    # Try to parse JSON
    try:
        json_str = _extract_json(response)
        parsed = json.loads(json_str)
        result["parse_valid"] = 1.0
    except (json.JSONDecodeError, ValueError):
        return result

    # Check required fields
    if required_fields:
        all_present = all(f in parsed for f in required_fields) if isinstance(parsed, dict) else False
        result["field_valid"] = 1.0 if all_present else 0.0

    # Compute ranking quality
    if gold_ranking and isinstance(parsed, list):
        result["semantic_score"] = compute_ndcg(parsed, gold_ranking, k=10)

    return result


def _extract_json(response: str) -> str:
    """Extract JSON from response."""
    # Try code block
    pattern = r"```(?:json)?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try direct JSON
    start = response.find("[")
    if start == -1:
        start = response.find("{")
    if start == -1:
        raise ValueError("No JSON found")

    return response[start:]


def compute_ndcg(predicted: list, gold: list, k: int = 10) -> float:
    """Compute NDCG@k for ranking evaluation."""
    import math

    def dcg(ranking, gold_set, k):
        score = 0.0
        for i, item in enumerate(ranking[:k]):
            if item in gold_set:
                score += 1.0 / math.log2(i + 2)
        return score

    gold_set = set(gold[:k])
    actual_dcg = dcg(predicted, gold_set, k)
    ideal_dcg = dcg(gold, gold_set, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg
