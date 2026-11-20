"""Deterministic scoring. Exact=1, otherwise 0.2 per correct component."""
import datetime as dt
import re


def score(output: str, expected: str) -> float:
    output = output.strip()
    expected = expected.strip()
    if output == expected:
        return 1.0
    if expected == 'INVALID':
        return 0.0
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', output):
        return 0.0
    try:
        dt.date.fromisoformat(output)
    except ValueError:
        return 0.0
    return round(0.2 * sum(a == b for a, b in zip(output.split('-'), expected.split('-'))), 2)


def grpo_reward(completions, expected_answer, **kwargs):
    """TRL GRPOTrainer reward callable (string or conversational completions)."""
    results = []
    for completion, expected in zip(completions, expected_answer):
        if isinstance(completion, list):
            completion = completion[-1]['content'] if completion else ''
        results.append(score(str(completion), expected))
    return results
