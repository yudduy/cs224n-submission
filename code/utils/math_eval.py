"""Answer extraction and equivalence checking for MATH-500 evaluation."""
from __future__ import annotations

import re
from typing import Optional


def extract_boxed(text: str) -> Optional[str]:
    """Extract last \\boxed{...} content, handling nested braces."""
    i = text.rfind("\\boxed{")
    if i == -1:
        return None
    depth, start = 0, i + 7
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            if depth == 0:
                return text[start:j].strip()
            depth -= 1
    return None


def extract_clean_answer(text: str) -> Optional[str]:
    """Extract answer: try \\boxed{} first, then last number."""
    ans = extract_boxed(text)
    if ans is not None:
        return ans
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def answers_match(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    a_clean = re.sub(r"\s+", "", a.lower().strip())
    b_clean = re.sub(r"\s+", "", b.lower().strip())
    if a_clean == b_clean:
        return True
    try:
        return abs(float(a_clean) - float(b_clean)) < 1e-6
    except (ValueError, OverflowError):
        return False
