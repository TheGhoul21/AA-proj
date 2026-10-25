"""Unit tests for classical string algorithms in patterns.py:
- KMP failure function (sp table)
- Z-array
- KMP exact search
"""

import random
import pytest

from patterns import kmp_failure, z_array, kmp_search, ALPHABET


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("ABABC", [0, 0, 1, 2, 0]),
        ("ABABA", [0, 0, 1, 2, 3]),
        ("AAAA", [0, 1, 2, 3]),
        ("A", [0]),
        ("ABCD", [0, 0, 0, 0]),
        ("AA", [0, 1]),
        ("AB", [0, 0]),
        ("AABAACAABAA", [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]),
    ],
)
def test_kmp_failure_known_and_edge_cases(pattern: str, expected: list[int]) -> None:
    assert kmp_failure(pattern) == expected


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("ABABC", [5, 0, 2, 0, 0]),
        ("ABABA", [5, 0, 3, 0, 1]),
        ("AAAA", [4, 3, 2, 1]),
        ("A", [1]),
        ("ABCD", [4, 0, 0, 0]),
        ("AA", [2, 1]),
        ("AB", [2, 0]),
    ],
)
def test_z_array_known_and_edge_cases(pattern: str, expected: list[int]) -> None:
    assert z_array(pattern) == expected


def _naive_search(text: str, pattern: str) -> list[int]:
    """Reference implementation using Python's native string slicing."""
    m = len(pattern)
    if m == 0 or len(text) < m:
        return []
    return [i for i in range(len(text) - m + 1) if text[i : i + m] == pattern]


@pytest.mark.parametrize(
    "text,pattern",
    [
        ("DABABABCDABABCDABAB", "ABABC"),
        ("AAAAAA", "AAAA"),
        ("DDDDDD", "ABABC"),
        ("ABABCXYZ", "ABABC"),
        ("XYZABABC", "ABABC"),
        ("ABABC", "ABABC"),
        ("ABC", "ABCD"),
        ("ABABABABAB", "ABA"),
        ("BACDABCDABCAABCD", "ABCD"),
    ],
)
def test_kmp_search_matches_reference(text: str, pattern: str) -> None:
    expected = _naive_search(text, pattern)
    actual = kmp_search(text, pattern)
    assert actual == expected


def test_kmp_search_random_texts() -> None:
    rng = random.Random(42)
    patterns = ["A", "AB", "ABA", "ABABC", "AAAA", "ABCD", "BCDA"]

    for pattern in patterns:
        for text_len in [10, 50, 200, 1000]:
            text = "".join(rng.choice(ALPHABET) for _ in range(text_len))
            expected = _naive_search(text, pattern)
            actual = kmp_search(text, pattern)
            assert actual == expected, (
                f"Mismatch for pattern={pattern!r} on text length {text_len}"
            )
