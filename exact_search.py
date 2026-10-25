"""Instrumented exact baselines: count actual character equality tests.

Preprocessing, indexing, additions and branch costs are not counted. The
bad-character-only variant is intentionally not full Boyer--Moore.
Nonempty patterns are required; overlapping matches are always reported.
"""

from patterns import kmp_failure


def naive_search(text: str, pattern: str) -> tuple[list[int], int]:
    if not pattern:
        raise ValueError("Pattern must be nonempty")
    matches, comparisons = [], 0
    for start in range(len(text) - len(pattern) + 1):
        for i, c in enumerate(pattern):
            comparisons += 1
            if text[start + i] != c:
                break
        else:
            matches.append(start)
    return matches, comparisons


def kmp_counted(text: str, pattern: str) -> tuple[list[int], int]:
    if not pattern:
        raise ValueError("Pattern must be nonempty")
    failure = kmp_failure(pattern)
    matches, comparisons, matched = [], 0, 0
    for i, c in enumerate(text):
        while True:
            comparisons += 1
            if pattern[matched] == c:
                matched += 1
                break
            if matched == 0:
                break
            matched = failure[matched - 1]
        if matched == len(pattern):
            matches.append(i - len(pattern) + 1)
            matched = failure[matched - 1]
    return matches, comparisons


def bad_character_search(text: str, pattern: str) -> tuple[list[int], int]:
    """Right-to-left comparisons, global last occurrence, no good-suffix rule.

    Shift one after a full match to preserve overlaps. No linear worst-case
    bound is claimed for this simplified baseline.
    """
    if not pattern:
        raise ValueError("Pattern must be nonempty")
    last = {c: i for i, c in enumerate(pattern)}
    matches, comparisons, start = [], 0, 0
    while start + len(pattern) <= len(text):
        i = len(pattern) - 1
        while i >= 0:
            comparisons += 1
            if pattern[i] != text[start + i]:
                break
            i -= 1
        if i < 0:
            matches.append(start)
            start += 1
        else:
            start += max(1, i - last.get(text[start + i], -1))
    return matches, comparisons
