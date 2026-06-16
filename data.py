"""Dataset generator: texts that contain complete pattern matches AND deliberate
partial-prefix near-misses. The near-misses are the cases where KMP's failure
function makes a difference in classical pattern matching.

A "partial token" of length k+1 is: pattern[:k] followed by a single character
that is NOT pattern[k] (so this prefix cannot be extended to a full match
without backtracking).
"""

import random
from dataclasses import dataclass

import numpy as np

from patterns import ALPHABET, CHAR_TO_IDX, SIGMA, kmp_search


def random_partial_token(pattern: str, k: int, rng: random.Random) -> str:
    """Prefix pattern[:k] followed by a char that is NOT pattern[k]."""
    assert 1 <= k < len(pattern)
    forbidden = pattern[k]
    choices = [c for c in ALPHABET if c != forbidden]
    return pattern[:k] + rng.choice(choices)


def generate_sample(
    pattern: str,
    target_length: int,
    n_complete: int,
    n_partial: int,
    rng: random.Random,
) -> tuple[str, np.ndarray]:
    """Generate one (text, labels) pair.

    labels[i] = 1 if pattern matches text starting at position i, else 0.
    Labels are length T - m + 1 (positions where a kernel of size m can start).
    """
    m = len(pattern)
    tokens: list[str] = [pattern] * n_complete
    for _ in range(n_partial):
        k = rng.randint(1, m - 1)
        tokens.append(random_partial_token(pattern, k, rng))

    base_len = sum(len(t) for t in tokens)
    if base_len > target_length:
        raise ValueError(
            f"Token block ({base_len}) exceeds target_length ({target_length}); "
            f"reduce n_complete/n_partial or increase target_length."
        )

    # Fill the remaining slots with single random characters, then shuffle the
    # whole token list so matches are not clustered at the start.
    n_filler = target_length - base_len
    tokens.extend(rng.choice(ALPHABET) for _ in range(n_filler))
    rng.shuffle(tokens)
    text = "".join(tokens)
    assert len(text) == target_length

    matches = kmp_search(text, pattern)
    n_pos = target_length - m + 1
    labels = np.zeros(n_pos, dtype=np.float32)
    for p in matches:
        if p < n_pos:
            labels[p] = 1.0
    return text, labels


def one_hot_encode(text: str) -> np.ndarray:
    """[T, SIGMA] one-hot encoding."""
    arr = np.zeros((len(text), SIGMA), dtype=np.float32)
    for i, c in enumerate(text):
        arr[i, CHAR_TO_IDX[c]] = 1.0
    return arr


@dataclass
class Dataset:
    x: np.ndarray  # [N, T, SIGMA]
    y: np.ndarray  # [N, T - m + 1]
    texts: list[str]


def build_dataset(
    pattern: str,
    n_samples: int,
    target_length: int,
    n_complete: int,
    n_partial: int,
    seed: int,
) -> Dataset:
    rng = random.Random(seed)
    texts: list[str] = []
    labels_list: list[np.ndarray] = []
    for _ in range(n_samples):
        t, y = generate_sample(pattern, target_length, n_complete, n_partial, rng)
        texts.append(t)
        labels_list.append(y)
    x = np.stack([one_hot_encode(t) for t in texts], axis=0)
    y = np.stack(labels_list, axis=0)
    return Dataset(x=x, y=y, texts=texts)


if __name__ == "__main__":
    rng = random.Random(0)
    text, labels = generate_sample("ABABC", 50, n_complete=2, n_partial=4, rng=rng)
    print(f"text   ({len(text)} chars): {text}")
    print(f"labels ({len(labels)}):     {''.join(str(int(v)) for v in labels)}")
    print(f"positive rate: {labels.sum() / len(labels):.3f}")

    ds = build_dataset("ABABC", n_samples=8, target_length=50, n_complete=2, n_partial=4, seed=0)
    print(f"\nDataset: x.shape={ds.x.shape}, y.shape={ds.y.shape}")
    print(f"avg matches per sample: {ds.y.sum(axis=1).mean():.2f}")
