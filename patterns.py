"""Classical exact pattern matching: KMP failure function and Z-array.

These are the reference structures we want to compare against CNN-learned weights.
"""

ALPHABET = "ABCD"
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
SIGMA = len(ALPHABET)


def kmp_failure(p: str) -> list[int]:
    """KMP failure function (a.k.a. sp table).

    sp[i] = length of the longest proper prefix of P[0..i] that is also a suffix.
    """
    m = len(p)
    sp = [0] * m
    k = 0
    for i in range(1, m):
        while k > 0 and p[k] != p[i]:
            k = sp[k - 1]
        if p[k] == p[i]:
            k += 1
        sp[i] = k
    return sp


def z_array(s: str) -> list[int]:
    """Z-array: z[i] = length of longest substring starting at i that matches a prefix of s."""
    n = len(s)
    z = [0] * n
    z[0] = n
    l, r = 0, 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z


def kmp_search(text: str, pattern: str) -> list[int]:
    """Return all start positions where pattern occurs in text."""
    sp = kmp_failure(pattern)
    matches = []
    k = 0
    for i, c in enumerate(text):
        while k > 0 and pattern[k] != c:
            k = sp[k - 1]
        if pattern[k] == c:
            k += 1
        if k == len(pattern):
            matches.append(i - len(pattern) + 1)
            k = sp[k - 1]
    return matches


if __name__ == "__main__":
    p = "ABABC"
    print(f"Pattern: {p}")
    print(f"  sp (KMP failure): {kmp_failure(p)}")
    print(f"  Z-array:          {z_array(p)}")
    text = "DABABABCDABABCDABAB"
    print(f"\nText: {text}")
    print(f"  matches at: {kmp_search(text, p)}")
