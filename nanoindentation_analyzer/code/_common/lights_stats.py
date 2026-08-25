"""

``jonckheere_terpstra`` tests a monotone trend across ordered groups. With
twelve gels in three groups of four there are 34 650 distinct assignments.

``exact_ranksum`` is the two-sided Wilcoxon rank-sum over all C(n+m, m)
assignments. At 4 v 4 there are 70, and the floor of
0.029 is reached only when the two groups do not overlap at all. Every
comparison that reaches the floor is reported as being at it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "EXACT_ENUMERATION_MAX",
    "cliffs_delta",
    "exact_ranksum",
    "jonckheere_terpstra",
]

#: Above this many distinct assignments the trend test falls back to a
#: permutation p and says so in its notes.
EXACT_ENUMERATION_MAX = 200_000
DEFAULT_RESAMPLES = 20000
DEFAULT_SEED = 20260821


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """P(b > a) - P(b < a); +1 means every b exceeds every a."""
    x = np.asarray(a, dtype=float)[:, None]
    y = np.asarray(b, dtype=float)[None, :]
    return float(np.sign(y - x).mean())


def _multinomial(sizes: Sequence[int]) -> int:
    """Number of distinct ways to deal ``sum(sizes)`` items into the groups."""
    total = 1
    remaining = int(sum(sizes))
    for size in sizes:
        total *= math.comb(remaining, int(size))
        remaining -= int(size)
    return total


def _ordered_partitions(
    pooled: np.ndarray, sizes: Sequence[int]
) -> Iterator[list[np.ndarray]]:
    """Yield every distinct assignment of ``pooled`` into groups of ``sizes``.
    """
    indices = list(range(pooled.size))

    def recurse(
        available: list[int], depth: int, chosen: list[list[int]]
    ) -> Iterator[list[np.ndarray]]:
        if depth == len(sizes) - 1:
            yield [pooled[list(g)] for g in chosen] + [pooled[available]]
            return
        for combo in itertools.combinations(available, int(sizes[depth])):
            rest = [i for i in available if i not in set(combo)]
            yield from recurse(rest, depth + 1, [*chosen, list(combo)])

    yield from recurse(indices, 0, [])


def jonckheere_terpstra(
    groups: Sequence[Sequence[float]],
    *,
    alternative: str = "increasing",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Trend test for an ordered alternative across ordered groups.

    ``groups`` must already be in the hypothesised order.

    The p-value is exact when the design is small enough to enumerate every
    distinct assignment of observations to the ordered levels, and a
    permutation p otherwise; the large-sample normal approximation is
    reported alongside in both cases.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    arrays = [a for a in arrays if a.size > 0]
    if len(arrays) < 3:
        return {
            "test": "Jonckheere-Terpstra",
            "statistic": math.nan,
            "p_value": math.nan,
            "z": math.nan,
            "n": int(sum(a.size for a in arrays)),
            "exact": False,
            "notes": "fewer than three ordered levels",
        }

    def jt_statistic(sets: Sequence[np.ndarray]) -> float:
        total = 0.0
        for i in range(len(sets) - 1):
            for j in range(i + 1, len(sets)):
                a, b = sets[i], sets[j]
                comparison = np.sign(b[None, :] - a[:, None])
                total += float(
                    np.sum(comparison > 0) + 0.5 * np.sum(comparison == 0)
                )
        return total

    observed = jt_statistic(arrays)
    sizes = [a.size for a in arrays]
    pooled = np.concatenate(arrays)
    n = pooled.size

    # Large-sample normal approximation
    mean_jt = (n**2 - sum(s**2 for s in sizes)) / 4.0
    var_jt = (
        n**2 * (2 * n + 3) - sum(s**2 * (2 * s + 3) for s in sizes)
    ) / 72.0
    z = (observed - mean_jt) / np.sqrt(var_jt) if var_jt > 0 else math.nan

    def is_extreme(value: float) -> bool:
        if alternative == "increasing":
            return value >= observed
        if alternative == "decreasing":
            return value <= observed
        return abs(value - mean_jt) >= abs(observed - mean_jt)

    n_arrangements = _multinomial(sizes)
    if n_arrangements <= EXACT_ENUMERATION_MAX:
        count = sum(
            is_extreme(jt_statistic(split))
            for split in _ordered_partitions(pooled, sizes)
        )
        pvalue = count / n_arrangements
        note = (
            f"exact p over all {n_arrangements} distinct assignments; "
            "no resampling floor"
        )
        exact = True
    else:
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(resamples):
            shuffled = rng.permutation(pooled)
            split, start = [], 0
            for size in sizes:
                split.append(shuffled[start : start + size])
                start += size
            count += is_extreme(jt_statistic(split))
        pvalue = (count + 1) / (resamples + 1)
        note = (
            f"permutation p over {resamples} resamples; smallest attainable "
            f"p is {1 / (resamples + 1):.2e}"
        )
        exact = False

    return {
        "test": f"Jonckheere-Terpstra ({alternative})",
        "statistic": float(observed),
        "p_value": float(pvalue),
        "z": float(z),
        "n": int(n),
        "exact": exact,
        "notes": note,
    }


def exact_ranksum(a: Sequence[float], b: Sequence[float]) -> dict:
    """Exact two-sided Wilcoxon rank-sum by complete enumeration"""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n, m = x.size, y.size
    pooled = np.concatenate([x, y])
    ranks = stats.rankdata(pooled)
    observed = float(ranks[n:].sum())
    total = math.comb(n + m, m)
    counts = np.empty(total)
    for i, idx in enumerate(itertools.combinations(range(n + m), m)):
        counts[i] = ranks[list(idx)].sum()
    centre = m * (n + m + 1) / 2.0
    p_two = float(
        np.mean(np.abs(counts - centre) >= abs(observed - centre) - 1e-12)
    )
    return {
        "rank_sum_b": observed,
        "n_arrangements": int(total),
        "p_exact_two_sided": p_two,
        "p_floor": 2.0 / total,
        "at_floor": bool(abs(p_two - 2.0 / total) < 1e-12),
        "cliffs_delta": cliffs_delta(x, y),
    }
