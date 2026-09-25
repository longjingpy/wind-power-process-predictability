"""Portable copy of the verified v5 SDA and 2015 conference OpSDA kernels.

Florita et al., doi:10.1109/GreenTech.2013.30, pp.2-3: parallel door hinges,
closing at the violating observation. Cui et al.,
doi:10.1109/PESGM.2015.7286272, p.2 Eqs.1-3: dynamic programming with squared
interval-length reward and a same-direction merging rule. Overlapping windows
and deterministic ties follow this study's declared implementation.
"""
import numpy as np


def original_sda(x, epsilon):
    x = np.asarray(x, float)
    if not np.isfinite(x).all() or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Finite signal and positive epsilon required")
    if len(x) < 2:
        return []
    start, lower, upper = 0, -np.inf, np.inf
    segments = []
    for end in range(1, len(x)):
        lower = max(lower, (x[end]-x[start]-epsilon)/(end-start))
        upper = min(upper, (x[end]-x[start]+epsilon)/(end-start))
        if lower >= upper-1e-12:
            segments.append((start, end))
            start, lower, upper = end, -np.inf, np.inf
    if start < len(x)-1:
        segments.append((start, len(x)-1))
    return segments


def optimize_window(x, knots, amplitude=.2):
    x, knots = np.asarray(x, float), np.asarray(knots, int)
    if len(knots) < 2:
        return [], 0.
    if (np.diff(knots) <= 0).any() or knots[0] < 0 or knots[-1] >= len(x):
        raise ValueError("Ordered in-range SDA endpoints required")
    n = len(knots)
    best, previous, chosen = np.zeros(n), np.full(n, -1, int), np.zeros(n, bool)
    for j in range(1, n):
        best[j], previous[j] = best[j-1], j-1
        for i in range(j):
            changes = np.diff(x[knots[i:j+1]])
            monotone = (changes >= 0).all() or (changes <= 0).all()
            if monotone and abs(x[knots[j]]-x[knots[i]]) >= amplitude:
                candidate = best[i]+float((knots[j]-knots[i])**2)
                if candidate > best[j]+1e-12:
                    best[j], previous[j], chosen[j] = candidate, i, True
    intervals, j = [], n-1
    while j > 0:
        i = previous[j]
        if chosen[j]:
            intervals.append((int(knots[i]), int(knots[j])))
        j = i
    return intervals[::-1], float(best[-1])


def opsda_2015(x, epsilon, horizon_steps, amplitude=.2):
    if horizon_steps < 1:
        raise ValueError("Positive horizon required")
    segments = original_sda(x, epsilon)
    if not segments:
        return []
    knots = np.array([segments[0][0]]+[b for _, b in segments])
    intervals = set()
    for i, start in enumerate(knots[:-1]):
        stop = np.searchsorted(knots, start+horizon_steps, side="right")
        ramps, _ = optimize_window(x, knots[i:stop], amplitude)
        intervals.update(ramps)
    return sorted(intervals)
