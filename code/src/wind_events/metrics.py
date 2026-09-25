"""Partition agreement and shared-calendar-block uncertainty.

ARI: Hubert and Arabie (1985), doi:10.1007/BF01908075, adjusted index.
Calendar block resampling follows the dependent-data bootstrap principle of
Kunsch (1989), doi:10.1214/aos/1176347265, with nonoverlapping calendar blocks
declared for this study. All turbines in one site share each block weight.
"""
import numpy as np
import pandas as pd


def contingency(a, b, k):
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(a) != len(b) or (a < 0).any() or (b < 0).any() or (a >= k).any() or (b >= k).any():
        raise ValueError("Equal nonnegative in-range cluster labels required")
    return np.bincount(a*k+b, minlength=k*k).reshape(k, k)


def scores_from_counts(counts):
    c = np.asarray(counts, float)
    if c.ndim == 2:
        c = c[None]
    n = c.sum(axis=(1, 2))
    row, col = c.sum(axis=2), c.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = c/n[:, None, None]
        pr, pc = row/n[:, None], col/n[:, None]
        mi = np.where(p > 0, p*np.log(p/(pr[:, :, None]*pc[:, None, :])), 0).sum(axis=(1, 2))
        hr = -np.where(pr > 0, pr*np.log(pr), 0).sum(axis=1)
        hc = -np.where(pc > 0, pc*np.log(pc), 0).sum(axis=1)
        nmi = 2*mi/(hr+hc)
        cells = (c*(c-1)/2).sum(axis=(1, 2))
        rr, cc = (row*(row-1)/2).sum(axis=1), (col*(col-1)/2).sum(axis=1)
        expected = rr*cc/(n*(n-1)/2)
        ari = (cells-expected)/(.5*(rr+cc)-expected)
        agreement = np.trace(c, axis1=1, axis2=2)/n
    informative = (row > 0).sum(axis=1) > 1
    informative &= (col > 0).sum(axis=1) > 1
    both_constant = ((row > 0).sum(axis=1) <= 1) & ((col > 0).sum(axis=1) <= 1)
    nmi[both_constant | (n < 2)] = np.nan
    ari[both_constant | (n < 2)] = np.nan
    return nmi, ari, agreement, informative


def block_intervals(a, b, times, k, days, repetitions=2000, seed=41):
    stamp = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns")
    blocks = stamp.asi8//pd.Timedelta(days=days).value
    unique, inverse = np.unique(blocks, return_inverse=True)
    counts = np.zeros((len(unique), k*k), np.int64)
    np.add.at(counts, (inverse, np.asarray(a, int)*k+np.asarray(b, int)), 1)
    if len(unique) < 2:
        return {"occupied_blocks": len(unique), "nmi_low": None, "nmi_high": None, "ari_low": None, "ari_high": None}
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=repetitions)
    sample = (weights@counts).reshape(repetitions, k, k)
    nmi, ari, _, _ = scores_from_counts(sample)
    result = {"occupied_blocks": len(unique), "bootstrap_valid_nmi": int(np.isfinite(nmi).sum()), "bootstrap_valid_ari": int(np.isfinite(ari).sum())}
    for name, values in (("nmi", nmi), ("ari", ari)):
        finite = values[np.isfinite(values)]
        low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
        result[name+"_low"], result[name+"_high"] = float(low), float(high)
    return result


def support_components(starts, ends, turbines):
    table = pd.DataFrame({"start": pd.to_datetime(starts, utc=True), "end": pd.to_datetime(ends, utc=True), "turbine": turbines})
    total = 0
    for _, group in table.groupby("turbine"):
        latest = None
        for start, end in group.sort_values(["start", "end"])[["start", "end"]].itertuples(index=False, name=None):
            if latest is None or start > latest:
                total += 1
            latest = end if latest is None else max(latest, end)
    return total
