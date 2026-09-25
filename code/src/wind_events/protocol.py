"""Physical-time contract shared by every site and resolution.

Study definition: docs/RESEARCH_ENHANCEMENT_V18.md. Train-only calibration follows
the leakage taxonomy of Kapoor and Narayanan, Patterns (2023),
doi:10.1016/j.patter.2023.100804. It keeps external evaluation independent of
test-distribution normalization; it does not identify nameplate capacity.
"""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Protocol:
    resolution_minutes: int = 30
    horizons_minutes: tuple = (60, 120, 240)
    history_minutes: int = 1440
    context_minutes: int = 120
    amplitude: float = .2
    tail_probability: float = .05
    tail_minimum_amplitude: float = .1
    corridor_widths: tuple = (.025, .05, .1)
    literature_epsilon: float = .025
    composite_max_minutes: int = 240
    composite_gap_minutes: int = 30

    def __post_init__(self):
        if self.resolution_minutes <= 0 or self.context_minutes <= 0:
            raise ValueError("Positive physical cadence and context required")
        if any(v % self.resolution_minutes for v in (*self.horizons_minutes, self.history_minutes, self.context_minutes)):
            raise ValueError("Detection, history and context must be integer sampling steps")
        if not 0 < self.tail_probability < .5 or self.amplitude <= 0:
            raise ValueError("Invalid detector threshold")

    def steps(self, minutes):
        if minutes % self.resolution_minutes:
            raise ValueError("Physical duration must contain integer sampling steps")
        return int(minutes // self.resolution_minutes)

    def to_dict(self):
        return asdict(self)


def chronological_split(times, boundaries=None):
    times = pd.DatetimeIndex(times)
    if len(times) < 2 or times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("Sorted unique nonmissing timestamps required")
    b1, b2 = boundaries if boundaries is not None else (times[0] + .6 * (times[-1] - times[0]), times[0] + .8 * (times[-1] - times[0]))
    if not times[0] < b1 < b2 <= times[-1]:
        raise ValueError("Split boundaries must lie inside the observation domain")
    return np.where(times < b1, "train", np.where(times < b2, "validation", "test")), (b1, b2)


def fit_scale(power, valid, splits, capacity=None):
    x = np.asarray(power, float)
    calibration = x[np.asarray(valid, bool) & (np.asarray(splits) == "train") & np.isfinite(x)]
    if not len(calibration):
        raise ValueError("No eligible training data for calibration")
    scale = float(capacity) if capacity is not None else float(np.quantile(calibration, .995))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive finite scale required")
    return scale, "nameplate" if capacity is not None else "training_q995"


def regularize(times, power, valid, native_minutes, target_minutes, timestamp_position="start"):
    """Require all native bins; preserve gaps and reject ambiguous duplicates.

    Canonical timestamp denotes a bin start. End-labelled source means are
    shifted by their native averaging period, before left-closed aggregation.
    No values are interpolated in this observation table.
    """
    if target_minutes < native_minutes or target_minutes % native_minutes:
        raise ValueError("Only integer-factor downsampling is supported")
    if timestamp_position not in ("start", "end", "unverified"):
        raise ValueError("Unknown timestamp convention")
    index = pd.DatetimeIndex(pd.to_datetime(times, errors="raise"))
    if index.hasnans:
        raise ValueError("Missing source timestamp")
    if timestamp_position == "end":
        index = index - pd.Timedelta(minutes=native_minutes)
    frame = pd.DataFrame({"power": np.asarray(power, float), "valid": np.asarray(valid, bool)}, index=index).sort_index()
    duplicate = frame.index.duplicated(keep=False)
    duplicate_count = int(duplicate.sum())
    frame.loc[duplicate, "valid"] = False
    frame = frame.loc[~frame.index.duplicated(keep="first")]
    frame["valid"] &= np.isfinite(frame.power)
    frame["power"] = frame.power.where(frame.valid)
    rule = f"{target_minutes}min"
    grid = frame.resample(rule, origin="epoch", label="left", closed="left")
    means = grid.power.mean()
    count = grid.valid.sum()
    expected = target_minutes // native_minutes
    total = grid.size()
    ok = count.eq(expected) & total.eq(expected)
    means = means.where(ok)
    audit = {"native_minutes": int(native_minutes), "resolution_minutes": int(target_minutes),
             "expected_samples_per_bin": int(expected), "duplicate_source_rows": duplicate_count,
             "partial_or_missing_bins": int((~ok).sum()), "timestamp_position": timestamp_position}
    return means.index, means.to_numpy(), ok.to_numpy(), audit
