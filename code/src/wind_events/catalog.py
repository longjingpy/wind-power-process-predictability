"""Detection and event morphology with durations expressed in physical minutes.

SDA: Florita et al. (2013), doi:10.1109/GreenTech.2013.30, pp.2-3.
OpSDA: Cui et al. (2015), doi:10.1109/PESGM.2015.7286272, p.2 Eqs.1-3.
These source-traced algorithms are imported from the archived implementation.
Threshold, historical tail, endpoint corridor and composite rules are declared
study definitions, not fitted physical-event truth.
"""
import numpy as np
import pandas as pd
from .protocol import Protocol, chronological_split, fit_scale
from .literature import original_sda, opsda_2015


def valid_runs(mask):
    delta = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(delta == 1), np.flatnonzero(delta == -1)))


def corridor(x, epsilon, history_steps):
    """Actual-endpoint corridor; adaptive width is frozen at each new start."""
    def width(start):
        if epsilon is not None:
            return float(epsilon), 0, "fixed"
        past = x[max(0, start-history_steps):start+1]
        change = np.diff(past)
        if not len(change):
            return .025, 0, "floor_no_history"
        mad = np.median(np.abs(change-np.median(change)))
        value = max(.025, 1.4826*mad, .01*np.median(np.abs(past)))
        return float(value), len(change), "full_history" if len(change) == history_steps else "partial_history"
    if len(x) < 2:
        return []
    start, low, high = 0, -np.inf, np.inf
    eps, n, mode = width(start)
    result = []
    for end in range(1, len(x)):
        slope = (x[end]-x[start])/(end-start)
        if slope < low-1e-12 or slope > high+1e-12:
            result.append((start, end-1, eps, n, mode))
            start, low, high = end-1, -np.inf, np.inf
            eps, n, mode = width(start)
        low = max(low, (x[end]-x[start]-eps)/(end-start))
        high = min(high, (x[end]-x[start]+eps)/(end-start))
    result.append((start, len(x)-1, eps, n, mode))
    return result


def merge_signed(intervals):
    result = []
    for a, b, sign in intervals:
        if result and sign == result[-1][2] and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]), sign)
        else:
            result.append((int(a), int(b), int(sign)))
    return result


def detect(values, protocol=Protocol(), configurations=None):
    x = np.asarray(values, float)
    if not np.isfinite(x).all():
        raise ValueError("Detection requires a continuous finite run")
    p = pd.Series(x)
    result = {}
    selected = None if configurations is None else set(configurations)
    for method in ("threshold", "financial_tail", "mean_shift"):
        for minutes in protocol.horizons_minutes:
            if selected is not None and f"{method}_{minutes}min" not in selected:
                continue
            lag = protocol.steps(minutes)
            movement = p.diff(lag)
            span = lag
            if method == "mean_shift":
                movement = p.rolling(lag).mean()-p.shift(lag).rolling(lag).mean()
                span = 2*lag-1
            hit = movement.abs().ge(protocol.amplitude)
            if method == "financial_tail":
                steps = protocol.steps(protocol.history_minutes)
                history = movement.shift(1).rolling(steps, min_periods=steps)
                hit = (movement.lt(history.quantile(protocol.tail_probability)) |
                       movement.gt(history.quantile(1-protocol.tail_probability))) & movement.abs().ge(protocol.tail_minimum_amplitude)
            ids = np.flatnonzero(hit.to_numpy())
            result[f"{method}_{minutes}min"] = [
                (a, b, s, None, None, "not_adaptive") for a, b, s in
                merge_signed([(i-span, i, np.sign(movement.iloc[i])) for i in ids])]
    for eps in protocol.corridor_widths:
        if selected is not None and f"endpoint_corridor_{eps:.3f}" not in selected:
            continue
        result[f"endpoint_corridor_{eps:.3f}"] = [
            (a, b, int(np.sign(x[b]-x[a])), e, n, m)
            for a, b, e, n, m in corridor(x, eps, protocol.steps(protocol.history_minutes))
            if abs(x[b]-x[a]) >= protocol.amplitude]
    eps = protocol.literature_epsilon
    if selected is None or f"sda_florita2013_{eps:.3f}" in selected:
        result[f"sda_florita2013_{eps:.3f}"] = [
            (a, b, int(np.sign(x[b]-x[a])), eps, None, "not_adaptive")
            for a, b in original_sda(x, eps) if abs(x[b]-x[a]) >= protocol.amplitude]
    for minutes in protocol.horizons_minutes:
        if selected is not None and f"opsda_cui2015_{minutes}min_{eps:.3f}" not in selected:
            continue
        result[f"opsda_cui2015_{minutes}min_{eps:.3f}"] = [
            (a, b, int(np.sign(x[b]-x[a])), eps, None, "not_adaptive")
            for a, b in opsda_2015(x, eps, protocol.steps(minutes), protocol.amplitude)]
    if selected is None or "adaptive_corridor" in selected:
        result["adaptive_corridor"] = [
            (a, b, int(np.sign(x[b]-x[a])), e, n, m)
            for a, b, e, n, m in corridor(x, None, protocol.steps(protocol.history_minutes))
            if abs(x[b]-x[a]) >= protocol.amplitude]
    if selected is not None and set(result) != selected:
        raise ValueError(f"Unknown detector configurations: {selected-set(result)}")
    return result


def describe(values, start, end, protocol=Protocol()):
    x = np.asarray(values, float)
    if not 0 <= start < end < len(x):
        raise ValueError("Nonempty positive-duration event required")
    window = x[start:end+1]
    delta = np.diff(window)
    context = protocol.steps(protocol.context_minutes)
    eligible = start >= context and end+context < len(x)
    minutes = protocol.resolution_minutes
    row = {"duration_hours": (end-start)*minutes/60, "amplitude": float(window[-1]-window[0]),
           "direction": int(np.sign(window[-1]-window[0])), "power_start": float(window[0]),
           "power_end": float(window[-1]), "power_range": float(np.ptp(window)),
           "total_variation": float(np.abs(delta).sum()),
           "max_abs_rate_per_hour": float(np.abs(delta).max()*60/minutes),
           "max_abs_chord_residual": float(np.abs(window-np.linspace(window[0], window[-1], len(window))).max()),
           "curvature_l1": float(np.abs(np.diff(window, n=2)).sum()),
           "pre_mean": float(x[start-context:start].mean()) if eligible else None,
           "post_mean": float(x[end+1:end+context+1].mean()) if eligible else None,
           "context_complete": bool(eligible), "representation_eligible": bool(eligible),
           "left_boundary": start == 0, "right_boundary": end == len(x)-1,
           "min_offset": int(window.argmin()), "max_offset": int(window.argmax())}
    shape = None
    row["shape_normalizer"] = None
    if eligible:
        # Four common physical context locations, even at native 10/15/60 min.
        pre = start + np.linspace(-context, -context/4, 4)
        post = end + np.linspace(context/4, context, 4)
        locations = np.r_[pre, np.linspace(start, end, 17), post]
        path = np.interp(locations, np.arange(len(x)), x)-x[start]
        normalizer = float(np.abs(path).max())
        if normalizer <= 0:
            row["representation_eligible"] = False
        else:
            shape = (path/normalizer).astype(np.float32)
            row["shape_normalizer"] = normalizer
    return row, shape


def composite_intervals(values, intervals, protocol=Protocol()):
    """Pair adjacent opposite ramps with an observed internal reversal.

    A composite retains two primitive indices and a signal-derived turning
    point. Overlapping pairs are preserved explicitly, rather than being
    counted as independent weather processes.
    """
    x = np.asarray(values, float)
    ordered = sorted(enumerate(intervals), key=lambda item: (item[1][0], item[1][1], item[0]))
    result = []
    seen = set()
    for (ia, first), (ib, second) in zip(ordered, ordered[1:]):
        a, b, sign = first[:3]
        c, d, other = second[:3]
        if sign == other or sign == 0 or other == 0 or d <= b:
            continue
        if max(0, c-b)*protocol.resolution_minutes > protocol.composite_gap_minutes:
            continue
        if (d-a)*protocol.resolution_minutes > protocol.composite_max_minutes:
            continue
        turn = a + int(np.argmax(x[a:d+1]) if sign > 0 else np.argmin(x[a:d+1]))
        if not a < turn < d:
            continue
        if min(abs(x[turn]-x[a]), abs(x[d]-x[turn])) < protocol.amplitude/2:
            continue
        key = (a, d, turn)
        if key in seen:
            continue
        seen.add(key)
        result.append((a, d, turn, "inverted_v" if sign > 0 else "v", ia, ib))
    return result


def build_catalog(site, turbine, times, power, valid, protocol=Protocol(), capacity=None,
                  calibration="metadata_first", clock_status="UTC_VERIFIED", include_composites=True,
                  split_boundaries=None, scale_override=None, external_test=False, configurations=None):
    t = pd.DatetimeIndex(times).as_unit("ns")
    if external_test:
        if scale_override is None:
            raise ValueError("External test catalogue requires a frozen reference scale")
        if t.hasnans or not t.is_unique or not t.is_monotonic_increasing:
            raise ValueError("Sorted unique nonmissing external clock required")
        split, boundaries = np.repeat("test", len(t)), (None, None)
    else:
        split, boundaries = chronological_split(t, split_boundaries)
    if not np.all(np.diff(t.asi8) == pd.Timedelta(minutes=protocol.resolution_minutes).value):
        raise ValueError("Regular observation grid required; missing bins must be explicit")
    valid = np.asarray(valid, bool) & np.isfinite(power)
    if calibration not in ("metadata_first", "training_q995"):
        raise ValueError("Unknown calibration mode")
    scale, basis = scale_override if external_test else fit_scale(power, valid, split, capacity if calibration == "metadata_first" else None)
    if scale_override is not None:
        scale, basis = scale_override
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Frozen reference scale must be positive and finite")
    x = np.asarray(power, float)/scale
    protocol_id = (f"r{protocol.resolution_minutes}_h{protocol.history_minutes}_c{protocol.context_minutes}"
                   f"_a{protocol.amplitude:g}_q{protocol.tail_probability:g}"
                   f"_f{protocol.tail_minimum_amplitude:g}_e{protocol.literature_epsilon:g}")
    rows, shapes = [], []
    for part in ("train", "validation", "test"):
        for lo, hi in valid_runs(valid & (split == part)):
            if hi-lo < 2:
                continue
            values = x[lo:hi]
            for config, intervals in detect(values, protocol, configurations).items():
                ids = []
                def append(a, b, sign, level, kind, epsilon=None, history_n=None, history_mode="not_adaptive", children="", turning=None):
                    desc, shape = describe(values, a, b, protocol)
                    aa, bb = int(lo+a), int(lo+b)
                    eid = f"{site}:{turbine}:{protocol_id}:{calibration}:{part}:{config}:{level}:{kind}:{aa}:{bb}"
                    desc.update(event_id=eid, site=site, turbine=str(turbine), split=part, config=config,
                                event_level=level, event_kind=kind, start_index=aa, end_index=bb,
                                time_start=t[aa], time_end=t[bb],
                                time_min=t[aa+desc.pop("min_offset")], time_max=t[aa+desc.pop("max_offset")],
                                detector_direction=int(sign), scale=scale, scale_basis=basis,
                                shape_row=len(shapes) if shape is not None else -1,
                                epsilon=epsilon, history_n=history_n, history_mode=history_mode,
                                run_start=int(lo), run_end_exclusive=int(hi),
                                resolution_minutes=protocol.resolution_minutes, clock_status=clock_status,
                                protocol_id=protocol_id,
                                child_event_ids=children, turning_index=None if turning is None else int(lo+turning),
                                quality_flag="complete_context" if shape is not None else "context_or_run_boundary")
                    rows.append(desc)
                    if shape is not None:
                        shapes.append(shape)
                    return eid
                for a, b, sign, eps, n, mode in intervals:
                    ids.append(append(a, b, sign, "primitive", "rise" if sign > 0 else "fall", eps, n, mode))
                if include_composites:
                    for a, b, turning, kind, ia, ib in composite_intervals(values, intervals, protocol):
                        append(a, b, np.sign(values[b]-values[a]), "composite", kind,
                               children=ids[ia]+"|"+ids[ib], turning=turning)
    table = pd.DataFrame(rows)
    array = np.asarray(shapes, dtype=np.float32).reshape(-1, 25)
    if len(table) and not table.event_id.is_unique:
        raise AssertionError("Duplicate event identity")
    if not np.isfinite(array).all():
        raise AssertionError("Nonfinite shape")
    audit = {"site": site, "turbine": str(turbine), "rows": len(t), "valid_rows": int(valid.sum()),
             "start": str(t[0]), "end": str(t[-1]), "b1": str(boundaries[0]) if boundaries[0] is not None else None,
             "b2": str(boundaries[1]) if boundaries[1] is not None else None,
             "split_design": "external_all_test" if external_test else "chronological_60_20_20",
             "scale": scale, "scale_basis": basis, "clock_status": clock_status,
             "candidates": len(table), "complete_shapes": len(array), "protocol": protocol.to_dict()}
    return table, array, audit
