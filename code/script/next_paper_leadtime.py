"""NP005: paired issue leads for one fixed future power process.

Geurts et al. (2006), Extremely randomized trees,
doi:10.1007/s10994-006-6226-1, randomized tree construction: fixed-budget
classifiers provide the primary information/lead-time comparison. Regressors
and conditional full paths provide a second forecast expression.
Meinshausen (2006), Quantile Regression Forests, JMLR 7:983-999, Sections 2-3,
equations (4)-(6): reuse NP004's normalized leaf sampling, without claiming
its consistency theorem applies to dependent rolling windows.
Worsnop et al. (2018), Generating wind power scenarios for probabilistic ramp
event prediction using multivariate statistical post-processing,
doi:10.5194/wes-3-371-2018: keep temporal dependence when deriving event risk.
Gneiting & Raftery (2007), Strictly Proper Scoring Rules, Prediction, and
Estimation, doi:10.1198/016214506000001437: inherited Brier/log/CRPS scores
measure probability skill. Kunsch (1989), doi:10.1214/aos/1176347265,
dependent-data resampling motivates shared calendar blocks, not IID windows.

The fixed-target pairing is this experiment's design, not a new algorithm.
All leads have identical outcomes and archived target-weather information;
only pre-issue observations become older. Forecast wind groups describe an
available model prediction, not independently verified atmospheric mechanisms.
"""
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from next_paper_trajectory import (ROOT, POWER, WEATHER, CAP, model_inputs,
    frame_and_splits, weather_features, digest, write_json, status)
from next_paper_wind_history import wind_history
from next_paper_probability import (probability_from_classes, probability_losses,
    empirical_crps, build_leaf_bank, draw_neighbors, paired_loss_effect)
from wind_events.paired_probability import block_design

OUT = ROOT / 'outputs/next_paper/np005'
LEADS = [15, 60, 120, 240, 480, 720]
ARMS = ['power', 'power_weather', 'power_weather_wind']
SEEDS = [41, 42, 43]
STATES = ['forecast_falling', 'forecast_middle', 'forecast_rising']
PARAMETERS = dict(n_estimators=150, max_depth=16, min_samples_leaf=32,
                  max_features=1., n_jobs=4)
STEPS = 16
BATCH = 256


def path_class(path, threshold=.2):
    """Apply NP004 ordered excursions within the future target only.

    Both endpoints are unobserved at issue. Inserting issue power would change
    the target across leads, so the first target point is the path's start.
    """
    path = np.asarray(path)
    if path.ndim not in (2, 3) or path.shape[-1] < 2 or not np.isfinite(path).all():
        raise ValueError('Finite future paths with at least two nodes required')
    up = np.max(path - np.minimum.accumulate(path, axis=-1), axis=-1)
    down = np.max(np.maximum.accumulate(path, axis=-1) - path, axis=-1)
    return (up >= threshold - 1e-12).astype(np.int8) + 2 * (down >= threshold - 1e-12).astype(np.int8)


def shift_rows(values, rows):
    """Positive rows use earlier input; keep edge gaps explicit."""
    result = np.full(np.shape(values), np.nan, dtype=float)
    if rows == 0:
        return np.asarray(values, float).copy()
    result[rows:] = values[:-rows]
    return result


def paired_masks(clock, valid, first, second):
    end = clock + pd.Timedelta(hours=4)
    history_start = clock - pd.Timedelta(minutes=max(LEADS)) - pd.Timedelta(hours=24)
    return {'train': valid & (end < first),
            'validation': valid & (history_start >= first) & (end < second),
            'test': valid & (history_start >= second)}


def build_data(frame, weather, first, second):
    clock = frame.index
    if not clock.is_unique or not (np.diff(clock.as_unit('ns').asi8) == pd.Timedelta(minutes=15).value).all():
        raise ValueError('Unique regular 15-minute clock required')
    shape, common, _ = model_inputs(frame, STEPS)
    power = np.delete(np.c_[common, shape], 2, axis=1)
    wind, _, _ = wind_history(frame)
    onsite = np.c_[frame.wind.to_numpy() / 20, wind]
    history = np.stack([shift_rows(power, lead // 15) for lead in LEADS])
    onsite = np.stack([shift_rows(onsite, lead // 15) for lead in LEADS])
    anchors = np.stack([shift_rows(frame.available_power.to_numpy() / CAP, lead // 15) for lead in LEADS])
    extra, _, columns = weather_features(weather, clock, STEPS)
    nominal_latest = (clock + pd.Timedelta(hours=4)).ceil('h') - pd.Timedelta(hours=24)
    earliest_issue = clock - pd.Timedelta(minutes=max(LEADS))
    margin = (earliest_issue - nominal_latest) / pd.Timedelta(hours=1)
    if not (margin > 0).all():
        raise ValueError('Target weather unavailable at the earliest issue')
    actual = np.column_stack([frame.target_power.shift(-k).to_numpy() / CAP for k in range(STEPS + 1)])
    # model_inputs replaces NaN shapes for its own interface. Explicitly check
    # every original power lag, so missing observations cannot become zeros.
    original_valid = np.isfinite(np.column_stack([frame.available_power.shift(k) for k in range(17)])).all(axis=1)
    valid_history = np.stack([shift_rows(original_valid.astype(float), lead // 15) == 1 for lead in LEADS]).all(axis=0)
    valid = (valid_history & np.isfinite(history).all(axis=(0, 2))
             & np.isfinite(onsite).all(axis=(0, 2)) & np.isfinite(anchors).all(axis=0)
             & np.isfinite(extra).all(axis=1) & np.isfinite(actual).all(axis=1)
             & ((actual >= -.05) & (actual <= 1.2)).all(axis=1))
    masks = paired_masks(clock, valid, first, second)
    keep = np.logical_or.reduce(list(masks.values()))
    hour = clock.tz_convert('Asia/Shanghai').hour + clock.minute / 60
    calendar = np.c_[np.sin(hour * 2 * np.pi / 24), np.cos(hour * 2 * np.pi / 24),
                     np.sin(clock.dayofyear * 2 * np.pi / 365.25), np.cos(clock.dayofyear * 2 * np.pi / 365.25)]
    # The target calendar is known at every issue. Keep its encoding identical
    # so tree splits do not change merely because the clock origin shifts.
    history[:, :, 4:8] = calendar[None]
    ui = [i for i, c in enumerate(columns) if c.startswith('gfs_global_') and c.endswith('_u')]
    vi = [i for i, c in enumerate(columns) if c.startswith('gfs_global_') and c.endswith('_v')]
    if len(ui) != 1 or len(vi) != 1:
        raise ValueError(f'GFS u/v columns not identifiable: {columns}')
    trend = np.hypot(extra[:, -8 + ui[0]], extra[:, -8 + vi[0]]) - np.hypot(extra[:, ui[0]], extra[:, vi[0]])
    cutoffs = np.quantile(trend[masks['train']], [.25, .75])
    states = np.select([trend <= cutoffs[0], trend >= cutoffs[1]], [STATES[0], STATES[2]], default=STATES[1])
    data = dict(history=history[:, keep], onsite=onsite[:, keep], anchor=anchors[:, keep],
                weather=extra[keep], actual=actual[keep], calendar=calendar[keep],
                times_ns=clock.as_unit('ns').asi8[keep], outcome=path_class(actual[keep]),
                forecast_trend=trend[keep], state=states[keep],
                **{s: m[keep] for s, m in masks.items()})
    metadata = dict(weather_columns=columns, weather_minimum_nominal_margin_hours=float(np.min(margin)),
                    state_cutoffs_ms=cutoffs.tolist(), raw_clock_rows=len(clock),
                    retained_rows=int(keep.sum()), history_features=33, weather_features=136, onsite_features=17)
    return data, metadata


def features(data, lead, arm, mask):
    parts = [data['history'][LEADS.index(lead), mask]]
    if arm != 'power':
        parts.append(data['weather'][mask])
    if arm == 'power_weather_wind':
        parts.append(data['onsite'][LEADS.index(lead), mask])
    return np.column_stack(parts)


def read_data(out):
    with np.load(out / 'dataset.npz') as source:
        return {k: source[k] for k in source.files}


def prepare(out):
    if out.exists():
        raise RuntimeError('New output directory required')
    frame, first, second = frame_and_splits()
    data, meta = build_data(frame, pd.read_parquet(WEATHER), first, second)
    out.mkdir(parents=True)
    np.savez_compressed(out / 'dataset.npz', **data)
    support = []
    for split in ('train', 'validation', 'test'):
        for state in ['all', *STATES]:
            mask = data[split] & (True if state == 'all' else data['state'] == state)
            times = pd.to_datetime(data['times_ns'][mask], utc=True)
            counts = np.bincount(data['outcome'][mask], minlength=4)
            support.append(dict(split=split, state=state, windows=int(mask.sum()),
                occupied_7d_blocks=len(np.unique(times.as_unit('ns').asi8 // pd.Timedelta(days=7).value)),
                first_target=str(times[0]), last_target=str(times[-1]), **{f'class{k}': int(n) for k, n in enumerate(counts)}))
    pd.DataFrame(support).to_csv(out / 'support.csv', index=False)
    files = [POWER, WEATHER, Path(__file__), ROOT / 'script/next_paper_trajectory.py',
             ROOT / 'script/next_paper_wind_history.py', ROOT / 'script/next_paper_probability.py',
             ROOT / 'script/forecast_penalty_v24.py', ROOT / 'src/wind_events/paired_probability.py']
    protocol = dict(experiment='NP005', stage='EXPLORATORY_PREVIOUSLY_STUDIED_CALENDAR',
        question='How do fixed-target event risks and matched information gains change as observations become older?',
        target='17 native power points in [T,T+4h]; target start is unobserved at every issue',
        target_threshold_pu=.2, lead_minutes=LEADS, arms=ARMS, seeds=SEEDS,
        primary_method='classifier', secondary_method='joint', model_parameters=PARAMETERS,
        selection='No parameter search or test selection; all models frozen after validation diagnostics',
        references=['train_frequency', 'calendar_classifier'],
        input_clock='Each issue is T-L; onsite measurements end no later than issue minus1min',
        calendar_encoding='Target T hour/day encodings identical across all leads, including the power-history arms',
        weather='Same fixed24h-lead target fields for every task lead; proxy site, actual publication clock unavailable',
        support='Identical target windows across all leads/arms; earliest issue has24h split purge, target end inside split',
        boundaries_utc=[str(first), str(second)],
        forecast_states='Training quartiles of GFS target-window endpoint wind-speed change; constant across task leads',
        sampling_seeds={'validation': 9401, 'test': 9402},
        calibration='None in this version; report event reliability and marginal coverage',
        inference='Paired target calendar blocks3/7/14d,2000 draws,seed41,primary7d; exploratory pointwise intervals',
        lead_range='Only contiguous supported grid leads from15min; no interpolation or theoretical limit',
        acceptance='Same targets, causal inputs, frozen artifacts; independently replay labels/probabilities/losses and primary intervals',
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in files},
        dataset_sha256=digest(out / 'dataset.npz'), **meta)
    write_json(out / 'protocol.json', protocol)
    status(out, 'PREPARED', support={s: int(data[s].sum()) for s in ('train', 'validation', 'test')})
    print(pd.DataFrame(support).to_string(index=False), flush=True)


def sources(out):
    p = json.loads((out / 'protocol.json').read_text())
    for file, sha in p['source_sha256'].items():
        if digest(ROOT / file) != sha:
            raise RuntimeError(f'Changed source: {file}')
    if digest(out / 'dataset.npz') != p['dataset_sha256']:
        raise RuntimeError('Changed dataset')
    return p


def class_probability(models, x):
    p = np.zeros((len(x), 4))
    for model in models:
        p[:, model.classes_] += model.predict_proba(x) / len(models)
    return p


def generate_cell(out, data, protocol, split, lead, arm, pack):
    mask, tr = data[split], data['train']
    j = LEADS.index(lead)
    x = features(data, lead, arm, mask)
    anchor = data['anchor'][j, mask]
    delta = data['actual'][tr] - data['anchor'][j, tr, None]
    neighbors = draw_neighbors(pack['regressors'], pack['bank'], x, protocol['sampling_seeds'][split])
    joint, means, crps, quantiles = [], [], [], []
    for start in range(0, len(x), BATCH):
        stop = min(start + BATCH, len(x))
        paths = np.clip(anchor[start:stop, None, None] + delta[neighbors[start:stop]], 0, 1)
        joint.append(probability_from_classes(path_class(paths)))
        means.append(paths.mean(axis=1))
        crps.append(empirical_crps(paths, data['actual'][mask][start:stop]))
        quantiles.append(np.quantile(paths, [.025, .1, .9, .975], axis=1))
    result = dict(neighbors=neighbors, joint=np.vstack(joint),
                  classifier=class_probability(pack['classifiers'], x), mean_path=np.vstack(means),
                  crps=np.vstack(crps), quantiles=np.concatenate(quantiles, axis=1))
    np.savez_compressed(out / f'{split}_{lead}_{arm}.npz', **result)
    return result


def validate(out):
    protocol = sources(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'PREPARED':
        raise RuntimeError('Fresh prepared run required')
    data = read_data(out); tr = data['train']; va = data['validation']
    calendars = [ExtraTreesClassifier(**PARAMETERS, random_state=s).fit(data['calendar'][tr], data['outcome'][tr]) for s in SEEDS]
    joblib.dump(calendars, out / 'calendar.joblib', compress=3)
    np.save(out / 'validation_calendar.npy', class_probability(calendars, data['calendar'][va]))
    records = []
    for lead in LEADS:
        for arm in ARMS:
            x = features(data, lead, arm, tr)
            delta = data['actual'][tr] - data['anchor'][LEADS.index(lead), tr, None]
            regressors, classifiers = [], []
            for seed in SEEDS:
                regression = ExtraTreesRegressor(**PARAMETERS, random_state=seed).fit(x, delta)
                regressors.append((seed, regression, None))
                classifiers.append(ExtraTreesClassifier(**PARAMETERS, random_state=seed).fit(x, data['outcome'][tr]))
                print(datetime.now(timezone.utc).isoformat(), 'fit', lead, arm, seed, flush=True)
            bank, error = build_leaf_bank(regressors, x, delta)
            pack = dict(regressors=regressors, classifiers=classifiers, bank=bank, leaf_mean_error=error)
            joblib.dump(pack, out / f'model_{lead}_{arm}.joblib', compress=3)
            result = generate_cell(out, data, protocol, 'validation', lead, arm, pack)
            for method in ('classifier', 'joint'):
                records.append(dict(lead_minutes=lead, arm=arm, method=method,
                    **{metric: float(v.mean()) for metric, v in probability_losses(data['outcome'][va], result[method]).items()}))
            print('validation', records[-2:], flush=True)
    pd.DataFrame(records).to_csv(out / 'validation_summary.csv', index=False)
    write_json(out / 'freeze.json', dict(state='FROZEN_BEFORE_TEST', frozen_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(out / 'protocol.json'), model_sha256={p.name: digest(p) for p in out.glob('*.joblib')},
        validation_sha256={p.name: digest(p) for p in out.glob('validation*')}))
    status(out, 'VALIDATION_FROZEN', fitted_models=111)


def frozen(out):
    protocol = sources(out)
    lock = json.loads((out / 'freeze.json').read_text())
    if digest(out / 'protocol.json') != lock['protocol_sha256']:
        raise RuntimeError('Changed protocol')
    for file, sha in {**lock['model_sha256'], **lock['validation_sha256']}.items():
        if digest(out / file) != sha:
            raise RuntimeError(f'Changed frozen artifact: {file}')
    return protocol


def loss_difference(times, candidate, reference, days):
    idx, weights = block_design(times, days, 2000, 41)
    sums = np.bincount(idx, weights=candidate - reference, minlength=weights.shape[1])
    counts = np.bincount(idx, minlength=weights.shape[1])
    draws = weights @ sums / (weights @ counts)
    low, high = np.quantile(draws, [.025, .975])
    return dict(loss_difference=float(np.mean(candidate - reference)), difference_low=float(low), difference_high=float(high))


def score(out, data):
    mask, tr = data['test'], data['train']
    y = data['outcome'][mask]; times = pd.to_datetime(data['times_ns'][mask], utc=True)
    frequency = np.bincount(data['outcome'][tr], minlength=4) / tr.sum()
    predictions = {('frequency', 0, 'reference'): np.tile(frequency, (len(y), 1)),
                   ('calendar', 0, 'reference'): np.load(out / 'test_calendar.npy')}
    margins = []
    for lead in LEADS:
        for arm in ARMS:
            with np.load(out / f'test_{lead}_{arm}.npz') as result:
                for method in ('classifier', 'joint'):
                    predictions[(arm, lead, method)] = result[method]
                q = result['quantiles']; actual = data['actual'][mask]
                for node in range(17):
                    margins.append(dict(lead_minutes=lead, arm=arm, target_offset_minutes=node * 15,
                        crps=float(result['crps'][:, node].mean()),
                        coverage80=float(((actual[:, node] >= q[1, :, node]) & (actual[:, node] <= q[2, :, node])).mean()),
                        coverage95=float(((actual[:, node] >= q[0, :, node]) & (actual[:, node] <= q[3, :, node])).mean()),
                        width80=float((q[2, :, node] - q[1, :, node]).mean())))
    losses = {key: probability_losses(y, p) for key, p in predictions.items()}
    scores, intervals, reliability = [], [], []
    comparisons = []
    for method in ('classifier', 'joint'):
        for lead in LEADS:
            for arm in ARMS:
                for baseline in ('frequency', 'calendar'):
                    comparisons.append(((arm, lead, method), (baseline, 0, 'reference'), 'skill'))
            comparisons.extend([(('power_weather', lead, method), ('power', lead, method), 'weather_information'),
                                (('power_weather_wind', lead, method), ('power_weather', lead, method), 'wind_information')])
        for earlier, later in zip(LEADS[1:], LEADS[:-1]):
            for arm in ARMS:
                comparisons.append(((arm, earlier, method), (arm, later, method), 'older_issue'))
    for state in ['all', *STATES]:
        subset = np.ones(len(y), bool) if state == 'all' else data['state'][mask] == state
        for key, values in losses.items():
            for metric, v in values.items():
                base = losses[('frequency', 0, 'reference')][metric]
                scores.append(dict(arm=key[0], lead_minutes=key[1], method=key[2], state=state, metric=metric,
                    windows=int(subset.sum()), return_windows=int(((y == 3) & subset).sum()),
                    loss=float(v[subset].mean()), skill_vs_frequency_pct=float(100 * (1 - v[subset].mean() / base[subset].mean()))))
        for candidate, reference, kind in comparisons:
            for metric in ('return_brier', 'up_brier', 'down_brier', 'multiclass_brier'):
                a, b = losses[candidate][metric][subset], losses[reference][metric][subset]
                for days in ([3, 7, 14] if state == 'all' else [7]):
                    intervals.append(dict(candidate=':'.join(map(str, candidate)), reference=':'.join(map(str, reference)),
                        comparison=kind, state=state, metric=metric, windows=int(subset.sum()),
                        **paired_loss_effect(times[subset], a, b, days), **loss_difference(times[subset], a, b, days)))
    for key, p in predictions.items():
        bins = np.minimum((p[:, 3] * 10).astype(int), 9)
        for k in range(10):
            subset = bins == k
            reliability.append(dict(arm=key[0], lead_minutes=key[1], method=key[2], bin_low=k / 10,
                windows=int(subset.sum()), mean_probability=float(p[subset, 3].mean()) if subset.any() else np.nan,
                observed_fraction=float((y[subset] == 3).mean()) if subset.any() else np.nan))
    pd.DataFrame(scores).to_csv(out / 'test_summary.csv', index=False)
    pd.DataFrame(intervals).to_csv(out / 'paired_intervals.csv', index=False)
    pd.DataFrame(reliability).to_csv(out / 'reliability.csv', index=False)
    pd.DataFrame(margins).to_csv(out / 'marginal_summary.csv', index=False)
    print(pd.DataFrame(scores).query("state == 'all' and metric == 'return_brier'").to_string(index=False), flush=True)


def test(out):
    protocol = frozen(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'VALIDATION_FROZEN':
        raise RuntimeError('Frozen untested run required')
    if list(out.glob('test_*')):
        raise RuntimeError('Existing test artifacts must not be overwritten')
    data = read_data(out)
    np.save(out / 'test_calendar.npy', class_probability(joblib.load(out / 'calendar.joblib'), data['calendar'][data['test']]))
    for lead in LEADS:
        for arm in ARMS:
            pack = joblib.load(out / f'model_{lead}_{arm}.joblib')
            generate_cell(out, data, protocol, 'test', lead, arm, pack)
            print('test generated', lead, arm, flush=True)
    score(out, data)
    status(out, 'COMPLETE', windows=int(data['test'].sum()), fitted_models=111)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'validate', 'test'], required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'validate': validate, 'test': test}[args.phase](args.output)


if __name__ == '__main__':
    main()
