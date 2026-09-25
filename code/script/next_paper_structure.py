"""NP002: farm-level event correspondence and validation-selected improvements.

The task reuses NP001 inputs and its frozen chronological support. Future
catalogue labels are evaluation metadata only, never model features.

Methods: the current event manuscript S26 (unpublished, no DOI) motivates
separating endpoint chords from internal geometry; this is a target transform,
not a newly invented physical law. Geurts et al. (2006), Extremely randomized
trees, doi:10.1007/s10994-006-6226-1, abstract/method, supplies the inherited
nonlinear reference. Hoerl & Kennard (1970), doi:10.1080/00401706.1970.10488634,
abstract, motivates a regularized linear comparison. Friedman (2001), Greedy
function approximation: A gradient boosting machine,
doi:10.1214/aos/1013203451, gradient-descent function approximation, supplies a
stronger learner control so decomposition is not confused with model capacity.
Use sklearn 1.7.2's histogram implementation with explicit fixed iteration
count and no random validation split. Calendar-block inference is inherited
from wind_events.paired_probability, motivated by Kunsch (1989),
doi:10.1214/aos/1176347265; no IID window resampling is performed.
"""
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from next_paper_trajectory import (ROOT, POWER, CAP, OUT as BASE, PREFIXES,
    digest, write_json, status, source_check, structure_losses, paired_effect)
from wind_events import Protocol, build_catalog

OUT = ROOT / 'outputs/next_paper/np002'
CONFIGS = ['ridge_a1', 'ridge_a10', 'et_l32', 'et_l128', 'hgb']
BASELINES = ['np001_extra_trees', 'np001_ridge']
GROUPS = ['complete_composite', 'complete_primitive', 'partial_only', 'no_event']
SOURCE_FILES = [Path(__file__), ROOT / 'src/wind_events/catalog.py',
                ROOT / 'src/wind_events/protocol.py']


def geometry(curves, anchor):
    """The NP001/S26 chord residual, anchored at issue minus one minute."""
    h = curves.shape[1]
    fraction = (15 * np.arange(1, h + 1) + 1) / (15 * h + 1)
    return curves - anchor[:, None] - (curves[:, -1] - anchor)[:, None] * fraction


def reconstruct(reference, anchor, internal, blend):
    if internal.shape != reference.shape or not np.allclose(internal[:, -1], 0, atol=1e-12):
        raise ValueError('Geometry must align with the reference and vanish at its endpoint')
    if not 0 <= blend <= 1:
        raise ValueError('Blend must be in [0,1]')
    q = reference + blend * (internal - geometry(reference, anchor))
    q[:, -1] = reference[:, -1]
    return np.clip(q, 0, 1)


def correspondence(times, events):
    """Use observed intervals only to label future four-hour evaluation windows.

    Catalogue rules come from the project's existing Protocol/build_catalog/
    composite_intervals implementation. Full containment is a closed interval;
    positive-duration overlap is open at the outer boundaries. Counts represent
    windows, with separate event links to avoid implying independent events.
    """
    times = pd.DatetimeIndex(times)
    ns = times.as_unit('ns').asi8
    horizon = pd.Timedelta(hours=4).value
    primitive = np.zeros(len(times), int)
    composite = np.zeros(len(times), int)
    overlap = np.zeros(len(times), int)
    links = []
    for e in events.itertuples():
        start, end = pd.Timestamp(e.time_start).value, pd.Timestamp(e.time_end).value
        lo = np.searchsorted(ns, start - horizon, side='right')
        hi = np.searchsorted(ns, end, side='left')
        overlap[lo:hi] += 1
        lo = np.searchsorted(ns, end - horizon, side='left')
        hi = np.searchsorted(ns, start, side='right')
        if lo >= hi:
            continue
        counts = composite if e.event_level == 'composite' else primitive
        counts[lo:hi] += 1
        for i in range(lo, hi):
            links.append({'issue_time': times[i], 'event_id': e.event_id,
                          'event_level': e.event_level})
    groups = np.select([composite > 0, primitive > 0, overlap > 0], GROUPS[:3], default=GROUPS[3])
    labels = pd.DataFrame({'issue_time': times, 'population': groups,
             'complete_composite_count': composite, 'complete_primitive_count': primitive,
             'overlap_count': overlap})
    return labels, pd.DataFrame(links, columns=['issue_time', 'event_id', 'event_level'])


def estimator(config):
    if config.startswith('ridge_'):
        return make_pipeline(StandardScaler(), Ridge(alpha=float(config.split('a')[-1])))
    if config.startswith('et_'):
        return ExtraTreesRegressor(n_estimators=150, max_depth=16,
            min_samples_leaf=int(config.split('l')[-1]), max_features=1., random_state=41, n_jobs=4)
    return MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=150, max_leaf_nodes=15, min_samples_leaf=64, learning_rate=.05,
        l2_regularization=1., early_stopping=False, random_state=41), n_jobs=1)


def predict_direct(out, name, x, anchor):
    if name in BASELINES:
        artifact = 'extra_trees_weather.joblib' if name.endswith('trees') else 'ridge_weather.joblib'
        return np.clip(joblib.load(BASE / artifact).predict(x), 0, 1)
    return np.clip(anchor[:, None] + joblib.load(out / f'{name}.joblib').predict(x), 0, 1)


def score(actual, prediction, anchor):
    return {key: float(value.mean()) for key, value in structure_losses(actual, prediction, anchor).items()}


def choose_geometry(rows, reference_mse):
    eligible = [r for r in rows if r['trajectory'] <= reference_mse + 1e-12]
    if not eligible:
        raise ValueError('The unchanged direct reference must be included')
    return min(eligible, key=lambda r: (r['internal_geometry'], r['blend'], r['model']))


def prepare(out):
    base_protocol = source_check(BASE)
    if json.loads((BASE / 'status.json').read_text())['state'] != 'COMPLETE':
        raise RuntimeError('The inherited baseline must be complete and frozen')
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(POWER)
    data = np.load(BASE / 'dataset.npz')
    p = Protocol(resolution_minutes=15)
    boundaries = tuple(pd.Timestamp(base_protocol[k]) for k in ['training_boundary_utc', 'validation_boundary_utc'])
    table, _, audit = build_catalog('pizhou_farm', 'aggregate', frame.index,
        frame.target_power.to_numpy(), frame.target_power.notna().to_numpy(), p,
        capacity=CAP, split_boundaries=boundaries, configurations=['threshold_60min'])
    table.to_parquet(out / 'events.parquet', index=False)
    labels, links = correspondence(frame.index, table)
    labels.to_parquet(out / 'window_labels.parquet', index=False)
    links.to_parquet(out / 'event_links.parquet', index=False)
    support = []
    for split in ['train', 'validation', 'test']:
        for group in ['all', *GROUPS]:
            mask = data[split] & (np.ones(len(frame), bool) if group == 'all' else labels.population.eq(group).to_numpy())
            times = frame.index[mask]
            paired = links[links.issue_time.isin(times)]
            support.append({'split': split, 'population': group, 'windows': len(times),
                'occupied_7d_blocks': len(np.unique(times.as_unit('ns').asi8 // pd.Timedelta(days=7).value)),
                'unique_complete_primitive': paired.loc[paired.event_level.eq('primitive'), 'event_id'].nunique(),
                'unique_complete_composite': paired.loc[paired.event_level.eq('composite'), 'event_id'].nunique()})
    pd.DataFrame(support).to_csv(out / 'support.csv', index=False)
    protocol = {'experiment': 'NP002', 'stage': 'EXPLORATORY_IMPROVEMENT_AFTER_VIEWED_NP001',
        'question': 'Can target decomposition or stronger learners improve internal trajectories on common information, including complete farm events?',
        'features': 'Unchanged NP001 history+weather, 170 columns; no future catalogue labels',
        'data_and_splits': str(BASE / 'protocol.json'), 'catalogue': audit,
        'catalogue_configuration': 'threshold_60min; native farm quarter-hour points; 0.2 rated capacity',
        'populations': GROUPS, 'event_role': 'Evaluation-only future containment/overlap; not physical weather classes',
        'direct_candidates': BASELINES + ['delta_' + c for c in CONFIGS],
        'geometry_candidates': ['geometry_' + c for c in CONFIGS], 'blends': [0, .25, .5, 1.],
        'hgb': {'max_iter': 150, 'max_leaf_nodes': 15, 'min_samples_leaf': 64,
                'learning_rate': .05, 'l2_regularization': 1., 'early_stopping': False, 'seed': 41},
        'selection': 'Best validation full-trajectory direct candidate first; then minimum geometry MSE subject to full-trajectory MSE not increasing; ties prefer smaller blend',
        'endpoint_constraint': 'Keep selected direct reference endpoint exactly; geometry last coordinate is zero',
        'primary_comparisons': 'selected vs strong_direct; selected vs np001_extra_trees; strong_direct vs np001_extra_trees',
        'reporting': 'All candidates and all overall prefixes; event strata at 4h only; paired 3/7/14-day blocks, 2000 draws; primary7d',
        'baseline_hashes': {'protocol.json': digest(BASE / 'protocol.json'),
             'dataset.npz': digest(BASE / 'dataset.npz'), 'selection.json': digest(BASE / 'selection.json'),
             **{p.name: digest(p) for p in BASE.glob('*.joblib')}},
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in SOURCE_FILES},
        'catalogue_sha256': {name: digest(out / name) for name in ['events.parquet', 'window_labels.parquet', 'event_links.parquet']}}
    write_json(out / 'protocol.json', protocol)
    status(out, 'PREPARED', catalogue_events=len(table))
    print(pd.DataFrame(support).to_string(index=False), flush=True)


def check_sources(out):
    source_check(BASE)
    protocol = json.loads((out / 'protocol.json').read_text())
    for name, sha in protocol['baseline_hashes'].items():
        if digest(BASE / name) != sha:
            raise RuntimeError(f'Frozen NP001 artifact changed: {name}')
    for path, sha in protocol['source_sha256'].items():
        if digest(ROOT / path) != sha:
            raise RuntimeError(f'NP002 source changed: {path}')
    for name, sha in protocol['catalogue_sha256'].items():
        if digest(out / name) != sha:
            raise RuntimeError(f'Catalogue artifact changed: {name}')


def validate(out):
    check_sources(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'PREPARED':
        raise RuntimeError('A fresh prepared run is required')
    data = np.load(BASE / 'dataset.npz')
    tr, va = data['train'], data['validation']
    x = data['weather']; y = data['actual']; anchor = data['anchor']
    rows, predictions = [], {}
    for name in BASELINES:
        predictions[name] = predict_direct(out, name, x[va], anchor[va])
    for config in CONFIGS:
        name = 'delta_' + config
        model = estimator(config).fit(x[tr], y[tr] - anchor[tr, None])
        joblib.dump(model, out / f'{name}.joblib', compress=3)
        predictions[name] = np.clip(anchor[va, None] + model.predict(x[va]), 0, 1)
        print('direct validation', name, score(y[va], predictions[name], anchor[va]), flush=True)
    for name, q in predictions.items():
        rows.append({'model': name, 'kind': 'direct', 'blend': 0., **score(y[va], q, anchor[va])})
    direct = min(rows, key=lambda r: (r['trajectory'], r['model']))
    reference = predictions[direct['model']]
    geometry_rows = [direct]
    target = geometry(y[tr], anchor[tr])[:, :-1]
    for config in CONFIGS:
        name = 'geometry_' + config
        model = estimator(config).fit(x[tr], target)
        joblib.dump(model, out / f'{name}.joblib', compress=3)
        internal = np.c_[model.predict(x[va]), np.zeros(int(va.sum()))]
        for blend in [.25, .5, 1.]:
            key = f'{name}_b{int(blend * 100)}'
            q = reconstruct(reference, anchor[va], internal, blend)
            row = {'model': key, 'kind': 'geometry', 'head': name, 'blend': blend,
                   **score(y[va], q, anchor[va])}
            rows.append(row); geometry_rows.append(row); predictions[key] = q
        pd.DataFrame(rows).to_csv(out / 'validation_candidates.csv', index=False)
        print('geometry validation', name, [r for r in rows if r.get('head') == name], flush=True)
    selected = choose_geometry(geometry_rows, direct['trajectory'])
    pd.DataFrame(rows).to_csv(out / 'validation_candidates.csv', index=False)
    np.savez_compressed(out / 'validation_predictions.npz', **predictions)
    write_json(out / 'selection.json', {'state': 'FROZEN_BEFORE_TEST', 'direct_reference': direct,
        'selected': selected, 'model_sha256': {p.name: digest(p) for p in out.glob('*.joblib')},
        'protocol_sha256': digest(out / 'protocol.json')})
    status(out, 'VALIDATION_FROZEN', direct=direct['model'], selected=selected['model'])
    print('FROZEN', direct, selected, flush=True)


def selected_predictions(out, data, mask, selection):
    x, anchor = data['weather'][mask], data['anchor'][mask]
    predictions = {name: predict_direct(out, name, x, anchor)
                   for name in BASELINES + ['delta_' + c for c in CONFIGS]}
    reference = predictions[selection['direct_reference']['model']]
    for config in CONFIGS:
        name = 'geometry_' + config
        internal = np.c_[joblib.load(out / f'{name}.joblib').predict(x), np.zeros(len(x))]
        for blend in [.25, .5, 1.]:
            predictions[f'{name}_b{int(blend * 100)}'] = reconstruct(reference, anchor, internal, blend)
    predictions['strong_direct'] = reference
    predictions['selected'] = predictions[selection['selected']['model']]
    return predictions


def test(out):
    check_sources(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'VALIDATION_FROZEN':
        raise RuntimeError('Frozen, untested run required')
    selection = json.loads((out / 'selection.json').read_text())
    if digest(out / 'protocol.json') != selection['protocol_sha256']:
        raise RuntimeError('Protocol changed after selection')
    for name, sha in selection['model_sha256'].items():
        if digest(out / name) != sha:
            raise RuntimeError('Selected model changed')
    if (out / 'test_summary.csv').exists():
        raise RuntimeError('Test outputs already exist')
    data = np.load(BASE / 'dataset.npz'); te = data['test']
    y, anchor = data['actual'][te], data['anchor'][te]
    times = pd.to_datetime(data['times_ns'][te], utc=True)
    predictions = selected_predictions(out, data, te, selection)
    labels = pd.read_parquet(out / 'window_labels.parquet').set_index('issue_time').loc[times]
    if not all(q.shape == y.shape and np.isfinite(q).all() for q in predictions.values()):
        raise AssertionError('Unaligned or invalid predictions')
    panel = pd.DataFrame({'issue_time': np.repeat(times, 16),
        'horizon_minutes': np.tile(np.arange(1, 17) * 15, len(y)),
        'actual_pu': y.ravel(), 'anchor_pu': np.repeat(anchor, 16),
        'population': np.repeat(labels.population.to_numpy(), 16)})
    for name, q in predictions.items():
        panel[name] = q.ravel()
    panel.to_parquet(out / 'test_predictions.parquet', index=False)
    summary, effects = [], []
    pairs = [('selected', 'np001_extra_trees'), ('strong_direct', 'np001_extra_trees'),
             ('selected', 'strong_direct')]
    for h in PREFIXES:
        losses = {name: structure_losses(y[:, :h], q[:, :h], anchor) for name, q in predictions.items()}
        groups = ['all', *GROUPS] if h == 16 else ['all']
        for group in groups:
            mask = np.ones(len(y), bool) if group == 'all' else labels.population.eq(group).to_numpy()
            if not mask.any():
                continue
            for name, metrics in losses.items():
                for metric, values in metrics.items():
                    summary.append({'model': name, 'population': group, 'horizon_minutes': 15 * h,
                         'metric': metric, 'windows': int(mask.sum()), 'rmse_pu': float(np.sqrt(values[mask].mean()))})
            for candidate, reference in pairs:
                for metric in losses[candidate]:
                    for days in [3, 7, 14]:
                        effect = paired_effect(times[mask], losses[candidate][metric][mask], losses[reference][metric][mask], days)
                        effects.append({'candidate': candidate, 'reference': reference, 'population': group,
                            'horizon_minutes': 15 * h, 'metric': metric, 'windows': int(mask.sum()), **effect})
    pd.DataFrame(summary).to_csv(out / 'test_summary.csv', index=False)
    pd.DataFrame(effects).to_csv(out / 'paired_intervals.csv', index=False)
    status(out, 'COMPLETE', windows=len(y), models=len(predictions), selected=selection['selected']['model'])
    e = pd.DataFrame(effects)
    print(e[(e.block_days == 7) & (e.horizon_minutes == 240) & e.population.eq('all')].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'validate', 'test'], required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'validate': validate, 'test': test}[args.phase](args.output)


if __name__ == '__main__':
    main()
