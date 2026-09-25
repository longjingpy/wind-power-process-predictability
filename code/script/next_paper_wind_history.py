"""NP003: matched information experiment for causal SCADA wind histories.

Observation motivation: Lochmann et al. (2023), Analysing wind power ramp
events and improving very short-term wind power predictions by including wind
speed observations, doi:10.1002/we.2816, abstract/results. Historical onsite
wind may add information beyond power and issued NWP; that is a hypothesis,
not a transferred result. Actual wind timestamps are inherited from
prepare_pizhou_policy_v24: complete 15-minute means ending at issue minus 1 min.

The sorted-history control retains each row's multiset of past wind values
but removes their order. It adapts the current event manuscript's S26 temporal
order controls (unpublished, no DOI) to pre-issue inputs. It tests the use of
order within this learning setup, not a causal atmospheric mechanism.

ExtraTrees, endpoint/geometry targets, and paired calendar blocks are reused
from NP002/NP001 with the same cited method sources and fixed configurations.
All arms are retrained on identical complete-history support; three matched
seeds distinguish information effects from one random tree realization.
"""
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from next_paper_trajectory import (ROOT, POWER, OUT as BASE, PREFIXES,
    digest, write_json, status, structure_losses, paired_effect)
from next_paper_structure import (OUT as PREVIOUS, GROUPS, estimator, geometry,
    reconstruct, predict_direct, check_sources)

OUT = ROOT / 'outputs/next_paper/np003'
ARMS = ['base', 'wind_history', 'wind_sorted']
SEEDS = [41, 42, 43]
STATES = ['wind_falling', 'wind_middle', 'wind_rising']


def wind_history(frame):
    values = np.column_stack([frame.wind.shift(k).to_numpy() / 20 for k in range(1, 17)])
    valid = np.isfinite(values).all(axis=1) & np.isfinite(frame.wind.to_numpy())
    trend = frame.wind.to_numpy() - frame.wind.shift(4).to_numpy()
    return values, trend, valid


def state_labels(trend, train):
    low, high = np.quantile(trend[train], [.25, .75])
    states = np.select([~np.isfinite(trend), trend <= low, trend >= high],
                       ['wind_missing', 'wind_falling', 'wind_rising'], default='wind_middle')
    return states, [float(low), float(high)]


def feature_arms(base, wind):
    return {'base': base, 'wind_history': np.c_[base, wind],
            'wind_sorted': np.c_[base, np.sort(wind, axis=1)]}


def original_prediction(data, mask):
    selection = json.loads((PREVIOUS / 'selection.json').read_text())
    x, anchor = data['weather'][mask], data['anchor'][mask]
    direct = predict_direct(PREVIOUS, selection['direct_reference']['model'], x, anchor)
    choice = selection['selected']
    if choice['kind'] == 'direct':
        return predict_direct(PREVIOUS, choice['model'], x, anchor)
    internal = np.c_[joblib.load(PREVIOUS / f"{choice['head']}.joblib").predict(x), np.zeros(len(x))]
    return reconstruct(direct, anchor, internal, choice['blend'])


def populations(labels):
    groups = {'all': np.ones(len(labels), bool)}
    groups.update({f'event:{g}': labels.population.eq(g).to_numpy() for g in GROUPS})
    groups.update({f'state:{s}': labels.wind_state.eq(s).to_numpy() for s in STATES})
    groups.update({f'composite:{s}': (labels.population.eq('complete_composite') & labels.wind_state.eq(s)).to_numpy() for s in STATES})
    return groups


def prepare(out):
    check_sources(PREVIOUS)
    if json.loads((PREVIOUS / 'status.json').read_text())['state'] != 'COMPLETE':
        raise RuntimeError('Frozen NP002 required')
    out.mkdir(parents=True, exist_ok=False)
    data = np.load(BASE / 'dataset.npz')
    frame = pd.read_parquet(POWER)
    wind, trend, valid = wind_history(frame)
    masks = {split: data[split] & valid for split in ['train', 'validation', 'test']}
    states, cutoffs = state_labels(trend, masks['train'])
    np.savez_compressed(out / 'wind_inputs.npz', wind=wind, trend=trend, **masks)
    labels = pd.read_parquet(PREVIOUS / 'window_labels.parquet')
    if not np.array_equal(pd.DatetimeIndex(labels.issue_time).as_unit('ns').asi8, data['times_ns']):
        raise ValueError('Inherited labels and issue grid differ')
    labels['wind_state'] = states
    labels.to_parquet(out / 'window_labels.parquet', index=False)
    support = []
    links = pd.read_parquet(PREVIOUS / 'event_links.parquet')
    for split, mask in masks.items():
        for name, subset in populations(labels).items():
            times = frame.index[mask & subset]
            complete = links[links.issue_time.isin(times)]
            support.append({'split': split, 'population': name, 'windows': len(times),
                'occupied_7d_blocks': len(np.unique(times.as_unit('ns').asi8 // pd.Timedelta(days=7).value)),
                'unique_complete_primitive': complete.loc[complete.event_level.eq('primitive'), 'event_id'].nunique(),
                'unique_complete_composite': complete.loc[complete.event_level.eq('composite'), 'event_id'].nunique()})
    pd.DataFrame(support).to_csv(out / 'support.csv', index=False)
    tracked = [PREVIOUS / 'protocol.json', PREVIOUS / 'selection.json',
        PREVIOUS / 'test_predictions.parquet', PREVIOUS / 'window_labels.parquet',
        PREVIOUS / 'event_links.parquet', *PREVIOUS.glob('*.joblib')]
    protocol = {'experiment': 'NP003', 'stage': 'EXPLORATORY_INFORMATION_EXTENSION_AFTER_VIEWED_NP002',
        'question': 'Does pre-issue SCADA wind history, and its ordering, add useful trajectory information?',
        'wind_source': str(POWER.relative_to(ROOT)),
        'wind_measurement': '33-turbine mean of complete trailing-15min SCADA wind means, ending at issue minus1min; height/location verification not added',
        'lags_minutes': [15 * k for k in range(1, 17)], 'scaling': 'wind / 20, inherited fixed scale',
        'arms': ARMS, 'features': {'base': 170, 'wind_history': 186, 'wind_sorted': 186},
        'common_support': {s: int(m.sum()) for s, m in masks.items()},
        'excluded_for_wind_history': {s: int((data[s] & ~valid).sum()) for s in masks},
        'states': 'Training-only quartiles of current SCADA wind minus its 1h lag; observable trends, not weather mechanisms',
        'state_cutoffs_ms': cutoffs, 'seeds': SEEDS,
        'direct_config': 'et_l32', 'geometry_config': 'et_l128', 'geometry_blend': 1.,
        'model_parameters': '150 trees, max_depth16, max_features1, n_jobs4; no parameter search',
        'selection': 'Minimum common validation trajectory MSE over six three-seed ensemble forecasts and frozen NP002',
        'primary_information_comparison': 'wind_history_split versus base_split, regardless of selected delivery forecast',
        'order_control': 'wind_history_split versus wind_sorted_split; sorted history retains row multiset',
        'reporting': 'All seeds and ensemble forecasts; all prefixes overall, event/state groups at4h; paired3/7/14d,2000 draws, primary7d',
        'baseline_context': 'Matched arms retrained on common support; frozen NP002 trained on its earlier larger cohort, so its comparison is operational, not pure information attribution',
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), POWER, *tracked]},
        'prepared_sha256': {p: digest(out / p) for p in ['wind_inputs.npz', 'window_labels.parquet']}}
    write_json(out / 'protocol.json', protocol)
    status(out, 'PREPARED', support=protocol['common_support'])
    print(json.dumps({k: protocol[k] for k in ['common_support', 'excluded_for_wind_history', 'state_cutoffs_ms']}), flush=True)


def sources(out):
    check_sources(PREVIOUS)
    p = json.loads((out / 'protocol.json').read_text())
    for file, sha in p['source_sha256'].items():
        if digest(ROOT / file) != sha:
            raise RuntimeError(f'Source changed: {file}')
    for file, sha in p['prepared_sha256'].items():
        if digest(out / file) != sha:
            raise RuntimeError(f'Prepared input changed: {file}')
    return p


def arm_predictions(models, x, anchor, arm):
    predictions = {}
    for seed, direct_model, geometry_model in models:
        direct = np.clip(anchor[:, None] + direct_model.predict(x), 0, 1)
        internal = np.c_[geometry_model.predict(x), np.zeros(len(x))]
        predictions[f'{arm}_direct_s{seed}'] = direct
        predictions[f'{arm}_split_s{seed}'] = reconstruct(direct, anchor, internal, 1.)
    for mode in ['direct', 'split']:
        predictions[f'{arm}_{mode}'] = np.mean([predictions[f'{arm}_{mode}_s{s}'] for s in SEEDS], axis=0)
    return predictions


def validate(out):
    sources(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'PREPARED':
        raise RuntimeError('Fresh prepared run required')
    data, extra = np.load(BASE / 'dataset.npz'), np.load(out / 'wind_inputs.npz')
    tr, va = extra['train'], extra['validation']
    x = feature_arms(data['weather'], extra['wind'])
    y, anchor = data['actual'], data['anchor']
    delta = y[tr] - anchor[tr, None]
    internal = geometry(y[tr], anchor[tr])[:, :-1]
    predictions = {'np002_frozen': original_prediction(data, va)}
    rows = []
    for arm in ARMS:
        models = []
        for seed in SEEDS:
            direct = estimator('et_l32').set_params(random_state=seed).fit(x[arm][tr], delta)
            head = estimator('et_l128').set_params(random_state=seed).fit(x[arm][tr], internal)
            models.append((seed, direct, head))
            print('fit', arm, seed, flush=True)
        joblib.dump(models, out / f'{arm}_models.joblib', compress=3)
        predictions.update(arm_predictions(models, x[arm][va], anchor[va], arm))
        print('validation', arm, float(np.mean((predictions[f'{arm}_split'] - y[va]) ** 2)), flush=True)
    for name, q in predictions.items():
        scores = {m: float(v.mean()) for m, v in structure_losses(y[va], q, anchor[va]).items()}
        rows.append({'model': name, 'selection_eligible': name == 'np002_frozen' or name.endswith(('direct', 'split')), **scores})
    chosen = min([r for r in rows if r['selection_eligible']], key=lambda r: (r['trajectory'], r['model']))
    pd.DataFrame(rows).to_csv(out / 'validation_candidates.csv', index=False)
    np.savez_compressed(out / 'validation_predictions.npz', **predictions)
    write_json(out / 'selection.json', {'state': 'FROZEN_BEFORE_TEST', 'selected': chosen,
        'model_sha256': {p.name: digest(p) for p in out.glob('*.joblib')},
        'protocol_sha256': digest(out / 'protocol.json')})
    status(out, 'VALIDATION_FROZEN', selected=chosen['model'])
    print('FROZEN', chosen, flush=True)


def test(out):
    sources(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'VALIDATION_FROZEN':
        raise RuntimeError('Frozen, untested run required')
    selection = json.loads((out / 'selection.json').read_text())
    if digest(out / 'protocol.json') != selection['protocol_sha256']:
        raise RuntimeError('Changed protocol')
    for name, sha in selection['model_sha256'].items():
        if digest(out / name) != sha:
            raise RuntimeError('Changed model')
    if (out / 'test_summary.csv').exists():
        raise RuntimeError('Test already exists')
    data, extra = np.load(BASE / 'dataset.npz'), np.load(out / 'wind_inputs.npz')
    te = extra['test']; anchor = data['anchor'][te]; y = data['actual'][te]
    times = pd.to_datetime(data['times_ns'][te], utc=True)
    x = feature_arms(data['weather'][te], extra['wind'][te])
    predictions = {'np002_frozen': original_prediction(data, te)}
    for arm in ARMS:
        predictions.update(arm_predictions(joblib.load(out / f'{arm}_models.joblib'), x[arm], anchor, arm))
    predictions['selected'] = predictions[selection['selected']['model']]
    labels = pd.read_parquet(out / 'window_labels.parquet').set_index('issue_time').loc[times]
    if not all(q.shape == y.shape and np.isfinite(q).all() for q in predictions.values()):
        raise AssertionError('Invalid or misaligned forecasts')
    panel = pd.DataFrame({'issue_time': np.repeat(times, 16),
        'horizon_minutes': np.tile(np.arange(1, 17) * 15, len(y)),
        'actual_pu': y.ravel(), 'anchor_pu': np.repeat(anchor, 16),
        'population': np.repeat(labels.population.to_numpy(), 16),
        'wind_state': np.repeat(labels.wind_state.to_numpy(), 16)})
    for name, q in predictions.items():
        panel[name] = q.ravel()
    panel.to_parquet(out / 'test_predictions.parquet', index=False)
    pairs = [('wind_history_split', 'base_split'), ('wind_sorted_split', 'base_split'),
        ('wind_history_split', 'wind_sorted_split'), ('selected', 'np002_frozen'),
        ('wind_history_direct', 'base_direct'), ('base_split', 'np002_frozen'),
        ('wind_history_split', 'np002_frozen')]
    summary, effects, seeds = [], [], []
    for h in PREFIXES:
        losses = {name: structure_losses(y[:, :h], q[:, :h], anchor) for name, q in predictions.items()}
        groups = populations(labels) if h == 16 else {'all': np.ones(len(y), bool)}
        for group, mask in groups.items():
            if not mask.any():
                continue
            for name, metrics in losses.items():
                for metric, value in metrics.items():
                    summary.append({'model': name, 'population': group, 'horizon_minutes': 15 * h,
                         'metric': metric, 'windows': int(mask.sum()), 'rmse_pu': float(np.sqrt(value[mask].mean()))})
            for candidate, reference in pairs:
                for metric in losses[candidate]:
                    for days in [3, 7, 14]:
                        effect = paired_effect(times[mask], losses[candidate][metric][mask], losses[reference][metric][mask], days)
                        effects.append({'candidate': candidate, 'reference': reference, 'population': group,
                            'horizon_minutes': 15 * h, 'metric': metric, 'windows': int(mask.sum()), **effect})
            if h == 16 and group in ['all', 'event:complete_composite']:
                for seed in SEEDS:
                    for metric in losses['base_split']:
                        a = losses[f'wind_history_split_s{seed}'][metric][mask].mean()
                        b = losses[f'base_split_s{seed}'][metric][mask].mean()
                        seeds.append({'population': group, 'metric': metric, 'seed': seed,
                                      'relative_rmse_reduction_pct': 100 * (1 - np.sqrt(a / b))})
    pd.DataFrame(summary).to_csv(out / 'test_summary.csv', index=False)
    pd.DataFrame(effects).to_csv(out / 'paired_intervals.csv', index=False)
    pd.DataFrame(seeds).to_csv(out / 'seed_effects.csv', index=False)
    status(out, 'COMPLETE', windows=len(y), selected=selection['selected']['model'])
    effect = pd.DataFrame(effects)
    print(effect[(effect.block_days == 7) & effect.population.isin(['all', 'event:complete_composite']) &
          effect.candidate.eq('wind_history_split') & effect.reference.eq('base_split') & (effect.horizon_minutes == 240)].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'validate', 'test'], required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'validate': validate, 'test': test}[args.phase](args.output)


if __name__ == '__main__':
    main()
