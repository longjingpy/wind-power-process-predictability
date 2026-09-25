"""NP008: information roles and update dependence in a small process model.

Reuse NP007's multinomial logistic fit, fixed mean-loss L2 penalty and mature
label protocol. The experiment varies representations, not information access
dates or hyperparameters. Existing NP007 full rolling forecasts are reused.

The added absolute changes control ordinary nonlinear amplitude effects.
Excess total variation is sum|dv|-|v_last-v_first|, divided by10. This is a
mathematical control designed here, motivated by the current event paper's
S26 distinction between endpoint change and internal evolution (unpublished,
no DOI); it is not a claimed new atmospheric mechanism. Use window boundaries
and internal native-hour nodes, not every interpolated quarter-hour value.
Boundaries may themselves be interpolated and that limitation is retained.

Gneiting & Raftery (2007), doi:10.1198/016214506000001437, and Kunsch (1989),
doi:10.1214/aos/1176347265: inherited proper losses and shared calendar blocks
compare the same targets. Interaction contrasts use absolute paired loss
differences rather than ratios of signed improvements.
"""
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from next_paper_causal_calibration import (ROOT, OUT as PARENT, BASE, LEADS, STATES, DAY, MINUTE,
    MATURITY, frozen as check_parent, read_inputs, snapshot, apply_map, config_name,
    digest, write_json, status)
from next_paper_leadtime import read_data, loss_difference
from next_paper_probability import probability_losses, paired_loss_effect

OUT = ROOT / 'outputs/next_paper/np008'
GROUPS = {'frequency': [], 'power': [0, 1], 'power_mean': [0, 1, 4],
          'power_trend': [0, 1, 2, 3, 4], 'weather': [2, 3, 4, 5],
          'additive': [0, 1, 2, 3, 4, 5], 'full': list(range(7)),
          'magnitude': [0, 1, 2, 3, 4, 7, 8],
          'magnitude_range': [0, 1, 2, 3, 4, 5, 7, 8],
          'magnitude_reversal': [0, 1, 2, 3, 4, 7, 8, 9]}
MODES = ['static', 'rolling']
INCREMENTS = [('power', 'frequency'), ('weather', 'frequency'), ('full', 'power'), ('full', 'weather'),
              ('power_mean', 'power'), ('power_trend', 'power_mean'), ('additive', 'power_trend'),
              ('full', 'additive'), ('magnitude_range', 'magnitude'),
              ('magnitude_reversal', 'magnitude'), ('magnitude_reversal', 'magnitude_range')]


def coarse_excess(weather, times_ns):
    forecast = np.asarray(weather).reshape(-1, 17, 8)
    times_ns = np.asarray(times_ns)
    if len(times_ns) != len(forecast) or (times_ns % (15 * MINUTE) != 0).any():
        raise ValueError('Aligned native quarter-hour target times required')
    speed = np.hypot(forecast[:, :, 4], forecast[:, :, 5])
    valid_times = times_ns[:, None] + np.arange(17)[None] * 15 * MINUTE
    keep = valid_times % (60 * MINUTE) == 0
    keep[:, [0, -1]] = True
    nodes = np.stack([np.pad(np.flatnonzero(row), (0, 6 - row.sum()), constant_values=16) for row in keep])
    sampled = np.take_along_axis(speed, nodes, axis=1)
    value = np.abs(np.diff(sampled, axis=1)).sum(axis=1) - np.abs(speed[:, -1] - speed[:, 0])
    return np.maximum(value, 0) / 10, nodes


def augment(context, weather, times_ns):
    variation, nodes = coarse_excess(weather, times_ns)
    augmented = np.concatenate([context, np.abs(context[:, :, 2:4]),
                                np.broadcast_to(variation[None, :, None], (*context.shape[:2], 1))], axis=2)
    return augmented, nodes


def prepare(out):
    _, selected = check_parent(PARENT)
    if json.loads((PARENT / 'status.json').read_text())['state'] != 'COMPLETE' or out.exists():
        raise RuntimeError('Completed parent and a fresh output directory required')
    data = read_inputs(PARENT); base = read_data(BASE)
    pool = base['validation'] | base['test']
    np.testing.assert_array_equal(data['times_ns'], base['times_ns'][pool])
    context, nodes = augment(data['context'], base['weather'][pool], data['times_ns'])
    arrays = {k: data[k] for k in ['times_ns', 'outcome', 'state', 'validation', 'test', 'original_validation']}
    arrays.update(context=context, coarse_nodes=nodes)
    chosen = selected['selected']; tag = 'validation_' + config_name(chosen['window_days'], chosen['lambda_'])
    out.mkdir(parents=True)
    np.savez_compressed(out / 'dataset.npz', **arrays)
    tracked = [Path(__file__), ROOT / 'script/next_paper_causal_calibration.py', ROOT / 'script/next_paper_probability.py',
               ROOT / 'script/next_paper_leadtime.py', BASE / 'dataset.npz',
               PARENT / 'protocol.json', PARENT / 'selection.json', PARENT / 'dataset.npz']
    for prefix in [tag, 'test']:
        tracked += [PARENT / f'{prefix}_predictions.npz', PARENT / f'{prefix}_snapshots.joblib']
    protocol = dict(experiment='NP008', stage='EXPLORATORY_MATCHED_INFORMATION_AND_UPDATE',
        question='Which power/forecast features and update policy explain process probability skill?',
        leads=LEADS, groups=GROUPS, modes=MODES, window_days=chosen['window_days'], lambda_=chosen['lambda_'],
        hyperparameter_source='NP007 shared selection, fixed before NP008 scoring; no further search or per-group selection',
        features=['issue_power', 'issue_power_squared', 'GFS_delta/5', 'JMA_delta/5', 'GFS_mean/20',
                  'GFS_range/10', 'issue_power*GFS_delta/5', 'abs_GFS_delta/5', 'abs_JMA_delta/5', 'GFS_coarse_excess_TV/10'],
        coarse_variation='Window boundaries and internal native hourly nodes; repeated final node pads variable lengths; boundary interpolation remains',
        maturity='T+4h+1min <= UTC midnight update <= issue; preceding56day target starts',
        reuse='frequency static/rolling and full rolling reuse frozen NP007 predictions and snapshots exactly',
        target='Unchanged17node future4h path and four event classes; paired6496test windows',
        comparisons=INCREMENTS, update_comparison='rolling versus static for every group',
        interaction_comparisons=['update benefit full versus power', 'update benefit full versus weather',
                                'range increment rolling versus static', 'coarse reversal increment rolling versus static'],
        uncertainty='Paired3/7/14day blocks,2000draws,seed41; primary7d; fixed sequential predictions, not refitted bootstrap',
        interpretation='Predictive information effects, not causal atmospheric attribution, new data products or independent confirmation',
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in tracked}, dataset_sha256=digest(out / 'dataset.npz'))
    write_json(out / 'protocol.json', protocol)
    status(out, 'PREPARED', validation_windows=int(data['validation'].sum()), test_windows=int(data['test'].sum()))
    print('PREPARED', protocol['window_days'], protocol['lambda_'], context.shape, flush=True)


def sources(out):
    p = json.loads((out / 'protocol.json').read_text())
    for name, sha in p['source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f'Changed source: {name}')
    if digest(out / 'dataset.npz') != p['dataset_sha256']:
        raise RuntimeError('Changed prepared dataset')
    return p


def run_phase(out, data, protocol, split):
    target = np.flatnonzero(data[split]); n = len(target)
    days, strength = protocol['window_days'], protocol['lambda_']
    parent_tag = 'test' if split == 'test' else 'validation_' + config_name(days, strength)
    parent_records = joblib.load(PARENT / f'{parent_tag}_snapshots.joblib')
    with np.load(PARENT / f'{parent_tag}_predictions.npz') as file:
        parent_prediction = {key: file[key] for key in file.files if key.startswith('reference__')}
    first_update = ((data['times_ns'][target].min() - max(LEADS) * MINUTE) // DAY) * DAY
    predictions, records = {}, {}
    for j, lead in enumerate(LEADS):
        issue = data['times_ns'][target] - lead * MINUTE
        for group, columns in GROUPS.items():
            x = data['context'][j][:, columns]
            for mode in MODES:
                key = f'{lead}__{group}__{mode}'
                schedule = np.full(n, first_update, np.int64) if mode == 'static' else issue // DAY * DAY
                inherited = group == 'frequency' or (group == 'full' and mode == 'rolling')
                parent_mode = f'{mode}_frequency' if group == 'frequency' else 'rolling_context_only'
                parent_key = f'reference__{lead}__{parent_mode}'
                if inherited:
                    predictions[key] = parent_prediction[parent_key]
                else:
                    predictions[key] = np.empty((n, 4))
                for stamp in np.unique(schedule):
                    take = np.flatnonzero(schedule == stamp)
                    if inherited:
                        record = parent_records[f'{parent_key}__{stamp}'].copy()
                    else:
                        record = snapshot(data, x, int(stamp), days, strength)
                        predictions[key][take] = apply_map(record, x[target[take]])
                    record['origin'] = 'NP007' if inherited else 'NP008'
                    assert stamp <= issue[take].min()
                    assert record['latest_label_available_ns'] is None or record['latest_label_available_ns'] <= stamp
                    records[f'{key}__{stamp}'] = record
        print(datetime.now(timezone.utc).isoformat(), split, 'lead', lead, 'snapshots', len(records), flush=True)
    np.savez_compressed(out / f'{split}_predictions.npz', **predictions)
    joblib.dump(records, out / f'{split}_snapshots.joblib', compress=3)
    audit = []
    for key, value in records.items():
        lead, group, mode, stamp = key.split('__')
        audit.append(dict(lead_minutes=int(lead), group=group, mode=mode, update_ns=int(stamp),
            **{k: value[k] for k in ['origin', 'kind', 'rows', 'row_sha256', 'latest_label_available_ns', 'n_iter']}))
    pd.DataFrame(audit).to_csv(out / f'{split}_update_audit.csv', index=False)
    return predictions


def evaluate(out, data, split, predictions):
    mask = data[split]; y = data['outcome'][mask]; times = pd.to_datetime(data['times_ns'][mask], utc=True)
    losses = {k: probability_losses(y, p) for k, p in predictions.items()}
    scores, biases, effects, interactions = [], [], [], []
    comparisons = []
    for lead in LEADS:
        for mode in MODES:
            comparisons += [(f'{lead}__{a}__{mode}', f'{lead}__{b}__{mode}', 'information') for a, b in INCREMENTS]
        comparisons += [(f'{lead}__{g}__rolling', f'{lead}__{g}__static', 'update') for g in GROUPS]
    for state in ['all', *STATES]:
        take = np.ones(len(y), bool) if state == 'all' else data['state'][mask] == state
        for key, metrics in losses.items():
            lead, group, mode = key.split('__')
            for metric, values in metrics.items():
                scores.append(dict(lead_minutes=int(lead), group=group, mode=mode, state=state, metric=metric,
                    windows=int(take.sum()), loss=float(values[take].mean())))
            for event, classes in [('up', [1, 3]), ('down', [2, 3]), ('return', [3])]:
                value = predictions[key][take][:, classes].sum(axis=1); actual = np.isin(y[take], classes)
                biases.append(dict(lead_minutes=int(lead), group=group, mode=mode, state=state, event=event,
                    windows=len(value), mean_probability=float(value.mean()), observed_fraction=float(actual.mean()),
                    probability_bias=float(value.mean() - actual.mean())))
        if split == 'test':
            for candidate, reference, kind in comparisons:
                for metric in ['multiclass_brier', 'return_brier', 'up_brier', 'down_brier']:
                    for days in ([3, 7, 14] if state == 'all' else [7]):
                        effects.append(dict(candidate=candidate, reference=reference, comparison=kind, state=state, metric=metric,
                            windows=int(take.sum()), **paired_loss_effect(times[take], losses[candidate][metric][take], losses[reference][metric][take], days)))
    if split == 'test':
        for lead in LEADS:
            for metric in ['multiclass_brier', 'return_brier', 'up_brier', 'down_brier']:
                def loss(group, mode):
                    return losses[f'{lead}__{group}__{mode}'][metric]
                contrasts = {
                    'update_full_minus_power': (loss('full', 'rolling') - loss('full', 'static'), loss('power', 'rolling') - loss('power', 'static')),
                    'update_full_minus_weather': (loss('full', 'rolling') - loss('full', 'static'), loss('weather', 'rolling') - loss('weather', 'static')),
                    'range_increment_rolling_minus_static': (loss('additive', 'rolling') - loss('power_trend', 'rolling'), loss('additive', 'static') - loss('power_trend', 'static')),
                    'reversal_increment_rolling_minus_static': (loss('magnitude_reversal', 'rolling') - loss('magnitude', 'rolling'), loss('magnitude_reversal', 'static') - loss('magnitude', 'static'))}
                for name, (a, b) in contrasts.items():
                    for days in [3, 7, 14]:
                        interactions.append(dict(lead_minutes=lead, metric=metric, contrast=name, block_days=days,
                            interpretation='Negative means the candidate contrast reduces loss more', **loss_difference(times, a, b, days)))
        pd.DataFrame(effects).to_csv(out / 'paired_intervals.csv', index=False)
        pd.DataFrame(interactions).to_csv(out / 'interaction_intervals.csv', index=False)
    pd.DataFrame(scores).to_csv(out / f'{split}_summary.csv', index=False)
    pd.DataFrame(biases).to_csv(out / f'{split}_conditional_bias.csv', index=False)
    print(pd.DataFrame(scores).query("lead_minutes == 720 and state == 'all' and metric == 'multiclass_brier'").to_string(index=False), flush=True)


def validate(out):
    p = sources(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'PREPARED':
        raise RuntimeError('Fresh prepared run required')
    data = read_inputs(out); predictions = run_phase(out, data, p, 'validation')
    evaluate(out, data, 'validation', predictions)
    write_json(out / 'freeze.json', dict(state='FROZEN_BEFORE_TEST_NO_NEW_SELECTION', frozen_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(out / 'protocol.json'), validation_sha256={p.name: digest(p) for p in out.glob('validation_*')}))
    status(out, 'VALIDATION_FROZEN')


def frozen(out):
    p = sources(out); lock = json.loads((out / 'freeze.json').read_text())
    assert digest(out / 'protocol.json') == lock['protocol_sha256']
    for name, sha in lock['validation_sha256'].items():
        assert digest(out / name) == sha
    return p


def test(out):
    p = frozen(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'VALIDATION_FROZEN' or list(out.glob('test_*')):
        raise RuntimeError('Frozen untested run required')
    data = read_inputs(out); predictions = run_phase(out, data, p, 'test')
    evaluate(out, data, 'test', predictions)
    status(out, 'COMPLETE', test_windows=int(data['test'].sum()), new_hyperparameter_search=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'validate', 'test'], required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {'prepare': prepare, 'validate': validate, 'test': test}[args.phase](args.output)
