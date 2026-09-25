"""NP007: static versus causally updated process-probability calibration.

Kull et al. (2019), Beyond temperature scaling: Obtaining well-calibrated
multiclass probabilities with Dirichlet calibration, NeurIPS32, Section3,
https://papers.neurips.cc/paper_files/paper/2019/file/8ca01ea920679a0fe3728441494041b9-Paper.pdf:
log probabilities followed by a linear map and softmax provide a multiclass
calibration map. We use ordinary L2, not the paper's specialized ODIR penalty.
The project's lidar_probability_calibration_v18.py already uses multinomial
logistic calibration; this experiment adds available context and daily causal
refitting. Those choices are experimental hypotheses, not a new algorithm.

Scikit-learn1.7.2 _logistic.py uses l2_reg_strength=1/(C*n_samples) for lbfgs.
Set C=1/(n*lambda) to hold the mean-loss penalty fixed across rolling cohorts.
Gneiting & Raftery (2007), doi:10.1198/016214506000001437, and the inherited
Kunsch-motivated paired calendar blocks supply proper scores and intervals.

This calibrates mutually exclusive event probabilities, not member paths or
marginal intervals. Targets mature four hours plus one minute after T. No
update may see an outcome that was unavailable at its issue-time cutoff.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from next_paper_leadtime import ROOT, OUT as BASE, LEADS, STATES, read_data, digest, write_json, status
from next_paper_observation_domain import OUT as PREVIOUS, frozen as check_parent
from next_paper_probability import probability_losses, paired_loss_effect

OUT = ROOT / 'outputs/next_paper/np007'
SOURCES = ['classifier', 'joint']
WINDOWS = [28, 56]
LAMBDAS = [.001, .01, .1]
MINUTE = pd.Timedelta(minutes=1).value
DAY = pd.Timedelta(days=1).value
MATURITY = 241 * MINUTE
VALIDATION_START = pd.Timestamp('2024-09-01', tz='UTC').value
MODES = ['static_global', 'static_context', 'rolling_global', 'rolling_context']


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()


def training_rows(times, update, days):
    times = np.asarray(times, dtype=np.int64)
    return np.flatnonzero((times >= update - days * DAY) & (times + MATURITY <= update))


def context_features(data, j):
    weather = data['weather'].reshape(-1, 17, 8)
    gfs = np.hypot(weather[:, :, 4], weather[:, :, 5])
    jma = np.hypot(weather[:, :, 0], weather[:, :, 1])
    change = (gfs[:, -1] - gfs[:, 0]) / 5
    anchor = data['anchor'][j]
    return np.c_[anchor, anchor ** 2, change, (jma[:, -1] - jma[:, 0]) / 5,
                 gfs.mean(axis=1) / 20, np.ptp(gfs, axis=1) / 10, anchor * change]


def calibration_features(probability, context, mode):
    if mode == 'context_only':
        return context
    logp = np.log(np.clip(probability, 1e-6, 1))
    return np.c_[logp, context] if mode.endswith('context') else logp


def fit_map(x, y, strength):
    counts = np.bincount(y, minlength=4)
    if len(y) < 100 or (counts == 0).any():
        return dict(kind='frequency_fallback', probability=(counts + 1) / (len(y) + 4),
                    class_counts=counts, n_iter=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = LogisticRegression(C=1 / (len(y) * strength), solver='lbfgs',
            penalty='l2', max_iter=1000, tol=1e-7, random_state=41).fit(x, y)
    if not np.array_equal(model.classes_, np.arange(4)):
        raise ValueError('Four classes required after the fallback gate')
    return dict(kind='logistic', coefficient=model.coef_, intercept=model.intercept_,
                class_counts=counts, n_iter=int(model.n_iter_.max()), C=model.C)


def apply_map(record, x):
    if record['kind'] != 'logistic':
        return np.tile(record['probability'], (len(x), 1))
    return softmax(x @ record['coefficient'].T + record['intercept'], axis=1)


def prepare(out):
    check_parent(PREVIOUS)
    if out.exists():
        raise RuntimeError('A new output directory is required')
    base = read_data(BASE)
    pool = base['validation'] | base['test']
    probabilities = np.empty((2, len(LEADS), pool.sum(), 4))
    tracked = [BASE / 'dataset.npz', BASE / 'protocol.json', PREVIOUS / 'protocol.json', PREVIOUS / 'freeze.json']
    for j, lead in enumerate(LEADS):
        for split in ['validation', 'test']:
            take = base[split][pool]
            for source, parent, field in [('classifier', BASE, 'classifier'), ('joint', PREVIOUS, 'joint')]:
                path = parent / f'{split}_{lead}_power_weather.npz'
                with np.load(path) as values:
                    probabilities[SOURCES.index(source), j, take] = values[field]
                tracked.append(path)
    data = dict(times_ns=base['times_ns'][pool], outcome=base['outcome'][pool],
        state=base['state'][pool], context=np.stack([context_features(base, j)[pool] for j in range(len(LEADS))]),
        probability=probabilities, validation=base['validation'][pool] & (base['times_ns'][pool] >= VALIDATION_START),
        test=base['test'][pool], original_validation=base['validation'][pool],
        training_frequency=np.bincount(base['outcome'][base['train']], minlength=4) / base['train'].sum())
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=-1), 1):
        raise ValueError('Finite normalized inherited probabilities required')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'dataset.npz', **data)
    support = []
    for split in ['original_validation', 'validation', 'test']:
        take = data[split]; time = pd.to_datetime(data['times_ns'][take], utc=True)
        support.append(dict(split=split, windows=int(take.sum()), first_target=str(time.min()), last_target=str(time.max()),
                            **{f'class{k}': int((data['outcome'][take] == k).sum()) for k in range(4)}))
    pd.DataFrame(support).to_csv(out / 'support.csv', index=False)
    tracked += [Path(__file__), ROOT / 'script/next_paper_probability.py', ROOT / 'script/next_paper_leadtime.py',
                ROOT / 'script/lidar_probability_calibration_v18.py', ROOT / 'src/wind_events/paired_probability.py']
    protocol = dict(experiment='NP007', stage='EXPLORATORY_PREQUENTIAL_AFTER_VIEWED_NP006',
        question='Can causal updates and available context repair time-varying process probabilities beyond recent frequency?',
        leads=LEADS, sources=SOURCES, windows_days=WINDOWS, lambda_candidates=LAMBDAS,
        context=['issue_power', 'issue_power_squared', 'GFS_delta_speed/5', 'JMA_delta_speed/5',
                 'GFS_mean_speed/20', 'GFS_speed_range/10', 'issue_power*GFS_delta_speed/5'],
        calibration='Multinomial log-probability linear map, optional7context inputs, softmax; L2, no class weighting',
        regularization='C=1/(n*lambda), preserving mean-loss penalty; installed sklearn1.7.2 source verified',
        modes=MODES, references=['raw', 'training_frequency', 'static_frequency', 'rolling_frequency', 'rolling_context_only'],
        label_available='Target T+4h+1min; include only labels available by UTC-midnight update <= issue T-lead',
        rolling_window='Target start T >= update-Wdays and T+4h+1min <= update',
        static_update='UTC midnight at or before the earliest issue across all six leads in the evaluation phase',
        validation_target_start_utc=str(pd.to_datetime(VALIDATION_START, utc=True)),
        selection='One shared W/lambda minimizes mean multiclass Brier across2sources*6leads*4calibration modes on validation; ties shorterW,strongerlambda',
        fit_pool='Only archived validation/test predictions with mature labels; no in-sample base-tree training predictions',
        fallback='Below100mature windows or missingclass: smoothed (count+1)/(n+4) frequency',
        solver='lbfgs,max_iter1000,tol1e-7; convergence warnings fail explicitly',
        sequential_scope='Early test outcomes may train later updates only after maturity; not a fully frozen independent test',
        output_scope='Normalized event probabilities only; member paths and marginal CRPS not recalibrated',
        inference='Paired3/7/14day target-calendar blocks,2000draws,seed41,primary7d; exploratory pointwise intervals',
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in tracked}, dataset_sha256=digest(out / 'dataset.npz'))
    write_json(out / 'protocol.json', protocol)
    status(out, 'PREPARED', validation_windows=int(data['validation'].sum()), test_windows=int(data['test'].sum()))
    print(pd.DataFrame(support).to_string(index=False), flush=True)


def sources(out):
    p = json.loads((out / 'protocol.json').read_text())
    for file, expected in p['source_sha256'].items():
        if digest(ROOT / file) != expected:
            raise RuntimeError(f'Source changed: {file}')
    if digest(out / 'dataset.npz') != p['dataset_sha256']:
        raise RuntimeError('Prepared inputs changed')
    return p


def read_inputs(out):
    with np.load(out / 'dataset.npz') as source:
        return {k: source[k] for k in source.files}


def config_name(days, strength):
    return f'w{days}_l{strength:g}'


def snapshot(data, x, update, days, strength, frequency=False):
    rows = training_rows(data['times_ns'], update, days)
    if frequency:
        counts = np.bincount(data['outcome'][rows], minlength=4)
        record = dict(kind='frequency', probability=(counts + 1) / (len(rows) + 4), class_counts=counts, n_iter=0)
    else:
        record = fit_map(x[rows], data['outcome'][rows], strength)
    record.update(update_ns=int(update), rows=len(rows), row_sha256=row_hash(rows),
        latest_label_available_ns=int(np.max(data['times_ns'][rows] + MATURITY)) if len(rows) else None,
        earliest_target_ns=int(data['times_ns'][rows].min()) if len(rows) else None)
    return record


def run_phase(out, data, split, days, strength, tag):
    target = np.flatnonzero(data[split]); n = len(target)
    first_update = ((data['times_ns'][target].min() - max(LEADS) * MINUTE) // DAY) * DAY
    predictions, records = {}, {}
    for j, lead in enumerate(LEADS):
        context = data['context'][j]
        issued = data['times_ns'][target] - lead * MINUTE
        updates = issued // DAY * DAY
        for source in ['reference', *SOURCES]:
            if source == 'reference':
                key = f'{source}__{lead}__training_frequency'
                predictions[key] = np.tile(data['training_frequency'], (n, 1))
                modes = ['static_frequency', 'rolling_frequency', 'rolling_context_only']
                probability = None
            else:
                probability = data['probability'][SOURCES.index(source), j]
                predictions[f'{source}__{lead}__raw'] = probability[target]
                modes = MODES
            for mode in modes:
                key = f'{source}__{lead}__{mode}'
                predictions[key] = np.empty((n, 4))
                frequency = mode.endswith('frequency')
                x = context if source == 'reference' else calibration_features(probability, context, mode)
                scheduled = np.full(n, first_update, dtype=np.int64) if mode.startswith('static') else updates
                for update in np.unique(scheduled):
                    record = snapshot(data, x, int(update), days, strength, frequency)
                    chosen = np.flatnonzero(scheduled == update)
                    if update > issued[chosen].min() or (record['latest_label_available_ns'] is not None and record['latest_label_available_ns'] > update):
                        raise AssertionError('Calibration uses unavailable outcomes')
                    predictions[key][chosen] = apply_map(record, x[target[chosen]])
                    records[f'{key}__{int(update)}'] = record
        print(datetime.now(timezone.utc).isoformat(), split, tag, 'lead', lead, 'snapshots', len(records), flush=True)
    np.savez_compressed(out / f'{tag}_predictions.npz', **predictions)
    joblib.dump(records, out / f'{tag}_snapshots.joblib', compress=3)
    audit = []
    for key, record in records.items():
        source, lead, mode, update = key.split('__')
        audit.append(dict(source=source, lead_minutes=int(lead), mode=mode, update_ns=int(update),
            **{k: record[k] for k in ['kind', 'rows', 'row_sha256', 'latest_label_available_ns', 'earliest_target_ns', 'n_iter']}))
    pd.DataFrame(audit).to_csv(out / f'{tag}_update_audit.csv', index=False)
    return predictions


def validate(out):
    sources(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'PREPARED':
        raise RuntimeError('Fresh prepared experiment required')
    data = read_inputs(out); y = data['outcome'][data['validation']]
    candidates, all_scores = [], []
    for days in WINDOWS:
        for strength in LAMBDAS:
            tag = 'validation_' + config_name(days, strength)
            predictions = run_phase(out, data, 'validation', days, strength, tag)
            objective = []
            for key, p in predictions.items():
                source, lead, mode = key.split('__')
                scores = {metric: float(value.mean()) for metric, value in probability_losses(y, p).items()}
                all_scores.append(dict(window_days=days, lambda_=strength, source=source, lead_minutes=int(lead), mode=mode, **scores))
                if source in SOURCES and mode in MODES:
                    objective.append(scores['multiclass_brier'])
            candidates.append(dict(window_days=days, lambda_=strength, multiclass_brier=float(np.mean(objective)), compared_modes=len(objective)))
            print('candidate', candidates[-1], flush=True)
    winner = min(candidates, key=lambda r: (r['multiclass_brier'], r['window_days'], -r['lambda_']))
    pd.DataFrame(candidates).to_csv(out / 'validation_candidates.csv', index=False)
    pd.DataFrame(all_scores).to_csv(out / 'validation_scores.csv', index=False)
    write_json(out / 'selection.json', dict(state='POLICY_FROZEN_BEFORE_TEST', selected=winner,
        frozen_at=datetime.now(timezone.utc).isoformat(), protocol_sha256=digest(out / 'protocol.json'),
        validation_sha256={p.name: digest(p) for p in out.glob('validation_*')}))
    status(out, 'VALIDATION_FROZEN', selected=winner)
    print('FROZEN', winner, flush=True)


def frozen(out):
    p = sources(out); selected = json.loads((out / 'selection.json').read_text())
    if digest(out / 'protocol.json') != selected['protocol_sha256']:
        raise RuntimeError('Changed protocol')
    for file, sha in selected['validation_sha256'].items():
        if digest(out / file) != sha:
            raise RuntimeError(f'Changed validation artifact: {file}')
    return p, selected


def score(out, data, predictions):
    mask = data['test']; y = data['outcome'][mask]; times = pd.to_datetime(data['times_ns'][mask], utc=True)
    losses = {key: probability_losses(y, p) for key, p in predictions.items()}
    comparisons = []
    for lead in LEADS:
        for source in SOURCES:
            prefix = f'{source}__{lead}__'
            for reference in ['raw', 'static_context', 'rolling_global']:
                comparisons.append((prefix + 'rolling_context', prefix + reference))
            for reference in ['rolling_frequency', 'rolling_context_only']:
                comparisons.append((prefix + 'rolling_context', f'reference__{lead}__' + reference))
            comparisons += [(prefix + 'rolling_global', prefix + 'static_global'),
                            (prefix + 'static_context', prefix + 'static_global'),
                            (prefix + 'rolling_global', f'reference__{lead}__rolling_frequency'),
                            (prefix + 'raw', f'reference__{lead}__training_frequency'),
                            (prefix + 'static_context', prefix + 'raw')]
    summary, effects, bias, reliability = [], [], [], []
    for state in ['all', *STATES]:
        subset = np.ones(len(y), bool) if state == 'all' else data['state'][mask] == state
        for key, metrics in losses.items():
            source, lead, mode = key.split('__')
            for metric, value in metrics.items():
                summary.append(dict(source=source, lead_minutes=int(lead), mode=mode, state=state, metric=metric,
                    windows=int(subset.sum()), loss=float(value[subset].mean())))
            p = predictions[key][subset]; truth = y[subset]
            for event, classes in [('up', [1, 3]), ('down', [2, 3]), ('return', [3])]:
                value = p[:, classes].sum(axis=1); actual = np.isin(truth, classes)
                bias.append(dict(source=source, lead_minutes=int(lead), mode=mode, state=state, event=event,
                    windows=len(truth), mean_probability=float(value.mean()), observed_fraction=float(actual.mean()),
                    probability_bias=float(value.mean() - actual.mean())))
        for candidate, reference in comparisons:
            for metric in ['multiclass_brier', 'return_brier', 'up_brier', 'down_brier']:
                for days in ([3, 7, 14] if state == 'all' else [7]):
                    effects.append(dict(candidate=candidate, reference=reference, state=state, metric=metric, windows=int(subset.sum()),
                        **paired_loss_effect(times[subset], losses[candidate][metric][subset], losses[reference][metric][subset], days)))
    for key, p in predictions.items():
        for event, classes in [('up', [1, 3]), ('down', [2, 3]), ('return', [3])]:
            value = p[:, classes].sum(axis=1); actual = np.isin(y, classes)
            bins = np.minimum((value * 10).astype(int), 9)
            for b in range(10):
                take = bins == b
                reliability.append(dict(model=key, event=event, bin_low=b / 10, windows=int(take.sum()),
                    mean_probability=float(value[take].mean()) if take.any() else np.nan,
                    observed_fraction=float(actual[take].mean()) if take.any() else np.nan))
    pd.DataFrame(summary).to_csv(out / 'test_summary.csv', index=False)
    pd.DataFrame(effects).to_csv(out / 'paired_intervals.csv', index=False)
    pd.DataFrame(bias).to_csv(out / 'conditional_bias.csv', index=False)
    pd.DataFrame(reliability).to_csv(out / 'reliability.csv', index=False)
    print(pd.DataFrame(summary).query("state == 'all' and lead_minutes == 720 and metric == 'multiclass_brier'").to_string(index=False), flush=True)


def test(out):
    _, record = frozen(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'VALIDATION_FROZEN' or list(out.glob('test_*')):
        raise RuntimeError('Frozen untested policy required')
    data = read_inputs(out); config = record['selected']
    predictions = run_phase(out, data, 'test', config['window_days'], config['lambda_'], 'test')
    score(out, data, predictions)
    status(out, 'COMPLETE', test_windows=int(data['test'].sum()), selected=config,
           sequential_test_updates=True, retrained_base_forests=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'validate', 'test'], required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {'prepare': prepare, 'validate': validate, 'test': test}[args.phase](args.output)
