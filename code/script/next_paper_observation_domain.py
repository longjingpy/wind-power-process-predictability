"""NP006: align conditional scenarios with their measured-power target.

The inherited target is native recorded power, including small values outside
the nominal [0, capacity] range. Training-observed bounds are an empirical
support correction, not physical limits or a new calibration algorithm.
No observations are clipped, and no new trees are fitted.

Gneiting & Raftery (2007), Strictly Proper Scoring Rules, Prediction, and
Estimation, doi:10.1198/016214506000001437, CRPS and interval scores: use
proper scores alongside coverage so arbitrarily wide intervals cannot pass
as an improvement. Conditional leaf sampling and process probabilities reuse
NP004/005 and their cited Meinshausen/Worsnop methods unchanged. Paired
calendar blocks retain the inherited Kunsch (1989) motivation.

SCADA wind changes are a local operating-measurement reference. JMA10m and
GFS100m forecasts use proxy coordinates, so their change correspondence is
not a direct same-height forecast verification or atmospheric causal proof.
"""
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from next_paper_leadtime import (ROOT, OUT as PARENT, POWER, LEADS, ARMS, STATES,
    read_data, frozen as check_parent, digest, write_json, status, path_class)
from next_paper_probability import (probability_from_classes, probability_losses,
    empirical_crps, paired_loss_effect)

OUT = ROOT / 'outputs/next_paper/np006'


def training_bounds(data):
    target = data['actual'][data['train']]
    return np.array([target.min(), target.max()])


def interval_score(actual, lower, upper, alpha):
    if not 0 < alpha < 1 or (np.asarray(lower) > np.asarray(upper)).any():
        raise ValueError('Ordered bounds and valid tail probability required')
    return upper - lower + 2 / alpha * (np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))


def node_metrics(actual, values):
    q = values['quantiles']
    return {'crps': values['crps'],
        'coverage80': ((actual >= q[1]) & (actual <= q[2])).astype(float),
        'coverage95': ((actual >= q[0]) & (actual <= q[3])).astype(float),
        'width80': q[2] - q[1], 'width95': q[3] - q[0],
        'interval_score80': interval_score(actual, q[1], q[2], .2),
        'interval_score95': interval_score(actual, q[0], q[3], .05)}


def load_npz(path):
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def wind_reference(data):
    frame = pd.read_parquet(POWER)
    indices = frame.index.get_indexer(pd.to_datetime(data['times_ns'], utc=True))
    if (indices < 0).any():
        raise ValueError('Unknown target clock')
    wind = np.column_stack([frame.wind.to_numpy()[indices + k] for k in range(17)])
    return wind, np.isfinite(wind).all(axis=1)


def prepare(out):
    check_parent(PARENT)
    if out.exists():
        raise RuntimeError('A fresh output directory is required')
    data = read_data(PARENT)
    bounds = training_bounds(data)
    wind, valid = wind_reference(data)
    anchor = data['anchor'][LEADS.index(720)]
    cuts = np.quantile(anchor[data['train']], [1 / 3, 2 / 3])
    levels = np.select([anchor <= cuts[0], anchor > cuts[1]], ['power_low', 'power_high'], default='power_middle')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'wind_reference.npz', wind=wind, valid=valid, levels=levels)
    inventory = []
    for split in ['train', 'validation', 'test']:
        actual = data['actual'][data[split]]
        below, above = actual < 0, actual > 1
        inventory.append(dict(split=split, windows=int(data[split].sum()), scored_nodes=actual.size,
            complete_wind_windows=int((data[split] & valid).sum()), minimum=float(actual.min()), maximum=float(actual.max()),
            below_zero_fraction=float(below.mean()), above_one_fraction=float(above.mean()),
            outside_nominal_fraction=float((below | above).mean()),
            maximum_possible_nominal_coverage=float((~below & ~above).mean()),
            outside_training_range_fraction=float(((actual < bounds[0]) | (actual > bounds[1])).mean())))
    pd.DataFrame(inventory).to_csv(out / 'domain_inventory.csv', index=False)
    tracked = [Path(__file__), ROOT / 'script/next_paper_leadtime.py', ROOT / 'script/next_paper_probability.py',
               POWER, PARENT / 'protocol.json', PARENT / 'freeze.json', PARENT / 'dataset.npz',
               *PARENT.glob('validation_*.npz'), *PARENT.glob('test_*.npz')]
    protocol = dict(experiment='NP006', stage='EXPLORATORY_SUPPORT_CORRECTION_AFTER_VIEWED_NP005',
        question='Can matching scenario support to the measured target restore reliability while retaining process skill?',
        secondary_question='How do forecast wind backgrounds correspond to later onsite wind and probability bias?',
        parent='NP005', leads=LEADS, arms=ARMS, training_observed_bounds=bounds.tolist(),
        bound_rule='Min/max of NP005 training actual paths only; fixed before new scores; no label clipping or physical-bound claim',
        observations='Native recorded power unchanged, including small negative and above-nameplate readings',
        sampling='Same saved 450 training neighbors as NP005; no new sampling or tree fit',
        selection='No model or bound search; fixed support correction evaluated on validation before frozen test',
        wind_reference='All17 complete trailing15min SCADA means ending one minute before each target node; diagnostic only',
        weather_reference='Archived JMA10m/GFS100m proxy-site magnitudes; changes are not same-height verification',
        power_context='12h issue power; training-only terciles; all levels retained', power_context_cutoffs=cuts.tolist(),
        inference='Paired3/7/14day target blocks;2000 draws seed41; primary7d; exploratory pointwise intervals',
        reporting='All18 cells; original and corrected distributions; classifier comparator; interval scores and coverage separated',
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in tracked},
        prepared_sha256={'wind_reference.npz': digest(out / 'wind_reference.npz')})
    write_json(out / 'protocol.json', protocol)
    status(out, 'PREPARED', training_observed_bounds=bounds.tolist())
    print(pd.DataFrame(inventory).to_string(index=False), flush=True)


def sources(out):
    p = json.loads((out / 'protocol.json').read_text())
    for name, sha in p['source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f'Changed inherited source: {name}')
    for name, sha in p['prepared_sha256'].items():
        if digest(out / name) != sha:
            raise RuntimeError(f'Changed prepared source: {name}')
    return p


def generate(out, data, protocol, split):
    mask, tr = data[split], data['train']
    lower, upper = protocol['training_observed_bounds']
    for j, lead in enumerate(LEADS):
        anchor = data['anchor'][j, mask]
        delta = data['actual'][tr] - data['anchor'][j, tr, None]
        for arm in ARMS:
            old = load_npz(PARENT / f'{split}_{lead}_{arm}.npz')
            probabilities, means, quantiles, crps = [], [], [], []
            maximum_nominal_probability_error = 0.
            for start in range(0, len(anchor), 256):
                stop = min(start + 256, len(anchor))
                raw = anchor[start:stop, None, None] + delta[old['neighbors'][start:stop]]
                nominal = np.clip(raw, 0, 1)
                reference = probability_from_classes(path_class(nominal))
                maximum_nominal_probability_error = max(maximum_nominal_probability_error,
                    float(np.max(np.abs(reference - old['joint'][start:stop]))))
                paths = np.clip(raw, lower, upper)
                probabilities.append(probability_from_classes(path_class(paths)))
                means.append(paths.mean(axis=1))
                quantiles.append(np.quantile(paths, [.025, .1, .9, .975], axis=1))
                crps.append(empirical_crps(paths, data['actual'][mask][start:stop]))
            if maximum_nominal_probability_error != 0:
                raise AssertionError('Inherited nominal scenario probabilities not reproduced')
            np.savez_compressed(out / f'{split}_{lead}_{arm}.npz', joint=np.vstack(probabilities),
                mean_path=np.vstack(means), crps=np.vstack(crps), quantiles=np.concatenate(quantiles, axis=1))
            print('generated', split, lead, arm, flush=True)


def score(out, data, split):
    mask = data[split]; actual = data['actual'][mask]; y = data['outcome'][mask]
    times = pd.to_datetime(data['times_ns'][mask], utc=True)
    probability_rows, marginal_rows, effects = [], [], []
    for lead in LEADS:
        for arm in ARMS:
            old = load_npz(PARENT / f'{split}_{lead}_{arm}.npz')
            new = load_npz(out / f'{split}_{lead}_{arm}.npz')
            probability = {'nominal': old['joint'], 'observed_range': new['joint'], 'classifier': old['classifier']}
            losses = {key: probability_losses(y, p) for key, p in probability.items()}
            nodes = {'nominal': node_metrics(actual, old), 'observed_range': node_metrics(actual, new)}
            for state in ['all', *STATES]:
                subset = np.ones(len(y), bool) if state == 'all' else data['state'][mask] == state
                for method, metrics in losses.items():
                    for metric, value in metrics.items():
                        probability_rows.append(dict(lead_minutes=lead, arm=arm, method=method, state=state,
                            metric=metric, windows=int(subset.sum()), loss=float(value[subset].mean())))
            for method, metrics in nodes.items():
                for node in range(17):
                    inside = (actual[:, node] >= 0) & (actual[:, node] <= 1)
                    marginal_rows.append(dict(lead_minutes=lead, arm=arm, method=method, target_offset_minutes=15 * node,
                        **{key: float(value[:, node].mean()) for key, value in metrics.items()},
                        coverage80_inside_nominal=float(metrics['coverage80'][inside, node].mean()),
                        coverage95_inside_nominal=float(metrics['coverage95'][inside, node].mean()),
                        nominal_inside_nodes=int(inside.sum())))
            if split == 'test':
                compare = [(metric, losses['observed_range'][metric], losses[reference][metric], reference)
                           for metric in ['return_brier', 'up_brier', 'down_brier', 'multiclass_brier']
                           for reference in ['nominal', 'classifier']]
                compare += [(metric, nodes['observed_range'][metric].mean(axis=1), nodes['nominal'][metric].mean(axis=1), 'nominal')
                            for metric in ['crps', 'interval_score80', 'interval_score95']]
                for metric, candidate, reference_loss, reference in compare:
                    for days in [3, 7, 14]:
                        effects.append(dict(lead_minutes=lead, arm=arm, metric=metric, reference=reference,
                            windows=len(y), **paired_loss_effect(times, candidate, reference_loss, days)))
    pd.DataFrame(probability_rows).to_csv(out / f'{split}_probability_scores.csv', index=False)
    marginal = pd.DataFrame(marginal_rows)
    marginal.to_csv(out / f'{split}_marginal_scores.csv', index=False)
    if effects:
        pd.DataFrame(effects).to_csv(out / 'paired_intervals.csv', index=False)
    print(marginal.groupby(['arm', 'lead_minutes', 'method'])[['crps', 'coverage80', 'coverage95', 'width80']].mean().round(5).to_string(), flush=True)


def physical_diagnostics(out, data):
    reference = load_npz(out / 'wind_reference.npz')
    wind = reference['wind']; valid = reference['valid']; levels = reference['levels']
    observation = wind[:, -1] - wind[:, 0]
    forecast = data['weather'].reshape(-1, 17, 8)
    shifts = {name: np.hypot(forecast[:, -1, k], forecast[:, -1, k + 1])
              - np.hypot(forecast[:, 0, k], forecast[:, 0, k + 1]) for name, k in [('jma10m', 0), ('gfs100m', 4)]}
    correspondence, context, calibration = [], [], []
    for split in ['train', 'validation', 'test']:
        probabilities = {}
        if split != 'train':
            for arm in ['power', 'power_weather']:
                with np.load(PARENT / f'{split}_720_{arm}.npz') as source:
                    probabilities[arm] = {method: source[method] for method in ['classifier', 'joint']}
        for state in ['all', *STATES]:
            support = data[split] & valid & (True if state == 'all' else data['state'] == state)
            for name, delta in shifts.items():
                pred, obs = delta[support], observation[support]
                nonzero = (pred != 0) & (obs != 0)
                correspondence.append(dict(split=split, state=state, weather=name, windows=int(support.sum()),
                    forecast_change=float(pred.mean()), observed_change=float(obs.mean()),
                    change_mae=float(np.abs(pred - obs).mean()), change_rmse=float(np.sqrt(np.mean((pred - obs) ** 2))),
                    correlation=float(np.corrcoef(pred, obs)[0, 1]),
                    sign_agreement=float((np.sign(pred[nonzero]) == np.sign(obs[nonzero])).mean()),
                    nonzero_pairs=int(nonzero.sum())))
            for level in ['all', 'power_low', 'power_middle', 'power_high']:
                subset = support & (True if level == 'all' else levels == level)
                if not subset.any():
                    continue
                y = data['outcome'][subset]
                context.append(dict(split=split, state=state, issue_power_level=level, windows=int(subset.sum()),
                    issue_power_mean=float(data['anchor'][LEADS.index(720), subset].mean()),
                    target_power_start_mean=float(data['actual'][subset, 0].mean()),
                    onsite_wind_change=float(observation[subset].mean()),
                    up_fraction=float(np.isin(y, [1, 3]).mean()), down_fraction=float(np.isin(y, [2, 3]).mean()),
                    return_fraction=float((y == 3).mean())))
                if split == 'train':
                    continue
                aligned = subset[data[split]]
                for arm in ['power', 'power_weather']:
                    for method in ['classifier', 'joint']:
                        prob = probabilities[arm][method][aligned]
                        for event, outcome, value in [('up', np.isin(y, [1, 3]), prob[:, 1] + prob[:, 3]),
                                                      ('down', np.isin(y, [2, 3]), prob[:, 2] + prob[:, 3]),
                                                      ('return', y == 3, prob[:, 3])]:
                            calibration.append(dict(split=split, state=state, issue_power_level=level, arm=arm, method=method,
                                event=event, windows=len(y), observed_fraction=float(outcome.mean()), mean_probability=float(value.mean()),
                                probability_bias=float(value.mean() - outcome.mean()), brier=float(np.mean((value - outcome) ** 2)),
                                auroc=float(roc_auc_score(outcome, value)) if len(np.unique(outcome)) == 2 else np.nan))
    pd.DataFrame(correspondence).to_csv(out / 'wind_correspondence.csv', index=False)
    pd.DataFrame(context).to_csv(out / 'response_context.csv', index=False)
    pd.DataFrame(calibration).to_csv(out / 'conditional_probability_diagnostics.csv', index=False)


def validate(out):
    p = sources(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'PREPARED':
        raise RuntimeError('Fresh prepared run required')
    data = read_data(PARENT)
    generate(out, data, p, 'validation')
    score(out, data, 'validation')
    write_json(out / 'freeze.json', dict(state='FROZEN_BEFORE_NEW_TEST', frozen_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(out / 'protocol.json'),
        validation_sha256={p.name: digest(p) for p in out.glob('validation_*')}))
    status(out, 'VALIDATION_FROZEN')


def frozen(out):
    p = sources(out)
    record = json.loads((out / 'freeze.json').read_text())
    assert digest(out / 'protocol.json') == record['protocol_sha256']
    for name, sha in record['validation_sha256'].items():
        assert digest(out / name) == sha
    return p


def test(out):
    p = frozen(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'VALIDATION_FROZEN' or list(out.glob('test_*')):
        raise RuntimeError('Frozen untested output required')
    data = read_data(PARENT)
    generate(out, data, p, 'test')
    score(out, data, 'test')
    physical_diagnostics(out, data)
    status(out, 'COMPLETE', test_windows=int(data['test'].sum()), retrained_models=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'validate', 'test'], required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'validate': validate, 'test': test}[args.phase](args.output)
