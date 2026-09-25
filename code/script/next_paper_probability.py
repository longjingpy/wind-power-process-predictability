"""NP004: coherent conditional trajectories and fixed-window return risk.

Meinshausen (2006), Quantile Regression Forests, JMLR 7:983-999,
https://www.jmlr.org/papers/v7/meinshausen06a.html, Sections 2-3, equations
(4)-(6): retain all responses in a terminal leaf and average normalized leaf
weights across trees. We reuse frozen ExtraTrees and sample one complete
training delta path per tree. This is an adaptation, not a reproduction of
that paper's forest settings or its IID consistency theorem.

Worsnop et al. (2018), Generating wind power scenarios for probabilistic ramp
event prediction using multivariate statistical post-processing,
doi:10.5194/wes-3-371-2018, multivariate scenarios and ramp evaluation:
temporal dependence matters for process probabilities. Our shuffled control
preserves each empirical marginal exactly and changes scenario correspondence.

Gneiting & Raftery (2007), Strictly Proper Scoring Rules, Prediction, and
Estimation, doi:10.1198/016214506000001437, quadratic/log scores and CRPS:
proper probability losses assess risk forecasts; CRPS of an empirical
distribution is mean|X-y| minus one half mean|X-X'|. Block inference reuses
the existing Kunsch-motivated calendar design. No alert threshold is tuned.
"""
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from next_paper_wind_history import (ROOT, BASE, OUT as PREVIOUS, STATES, SEEDS,
    sources as check_parent_sources, feature_arms, original_prediction)
from next_paper_trajectory import digest, write_json, status
from wind_events.paired_probability import block_design

OUT = ROOT / 'outputs/next_paper/np004'
ARMS = ['base', 'wind_history']
AMPLITUDE = .2
BATCH = 256


def process_class(future, anchor, threshold=AMPLITUDE):
    """NP002's capacity scale, applied to a fixed future window, not its catalogue.

    Maximum ordered drawup/drawdown distinguish return paths from a small net
    change. This reuses the project's S26 process idea with power units and an
    explicit threshold. All future coordinates are outcomes, never inputs.
    """
    future, anchor = np.asarray(future), np.asarray(anchor)
    if future.ndim not in [2, 3] or len(future) != len(anchor):
        raise ValueError('Aligned paths and known anchors required')
    if not np.isfinite(future).all() or not np.isfinite(anchor).all():
        raise ValueError('Finite paths required')
    shape = (*future.shape[:-1], 1)
    start = np.broadcast_to(anchor.reshape((len(anchor),) + (1,) * (future.ndim - 1)), shape)
    path = np.concatenate([start, future], axis=-1)
    up = np.max(path - np.minimum.accumulate(path, axis=-1), axis=-1)
    down = np.max(np.maximum.accumulate(path, axis=-1) - path, axis=-1)
    return (up >= threshold - 1e-12).astype(np.int8) + 2 * (down >= threshold - 1e-12).astype(np.int8)


def probability_from_classes(labels):
    return np.stack([(labels == k).mean(axis=1) for k in range(4)], axis=1)


def probability_losses(actual, probability):
    p = np.asarray(probability)
    if p.shape != (len(actual), 4) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Normalized four-class probabilities required')
    y = np.asarray(actual, int)
    if ((y < 0) | (y > 3)).any():
        raise ValueError('Outcomes must be in the four declared classes')
    return {'return_brier': (p[:, 3] - (y == 3)) ** 2,
        'multiclass_brier': ((p - np.eye(4)[y]) ** 2).sum(axis=1),
        'log_loss': -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)),
        'up_brier': (p[:, 1] + p[:, 3] - ((y == 1) | (y == 3))) ** 2,
        'down_brier': (p[:, 2] + p[:, 3] - ((y == 2) | (y == 3))) ** 2}


def empirical_crps(samples, actual):
    ordered = np.sort(samples, axis=1)
    m = samples.shape[1]
    coefficient = 2 * np.arange(1, m + 1) - m - 1
    half_pair_distance = np.einsum('m,nmh->nh', coefficient, ordered) / m ** 2
    return np.mean(np.abs(samples - actual[:, None, :]), axis=1) - half_pair_distance


def build_leaf_bank(models, x, delta):
    bank, maximum_error = [], 0.
    for _, forest, _ in models:
        leaves = forest.apply(x)
        for j, tree in enumerate(forest.estimators_):
            ids = leaves[:, j]
            counts = np.bincount(ids, minlength=tree.tree_.node_count)
            starts = np.cumsum(counts) - counts
            order = np.argsort(ids, kind='stable').astype(np.int32)
            occupied = np.flatnonzero(counts)
            means = np.stack([np.bincount(ids, weights=delta[:, k], minlength=len(counts))[occupied] / counts[occupied]
                              for k in range(delta.shape[1])], axis=1)
            error = np.max(np.abs(means - tree.tree_.value[occupied, :, 0]))
            maximum_error = max(maximum_error, float(error))
            if not np.array_equal(counts[occupied], tree.tree_.n_node_samples[occupied]):
                raise ValueError('Leaf population does not match the frozen training set')
            bank.append((order, starts.astype(np.int32), counts.astype(np.int32)))
    if maximum_error > 1e-10:
        raise ValueError('Leaf target means do not reproduce the frozen forest')
    return bank, maximum_error


def draw_neighbors(models, bank, x, seed):
    leaves = np.column_stack([forest.apply(x) for _, forest, _ in models])
    uniform = np.random.default_rng(seed).random(leaves.shape)
    neighbors = np.empty(leaves.shape, np.int32)
    for j, (order, starts, counts) in enumerate(bank):
        size = counts[leaves[:, j]]
        if (size <= 0).any():
            raise ValueError('Empty predictive leaf')
        offset = starts[leaves[:, j]] + np.floor(uniform[:, j] * size).astype(int)
        neighbors[:, j] = order[offset]
    return neighbors


def parent_data():
    data = np.load(BASE / 'dataset.npz')
    wind = np.load(PREVIOUS / 'wind_inputs.npz')
    x = feature_arms(data['weather'], wind['wind'])
    return data, wind, x


def prepare(out):
    check_parent_sources(PREVIOUS)
    parent_selection = json.loads((PREVIOUS / 'selection.json').read_text())
    for name, sha in parent_selection['model_sha256'].items():
        if digest(PREVIOUS / name) != sha:
            raise RuntimeError('NP003 model changed')
    out.mkdir(parents=True, exist_ok=False)
    data, wind, x = parent_data()
    tr = wind['train']; y, anchor = data['actual'], data['anchor']
    labels = pd.read_parquet(PREVIOUS / 'window_labels.parquet')[['issue_time', 'wind_state']].copy()
    labels['outcome'] = -1
    union = tr | wind['validation'] | wind['test']
    labels.loc[union, 'outcome'] = process_class(y[union], anchor[union])
    labels.to_parquet(out / 'targets.parquet', index=False)
    support = []
    for split in ['train', 'validation', 'test']:
        for state in ['all', *STATES]:
            mask = wind[split] & (np.ones(len(y), bool) if state == 'all' else labels.wind_state.eq(state).to_numpy())
            support.append({'split': split, 'state': state, 'windows': int(mask.sum()),
                **{f'class_{k}': int((labels.loc[mask, 'outcome'] == k).sum()) for k in range(4)}})
    pd.DataFrame(support).to_csv(out / 'support.csv', index=False)
    errors = {}
    for arm in ARMS:
        models = joblib.load(PREVIOUS / f'{arm}_models.joblib')
        bank, error = build_leaf_bank(models, x[arm][tr], y[tr] - anchor[tr, None])
        joblib.dump(bank, out / f'{arm}_leaf_bank.joblib', compress=3)
        errors[arm] = error
        print('leaf bank', arm, len(bank), 'maximum_mean_error', error, flush=True)
    tracked = [Path(__file__), PREVIOUS / 'protocol.json', PREVIOUS / 'selection.json',
        PREVIOUS / 'wind_inputs.npz', PREVIOUS / 'window_labels.parquet',
        *[PREVIOUS / f'{arm}_models.joblib' for arm in ARMS]]
    protocol = {'experiment': 'NP004', 'stage': 'EXPLORATORY_FIXED_4H_PROCESS_PROBABILITY',
        'target': '0=neither,1=up only,2=down only,3=both ordered drawup and drawdown >=0.2 capacity within future4h',
        'catalogue_distinction': 'Fixed-window excursion class, not NP002 complete-composite catalogue membership',
        'amplitude_pu': AMPLITUDE, 'anchor': 'observed power at issue minus1min',
        'common_support': {s: int(wind[s].sum()) for s in ['train', 'validation', 'test']},
        'training_class_counts': np.bincount(labels.loc[tr, 'outcome'], minlength=4).tolist(),
        'arms': ARMS, 'scenario_members': 450, 'batch_size': BATCH,
        'sampling': 'One uniform training member per frozen tree; common quantiles between arms; full delta path plus new anchor clipped[0,1]',
        'sampling_seeds': {'validation': 8401, 'test': 8402}, 'shuffle_seeds': {'validation': 8501, 'test': 8502},
        'temporal_control': 'Independent member permutation per origin and lead, preserving exact empirical marginals',
        'classifier': '3 seeds41/42/43;150 trees;max_depth16;min_samples_leaf32;no class reweighting',
        'calibration': 'No fitted calibration; raw probabilistic reliability reported',
        'selection': 'Lowest validation return Brier among joint scenarios, direct classifiers and train frequency; ties by multiclass Brier then name',
        'scores': ['return_brier', 'multiclass_brier', 'log_loss', 'up_brier', 'down_brier'],
        'log_probability_floor': 1e-12,
        'inference': 'Paired3/7/14d calendar blocks,2000 draws,primary7d; full prospective populations and pre-issue wind states',
        'leaf_mean_maximum_errors': errors,
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in tracked},
        'prepared_sha256': {p.name: digest(p) for p in [out / 'targets.parquet', *out.glob('*leaf_bank.joblib')]}}
    write_json(out / 'protocol.json', protocol)
    status(out, 'PREPARED', training_counts=protocol['training_class_counts'])


def sources(out):
    check_parent_sources(PREVIOUS)
    p = json.loads((out / 'protocol.json').read_text())
    for name, sha in p['source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f'Changed source: {name}')
    for name, sha in p['prepared_sha256'].items():
        if digest(out / name) != sha:
            raise RuntimeError(f'Changed prepared artifact: {name}')
    return p


def probability_panel(labels, probabilities):
    panel = labels.rename_axis('issue_time').reset_index()[['issue_time', 'wind_state', 'outcome']]
    for name, p in probabilities.items():
        probability_losses(labels.outcome.to_numpy(), p)
        for k in range(4):
            panel[f'{name}_p{k}'] = p[:, k]
    return panel


def generate(out, split, classifiers):
    protocol = sources(out)
    data, wind, x = parent_data()
    mask, tr = wind[split], wind['train']
    anchor, y = data['anchor'][mask], data['actual'][mask]
    delta_train = data['actual'][tr] - data['anchor'][tr, None]
    times = pd.to_datetime(data['times_ns'][mask], utc=True)
    labels = pd.read_parquet(out / 'targets.parquet').set_index('issue_time').loc[times]
    frequency = np.array(protocol['training_class_counts']) / int(tr.sum())
    probabilities = {'train_frequency': np.tile(frequency, (len(y), 1)),
        'np002_point': np.eye(4)[process_class(original_prediction(data, mask), anchor)]}
    marginals, generation = [], {}
    for arm in ARMS:
        models = joblib.load(PREVIOUS / f'{arm}_models.joblib')
        bank = joblib.load(out / f'{arm}_leaf_bank.joblib')
        neighbors = draw_neighbors(models, bank, x[arm][mask], protocol['sampling_seeds'][split])
        np.savez_compressed(out / f'{split}_{arm}_neighbors.npz', neighbors=neighbors)
        joint_p, shuffled_p, mean_p = [], [], []
        max_difference = 0.
        shuffle_rng = np.random.default_rng(protocol['shuffle_seeds'][split])
        for start in range(0, len(y), BATCH):
            end = min(start + BATCH, len(y))
            paths = np.clip(anchor[start:end, None, None] + delta_train[neighbors[start:end]], 0, 1)
            shuffled = shuffle_rng.permuted(paths, axis=1)
            ordered = np.sort(paths, axis=1)
            difference = float(np.max(np.abs(ordered - np.sort(shuffled, axis=1))))
            max_difference = max(max_difference, difference)
            joint_p.append(probability_from_classes(process_class(paths, anchor[start:end])))
            shuffled_p.append(probability_from_classes(process_class(shuffled, anchor[start:end])))
            mean_p.append(np.eye(4)[process_class(paths.mean(axis=1), anchor[start:end])])
            crps = empirical_crps(paths, y[start:end])
            quantiles = np.quantile(paths, [.025, .1, .9, .975], axis=1)
            marginals.append(pd.DataFrame({'issue_time': np.repeat(times[start:end], 16), 'arm': arm,
                'horizon_minutes': np.tile(np.arange(1, 17) * 15, end - start),
                'actual_pu': y[start:end].ravel(), 'crps': crps.ravel(),
                'lower80': quantiles[1].ravel(), 'upper80': quantiles[2].ravel(),
                'lower95': quantiles[0].ravel(), 'upper95': quantiles[3].ravel()}))
        if max_difference != 0:
            raise AssertionError('Temporal control changed marginal values')
        probabilities[f'{arm}_joint'] = np.vstack(joint_p)
        probabilities[f'{arm}_shuffled'] = np.vstack(shuffled_p)
        probabilities[f'{arm}_mean'] = np.vstack(mean_p)
        p = np.zeros((len(y), 4))
        for classifier in classifiers[arm]:
            p[:, classifier.classes_] += classifier.predict_proba(x[arm][mask]) / len(SEEDS)
        probabilities[f'{arm}_classifier'] = p
        generation[arm] = {'members': neighbors.shape[1], 'minimum_neighbor': int(neighbors.min()),
            'maximum_neighbor': int(neighbors.max()), 'maximum_marginal_difference': max_difference}
        print('generated', split, arm, len(y), flush=True)
    panel = probability_panel(labels, probabilities)
    panel.to_parquet(out / f'{split}_probabilities.parquet', index=False)
    pd.concat(marginals, ignore_index=True).to_parquet(out / f'{split}_marginals.parquet', index=False)
    write_json(out / f'{split}_generation.json', generation)
    return labels, probabilities


def validate(out):
    sources(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'PREPARED':
        raise RuntimeError('Fresh prepared run required')
    data, wind, x = parent_data(); tr = wind['train']
    target = process_class(data['actual'][tr], data['anchor'][tr])
    classifiers = {}
    for arm in ARMS:
        classifiers[arm] = []
        for seed in SEEDS:
            model = ExtraTreesClassifier(n_estimators=150, max_depth=16, min_samples_leaf=32,
                max_features=1., random_state=seed, n_jobs=4).fit(x[arm][tr], target)
            classifiers[arm].append(model)
        joblib.dump(classifiers[arm], out / f'{arm}_classifiers.joblib', compress=3)
    labels, probabilities = generate(out, 'validation', classifiers)
    candidates = []
    for name, p in probabilities.items():
        scores = {m: float(v.mean()) for m, v in probability_losses(labels.outcome.to_numpy(), p).items()}
        candidates.append({'model': name, 'eligible': name == 'train_frequency' or name.endswith(('_joint', '_classifier')), **scores})
    winner = min([c for c in candidates if c['eligible']], key=lambda c: (c['return_brier'], c['multiclass_brier'], c['model']))
    pd.DataFrame(candidates).to_csv(out / 'validation_candidates.csv', index=False)
    write_json(out / 'selection.json', {'state': 'FROZEN_BEFORE_TEST', 'selected': winner,
        'protocol_sha256': digest(out / 'protocol.json'),
        'classifier_sha256': {p.name: digest(p) for p in out.glob('*classifiers.joblib')}})
    status(out, 'VALIDATION_FROZEN', selected=winner['model'])
    print('FROZEN', winner, flush=True)


def paired_loss_effect(times, candidate, reference, days):
    index, weights = block_design(times, days, 2000, 41)
    a = np.bincount(index, weights=candidate, minlength=weights.shape[1])
    b = np.bincount(index, weights=reference, minlength=weights.shape[1])
    draws = 100 * (1 - (weights @ a) / (weights @ b))
    low, high = np.quantile(draws, [.025, .975])
    return {'relative_loss_reduction_pct': 100 * (1 - candidate.mean() / reference.mean()),
        'low': low, 'high': high, 'block_days': days, 'occupied_blocks': weights.shape[1]}


def test(out):
    sources(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'VALIDATION_FROZEN':
        raise RuntimeError('Frozen untested run required')
    selection = json.loads((out / 'selection.json').read_text())
    if digest(out / 'protocol.json') != selection['protocol_sha256']:
        raise RuntimeError('Changed protocol')
    for name, sha in selection['classifier_sha256'].items():
        if digest(out / name) != sha:
            raise RuntimeError('Changed classifier')
    if (out / 'test_probabilities.parquet').exists():
        raise RuntimeError('Test already generated')
    classifiers = {a: joblib.load(out / f'{a}_classifiers.joblib') for a in ARMS}
    labels, probabilities = generate(out, 'test', classifiers)
    probabilities['selected'] = probabilities[selection['selected']['model']]
    actual = labels.outcome.to_numpy(); times = pd.DatetimeIndex(labels.index)
    losses = {name: probability_losses(actual, p) for name, p in probabilities.items()}
    pairs = [('base_joint', 'base_shuffled'), ('base_joint', 'train_frequency'),
        ('wind_history_joint', 'base_joint'), ('base_joint', 'base_mean'),
        ('wind_history_joint', 'wind_history_shuffled'), ('base_joint', 'base_classifier'),
        ('selected', 'train_frequency'), ('selected', 'np002_point'),
        ('wind_history_joint', 'wind_history_classifier')]
    summary, effects, reliability = [], [], []
    for state in ['all', *STATES]:
        mask = np.ones(len(actual), bool) if state == 'all' else labels.wind_state.eq(state).to_numpy()
        for name, scores in losses.items():
            for metric, values in scores.items():
                summary.append({'model': name, 'state': state, 'metric': metric, 'windows': int(mask.sum()),
                    'return_windows': int(((actual == 3) & mask).sum()), 'loss': float(values[mask].mean()),
                    'skill_vs_train_frequency_pct': 100 * (1 - values[mask].mean() / losses['train_frequency'][metric][mask].mean())})
        for candidate, reference in pairs:
            for metric in losses[candidate]:
                for days in [3, 7, 14]:
                    effect = paired_loss_effect(times[mask], losses[candidate][metric][mask], losses[reference][metric][mask], days)
                    effects.append({'candidate': candidate, 'reference': reference, 'state': state,
                        'metric': metric, 'windows': int(mask.sum()), **effect})
    for name, p in probabilities.items():
        bins = np.minimum((p[:, 3] * 10).astype(int), 9)
        for b in range(10):
            mask = bins == b
            reliability.append({'model': name, 'bin_low': b / 10, 'bin_high': (b + 1) / 10,
                'windows': int(mask.sum()), 'mean_probability': p[mask, 3].mean() if mask.any() else np.nan,
                'observed_return_fraction': (actual[mask] == 3).mean() if mask.any() else np.nan})
    pd.DataFrame(summary).to_csv(out / 'test_summary.csv', index=False)
    pd.DataFrame(effects).to_csv(out / 'paired_intervals.csv', index=False)
    pd.DataFrame(reliability).to_csv(out / 'reliability.csv', index=False)
    marginal = pd.read_parquet(out / 'test_marginals.parquet')
    marginal['covered80'] = marginal.actual_pu.between(marginal.lower80, marginal.upper80)
    marginal['covered95'] = marginal.actual_pu.between(marginal.lower95, marginal.upper95)
    marginal['width80'] = marginal.upper80 - marginal.lower80
    marginal.groupby(['arm', 'horizon_minutes']).agg(crps=('crps', 'mean'), coverage80=('covered80', 'mean'),
        coverage95=('covered95', 'mean'), width80=('width80', 'mean')).reset_index().to_csv(out / 'marginal_summary.csv', index=False)
    status(out, 'COMPLETE', windows=len(actual), return_windows=int((actual == 3).sum()), selected=selection['selected']['model'])
    result = pd.DataFrame(summary)
    print(result[result.state.eq('all') & result.metric.eq('return_brier')].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'validate', 'test'], required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'validate': validate, 'test': test}[args.phase](args.output)


if __name__ == '__main__':
    main()
