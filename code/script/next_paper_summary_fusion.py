"""NP015: directly fuse prequential local-wind summaries into event forecasts.

The saved NP013/NP014 paths are reconstructed from rolling historical neighbors.
The target future wind is never used as an operating feature. Validation is a
fit period and test is a chronological holdout. No test selection is done.
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from next_paper_local_wind import BASE, ROOT, inputs
from next_paper_range_distribution import rank_assign, reshape_range
from next_paper_causal_calibration import row_hash
from next_paper_probability import probability_losses, paired_loss_effect
from next_paper_trajectory import digest, write_json, status

OUT = BASE / 'np015'
LEADS = [15, 720]
SOURCES = ['np013', 'np014']
FAMILIES = ['logistic', 'forest']
ARMS = ['direct', 'aug_np013', 'aug_np014', 'context_np013', 'context_np014']
PARAMS = dict(n_estimators=150, max_depth=16, min_samples_leaf=32,
              max_features=1.0, random_state=41, n_jobs=4)


def summary_features(paths):
    paths = np.asarray(paths, float)
    if paths.ndim != 3 or paths.shape[-1] != 17 or not np.isfinite(paths).all():
        raise ValueError('Finite member paths with 17 nodes required')
    mean_path = paths.mean(axis=1)
    ranges = np.ptp(paths, axis=2)
    nets = paths[:, :, -1] - paths[:, :, 0]
    return np.column_stack([mean_path.mean(axis=1) / 20, ranges.mean(axis=1) / 10,
                            nets.mean(axis=1) / 5, np.quantile(ranges, .1, axis=1) / 10,
                            np.quantile(ranges, .9, axis=1) / 10,
                            mean_path[:, 0] / 20, mean_path[:, -1] / 20])


def source_paths(data, split, lead, source):
    target = np.flatnonzero(data[split])
    if source == 'np013':
        neighbors = np.load(BASE / 'np013' / f'{split}_neighbors.npz')[f'{lead}__conditional']
        return np.maximum(data['gfs'][target, None, :] +
                          data['wind'][neighbors] - data['gfs'][neighbors], 0)
    old_neighbors = np.load(BASE / 'np013' / f'{split}_neighbors.npz')[f'{lead}__conditional']
    range_neighbors = np.load(BASE / 'np014' / f'{split}_range_neighbors.npz')[f'{lead}__conditional']
    donors = data['wind'][old_neighbors]
    old = np.maximum(data['gfs'][target, None, :] + data['wind'][old_neighbors] -
                     data['gfs'][old_neighbors], 0)
    ranges = np.ptp(data['wind'][range_neighbors], axis=2)
    assigned = rank_assign(np.ptp(old, axis=2), ranges)
    paths, _ = reshape_range(old, assigned, donors)
    return paths


def arm_features(data, split, lead, arm):
    idx = np.flatnonzero(data[split]); x = data[f'x_{lead}'][idx]
    if arm == 'direct': return x
    source = 'np013' if 'np013' in arm else 'np014'
    summaries = data[f'{split}_{lead}_{source}']
    if arm.startswith('aug'): return np.c_[x, summaries]
    return np.c_[data[f'known_{lead}'][idx], summaries]


def prepare(out):
    if out.exists(): raise RuntimeError('Fresh output directory required')
    source = dict(np.load(BASE / 'np013/dataset.npz'))
    data = {k: source[k] for k in ['times_ns', 'outcome', 'validation', 'test', 'wind', 'wind_valid', 'gfs']}
    for lead in LEADS:
        x, known = inputs(source, lead); data[f'x_{lead}'] = x; data[f'known_{lead}'] = known
        for split in ['validation', 'test']:
            for name in SOURCES:
                data[f'{split}_{lead}_{name}'] = summary_features(source_paths(source, split, lead, name))
    out.mkdir(parents=True); (out / 'models').mkdir()
    np.savez_compressed(out / 'dataset.npz', **data)
    support = [dict(split=s, windows=int(data[s].sum()), wind_complete=int(data['wind_valid'][data[s]].sum()),
                    **{f'class{i}': int((data['outcome'][data[s]] == i).sum()) for i in range(4)})
               for s in ['validation', 'test']]
    pd.DataFrame(support).to_csv(out / 'support.csv', index=False)
    tracked = [Path(__file__), BASE / 'np013/dataset.npz', BASE / 'np013/protocol.json',
               BASE / 'np013/verification.json', BASE / 'np014/protocol.json', BASE / 'np014/verification.json']
    tracked += [BASE / f'np013/{s}_neighbors.npz' for s in ['validation', 'test']]
    tracked += [BASE / f'np014/{s}_range_neighbors.npz' for s in ['validation', 'test']]
    tracked += [ROOT / 'script/next_paper_local_wind.py', ROOT / 'script/next_paper_range_distribution.py']
    protocol = dict(experiment='NP015', stage='PREQUENTIAL_LOCAL_WIND_SUMMARY_EVENT_FUSION',
                    question='Can predicted local-wind summaries improve event discrimination beyond the same issue-time inputs?',
                    leads=LEADS, sources=SOURCES, families=FAMILIES, arms=ARMS,
                    summary='Seven scaled features: mean wind, mean range, mean net, range q10/q90, mean start/end',
                    source='NP013 conditional paths and NP014 coupled range-constrained paths reconstructed from rolling neighbors',
                    no_future_input=True, fitting='Fit on post-2024-09-01 validation predictions; chronological test holdout',
                    logistic='L2 multinomial lambda .001', forest=PARAMS,
                    direct='NP013 186 issue-time features; augmented arms append seven predicted summaries',
                    context='Known 9-dimensional anchor/calendar context plus seven summaries',
                    selection='No test selection; frozen NP013, NP014 and GFS references retained',
                    inference='Paired 3/7/14-day calendar blocks, 2000 draws, seed41',
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in tracked},
                    dataset_sha256=digest(out / 'dataset.npz'))
    write_json(out / 'protocol.json', protocol); status(out, 'PREPARED', support=support)
    print(pd.DataFrame(support).to_string(index=False), flush=True)


def sources(out):
    p = json.loads((out / 'protocol.json').read_text())
    for name, sha in p['source_sha256'].items(): assert digest(ROOT / name) == sha, name
    assert digest(out / 'dataset.npz') == p['dataset_sha256']
    return p


def fit_model(x, y, family):
    if family == 'logistic':
        model = LogisticRegression(C=1 / (len(y) * .001), solver='lbfgs', penalty='l2',
                                   max_iter=2000, tol=1e-7, random_state=41).fit(x, y)
    else: model = ExtraTreesClassifier(**PARAMS).fit(x, y)
    np.testing.assert_array_equal(model.classes_, np.arange(4)); return model


def run_fit(out, data):
    train = np.flatnonzero(data['validation']); y = data['outcome'][train]; models = {}; audit = []; hashes = {}
    for lead in LEADS:
        for arm in ARMS:
            x = arm_features(data, 'validation', lead, arm)
            for family in FAMILIES:
                key = f'{lead}__{arm}__{family}'; model = fit_model(x, y, family)
                name = f'models/{key}.joblib'; joblib.dump(model, out / name, compress=3); hashes[name] = digest(out / name)
                models[key] = model; audit.append(dict(key=key, path=name, rows=len(train),
                                                       row_sha256=row_hash(train), feature_count=x.shape[1], family=family))
    pd.DataFrame(audit).to_csv(out / 'fit_audit.csv', index=False); write_json(out / 'models.json', hashes); return models


def references(split):
    p13 = np.load(BASE / 'np013' / f'{split}_predictions.npz'); p14 = np.load(BASE / 'np014' / f'{split}_predictions.npz')
    return {f'{l}__np013_selected': p13[f'{l}__selected'] for l in LEADS} | \
           {f'{l}__np014_selected': p14[f'{l}__selected'] for l in LEADS} | \
           {f'{l}__frozen_gfs': p14[f'{l}__frozen_gfs'] for l in LEADS}


def predict(out, data, split, models):
    result = {}
    for lead in LEADS:
        for arm in ARMS:
            x = arm_features(data, split, lead, arm)
            for family in FAMILIES: result[f'{lead}__{arm}__{family}'] = models[f'{lead}__{arm}__{family}'].predict_proba(x)
    result.update(references(split)); np.savez_compressed(out / f'{split}_predictions.npz', **result); return result


def evaluate(out, data, split, predictions):
    idx = np.flatnonzero(data[split]); y = data['outcome'][idx]; losses = {k: probability_losses(y, v) for k, v in predictions.items()}
    summary = [dict(model=k, metric=m, windows=len(y), loss=float(v.mean())) for k, z in losses.items() for m, v in z.items()]
    pd.DataFrame(summary).to_csv(out / f'{split}_summary.csv', index=False)
    if split != 'test': return
    clock = pd.to_datetime(data['times_ns'][idx], utc=True); rows = []; comparisons = []
    for lead in LEADS:
        for family in FAMILIES:
            for arm in ['aug_np013', 'aug_np014', 'context_np013', 'context_np014']:
                comparisons.append((f'{lead}__{arm}__{family}', f'{lead}__direct__{family}', 'summary_vs_direct'))
            comparisons.append((f'{lead}__aug_np013__{family}', f'{lead}__np013_selected', 'vs_np013'))
            comparisons.append((f'{lead}__aug_np014__{family}', f'{lead}__frozen_gfs', 'vs_gfs'))
    for a, b, kind in comparisons:
        for metric in ['multiclass_brier', 'return_brier', 'up_brier', 'down_brier']:
            for days in [3, 7, 14]: rows.append(dict(candidate=a, reference=b, comparison=kind, metric=metric,
                windows=len(y), **paired_loss_effect(clock, losses[a][metric], losses[b][metric], days)))
    pd.DataFrame(rows).to_csv(out / 'paired_intervals.csv', index=False)


def validate(out):
    sources(out); data = dict(np.load(out / 'dataset.npz')); status(out, 'VALIDATING')
    models = run_fit(out, data); evaluate(out, data, 'validation', predict(out, data, 'validation', models))
    write_json(out / 'freeze.json', dict(protocol_sha256=digest(out / 'protocol.json'),
        model_sha256=json.loads((out / 'models.json').read_text()), fit_audit_sha256=digest(out / 'fit_audit.csv')))
    status(out, 'VALIDATION_FROZEN', models=len(models))


def test(out):
    sources(out); f = json.loads((out / 'freeze.json').read_text())
    assert digest(out / 'protocol.json') == f['protocol_sha256'] and digest(out / 'fit_audit.csv') == f['fit_audit_sha256']
    for name, sha in f['model_sha256'].items(): assert digest(out / name) == sha
    data = dict(np.load(out / 'dataset.npz')); models = {}
    for name in f['model_sha256']:
        lead, arm, family = Path(name).stem.split('__'); models[f'{lead}__{arm}__{family}'] = joblib.load(out / name)
    status(out, 'TESTING'); evaluate(out, data, 'test', predict(out, data, 'test', models)); status(out, 'COMPLETE', models=len(models))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['prepare', 'validate', 'test'], required=True)
    parser.add_argument('--output', type=Path, default=OUT); args = parser.parse_args()
    with threadpool_limits(limits=1): {'prepare': prepare, 'validate': validate, 'test': test}[args.phase](args.output)
