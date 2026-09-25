"""NP016: prequential event-probability calibration with predicted wind summaries.

NP013/NP014 already produce rolling event probabilities from issue-time inputs.
This experiment freezes those probabilities and fits one validation-period
multinomial calibration map, with and without seven predicted local-wind
summaries. The summaries come from rolling historical neighbors and never use
the target future wind. Test is a chronological holdout, so this isolates
summary information from the static-versus-rolling model difference seen in
NP015. This is a calibration application, not a new calibration algorithm.
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from next_paper_local_wind import BASE, ROOT
from next_paper_causal_calibration import row_hash
from next_paper_probability import probability_losses, paired_loss_effect
from next_paper_trajectory import digest, write_json, status

OUT = BASE / 'np016'
LEADS = [15, 720]
MODES = ['base', 'np013', 'np014', 'context_np013', 'context_np014']


def features(probability, known, summary=None, include_known=True):
    logp = np.log(np.clip(probability, 1e-6, 1))
    if summary is None: return np.c_[logp, known] if include_known else logp
    return np.c_[logp, known, summary] if include_known else np.c_[logp, summary]


def prepare(out):
    if out.exists(): raise RuntimeError('Fresh output directory required')
    p13 = dict(np.load(BASE / 'np013/dataset.npz'))
    data = {k: p13[k] for k in ['times_ns', 'outcome', 'validation', 'test']}
    p13v = np.load(BASE / 'np013/validation_predictions.npz'); p13t = np.load(BASE / 'np013/test_predictions.npz')
    p14v = np.load(BASE / 'np014/validation_predictions.npz'); p14t = np.load(BASE / 'np014/test_predictions.npz')
    for lead in LEADS:
        j = LEADS.index(lead)
        data[f'known_{lead}'] = np.c_[p13['anchor'][j], p13['anchor'][j] ** 2, p13['calendar']]
        data[f'np013_prob_validation_{lead}'] = p13v[f'{lead}__selected']
        data[f'np013_prob_test_{lead}'] = p13t[f'{lead}__selected']
        data[f'np014_prob_validation_{lead}'] = p14v[f'{lead}__selected']
        data[f'np014_prob_test_{lead}'] = p14t[f'{lead}__selected']
        # Summary features are held in the NP015 prepared dataset, generated
        # before this map and from saved rolling future-wind neighbors.
        p15 = np.load(BASE / 'np015/dataset.npz')
        data[f'np013_summary_validation_{lead}'] = p15[f'validation_{lead}_np013']
        data[f'np013_summary_test_{lead}'] = p15[f'test_{lead}_np013']
        data[f'np014_summary_validation_{lead}'] = p15[f'validation_{lead}_np014']
        data[f'np014_summary_test_{lead}'] = p15[f'test_{lead}_np014']
    out.mkdir(parents=True)
    np.savez_compressed(out / 'dataset.npz', **data)
    pd.DataFrame([dict(split=s, windows=int(data[s].sum()),
                       **{f'class{i}': int((data['outcome'][data[s]] == i).sum()) for i in range(4)})
                  for s in ['validation', 'test']]).to_csv(out / 'support.csv', index=False)
    tracked = [Path(__file__), BASE / 'np013/dataset.npz', BASE / 'np013/protocol.json',
               BASE / 'np013/verification.json', BASE / 'np014/protocol.json', BASE / 'np014/verification.json',
               BASE / 'np015/dataset.npz', BASE / 'np015/protocol.json']
    tracked += [BASE / f'np013/{s}_predictions.npz' for s in ['validation', 'test']]
    tracked += [BASE / f'np014/{s}_predictions.npz' for s in ['validation', 'test']]
    protocol = dict(experiment='NP016', stage='PREQUENTIAL_LOCAL_WIND_SUMMARY_CALIBRATION',
                    question='Do predicted local-wind summaries add information to an already rolling event probability?',
                    modes=MODES, leads=LEADS, validation_windows=2512, test_windows=6496,
                    calibration='Multinomial log-probability map plus known anchor/calendar context; lambda=.001; C=1/(n*lambda)',
                    base='Frozen NP013 selected probability, with NP014 selected probability as source control',
                    summary='NP015 seven scaled local-wind summaries from rolling historical neighbors; no true future wind',
                    selection='No mode/test selection; every map and frozen probability retained',
                    inference='Fixed-test paired 3/7/14-day calendar blocks, 2000 draws, seed41',
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in tracked},
                    dataset_sha256=digest(out / 'dataset.npz'))
    write_json(out / 'protocol.json', protocol); status(out, 'PREPARED')


def sources(out):
    p = json.loads((out / 'protocol.json').read_text())
    for name, sha in p['source_sha256'].items(): assert digest(ROOT / name) == sha, name
    assert digest(out / 'dataset.npz') == p['dataset_sha256']; return p


def fit_maps(out, data):
    train = np.flatnonzero(data['validation']); y = data['outcome'][train]; records = {}; hashes = {}; audit = []
    for lead in LEADS:
        for mode in MODES:
            prob = data[f'np013_prob_validation_{lead}'] if mode != 'context_np014' and mode != 'np014' else data[f'np014_prob_validation_{lead}']
            source = 'np014' if 'np014' in mode else 'np013'
            summary = None if mode == 'base' else data[f'{source}_summary_validation_{lead}']
            x = features(prob, data[f'known_{lead}'][train], summary, include_known=not mode.startswith('context_'))
            model = LogisticRegression(C=1 / (len(train) * .001), solver='lbfgs', penalty='l2',
                                       max_iter=2000, tol=1e-7, random_state=41).fit(x, y)
            key = f'{lead}__{mode}'; name = f'{key}.joblib'; joblib.dump(model, out / name, compress=3)
            hashes[name] = digest(out / name); records[key] = model
            audit.append(dict(key=key, path=name, rows=len(train), row_sha256=row_hash(train), feature_count=x.shape[1]))
    pd.DataFrame(audit).to_csv(out / 'fit_audit.csv', index=False); write_json(out / 'models.json', hashes); return records


def predict(data, models, split):
    result = {}
    for lead in LEADS:
        for mode in MODES:
            source = 'np014' if 'np014' in mode else 'np013'
            prob = data[f'np013_prob_{split}_{lead}'] if source == 'np013' else data[f'np014_prob_{split}_{lead}']
            summary = None if mode == 'base' else data[f'{source}_summary_{split}_{lead}']
            x = features(prob, data[f'known_{lead}'][np.flatnonzero(data[split])], summary, include_known=not mode.startswith('context_'))
            result[f'{lead}__{mode}'] = models[f'{lead}__{mode}'].predict_proba(x)
    for lead in LEADS:
        result[f'{lead}__raw_np013'] = data[f'np013_prob_{split}_{lead}']
        result[f'{lead}__raw_np014'] = data[f'np014_prob_{split}_{lead}']
    np.savez_compressed(OUT / f'{split}_predictions.npz', **result); return result


def evaluate(out, data, split, predictions):
    idx = np.flatnonzero(data[split]); y = data['outcome'][idx]
    losses = {k: probability_losses(y, v) for k, v in predictions.items()}
    pd.DataFrame([dict(model=k, metric=m, windows=len(y), loss=float(v.mean()))
                  for k, z in losses.items() for m, v in z.items()]).to_csv(out / f'{split}_summary.csv', index=False)
    if split == 'test':
        clock = pd.to_datetime(data['times_ns'][idx], utc=True); rows = []
        for lead in LEADS:
            for mode in ['np013', 'np014', 'context_np013', 'context_np014']:
                for family, ref in [('summary', f'{lead}__base'), ('raw', f'{lead}__raw_np013')]:
                    candidate = f'{lead}__{mode}' if family == 'summary' else ref
                    if family == 'raw': continue
                    for metric in ['multiclass_brier', 'return_brier', 'up_brier', 'down_brier']:
                        for days in [3, 7, 14]: rows.append(dict(candidate=candidate, reference=ref, comparison='summary_vs_base', metric=metric,
                            windows=len(y), **paired_loss_effect(clock, losses[candidate][metric], losses[ref][metric], days)))
            for candidate, ref in [(f'{lead}__np013', f'{lead}__raw_np013'), (f'{lead}__context_np013', f'{lead}__raw_np013'),
                                   (f'{lead}__np014', f'{lead}__raw_np014')]:
                for metric in ['multiclass_brier', 'return_brier', 'up_brier', 'down_brier']:
                    for days in [3, 7, 14]: rows.append(dict(candidate=candidate, reference=ref, comparison='vs_raw', metric=metric,
                        windows=len(y), **paired_loss_effect(clock, losses[candidate][metric], losses[ref][metric], days)))
        pd.DataFrame(rows).to_csv(out / 'paired_intervals.csv', index=False)


def validate(out):
    sources(out); data = dict(np.load(out / 'dataset.npz')); status(out, 'VALIDATING')
    models = fit_maps(out, data); predict(data, models, 'validation'); evaluate(out, data, 'validation', predict(data, models, 'validation'))
    write_json(out / 'freeze.json', dict(protocol_sha256=digest(out / 'protocol.json'), model_sha256=json.loads((out / 'models.json').read_text()), fit_audit_sha256=digest(out / 'fit_audit.csv')))
    status(out, 'VALIDATION_FROZEN', models=len(models))


def test(out):
    sources(out); f = json.loads((out / 'freeze.json').read_text()); assert digest(out / 'protocol.json') == f['protocol_sha256']
    assert digest(out / 'fit_audit.csv') == f['fit_audit_sha256']
    for name, sha in f['model_sha256'].items(): assert digest(out / name) == sha
    data = dict(np.load(out / 'dataset.npz')); models = {}
    for name in f['model_sha256']:
        lead, mode = Path(name).stem.split('__'); models[f'{lead}__{mode}'] = joblib.load(out / name)
    status(out, 'TESTING'); pred = predict(data, models, 'test'); evaluate(out, data, 'test', pred); status(out, 'COMPLETE', models=len(models))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['prepare', 'validate', 'test'], required=True)
    parser.add_argument('--output', type=Path, default=OUT); args = parser.parse_args()
    with threadpool_limits(limits=1): {'prepare': prepare, 'validate': validate, 'test': test}[args.phase](args.output)
