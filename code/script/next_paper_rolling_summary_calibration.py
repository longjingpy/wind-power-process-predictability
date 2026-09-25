"""NP017: rolling calibration of NP013/NP014 probabilities with local summaries.

NP016 fit one calibration map on the validation period and then applied it to
the test period. NP017 uses a fixed 56-day window and UTC Monday updates, with
only matured validation/test summaries available at each update. This isolates
distribution shift from the summary signal. The raw NP013/NP014 rolling
probabilities remain unchanged references; a single validation-selected blend
weight is frozen before test.
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

from next_paper_future_wind import monday_floor
from next_paper_summary_calibration import features
from next_paper_causal_calibration import row_hash
from next_paper_probability import probability_losses, paired_loss_effect
from next_paper_trajectory import digest, write_json, status

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
OUT = ROOT / 'outputs/next_paper/np017'
P13 = ROOT / 'outputs/next_paper/np013'
P14 = ROOT / 'outputs/next_paper/np014'
P15 = ROOT / 'outputs/next_paper/np015'
LEADS = [15, 720]
MODES = ['raw_np013', 'raw_np014', 'base_np013', 'summary_np013', 'summary_np014']
WEIGHTS = [0., .25, .5, .75, 1.]
DAY = pd.Timedelta(days=1).value
MINUTE = pd.Timedelta(minutes=1).value
HOUR = pd.Timedelta(hours=1).value


def maturity(times):
    return ((np.asarray(times, dtype=np.int64) + 4 * HOUR + HOUR - 1) // HOUR) * HOUR + MINUTE


def fit_map(x, y):
    counts = np.bincount(y, minlength=4)
    if len(y) < 100 or (counts == 0).any():
        return dict(kind='frequency', probability=(counts + 1) / (len(y) + 4), counts=counts, rows=len(y))
    model = LogisticRegression(C=1 / (len(y) * .001), solver='lbfgs', penalty='l2',
                               max_iter=2000, tol=1e-7, random_state=41).fit(x, y)
    return dict(kind='logistic', coefficient=model.coef_, intercept=model.intercept_,
                classes=model.classes_, counts=counts, rows=len(y), C=model.C, n_iter=int(model.n_iter_.max()))


def apply(record, x):
    if record['kind'] == 'frequency': return np.tile(record['probability'], (len(x), 1))
    return softmax(x @ record['coefficient'].T + record['intercept'], axis=1)


def prepare(out):
    if out.exists(): raise RuntimeError('Fresh output directory required')
    p13 = dict(np.load(P13 / 'dataset.npz')); p15 = dict(np.load(P15 / 'dataset.npz'))
    data = {k: p13[k] for k in ['times_ns', 'outcome', 'validation', 'test']}
    data['maturity_ns'] = maturity(data['times_ns']); data['summary_valid'] = np.zeros(len(data['times_ns']), bool)
    data['summary_valid'] = p13['validation'] | p13['test']
    for lead in LEADS:
        j = [15, 720].index(lead)
        data[f'known_{lead}'] = np.c_[p13['anchor'][j], p13['anchor'][j] ** 2, p13['calendar']]
        for split in ['validation', 'test']:
            data[f'p13_{split}_{lead}'] = np.load(P13 / f'{split}_predictions.npz')[f'{lead}__selected']
            data[f'p14_{split}_{lead}'] = np.load(P14 / f'{split}_predictions.npz')[f'{lead}__selected']
            data[f's13_{split}_{lead}'] = p15[f'{split}_{lead}_np013']
            data[f's14_{split}_{lead}'] = p15[f'{split}_{lead}_np014']
    out.mkdir(parents=True); (out / 'models').mkdir(); np.savez_compressed(out / 'dataset.npz', **data)
    tracked = [Path(__file__), P13 / 'dataset.npz', P13 / 'protocol.json', P13 / 'verification.json',
               P14 / 'protocol.json', P14 / 'verification.json', P15 / 'dataset.npz', P15 / 'protocol.json']
    tracked += [P13 / f'{s}_predictions.npz' for s in ['validation', 'test']]
    tracked += [P14 / f'{s}_predictions.npz' for s in ['validation', 'test']]
    protocol = dict(experiment='NP017', stage='ROLLING_LOCAL_WIND_SUMMARY_CALIBRATION', leads=LEADS, modes=MODES,
                    window_days=56, update='UTC Monday 00, labels matured at ceil(T+4h)+1min',
                    summary='NP015 prequential NP013/NP014 seven-dimensional predicted wind summaries',
                    fitting='Only summary-valid rows in the preceding 56 days and mature before update',
                    lambda_=0.001, weights=WEIGHTS, selection='One global blend weight on validation; no test selection',
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in tracked}, dataset_sha256=digest(out / 'dataset.npz'))
    write_json(out / 'protocol.json', protocol); status(out, 'PREPARED')


def sources(out):
    p = json.loads((out / 'protocol.json').read_text())
    for name, sha in p['source_sha256'].items(): assert digest(ROOT / name) == sha, name
    assert digest(out / 'dataset.npz') == p['dataset_sha256']; return p


def mode_features(data, split, lead, mode, rows):
    source = 'p14' if 'np014' in mode else 'p13'
    probability = data[f'{source}_{split}_{lead}']
    known = data[f'known_{lead}'][rows]
    if mode.startswith('raw'): return probability
    summary = None if mode.startswith('base') else data[f's{source[-3:]}_{split}_{lead}']
    return features(probability, known, summary, include_known=True)


def fit_for_update(data, update, lead, mode):
    valid = data['summary_valid'] & (data['times_ns'] >= update - 56 * DAY) & (data['maturity_ns'] <= update)
    rows = np.flatnonzero(valid)
    split = 'validation' if rows.size and np.all(data['validation'][rows]) else 'test'
    # Features are stored by split. Build aligned arrays directly from their
    # global row locations, avoiding any future truth or summary reconstruction.
    if mode == 'raw_np013':
        p = np.load(P13 / 'validation_predictions.npz')
    counts = np.bincount(data['outcome'][rows], minlength=4)
    if len(rows) < 100 or (counts == 0).any():
        return fit_map(np.zeros((len(rows), 1)), data['outcome'][rows]), rows
    # Rows are drawn from validation then test; use saved global arrays assembled below.
    x = data[f'features_{lead}_{mode}'][rows]
    return fit_map(x, data['outcome'][rows]), rows


def build_features(data):
    for lead in LEADS:
        for mode in MODES:
            source = 'p14' if 'np014' in mode else 'p13'
            p = np.r_[data[f'p13_validation_{lead}'] if source == 'p13' else data[f'p14_validation_{lead}'],
                      data[f'p13_test_{lead}'] if source == 'p13' else data[f'p14_test_{lead}']]
            s = np.r_[data[f's13_validation_{lead}'] if source == 'p13' else data[f's14_validation_{lead}'],
                      data[f's13_test_{lead}'] if source == 'p13' else data[f's14_test_{lead}']]
            # Validation and test arrays are concatenated in their own order;
            # map them back to global row indices.
            global_rows = np.r_[np.flatnonzero(data['validation']), np.flatnonzero(data['test'])]
            known = data[f'known_{lead}'][global_rows]
            if mode.startswith('raw'): x = p
            elif mode.startswith('base'): x = features(p, known, None)
            else: x = features(p, known, s, include_known=True)
            arr = np.empty((len(data['times_ns']), x.shape[1])); arr[:] = np.nan; arr[global_rows] = x
            data[f'features_{lead}_{mode}'] = arr
    return data


def run_phase(out, data, split):
    target = np.flatnonzero(data[split]); predictions = {}; audit = []; hashes = {}
    for lead in LEADS:
        schedule = monday_floor(data['times_ns'][target] - lead * MINUTE)
        for mode in MODES: predictions[f'{lead}__{mode}'] = np.empty((len(target), 4))
        for update in np.unique(schedule):
            rows_base = np.flatnonzero(data['summary_valid'] & (data['times_ns'] >= update - 56 * DAY) & (data['maturity_ns'] <= update))
            take = np.flatnonzero(schedule == update); y = data['outcome'][rows_base]
            for mode in MODES:
                x = data[f'features_{lead}_{mode}'][rows_base]; record = fit_map(x, y)
                xt = data[f'features_{lead}_{mode}'][target[take]]; predictions[f'{lead}__{mode}'][take] = apply(record, xt)
                audit.append(dict(lead=lead, mode=mode, update_ns=int(update), rows=len(rows_base),
                                  row_sha256=row_hash(rows_base), kind=record['kind'], n_iter=record.get('n_iter', 0)))
            print('fit', split, lead, pd.Timestamp(update, tz='UTC'), len(rows_base), flush=True)
    # Raw references are identical to the source rolling probabilities.
    for lead in LEADS:
        source = 'p13' if split else 'p13'; predictions[f'{lead}__raw_source'] = data[f'p13_{split}_{lead}']
    np.savez_compressed(out / f'{split}_predictions.npz', **predictions); pd.DataFrame(audit).to_csv(out / f'{split}_audit.csv', index=False)
    return predictions


def evaluate(out, data, split, pred):
    idx = np.flatnonzero(data[split]); y = data['outcome'][idx]; losses = {k: probability_losses(y, v) for k, v in pred.items()}
    pd.DataFrame([dict(model=k, metric=m, windows=len(y), loss=float(v.mean())) for k, z in losses.items() for m, v in z.items()]).to_csv(out / f'{split}_summary.csv', index=False)
    if split == 'test':
        clock = pd.to_datetime(data['times_ns'][idx], utc=True); rows = []
        for lead in LEADS:
            for mode in ['base_np013', 'summary_np013', 'summary_np014']:
                for metric in ['multiclass_brier', 'return_brier', 'up_brier', 'down_brier']:
                    for days in [3, 7, 14]: rows.append(dict(candidate=f'{lead}__{mode}', reference=f'{lead}__raw_source', comparison='rolling_summary_vs_raw', metric=metric, windows=len(y), **paired_loss_effect(clock, losses[f'{lead}__{mode}'][metric], losses[f'{lead}__raw_source'][metric], days)))
        pd.DataFrame(rows).to_csv(out / 'paired_intervals.csv', index=False)


def validate(out):
    sources(out); data = build_features(dict(np.load(out / 'dataset.npz'))); status(out, 'VALIDATING')
    pred = run_phase(out, data, 'validation'); evaluate(out, data, 'validation', pred)
    write_json(out / 'freeze.json', dict(protocol_sha256=digest(out / 'protocol.json'), dataset_sha256=digest(out / 'dataset.npz'),
                                         validation_audit_sha256=digest(out / 'validation_audit.csv'))); status(out, 'VALIDATION_FROZEN')


def test(out):
    sources(out); f = json.loads((out / 'freeze.json').read_text()); assert digest(out / 'protocol.json') == f['protocol_sha256']
    assert digest(out / 'validation_audit.csv') == f['validation_audit_sha256']; data = build_features(dict(np.load(out / 'dataset.npz'))); status(out, 'TESTING')
    pred = run_phase(out, data, 'test'); evaluate(out, data, 'test', pred); status(out, 'COMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['prepare', 'validate', 'test'], required=True); parser.add_argument('--output', type=Path, default=OUT); args = parser.parse_args()
    with threadpool_limits(limits=1): {'prepare': prepare, 'validate': validate, 'test': test}[args.phase](args.output)
