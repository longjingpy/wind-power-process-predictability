"""NP001: issue-time power trajectory information, with frozen validation.

Source methods and their role:
* v24 forecast_penalty_v24.model_inputs and frame_and_splits: reuse the actual
  causal-prefix interface and chronological calendar, not the fee objective.
* Current event manuscript, Supplementary S26 (unpublished; no DOI), complete
  wind-process reconstruction: subtract each curve's own endpoint chord to
  measure internal evolution separately from its endpoint. Here the target is
  FUTURE power, with its anchor observed one minute before issue.
* Geurts et al. (2006), Extremely randomized trees,
  doi:10.1007/s10994-006-6226-1, abstract/method: randomized regression trees
  supply a nonlinear reference so information effects are not judged by one
  linear model alone; use sklearn's native multi-output squared-error fit.
* Hoerl & Kennard (1970), Ridge Regression: Biased Estimation for Nonorthogonal
  Problems, doi:10.1080/00401706.1970.10488634, abstract: regularized linear
  regression controls correlated historical/weather predictors as a reference.
* Kunsch (1989), doi:10.1214/aos/1176347265: dependent-data resampling motivates
  the inherited paired calendar-block implementation, not IID event sampling.

This is an exploratory deterministic-trajectory pilot, not a probability
model, a weather-mechanism experiment, or an estimate of theoretical limits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from forecast_penalty_v24 import CAP, model_inputs, frame_and_splits
from wind_events.paired_probability import block_design

OUT = ROOT / 'outputs/next_paper/np001'
POWER = ROOT / 'outputs/protocol_benchmark_v24/economics/jiangsu_native/farm_15min.parquet'
WEATHER = ROOT / 'data/weather_v24/pizhou_fixed_lead/fixed_lead_hourly.parquet'
STEPS = 16
PREFIXES = [2, 4, 8, 16]
SOURCES = [POWER, WEATHER, Path(__file__),
           ROOT / 'script/forecast_penalty_v24.py',
           ROOT / 'script/prepare_pizhou_policy_v24.py',
           ROOT / 'script/fetch_pizhou_forecasts_v24.py',
           ROOT / 'script/physical_process_v26.py',
           ROOT / 'src/wind_events/paired_probability.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def status(out, state, **extra):
    write_json(out / 'status.json', {'state': state,
               'updated_at': datetime.now(timezone.utc).isoformat(), **extra})


def weather_features(data, clock, steps=STEPS):
    """Official Previous Runs API: previous_day1 = valid time minus 24 h.

    https://open-meteo.com/en/docs/previous-runs-api, lead-time description.
    Validate BOTH hourly endpoints used by interpolation. Exact hourly samples
    do not borrow another hour. No actual publication timestamp is fabricated.
    """
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError('Unique sorted forecast validity times required')
    columns = [c for c in data if c.startswith(('jma_gsm_', 'gfs_global_'))]
    if len(columns) != 8:
        raise ValueError('Expected the eight archived JMA/GFS fields')
    parts, margins = [], []
    for step in range(steps + 1):
        target = clock + pd.Timedelta(minutes=15 * step)
        lower, upper = target.floor('h'), target.ceil('h')
        nominal_latest = upper - pd.Timedelta(hours=24)
        if not (nominal_latest < clock).all():
            raise ValueError('Weather forecast vintage must precede task issue')
        margins.append(np.asarray((clock - nominal_latest) / pd.Timedelta(hours=1)))
        weight = np.asarray((target - lower) / pd.Timedelta(hours=1))[:, None]
        left = data[columns].reindex(lower).to_numpy()
        right = data[columns].reindex(upper).to_numpy()
        parts.append(left * (1 - weight) + right * weight)
    return np.column_stack(parts), float(np.min(margins)), columns


def build_arrays(frame, weather):
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Unique sorted issue grid required')
    if not (np.diff(frame.index.as_unit('ns').asi8) == pd.Timedelta(minutes=15).value).all():
        raise ValueError('An explicit regular 15-minute grid with missing values is required')
    shapes, common, valid = model_inputs(frame, STEPS)
    history = np.column_stack([common, shapes])
    extra, margin, columns = weather_features(weather, frame.index)
    y = np.column_stack([frame.target_power.shift(-k).to_numpy() / CAP
                         for k in range(1, STEPS + 1)])
    daily = np.column_stack([frame.target_power.shift(96 - k).to_numpy() / CAP
                             for k in range(1, STEPS + 1)])
    valid &= np.isfinite(extra).all(axis=1) & np.isfinite(y).all(axis=1)
    valid &= np.isfinite(daily).all(axis=1) & ((y >= -.05) & (y <= 1.2)).all(axis=1)
    arrays = {'history': history, 'weather': np.column_stack([history, extra]),
              'actual': y, 'anchor': frame.available_power.to_numpy() / CAP,
              'daily': np.clip(daily, 0, 1), 'eligible': valid,
              'times_ns': frame.index.as_unit('ns').asi8}
    return arrays, {'nominal_minimum_issue_margin_hours': margin,
                    'weather_columns': columns,
                    'history_features': history.shape[1],
                    'history_plus_weather_features': history.shape[1] + extra.shape[1]}


def split_masks(clock, eligible, first, second):
    end = clock + pd.Timedelta(hours=4)
    past = clock - pd.Timedelta(hours=24)
    return {'train': eligible & (end < first),
            'validation': eligible & (past >= first) & (end < second),
            'test': eligible & (past >= second)}


def structure_losses(actual, predicted, anchor):
    """S26 chord residual adapted to a one-minute pre-issue power anchor.

    Target k is 15*k+1 minutes after the anchor, so use physical elapsed time
    rather than treating that anchor as an observation at the issue itself.
    The true endpoint is used for evaluation only, never as an input.
    """
    if actual.shape != predicted.shape or len(anchor) != len(actual):
        raise ValueError('Aligned actual, prediction and anchor required')
    h = actual.shape[1]
    fraction = (15 * np.arange(1, h + 1) + 1) / (15 * h + 1)
    a = actual - anchor[:, None]
    p = predicted - anchor[:, None]
    a_internal = a - a[:, -1, None] * fraction
    p_internal = p - p[:, -1, None] * fraction
    return {'trajectory': np.mean((predicted - actual) ** 2, axis=1),
            'endpoint': (predicted[:, -1] - actual[:, -1]) ** 2,
            'internal_geometry': np.mean((p_internal - a_internal) ** 2, axis=1)}


def source_check(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    for relative, expected in protocol['source_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Source changed after preparation: {relative}')
    if digest(out / 'dataset.npz') != protocol['dataset_sha256']:
        raise RuntimeError('Prepared dataset changed')
    return protocol


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    frame, first, second = frame_and_splits()
    arrays, meta = build_arrays(frame, pd.read_parquet(WEATHER))
    arrays.update(split_masks(frame.index, arrays['eligible'], first, second))
    np.savez_compressed(out / 'dataset.npz', **arrays)
    support = []
    for split in ['train', 'validation', 'test']:
        times = frame.index[arrays[split]]
        if not len(times):
            raise ValueError(f'No common support in {split}')
        support.append({'split': split, 'windows': len(times), 'first_issue': str(times[0]),
                        'last_issue': str(times[-1]),
                        'first_target': str(times[0] + pd.Timedelta(minutes=15)),
                        'last_target': str(times[-1] + pd.Timedelta(hours=4))})
    pd.DataFrame(support).to_csv(out / 'support.csv', index=False)
    protocol = {'experiment': 'NP001', 'stage': 'EXPLORATORY_PREVIOUSLY_STUDIED_CALENDAR',
                'question': 'Do issued weather fields affect endpoint and internal trajectory forecasts differently?',
                'claims': 'Initial I1/I2 evidence only; no physical-regime or probabilistic claim',
                'capacity_mw': CAP, 'target': 'Native future quarter-hour point power / capacity',
                'forecast_steps': STEPS, 'reported_prefix_minutes': [15 * h for h in PREFIXES],
                'training_boundary_utc': str(first), 'validation_boundary_utc': str(second),
                'purge': '24 h history (including daily-repeat baseline) and 4 h future support within split',
                'comparison_support': 'all methods share all sixteen future points and both weather models',
                'weather': 'Fixed-24h-lead Previous Runs fields; administrative proxy; may span different cycles',
                'publication_time': 'Actual provider publish times unavailable; nominal endpoint margin verified',
                'history': 'Inherited v24 four-hour normalized raw25 plus nine causal scalar features',
                'baselines': ['persistence', 'daily_repeat'],
                'candidate_parameters': {'ridge_alpha': [1., 10.], 'extra_trees_min_leaf': [8, 32]},
                'extra_trees': {'n_estimators': 150, 'max_depth': 16, 'max_features': 1., 'seed': 41, 'n_jobs': 4},
                'selection': 'One shared parameter per family minimizes mean validation trajectory MSE of its history/weather arms',
                'refit': 'None after validation; no test selection',
                'uncertainty': 'Paired fixed calendar blocks 3/7/14 days, 2000 draws, seed41; primary7d; exploratory pointwise intervals',
                'scope': 'All eligible rolling windows, deterministic mean trajectories; not independent event samples',
                'runtime': {'python': sys.executable, 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in SOURCES},
                'dataset_sha256': digest(out / 'dataset.npz'), **meta}
    write_json(out / 'protocol.json', protocol)
    status(out, 'PREPARED', support=support)
    print(json.dumps(support, ensure_ascii=False), flush=True)


def validate(out):
    source_check(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'PREPARED':
        raise RuntimeError('Validation requires a fresh prepared run')
    data = np.load(out / 'dataset.npz')
    tr, va = data['train'], data['validation']
    y = data['actual']
    rows, choices = [], []
    for family, candidates in [('ridge', [1., 10.]), ('extra_trees', [8, 32])]:
        best = None
        for parameter in candidates:
            models, scores = {}, []
            for arm in ['history', 'weather']:
                if family == 'ridge':
                    model = make_pipeline(StandardScaler(), Ridge(alpha=parameter))
                else:
                    model = ExtraTreesRegressor(n_estimators=150, max_depth=16,
                        min_samples_leaf=int(parameter), max_features=1., random_state=41, n_jobs=4)
                model.fit(data[arm][tr], y[tr])
                q = np.clip(model.predict(data[arm][va]), 0, 1)
                mse = float(np.mean((q - y[va]) ** 2))
                models[arm] = model
                scores.append(mse)
                rows.append({'family': family, 'parameter': parameter, 'arm': arm,
                             'validation_mse': mse, 'validation_windows': int(va.sum())})
                pd.DataFrame(rows).to_csv(out / 'validation_candidates.csv', index=False)
                print('validation', family, parameter, arm, mse, flush=True)
            mean = float(np.mean(scores))
            if best is None or mean < best[0]:
                best = (mean, parameter, models)
        score, parameter, models = best
        for arm, model in models.items():
            joblib.dump(model, out / f'{family}_{arm}.joblib', compress=3)
        choices.append({'family': family, 'shared_parameter': parameter, 'pooled_validation_mse': score})
    # References are evaluated on the same validation population and never selected on test.
    references = {'persistence': np.repeat(data['anchor'][va, None], STEPS, axis=1),
                  'daily_repeat': data['daily'][va]}
    reference_scores = {name: float(np.mean((q - y[va]) ** 2)) for name, q in references.items()}
    artifacts = {p.name: digest(p) for p in sorted(out.glob('*.joblib'))}
    write_json(out / 'selection.json', {'state': 'FROZEN_BEFORE_TEST', 'choices': choices,
               'validation_references_mse': reference_scores, 'model_sha256': artifacts,
               'protocol_sha256': digest(out / 'protocol.json')})
    status(out, 'VALIDATION_FROZEN', choices=choices)


def paired_effect(times, candidate, reference, days):
    index, weights = block_design(times, days, 2000, 41)
    size = weights.shape[1]
    a = np.bincount(index, weights=candidate, minlength=size)
    b = np.bincount(index, weights=reference, minlength=size)
    denominator = weights @ b
    draws = 100 * (1 - np.sqrt(np.divide(weights @ a, denominator,
                 out=np.full(len(weights), np.nan), where=denominator > 0)))
    finite = draws[np.isfinite(draws)]
    lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
    return {'relative_rmse_reduction_pct': 100 * (1 - np.sqrt(candidate.mean() / reference.mean()))
                if reference.mean() > 0 else np.nan,
            'low': lo, 'high': hi, 'block_days': days,
            'occupied_blocks': size, 'valid_draws': len(finite)}


def test(out):
    source_check(out)
    if json.loads((out / 'status.json').read_text())['state'] != 'VALIDATION_FROZEN':
        raise RuntimeError('A frozen, not-yet-tested run is required')
    selection = json.loads((out / 'selection.json').read_text())
    if selection['protocol_sha256'] != digest(out / 'protocol.json'):
        raise RuntimeError('Protocol changed after selection')
    for name, expected in selection['model_sha256'].items():
        if digest(out / name) != expected:
            raise RuntimeError('Selected model changed')
    if (out / 'test_summary.csv').exists():
        raise RuntimeError('Test results already exist')
    data = np.load(out / 'dataset.npz')
    te = data['test']
    y, anchor = data['actual'][te], data['anchor'][te]
    times = pd.to_datetime(data['times_ns'][te], utc=True)
    predictions = {'persistence': np.clip(np.repeat(anchor[:, None], STEPS, axis=1), 0, 1),
                   'daily_repeat': data['daily'][te]}
    for family in ['ridge', 'extra_trees']:
        for arm in ['history', 'weather']:
            model = joblib.load(out / f'{family}_{arm}.joblib')
            predictions[f'{family}_{arm}'] = np.clip(model.predict(data[arm][te]), 0, 1)
    if not all(np.isfinite(p).all() and p.shape == y.shape for p in predictions.values()):
        raise AssertionError('Nonfinite or misaligned predictions')
    panel = pd.DataFrame({'issue_time': np.repeat(times, STEPS),
             'horizon_minutes': np.tile(np.arange(1, STEPS + 1) * 15, len(y)),
             'actual_pu': y.ravel(), 'anchor_pu': np.repeat(anchor, STEPS)})
    for name, prediction in predictions.items():
        panel[name] = prediction.ravel()
    panel.to_parquet(out / 'test_predictions.parquet', index=False)
    rows, effects = [], []
    comparisons = [('extra_trees_weather', 'extra_trees_history'), ('ridge_weather', 'ridge_history')]
    comparisons += [(name, 'persistence') for name in predictions if name != 'persistence']
    for h in PREFIXES:
        losses = {name: structure_losses(y[:, :h], p[:, :h], anchor)
                  for name, p in predictions.items()}
        for name, metrics in losses.items():
            for metric, values in metrics.items():
                rows.append({'model': name, 'horizon_minutes': h * 15, 'metric': metric,
                             'windows': len(y), 'rmse_pu': float(np.sqrt(values.mean()))})
        for candidate, reference in comparisons:
            for metric in losses[candidate]:
                for days in [3, 7, 14]:
                    effect = paired_effect(times, losses[candidate][metric], losses[reference][metric], days)
                    effects.append({'candidate': candidate, 'reference': reference,
                        'horizon_minutes': h * 15, 'metric': metric, 'windows': len(y), **effect})
    pd.DataFrame(rows).to_csv(out / 'test_summary.csv', index=False)
    effects = pd.DataFrame(effects)
    effects.to_csv(out / 'paired_intervals.csv', index=False)
    status(out, 'COMPLETE', test_windows=len(y), test_prediction_rows=len(panel),
           scope='Exploratory rolling-window deterministic trajectory information; no event/weather causality claim')
    print(effects[(effects.block_days == 7) & (effects.reference == 'extra_trees_history')].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'validate', 'test'], required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'validate': validate, 'test': test}[args.phase](args.output)


if __name__ == '__main__':
    main()
