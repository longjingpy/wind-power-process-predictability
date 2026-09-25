from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

try:  # optional dependency
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except Exception:  # pragma: no cover - optional runtime dependency
    ARIMA = None  # type: ignore[assignment]
    SARIMAX = None  # type: ignore[assignment]
    ConvergenceWarning = Warning  # type: ignore[assignment]


@dataclass
class ArimaSpec:
    order: tuple[int, int, int]
    aic: float


@dataclass
class ArimaFitResult:
    order: tuple[int, int, int]
    aic: float
    model_result: object


@dataclass
class SarimaFitResult:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float
    model_result: object


def _require_statsmodels() -> None:
    if ARIMA is None:
        raise ImportError(
            "statsmodels is required for ARIMA baseline. "
            "Install with: pip install statsmodels"
        )


def _require_statsmodels_sarima() -> None:
    if SARIMAX is None:
        raise ImportError(
            "statsmodels is required for SARIMA baseline. "
            "Install with: pip install statsmodels"
        )


def auto_arima_order(
    series: np.ndarray,
    max_p: int,
    max_d: int,
    max_q: int,
    *,
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> ArimaSpec:
    """Select ARIMA order by AIC grid search.

    References:
      - Akaike, H. (1974) "A new look at the statistical model identification",
        IEEE Transactions on Automatic Control. DOI: 10.1109/TAC.1974.1100705
    """
    _require_statsmodels()
    best = ArimaSpec(order=(1, 0, 0), aic=float("inf"))
    y = np.asarray(series, dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and d == 0 and q == 0:
                        continue
                    try:
                        res = ARIMA(
                            y,
                            order=(p, d, q),
                            enforce_stationarity=enforce_stationarity,
                            enforce_invertibility=enforce_invertibility,
                        ).fit()
                        aic = float(res.aic)
                        if np.isfinite(aic) and aic < best.aic:
                            best = ArimaSpec(order=(p, d, q), aic=aic)
                    except Exception:
                        continue
    if not np.isfinite(best.aic):
        best = ArimaSpec(order=(1, 0, 0), aic=float("inf"))
    return best


def fit_arima(
    series: np.ndarray,
    order: tuple[int, int, int],
    *,
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> ArimaFitResult:
    """Fit ARIMA baseline.

    References:
      - Box, G. E. P. & Jenkins, G. M. (1970) "Time Series Analysis: Forecasting and Control",
        Holden-Day.
    """
    _require_statsmodels()
    y = np.asarray(series, dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        res = ARIMA(
            y,
            order=order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        ).fit()
    return ArimaFitResult(order=order, aic=float(res.aic), model_result=res)


def fit_sarima(
    series: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    *,
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> SarimaFitResult:
    """Fit SARIMA (seasonal ARIMA) baseline.

    References:
      - Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015)
        "Time Series Analysis: Forecasting and Control", Wiley. DOI: 10.1002/9781118619193
    """
    _require_statsmodels_sarima()
    y = np.asarray(series, dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        res = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        ).fit(disp=False)
    return SarimaFitResult(
        order=order,
        seasonal_order=seasonal_order,
        aic=float(res.aic),
        model_result=res,
    )


def rolling_forecast(
    series: np.ndarray,
    train_len: int,
    steps_ahead: int,
    *,
    order: tuple[int, int, int] | None = None,
    auto: bool = True,
    max_p: int = 2,
    max_d: int = 1,
    max_q: int = 2,
) -> tuple[np.ndarray, ArimaFitResult]:
    """Rolling multi-step forecasts using ARIMA fitted on training slice.

    Notes:
    - For each origin time t, forecast steps_ahead using data up to t.
    """
    _require_statsmodels()
    if steps_ahead < 1:
        raise ValueError("steps_ahead must be >= 1")
    y = np.asarray(series, dtype=float)
    if train_len < 10:
        raise ValueError("train_len too small for ARIMA baseline")
    y_train = y[:train_len]
    if order is None and auto:
        spec = auto_arima_order(y_train, max_p, max_d, max_q)
        order = spec.order
    if order is None:
        order = (1, 0, 0)
    fit = fit_arima(y_train, order)

    preds = np.full_like(y, fill_value=np.nan, dtype=float)
    res = fit.model_result
    last_idx = len(y) - steps_ahead
    for t_idx in range(train_len - 1, last_idx):
        if t_idx >= train_len:
            # update state with actual observation at t_idx (no refit)
            try:
                res = res.append([y[t_idx]], refit=False)
            except Exception:
                # fallback: re-fit on expanding window if append is unavailable
                res = fit_arima(y[: t_idx + 1], order).model_result
        fc = res.forecast(steps=steps_ahead)
        preds[t_idx + steps_ahead] = float(fc[-1])
    return preds, fit


def rolling_forecast_multi(
    series: np.ndarray,
    train_len: int,
    max_steps: int,
    *,
    order: tuple[int, int, int] | None = None,
    auto: bool = True,
    max_p: int = 2,
    max_d: int = 1,
    max_q: int = 2,
) -> tuple[np.ndarray, ArimaFitResult]:
    """Rolling multi-step forecasts (all steps) using ARIMA fitted on training slice."""
    _require_statsmodels()
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    y = np.asarray(series, dtype=float)
    if train_len < 10:
        raise ValueError("train_len too small for ARIMA baseline")
    y_train = y[:train_len]
    if order is None and auto:
        spec = auto_arima_order(y_train, max_p, max_d, max_q)
        order = spec.order
    if order is None:
        order = (1, 0, 0)
    fit = fit_arima(y_train, order)

    preds = np.full((len(y), max_steps), fill_value=np.nan, dtype=float)
    res = fit.model_result
    last_idx = len(y) - max_steps
    for t_idx in range(train_len - 1, last_idx):
        if t_idx >= train_len:
            try:
                res = res.append([y[t_idx]], refit=False)
            except Exception:
                res = fit_arima(y[: t_idx + 1], order).model_result
        fc = res.forecast(steps=max_steps)
        for k in range(max_steps):
            preds[t_idx + k + 1, k] = float(fc[k])
    return preds, fit


def rolling_forecast_multi_sarima(
    series: np.ndarray,
    train_len: int,
    max_steps: int,
    *,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    stride: int = 1,
) -> tuple[np.ndarray, SarimaFitResult]:
    """Rolling multi-step forecasts using SARIMA fitted on training slice."""
    _require_statsmodels_sarima()
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    y = np.asarray(series, dtype=float)
    if train_len < 10:
        raise ValueError("train_len too small for SARIMA baseline")
    fit = fit_sarima(y[:train_len], order, seasonal_order)

    preds = np.full((len(y), max_steps), fill_value=np.nan, dtype=float)
    res = fit.model_result
    last_idx = len(y) - max_steps
    for t_idx in range(train_len - 1, last_idx, stride):
        if t_idx >= train_len:
            try:
                res = res.append([y[t_idx]], refit=False)
            except Exception:
                res = fit_sarima(y[: t_idx + 1], order, seasonal_order).model_result
        fc = res.forecast(steps=max_steps)
        for k in range(max_steps):
            preds[t_idx + k + 1, k] = float(fc[k])
    return preds, fit


def sarima_one_step_predictions(
    series: np.ndarray,
    train_len: int,
    *,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> tuple[np.ndarray, SarimaFitResult]:
    """One-step-ahead SARIMA predictions after an initial training window."""
    _require_statsmodels_sarima()
    y = np.asarray(series, dtype=float)
    if train_len < 10:
        raise ValueError("train_len too small for SARIMA baseline")
    fit = fit_sarima(y[:train_len], order, seasonal_order)
    res = fit.model_result
    preds = np.full_like(y, fill_value=np.nan, dtype=float)
    try:
        pred_res = res.get_prediction(start=train_len, end=len(y) - 1, dynamic=False)
        preds[train_len:] = np.asarray(pred_res.predicted_mean, dtype=float)
    except Exception:
        # fallback: simple rolling one-step forecasts (slower)
        last_idx = len(y) - 1
        for t_idx in range(train_len - 1, last_idx):
            if t_idx >= train_len:
                try:
                    res = res.append([y[t_idx]], refit=False)
                except Exception:
                    res = fit_sarima(y[: t_idx + 1], order, seasonal_order).model_result
            fc = res.forecast(steps=1)
            preds[t_idx + 1] = float(fc[0])
    return preds, fit
