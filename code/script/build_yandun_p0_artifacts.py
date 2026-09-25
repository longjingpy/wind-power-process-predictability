"""Build reviewer-facing Yandun P0 artifacts from registered JSON/CSV inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MODELS = ("Temporal-Only", "GNN without GNSS", "LSTM+GNSS-GNN")
CONTRASTS = (
    ("GNN without GNSS", "Temporal-Only"),
    ("LSTM+GNSS-GNN", "GNN without GNSS"),
    ("LSTM+GNSS-GNN", "Temporal-Only"),
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_row(model: str, seed: int, metrics: dict[str, Any], capacity_kw: float) -> dict[str, Any]:
    overall = metrics["overall"]
    return {
        "model": model,
        "seed": seed,
        "mae_kw": float(overall["mae"]),
        "rmse_kw": float(overall["rmse"]),
        "nmae_pct": float(overall["mae"]) / capacity_kw * 100.0,
        "nrmse_pct": float(overall["rmse"]) / capacity_kw * 100.0,
    }


def _lead_rows(model: str, seed: int, metrics: dict[str, Any], capacity_kw: float) -> list[dict[str, Any]]:
    rows = []
    for item in metrics["step_metrics"]:
        values = item["metrics"]
        rows.append(
            {
                "model": model,
                "seed": seed,
                "lead_min": int(item["step_index"]) * 15,
                "mae_kw": float(values["mae"]),
                "rmse_kw": float(values["rmse"]),
                "nmae_pct": float(values["mae"]) / capacity_kw * 100.0,
                "nrmse_pct": float(values["rmse"]) / capacity_kw * 100.0,
            }
        )
    return rows


def _load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["origin_time", "datetime"])
    required = {"origin_time", "step_index", "target_power", "pred_power", "persistence_power"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} misses prediction columns: {sorted(missing)}")
    frame = frame.sort_values(["origin_time", "step_index"]).reset_index(drop=True)
    if frame.duplicated(["origin_time", "step_index"]).any():
        raise ValueError(f"Duplicate origin-step keys in {path}")
    return frame


def _baseline_rows(
    records: list[dict[str, Any]], capacity_kw: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate: list[dict[str, Any]] = []
    leadwise: list[dict[str, Any]] = []
    temporal = [record for record in records if record["model"] == "Temporal-Only"]
    for record in temporal:
        seed = int(record["seed"])
        metrics = record["metrics_data"]
        prediction = record["prediction_data"]
        target = prediction["target_power"].to_numpy(float)
        persistence = prediction["persistence_power"].to_numpy(float)
        error = persistence - target
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
        aggregate.append(
            {"model": "Persistence", "seed": seed, "mae_kw": mae, "rmse_kw": rmse,
             "nmae_pct": mae / capacity_kw * 100.0, "nrmse_pct": rmse / capacity_kw * 100.0}
        )
        for step, subset in prediction.groupby("step_index", sort=True):
            step_error = subset["persistence_power"].to_numpy(float) - subset["target_power"].to_numpy(float)
            step_mae = float(np.mean(np.abs(step_error)))
            step_rmse = float(np.sqrt(np.mean(np.square(step_error))))
            leadwise.append(
                {"model": "Persistence", "seed": seed, "lead_min": (int(step) + 1) * 15,
                 "mae_kw": step_mae, "rmse_kw": step_rmse,
                 "nmae_pct": step_mae / capacity_kw * 100.0,
                 "nrmse_pct": step_rmse / capacity_kw * 100.0}
            )
        baseline = metrics["baseline"]["overall"]
        aggregate.append(
            {"model": "ARX", "seed": seed, "mae_kw": float(baseline["mae"]),
             "rmse_kw": float(baseline["rmse"]),
             "nmae_pct": float(baseline["mae"]) / capacity_kw * 100.0,
             "nrmse_pct": float(baseline["rmse"]) / capacity_kw * 100.0}
        )
        for item in metrics["step_metrics"]:
            values = item["baseline"]
            leadwise.append(
                {"model": "ARX", "seed": seed, "lead_min": int(item["step_index"]) * 15,
                 "mae_kw": float(values["mae"]), "rmse_kw": float(values["rmse"]),
                 "nmae_pct": float(values["mae"]) / capacity_kw * 100.0,
                 "nrmse_pct": float(values["rmse"]) / capacity_kw * 100.0}
            )
    return aggregate, leadwise


def _bootstrap_mean_ci(
    values: np.ndarray, resamples: int, seed: int, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_origins = values.shape[0]
    samples = np.empty((resamples, values.shape[1]), dtype=np.float64)
    probabilities = np.full(n_origins, 1.0 / n_origins)
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        counts = rng.multinomial(n_origins, probabilities, size=stop - start)
        samples[start:stop] = counts @ values / n_origins
    return (
        np.quantile(samples, alpha / 2.0, axis=0),
        np.quantile(samples, 1.0 - alpha / 2.0, axis=0),
    )


def _paired_frames(
    records: list[dict[str, Any]], capacity_kw: float, resamples: int, bootstrap_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    primary = [record for record in records if record["model"] in PRIMARY_MODELS]
    keyed: dict[tuple[str, int], pd.DataFrame] = {
        (record["model"], int(record["seed"])): record["prediction_data"] for record in primary
    }
    seeds = sorted({seed for _, seed in keyed})
    reference = keyed[("Temporal-Only", seeds[0])][["origin_time", "step_index", "target_power"]]
    for key, frame in keyed.items():
        if len(frame) != len(reference):
            raise ValueError(f"Prediction length mismatch for {key}")
        if not frame[["origin_time", "step_index"]].equals(reference[["origin_time", "step_index"]]):
            raise ValueError(f"Origin-step mismatch for {key}")
        if not np.allclose(frame["target_power"], reference["target_power"], atol=1e-3, rtol=0):
            raise ValueError(f"Target mismatch for {key}")

    shape = (reference["origin_time"].nunique(), reference["step_index"].nunique())
    errors: dict[tuple[str, int], np.ndarray] = {}
    for key, frame in keyed.items():
        errors[key] = np.abs(frame["pred_power"].to_numpy(float) - frame["target_power"].to_numpy(float)).reshape(shape)

    combined_rows: list[dict[str, Any]] = []
    seedwise_rows: list[dict[str, Any]] = []
    alpha_lead = 0.05 / shape[1]
    for contrast_index, (model_a, model_b) in enumerate(CONTRASTS):
        seed_diffs = np.stack([errors[(model_a, seed)] - errors[(model_b, seed)] for seed in seeds])
        mean_diff = seed_diffs.mean(axis=0)
        lead_low, lead_high = _bootstrap_mean_ci(
            mean_diff, resamples, bootstrap_seed + contrast_index, alpha_lead
        )
        aggregate_diff = mean_diff.mean(axis=1, keepdims=True)
        agg_low, agg_high = _bootstrap_mean_ci(
            aggregate_diff, resamples, bootstrap_seed + 100 + contrast_index, 0.05
        )
        contrast = f"{model_a} - {model_b}"
        for step in range(shape[1]):
            estimate = float(mean_diff[:, step].mean())
            combined_rows.append(
                {"scope": "lead", "contrast": contrast, "seed": 0, "lead_min": (step + 1) * 15,
                 "estimate_mae_kw": estimate, "ci_low_mae_kw": float(lead_low[step]),
                 "ci_high_mae_kw": float(lead_high[step]),
                 "estimate_nmae_pct": estimate / capacity_kw * 100.0,
                 "ci_low_nmae_pct": float(lead_low[step]) / capacity_kw * 100.0,
                 "ci_high_nmae_pct": float(lead_high[step]) / capacity_kw * 100.0,
                 "interval": "Bonferroni-adjusted 95% family-wise CI", "seed_set": "/".join(map(str, seeds))}
            )
        estimate = float(aggregate_diff.mean())
        combined_rows.append(
            {"scope": "aggregate", "contrast": contrast, "seed": 0, "lead_min": 0,
             "estimate_mae_kw": estimate, "ci_low_mae_kw": float(agg_low[0]),
             "ci_high_mae_kw": float(agg_high[0]),
             "estimate_nmae_pct": estimate / capacity_kw * 100.0,
             "ci_low_nmae_pct": float(agg_low[0]) / capacity_kw * 100.0,
             "ci_high_nmae_pct": float(agg_high[0]) / capacity_kw * 100.0,
             "interval": "95% paired origin bootstrap CI", "seed_set": "/".join(map(str, seeds))}
        )
        for seed_index, seed in enumerate(seeds):
            diff = seed_diffs[seed_index]
            for step in range(shape[1]):
                seedwise_rows.append(
                    {"scope": "lead", "contrast": contrast, "seed": seed, "lead_min": (step + 1) * 15,
                     "estimate_mae_kw": float(diff[:, step].mean()),
                     "estimate_nmae_pct": float(diff[:, step].mean()) / capacity_kw * 100.0}
                )
            seedwise_rows.append(
                {"scope": "aggregate", "contrast": contrast, "seed": seed, "lead_min": 0,
                 "estimate_mae_kw": float(diff.mean()),
                 "estimate_nmae_pct": float(diff.mean()) / capacity_kw * 100.0}
            )
    audit = {
        "common_origins": shape[0], "leads": shape[1], "origin_lead_rows": len(reference),
        "seeds": seeds, "target_identity_verified": True, "origin_step_identity_verified": True,
        "primary_endpoint": "4 h aggregate nMAE", "difference_sign": "model A minus model B; negative favors model A",
        "aggregate_interval": f"paired forecast-origin bootstrap, {resamples} resamples, percentile 95% CI",
        "lead_interval": f"paired forecast-origin bootstrap, {resamples} resamples, Bonferroni FWER 0.05 across 16 leads",
        "bootstrap_seed": bootstrap_seed,
        "seed_variance_policy": "Report model metric mean and SD across seeds separately from origin bootstrap CI."
    }
    return pd.DataFrame(combined_rows), pd.DataFrame(seedwise_rows), audit


def _ablation_frame(records: list[dict[str, Any]], capacity_kw: float) -> pd.DataFrame:
    rows = []
    for record in records:
        if record["model"] != "LSTM+GNSS-GNN":
            continue
        metrics = record["metrics_data"]
        diagnostic = metrics["temporal_vs_final_mae_stepwise"]
        temporal_mae = diagnostic["temporal_mae"]
        final_mae = diagnostic["final_mae"]
        improvement = diagnostic["improvement"]
        gate = metrics.get("gate_mean_stepwise", [0.0] * len(temporal_mae))
        correction_ratio = metrics.get("spatial_delta_abs_ratio_stepwise", [0.0] * len(temporal_mae))
        for step, (base_value, final_value, improvement_value) in enumerate(
            zip(temporal_mae, final_mae, improvement, strict=True)
        ):
            rows.append(
                {"seed": int(record["seed"]), "lead_min": (step + 1) * 15,
                 "temporal_base_mae_kw": float(base_value), "final_mae_kw": float(final_value),
                 "mae_improvement_kw": float(improvement_value),
                 "nmae_improvement_pct_point": float(improvement_value) / capacity_kw * 100.0,
                 "spatial_delta_abs_ratio": float(correction_ratio[step]),
                 "mean_residual_gate": float(gate[step]),
                 "source": "metrics.json:temporal_vs_final_mae_stepwise"}
            )
    return pd.DataFrame(rows)


def _data_contract(config: dict[str, Any], audit: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    datasets: dict[str, Any] = {}
    features: list[dict[str, Any]] = []
    role_map = {
        "farm_feature_columns": "forecast_input", "turbine_feature_columns": "forecast_input",
        "gnss_feature_columns": "forecast_input",
        "gnss_station_feature_columns": "forecast_input", "gnss_station_static_columns": "static_geometry",
        "static_node_columns": "static_geometry", "target_columns": "offline_label",
        "target_flow_u_columns": "offline_label", "target_flow_v_columns": "offline_label",
    }
    for dataset_name, dataset_value in config["datasets"].items():
        root = _path(dataset_value)
        meta = _json(root / "dataset_meta.json")
        index_path = root / "sample_index.parquet"
        index = pd.read_parquet(index_path)
        time_column = next(column for column in ("origin_time", "datetime") if column in index.columns)
        times = pd.to_datetime(index[time_column])
        datasets[dataset_name] = {
            "path": str(root), "sample_index_sha256": _sha256(index_path),
            "candidate_origins": int(len(index)), "candidate_time_start": str(times.min()),
            "candidate_time_end": str(times.max()), "dataset_version": meta.get("dataset_version"),
            "frequency_min": meta.get("freq_min"), "input_steps": meta.get("t_in"),
            "horizon_steps": meta.get("horizon"), "lead_spacing_min": meta.get("step_out_min"),
            "selected_stations": meta.get("gnss_selected_stations", []),
            "station_coverage_ratio": meta.get("gnss_station_coverage_ratio"),
            "gnss_station_policy": meta.get("gnss_station_policy"),
            "gnss_processing_route": meta.get("gnss_processing_route"),
            "gnss_product_source": meta.get("gnss_product_source"),
            "farm_capacity_mw": meta.get("farm_capacity_mw"), "fill_method": meta.get("fill_method"),
            "fill_limit": meta.get("fill_limit"), "raw_scada_pattern": meta.get("raw_scada_pattern"),
        }
        for group, role in role_map.items():
            for order, column in enumerate(meta.get(group, []), start=1):
                features.append(
                    {"dataset": dataset_name, "feature_group": group, "order": order,
                     "column": column, "role": role, "inference_available": role in {"forecast_input", "static_geometry"}}
                )
        for order, column in enumerate(
            [name for name in index.columns if name.startswith(("window_", "is_"))], start=1
        ):
            features.append(
                {"dataset": dataset_name, "feature_group": "sample_index_audit_columns", "order": order,
                 "column": column, "role": "audit_only", "inference_available": False}
            )
    if datasets["scada"]["sample_index_sha256"] != datasets["gnss"]["sample_index_sha256"]:
        raise ValueError("SCADA and GNSS sample-index hashes differ")
    contract = {
        "schema_version": "yandun-data-contract-v1", "status": "frozen_from_actual_artifacts",
        "datasets": datasets, "registered_evaluation": audit,
        "split": {"mode": "window_overlap", "window_days": 40, "validation_days": 10,
                  "window_end": "2024-08-24 00:00:00", "purge_gap_min": 0,
                  "actual_eval_start": "2024-07-09 01:40:00", "actual_eval_end": "2024-08-23 22:55:00"},
        "inference_exclusions": ["ERA5/reanalysis dynamic fields", "target flow labels", "future power targets"],
        "feature_roles_csv": "site_ordered_feature_contract.csv",
    }
    return contract, pd.DataFrame(features)


def _model_contract(records: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    models = []
    for record in records:
        meta = record["meta_data"]
        checkpoint = record["checkpoint_path"]
        run_dir = record["meta_path"].parent
        runtime_path = run_dir / "train_runtime.json"
        runtime = _json(runtime_path) if runtime_path.exists() else {}
        recipe = meta.get("training_recipe") or {
            "epochs": 20, "batch_size": 64, "learning_rate": 1e-4, "optimizer": "adam",
            "weight_decay": 1e-4, "lr_scheduler": "plateau", "source": "scripts/run_yandun_remote_linux.sh"
        }
        models.append(
            {"model": record["model"], "run_name": meta.get("model"), "seed": int(record["seed"]),
             "meta_path": str(record["meta_path"]), "checkpoint_path": str(checkpoint),
             "checkpoint_sha256": _sha256(checkpoint), "training_recipe": recipe,
             "exact_argv": runtime.get("argv"), "runtime": {key: runtime.get(key) for key in
                 ("started_at", "finished_at", "python", "python_version", "platform", "torch", "torch_cuda", "cuda_device_name", "timing_sec")},
             "architecture": {key: meta.get(key) for key in
                 ("arch", "farm_v2_graph_mode", "farm_v2_temporal_readout", "lstm_spatial_branch_mode",
                  "lstm_main_hidden_dim", "lstm_main_layers", "lstm_main_dropout", "power_loss",
                  "power_val_metric", "power_residual_baseline", "gnss_flow_enabled", "gnss_prior_source",
                  "gnss_flow_active_steps", "gnss_prior_confidence_gating")},
             "training_data_boundary": {key: meta.get(key) for key in
                 ("split_mode", "split_window_days", "val_window_days", "split_window_start",
                  "split_window_val_start", "split_window_end", "split_buffer_min",
                  "farm_v2_filtered_sample_count", "power_weight_stats")},
             "loss_and_weighting": {key: meta.get(key) for key in
                 ("power_loss", "farm_v2_high_power_quantile", "farm_v2_high_power_weight",
                  "farm_v2_ramp_main_loss_mult", "farm_v2_ramp_loss_bonus",
                  "farm_v2_gap_fill_loss_discount", "gnss_flow_loss_weight",
                  "gnss_flow_loss_normalized")},
             "normalization_state": {
                 "fit_boundary": "training split only; frozen for validation/evaluation",
                 "farm_feature_mean": meta.get("farm_feature_mean"),
                 "farm_feature_std": meta.get("farm_feature_std"),
                 "turbine_feature_mean": meta.get("turbine_feature_mean"),
                 "turbine_feature_std": meta.get("turbine_feature_std"),
                 "gnss_prior_calibration_state": meta.get("gnss_prior_calibration_state"),
             },
             "selection": {key: meta.get(key) for key in
                 ("best_epoch", "power_val_metric", "power_loss", "best_epoch_rejected_by_shift")},
             "deterministic_policy": {
                 "python_random_seed": int(record["seed"]), "numpy_seed": int(record["seed"]),
                 "torch_seed": int(record["seed"]), "cuda_seed_all": int(record["seed"]),
                 "cudnn_benchmark": False, "cudnn_deterministic": True,
                 "source": "script/train.py:_set_random_seed"
             },
             "input_boundary": {key: meta.get(key, []) for key in
                 ("feature_columns", "turbine_feature_columns", "gnss_feature_columns", "gnss_station_feature_columns")},
             "evaluation": {
                 "common_origins": audit["common_origins"], "leads": audit["leads"],
                 "persistence_fallback_applied": record["metrics_data"].get("persistence_fallback_applied"),
                 "persistence_gate_passed": record["metrics_data"].get("persistence_gate_passed"),
                 "effective_split": record["metrics_data"].get("effective_split"),
                 "eval_window": record["metrics_data"].get("eval_window"),
             }}
        )
    return {"schema_version": "yandun-model-contract-v1", "status": "generated_from_actual_meta_and_checkpoints", "models": models}


def _difference_matrix(contract: dict[str, Any]) -> pd.DataFrame:
    seed42 = {item["model"]: item for item in contract["models"] if item["seed"] == 42}
    rows = []
    attributes = {
        "eligible origins": lambda item: item["evaluation"]["common_origins"],
        "temporal readout": lambda item: item["architecture"]["farm_v2_temporal_readout"],
        "temporal hidden/layers": lambda item: f"{item['architecture']['lstm_main_hidden_dim']}/{item['architecture']['lstm_main_layers']}",
        "graph mode": lambda item: item["architecture"]["farm_v2_graph_mode"],
        "GNSS inference features": lambda item: bool(item["input_boundary"]["gnss_station_feature_columns"]),
        "GNSS prior source": lambda item: item["architecture"]["gnss_prior_source"] if item["architecture"]["gnss_flow_enabled"] else "none",
        "training epochs": lambda item: item["training_recipe"]["epochs"],
        "learning rate": lambda item: item["training_recipe"]["learning_rate"],
        "batch size": lambda item: item["training_recipe"]["batch_size"],
    }
    for attribute, getter in attributes.items():
        values = {model: getter(seed42[model]) for model in PRIMARY_MODELS}
        rows.append({"attribute": attribute, **values, "all_equal": len({str(value) for value in values.values()}) == 1})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _json(args.sources)
    capacity_kw = float(config["capacity_kw"])
    records: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    for source in config["models"]:
        metrics_path, meta_path = _path(source["metrics"]), _path(source["meta"])
        record = dict(source)
        record.update(
            {"metrics_data": _json(metrics_path), "meta_data": _json(meta_path),
             "meta_path": meta_path, "checkpoint_path": _path(source["checkpoint"])}
        )
        if "predictions" in source:
            record["prediction_data"] = _load_prediction(_path(source["predictions"]))
            prediction = record["prediction_data"]
            error = prediction["pred_power"].to_numpy(float) - prediction["target_power"].to_numpy(float)
            exported_mae = float(np.mean(np.abs(error)))
            exported_rmse = float(np.sqrt(np.mean(np.square(error))))
            expected = record["metrics_data"]["overall"]
            if not np.isclose(exported_mae, float(expected["mae"]), atol=1e-2, rtol=0):
                raise ValueError(f"Prediction/metrics MAE mismatch for {source['model']} seed {source['seed']}")
            if not np.isclose(exported_rmse, float(expected["rmse"]), atol=1e-2, rtol=0):
                raise ValueError(f"Prediction/metrics RMSE mismatch for {source['model']} seed {source['seed']}")
        records.append(record)
        aggregate_rows.append(_metric_row(source["model"], int(source["seed"]), record["metrics_data"], capacity_kw))
        lead_rows.extend(_lead_rows(source["model"], int(source["seed"]), record["metrics_data"], capacity_kw))
    baseline_aggregate, baseline_leads = _baseline_rows(records, capacity_kw)
    aggregate_rows.extend(baseline_aggregate)
    lead_rows.extend(baseline_leads)
    paired, paired_seedwise, audit = _paired_frames(
        records, capacity_kw, int(config["bootstrap_resamples"]), int(config["bootstrap_seed"])
    )
    data_contract, features = _data_contract(config, audit)
    development_dataset = data_contract["datasets"].pop("development")
    model_contract = _model_contract(records, audit)
    difference = _difference_matrix(model_contract)
    ablation = _ablation_frame(records, capacity_kw)
    configuration_rows: list[dict[str, Any]] = []
    configuration_sources = []
    for source in config.get("configuration_ablation_models", []):
        metrics = _json(_path(source["metrics"]))
        configuration_sources.append(source)
        aggregate = _metric_row(source["variant"], int(source["seed"]), metrics, capacity_kw)
        configuration_rows.append({"scope": "aggregate", "lead_min": 0, **aggregate})
        configuration_rows.extend(
            {"scope": "lead", **row}
            for row in _lead_rows(source["variant"], int(source["seed"]), metrics, capacity_kw)
        )
    for record in records:
        if record["model"] != "LSTM+GNSS-GNN":
            continue
        variant = "Learned-prior GNSS-GNN (60 epochs, 3e-5)"
        aggregate = _metric_row(variant, int(record["seed"]), record["metrics_data"], capacity_kw)
        configuration_rows.append({"scope": "aggregate", "lead_min": 0, **aggregate})
        configuration_rows.extend(
            {"scope": "lead", **row}
            for row in _lead_rows(variant, int(record["seed"]), record["metrics_data"], capacity_kw)
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(lead_rows).to_csv(args.out_dir / "yandun_leadwise_metrics.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(args.out_dir / "yandun_aggregate_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "yandun_paired_inference.csv", index=False)
    paired_seedwise.to_csv(args.out_dir / "yandun_paired_inference_seedwise.csv", index=False)
    ablation.to_csv(args.out_dir / "yandun_gnss_residual_ablation.csv", index=False)
    pd.DataFrame(configuration_rows).to_csv(args.out_dir / "yandun_gnss_configuration_ablation.csv", index=False)
    features.to_csv(args.out_dir / "site_ordered_feature_contract.csv", index=False)
    difference.to_csv(args.out_dir / "yandun_contract_difference_matrix.csv", index=False)
    (args.out_dir / "yandun_data_contract.json").write_text(
        json.dumps(data_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    development_contract = {
        "schema_version": "development-data-contract-v1",
        "status": data_contract["status"],
        "dataset": development_dataset,
        "feature_roles_csv": data_contract["feature_roles_csv"],
        "inference_exclusions": data_contract["inference_exclusions"],
    }
    development_meta = _json(_path(config["development_reference_meta"]))
    development_contract["registered_model_boundary"] = {
        "eligible_origins": development_meta.get("farm_v2_filtered_sample_count"),
        "split_mode": development_meta.get("split_mode"),
        "split_window_start": development_meta.get("split_window_start"),
        "split_window_val_start": development_meta.get("split_window_val_start"),
        "split_window_end": development_meta.get("split_window_end"),
        "split_buffer_min": development_meta.get("split_buffer_min"),
        "gnss_prior_source": development_meta.get("gnss_prior_source"),
        "gnss_prior_calibration_state": development_meta.get("gnss_prior_calibration_state"),
        "source_meta": config["development_reference_meta"],
    }
    (args.out_dir / "development_data_contract.json").write_text(
        json.dumps(development_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "yandun_model_contract.json").write_text(
        json.dumps(model_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "yandun_paired_inference_method.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    eval_commands = []
    for record in records:
        if "predictions" not in record:
            continue
        dataset = config["datasets"]["gnss" if record["model"] == "LSTM+GNSS-GNN" else "scada"]
        model_dir = record.get("eval_model_dir", str(record["meta_path"].parent))
        out_dir = str(Path(record["predictions"]).parent.parent)
        eval_commands.append(
            "D:/WindPowerForcast/.venv/Scripts/python.exe script/eval.py "
            f"--data {dataset} --model-dir {model_dir} --out-dir {out_dir} --split all "
            "--split-mode window_overlap --split-window-days 40 --val-window-days 10 "
            "--split-window-end 2024-08-24 --split-buffer-min 0 --ignore-training-contract "
            "--all-months --batch-size 256 --device cuda --save-predictions-csv"
        )
    reproducibility_input = {
        "schema_version": "yandun-reproducibility-v2",
        "experiment_id": "yandun_p0_t12x5_actual_20260719",
        "exact_commands": [
            "bash scripts/run_yandun_remote_linux.sh",
            "bash scripts/run_yandun_gnss_gnn_refine_linux.sh",
            *eval_commands,
            "D:/WindPowerForcast/.venv/Scripts/python.exe script/build_yandun_p0_artifacts.py --sources docs/paper/yandun_p0_sources_20260719.json --out-dir outputs/p0/yandun_20260719/artifacts",
            "D:/WindPowerForcast/.venv/Scripts/python.exe script/plot_yandun_p0_results.py --leadwise-csv outputs/p0/yandun_20260719/artifacts/yandun_leadwise_metrics.csv --aggregate-csv outputs/p0/yandun_20260719/artifacts/yandun_aggregate_metrics.csv --paired-csv outputs/p0/yandun_20260719/artifacts/yandun_paired_inference.csv --ablation-csv outputs/p0/yandun_20260719/artifacts/yandun_gnss_residual_ablation.csv --out-dir outputs/p0/yandun_20260719/artifacts/figures"
        ],
        "code_paths": [
            "script/train.py", "script/eval.py", "script/build_yandun_p0_artifacts.py",
            "script/plot_yandun_p0_results.py", "script/freeze_yandun_reproducibility.py",
            "src/farm_v2_dataset.py", "src/farm_v2_runtime.py", "src/gnss_flow.py",
            "src/lstm_spatial_residual_adapter.py", "scripts/run_yandun_remote_linux.sh",
            "scripts/run_yandun_gnss_gnn_refine_linux.sh"
        ],
        "dataset_paths": list(config["datasets"].values()),
        "origin_index_paths": [
            f"{config['datasets']['development']}/sample_index.parquet",
            f"{config['datasets']['scada']}/sample_index.parquet",
            f"{config['datasets']['gnss']}/sample_index.parquet"
        ],
        "log_paths": [
            str(record["meta_path"].parent / name)
            for record in records for name in ("train_runtime.json", "checkpoint_metrics.jsonl")
            if (record["meta_path"].parent / name).exists()
        ] + [
            str(_path(record["predictions"]).parent / name)
            for record in records if "predictions" in record for name in ("metrics.json", "eval_runtime.json")
        ] + ["models/yandun_gnss_proxy_prior_e40_seed42/20260719_111452/checkpoint_metrics.jsonl"],
        "checkpoint_paths": [str(record["checkpoint_path"]) for record in records] + [
            source["checkpoint"] for source in configuration_sources
        ] + [
            "models/yandun_gnss_proxy_prior_e40_seed42/20260719_111452/checkpoint_best.pt"
        ],
        "contract_paths": [
            str(args.sources), config["development_reference_meta"], str(args.out_dir / "yandun_data_contract.json"),
            str(args.out_dir / "development_data_contract.json"),
            str(args.out_dir / "yandun_model_contract.json"),
            str(args.out_dir / "yandun_paired_inference_method.json"),
            "docs/paper/revision_evidence_log.md",
            "docs/paper/yandun_p0_completion_audit_20260719.md",
            "models/yandun_gnss_proxy_prior_e40_seed42/20260719_111452/meta.json",
            "models/yandun_gnss_proxy_prior_e40_seed42/20260719_111452/summary.json"
        ],
        "registered_common_origins": audit["common_origins"],
        "registered_seeds": audit["seeds"],
    }
    (args.out_dir / "yandun_reproducibility_input.json").write_text(
        json.dumps(reproducibility_input, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved Yandun P0 artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
