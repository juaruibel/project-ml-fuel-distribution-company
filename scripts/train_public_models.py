#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.shared import functions as fc
from src.ml.shared.day05_tabular import (  # noqa: E402
    V41_TRANSPORT_COLUMNS,
    build_one_hot_train_test_matrices,
    prepare_day05_model_frame,
    split_day05_by_cutoff,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "public"
CONTRACT_PATH = ARTIFACTS_DIR / "build_contract.json"
TRAINING_SUMMARY_PATH = ARTIFACTS_DIR / "models" / "public_training_summary.json"
MODEL_REGISTRY_PATH = ARTIFACTS_DIR / "models" / "public_model_registry.csv"


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _compare_metrics(actual: dict, expected: dict, atol: float = 1e-6) -> dict:
    checks = []
    status = "PASS"
    for metric_name, expected_value in expected.items():
        actual_value = actual.get(metric_name)
        matches = bool(np.isclose(float(actual_value), float(expected_value), atol=atol, rtol=0.0))
        if not matches:
            status = "FAIL"
        checks.append(
            {
                "metric": metric_name,
                "actual": float(actual_value),
                "expected": float(expected_value),
                "matches": matches,
            }
        )
    return {"status": status, "checks": checks}


def _train_public_baseline(contract: dict) -> dict:
    spec = contract["training"]["baseline"]
    dataset_path = PROJECT_ROOT / spec["dataset_path"]
    dataset_df = pd.read_csv(dataset_path, keep_default_na=False)

    feature_cols_num, feature_cols_cat, target_col = fc.get_feature_columns_v2()
    model_df = dataset_df.copy()
    for column in feature_cols_num:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce").fillna(0.0)
    for column in feature_cols_cat:
        model_df[column] = model_df[column].astype("string").fillna("UNKNOWN").str.strip().replace("", "UNKNOWN")
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce").fillna(0).astype(int)
    model_df["fecha_evento"] = pd.to_datetime(model_df["fecha_evento"], errors="coerce")
    model_df = model_df.dropna(subset=["fecha_evento"]).reset_index(drop=True)

    cutoff_date = spec["cutoff_date"]
    train_df = model_df[model_df["fecha_evento"] <= pd.to_datetime(cutoff_date)].copy()
    test_df = model_df[model_df["fecha_evento"] > pd.to_datetime(cutoff_date)].copy()
    X_train, X_test, y_train, y_test = fc.dummificar_train_test(
        train=train_df,
        test=test_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    variant_map = {name: (model, sampler) for name, model, sampler in fc.build_default_lr_balance_variants()}
    baseline_model, baseline_sampler = variant_map["LR_smote_0.5"]
    baseline_row, baseline_result = fc.evaluate_balance_variant(
        name="LR_smote_0.5",
        model=clone(baseline_model),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_df=train_df,
        test_df=test_df,
        sampler=clone(baseline_sampler),
    )

    metrics = {
        "accuracy": float(baseline_row["acc"]),
        "balanced_accuracy": float(baseline_row["bal_acc"]),
        "f1_pos": float(baseline_row["f1_pos"]),
        "top1_hit": float(baseline_row["top1_hit"]),
        "top2_hit": float(baseline_row["top2_hit"]),
        "coverage": 1.0,
        "test_events": int(test_df["event_id"].nunique()),
    }
    metadata_metrics = {
        "top1_hit": metrics["top1_hit"],
        "top2_hit": metrics["top2_hit"],
        "test_acc": metrics["accuracy"],
        "test_bal_acc": metrics["balanced_accuracy"],
        "test_f1_pos": metrics["f1_pos"],
        "cv_bal_acc": None,
    }
    model_path, metadata_path, _ = fc.save_champion_artifacts(
        model=baseline_result["model"],
        model_dir=PROJECT_ROOT / "models" / "public" / "baseline",
        model_name=spec["public_model_name"],
        metrics=metadata_metrics,
        cutoff_date=cutoff_date,
        dataset_name=dataset_path.name,
        feature_columns=list(X_train.columns),
        selection_rule=spec["selection_rule"],
    )
    return {
        "role": "baseline",
        "model_variant": spec["reference_variant"],
        "public_model_name": spec["public_model_name"],
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "cutoff_date": cutoff_date,
        "metrics": metrics,
        "expected_metrics": spec["expected_metrics"],
        "parity": _compare_metrics(metrics, spec["expected_metrics"]),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
    }


def _train_public_champion(contract: dict) -> dict:
    spec = contract["training"]["champion_pure"]
    dataset_path = PROJECT_ROOT / spec["dataset_path"]
    dataset_df = pd.read_csv(dataset_path, keep_default_na=False)

    feature_cols_num, feature_cols_cat, target_col = fc.get_feature_columns_v2()
    feature_cols_num = list(feature_cols_num) + list(V41_TRANSPORT_COLUMNS)
    model_df = prepare_day05_model_frame(
        dataset_df=dataset_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )
    cutoff_date = spec["cutoff_date"]
    train_df, test_df = split_day05_by_cutoff(model_df, cutoff_date=cutoff_date)
    X_train, X_test, y_train, y_test = build_one_hot_train_test_matrices(
        train_df=train_df,
        test_df=test_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )

    model = LGBMClassifier(**spec["params"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    eval_df = test_df[["event_id", target_col]].copy()
    eval_df["score_model"] = y_score
    eval_df[target_col] = eval_df[target_col].astype(int)
    metrics = {
        "accuracy": float(fc.compute_row_metrics(y_test, y_pred)["accuracy"]),
        "balanced_accuracy": float(fc.compute_row_metrics(y_test, y_pred)["balanced_accuracy"]),
        "f1_pos": float(fc.compute_row_metrics(y_test, y_pred)["f1_pos"]),
        "top1_hit": float(fc.topk_hit_by_event(eval_df.rename(columns={target_col: "target_elegido"}), "score_model", k=1)),
        "top2_hit": float(fc.topk_hit_by_event(eval_df.rename(columns={target_col: "target_elegido"}), "score_model", k=2)),
        "coverage": 1.0,
        "test_events": int(test_df["event_id"].nunique()),
    }
    model_path, metadata_path, _ = fc.save_champion_artifacts(
        model=model,
        model_dir=PROJECT_ROOT / "models" / "public" / "champion_pure",
        model_name=spec["public_model_name"],
        metrics=metrics,
        cutoff_date=cutoff_date,
        dataset_name=dataset_path.name,
        feature_columns=list(X_train.columns),
        selection_rule=spec["selection_rule"],
    )
    return {
        "role": "champion_pure",
        "model_variant": spec["reference_variant"],
        "public_model_name": spec["public_model_name"],
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "cutoff_date": cutoff_date,
        "metrics": metrics,
        "expected_metrics": spec["expected_metrics"],
        "parity": _compare_metrics(metrics, spec["expected_metrics"]),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    contract = _load_contract()
    (ARTIFACTS_DIR / "models").mkdir(parents=True, exist_ok=True)

    baseline = _train_public_baseline(contract)
    champion = _train_public_champion(contract)

    summary = {
        "status": "PASS" if baseline["parity"]["status"] == "PASS" and champion["parity"]["status"] == "PASS" else "FAIL",
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "models": [baseline, champion],
    }
    TRAINING_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    registry_rows = []
    for payload in [baseline, champion]:
        row = {
            "role": payload["role"],
            "reference_variant": payload["model_variant"],
            "public_model_name": payload["public_model_name"],
            "dataset_path": payload["dataset_path"],
            "cutoff_date": payload["cutoff_date"],
            "model_path": payload["model_path"],
            "metadata_path": payload["metadata_path"],
            "parity_status": payload["parity"]["status"],
        }
        row.update(payload["metrics"])
        registry_rows.append(row)
    pd.DataFrame(registry_rows).to_csv(MODEL_REGISTRY_PATH, index=False, encoding="utf-8")

    if summary["status"] != "PASS":
        raise SystemExit("Public model training parity failed. Review artifacts/public/models/public_training_summary.json")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
