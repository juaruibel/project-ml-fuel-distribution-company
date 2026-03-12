#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from src.ml.metrics import registry
    from src.ml.metrics.postinference_metrics import compute_postinference_metrics
    from src.ml.rules import engine as rengine
    from src.ml.shared import functions as fc
    from src.ml.shared.day05_tabular import (
        build_catboost_train_test_matrices,
        build_one_hot_train_test_matrices,
        build_temporal_event_folds,
        get_day05_dataset_catalog,
        prepare_day05_model_frame,
        split_day05_by_cutoff,
    )
    from src.ml.shared.helpers import build_postinference_audit_paths, build_run_id, utc_now_iso, write_json
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.ml.metrics import registry
    from src.ml.metrics.postinference_metrics import compute_postinference_metrics
    from src.ml.rules import engine as rengine
    from src.ml.shared import functions as fc
    from src.ml.shared.day05_tabular import (
        build_catboost_train_test_matrices,
        build_one_hot_train_test_matrices,
        build_temporal_event_folds,
        get_day05_dataset_catalog,
        prepare_day05_model_frame,
        split_day05_by_cutoff,
    )
    from src.ml.shared.helpers import build_postinference_audit_paths, build_run_id, utc_now_iso, write_json

MODEL_FAMILIES = ("CATBOOST", "LIGHTGBM", "XGBOOST")
BALANCE_TAGS = (
    "AUTO_CLASS_WEIGHTS_BALANCED",
    "CLASS_WEIGHT_BALANCED",
    "SCALE_POS_WEIGHT",
    "ROS",
    "SMOTE",
)
METRIC_SORT_COLUMNS = [
    "delta_top2_vs_baseline",
    "delta_bal_acc_vs_baseline",
    "top2_hit",
    "balanced_accuracy",
    "f1_pos",
]


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 05 tabular experimentation runner."""
    parser = argparse.ArgumentParser(
        description="Orquesta Day 05: tabulares puros sobre bases post-fix con registry limpio y search logs separados."
    )
    parser.add_argument(
        "--phase",
        choices=["smoke-imports", "phase1", "all"],
        required=True,
        help="Fase a ejecutar: smoke-imports, phase1 o all.",
    )
    parser.add_argument(
        "--dataset-aliases",
        nargs="*",
        default=[],
        help="Aliases Day 05 a ejecutar. Si vacío, se usan todos los datasets core.",
    )
    parser.add_argument(
        "--model-families",
        nargs="*",
        default=[],
        help="Familias de modelo a ejecutar. Si vacío, se usan CATBOOST/LIGHTGBM/XGBOOST.",
    )
    parser.add_argument(
        "--baseline-metadata-path",
        type=Path,
        default=Path("models/public/baseline/metadata.json"),
        help="Metadata del baseline oficial para resolver cutoff y contrato.",
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("artifacts/public/metrics/final_baseline_vs_candidates.csv"),
        help="Registry oficial baseline vs candidatos.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports"),
        help="Directorio raíz de reportes.",
    )
    parser.add_argument(
        "--candidate-model-dir",
        type=Path,
        default=Path("models/candidates/day05_tabular"),
        help="Directorio raíz de artefactos Day 05.",
    )
    parser.add_argument(
        "--rules-csv",
        type=Path,
        default=Path("config/business_blocklist_rules.csv"),
        help="CSV de reglas de negocio para la comparativa secundaria con policy.",
    )
    parser.add_argument(
        "--albaran-policy",
        type=str,
        default="PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009",
        choices=["PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009"],
        help="Policy Day 03 para la comparativa secundaria.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k operativo para scoring Day 05.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Número de folds temporales expansivos sobre eventos para Fase 2.",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="No añadir filas al registry oficial. Útil para smoke checks.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id opcional para reproducibilidad externa.",
    )
    return parser.parse_args()


# SECTION: Import checks
def resolve_import_status() -> dict[str, dict[str, str]]:
    """Return import status for the external tabular dependencies required by Day 05."""
    statuses: dict[str, dict[str, str]] = {}
    for module_name in ["catboost", "lightgbm", "xgboost"]:
        try:
            importlib.import_module(module_name)
            statuses[module_name] = {"status": "ok", "detail": ""}
        except Exception as exc:
            statuses[module_name] = {"status": "fail", "detail": f"{type(exc).__name__}: {exc}"}
    return statuses


# SECTION: Import checks
def smoke_imports() -> int:
    """Print dependency import status and return a shell-compatible exit code."""
    statuses = resolve_import_status()
    failed = False
    for module_name, payload in statuses.items():
        line = f"{module_name}: {payload['status'].upper()}"
        if payload["detail"]:
            line = f"{line}: {payload['detail']}"
        print(line)
        failed = failed or payload["status"] != "ok"
    return 1 if failed else 0


# SECTION: Path helpers
def resolve_project_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve all repo-root-relative paths used by the Day 05 runner."""
    project_root = Path(__file__).resolve().parents[3]
    baseline_metadata_path = (project_root / args.baseline_metadata_path).resolve()
    registry_csv = (project_root / args.registry_csv).resolve()
    reports_root = (project_root / args.reports_root).resolve()
    candidate_model_dir = (project_root / args.candidate_model_dir).resolve()
    rules_csv = (project_root / args.rules_csv).resolve()
    return {
        "project_root": project_root,
        "baseline_metadata_path": baseline_metadata_path,
        "registry_csv": registry_csv,
        "reports_root": reports_root,
        "candidate_model_dir": candidate_model_dir,
        "rules_csv": rules_csv,
    }


# SECTION: Path helpers
def build_day05_output_paths(reports_root: Path, run_id: str) -> dict[str, Path]:
    """Build the Day 05 summary and search-log paths for one run id."""
    run_date = run_id[:8]
    metrics_root = reports_root / "metrics"
    day05_root = metrics_root / "day05"
    search_root = metrics_root / "search" / run_date
    candidates_root = metrics_root / "candidates" / run_date
    return {
        "day05_root": day05_root,
        "search_root": search_root,
        "candidates_root": candidates_root,
        "canonical_csv": day05_root / f"{run_id}_canonical_candidates.csv",
        "canonical_json": day05_root / f"{run_id}_canonical_candidates.json",
        "selection_json": day05_root / f"{run_id}_selection_decisions.json",
        "run_summary_json": day05_root / f"{run_id}_run_summary.json",
        "phase2_trials_csv": search_root / f"{run_id}_phase2_trials.csv",
        "phase2_trials_json": search_root / f"{run_id}_phase2_trials.json",
        "phase3_trials_csv": search_root / f"{run_id}_phase3_trials.csv",
        "phase3_trials_json": search_root / f"{run_id}_phase3_trials.json",
        "policy_summary_json": day05_root / f"{run_id}_policy_summary.json",
    }


# SECTION: Metadata helpers
def load_json(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


# SECTION: Metadata helpers
def resolve_cutoff_date(metadata_path: Path) -> str:
    """Resolve the fixed official cutoff date from the baseline metadata."""
    metadata = load_json(metadata_path)
    cutoff_date = str(metadata.get("cutoff_date", "")).strip()
    if cutoff_date == "":
        raise ValueError(f"No se pudo resolver cutoff_date desde {metadata_path}")
    return cutoff_date


# SECTION: Metadata helpers
def slugify_variant(model_variant: str) -> str:
    """Build a filesystem-safe slug from a canonical model variant name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", model_variant).strip("_").lower()


# SECTION: Registry helpers
def load_registry_reference_rows(registry_csv: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the baseline row and the LR-equivalent reference rows from the official registry."""
    rows = registry.read_registry_rows(registry_csv)
    baseline_row = next((row for row in rows if row.get("model_role") == "baseline"), None)
    if baseline_row is None:
        raise ValueError(f"No existe baseline en el registry oficial: {registry_csv}")

    lr_variants = {
        "LR_smote_0.5",
        "V2_TRANSPORT_ONLY_LR_smote_0.5_v1",
        "V2_DISPERSION_LR_smote_0.5_v1",
        "V2_COMPETITION_LR_smote_0.5_v1",
        "V2_TRANSPORT_CARRY30D_ONLY_LR_smote_0.5_v1",
    }
    lr_reference_rows = {
        row["model_variant"]: row
        for row in rows
        if row.get("model_variant") in lr_variants and row.get("model_role") in {"baseline", "candidate"}
    }
    return baseline_row, lr_reference_rows


# SECTION: Registry helpers
def extract_reference_metrics(row: dict[str, Any]) -> dict[str, float]:
    """Extract the official metrics used for comparisons from one registry row."""
    return {
        "top1_hit": float(row["top1_hit"]),
        "top2_hit": float(row["top2_hit"]),
        "balanced_accuracy": float(row["balanced_accuracy"]),
        "accuracy": float(row["accuracy"]),
        "f1_pos": float(row["f1_pos"]),
        "coverage": float(row["coverage"]),
        "test_events": float(row["test_events"]),
    }


# SECTION: Model factories
def build_base_model(model_family: str, params: dict[str, Any], cat_feature_names: list[str] | None = None):
    """Build one unfitted estimator for the requested Day 05 model family."""
    family = model_family.upper()
    if family == "CATBOOST":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params["l2_leaf_reg"],
            loss_function="Logloss",
            random_seed=params["random_state"],
            allow_writing_files=False,
            verbose=False,
            cat_features=cat_feature_names or [],
            auto_class_weights=params.get("auto_class_weights"),
        )
    if family == "LIGHTGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            class_weight=params.get("class_weight"),
            n_jobs=-1,
            random_state=params["random_state"],
        )
    if family == "XGBOOST":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            scale_pos_weight=params.get("scale_pos_weight", 1.0),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=params["random_state"],
        )
    raise ValueError(f"Familia de modelo no soportada en Day 05: {model_family}")


# SECTION: Model factories
def get_phase1_default_params(model_family: str) -> dict[str, Any]:
    """Return the fixed defendable Day 05 base parameters for one model family."""
    family = model_family.upper()
    if family == "CATBOOST":
        return {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3,
            "random_state": 0,
        }
    if family == "LIGHTGBM":
        return {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 0,
        }
    if family == "XGBOOST":
        return {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "random_state": 0,
        }
    raise ValueError(f"Familia de modelo no soportada en Day 05: {model_family}")


# SECTION: Search grids
def get_phase2_trial_params(model_family: str) -> list[dict[str, Any]]:
    """Return the bounded Day 05 tuning grid with at most eight trials per family."""
    family = model_family.upper()
    if family == "CATBOOST":
        return [
            {"iterations": 300, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 3, "random_state": 0},
            {"iterations": 300, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3, "random_state": 0},
            {"iterations": 300, "depth": 8, "learning_rate": 0.10, "l2_leaf_reg": 5, "random_state": 0},
            {"iterations": 600, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5, "random_state": 0},
            {"iterations": 600, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5, "random_state": 0},
            {"iterations": 600, "depth": 8, "learning_rate": 0.10, "l2_leaf_reg": 7, "random_state": 0},
            {"iterations": 600, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7, "random_state": 0},
            {"iterations": 300, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 7, "random_state": 0},
        ]
    if family == "LIGHTGBM":
        return [
            {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 0},
            {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0, "random_state": 0},
            {"n_estimators": 300, "learning_rate": 0.10, "num_leaves": 63, "max_depth": 10, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 1.0, "random_state": 0},
            {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1, "min_child_samples": 50, "subsample": 1.0, "colsample_bytree": 0.8, "random_state": 0},
            {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 1.0, "random_state": 0},
            {"n_estimators": 600, "learning_rate": 0.10, "num_leaves": 63, "max_depth": 10, "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 0.8, "random_state": 0},
            {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "max_depth": -1, "min_child_samples": 50, "subsample": 1.0, "colsample_bytree": 1.0, "random_state": 0},
            {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 10, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 0},
        ]
    if family == "XGBOOST":
        return [
            {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "random_state": 0},
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0, "random_state": 0},
            {"n_estimators": 300, "learning_rate": 0.10, "max_depth": 8, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 1.0, "reg_lambda": 5.0, "random_state": 0},
            {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 5, "subsample": 1.0, "colsample_bytree": 0.8, "reg_lambda": 1.0, "random_state": 0},
            {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 1.0, "reg_lambda": 5.0, "random_state": 0},
            {"n_estimators": 600, "learning_rate": 0.10, "max_depth": 8, "min_child_weight": 5, "subsample": 1.0, "colsample_bytree": 0.8, "reg_lambda": 1.0, "random_state": 0},
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 5.0, "random_state": 0},
            {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0, "random_state": 0},
        ]
    raise ValueError(f"Familia de modelo no soportada en Day 05: {model_family}")


# SECTION: Search grids
def get_phase3_trial_specs(
    model_family: str,
    base_params: dict[str, Any],
    class_ratio: float,
) -> list[dict[str, Any]]:
    """Return the bounded balance-variant candidates for Day 05 Phase 3."""
    family = model_family.upper()
    if family == "CATBOOST":
        return [
            {"balance_tag": "AUTO_CLASS_WEIGHTS_BALANCED", "params": {**base_params, "auto_class_weights": "Balanced"}, "resampler": None},
            {"balance_tag": "ROS", "params": dict(base_params), "resampler": "ros"},
        ]
    if family == "LIGHTGBM":
        return [
            {"balance_tag": "CLASS_WEIGHT_BALANCED", "params": {**base_params, "class_weight": "balanced"}, "resampler": None},
            {"balance_tag": "ROS", "params": dict(base_params), "resampler": "ros"},
            {"balance_tag": "SMOTE", "params": dict(base_params), "resampler": "smote"},
        ]
    if family == "XGBOOST":
        return [
            {"balance_tag": "SCALE_POS_WEIGHT", "params": {**base_params, "scale_pos_weight": class_ratio}, "resampler": None},
            {"balance_tag": "ROS", "params": dict(base_params), "resampler": "ros"},
            {"balance_tag": "SMOTE", "params": dict(base_params), "resampler": "smote"},
        ]
    raise ValueError(f"Familia de modelo no soportada en Day 05: {model_family}")


# SECTION: Sampling helpers
def build_resampler(resampler_name: str | None):
    """Build one optional resampler for Day 05 balance experiments."""
    if resampler_name is None:
        return None
    if resampler_name == "ros":
        from imblearn.over_sampling import RandomOverSampler

        return RandomOverSampler(random_state=0)
    if resampler_name == "smote":
        from imblearn.over_sampling import SMOTE

        return SMOTE(random_state=0)
    raise ValueError(f"Resampler Day 05 no soportado: {resampler_name}")


# SECTION: Sampling helpers
def resample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    resampler_name: str | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply one optional resampler on train only while preserving column names when possible."""
    if resampler_name is None:
        return X_train, y_train
    sampler = build_resampler(resampler_name)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    if isinstance(X_train, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    if isinstance(y_train, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y_train.name)
    return X_resampled, y_resampled


# SECTION: Feature builders
def build_train_test_inputs(
    model_family: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build train/test inputs according to the model-family-specific feature recipe."""
    family = model_family.upper()
    if family == "CATBOOST":
        return build_catboost_train_test_matrices(
            train_df=train_df,
            test_df=test_df,
            feature_cols_num=feature_cols_num,
            feature_cols_cat=feature_cols_cat,
            target_col=target_col,
        )
    return build_one_hot_train_test_matrices(
        train_df=train_df,
        test_df=test_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )


# SECTION: Evaluation helpers
def fit_and_evaluate_model(
    *,
    model_family: str,
    estimator,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
    resampler_name: str | None = None,
) -> dict[str, Any]:
    """Fit one candidate and return fitted estimator, row metrics, event metrics, and eval frame."""
    X_train, X_test, y_train, y_test = build_train_test_inputs(
        model_family=model_family,
        train_df=train_df,
        test_df=test_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )
    X_train_fit, y_train_fit = resample_training_data(X_train, y_train, resampler_name=resampler_name)
    model = estimator
    model.fit(X_train_fit, y_train_fit)
    scored_payload = score_fitted_model(
        model_family=model_family,
        model=model,
        train_df=train_df,
        test_df=test_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )
    return {
        "model": model,
        "row_metrics": scored_payload["row_metrics"],
        "event_metrics": scored_payload["event_metrics"],
        "eval_frame": scored_payload["eval_frame"],
        "feature_columns": scored_payload["feature_columns"],
        "train_pos_rate_after": float(pd.Series(y_train_fit).mean()) if len(y_train_fit) else 0.0,
    }


# SECTION: Evaluation helpers
def score_fitted_model(
    *,
    model_family: str,
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
) -> dict[str, Any]:
    """Score one already-fitted model on the fixed holdout using the family-specific feature contract."""
    _, X_test, _, y_test = build_train_test_inputs(
        model_family=model_family,
        train_df=train_df,
        test_df=test_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )

    y_pred = np.asarray(model.predict(X_test)).astype(int)
    if hasattr(model, "predict_proba"):
        y_score = np.asarray(model.predict_proba(X_test))[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = np.asarray(model.decision_function(X_test))
        y_score = raw_scores[:, 1] if raw_scores.ndim > 1 else raw_scores
    else:
        y_score = y_pred.astype(float)

    row_metrics = fc.compute_row_metrics(y_test, y_pred)
    eval_frame = fc.build_eval_frame(test_df, y_test, y_pred, y_score)
    coverage = float(eval_frame["event_id"].nunique() / test_df["event_id"].nunique()) if not test_df.empty else 0.0
    event_metrics = {
        "top1_hit": float(fc.topk_hit_by_event(eval_frame, "score_model", k=1)),
        "top2_hit": float(fc.topk_hit_by_event(eval_frame, "score_model", k=2)),
        "coverage": coverage,
        "test_events": int(test_df["event_id"].nunique()),
    }
    feature_columns = [str(column) for column in X_test.columns]
    return {
        "row_metrics": row_metrics,
        "event_metrics": event_metrics,
        "eval_frame": eval_frame,
        "feature_columns": feature_columns,
    }


# SECTION: Evaluation helpers
def build_fold_frames(train_df: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project one temporal event fold back to candidate-row train and validation frames."""
    train_events = set(fold["train_event_ids"])
    valid_events = set(fold["valid_event_ids"])
    fold_train = train_df[train_df["event_id"].astype(str).isin(train_events)].copy()
    fold_valid = train_df[train_df["event_id"].astype(str).isin(valid_events)].copy()
    if fold_train.empty or fold_valid.empty:
        raise ValueError(
            "Fold temporal Day 05 vacío tras proyectar a filas. "
            f"fold_idx={fold['fold_idx']} train_rows={len(fold_train)} valid_rows={len(fold_valid)}"
        )
    return fold_train, fold_valid


# SECTION: Evaluation helpers
def evaluate_cv_trial(
    *,
    model_family: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
    n_splits: int,
    resampler_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate one tuning/balance trial with expanding temporal folds over event groups."""
    folds = build_temporal_event_folds(train_df, n_splits=n_splits)
    fold_results: list[dict[str, Any]] = []

    for fold in folds:
        fold_train, fold_valid = build_fold_frames(train_df, fold)
        estimator = build_base_model(
            model_family=model_family,
            params=params,
            cat_feature_names=feature_cols_cat if model_family.upper() == "CATBOOST" else None,
        )
        result = fit_and_evaluate_model(
            model_family=model_family,
            estimator=estimator,
            train_df=fold_train,
            test_df=fold_valid,
            feature_cols_num=feature_cols_num,
            feature_cols_cat=feature_cols_cat,
            target_col=target_col,
            resampler_name=resampler_name,
        )
        fold_results.append(
            {
                "fold_idx": fold["fold_idx"],
                "top2_hit": result["event_metrics"]["top2_hit"],
                "balanced_accuracy": result["row_metrics"]["balanced_accuracy"],
                "f1_pos": result["row_metrics"]["f1_pos"],
                "coverage": result["event_metrics"]["coverage"],
                "train_events": len(fold["train_event_ids"]),
                "valid_events": len(fold["valid_event_ids"]),
            }
        )

    fold_df = pd.DataFrame(fold_results)
    return {
        "folds": fold_results,
        "cv_top2_hit_mean": float(fold_df["top2_hit"].mean()),
        "cv_bal_acc_mean": float(fold_df["balanced_accuracy"].mean()),
        "cv_f1_pos_mean": float(fold_df["f1_pos"].mean()),
        "cv_coverage_mean": float(fold_df["coverage"].mean()),
    }


# SECTION: Variant naming
def build_model_variant(
    dataset_alias: str,
    model_family: str,
    suffix: str = "v1",
) -> str:
    """Build the canonical Day 05 pure variant name from dataset alias and model family."""
    return f"{dataset_alias}_{model_family}_{suffix}"


# SECTION: Variant naming
def build_policy_variant(model_variant: str) -> str:
    """Build the canonical Day 05 policy-comparison variant name from one pure candidate."""
    return f"{model_variant}_WITH_DETERMINISTIC_LAYER_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009_v1"


# SECTION: Metrics helpers
def build_pure_metrics_payload(
    *,
    run_id: str,
    ts_utc: str,
    day_id: str,
    cutoff_date: str,
    model_variant: str,
    dataset_alias: str,
    dataset_path: Path,
    model_family: str,
    variant_stage: str,
    metrics: dict[str, float | int],
    model_path: Path,
    metadata_path: Path,
    search_log_path: Path | None = None,
) -> dict[str, Any]:
    """Build the canonical metrics JSON consumed by the official registry for pure models."""
    payload: dict[str, Any] = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": day_id,
        "scope": "model_retrain",
        "eval_scope": "day02_test_like",
        "cutoff_applied": cutoff_date,
        "model_variant": model_variant,
        "dataset_alias": dataset_alias,
        "model_family": model_family,
        "variant_stage": variant_stage,
        "metrics": metrics,
        "sources": {
            "dataset_csv": str(dataset_path),
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
        },
    }
    if search_log_path is not None:
        payload["sources"]["search_log_path"] = str(search_log_path)
    return payload


# SECTION: Metrics helpers
def compute_gate_flags(metrics: dict[str, float | int], baseline_metrics: dict[str, float]) -> dict[str, Any]:
    """Compute the pure-model registry gate flags locally to mirror the official registry logic."""
    delta_top2 = float(metrics["top2_hit"]) - baseline_metrics["top2_hit"]
    delta_bal = float(metrics["balanced_accuracy"]) - baseline_metrics["balanced_accuracy"]
    delta_cov = float(metrics["coverage"]) - baseline_metrics["coverage"]
    gate_top2_ok = delta_top2 >= registry.GATE_MIN_DELTA_TOP2
    gate_bal_ok = delta_bal >= registry.GATE_MIN_DELTA_BAL_ACC
    gate_cov_ok = delta_cov >= registry.GATE_MIN_DELTA_COVERAGE
    gate_pass = gate_top2_ok and gate_bal_ok and gate_cov_ok
    return {
        "delta_top2_vs_baseline": delta_top2,
        "delta_bal_acc_vs_baseline": delta_bal,
        "delta_coverage_vs_baseline": delta_cov,
        "gate_top2_ok": gate_top2_ok,
        "gate_bal_acc_ok": gate_bal_ok,
        "gate_coverage_ok": gate_cov_ok,
        "gate_pass": gate_pass,
        "promotion_decision": "promote" if gate_pass else "keep_baseline",
    }


# SECTION: Metrics helpers
def compute_lr_deltas(metrics: dict[str, float | int], lr_reference_metrics: dict[str, float]) -> dict[str, float]:
    """Compute deltas against the LR-equivalent reference for the same dataset."""
    return {
        "delta_top2_vs_lr_equivalente": float(metrics["top2_hit"]) - lr_reference_metrics["top2_hit"],
        "delta_bal_acc_vs_lr_equivalente": float(metrics["balanced_accuracy"]) - lr_reference_metrics["balanced_accuracy"],
        "delta_coverage_vs_lr_equivalente": float(metrics["coverage"]) - lr_reference_metrics["coverage"],
    }


# SECTION: Metrics helpers
def compute_phase2_eligibility(
    gate_flags: dict[str, Any],
    baseline_metrics: dict[str, float],
    lr_deltas: dict[str, float],
    lr_reference_metrics: dict[str, float],
    metrics: dict[str, float | int],
) -> dict[str, bool]:
    """Compute primary and secondary Day 05 eligibility flags for Fase 2."""
    primary = (
        (gate_flags["delta_top2_vs_baseline"] > 0 or gate_flags["delta_bal_acc_vs_baseline"] > 0)
        and gate_flags["delta_top2_vs_baseline"] >= -0.005
        and gate_flags["delta_bal_acc_vs_baseline"] >= -0.005
        and float(metrics["coverage"]) >= baseline_metrics["coverage"] - 0.005
    )
    secondary_improves = (
        lr_deltas["delta_top2_vs_lr_equivalente"] >= 0.002
        or lr_deltas["delta_bal_acc_vs_lr_equivalente"] >= 0.002
    )
    secondary_other_metric_ok = (
        lr_deltas["delta_top2_vs_lr_equivalente"] >= -0.003
        and lr_deltas["delta_bal_acc_vs_lr_equivalente"] >= -0.003
    )
    secondary = (
        secondary_improves
        and secondary_other_metric_ok
        and float(metrics["coverage"]) >= lr_reference_metrics["coverage"] - 0.005
    )
    return {
        "eligible_primary_phase2": bool(primary),
        "eligible_secondary_phase2": bool(secondary),
        "eligible_phase2": bool(primary or secondary),
    }


# SECTION: Metrics helpers
def is_policy_eligible(row: pd.Series) -> bool:
    """Return whether one pure Day 05 row is defendible enough for the single policy comparison."""
    return bool(
        row["gate_pass"]
        or (
            row["delta_top2_vs_baseline"] >= -0.005
            and row["delta_bal_acc_vs_baseline"] >= -0.005
            and row["eligible_secondary_phase2"]
        )
    )


# SECTION: Search helpers
def serialize_params(params: dict[str, Any]) -> str:
    """Serialize one params dictionary with stable JSON formatting."""
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


# SECTION: Search helpers
def ranking_tuple(row: pd.Series) -> tuple[float, float, float, float, float]:
    """Return the canonical Day 05 ranking tuple for candidate ordering."""
    return (
        float(row["delta_top2_vs_baseline"]),
        float(row["delta_bal_acc_vs_baseline"]),
        float(row["top2_hit"]),
        float(row["balanced_accuracy"]),
        float(row["f1_pos"]),
    )


# SECTION: Search helpers
def choose_phase2_candidates(canonical_df: pd.DataFrame) -> list[str]:
    """Choose up to two Day 05 candidates for Phase 2 using anti-redundancy by dataset."""
    eligible = canonical_df[canonical_df["eligible_phase2"]].copy()
    if eligible.empty:
        return []

    eligible = eligible.sort_values(METRIC_SORT_COLUMNS, ascending=False).reset_index(drop=True)
    dataset_best = eligible.groupby("dataset_alias", as_index=False).first()
    dataset_best = dataset_best.sort_values(METRIC_SORT_COLUMNS, ascending=False).reset_index(drop=True)

    first = dataset_best.iloc[0]
    selected = [str(first["model_variant"])]
    best_other_dataset = dataset_best.iloc[1] if len(dataset_best) > 1 else None

    same_dataset_candidates = eligible[
        (eligible["dataset_alias"] == first["dataset_alias"])
        & (eligible["model_variant"] != first["model_variant"])
    ].copy()
    same_dataset_second = same_dataset_candidates.iloc[0] if not same_dataset_candidates.empty else None

    if same_dataset_second is not None:
        if best_other_dataset is None:
            selected.append(str(same_dataset_second["model_variant"]))
        else:
            clear_advantage = (
                (
                    float(same_dataset_second["delta_top2_vs_baseline"]) - float(best_other_dataset["delta_top2_vs_baseline"]) >= 0.005
                    and float(same_dataset_second["delta_bal_acc_vs_baseline"]) - float(best_other_dataset["delta_bal_acc_vs_baseline"]) >= -0.002
                )
                or (
                    float(same_dataset_second["delta_bal_acc_vs_baseline"]) - float(best_other_dataset["delta_bal_acc_vs_baseline"]) >= 0.005
                    and float(same_dataset_second["delta_top2_vs_baseline"]) - float(best_other_dataset["delta_top2_vs_baseline"]) >= -0.002
                )
            )
            near_tie = (
                abs(float(same_dataset_second["delta_top2_vs_baseline"]) - float(best_other_dataset["delta_top2_vs_baseline"])) <= 0.002
                and abs(float(same_dataset_second["delta_bal_acc_vs_baseline"]) - float(best_other_dataset["delta_bal_acc_vs_baseline"])) <= 0.002
            )
            if clear_advantage and not near_tie:
                selected.append(str(same_dataset_second["model_variant"]))
            else:
                selected.append(str(best_other_dataset["model_variant"]))
    elif best_other_dataset is not None:
        selected.append(str(best_other_dataset["model_variant"]))

    return selected[:2]


# SECTION: Search helpers
def choose_best_final_pure_variant(canonical_df: pd.DataFrame) -> str:
    """Choose the best final pure Day 05 candidate after all executed phases."""
    pure_df = canonical_df[canonical_df["variant_stage"] != "policy"].copy()
    pure_df["eligible_policy_check"] = pure_df.apply(is_policy_eligible, axis=1)
    pure_df = pure_df.sort_values(
        ["gate_pass", "eligible_primary_phase2", "eligible_secondary_phase2", "top2_hit", "balanced_accuracy", "f1_pos"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    return str(pure_df.iloc[0]["model_variant"])


# SECTION: Artifact helpers
def save_candidate_artifacts(
    *,
    candidate_model_dir: Path,
    run_id: str,
    model_variant: str,
    model,
    metrics: dict[str, Any],
    cutoff_date: str,
    dataset_name: str,
    feature_columns: list[str],
    selection_rule: str,
    extra_metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist one Day 05 candidate model and enrich its metadata with Day 05-specific context."""
    model_dir = candidate_model_dir / slugify_variant(model_variant) / run_id
    model_path, metadata_path, metadata = fc.save_champion_artifacts(
        model=model,
        model_dir=model_dir,
        model_name=model_variant,
        metrics=metrics,
        cutoff_date=cutoff_date,
        dataset_name=dataset_name,
        feature_columns=feature_columns,
        selection_rule=selection_rule,
    )
    metadata.update(extra_metadata)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, metadata_path


# SECTION: Artifact helpers
def write_metrics_payload(
    payload: dict[str, Any],
    candidates_root: Path,
    model_variant: str,
) -> Path:
    """Persist one metrics JSON payload in the official candidates directory."""
    candidates_root.mkdir(parents=True, exist_ok=True)
    output_path = candidates_root / f"{payload['run_id']}_{model_variant}_metrics.json"
    return write_json(output_path, payload)


# SECTION: Artifact helpers
def append_registry_candidate(
    *,
    registry_csv: Path,
    run_id: str,
    day_id: str,
    model_variant: str,
    metadata_path: Path,
    metrics_json_path: Path,
    dataset_csv: Path,
    model_path: Path,
) -> None:
    """Append one canonical Day 05 candidate row to the official registry."""
    args = argparse.Namespace(
        command="append-candidate",
        output=str(registry_csv),
        run_id=run_id,
        day_id=day_id,
        model_variant=model_variant,
        metadata=str(metadata_path),
        metrics_json=str(metrics_json_path),
        dataset=str(dataset_csv),
        coverage=None,
        test_events=None,
        model_path=str(model_path),
        gate_pass="auto",
        promotion_decision="auto",
    )
    registry.append_candidate(args)


# SECTION: Artifact helpers
def write_dataframe_outputs(csv_path: Path, json_path: Path, frame: pd.DataFrame) -> None:
    """Persist one dataframe as both CSV and JSON records for notebook-friendly consumption."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")


# SECTION: Canonical row builders
def build_canonical_row(
    *,
    run_id: str,
    cutoff_date: str,
    dataset_alias: str,
    dataset_path: Path,
    lr_equivalent_variant: str,
    model_family: str,
    model_variant: str,
    variant_stage: str,
    balance_tag: str,
    metrics: dict[str, float | int],
    baseline_metrics: dict[str, float],
    lr_reference_metrics: dict[str, float],
    search_log_path: Path | None,
) -> dict[str, Any]:
    """Build one canonical Day 05 summary row with baseline and LR comparison deltas."""
    gate_flags = compute_gate_flags(metrics, baseline_metrics)
    lr_deltas = compute_lr_deltas(metrics, lr_reference_metrics)
    phase2_flags = compute_phase2_eligibility(
        gate_flags,
        baseline_metrics,
        lr_deltas,
        lr_reference_metrics,
        metrics,
    )
    row = {
        "run_id": run_id,
        "cutoff_date": cutoff_date,
        "dataset_alias": dataset_alias,
        "dataset_path": str(dataset_path),
        "lr_equivalent_variant": lr_equivalent_variant,
        "model_family": model_family,
        "model_variant": model_variant,
        "variant_stage": variant_stage,
        "balance_tag": balance_tag,
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "f1_pos": float(metrics["f1_pos"]),
        "top1_hit": float(metrics["top1_hit"]),
        "top2_hit": float(metrics["top2_hit"]),
        "coverage": float(metrics["coverage"]),
        "test_events": int(metrics["test_events"]),
        "search_log_path": str(search_log_path) if search_log_path is not None else "",
        **gate_flags,
        **lr_deltas,
        **phase2_flags,
    }
    row["eligible_phase3"] = bool(row["eligible_phase2"])
    row["eligible_policy_check"] = bool(is_policy_eligible(pd.Series(row)))
    return row


# SECTION: Phase 1
def run_phase1(
    *,
    run_id: str,
    ts_utc: str,
    cutoff_date: str,
    args: argparse.Namespace,
    paths: dict[str, Path],
    output_paths: dict[str, Path],
    baseline_metrics: dict[str, float],
    lr_reference_rows: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Run the canonical Day 05 Phase 1 matrix and return the canonical summary dataframe."""
    project_root = paths["project_root"]
    dataset_catalog = get_day05_dataset_catalog(project_root=project_root)
    dataset_aliases = args.dataset_aliases or list(dataset_catalog.keys())
    model_families = [family.upper() for family in (args.model_families or MODEL_FAMILIES)]
    canonical_rows: list[dict[str, Any]] = []

    for dataset_alias in dataset_aliases:
        if dataset_alias not in dataset_catalog:
            raise ValueError(f"Dataset alias Day 05 no soportado: {dataset_alias}")
        dataset_spec = dataset_catalog[dataset_alias]
        dataset_df = pd.read_csv(dataset_spec["dataset_path"], keep_default_na=False)
        model_df = prepare_day05_model_frame(
            dataset_df=dataset_df,
            feature_cols_num=dataset_spec["feature_cols_num"],
            feature_cols_cat=dataset_spec["feature_cols_cat"],
            target_col=dataset_spec["target_col"],
        )
        train_df, test_df = split_day05_by_cutoff(model_df, cutoff_date=cutoff_date)
        lr_reference_metrics = extract_reference_metrics(lr_reference_rows[dataset_spec["lr_equivalent_variant"]])

        for model_family in model_families:
            params = get_phase1_default_params(model_family)
            model_variant = build_model_variant(dataset_alias, model_family)
            estimator = build_base_model(
                model_family=model_family,
                params=params,
                cat_feature_names=dataset_spec["feature_cols_cat"] if model_family == "CATBOOST" else None,
            )
            result = fit_and_evaluate_model(
                model_family=model_family,
                estimator=estimator,
                train_df=train_df,
                test_df=test_df,
                feature_cols_num=dataset_spec["feature_cols_num"],
                feature_cols_cat=dataset_spec["feature_cols_cat"],
                target_col=dataset_spec["target_col"],
            )
            metrics = {
                "accuracy": result["row_metrics"]["accuracy"],
                "balanced_accuracy": result["row_metrics"]["balanced_accuracy"],
                "f1_pos": result["row_metrics"]["f1_pos"],
                "top1_hit": result["event_metrics"]["top1_hit"],
                "top2_hit": result["event_metrics"]["top2_hit"],
                "coverage": result["event_metrics"]["coverage"],
                "test_events": result["event_metrics"]["test_events"],
            }
            selection_rule = (
                f"day05_phase1_base({model_family} on {dataset_alias}) -> "
                "registry_gate(top2>=baseline+0.01 & bal_acc>=baseline+0.01 & coverage>=baseline-0.005)"
            )
            model_path, metadata_path = save_candidate_artifacts(
                candidate_model_dir=paths["candidate_model_dir"],
                run_id=run_id,
                model_variant=model_variant,
                model=result["model"],
                metrics=metrics,
                cutoff_date=cutoff_date,
                dataset_name=dataset_spec["dataset_path"].name,
                feature_columns=result["feature_columns"],
                selection_rule=selection_rule,
                extra_metadata={
                    "day_id": "Day 05",
                    "phase": "phase1",
                    "dataset_alias": dataset_alias,
                    "model_family": model_family,
                    "balance_tag": "",
                    "search_log_path": "",
                },
            )
            metrics_payload = build_pure_metrics_payload(
                run_id=run_id,
                ts_utc=ts_utc,
                day_id="Day 05",
                cutoff_date=cutoff_date,
                model_variant=model_variant,
                dataset_alias=dataset_alias,
                dataset_path=dataset_spec["dataset_path"],
                model_family=model_family,
                variant_stage="phase1",
                metrics=metrics,
                model_path=model_path,
                metadata_path=metadata_path,
            )
            metrics_json_path = write_metrics_payload(
                payload=metrics_payload,
                candidates_root=output_paths["candidates_root"],
                model_variant=model_variant,
            )
            if not args.skip_registry:
                append_registry_candidate(
                    registry_csv=paths["registry_csv"],
                    run_id=run_id,
                    day_id="Day 05",
                    model_variant=model_variant,
                    metadata_path=metadata_path,
                    metrics_json_path=metrics_json_path,
                    dataset_csv=dataset_spec["dataset_path"],
                    model_path=model_path,
                )
            canonical_rows.append(
                build_canonical_row(
                    run_id=run_id,
                    cutoff_date=cutoff_date,
                    dataset_alias=dataset_alias,
                    dataset_path=dataset_spec["dataset_path"],
                    lr_equivalent_variant=dataset_spec["lr_equivalent_variant"],
                    model_family=model_family,
                    model_variant=model_variant,
                    variant_stage="phase1",
                    balance_tag="",
                    metrics=metrics,
                    baseline_metrics=baseline_metrics,
                    lr_reference_metrics=lr_reference_metrics,
                    search_log_path=None,
                )
            )

    canonical_df = pd.DataFrame(canonical_rows).sort_values(METRIC_SORT_COLUMNS, ascending=False).reset_index(drop=True)
    phase2_selected = choose_phase2_candidates(canonical_df)
    canonical_df["selected_for_phase2"] = canonical_df["model_variant"].isin(phase2_selected)
    canonical_df["selected_for_phase3"] = False
    write_dataframe_outputs(output_paths["canonical_csv"], output_paths["canonical_json"], canonical_df)
    write_json(
        output_paths["selection_json"],
        {
            "run_id": run_id,
            "phase2_selected_variants": phase2_selected,
            "phase3_selected_variants": [],
            "best_final_pure_variant": choose_best_final_pure_variant(canonical_df),
        },
    )
    return canonical_df


# SECTION: Phase 2
def run_phase2(
    *,
    run_id: str,
    ts_utc: str,
    cutoff_date: str,
    canonical_df: pd.DataFrame,
    args: argparse.Namespace,
    paths: dict[str, Path],
    output_paths: dict[str, Path],
    baseline_metrics: dict[str, float],
    lr_reference_rows: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run bounded tuning for the selected Day 05 candidates and append final tuned rows."""
    project_root = paths["project_root"]
    dataset_catalog = get_day05_dataset_catalog(project_root=project_root)
    selected = canonical_df.loc[canonical_df["selected_for_phase2"], "model_variant"].astype(str).tolist()
    if not selected:
        return canonical_df, []

    phase2_trials: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    for model_variant in selected:
        base_row = canonical_df[canonical_df["model_variant"] == model_variant].iloc[0]
        dataset_alias = str(base_row["dataset_alias"])
        model_family = str(base_row["model_family"])
        dataset_spec = dataset_catalog[dataset_alias]
        dataset_df = pd.read_csv(dataset_spec["dataset_path"], keep_default_na=False)
        model_df = prepare_day05_model_frame(
            dataset_df=dataset_df,
            feature_cols_num=dataset_spec["feature_cols_num"],
            feature_cols_cat=dataset_spec["feature_cols_cat"],
            target_col=dataset_spec["target_col"],
        )
        train_df, test_df = split_day05_by_cutoff(model_df, cutoff_date=cutoff_date)
        lr_reference_metrics = extract_reference_metrics(lr_reference_rows[dataset_spec["lr_equivalent_variant"]])

        trial_candidates = get_phase2_trial_params(model_family)
        best_trial: dict[str, Any] | None = None

        for trial_idx, params in enumerate(trial_candidates, start=1):
            cv_payload = evaluate_cv_trial(
                model_family=model_family,
                params=params,
                train_df=train_df,
                feature_cols_num=dataset_spec["feature_cols_num"],
                feature_cols_cat=dataset_spec["feature_cols_cat"],
                target_col=dataset_spec["target_col"],
                n_splits=args.n_splits,
            )
            trial_row = {
                "run_id": run_id,
                "base_model_variant": model_variant,
                "dataset_alias": dataset_alias,
                "model_family": model_family,
                "trial_idx": trial_idx,
                "params_json": serialize_params(params),
                "cv_top2_hit_mean": cv_payload["cv_top2_hit_mean"],
                "cv_bal_acc_mean": cv_payload["cv_bal_acc_mean"],
                "cv_f1_pos_mean": cv_payload["cv_f1_pos_mean"],
                "cv_coverage_mean": cv_payload["cv_coverage_mean"],
                "folds_json": json.dumps(cv_payload["folds"], ensure_ascii=False),
            }
            phase2_trials.append(trial_row)
            if best_trial is None:
                best_trial = {"params": params, **trial_row}
                continue
            if (
                trial_row["cv_top2_hit_mean"],
                trial_row["cv_bal_acc_mean"],
                trial_row["cv_f1_pos_mean"],
            ) > (
                best_trial["cv_top2_hit_mean"],
                best_trial["cv_bal_acc_mean"],
                best_trial["cv_f1_pos_mean"],
            ):
                best_trial = {"params": params, **trial_row}

        if best_trial is None:
            continue

        tuned_variant = build_model_variant(dataset_alias, model_family, suffix="TUNED_v1")
        estimator = build_base_model(
            model_family=model_family,
            params=best_trial["params"],
            cat_feature_names=dataset_spec["feature_cols_cat"] if model_family == "CATBOOST" else None,
        )
        holdout_result = fit_and_evaluate_model(
            model_family=model_family,
            estimator=estimator,
            train_df=train_df,
            test_df=test_df,
            feature_cols_num=dataset_spec["feature_cols_num"],
            feature_cols_cat=dataset_spec["feature_cols_cat"],
            target_col=dataset_spec["target_col"],
        )
        metrics = {
            "accuracy": holdout_result["row_metrics"]["accuracy"],
            "balanced_accuracy": holdout_result["row_metrics"]["balanced_accuracy"],
            "f1_pos": holdout_result["row_metrics"]["f1_pos"],
            "top1_hit": holdout_result["event_metrics"]["top1_hit"],
            "top2_hit": holdout_result["event_metrics"]["top2_hit"],
            "coverage": holdout_result["event_metrics"]["coverage"],
            "test_events": holdout_result["event_metrics"]["test_events"],
        }
        selection_rule = (
            f"day05_phase2_tuned({model_family} on {dataset_alias}) -> "
            "cv_rank(top2_hit, balanced_accuracy, f1_pos) -> registry_gate(top2>=baseline+0.01 & bal_acc>=baseline+0.01 & coverage>=baseline-0.005)"
        )
        model_path, metadata_path = save_candidate_artifacts(
            candidate_model_dir=paths["candidate_model_dir"],
            run_id=run_id,
            model_variant=tuned_variant,
            model=holdout_result["model"],
            metrics=metrics,
            cutoff_date=cutoff_date,
            dataset_name=dataset_spec["dataset_path"].name,
            feature_columns=holdout_result["feature_columns"],
            selection_rule=selection_rule,
            extra_metadata={
                "day_id": "Day 05",
                "phase": "phase2",
                "dataset_alias": dataset_alias,
                "model_family": model_family,
                "balance_tag": "",
                "best_trial_params": best_trial["params"],
                "search_log_path": str(output_paths["phase2_trials_csv"]),
            },
        )
        metrics_payload = build_pure_metrics_payload(
            run_id=run_id,
            ts_utc=ts_utc,
            day_id="Day 05",
            cutoff_date=cutoff_date,
            model_variant=tuned_variant,
            dataset_alias=dataset_alias,
            dataset_path=dataset_spec["dataset_path"],
            model_family=model_family,
            variant_stage="phase2",
            metrics=metrics,
            model_path=model_path,
            metadata_path=metadata_path,
            search_log_path=output_paths["phase2_trials_csv"],
        )
        metrics_json_path = write_metrics_payload(
            payload=metrics_payload,
            candidates_root=output_paths["candidates_root"],
            model_variant=tuned_variant,
        )
        if not args.skip_registry:
            append_registry_candidate(
                registry_csv=paths["registry_csv"],
                run_id=run_id,
                day_id="Day 05",
                model_variant=tuned_variant,
                metadata_path=metadata_path,
                metrics_json_path=metrics_json_path,
                dataset_csv=dataset_spec["dataset_path"],
                model_path=model_path,
            )
        final_rows.append(
            build_canonical_row(
                run_id=run_id,
                cutoff_date=cutoff_date,
                dataset_alias=dataset_alias,
                dataset_path=dataset_spec["dataset_path"],
                lr_equivalent_variant=dataset_spec["lr_equivalent_variant"],
                model_family=model_family,
                model_variant=tuned_variant,
                variant_stage="phase2",
                balance_tag="",
                metrics=metrics,
                baseline_metrics=baseline_metrics,
                lr_reference_metrics=lr_reference_metrics,
                search_log_path=output_paths["phase2_trials_csv"],
            )
        )

    phase2_trials_df = pd.DataFrame(phase2_trials)
    if not phase2_trials_df.empty:
        phase2_trials_df.to_csv(output_paths["phase2_trials_csv"], index=False)
        output_paths["phase2_trials_json"].parent.mkdir(parents=True, exist_ok=True)
        output_paths["phase2_trials_json"].write_text(
            phase2_trials_df.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )

    if final_rows:
        final_df = pd.DataFrame(final_rows)
        final_df["selected_for_phase2"] = True
        final_df["selected_for_phase3"] = final_df["eligible_phase3"]
        canonical_df = pd.concat([canonical_df, final_df], ignore_index=True)
        canonical_df = canonical_df.sort_values(METRIC_SORT_COLUMNS, ascending=False).reset_index(drop=True)
        write_dataframe_outputs(output_paths["canonical_csv"], output_paths["canonical_json"], canonical_df)
        write_json(
            output_paths["selection_json"],
            {
                "run_id": run_id,
                "phase2_selected_variants": selected,
                "phase3_selected_variants": final_df.loc[final_df["selected_for_phase3"], "model_variant"].astype(str).tolist(),
                "best_final_pure_variant": choose_best_final_pure_variant(canonical_df),
            },
        )
    return canonical_df, phase2_trials


# SECTION: Phase 3
def run_phase3(
    *,
    run_id: str,
    ts_utc: str,
    cutoff_date: str,
    canonical_df: pd.DataFrame,
    args: argparse.Namespace,
    paths: dict[str, Path],
    output_paths: dict[str, Path],
    baseline_metrics: dict[str, float],
    lr_reference_rows: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run bounded balance experiments for tuned candidates that remain promising."""
    project_root = paths["project_root"]
    dataset_catalog = get_day05_dataset_catalog(project_root=project_root)
    selected = canonical_df.loc[
        (canonical_df["variant_stage"] == "phase2") & canonical_df["eligible_phase3"],
        "model_variant",
    ].astype(str).tolist()
    if not selected:
        return canonical_df, []

    phase3_trials: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    for tuned_variant in selected:
        tuned_row = canonical_df[canonical_df["model_variant"] == tuned_variant].iloc[0]
        dataset_alias = str(tuned_row["dataset_alias"])
        model_family = str(tuned_row["model_family"])
        dataset_spec = dataset_catalog[dataset_alias]
        dataset_df = pd.read_csv(dataset_spec["dataset_path"], keep_default_na=False)
        model_df = prepare_day05_model_frame(
            dataset_df=dataset_df,
            feature_cols_num=dataset_spec["feature_cols_num"],
            feature_cols_cat=dataset_spec["feature_cols_cat"],
            target_col=dataset_spec["target_col"],
        )
        train_df, test_df = split_day05_by_cutoff(model_df, cutoff_date=cutoff_date)
        y_train = train_df[dataset_spec["target_col"]].astype(int)
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        class_ratio = float(neg / pos) if pos else 1.0
        lr_reference_metrics = extract_reference_metrics(lr_reference_rows[dataset_spec["lr_equivalent_variant"]])
        tuned_metadata_path = paths["candidate_model_dir"] / slugify_variant(tuned_variant) / run_id / "metadata.json"
        tuned_metadata = load_json(tuned_metadata_path)
        tuned_params = {}
        for key, value in tuned_metadata.get("best_trial_params", {}).items():
            tuned_params[key] = value
        if not tuned_params:
            tuned_params = get_phase1_default_params(model_family)

        trial_specs = get_phase3_trial_specs(model_family, tuned_params, class_ratio)
        best_trial: dict[str, Any] | None = None

        for trial_idx, trial_spec in enumerate(trial_specs, start=1):
            if model_family == "CATBOOST" and trial_spec["balance_tag"] == "SMOTE":
                continue
            cv_payload = evaluate_cv_trial(
                model_family=model_family,
                params=trial_spec["params"],
                train_df=train_df,
                feature_cols_num=dataset_spec["feature_cols_num"],
                feature_cols_cat=dataset_spec["feature_cols_cat"],
                target_col=dataset_spec["target_col"],
                n_splits=args.n_splits,
                resampler_name=trial_spec["resampler"],
            )
            trial_row = {
                "run_id": run_id,
                "base_model_variant": tuned_variant,
                "dataset_alias": dataset_alias,
                "model_family": model_family,
                "balance_tag": trial_spec["balance_tag"],
                "trial_idx": trial_idx,
                "params_json": serialize_params(trial_spec["params"]),
                "resampler": trial_spec["resampler"] or "",
                "cv_top2_hit_mean": cv_payload["cv_top2_hit_mean"],
                "cv_bal_acc_mean": cv_payload["cv_bal_acc_mean"],
                "cv_f1_pos_mean": cv_payload["cv_f1_pos_mean"],
                "cv_coverage_mean": cv_payload["cv_coverage_mean"],
                "folds_json": json.dumps(cv_payload["folds"], ensure_ascii=False),
            }
            phase3_trials.append(trial_row)
            if best_trial is None:
                best_trial = {"trial_spec": trial_spec, **trial_row}
                continue
            if (
                trial_row["cv_top2_hit_mean"],
                trial_row["cv_bal_acc_mean"],
                trial_row["cv_f1_pos_mean"],
            ) > (
                best_trial["cv_top2_hit_mean"],
                best_trial["cv_bal_acc_mean"],
                best_trial["cv_f1_pos_mean"],
            ):
                best_trial = {"trial_spec": trial_spec, **trial_row}

        if best_trial is None:
            continue

        balance_tag = str(best_trial["balance_tag"])
        balanced_variant = build_model_variant(dataset_alias, model_family, suffix=f"{balance_tag}_v1")
        estimator = build_base_model(
            model_family=model_family,
            params=best_trial["trial_spec"]["params"],
            cat_feature_names=dataset_spec["feature_cols_cat"] if model_family == "CATBOOST" else None,
        )
        holdout_result = fit_and_evaluate_model(
            model_family=model_family,
            estimator=estimator,
            train_df=train_df,
            test_df=test_df,
            feature_cols_num=dataset_spec["feature_cols_num"],
            feature_cols_cat=dataset_spec["feature_cols_cat"],
            target_col=dataset_spec["target_col"],
            resampler_name=best_trial["trial_spec"]["resampler"],
        )
        metrics = {
            "accuracy": holdout_result["row_metrics"]["accuracy"],
            "balanced_accuracy": holdout_result["row_metrics"]["balanced_accuracy"],
            "f1_pos": holdout_result["row_metrics"]["f1_pos"],
            "top1_hit": holdout_result["event_metrics"]["top1_hit"],
            "top2_hit": holdout_result["event_metrics"]["top2_hit"],
            "coverage": holdout_result["event_metrics"]["coverage"],
            "test_events": holdout_result["event_metrics"]["test_events"],
        }
        selection_rule = (
            f"day05_phase3_balance({model_family} on {dataset_alias}, {balance_tag}) -> "
            "cv_rank(top2_hit, balanced_accuracy, f1_pos) -> registry_gate(top2>=baseline+0.01 & bal_acc>=baseline+0.01 & coverage>=baseline-0.005)"
        )
        model_path, metadata_path = save_candidate_artifacts(
            candidate_model_dir=paths["candidate_model_dir"],
            run_id=run_id,
            model_variant=balanced_variant,
            model=holdout_result["model"],
            metrics=metrics,
            cutoff_date=cutoff_date,
            dataset_name=dataset_spec["dataset_path"].name,
            feature_columns=holdout_result["feature_columns"],
            selection_rule=selection_rule,
            extra_metadata={
                "day_id": "Day 05",
                "phase": "phase3",
                "dataset_alias": dataset_alias,
                "model_family": model_family,
                "balance_tag": balance_tag,
                "best_trial_params": best_trial["trial_spec"]["params"],
                "resampler": best_trial["trial_spec"]["resampler"] or "",
                "search_log_path": str(output_paths["phase3_trials_csv"]),
            },
        )
        metrics_payload = build_pure_metrics_payload(
            run_id=run_id,
            ts_utc=ts_utc,
            day_id="Day 05",
            cutoff_date=cutoff_date,
            model_variant=balanced_variant,
            dataset_alias=dataset_alias,
            dataset_path=dataset_spec["dataset_path"],
            model_family=model_family,
            variant_stage="phase3",
            metrics=metrics,
            model_path=model_path,
            metadata_path=metadata_path,
            search_log_path=output_paths["phase3_trials_csv"],
        )
        metrics_json_path = write_metrics_payload(
            payload=metrics_payload,
            candidates_root=output_paths["candidates_root"],
            model_variant=balanced_variant,
        )
        if not args.skip_registry:
            append_registry_candidate(
                registry_csv=paths["registry_csv"],
                run_id=run_id,
                day_id="Day 05",
                model_variant=balanced_variant,
                metadata_path=metadata_path,
                metrics_json_path=metrics_json_path,
                dataset_csv=dataset_spec["dataset_path"],
                model_path=model_path,
            )
        final_rows.append(
            build_canonical_row(
                run_id=run_id,
                cutoff_date=cutoff_date,
                dataset_alias=dataset_alias,
                dataset_path=dataset_spec["dataset_path"],
                lr_equivalent_variant=dataset_spec["lr_equivalent_variant"],
                model_family=model_family,
                model_variant=balanced_variant,
                variant_stage="phase3",
                balance_tag=balance_tag,
                metrics=metrics,
                baseline_metrics=baseline_metrics,
                lr_reference_metrics=lr_reference_metrics,
                search_log_path=output_paths["phase3_trials_csv"],
            )
        )

    phase3_trials_df = pd.DataFrame(phase3_trials)
    if not phase3_trials_df.empty:
        phase3_trials_df.to_csv(output_paths["phase3_trials_csv"], index=False)
        output_paths["phase3_trials_json"].parent.mkdir(parents=True, exist_ok=True)
        output_paths["phase3_trials_json"].write_text(
            phase3_trials_df.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )

    if final_rows:
        final_df = pd.DataFrame(final_rows)
        final_df["selected_for_phase2"] = False
        final_df["selected_for_phase3"] = True
        canonical_df = pd.concat([canonical_df, final_df], ignore_index=True)
        canonical_df = canonical_df.sort_values(METRIC_SORT_COLUMNS, ascending=False).reset_index(drop=True)
        write_dataframe_outputs(output_paths["canonical_csv"], output_paths["canonical_json"], canonical_df)
        write_json(
            output_paths["selection_json"],
            {
                "run_id": run_id,
                "phase2_selected_variants": canonical_df.loc[canonical_df["selected_for_phase2"], "model_variant"].astype(str).tolist(),
                "phase3_selected_variants": final_df["model_variant"].astype(str).tolist(),
                "best_final_pure_variant": choose_best_final_pure_variant(canonical_df),
            },
        )
    return canonical_df, phase3_trials


# SECTION: Policy helpers
def build_scored_holdout(detail_df: pd.DataFrame, eval_frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Reattach scores and ranks to the holdout candidate rows for the single policy comparison."""
    scored_df = detail_df.reset_index(drop=True).copy()
    scored_df["score_model"] = pd.to_numeric(eval_frame["score_model"], errors="coerce").fillna(0.0).to_numpy()
    scored_df["pred_label"] = pd.to_numeric(eval_frame["pred_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    scored_df["rank_event_score"] = (
        scored_df.groupby("event_id")["score_model"].rank(method="first", ascending=False).astype(int)
    )
    scored_df["is_top1"] = (scored_df["rank_event_score"] == 1).astype(int)
    scored_df["is_topk"] = (scored_df["rank_event_score"] <= top_k).astype(int)
    scored_df["fecha_evento"] = pd.to_datetime(scored_df["fecha_evento"], errors="coerce").dt.strftime("%Y-%m-%d")
    return scored_df


# SECTION: Policy helpers
def build_policy_metrics_payload(
    *,
    run_id: str,
    ts_utc: str,
    day_id: str = "Day 05",
    cutoff_date: str,
    model_variant: str,
    dataset_path: Path,
    pure_metrics: dict[str, Any],
    policy_metrics: dict[str, Any],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    """Build the after-policy metrics payload for one single secondary comparison."""
    return {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": day_id,
        "scope": "after_policy",
        "eval_scope": "day02_test_like",
        "cutoff_applied": cutoff_date,
        "policy": "PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009",
        "model_variant": build_policy_variant(model_variant),
        "metrics": {
            "accuracy": float(pure_metrics["accuracy"]),
            "balanced_accuracy": float(pure_metrics["balanced_accuracy"]),
            "f1_pos": float(pure_metrics["f1_pos"]),
            "top1_hit": float(policy_metrics["top1_hit_after"]),
            "top2_hit": float(policy_metrics["top2_hit_after"]),
            "coverage": float(policy_metrics["coverage_after"]),
            "test_events": int(policy_metrics["test_events"]),
            "top1_hit_before": float(policy_metrics["top1_hit_before"]),
            "top2_hit_before": float(policy_metrics["top2_hit_before"]),
            "coherence_before": float(policy_metrics["coherence_before"]),
            "coherence_after": float(policy_metrics["coherence_after"]),
            "coherence_delta": float(policy_metrics["coherence_delta"]),
            "overrides_count": int(policy_metrics["overrides_count"]),
            "overrides_improved": int(policy_metrics["overrides_improved"]),
            "overrides_harmed": int(policy_metrics["overrides_harmed"]),
            "overrides_neutral": int(policy_metrics["overrides_neutral"]),
            "coverage_before": float(policy_metrics["coverage_before"]),
            "coverage_after": float(policy_metrics["coverage_after"]),
        },
        "sources": {
            "dataset_csv": str(dataset_path),
            **source_paths,
        },
    }


# SECTION: Policy helpers
def run_single_policy_check(
    *,
    run_id: str,
    ts_utc: str,
    day_id: str = "Day 05",
    cutoff_date: str,
    canonical_df: pd.DataFrame,
    args: argparse.Namespace,
    paths: dict[str, Path],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    """Run one single policy comparison only if the best final pure candidate is defendible."""
    dataset_catalog = get_day05_dataset_catalog(project_root=paths["project_root"])
    best_pure_variant = choose_best_final_pure_variant(canonical_df)
    best_row = canonical_df[canonical_df["model_variant"] == best_pure_variant].iloc[0]
    if not is_policy_eligible(best_row):
        summary = {
            "run_id": run_id,
            "executed": False,
            "reason": "best_pure_not_policy_eligible",
            "best_final_pure_variant": best_pure_variant,
        }
        write_json(output_paths["policy_summary_json"], summary)
        return summary

    dataset_alias = str(best_row["dataset_alias"])
    dataset_spec = dataset_catalog[dataset_alias]
    dataset_df = pd.read_csv(dataset_spec["dataset_path"], keep_default_na=False)
    model_df = prepare_day05_model_frame(
        dataset_df=dataset_df,
        feature_cols_num=dataset_spec["feature_cols_num"],
        feature_cols_cat=dataset_spec["feature_cols_cat"],
        target_col=dataset_spec["target_col"],
    )
    train_df, test_df = split_day05_by_cutoff(model_df, cutoff_date=cutoff_date)
    variant_dir = paths["candidate_model_dir"] / slugify_variant(best_pure_variant) / run_id
    model_path = variant_dir / "model.pkl"
    metadata_path = variant_dir / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "No existen artefactos del mejor tabular puro para la comparativa con policy. "
            f"variant={best_pure_variant}"
        )

    metadata = load_json(metadata_path)
    model = joblib.load(model_path)
    expected_feature_columns = [str(column) for column in metadata.get("feature_columns", [])]
    score_payload = score_fitted_model(
        model_family=str(best_row["model_family"]),
        model=model,
        train_df=train_df,
        test_df=test_df,
        feature_cols_num=dataset_spec["feature_cols_num"],
        feature_cols_cat=dataset_spec["feature_cols_cat"],
        target_col=dataset_spec["target_col"],
    )
    actual_feature_columns = [str(column) for column in score_payload["feature_columns"]]
    if actual_feature_columns != expected_feature_columns:
        raise ValueError(
            "Contrato de features inconsistente al rehacer scoring para la comparativa con policy. "
            f"variant={best_pure_variant}"
        )
    scored_holdout = build_scored_holdout(
        detail_df=test_df,
        eval_frame=score_payload["eval_frame"],
        top_k=args.top_k,
    )
    raw_output_path = output_paths["day05_root"] / f"{run_id}_{best_pure_variant}_policy_input.csv"
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_holdout.to_csv(raw_output_path, index=False)

    policy_run_id = f"{run_id}_policy"
    audit_paths = build_postinference_audit_paths(
        report_root=paths["reports_root"],
        raw_output_path=raw_output_path,
        run_id=policy_run_id,
        mode="assist",
        albaran_policy=args.albaran_policy,
    )
    rengine.run(
        input_csv=raw_output_path,
        output_csv=audit_paths["detail"],
        output_resumen_csv=audit_paths["resumen_evento"],
        output_resumen_albaran_csv=audit_paths["resumen_albaran"],
        rules_csv=paths["rules_csv"],
        mode="assist",
        albaran_policy=args.albaran_policy,
        summary_json=audit_paths["summary"],
    )

    detail_df = pd.read_csv(audit_paths["detail"], keep_default_na=False)
    resumen_df = pd.read_csv(audit_paths["resumen_evento"], keep_default_na=False)
    resumen_albaran_df = pd.read_csv(audit_paths["resumen_albaran"], keep_default_na=False)
    policy_metrics = compute_postinference_metrics(detail_df, resumen_df, resumen_albaran_df)
    pure_metrics = {
        "accuracy": float(best_row["accuracy"]),
        "balanced_accuracy": float(best_row["balanced_accuracy"]),
        "f1_pos": float(best_row["f1_pos"]),
    }
    policy_variant = build_policy_variant(best_pure_variant)
    metrics_payload = build_policy_metrics_payload(
        run_id=run_id,
        ts_utc=ts_utc,
        day_id=day_id,
        cutoff_date=cutoff_date,
        model_variant=best_pure_variant,
        dataset_path=dataset_spec["dataset_path"],
        pure_metrics=pure_metrics,
        policy_metrics=policy_metrics,
        source_paths={
            "raw_output_csv": str(raw_output_path),
            "detail_output_csv": str(audit_paths["detail"]),
            "resumen_output_csv": str(audit_paths["resumen_evento"]),
            "resumen_albaran_output_csv": str(audit_paths["resumen_albaran"]),
            "policy_summary_json": str(audit_paths["summary"]),
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
        },
    )
    metrics_json_path = write_metrics_payload(
        payload=metrics_payload,
        candidates_root=output_paths["candidates_root"],
        model_variant=policy_variant,
    )
    if not args.skip_registry:
        append_registry_candidate(
            registry_csv=paths["registry_csv"],
            run_id=run_id,
            day_id=day_id,
            model_variant=policy_variant,
            metadata_path=metadata_path,
            metrics_json_path=metrics_json_path,
            dataset_csv=dataset_spec["dataset_path"],
            model_path=model_path,
        )

    summary = {
        "run_id": run_id,
        "executed": True,
        "best_final_pure_variant": best_pure_variant,
        "policy_variant": policy_variant,
        "policy_metrics_path": str(metrics_json_path),
        "policy_summary_path": str(audit_paths["summary"]),
    }
    write_json(output_paths["policy_summary_json"], summary)
    return summary


# SECTION: Orchestration
def build_run_summary(
    *,
    run_id: str,
    cutoff_date: str,
    canonical_df: pd.DataFrame,
    phase2_trials: list[dict[str, Any]],
    phase3_trials: list[dict[str, Any]],
    policy_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build the final Day 05 run summary persisted after the runner finishes."""
    return {
        "run_id": run_id,
        "cutoff_date": cutoff_date,
        "canonical_variants_total": int(len(canonical_df)),
        "phase1_variants": canonical_df.loc[canonical_df["variant_stage"] == "phase1", "model_variant"].astype(str).tolist(),
        "phase2_variants": canonical_df.loc[canonical_df["variant_stage"] == "phase2", "model_variant"].astype(str).tolist(),
        "phase3_variants": canonical_df.loc[canonical_df["variant_stage"] == "phase3", "model_variant"].astype(str).tolist(),
        "phase2_trials_total": int(len(phase2_trials)),
        "phase3_trials_total": int(len(phase3_trials)),
        "phase2_selected_variants": canonical_df.loc[canonical_df["selected_for_phase2"], "model_variant"].astype(str).tolist(),
        "phase3_selected_variants": canonical_df.loc[canonical_df["selected_for_phase3"], "model_variant"].astype(str).tolist(),
        "best_final_pure_variant": choose_best_final_pure_variant(canonical_df),
        "policy_summary": policy_summary,
    }


# SECTION: Orchestration
def run_all(args: argparse.Namespace) -> int:
    """Execute Day 05 end-to-end: phase 1, phase 2, phase 3, policy check, and final summaries."""
    paths = resolve_project_paths(args)
    run_id = args.run_id.strip() or build_run_id()
    ts_utc = utc_now_iso()
    cutoff_date = resolve_cutoff_date(paths["baseline_metadata_path"])
    output_paths = build_day05_output_paths(paths["reports_root"], run_id)
    baseline_row, lr_reference_rows = load_registry_reference_rows(paths["registry_csv"])
    baseline_metrics = extract_reference_metrics(baseline_row)

    canonical_df = run_phase1(
        run_id=run_id,
        ts_utc=ts_utc,
        cutoff_date=cutoff_date,
        args=args,
        paths=paths,
        output_paths=output_paths,
        baseline_metrics=baseline_metrics,
        lr_reference_rows=lr_reference_rows,
    )
    phase2_trials: list[dict[str, Any]] = []
    phase3_trials: list[dict[str, Any]] = []
    if not canonical_df.loc[canonical_df["selected_for_phase2"]].empty:
        canonical_df, phase2_trials = run_phase2(
            run_id=run_id,
            ts_utc=ts_utc,
            cutoff_date=cutoff_date,
            canonical_df=canonical_df,
            args=args,
            paths=paths,
            output_paths=output_paths,
            baseline_metrics=baseline_metrics,
            lr_reference_rows=lr_reference_rows,
        )
        if not canonical_df.loc[(canonical_df["variant_stage"] == "phase2") & canonical_df["eligible_phase3"]].empty:
            canonical_df, phase3_trials = run_phase3(
                run_id=run_id,
                ts_utc=ts_utc,
                cutoff_date=cutoff_date,
                canonical_df=canonical_df,
                args=args,
                paths=paths,
                output_paths=output_paths,
                baseline_metrics=baseline_metrics,
                lr_reference_rows=lr_reference_rows,
            )

    policy_summary = run_single_policy_check(
        run_id=run_id,
        ts_utc=ts_utc,
        cutoff_date=cutoff_date,
        canonical_df=canonical_df,
        args=args,
        paths=paths,
        output_paths=output_paths,
    )
    write_json(
        output_paths["run_summary_json"],
        build_run_summary(
            run_id=run_id,
            cutoff_date=cutoff_date,
            canonical_df=canonical_df,
            phase2_trials=phase2_trials,
            phase3_trials=phase3_trials,
            policy_summary=policy_summary,
        ),
    )
    return 0


# SECTION: Orchestration
def run_phase1_only(args: argparse.Namespace) -> int:
    """Execute only Day 05 Phase 1 and persist the canonical summary outputs."""
    paths = resolve_project_paths(args)
    run_id = args.run_id.strip() or build_run_id()
    ts_utc = utc_now_iso()
    cutoff_date = resolve_cutoff_date(paths["baseline_metadata_path"])
    output_paths = build_day05_output_paths(paths["reports_root"], run_id)
    baseline_row, lr_reference_rows = load_registry_reference_rows(paths["registry_csv"])
    baseline_metrics = extract_reference_metrics(baseline_row)
    run_phase1(
        run_id=run_id,
        ts_utc=ts_utc,
        cutoff_date=cutoff_date,
        args=args,
        paths=paths,
        output_paths=output_paths,
        baseline_metrics=baseline_metrics,
        lr_reference_rows=lr_reference_rows,
    )
    return 0


# SECTION: Main
def main() -> None:
    """Dispatch the requested Day 05 runner mode and exit with an explicit status code."""
    args = parse_args()
    if args.phase == "smoke-imports":
        raise SystemExit(smoke_imports())
    if args.phase == "phase1":
        raise SystemExit(run_phase1_only(args))
    if args.phase == "all":
        raise SystemExit(run_all(args))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
