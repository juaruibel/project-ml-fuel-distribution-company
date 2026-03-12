#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone

try:
    from src.ml.metrics import registry
    from src.ml.shared import functions as fc
    from src.ml.shared.helpers import build_run_id, utc_now_iso, write_json
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.ml.metrics import registry
    from src.ml.shared import functions as fc
    from src.ml.shared.helpers import build_run_id, utc_now_iso, write_json


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for rebuilding the official Day 04 baseline on corrected V2."""
    parser = argparse.ArgumentParser(
        description="Reconstruye el baseline oficial Day 04 (LR_smote_0.5) sobre V2 corregido."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v2_candidates.csv"),
        help="Dataset V2 corregido.",
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("artifacts/public/metrics/final_baseline_vs_candidates.csv"),
        help="Registry oficial a reinicializar con la fila baseline.",
    )
    parser.add_argument(
        "--baseline-model-dir",
        type=Path,
        default=Path("models/day04_champion_tuned"),
        help="Directorio del champion baseline corregido.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports"),
        help="Directorio raíz de reportes.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2028-02-21",
        help="Cutoff temporal oficial.",
    )
    parser.add_argument(
        "--day-id",
        type=str,
        default="Day 02",
        help="Day id a usar en la fila baseline del registry.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id opcional para reproducibilidad externa.",
    )
    return parser.parse_args()


# SECTION: Helpers
def _build_metrics_payload(
    *,
    run_id: str,
    ts_utc: str,
    day_id: str,
    cutoff_date: str,
    dataset_path: Path,
    model_path: Path,
    metadata_path: Path,
    metrics: dict[str, float | int | None],
) -> dict[str, Any]:
    """Build the canonical baseline metrics JSON payload consumed by the registry."""
    return {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": day_id,
        "scope": "baseline_retrain",
        "eval_scope": "day02_test_like",
        "cutoff_applied": cutoff_date,
        "model_variant": "LR_smote_0.5",
        "metrics": metrics,
        "sources": {
            "dataset_csv": str(dataset_path),
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
        },
    }


# SECTION: Data preparation
def _prepare_model_frame(
    dataset_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Prepare V2 for model training while preserving the fixed train/test contract."""
    required = {"event_id", "fecha_evento", target_col}
    missing = required - set(dataset_df.columns)
    if missing:
        raise ValueError(f"Dataset V2 incompatible. Faltan columnas requeridas: {sorted(missing)}")

    working = dataset_df.copy()
    for column in feature_cols_num:
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
    for column in feature_cols_cat:
        working[column] = working[column].astype("string").fillna("UNKNOWN").str.strip().replace("", "UNKNOWN")
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce").fillna(0).astype(int)
    working["fecha_evento"] = pd.to_datetime(working["fecha_evento"], errors="coerce")
    working = working.dropna(subset=["fecha_evento"]).reset_index(drop=True)
    return working


# SECTION: Data preparation
def _split_by_cutoff(dataset_df: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the corrected V2 dataset using the official fixed cutoff."""
    cutoff_dt = pd.to_datetime(cutoff_date, errors="raise")
    train_df = dataset_df[dataset_df["fecha_evento"] <= cutoff_dt].copy()
    test_df = dataset_df[dataset_df["fecha_evento"] > cutoff_dt].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Split temporal inválido para baseline corregido. "
            f"train_rows={len(train_df)} test_rows={len(test_df)} cutoff={cutoff_date}"
        )
    return train_df, test_df


# SECTION: Registry
def _init_registry_baseline(
    *,
    registry_csv: Path,
    run_id: str,
    day_id: str,
    metadata_path: Path,
    metrics_json_path: Path,
    dataset_csv: Path,
    coverage: float,
    test_events: int,
    model_path: Path,
) -> None:
    """Initialize the official registry with the corrected baseline row."""
    args = argparse.Namespace(
        command="init-baseline",
        output=str(registry_csv),
        run_id=run_id,
        day_id=day_id,
        model_variant="LR_smote_0.5",
        metadata=str(metadata_path),
        metrics_json=str(metrics_json_path),
        dataset=str(dataset_csv),
        coverage=coverage,
        test_events=test_events,
        model_path=str(model_path),
        overwrite=True,
    )
    registry.init_baseline(args)


# SECTION: Main pipeline
def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild the corrected baseline artifact, metrics JSON, and registry baseline row."""
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = (project_root / args.dataset_csv).resolve()
    registry_csv = (project_root / args.registry_csv).resolve()
    baseline_model_dir = (project_root / args.baseline_model_dir).resolve()
    reports_root = (project_root / args.reports_root).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe dataset V2: {dataset_path}")

    run_id = args.run_id.strip() or build_run_id()
    ts_utc = utc_now_iso()
    run_date = run_id[:8]

    dataset_df = pd.read_csv(dataset_path, keep_default_na=False)
    feature_cols_num, feature_cols_cat, target_col = fc.get_feature_columns_v2()
    model_df = _prepare_model_frame(
        dataset_df=dataset_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )
    train_df, test_df = _split_by_cutoff(dataset_df=model_df, cutoff_date=args.cutoff_date)
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
    if "LR_smote_0.5" not in variant_map:
        raise ValueError("No se encontró la receta LR_smote_0.5 para reconstruir baseline.")

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

    baseline_metrics = {
        "accuracy": float(baseline_row["acc"]),
        "balanced_accuracy": float(baseline_row["bal_acc"]),
        "f1_pos": float(baseline_row["f1_pos"]),
        "top1_hit": float(baseline_row["top1_hit"]),
        "top2_hit": float(baseline_row["top2_hit"]),
        "coverage": 1.0,
        "test_events": int(test_df["event_id"].nunique()),
    }
    metadata_metrics = {
        "top1_hit": baseline_metrics["top1_hit"],
        "top2_hit": baseline_metrics["top2_hit"],
        "test_acc": baseline_metrics["accuracy"],
        "test_bal_acc": baseline_metrics["balanced_accuracy"],
        "test_f1_pos": baseline_metrics["f1_pos"],
        "cv_bal_acc": None,
    }

    model_path, metadata_path, _ = fc.save_champion_artifacts(
        model=baseline_result["model"],
        model_dir=baseline_model_dir,
        model_name="LR_smote_0.5",
        metrics=metadata_metrics,
        cutoff_date=args.cutoff_date,
        dataset_name=dataset_path.name,
        feature_columns=list(X_train.columns),
        selection_rule="top2_hit DESC -> test_bal_acc DESC -> test_f1_pos DESC",
    )

    metrics_dir = reports_root / "metrics" / "baseline"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics_path = metrics_dir / f"{run_id}_baseline_metrics.json"
    baseline_summary_path = metrics_dir / f"{run_id}_baseline_run_summary.json"

    payload = _build_metrics_payload(
        run_id=run_id,
        ts_utc=ts_utc,
        day_id=args.day_id,
        cutoff_date=args.cutoff_date,
        dataset_path=dataset_path,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics=baseline_metrics,
    )
    write_json(baseline_metrics_path, payload)
    _init_registry_baseline(
        registry_csv=registry_csv,
        run_id=run_id,
        day_id=args.day_id,
        metadata_path=metadata_path,
        metrics_json_path=baseline_metrics_path,
        dataset_csv=dataset_path,
        coverage=float(baseline_metrics["coverage"]),
        test_events=int(baseline_metrics["test_events"]),
        model_path=model_path,
    )

    summary = {
        "status": "ok",
        "run_id": run_id,
        "ts_utc": ts_utc,
        "cutoff_date": args.cutoff_date,
        "dataset_csv": str(dataset_path),
        "registry_csv": str(registry_csv),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics_json": str(baseline_metrics_path),
        "metrics": baseline_metrics,
        "run_date": run_date,
    }
    write_json(baseline_summary_path, summary)
    return summary


# SECTION: CLI
def main() -> None:
    """Parse CLI args, rebuild the corrected baseline, and print a machine-readable summary."""
    args = parse_args()
    summary = run_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
