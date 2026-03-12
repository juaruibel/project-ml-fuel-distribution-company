#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone

try:
    from src.etl.marts import build_dataset_modelo_day041_ablation as day041_builder
    from src.ml.metrics import registry
    from src.ml.metrics.postinference_metrics import compute_postinference_metrics
    from src.ml.rules import engine as rengine
    from src.ml.shared import functions as fc
    from src.ml.shared.helpers import build_postinference_audit_paths, build_run_id, utc_now_iso, write_json
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.etl.marts import build_dataset_modelo_day041_ablation as day041_builder
    from src.ml.metrics import registry
    from src.ml.metrics.postinference_metrics import compute_postinference_metrics
    from src.ml.rules import engine as rengine
    from src.ml.shared import functions as fc
    from src.ml.shared.helpers import build_postinference_audit_paths, build_run_id, utc_now_iso, write_json

VARIANT_NAME_MAP = {
    day041_builder.SOURCE_QUALITY_VARIANT: "V2_SOURCE_QUALITY_LR_smote_0.5_v1",
    day041_builder.DISPERSION_VARIANT: "V2_DISPERSION_LR_smote_0.5_v1",
    day041_builder.COMPETITION_VARIANT: "V2_COMPETITION_LR_smote_0.5_v1",
    day041_builder.TRANSPORT_VARIANT: "V2_TRANSPORT_ONLY_LR_smote_0.5_v1",
    day041_builder.SELECTED_VARIANT: "V3_A2_SELECTED_SIGNALS_LR_smote_0.5_v1",
}


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 04.1 ablation training pipeline."""
    parser = argparse.ArgumentParser(
        description="Entrena candidatos Day 04.1 sobre la matriz de ablación y registra métricas oficiales."
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("artifacts/public/data_quality_day041_ablation_matrix.json"),
        help="Reporte de calidad generado por el builder Day 04.1.",
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("artifacts/public/metrics/final_baseline_vs_candidates.csv"),
        help="Registro oficial baseline vs candidatos.",
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
        default=Path("models/candidates/day041_ablation"),
        help="Directorio raíz para artefactos Day 04.1.",
    )
    parser.add_argument(
        "--baseline-metadata-path",
        type=Path,
        default=Path("models/public/baseline/metadata.json"),
        help="Metadata baseline oficial para cutoff de referencia.",
    )
    parser.add_argument(
        "--rules-csv",
        type=Path,
        default=Path("config/business_blocklist_rules.csv"),
        help="Reglas deterministas para evaluación secundaria.",
    )
    parser.add_argument(
        "--v2-input",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v2_candidates.csv"),
        help="Input base V2 requerido si se construye V3_A2.",
    )
    parser.add_argument(
        "--ofertas-typed-input",
        type=Path,
        default=Path("data/public/support/ofertas_typed.csv"),
        help="Input staging de ofertas tipadas requerido si se construye V3_A2.",
    )
    parser.add_argument(
        "--transport-input",
        type=Path,
        default=Path("data/public/support/ofertas_transport_signals.csv"),
        help="Input parser de transporte requerido si se construye V3_A2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/public/day041"),
        help="Directorio de datasets Day 04.1.",
    )
    parser.add_argument(
        "--day-id",
        type=str,
        default="Day 04.1",
        help="Identificador day para trazabilidad.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k de scoring para métricas por evento.",
    )
    parser.add_argument(
        "--albaran-policy",
        type=str,
        default="PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009",
        choices=["PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009"],
        help="Policy Day03 para la evaluación secundaria.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id opcional.",
    )
    return parser.parse_args()


# SECTION: Shared helpers
def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


# SECTION: Shared helpers
def _slugify_variant(model_variant: str) -> str:
    """Build a filesystem-friendly slug from the model variant name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_variant).strip("_").lower()
    return slug


# SECTION: Shared helpers
def _resolve_cutoff_date(metadata_path: Path) -> str:
    """Resolve the fixed cutoff date from the baseline metadata."""
    metadata = _load_json(metadata_path)
    cutoff_date = str(metadata.get("cutoff_date", "")).strip()
    if cutoff_date == "":
        raise ValueError(f"No se pudo resolver cutoff_date desde {metadata_path}")
    return cutoff_date


# SECTION: Shared helpers
def _prepare_model_frame(
    dataset_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Prepare one dataset variant for model training while preserving audit columns."""
    working = dataset_df.copy()
    for column in feature_cols_num:
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
    for column in feature_cols_cat:
        working[column] = working[column].astype("string").fillna("UNKNOWN").str.strip().replace("", "UNKNOWN")
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce").fillna(0).astype(int)
    working["fecha_evento"] = pd.to_datetime(working["fecha_evento"], errors="coerce")
    working = working.dropna(subset=["fecha_evento"]).reset_index(drop=True)
    return working


# SECTION: Shared helpers
def _split_by_cutoff(dataset_df: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the variant dataset using the official fixed cutoff date."""
    cutoff_dt = pd.to_datetime(cutoff_date, errors="raise")
    train_df = dataset_df[dataset_df["fecha_evento"] <= cutoff_dt].copy()
    test_df = dataset_df[dataset_df["fecha_evento"] > cutoff_dt].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Split temporal inválido para Day 04.1. "
            f"train_rows={len(train_df)} test_rows={len(test_df)} cutoff={cutoff_date}"
        )
    return train_df, test_df


# SECTION: Shared helpers
def _build_train_test_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Dummify one variant dataset with the same recipe used by the baseline champion."""
    X_train, X_test, y_train, y_test = fc.dummificar_train_test(
        train=train_df,
        test=test_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return X_train, X_test, y_train.astype(int), y_test.astype(int)


# SECTION: Shared helpers
def _append_registry_row(
    *,
    registry_csv: Path,
    run_id: str,
    day_id: str,
    model_variant: str,
    metadata_path: Path,
    metrics_json_path: Path,
    dataset_csv: Path,
    model_path: Path,
    promotion_decision: str,
) -> None:
    """Append one candidate row to the official registry."""
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
        promotion_decision=promotion_decision,
    )
    registry.append_candidate(args)


# SECTION: Shared helpers
def _registry_has_row(registry_csv: Path, run_id: str, model_variant: str) -> bool:
    """Return whether the registry already contains a row for one run/model pair."""
    rows = registry.read_registry_rows(registry_csv)
    return any(row.get("run_id") == run_id and row.get("model_variant") == model_variant for row in rows)


# SECTION: Shared helpers
def _build_scored_holdout(
    test_df: pd.DataFrame,
    eval_frame: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Reattach scores and predictions to the holdout rows for policy evaluation."""
    scored_df = test_df.reset_index(drop=True).copy()
    scored_df["score_model"] = pd.to_numeric(eval_frame["score_model"], errors="coerce").fillna(0.0).to_numpy()
    scored_df["pred_label"] = pd.to_numeric(eval_frame["pred_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    scored_df["rank_event_score"] = (
        scored_df.groupby("event_id")["score_model"].rank(method="first", ascending=False).astype(int)
    )
    scored_df["is_top1"] = (scored_df["rank_event_score"] == 1).astype(int)
    scored_df["is_topk"] = (scored_df["rank_event_score"] <= top_k).astype(int)
    scored_df["fecha_evento"] = scored_df["fecha_evento"].dt.strftime("%Y-%m-%d")
    return scored_df


# SECTION: Shared helpers
def _selection_rule_for_variant(model_variant: str) -> str:
    """Return the selection rule string stored in model metadata for one variant."""
    return (
        f"fixed_recipe(LR_smote_0.5 on {model_variant}) -> registry_gate("
        "top2>=baseline+0.01 & bal_acc>=baseline+0.01 & coverage>=baseline-0.005)"
    )


# SECTION: Shared helpers
def _variant_features_from_quality(
    quality_payload: dict[str, Any],
    variant_name: str,
) -> tuple[list[str], list[str], str]:
    """Resolve numeric/categorical/target columns for one ablation variant."""
    base_num = list(quality_payload["base_feature_columns_num"])
    base_cat = list(quality_payload["base_feature_columns_cat"])
    target_col = str(quality_payload["target_column"])
    variant_info = quality_payload["variants"][variant_name]
    feature_cols_num = base_num + list(variant_info["added_columns"]) + list(variant_info.get("missing_flag_columns", []))
    return feature_cols_num, base_cat, target_col


# SECTION: Shared helpers
def _baseline_metrics_from_registry(registry_csv: Path) -> dict[str, float]:
    """Load baseline Top-2 and balanced accuracy to choose Day 04.1 selected families."""
    rows = registry.read_registry_rows(registry_csv)
    baseline_row = next(row for row in rows if row.get("model_role") == "baseline")
    return {
        "top2_hit": float(baseline_row["top2_hit"]),
        "balanced_accuracy": float(baseline_row["balanced_accuracy"]),
    }


# SECTION: Shared helpers
def _day041_summary_output_path(reports_root: Path, run_id: str) -> Path:
    """Build the persisted Day 04.1 run summary path consumed by notebook 12."""
    return reports_root / "metrics" / "candidates" / run_id[:8] / f"{run_id}_day041_ablation_run_summary.json"


# SECTION: Model training
def _train_single_variant(
    *,
    run_id: str,
    ts_utc: str,
    day_id: str,
    cutoff_date: str,
    model_variant: str,
    dataset_csv: Path,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
    registry_csv: Path,
    reports_root: Path,
    candidate_model_root: Path,
) -> dict[str, Any]:
    """Train and register one Day 04.1 variant using the fixed LR_smote_0.5 recipe."""
    dataset_df = pd.read_csv(dataset_csv, keep_default_na=False)
    model_df = _prepare_model_frame(
        dataset_df=dataset_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )
    train_df, test_df = _split_by_cutoff(dataset_df=model_df, cutoff_date=cutoff_date)
    X_train, X_test, y_train, y_test = _build_train_test_matrices(
        train_df=train_df,
        test_df=test_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )

    variants = fc.build_default_lr_balance_variants()
    variant_map = {name: (model, sampler) for name, model, sampler in variants}
    pure_model, pure_sampler = variant_map["LR_smote_0.5"]
    result_row, result_payload = fc.evaluate_balance_variant(
        name="LR_smote_0.5",
        model=clone(pure_model),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_df=train_df,
        test_df=test_df,
        sampler=clone(pure_sampler),
    )

    metrics = {
        "accuracy": float(result_row["acc"]),
        "balanced_accuracy": float(result_row["bal_acc"]),
        "f1_pos": float(result_row["f1_pos"]),
        "top1_hit": float(result_row["top1_hit"]),
        "top2_hit": float(result_row["top2_hit"]),
        "coverage": 1.0,
        "test_events": int(test_df["event_id"].nunique()),
    }

    model_variant_slug = _slugify_variant(model_variant)
    model_dir = candidate_model_root / model_variant_slug / run_id
    model_path, metadata_path, _ = fc.save_champion_artifacts(
        model=result_payload["model"],
        model_dir=model_dir,
        model_name=model_variant,
        metrics=metrics,
        cutoff_date=cutoff_date,
        dataset_name=dataset_csv.name,
        feature_columns=list(X_train.columns),
        selection_rule=_selection_rule_for_variant(model_variant=model_variant),
    )

    metrics_dir = reports_root / "metrics" / "candidates" / run_id[:8]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = metrics_dir / f"{run_id}_{model_variant}_metrics.json"
    summary_json_path = metrics_dir / f"{run_id}_{model_variant}_run_summary.json"
    metrics_payload = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": day_id,
        "scope": "model_retrain",
        "eval_scope": "day02_test_like",
        "cutoff_applied": cutoff_date,
        "model_variant": model_variant,
        "metrics": metrics,
        "feature_contract": {
            "feature_columns_num": feature_cols_num,
            "feature_columns_cat": feature_cols_cat,
        },
        "sources": {
            "dataset_csv": str(dataset_csv),
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
        },
    }
    write_json(metrics_json_path, metrics_payload)
    if not _registry_has_row(registry_csv=registry_csv, run_id=run_id, model_variant=model_variant):
        _append_registry_row(
            registry_csv=registry_csv,
            run_id=run_id,
            day_id=day_id,
            model_variant=model_variant,
            metadata_path=metadata_path,
            metrics_json_path=metrics_json_path,
            dataset_csv=dataset_csv,
            model_path=model_path,
            promotion_decision="auto",
        )

    registry_rows = registry.read_registry_rows(registry_csv)
    registry_row = next(
        row for row in registry_rows if row.get("run_id") == run_id and row.get("model_variant") == model_variant
    )
    summary_payload = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": day_id,
        "model_variant": model_variant,
        "cutoff_applied": cutoff_date,
        "metrics": metrics,
        "registry_row": registry_row,
        "artifact_paths": {
            "dataset_csv": str(dataset_csv),
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "metrics_json": str(metrics_json_path),
        },
    }
    write_json(summary_json_path, summary_payload)
    return {
        "model_variant": model_variant,
        "dataset_csv": dataset_csv,
        "feature_cols_num": feature_cols_num,
        "feature_cols_cat": feature_cols_cat,
        "target_col": target_col,
        "metrics": metrics,
        "registry_row": registry_row,
        "metrics_json_path": metrics_json_path,
        "summary_json_path": summary_json_path,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "test_df": test_df,
        "eval_frame": result_payload["eval_frame"],
    }


# SECTION: Policy evaluation
def _evaluate_policy_for_best_variant(
    *,
    best_result: dict[str, Any],
    run_id: str,
    ts_utc: str,
    day_id: str,
    cutoff_date: str,
    rules_csv: Path,
    reports_root: Path,
    registry_csv: Path,
    top_k: int,
    albaran_policy: str,
) -> dict[str, Any]:
    """Run the Day03 deterministic layer over the best pure Day 04.1 variant."""
    policy_variant = f"{best_result['model_variant'].replace('_v1', '')}_WITH_DETERMINISTIC_LAYER_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009_v1"
    scored_df = _build_scored_holdout(
        test_df=best_result["test_df"],
        eval_frame=best_result["eval_frame"],
        top_k=top_k,
    )
    audit_dir = reports_root / "postinferencia" / "audits" / run_id[:8]
    audit_dir.mkdir(parents=True, exist_ok=True)
    scored_input_path = audit_dir / f"{_slugify_variant(best_result['model_variant'])}_scored_input_{run_id}.csv"
    scored_df.to_csv(scored_input_path, index=False)

    audit_paths = build_postinference_audit_paths(
        report_root=reports_root,
        raw_output_path=scored_input_path,
        run_id=run_id,
        mode="assist",
        albaran_policy=albaran_policy,
    )
    engine_summary = rengine.run(
        input_csv=scored_input_path,
        output_csv=audit_paths["detail"],
        output_resumen_csv=audit_paths["resumen_evento"],
        output_resumen_albaran_csv=audit_paths["resumen_albaran"],
        rules_csv=rules_csv,
        mode="assist",
        albaran_policy=albaran_policy,
        summary_json=audit_paths["summary"],
    )

    detail_df = pd.read_csv(audit_paths["detail"], keep_default_na=False)
    resumen_df = pd.read_csv(audit_paths["resumen_evento"], keep_default_na=False)
    resumen_albaran_df = pd.read_csv(audit_paths["resumen_albaran"], keep_default_na=False)
    policy_metrics = compute_postinference_metrics(
        detail_df=detail_df,
        resumen_df=resumen_df,
        resumen_albaran_df=resumen_albaran_df,
    )

    metrics_dir = reports_root / "metrics" / "candidates" / run_id[:8]
    metrics_json_path = metrics_dir / f"{run_id}_{policy_variant}_metrics.json"
    summary_json_path = metrics_dir / f"{run_id}_{policy_variant}_run_summary.json"
    metrics_payload = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": day_id,
        "scope": "after_policy",
        "eval_scope": "day02_test_like",
        "cutoff_applied": cutoff_date,
        "model_variant": policy_variant,
        "metrics": {
            "accuracy": float(best_result["metrics"]["accuracy"]),
            "balanced_accuracy": float(best_result["metrics"]["balanced_accuracy"]),
            "f1_pos": float(best_result["metrics"]["f1_pos"]),
            "top1_hit": float(policy_metrics["top1_hit_after"]),
            "top2_hit": float(policy_metrics["top2_hit_after"]),
            "coverage": float(policy_metrics["coverage_after"]),
            "test_events": int(policy_metrics["test_events"]),
            "top1_hit_before": float(policy_metrics["top1_hit_before"]),
            "top2_hit_before": float(policy_metrics["top2_hit_before"]),
            "top1_hit_after": float(policy_metrics["top1_hit_after"]),
            "top2_hit_after": float(policy_metrics["top2_hit_after"]),
            "pair_groups_PRODUCT_002_PRODUCT_003": int(policy_metrics["pair_groups_PRODUCT_002_PRODUCT_003"]),
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
            "dataset_csv": str(best_result["dataset_csv"]),
            "model_path": str(best_result["model_path"]),
            "metadata_path": str(best_result["metadata_path"]),
            "scored_input_csv": str(scored_input_path),
            "detail_csv": str(audit_paths["detail"]),
            "resumen_evento_csv": str(audit_paths["resumen_evento"]),
            "resumen_albaran_csv": str(audit_paths["resumen_albaran"]),
            "summary_json": str(audit_paths["summary"]),
        },
    }
    write_json(metrics_json_path, metrics_payload)
    if not _registry_has_row(registry_csv=registry_csv, run_id=run_id, model_variant=policy_variant):
        _append_registry_row(
            registry_csv=registry_csv,
            run_id=run_id,
            day_id=day_id,
            model_variant=policy_variant,
            metadata_path=best_result["metadata_path"],
            metrics_json_path=metrics_json_path,
            dataset_csv=best_result["dataset_csv"],
            model_path=best_result["model_path"],
            promotion_decision="auto",
        )

    registry_rows = registry.read_registry_rows(registry_csv)
    registry_row = next(
        row for row in registry_rows if row.get("run_id") == run_id and row.get("model_variant") == policy_variant
    )
    summary_payload = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": day_id,
        "model_variant": policy_variant,
        "cutoff_applied": cutoff_date,
        "metrics": metrics_payload["metrics"],
        "registry_row": registry_row,
        "engine_summary": engine_summary,
        "artifact_paths": {
            "metrics_json": str(metrics_json_path),
            "detail_csv": str(audit_paths["detail"]),
            "resumen_evento_csv": str(audit_paths["resumen_evento"]),
            "resumen_albaran_csv": str(audit_paths["resumen_albaran"]),
            "summary_json": str(audit_paths["summary"]),
        },
    }
    write_json(summary_json_path, summary_payload)
    return {
        "policy_variant": policy_variant,
        "metrics_json_path": metrics_json_path,
        "summary_json_path": summary_json_path,
    }


# SECTION: Quality report updates
def _update_quality_report_with_selected_variant(
    quality_report_path: Path,
    selected_summary: dict[str, Any],
) -> None:
    """Append the optional V3_A2 summary to the Day 04.1 quality report."""
    payload = _load_json(quality_report_path)
    payload["variants"][day041_builder.SELECTED_VARIANT] = selected_summary
    quality_report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# SECTION: Main pipeline
def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Train all Day 04.1 variants, optionally build V3_A2, and run policy evaluation on the best pure variant."""
    project_root = Path(__file__).resolve().parents[3]
    quality_report_path = (project_root / args.quality_report).resolve()
    registry_csv = (project_root / args.registry_csv).resolve()
    reports_root = (project_root / args.reports_root).resolve()
    candidate_model_root = (project_root / args.candidate_model_dir).resolve()
    baseline_metadata_path = (project_root / args.baseline_metadata_path).resolve()
    rules_csv = (project_root / args.rules_csv).resolve()
    v2_input_path = (project_root / args.v2_input).resolve()
    ofertas_typed_input_path = (project_root / args.ofertas_typed_input).resolve()
    transport_input_path = (project_root / args.transport_input).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    run_id = args.run_id.strip() or build_run_id()
    ts_utc = utc_now_iso()
    cutoff_date = _resolve_cutoff_date(metadata_path=baseline_metadata_path)
    quality_payload = _load_json(quality_report_path)

    trained_results: list[dict[str, Any]] = []
    pure_variant_names = [
        day041_builder.SOURCE_QUALITY_VARIANT,
        day041_builder.DISPERSION_VARIANT,
        day041_builder.COMPETITION_VARIANT,
        day041_builder.TRANSPORT_VARIANT,
    ]

    for variant_name in pure_variant_names:
        variant_info = quality_payload["variants"][variant_name]
        if variant_info.get("status") != "built":
            continue
        dataset_csv = Path(variant_info["dataset_path"])
        feature_cols_num, feature_cols_cat, target_col = _variant_features_from_quality(
            quality_payload=quality_payload,
            variant_name=variant_name,
        )
        trained_results.append(
            _train_single_variant(
                run_id=run_id,
                ts_utc=ts_utc,
                day_id=args.day_id,
                cutoff_date=cutoff_date,
                model_variant=VARIANT_NAME_MAP[variant_name],
                dataset_csv=dataset_csv,
                feature_cols_num=feature_cols_num,
                feature_cols_cat=feature_cols_cat,
                target_col=target_col,
                registry_csv=registry_csv,
                reports_root=reports_root,
                candidate_model_root=candidate_model_root,
            )
        )

    baseline_metrics = _baseline_metrics_from_registry(registry_csv=registry_csv)
    selected_families: list[str] = []
    for variant_name in pure_variant_names:
        if variant_name == day041_builder.TRANSPORT_VARIANT and quality_payload["variants"][variant_name].get("status") != "built":
            continue
        model_variant = VARIANT_NAME_MAP[variant_name]
        result = next((item for item in trained_results if item["model_variant"] == model_variant), None)
        if result is None:
            continue
        if (
            float(result["metrics"]["top2_hit"]) >= baseline_metrics["top2_hit"]
            and float(result["metrics"]["balanced_accuracy"]) >= baseline_metrics["balanced_accuracy"]
        ):
            selected_families.append(variant_name)

    selected_result: dict[str, Any] | None = None
    if len(selected_families) >= 2:
        dataset_v2 = pd.read_csv(v2_input_path, keep_default_na=False)
        ofertas_typed = pd.read_csv(ofertas_typed_input_path, keep_default_na=False)
        transport_df = pd.read_csv(transport_input_path, keep_default_na=False) if transport_input_path.exists() else pd.DataFrame()
        variant_frames = day041_builder._variant_frames(
            dataset_v2=dataset_v2,
            ofertas_typed=ofertas_typed,
            transport_df=transport_df,
        )
        selected_build = day041_builder.build_selected_signals_dataset(
            dataset_v2=dataset_v2,
            variant_frames=variant_frames,
            selected_families=selected_families,
            output_dir=output_dir,
            cutoff_date=cutoff_date,
        )
        if selected_build is not None:
            selected_path, selected_summary = selected_build
            selected_summary["status"] = "built_after_ablation"
            _update_quality_report_with_selected_variant(
                quality_report_path=quality_report_path,
                selected_summary=selected_summary,
            )
            feature_cols_num = list(quality_payload["base_feature_columns_num"]) + list(selected_summary["added_columns"]) + list(
                selected_summary.get("missing_flag_columns", [])
            )
            feature_cols_cat = list(quality_payload["base_feature_columns_cat"])
            selected_result = _train_single_variant(
                run_id=run_id,
                ts_utc=ts_utc,
                day_id=args.day_id,
                cutoff_date=cutoff_date,
                model_variant=VARIANT_NAME_MAP[day041_builder.SELECTED_VARIANT],
                dataset_csv=selected_path,
                feature_cols_num=feature_cols_num,
                feature_cols_cat=feature_cols_cat,
                target_col=str(quality_payload["target_column"]),
                registry_csv=registry_csv,
                reports_root=reports_root,
                candidate_model_root=candidate_model_root,
            )

    best_pure_result = max(
        trained_results,
        key=lambda item: (
            float(item["metrics"]["top2_hit"]),
            float(item["metrics"]["balanced_accuracy"]),
            float(item["metrics"]["f1_pos"]),
        ),
    )
    policy_result = _evaluate_policy_for_best_variant(
        best_result=best_pure_result,
        run_id=run_id,
        ts_utc=ts_utc,
        day_id=args.day_id,
        cutoff_date=cutoff_date,
        rules_csv=rules_csv,
        reports_root=reports_root,
        registry_csv=registry_csv,
        top_k=args.top_k,
        albaran_policy=args.albaran_policy,
    )

    summary = {
        "status": "ok",
        "run_id": run_id,
        "trained_variants": [result["model_variant"] for result in trained_results],
        "selected_families": selected_families,
        "selected_variant_trained": selected_result["model_variant"] if selected_result is not None else "",
        "best_pure_variant": best_pure_result["model_variant"],
        "policy_variant": policy_result["policy_variant"],
        "transport_variant_status": quality_payload["variants"][day041_builder.TRANSPORT_VARIANT].get("status", ""),
        "transport_gate_pass": bool(
            quality_payload["variants"][day041_builder.TRANSPORT_VARIANT].get("transport_gate_pass", False)
        ),
    }
    write_json(_day041_summary_output_path(reports_root=reports_root, run_id=run_id), summary)
    return summary


# SECTION: CLI entrypoint
def main() -> None:
    """Run the CLI entrypoint for the Day 04.1 training pipeline."""
    args = parse_args()
    summary = run_pipeline(args=args)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
