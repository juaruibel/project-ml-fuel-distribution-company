#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ml.metrics import registry
from src.ml.metrics.postinference_metrics import compute_postinference_metrics
from src.ml.product import recommend_supplier as rs
from src.ml.rules import engine as rengine
from src.ml.shared import functions as fc
from src.ml.shared.helpers import (
    build_postinference_audit_paths,
    build_run_id,
    utc_now_iso,
    write_json,
)

POLICY_VARIANT = "BASELINE_WITH_DETERMINISTIC_LAYER_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest Day03: postinferencia por albarán (PRODUCT_002/PRODUCT_003) con persistencia before/after."
    )
    parser.add_argument(
        "--dataset-csv",
        default="data/public/dataset_modelo_proveedor_v2_candidates.csv",
        help="Dataset de entrada para backtest.",
    )
    parser.add_argument(
        "--model-path",
        default="models/public/baseline/model.pkl",
        help="Ruta model.pkl champion.",
    )
    parser.add_argument(
        "--metadata-path",
        default="models/public/baseline/metadata.json",
        help="Ruta metadata.json champion.",
    )
    parser.add_argument(
        "--rules-csv",
        default="config/business_blocklist_rules.csv",
        help="Reglas de blocklist por candidato.",
    )
    parser.add_argument(
        "--reports-root",
        default="reports",
        help="Directorio raíz de reportes.",
    )
    parser.add_argument(
        "--registry-csv",
        default="artifacts/public/metrics/final_baseline_vs_candidates.csv",
        help="Registro oficial baseline vs candidates.",
    )
    parser.add_argument(
        "--day-id",
        default="Day 03",
        help="Identificador day para registro.",
    )
    parser.add_argument(
        "--baseline-variant",
        default="LR_smote_0.5",
        help="Nombre baseline en el registro.",
    )
    parser.add_argument(
        "--albaran-policy",
        default="PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009",
        choices=["PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009"],
        help="Política de coherencia a evaluar.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k de inferencia base del modelo.",
    )
    parser.add_argument(
        "--eval-scope",
        default="day02_test_like",
        choices=["day02_test_like", "full_dataset"],
        help="Scope de evaluación. `day02_test_like` fuerza comparación 1:1 con Day02 (fecha_evento > cutoff).",
    )
    parser.add_argument(
        "--cutoff-date",
        default="",
        help="Cutoff opcional (YYYY-MM-DD). Si vacío, usa `cutoff_date` de metadata.",
    )
    return parser.parse_args()


def _build_row_metrics(detail_df: pd.DataFrame) -> dict[str, float | None]:
    required = {"target_elegido", "pred_label"}
    if not required.issubset(detail_df.columns):
        return {"accuracy": None, "balanced_accuracy": None, "f1_pos": None}

    y_true = pd.to_numeric(detail_df["target_elegido"], errors="coerce").fillna(0).astype(int)
    y_pred = pd.to_numeric(detail_df["pred_label"], errors="coerce").fillna(0).astype(int)
    metrics = fc.compute_row_metrics(y_true, y_pred)
    return {
        "accuracy": float(metrics.get("accuracy")),
        "balanced_accuracy": float(metrics.get("balanced_accuracy")),
        "f1_pos": float(metrics.get("f1_pos")),
    }


def _build_metrics_payload(
    *,
    run_id: str,
    ts_utc: str,
    day_id: str,
    policy: str,
    scope: str,
    eval_scope: str,
    cutoff_applied: str,
    metrics: dict[str, float | int | None],
    source_paths: dict[str, str],
) -> dict:
    payload = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": day_id,
        "scope": scope,
        "eval_scope": eval_scope,
        "cutoff_applied": cutoff_applied,
        "policy": policy,
        "metrics": metrics,
        "sources": source_paths,
    }
    return payload


def _extract_registry_metrics(
    *,
    row_metrics: dict[str, float | None],
    day03_metrics: dict[str, float | int],
    scope: str,
) -> dict[str, float | int | None]:
    if scope == "before_policy":
        top1 = float(day03_metrics.get("top1_hit_before", 0.0))
        top2 = float(day03_metrics.get("top2_hit_before", 0.0))
        coverage = float(day03_metrics.get("coverage_before", 0.0))
    else:
        top1 = float(day03_metrics.get("top1_hit_after", 0.0))
        top2 = float(day03_metrics.get("top2_hit_after", 0.0))
        coverage = float(day03_metrics.get("coverage_after", 0.0))

    return {
        "accuracy": row_metrics.get("accuracy"),
        "balanced_accuracy": row_metrics.get("balanced_accuracy"),
        "f1_pos": row_metrics.get("f1_pos"),
        "top1_hit": top1,
        "top2_hit": top2,
        "coverage": coverage,
        "test_events": int(day03_metrics.get("test_events", 0)),
    }


def _init_registry_if_needed(
    *,
    registry_csv: Path,
    run_id: str,
    day_id: str,
    baseline_variant: str,
    metadata_path: Path,
    baseline_metrics_json: Path,
    dataset_csv: Path,
    baseline_metrics: dict[str, float | int | None],
    model_path: Path,
) -> None:
    if registry_csv.exists() and registry.read_registry_rows(registry_csv):
        return

    args = argparse.Namespace(
        command="init-baseline",
        output=str(registry_csv),
        run_id=f"{run_id}_BASELINE",
        day_id=day_id,
        model_variant=baseline_variant,
        metadata=str(metadata_path),
        metrics_json=str(baseline_metrics_json),
        dataset=str(dataset_csv),
        coverage=baseline_metrics.get("coverage"),
        test_events=baseline_metrics.get("test_events"),
        model_path=str(model_path),
        overwrite=False,
    )
    registry.init_baseline(args)


def _filter_eval_scope(
    input_df: pd.DataFrame,
    metadata: dict,
    eval_scope: str,
    cutoff_override: str,
) -> tuple[pd.DataFrame, str]:
    if eval_scope == "full_dataset":
        return input_df.copy(), ""

    cutoff = str(cutoff_override).strip() or str(metadata.get("cutoff_date", "")).strip()
    if cutoff == "":
        raise ValueError("No se pudo resolver cutoff para eval-scope day02_test_like.")

    working = input_df.copy()
    working["fecha_evento_dt"] = pd.to_datetime(working["fecha_evento"], errors="coerce")
    cutoff_dt = pd.to_datetime(cutoff, errors="raise")
    filtered = working[working["fecha_evento_dt"] > cutoff_dt].copy()
    filtered = filtered.drop(columns=["fecha_evento_dt"])
    if filtered.empty:
        raise ValueError(
            "Eval-scope day02_test_like produjo dataset vacío. "
            f"cutoff={cutoff}"
        )
    return filtered, cutoff


def run_backtest(args: argparse.Namespace) -> dict:
    project_root = Path(__file__).resolve().parents[4]

    dataset_csv = (project_root / args.dataset_csv).resolve()
    model_path = (project_root / args.model_path).resolve()
    metadata_path = (project_root / args.metadata_path).resolve()
    rules_csv = (project_root / args.rules_csv).resolve()
    reports_root = (project_root / args.reports_root).resolve()
    registry_csv = (project_root / args.registry_csv).resolve()

    if not dataset_csv.exists():
        raise FileNotFoundError(f"No existe dataset: {dataset_csv}")

    run_id = build_run_id()
    ts_utc = utc_now_iso()
    run_date = run_id[:8]

    input_df = pd.read_csv(dataset_csv, dtype=str, keep_default_na=False)
    model, metadata, expected_feature_columns = rs.load_model_bundle(
        model_path=model_path,
        metadata_path=metadata_path,
    )
    eval_df, cutoff_applied = _filter_eval_scope(
        input_df=input_df,
        metadata=metadata,
        eval_scope=args.eval_scope,
        cutoff_override=args.cutoff_date,
    )

    scored_df = rs.run_inference_dataframe(
        input_df=eval_df,
        model=model,
        expected_feature_columns=expected_feature_columns,
        event_col="event_id",
        top_k=args.top_k,
    )

    audit_dir = reports_root / "postinferencia" / "audits" / run_date
    audit_dir.mkdir(parents=True, exist_ok=True)
    scored_input_path = audit_dir / f"day03_scored_input_{run_id}.csv"
    scored_df.to_csv(scored_input_path, index=False)

    audit_paths = build_postinference_audit_paths(
        report_root=reports_root,
        raw_output_path=scored_input_path,
        run_id=run_id,
        mode="assist",
        albaran_policy=args.albaran_policy,
    )

    engine_summary = rengine.run(
        input_csv=scored_input_path,
        output_csv=audit_paths["detail"],
        output_resumen_csv=audit_paths["resumen_evento"],
        output_resumen_albaran_csv=audit_paths["resumen_albaran"],
        rules_csv=rules_csv,
        mode="assist",
        albaran_policy=args.albaran_policy,
        summary_json=audit_paths["summary"],
    )

    detail_df = pd.read_csv(audit_paths["detail"], keep_default_na=False)
    resumen_df = pd.read_csv(audit_paths["resumen_evento"], keep_default_na=False)
    resumen_albaran_df = pd.read_csv(audit_paths["resumen_albaran"], keep_default_na=False)

    day03_metrics = compute_postinference_metrics(
        detail_df=detail_df,
        resumen_df=resumen_df,
        resumen_albaran_df=resumen_albaran_df,
    )
    row_metrics = _build_row_metrics(detail_df)

    baseline_registry_metrics = _extract_registry_metrics(
        row_metrics=row_metrics,
        day03_metrics=day03_metrics,
        scope="before_policy",
    )
    candidate_registry_metrics = _extract_registry_metrics(
        row_metrics=row_metrics,
        day03_metrics=day03_metrics,
        scope="after_policy",
    )

    baseline_metrics_path = (
        reports_root
        / "metrics"
        / "baseline"
        / f"{run_date}_postinferencia_albaran_baseline_metrics.json"
    )
    candidate_metrics_path = (
        reports_root
        / "metrics"
        / "candidates"
        / run_date
        / f"{run_id}_{POLICY_VARIANT}_metrics.json"
    )

    source_paths = {
        "dataset_csv": str(dataset_csv),
        "eval_scope": args.eval_scope,
        "cutoff_applied": cutoff_applied,
        "scored_input_csv": str(scored_input_path),
        "detail_csv": str(audit_paths["detail"]),
        "resumen_evento_csv": str(audit_paths["resumen_evento"]),
        "resumen_albaran_csv": str(audit_paths["resumen_albaran"]),
        "summary_json": str(audit_paths["summary"]),
    }

    baseline_payload = _build_metrics_payload(
        run_id=run_id,
        ts_utc=ts_utc,
        day_id=args.day_id,
        policy=args.albaran_policy,
        scope="before_policy",
        eval_scope=args.eval_scope,
        cutoff_applied=cutoff_applied,
        metrics={
            **baseline_registry_metrics,
            **{key: day03_metrics[key] for key in [
                "pair_groups_PRODUCT_002_PRODUCT_003",
                "coherence_before",
                "coherence_after",
                "coherence_delta",
                "overrides_count",
                "overrides_improved",
                "overrides_harmed",
                "overrides_neutral",
            ] if key in day03_metrics},
        },
        source_paths=source_paths,
    )
    candidate_payload = _build_metrics_payload(
        run_id=run_id,
        ts_utc=ts_utc,
        day_id=args.day_id,
        policy=args.albaran_policy,
        scope="after_policy",
        eval_scope=args.eval_scope,
        cutoff_applied=cutoff_applied,
        metrics={
            **candidate_registry_metrics,
            **day03_metrics,
        },
        source_paths=source_paths,
    )

    write_json(baseline_metrics_path, baseline_payload)
    write_json(candidate_metrics_path, candidate_payload)

    _init_registry_if_needed(
        registry_csv=registry_csv,
        run_id=run_id,
        day_id=args.day_id,
        baseline_variant=args.baseline_variant,
        metadata_path=metadata_path,
        baseline_metrics_json=baseline_metrics_path,
        dataset_csv=dataset_csv,
        baseline_metrics=baseline_registry_metrics,
        model_path=model_path,
    )

    append_args = argparse.Namespace(
        command="append-candidate",
        output=str(registry_csv),
        run_id=run_id,
        day_id=args.day_id,
        model_variant=POLICY_VARIANT,
        metadata=str(metadata_path),
        metrics_json=str(candidate_metrics_path),
        dataset=str(dataset_csv),
        coverage=candidate_registry_metrics.get("coverage"),
        test_events=candidate_registry_metrics.get("test_events"),
        model_path=str(model_path),
        gate_pass="auto",
        promotion_decision="auto",
    )
    registry.append_candidate(append_args)

    run_summary = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "policy_variant": POLICY_VARIANT,
        "engine_summary": engine_summary,
        "day03_metrics": day03_metrics,
        "row_metrics": row_metrics,
        "baseline_metrics_json": str(baseline_metrics_path),
        "candidate_metrics_json": str(candidate_metrics_path),
        "registry_csv": str(registry_csv),
        "audit_paths": {key: str(path) for key, path in audit_paths.items()},
    }

    summary_path = reports_root / "metrics" / "candidates" / run_date / f"{run_id}_{POLICY_VARIANT}_run_summary.json"
    write_json(summary_path, run_summary)
    run_summary["run_summary_json"] = str(summary_path)

    return run_summary


def main() -> None:
    args = parse_args()
    summary = run_backtest(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
