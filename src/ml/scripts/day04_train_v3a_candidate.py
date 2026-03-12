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
    from src.etl.marts.build_dataset_modelo_v3_context import get_feature_columns_v3
    from src.ml.metrics import registry
    from src.ml.metrics.postinference_metrics import compute_postinference_metrics
    from src.ml.rules import engine as rengine
    from src.ml.shared import functions as fc
    from src.ml.shared.helpers import build_postinference_audit_paths, build_run_id, utc_now_iso, write_json
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.etl.marts.build_dataset_modelo_v3_context import get_feature_columns_v3
    from src.ml.metrics import registry
    from src.ml.metrics.postinference_metrics import compute_postinference_metrics
    from src.ml.rules import engine as rengine
    from src.ml.shared import functions as fc
    from src.ml.shared.helpers import build_postinference_audit_paths, build_run_id, utc_now_iso, write_json

PURE_MODEL_VARIANT = "V3_A_LR_smote_0.5_v1"
POLICY_MODEL_VARIANT = "V3_A_WITH_DETERMINISTIC_LAYER_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009_v1"


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 04 V3_A training/evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa el candidato Day 04 V3_A con comparación 1:1 contra baseline."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v3_context.csv"),
        help="Dataset V3_A a evaluar.",
    )
    parser.add_argument(
        "--baseline-model-path",
        type=Path,
        default=Path("models/public/baseline/model.pkl"),
        help="Ruta del champion actual solo para trazabilidad.",
    )
    parser.add_argument(
        "--baseline-metadata-path",
        type=Path,
        default=Path("models/public/baseline/metadata.json"),
        help="Ruta de metadata del champion actual para resolver cutoff oficial.",
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("artifacts/public/metrics/final_baseline_vs_candidates.csv"),
        help="Registro oficial baseline vs candidatos.",
    )
    parser.add_argument(
        "--rules-csv",
        type=Path,
        default=Path("config/business_blocklist_rules.csv"),
        help="Reglas deterministas para la evaluación secundaria con policy Day03.",
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
        default=Path("models/candidates/day04_v3_a_lr_smote_0_5_v1"),
        help="Directorio donde persistir el artefacto del candidato Day 04.",
    )
    parser.add_argument(
        "--day-id",
        type=str,
        default="Day 04",
        help="Identificador day para trazabilidad.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k de scoring base del modelo.",
    )
    parser.add_argument(
        "--albaran-policy",
        type=str,
        default="PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009",
        choices=["PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009"],
        help="Policy de coherencia a evaluar como comparativa secundaria.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="",
        help="Cutoff opcional. Si vacío, se usa el del baseline oficial.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id opcional para reproducibilidad externa.",
    )
    return parser.parse_args()


# SECTION: Shared helpers
def _load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


# SECTION: Shared helpers
def _resolve_cutoff_date(metadata_path: Path, cutoff_override: str) -> str:
    """Resolve the official cutoff date using CLI override or baseline metadata."""
    if cutoff_override.strip():
        return cutoff_override.strip()
    metadata = _load_json(metadata_path)
    cutoff_date = str(metadata.get("cutoff_date", "")).strip()
    if cutoff_date == "":
        raise ValueError(f"No se pudo resolver cutoff_date desde {metadata_path}")
    return cutoff_date


# SECTION: Shared helpers
def _resolve_feature_columns(dataset_df: pd.DataFrame) -> tuple[list[str], list[str], str]:
    """Resolve the final numeric/categorical/target columns for the V3_A dataset."""
    feature_cols_num, feature_cols_cat, target_col = get_feature_columns_v3()
    missing_flags = sorted([column for column in dataset_df.columns if column.endswith("_missing_flag")])
    return feature_cols_num + missing_flags, feature_cols_cat, target_col


# SECTION: Data preparation
def _prepare_model_frame(
    dataset_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Build the model dataframe while keeping the original scoring/audit columns intact."""
    required = {"event_id", "fecha_evento", target_col, "proveedor_candidato", "producto_canonico", "terminal_compra"}
    missing = required - set(dataset_df.columns)
    if missing:
        raise ValueError(f"Dataset V3_A incompatible. Faltan columnas requeridas: {sorted(missing)}")

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
    """Split the dataset using the official fixed cutoff date."""
    cutoff_dt = pd.to_datetime(cutoff_date, errors="raise")
    train_df = dataset_df[dataset_df["fecha_evento"] <= cutoff_dt].copy()
    test_df = dataset_df[dataset_df["fecha_evento"] > cutoff_dt].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Split temporal inválido para Day 04. "
            f"train_rows={len(train_df)} test_rows={len(test_df)} cutoff={cutoff_date}"
        )
    return train_df, test_df


# SECTION: Data preparation
def _build_train_test_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Dummify train/test separately and align both matrices."""
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


# SECTION: Metrics payloads
def _build_pure_metrics_payload(
    *,
    run_id: str,
    ts_utc: str,
    day_id: str,
    cutoff_date: str,
    dataset_path: Path,
    metrics: dict[str, float | int],
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    baseline_model_path: Path,
    baseline_metadata_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    """Build the JSON payload for the pure Day 04 model candidate."""
    return {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": day_id,
        "scope": "model_retrain",
        "eval_scope": "day02_test_like",
        "cutoff_applied": cutoff_date,
        "model_variant": PURE_MODEL_VARIANT,
        "metrics": metrics,
        "feature_contract": {
            "feature_columns_num": feature_cols_num,
            "feature_columns_cat": feature_cols_cat,
        },
        "sources": {
            "dataset_csv": str(dataset_path),
            "baseline_model_path": str(baseline_model_path),
            "baseline_metadata_path": str(baseline_metadata_path),
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
        },
    }


# SECTION: Metrics payloads
def _build_policy_metrics_payload(
    *,
    run_id: str,
    ts_utc: str,
    day_id: str,
    cutoff_date: str,
    policy: str,
    dataset_path: Path,
    model_metrics: dict[str, float | int],
    policy_metrics: dict[str, float | int],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    """Build the JSON payload for the secondary Day 04 policy comparison."""
    return {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": day_id,
        "scope": "after_policy",
        "eval_scope": "day02_test_like",
        "cutoff_applied": cutoff_date,
        "policy": policy,
        "model_variant": POLICY_MODEL_VARIANT,
        "metrics": {
            "accuracy": float(model_metrics["accuracy"]),
            "balanced_accuracy": float(model_metrics["balanced_accuracy"]),
            "f1_pos": float(model_metrics["f1_pos"]),
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
            "dataset_csv": str(dataset_path),
            **source_paths,
        },
    }


# SECTION: Registry
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
    """Append a candidate row to the official registry using the current metrics payload."""
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


# SECTION: Registry
def _registry_has_row(registry_csv: Path, run_id: str, model_variant: str) -> bool:
    """Return whether the registry already contains a row for the given run/model pair."""
    rows = registry.read_registry_rows(registry_csv)
    return any(row.get("run_id") == run_id and row.get("model_variant") == model_variant for row in rows)


# SECTION: Policy evaluation
def _build_scored_holdout(
    test_df: pd.DataFrame,
    eval_frame: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Reattach scores and predictions to the full holdout rows for post-inference evaluation."""
    scored_df = test_df.reset_index(drop=True).copy()
    scored_df["score_model"] = pd.to_numeric(eval_frame["score_model"], errors="coerce").fillna(0.0).to_numpy()
    scored_df["pred_label"] = (
        pd.to_numeric(eval_frame["pred_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    )
    scored_df["rank_event_score"] = (
        scored_df.groupby("event_id")["score_model"].rank(method="first", ascending=False).astype(int)
    )
    scored_df["is_top1"] = (scored_df["rank_event_score"] == 1).astype(int)
    scored_df["is_topk"] = (scored_df["rank_event_score"] <= top_k).astype(int)
    scored_df["fecha_evento"] = scored_df["fecha_evento"].dt.strftime("%Y-%m-%d")
    return scored_df


# SECTION: Main pipeline
def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full Day 04 pipeline: train V3_A, evaluate, register, and persist artifacts."""
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = (project_root / args.dataset_csv).resolve()
    baseline_model_path = (project_root / args.baseline_model_path).resolve()
    baseline_metadata_path = (project_root / args.baseline_metadata_path).resolve()
    registry_csv = (project_root / args.registry_csv).resolve()
    rules_csv = (project_root / args.rules_csv).resolve()
    reports_root = (project_root / args.reports_root).resolve()
    base_candidate_model_dir = (project_root / args.candidate_model_dir).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe dataset V3_A: {dataset_path}")
    if not baseline_metadata_path.exists():
        raise FileNotFoundError(f"No existe metadata baseline: {baseline_metadata_path}")
    if not registry_csv.exists():
        raise FileNotFoundError(f"No existe registro oficial: {registry_csv}")

    run_id = args.run_id.strip() or build_run_id()
    ts_utc = utc_now_iso()
    run_date = run_id[:8]
    candidate_model_dir = base_candidate_model_dir / run_id
    cutoff_date = _resolve_cutoff_date(metadata_path=baseline_metadata_path, cutoff_override=args.cutoff_date)

    dataset_df = pd.read_csv(dataset_path, keep_default_na=False)
    feature_cols_num, feature_cols_cat, target_col = _resolve_feature_columns(dataset_df=dataset_df)
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
    if "LR_smote_0.5" not in variant_map:
        raise ValueError("No se encontró la receta LR_smote_0.5 en build_default_lr_balance_variants().")

    pure_model, pure_sampler = variant_map["LR_smote_0.5"]
    pure_row, pure_result = fc.evaluate_balance_variant(
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

    pure_metrics = {
        "accuracy": float(pure_row["acc"]),
        "balanced_accuracy": float(pure_row["bal_acc"]),
        "f1_pos": float(pure_row["f1_pos"]),
        "top1_hit": float(pure_row["top1_hit"]),
        "top2_hit": float(pure_row["top2_hit"]),
        "coverage": 1.0,
        "test_events": int(test_df["event_id"].nunique()),
    }

    selection_rule = (
        "fixed_recipe(LR_smote_0.5 on V3_A) -> registry_gate("
        "top2>=baseline+0.01 & bal_acc>=baseline+0.01 & coverage>=baseline-0.005)"
    )
    model_path, metadata_path, _ = fc.save_champion_artifacts(
        model=pure_result["model"],
        model_dir=candidate_model_dir,
        model_name=PURE_MODEL_VARIANT,
        metrics=pure_metrics,
        cutoff_date=cutoff_date,
        dataset_name=dataset_path.name,
        feature_columns=list(X_train.columns),
        selection_rule=selection_rule,
    )

    metrics_dir = reports_root / "metrics" / "candidates" / run_date
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pure_metrics_path = metrics_dir / f"{run_id}_{PURE_MODEL_VARIANT}_metrics.json"
    pure_summary_path = metrics_dir / f"{run_id}_{PURE_MODEL_VARIANT}_run_summary.json"

    pure_payload = _build_pure_metrics_payload(
        run_id=run_id,
        ts_utc=ts_utc,
        day_id=args.day_id,
        cutoff_date=cutoff_date,
        dataset_path=dataset_path,
        metrics=pure_metrics,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        baseline_model_path=baseline_model_path,
        baseline_metadata_path=baseline_metadata_path,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    write_json(pure_metrics_path, pure_payload)
    if not _registry_has_row(registry_csv=registry_csv, run_id=run_id, model_variant=PURE_MODEL_VARIANT):
        _append_registry_row(
            registry_csv=registry_csv,
            run_id=run_id,
            day_id=args.day_id,
            model_variant=PURE_MODEL_VARIANT,
            metadata_path=metadata_path,
            metrics_json_path=pure_metrics_path,
            dataset_csv=dataset_path,
            model_path=model_path,
            promotion_decision="auto",
        )

    scored_df = _build_scored_holdout(
        test_df=test_df,
        eval_frame=pure_result["eval_frame"],
        top_k=args.top_k,
    )
    audit_dir = reports_root / "postinferencia" / "audits" / run_date
    audit_dir.mkdir(parents=True, exist_ok=True)
    scored_input_path = audit_dir / f"day04_v3a_scored_input_{run_id}.csv"
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
    policy_metrics = compute_postinference_metrics(
        detail_df=detail_df,
        resumen_df=resumen_df,
        resumen_albaran_df=resumen_albaran_df,
    )

    policy_metrics_path = metrics_dir / f"{run_id}_{POLICY_MODEL_VARIANT}_metrics.json"
    policy_summary_path = metrics_dir / f"{run_id}_{POLICY_MODEL_VARIANT}_run_summary.json"
    policy_payload = _build_policy_metrics_payload(
        run_id=run_id,
        ts_utc=ts_utc,
        day_id=args.day_id,
        cutoff_date=cutoff_date,
        policy=args.albaran_policy,
        dataset_path=dataset_path,
        model_metrics=pure_metrics,
        policy_metrics=policy_metrics,
        source_paths={
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "scored_input_csv": str(scored_input_path),
            "detail_csv": str(audit_paths["detail"]),
            "resumen_evento_csv": str(audit_paths["resumen_evento"]),
            "resumen_albaran_csv": str(audit_paths["resumen_albaran"]),
            "summary_json": str(audit_paths["summary"]),
        },
    )
    write_json(policy_metrics_path, policy_payload)
    if not _registry_has_row(registry_csv=registry_csv, run_id=run_id, model_variant=POLICY_MODEL_VARIANT):
        _append_registry_row(
            registry_csv=registry_csv,
            run_id=run_id,
            day_id=args.day_id,
            model_variant=POLICY_MODEL_VARIANT,
            metadata_path=metadata_path,
            metrics_json_path=policy_metrics_path,
            dataset_csv=dataset_path,
            model_path=model_path,
            promotion_decision="keep_baseline",
        )

    pure_registry_rows = registry.read_registry_rows(registry_csv)
    pure_registry_row = next(
        row
        for row in pure_registry_rows
        if row.get("run_id") == run_id and row.get("model_variant") == PURE_MODEL_VARIANT
    )
    policy_registry_row = next(
        row
        for row in pure_registry_rows
        if row.get("run_id") == run_id and row.get("model_variant") == POLICY_MODEL_VARIANT
    )

    pure_summary = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": args.day_id,
        "model_variant": PURE_MODEL_VARIANT,
        "cutoff_applied": cutoff_date,
        "metrics": pure_metrics,
        "registry_row": pure_registry_row,
        "artifact_paths": {
            "dataset_csv": str(dataset_path),
            "baseline_model_path": str(baseline_model_path),
            "baseline_metadata_path": str(baseline_metadata_path),
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "metrics_json": str(pure_metrics_path),
        },
    }
    policy_summary = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": args.day_id,
        "model_variant": POLICY_MODEL_VARIANT,
        "cutoff_applied": cutoff_date,
        "metrics": policy_payload["metrics"],
        "registry_row": policy_registry_row,
        "engine_summary": engine_summary,
        "artifact_paths": {
            "metrics_json": str(policy_metrics_path),
            "detail_csv": str(audit_paths["detail"]),
            "resumen_evento_csv": str(audit_paths["resumen_evento"]),
            "resumen_albaran_csv": str(audit_paths["resumen_albaran"]),
            "summary_json": str(audit_paths["summary"]),
        },
    }
    write_json(pure_summary_path, pure_summary)
    write_json(policy_summary_path, policy_summary)

    return {
        "status": "ok",
        "run_id": run_id,
        "pure_summary_path": str(pure_summary_path),
        "policy_summary_path": str(policy_summary_path),
        "pure_metrics_path": str(pure_metrics_path),
        "policy_metrics_path": str(policy_metrics_path),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }


# SECTION: CLI entrypoint
def main() -> None:
    """Run the CLI entrypoint for the Day 04 V3_A training pipeline."""
    args = parse_args()
    summary = run_pipeline(args=args)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
