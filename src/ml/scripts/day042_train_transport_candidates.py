#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from src.etl.marts import build_dataset_modelo_day042_transport as day042_builder
    from src.ml.metrics import registry
    from src.ml.scripts import day041_train_ablation_candidates as day041_train
    from src.ml.shared.helpers import build_run_id, utc_now_iso, write_json
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.etl.marts import build_dataset_modelo_day042_transport as day042_builder
    from src.ml.metrics import registry
    from src.ml.scripts import day041_train_ablation_candidates as day041_train
    from src.ml.shared.helpers import build_run_id, utc_now_iso, write_json

VARIANT_NAME_MAP = {
    day042_builder.TRANSPORT_REBUILT_VARIANT: "V2_TRANSPORT_REBUILT_ONLY_LR_smote_0.5_v1",
    day042_builder.DISPERSION_PLUS_TRANSPORT_VARIANT: "V2_DISPERSION_PLUS_TRANSPORT_REBUILT_LR_smote_0.5_v1",
}


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 04.2 training pipeline."""
    parser = argparse.ArgumentParser(
        description="Entrena variantes Day 04.2 solo si el gate de cobertura/leakage lo permite."
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("artifacts/public/data_quality_day042_transport_matrix.json"),
        help="Reporte de calidad Day 04.2 generado por el builder.",
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
        help="Directorio raiz de reportes.",
    )
    parser.add_argument(
        "--candidate-model-dir",
        type=Path,
        default=Path("models/candidates/day042_transport"),
        help="Directorio raiz para artefactos Day 04.2.",
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
        help="Reglas deterministas para evaluacion secundaria.",
    )
    parser.add_argument(
        "--day-id",
        type=str,
        default="Day 04.2",
        help="Identificador day para trazabilidad.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k de scoring para metricas por evento.",
    )
    parser.add_argument(
        "--albaran-policy",
        type=str,
        default="PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009",
        choices=["PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009"],
        help="Policy Day03 para la evaluacion secundaria.",
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
    """Build a filesystem-friendly slug from the variant name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", model_variant).strip("_").lower()


# SECTION: Shared helpers
def _summary_output_path(reports_root: Path, run_id: str) -> Path:
    """Build the persisted Day 04.2 run summary path consumed by notebook 13."""
    return reports_root / "metrics" / "candidates" / run_id[:8] / f"{run_id}_day042_transport_run_summary.json"


# SECTION: Shared helpers
def _variant_features_from_quality(
    quality_payload: dict[str, Any],
    variant_name: str,
) -> tuple[list[str], list[str], str]:
    """Resolve numeric/categorical/target columns for one Day 04.2 variant."""
    base_num = list(quality_payload["base_feature_columns_num"])
    base_cat = list(quality_payload["base_feature_columns_cat"])
    target_col = str(quality_payload["target_column"])
    variant_info = quality_payload["variants"][variant_name]
    feature_cols_num = base_num + list(variant_info["added_columns"]) + list(variant_info.get("missing_flag_columns", []))
    return feature_cols_num, base_cat, target_col


# SECTION: Shared helpers
def _baseline_metrics_from_registry(registry_csv: Path) -> dict[str, float]:
    """Load baseline metrics to report Day 04.2 deltas in the run summary."""
    rows = registry.read_registry_rows(registry_csv)
    baseline_row = next(row for row in rows if row.get("model_role") == "baseline")
    return {
        "top2_hit": float(baseline_row["top2_hit"]),
        "balanced_accuracy": float(baseline_row["balanced_accuracy"]),
        "top1_hit": float(baseline_row["top1_hit"]),
    }


# SECTION: Main pipeline
def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Train Day 04.2 candidates only when the dataset gate allows it."""
    project_root = Path(__file__).resolve().parents[3]
    quality_report_path = (project_root / args.quality_report).resolve()
    registry_csv = (project_root / args.registry_csv).resolve()
    reports_root = (project_root / args.reports_root).resolve()
    candidate_model_root = (project_root / args.candidate_model_dir).resolve()
    baseline_metadata_path = (project_root / args.baseline_metadata_path).resolve()
    rules_csv = (project_root / args.rules_csv).resolve()

    run_id = args.run_id.strip() or build_run_id()
    ts_utc = utc_now_iso()
    quality_payload = _load_json(quality_report_path)
    gate_payload = quality_payload["gate"]
    cutoff_date = day041_train._resolve_cutoff_date(metadata_path=baseline_metadata_path)
    baseline_metrics = _baseline_metrics_from_registry(registry_csv=registry_csv)

    summary = {
        "status": "ok",
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": args.day_id,
        "gate": gate_payload,
        "baseline_metrics": baseline_metrics,
        "trained_variants": [],
        "policy_variant": "",
    }

    if not gate_payload.get("training_allowed", False):
        summary["status"] = "gated_no_train"
        summary["message"] = "Day 04.2 no entrena porque el artefacto reconstruido no supera cobertura/leakage."
        write_json(_summary_output_path(reports_root=reports_root, run_id=run_id), summary)
        return summary

    trained_results: list[dict[str, Any]] = []
    for variant_name in [day042_builder.TRANSPORT_REBUILT_VARIANT, day042_builder.DISPERSION_PLUS_TRANSPORT_VARIANT]:
        variant_info = quality_payload["variants"][variant_name]
        if variant_info.get("status") != "built":
            continue
        dataset_csv = Path(variant_info["dataset_path"])
        feature_cols_num, feature_cols_cat, target_col = _variant_features_from_quality(
            quality_payload=quality_payload,
            variant_name=variant_name,
        )
        trained_results.append(
            day041_train._train_single_variant(
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

    if not trained_results:
        summary["status"] = "gated_no_train"
        summary["message"] = "Day 04.2 no encontro variantes construidas para entrenar."
        write_json(_summary_output_path(reports_root=reports_root, run_id=run_id), summary)
        return summary

    best_result = max(
        trained_results,
        key=lambda item: (
            float(item["metrics"]["top2_hit"]),
            float(item["metrics"]["balanced_accuracy"]),
            float(item["metrics"]["f1_pos"]),
        ),
    )
    policy_result = day041_train._evaluate_policy_for_best_variant(
        best_result=best_result,
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

    summary.update(
        {
            "trained_variants": [result["model_variant"] for result in trained_results],
            "best_pure_variant": best_result["model_variant"],
            "best_pure_metrics": best_result["metrics"],
            "policy_variant": policy_result["policy_variant"],
            "transport_only_dataset": quality_payload["variants"][day042_builder.TRANSPORT_REBUILT_VARIANT]["dataset_path"],
            "transport_plus_dispersion_dataset": quality_payload["variants"][
                day042_builder.DISPERSION_PLUS_TRANSPORT_VARIANT
            ]["dataset_path"],
        }
    )
    write_json(_summary_output_path(reports_root=reports_root, run_id=run_id), summary)
    return summary


# SECTION: CLI entrypoint
def main() -> None:
    """Run the CLI entrypoint for the Day 04.2 training pipeline."""
    summary = run_pipeline(parse_args())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
