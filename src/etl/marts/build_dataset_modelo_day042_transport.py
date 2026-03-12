#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.etl.marts import build_dataset_modelo_day041_ablation as day041_builder
    from src.etl.marts import build_dataset_modelo_v3_context as day04_v3
    from src.etl.transform import rebuild_ofertas_transport_signals_day042 as day042_rebuild
    from src.ml.shared import functions as fc
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.etl.marts import build_dataset_modelo_day041_ablation as day041_builder
    from src.etl.marts import build_dataset_modelo_v3_context as day04_v3
    from src.etl.transform import rebuild_ofertas_transport_signals_day042 as day042_rebuild
    from src.ml.shared import functions as fc

TRANSPORT_REBUILT_VARIANT = "transport_rebuilt_only"
DISPERSION_PLUS_TRANSPORT_VARIANT = "dispersion_plus_transport_rebuilt"

TRANSPORT_REBUILT_COLUMNS = [
    "v42_transport_cost_min_day_provider",
    "v42_transport_cost_mean_day_provider",
    "v42_transport_cost_range_day_provider",
    "v42_transport_observations",
    "v42_transport_unique_terminal_count",
    "v42_transport_multi_terminal_share",
    "v42_transport_rank_event",
    "v42_transport_gap_vs_min_event",
    "v42_transport_ratio_vs_min_event",
    "v42_transport_source_raw_explicit",
    "v42_transport_source_parser_fix",
    "v42_transport_source_deterministic_rebuild",
    "v42_transport_source_heuristic_same_month",
    "v42_transport_source_missing",
    "v42_transport_imputed_flag",
    "v42_transport_days_gap",
    "v42_transport_lookahead_flag",
]

VARIANT_OUTPUTS = {
    TRANSPORT_REBUILT_VARIANT: "dataset_modelo_v2_transport_rebuilt_only.csv",
    DISPERSION_PLUS_TRANSPORT_VARIANT: "dataset_modelo_v2_dispersion_plus_transport_rebuilt.csv",
}


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 04.2 mart builder."""
    parser = argparse.ArgumentParser(
        description="Construye datasets Day 04.2 con transporte reconstruido y gate previo a entrenamiento."
    )
    parser.add_argument(
        "--v2-input",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v2_candidates.csv"),
        help="Input base V2.",
    )
    parser.add_argument(
        "--ofertas-typed-input",
        type=Path,
        default=Path("data/public/support/ofertas_typed.csv"),
        help="Input staging de ofertas tipadas para dispersion Day04.",
    )
    parser.add_argument(
        "--transport-rebuilt-input",
        type=Path,
        default=Path("data/public/support/ofertas_transport_signals_day042.csv"),
        help="Output staging Day 04.2 con transporte reconstruido.",
    )
    parser.add_argument(
        "--missingness-report",
        type=Path,
        default=Path("artifacts/public/transport_missingness_day042.json"),
        help="Reporte de missingness Day 04.2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/public/day042"),
        help="Directorio de salida para datasets Day 04.2.",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("artifacts/public/data_quality_day042_transport_matrix.json"),
        help="Reporte de calidad/gates Day 04.2.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2028-02-21",
        help="Cutoff temporal oficial.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.80,
        help="Cobertura minima requerida para permitir entrenamiento Day 04.2.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id opcional.",
    )
    return parser.parse_args()


# SECTION: Shared helpers
def _compute_source_stage_coverage(dataset_df: pd.DataFrame, cutoff_date: str) -> dict[str, dict[str, float]]:
    """Compute transport coverage by provenance stage on train/test splits."""
    working = dataset_df.copy()
    working["fecha_evento_dt"] = pd.to_datetime(working["fecha_evento"], errors="coerce")
    cutoff_dt = pd.to_datetime(cutoff_date, errors="raise")
    train_df = working[working["fecha_evento_dt"] <= cutoff_dt].copy()
    test_df = working[working["fecha_evento_dt"] > cutoff_dt].copy()

    stage_masks = {
        "raw_explicit": working["transport_source_kind"].eq(day042_rebuild.RAW_EXPLICIT_KIND),
        "parser_fix_plus_deterministic": working["transport_source_kind"].isin(
            [
                day042_rebuild.RAW_EXPLICIT_KIND,
                day042_rebuild.PARSER_FIX_KIND,
                day042_rebuild.DETERMINISTIC_KIND,
            ]
        ),
        "final_after_heuristic": working["transport_source_kind"].ne(day042_rebuild.MISSING_KIND),
        "final_after_heuristic_no_lookahead": working["transport_source_kind"].isin(
            [
                day042_rebuild.RAW_EXPLICIT_KIND,
                day042_rebuild.PARSER_FIX_KIND,
                day042_rebuild.DETERMINISTIC_KIND,
            ]
        )
        | (
            working["transport_source_kind"].eq(day042_rebuild.HEURISTIC_KIND)
            & working["transport_lookahead_flag"].fillna(0).eq(0)
        ),
    }
    coverage: dict[str, dict[str, float]] = {}
    for stage_name, mask in stage_masks.items():
        coverage[stage_name] = {
            "coverage_train": float(mask.loc[train_df.index].mean()) if not train_df.empty else 0.0,
            "coverage_test": float(mask.loc[test_df.index].mean()) if not test_df.empty else 0.0,
        }
    return coverage


# SECTION: Shared helpers
def _build_transport_day_frame(transport_rebuilt_df: pd.DataFrame) -> pd.DataFrame:
    """Project the rebuilt Day 04.2 staging into transport features at day-product-provider grain."""
    working = transport_rebuilt_df.copy()
    working["fecha_evento"] = pd.to_datetime(working["fecha_oferta"], errors="coerce").dt.strftime("%Y-%m-%d")
    numeric_columns = [
        "transport_cost_value",
        "transport_cost_mean_day_provider",
        "transport_cost_range_day_provider",
        "transport_observations",
        "transport_unique_terminal_count",
        "transport_multi_terminal_share",
        "transport_days_gap",
    ]
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["transport_lookahead_flag"] = pd.to_numeric(
        working["transport_lookahead_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    working["transport_imputed_flag"] = pd.to_numeric(
        working["transport_imputed_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    working["transport_source_kind"] = working["transport_source_kind"].astype("string").fillna(day042_rebuild.MISSING_KIND)

    working["v42_transport_cost_min_day_provider"] = working["transport_cost_value"]
    working["v42_transport_cost_mean_day_provider"] = working["transport_cost_mean_day_provider"]
    working["v42_transport_cost_range_day_provider"] = working["transport_cost_range_day_provider"]
    working["v42_transport_observations"] = working["transport_observations"]
    working["v42_transport_unique_terminal_count"] = working["transport_unique_terminal_count"]
    working["v42_transport_multi_terminal_share"] = working["transport_multi_terminal_share"]
    working["v42_transport_source_raw_explicit"] = working["transport_source_kind"].eq(day042_rebuild.RAW_EXPLICIT_KIND).astype(int)
    working["v42_transport_source_parser_fix"] = working["transport_source_kind"].eq(day042_rebuild.PARSER_FIX_KIND).astype(int)
    working["v42_transport_source_deterministic_rebuild"] = (
        working["transport_source_kind"].eq(day042_rebuild.DETERMINISTIC_KIND).astype(int)
    )
    working["v42_transport_source_heuristic_same_month"] = (
        working["transport_source_kind"].eq(day042_rebuild.HEURISTIC_KIND).astype(int)
    )
    working["v42_transport_source_missing"] = working["transport_source_kind"].eq(day042_rebuild.MISSING_KIND).astype(int)
    working["v42_transport_imputed_flag"] = working["transport_imputed_flag"]
    working["v42_transport_days_gap"] = working["transport_days_gap"]
    working["v42_transport_lookahead_flag"] = working["transport_lookahead_flag"]

    day_level_columns = [
        "fecha_evento",
        "producto_canonico",
        "proveedor_candidato",
        "transport_source_kind",
        "v42_transport_cost_min_day_provider",
        "v42_transport_cost_mean_day_provider",
        "v42_transport_cost_range_day_provider",
        "v42_transport_observations",
        "v42_transport_unique_terminal_count",
        "v42_transport_multi_terminal_share",
        "v42_transport_source_raw_explicit",
        "v42_transport_source_parser_fix",
        "v42_transport_source_deterministic_rebuild",
        "v42_transport_source_heuristic_same_month",
        "v42_transport_source_missing",
        "v42_transport_imputed_flag",
        "v42_transport_days_gap",
        "v42_transport_lookahead_flag",
    ]
    return working[day_level_columns].copy()


# SECTION: Shared helpers
def _build_transport_event_frame(dataset_v2: pd.DataFrame, transport_day_frame: pd.DataFrame) -> pd.DataFrame:
    """Project day-level rebuilt transport features back to candidate-event rows and add event ranking features."""
    merged = dataset_v2[
        ["event_id", "fecha_evento", "producto_canonico", "proveedor_candidato"]
    ].merge(
        transport_day_frame,
        on=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        how="left",
    )
    merged["v42_transport_cost_min_day_provider"] = pd.to_numeric(
        merged["v42_transport_cost_min_day_provider"],
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []
    for event_id, group in merged.groupby("event_id", sort=False):
        valid_costs = group["v42_transport_cost_min_day_provider"].dropna().astype(float)
        min_cost = float(valid_costs.min()) if not valid_costs.empty else np.nan
        rank_lookup = valid_costs.rank(method="dense", ascending=True).to_dict() if not valid_costs.empty else {}
        for row_index, row in group.iterrows():
            current_cost = row.get("v42_transport_cost_min_day_provider", np.nan)
            if pd.isna(current_cost) or pd.isna(min_cost):
                transport_rank = np.nan
                transport_gap = np.nan
                transport_ratio = np.nan
            else:
                transport_rank = float(rank_lookup.get(row_index, np.nan))
                transport_gap = float(current_cost - min_cost)
                transport_ratio = float(current_cost / min_cost) if min_cost not in {0.0, np.nan} else np.nan
            rows.append(
                {
                    "event_id": str(event_id),
                    "proveedor_candidato": str(row["proveedor_candidato"]),
                    "transport_source_kind": row.get("transport_source_kind", day042_rebuild.MISSING_KIND),
                    "v42_transport_cost_min_day_provider": row.get("v42_transport_cost_min_day_provider", np.nan),
                    "v42_transport_cost_mean_day_provider": row.get("v42_transport_cost_mean_day_provider", np.nan),
                    "v42_transport_cost_range_day_provider": row.get("v42_transport_cost_range_day_provider", np.nan),
                    "v42_transport_observations": row.get("v42_transport_observations", np.nan),
                    "v42_transport_unique_terminal_count": row.get("v42_transport_unique_terminal_count", np.nan),
                    "v42_transport_multi_terminal_share": row.get("v42_transport_multi_terminal_share", np.nan),
                    "v42_transport_rank_event": transport_rank,
                    "v42_transport_gap_vs_min_event": transport_gap,
                    "v42_transport_ratio_vs_min_event": transport_ratio,
                    "v42_transport_source_raw_explicit": row.get("v42_transport_source_raw_explicit", 0),
                    "v42_transport_source_parser_fix": row.get("v42_transport_source_parser_fix", 0),
                    "v42_transport_source_deterministic_rebuild": row.get(
                        "v42_transport_source_deterministic_rebuild",
                        0,
                    ),
                    "v42_transport_source_heuristic_same_month": row.get(
                        "v42_transport_source_heuristic_same_month",
                        0,
                    ),
                    "v42_transport_source_missing": row.get("v42_transport_source_missing", 1),
                    "v42_transport_imputed_flag": row.get("v42_transport_imputed_flag", 0),
                    "v42_transport_days_gap": row.get("v42_transport_days_gap", np.nan),
                    "v42_transport_lookahead_flag": row.get("v42_transport_lookahead_flag", 0),
                }
            )
    return pd.DataFrame(rows)


# SECTION: Shared helpers
def _compute_added_feature_coverage(
    dataset_df: pd.DataFrame,
    feature_columns: list[str],
    cutoff_date: str,
) -> dict[str, dict[str, float]]:
    """Compute train/test non-null coverage for added numeric features."""
    return day041_builder._compute_feature_coverage(
        dataset_df=dataset_df,
        feature_columns=feature_columns,
        cutoff_date=cutoff_date,
    )


# SECTION: Dataset building
def _build_variant_dataset(
    *,
    dataset_v2: pd.DataFrame,
    variant_name: str,
    transport_event_frame: pd.DataFrame,
    dispersion_frame: pd.DataFrame,
    cutoff_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one Day 04.2 dataset variant and return its quality summary."""
    dataset = dataset_v2.merge(
        transport_event_frame,
        on=["event_id", "proveedor_candidato"],
        how="left",
    )
    added_columns = list(TRANSPORT_REBUILT_COLUMNS)
    if variant_name == DISPERSION_PLUS_TRANSPORT_VARIANT:
        dataset = dataset.merge(
            dispersion_frame,
            on=["fecha_evento", "producto_canonico", "proveedor_candidato"],
            how="left",
        )
        added_columns.extend(day04_v3.V3_SIGNAL_FAMILIES[day04_v3.DISPERSION_FAMILY])

    feature_coverage = _compute_added_feature_coverage(
        dataset_df=dataset,
        feature_columns=added_columns,
        cutoff_date=cutoff_date,
    )
    missing_flag_columns: list[str] = []
    for column in added_columns:
        if dataset[column].isna().any():
            flag_column = f"{column}_missing_flag"
            dataset[flag_column] = dataset[column].isna().astype(int)
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce").fillna(0.0)
            missing_flag_columns.append(flag_column)
        else:
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    positive_per_event = dataset.groupby("event_id")["target_elegido"].sum()
    summary = {
        "variant_name": variant_name,
        "dataset_name": VARIANT_OUTPUTS[variant_name],
        "rows_output": int(len(dataset)),
        "events_output": int(dataset["event_id"].nunique()),
        "rows_vs_v2_match": int(len(dataset) == len(dataset_v2)),
        "events_vs_v2_match": int(dataset["event_id"].nunique() == dataset_v2["event_id"].nunique()),
        "events_with_invalid_positive_count": int((positive_per_event != 1).sum()) if len(positive_per_event) else 0,
        "added_columns": added_columns,
        "missing_flag_columns": missing_flag_columns,
        "feature_coverage": feature_coverage,
    }
    return dataset, summary


# SECTION: Main pipeline
def run(
    *,
    v2_input_path: Path,
    ofertas_typed_input_path: Path,
    transport_rebuilt_input_path: Path,
    missingness_report_path: Path,
    output_dir: Path,
    quality_report_path: Path,
    cutoff_date: str,
    min_coverage: float,
    run_id: str,
) -> dict[str, Any]:
    """Build Day 04.2 datasets, compute stage coverage, and decide if training is allowed."""
    execution_run_id = run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_ts_utc = datetime.now(timezone.utc).isoformat()

    dataset_v2 = pd.read_csv(v2_input_path, keep_default_na=False)
    ofertas_typed = pd.read_csv(ofertas_typed_input_path, keep_default_na=False)
    transport_rebuilt = pd.read_csv(transport_rebuilt_input_path, keep_default_na=False)
    missingness_report = json.loads(missingness_report_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)

    transport_day_frame = _build_transport_day_frame(transport_rebuilt_df=transport_rebuilt)
    transport_event_frame = _build_transport_event_frame(
        dataset_v2=dataset_v2,
        transport_day_frame=transport_day_frame,
    )
    dispersion_frame = day04_v3._build_source_quality_frame(ofertas_typed=ofertas_typed)[
        ["fecha_evento", "producto_canonico", "proveedor_candidato"] + day04_v3.V3_SIGNAL_FAMILIES[day04_v3.DISPERSION_FAMILY]
    ].copy()

    stage_probe = dataset_v2[
        ["event_id", "fecha_evento", "producto_canonico", "proveedor_candidato", "target_elegido"]
    ].merge(
        transport_rebuilt[
            [
                "fecha_oferta",
                "producto_canonico",
                "proveedor_candidato",
                "transport_source_kind",
                "transport_lookahead_flag",
            ]
        ].rename(columns={"fecha_oferta": "fecha_evento"}),
        on=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        how="left",
    )
    stage_probe["transport_source_kind"] = stage_probe["transport_source_kind"].fillna(day042_rebuild.MISSING_KIND)
    stage_probe["transport_lookahead_flag"] = pd.to_numeric(
        stage_probe["transport_lookahead_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    stage_coverage = _compute_source_stage_coverage(dataset_df=stage_probe, cutoff_date=cutoff_date)
    lookahead_rows_v2 = int(
        stage_probe.loc[stage_probe["transport_source_kind"].eq(day042_rebuild.HEURISTIC_KIND), "transport_lookahead_flag"].sum()
    )
    lookahead_unique_keys = int(
        transport_rebuilt.loc[
            transport_rebuilt["transport_source_kind"].eq(day042_rebuild.HEURISTIC_KIND),
            "transport_lookahead_flag",
        ].sum()
    )

    variants_summary: dict[str, Any] = {}
    for variant_name in [TRANSPORT_REBUILT_VARIANT, DISPERSION_PLUS_TRANSPORT_VARIANT]:
        dataset_variant, summary = _build_variant_dataset(
            dataset_v2=dataset_v2,
            variant_name=variant_name,
            transport_event_frame=transport_event_frame,
            dispersion_frame=dispersion_frame,
            cutoff_date=cutoff_date,
        )
        output_path = output_dir / VARIANT_OUTPUTS[variant_name]
        dataset_variant.to_csv(output_path, index=False, encoding="utf-8")
        summary["dataset_path"] = str(output_path)
        summary["status"] = "built"
        variants_summary[variant_name] = summary

    coverage_gate_pass = (
        stage_coverage["final_after_heuristic"]["coverage_train"] >= min_coverage
        and stage_coverage["final_after_heuristic"]["coverage_test"] >= min_coverage
    )
    leakage_gate_pass = lookahead_rows_v2 == 0
    structural_gate_pass = all(
        summary["rows_vs_v2_match"] == 1
        and summary["events_vs_v2_match"] == 1
        and summary["events_with_invalid_positive_count"] == 0
        for summary in variants_summary.values()
    )
    training_allowed = bool(coverage_gate_pass and leakage_gate_pass and structural_gate_pass)

    quality_report = {
        "status": "ok",
        "run_id": execution_run_id,
        "generated_ts_utc": generated_ts_utc,
        "cutoff_date": cutoff_date,
        "min_coverage": float(min_coverage),
        "input_v2_dataset": str(v2_input_path),
        "input_ofertas_typed": str(ofertas_typed_input_path),
        "input_transport_rebuilt": str(transport_rebuilt_input_path),
        "input_missingness_report": str(missingness_report_path),
        "output_dir": str(output_dir),
        "variants": variants_summary,
        "stage_coverage": stage_coverage,
        "lookahead": {
            "heuristic_unique_keys_with_lookahead": lookahead_unique_keys,
            "heuristic_v2_rows_with_lookahead": lookahead_rows_v2,
        },
        "gate": {
            "coverage_gate_pass": bool(coverage_gate_pass),
            "leakage_gate_pass": bool(leakage_gate_pass),
            "structural_gate_pass": bool(structural_gate_pass),
            "training_allowed": bool(training_allowed),
            "coverage_train_final": float(stage_coverage["final_after_heuristic"]["coverage_train"]),
            "coverage_test_final": float(stage_coverage["final_after_heuristic"]["coverage_test"]),
            "coverage_train_final_no_lookahead": float(
                stage_coverage["final_after_heuristic_no_lookahead"]["coverage_train"]
            ),
            "coverage_test_final_no_lookahead": float(
                stage_coverage["final_after_heuristic_no_lookahead"]["coverage_test"]
            ),
            "failure_reasons": [
                reason
                for reason, failed in [
                    ("coverage_below_min_threshold", not coverage_gate_pass),
                    ("heuristic_lookahead_detected", not leakage_gate_pass),
                    ("dataset_contract_mismatch", not structural_gate_pass),
                ]
                if failed
            ],
        },
        "transport_missingness_summary": {
            "bucket_counts_v2_rows": missingness_report.get("bucket_counts_v2_rows", {}),
            "no_raw_subreason_counts_v2_rows": missingness_report.get("no_raw_subreason_counts_v2_rows", {}),
        },
        "base_feature_columns_num": fc.get_feature_columns_v2()[0],
        "base_feature_columns_cat": fc.get_feature_columns_v2()[1],
        "target_column": fc.get_feature_columns_v2()[2],
    }
    quality_report_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return quality_report


# SECTION: CLI entrypoint
def main() -> None:
    """Run the CLI entrypoint for the Day 04.2 dataset builder."""
    args = parse_args()
    summary = run(
        v2_input_path=args.v2_input,
        ofertas_typed_input_path=args.ofertas_typed_input,
        transport_rebuilt_input_path=args.transport_rebuilt_input,
        missingness_report_path=args.missingness_report,
        output_dir=args.output_dir,
        quality_report_path=args.quality_report,
        cutoff_date=args.cutoff_date,
        min_coverage=args.min_coverage,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
