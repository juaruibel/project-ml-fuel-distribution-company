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
    from src.etl.transform import rebuild_ofertas_transport_signals_day043 as day043_rebuild
    from src.ml.shared import functions as fc
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.etl.marts import build_dataset_modelo_day041_ablation as day041_builder
    from src.etl.marts import build_dataset_modelo_v3_context as day04_v3
    from src.etl.transform import rebuild_ofertas_transport_signals_day043 as day043_rebuild
    from src.ml.shared import functions as fc

TRANSPORT_CARRY_VARIANT = "transport_carry30d_only"
DISPERSION_PLUS_TRANSPORT_VARIANT = "dispersion_plus_transport_carry30d"

TRANSPORT_CARRY_COLUMNS = [
    "v43_transport_cost_min_day_provider",
    "v43_transport_cost_mean_day_provider",
    "v43_transport_cost_range_day_provider",
    "v43_transport_observations",
    "v43_transport_unique_terminal_count",
    "v43_transport_multi_terminal_share",
    "v43_transport_rank_event",
    "v43_transport_gap_vs_min_event",
    "v43_transport_ratio_vs_min_event",
    "v43_transport_source_raw_explicit",
    "v43_transport_source_parser_fix",
    "v43_transport_source_deterministic_rebuild",
    "v43_transport_source_carry_forward_30d",
    "v43_transport_source_missing",
    "v43_transport_imputed_flag",
    "v43_transport_days_since_last_transport",
]

VARIANT_OUTPUTS = {
    TRANSPORT_CARRY_VARIANT: "dataset_modelo_v2_transport_carry30d_only.csv",
    DISPERSION_PLUS_TRANSPORT_VARIANT: "dataset_modelo_v2_dispersion_plus_transport_carry30d.csv",
}


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 04.3 mart builder."""
    parser = argparse.ArgumentParser(
        description="Construye datasets Day 04.3 con transporte carry-forward 30d y gate tecnico previo a entrenamiento."
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
        help="Input staging de ofertas tipadas para la familia dispersion.",
    )
    parser.add_argument(
        "--transport-rebuilt-input",
        type=Path,
        default=Path("data/public/support/ofertas_transport_signals_day043.csv"),
        help="Output staging Day 04.3 con carry-forward 30d.",
    )
    parser.add_argument(
        "--imputation-report",
        type=Path,
        default=Path("artifacts/public/transport_imputation_day043.json"),
        help="Reporte Day 04.3 con cobertura e imputacion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/public/day043"),
        help="Directorio de salida para datasets Day 04.3.",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("artifacts/public/data_quality_day043_transport_matrix.json"),
        help="Reporte de calidad/gates Day 04.3.",
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
        help="Cobertura minima requerida para permitir entrenamiento Day 04.3.",
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
    """Compute transport coverage by provenance stage on the official train/test split."""
    working = dataset_df.copy()
    working["fecha_evento_dt"] = pd.to_datetime(working["fecha_evento"], errors="coerce")
    cutoff_dt = pd.to_datetime(cutoff_date, errors="raise")
    train_df = working[working["fecha_evento_dt"] <= cutoff_dt].copy()
    test_df = working[working["fecha_evento_dt"] > cutoff_dt].copy()

    stage_masks = {
        "raw_explicit": working["transport_source_kind"].eq(day043_rebuild.day042_rebuild.RAW_EXPLICIT_KIND),
        "parser_fix_or_rebuild": working["transport_source_kind"].isin(list(day043_rebuild.DONOR_KINDS)),
        "final_after_carry_forward_30d": working["transport_source_kind"].ne(day043_rebuild.MISSING_KIND),
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
    """Project Day 04.3 staging to day-product-provider grain with stable transport features."""
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

    working["transport_imputed_flag"] = pd.to_numeric(
        working["transport_imputed_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    working["transport_lookahead_flag"] = pd.to_numeric(
        working["transport_lookahead_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    working["transport_source_kind"] = working["transport_source_kind"].astype("string").fillna(day043_rebuild.MISSING_KIND)

    working["v43_transport_cost_min_day_provider"] = working["transport_cost_value"]
    working["v43_transport_cost_mean_day_provider"] = working["transport_cost_mean_day_provider"]
    working["v43_transport_cost_range_day_provider"] = working["transport_cost_range_day_provider"]
    working["v43_transport_observations"] = working["transport_observations"]
    working["v43_transport_unique_terminal_count"] = working["transport_unique_terminal_count"]
    working["v43_transport_multi_terminal_share"] = working["transport_multi_terminal_share"]
    working["v43_transport_source_raw_explicit"] = working["transport_source_kind"].eq(
        day043_rebuild.day042_rebuild.RAW_EXPLICIT_KIND
    ).astype(int)
    working["v43_transport_source_parser_fix"] = working["transport_source_kind"].eq(
        day043_rebuild.day042_rebuild.PARSER_FIX_KIND
    ).astype(int)
    working["v43_transport_source_deterministic_rebuild"] = working["transport_source_kind"].eq(
        day043_rebuild.day042_rebuild.DETERMINISTIC_KIND
    ).astype(int)
    working["v43_transport_source_carry_forward_30d"] = working["transport_source_kind"].eq(
        day043_rebuild.CARRY_FORWARD_KIND
    ).astype(int)
    working["v43_transport_source_missing"] = working["transport_source_kind"].eq(day043_rebuild.MISSING_KIND).astype(int)
    working["v43_transport_imputed_flag"] = working["transport_imputed_flag"]
    working["v43_transport_days_since_last_transport"] = working["transport_days_gap"]

    return working[
        [
            "fecha_evento",
            "producto_canonico",
            "proveedor_candidato",
            "transport_source_kind",
            "transport_lookahead_flag",
            "v43_transport_cost_min_day_provider",
            "v43_transport_cost_mean_day_provider",
            "v43_transport_cost_range_day_provider",
            "v43_transport_observations",
            "v43_transport_unique_terminal_count",
            "v43_transport_multi_terminal_share",
            "v43_transport_source_raw_explicit",
            "v43_transport_source_parser_fix",
            "v43_transport_source_deterministic_rebuild",
            "v43_transport_source_carry_forward_30d",
            "v43_transport_source_missing",
            "v43_transport_imputed_flag",
            "v43_transport_days_since_last_transport",
        ]
    ].copy()


# SECTION: Shared helpers
def _build_transport_event_frame(dataset_v2: pd.DataFrame, transport_day_frame: pd.DataFrame) -> pd.DataFrame:
    """Project day-level Day 04.3 transport features back to candidate-event rows and derive event ranks."""
    merged = dataset_v2[
        ["event_id", "fecha_evento", "producto_canonico", "proveedor_candidato"]
    ].merge(
        transport_day_frame,
        on=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        how="left",
    )
    merged["v43_transport_cost_min_day_provider"] = pd.to_numeric(
        merged["v43_transport_cost_min_day_provider"],
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []
    for event_id, group in merged.groupby("event_id", sort=False):
        valid_costs = group["v43_transport_cost_min_day_provider"].dropna().astype(float)
        min_cost = float(valid_costs.min()) if not valid_costs.empty else np.nan
        rank_lookup = valid_costs.rank(method="dense", ascending=True).to_dict() if not valid_costs.empty else {}
        for row_index, row in group.iterrows():
            current_cost = row.get("v43_transport_cost_min_day_provider", np.nan)
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
                    "transport_source_kind": row.get("transport_source_kind", day043_rebuild.MISSING_KIND),
                    "v43_transport_cost_min_day_provider": row.get("v43_transport_cost_min_day_provider", np.nan),
                    "v43_transport_cost_mean_day_provider": row.get("v43_transport_cost_mean_day_provider", np.nan),
                    "v43_transport_cost_range_day_provider": row.get("v43_transport_cost_range_day_provider", np.nan),
                    "v43_transport_observations": row.get("v43_transport_observations", np.nan),
                    "v43_transport_unique_terminal_count": row.get("v43_transport_unique_terminal_count", np.nan),
                    "v43_transport_multi_terminal_share": row.get("v43_transport_multi_terminal_share", np.nan),
                    "v43_transport_rank_event": transport_rank,
                    "v43_transport_gap_vs_min_event": transport_gap,
                    "v43_transport_ratio_vs_min_event": transport_ratio,
                    "v43_transport_source_raw_explicit": row.get("v43_transport_source_raw_explicit", 0),
                    "v43_transport_source_parser_fix": row.get("v43_transport_source_parser_fix", 0),
                    "v43_transport_source_deterministic_rebuild": row.get(
                        "v43_transport_source_deterministic_rebuild",
                        0,
                    ),
                    "v43_transport_source_carry_forward_30d": row.get(
                        "v43_transport_source_carry_forward_30d",
                        0,
                    ),
                    "v43_transport_source_missing": row.get("v43_transport_source_missing", 1),
                    "v43_transport_imputed_flag": row.get("v43_transport_imputed_flag", 0),
                    "v43_transport_days_since_last_transport": row.get(
                        "v43_transport_days_since_last_transport",
                        np.nan,
                    ),
                    "transport_lookahead_flag": row.get("transport_lookahead_flag", 0),
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
    """Build one Day 04.3 dataset variant and return its quality summary."""
    dataset = dataset_v2.merge(
        transport_event_frame,
        on=["event_id", "proveedor_candidato"],
        how="left",
    )
    added_columns = list(TRANSPORT_CARRY_COLUMNS)
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
    imputation_report_path: Path,
    output_dir: Path,
    quality_report_path: Path,
    cutoff_date: str,
    min_coverage: float,
    run_id: str,
) -> dict[str, Any]:
    """Build Day 04.3 datasets, compute gates, and decide whether training is allowed."""
    execution_run_id = run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_ts_utc = datetime.now(timezone.utc).isoformat()

    dataset_v2 = pd.read_csv(v2_input_path, keep_default_na=False)
    ofertas_typed = pd.read_csv(ofertas_typed_input_path, keep_default_na=False)
    transport_rebuilt = pd.read_csv(transport_rebuilt_input_path, keep_default_na=False)
    imputation_report = json.loads(imputation_report_path.read_text(encoding="utf-8"))

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
    stage_probe["transport_source_kind"] = stage_probe["transport_source_kind"].fillna(day043_rebuild.MISSING_KIND)
    stage_probe["transport_lookahead_flag"] = pd.to_numeric(
        stage_probe["transport_lookahead_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    stage_coverage = _compute_source_stage_coverage(dataset_df=stage_probe, cutoff_date=cutoff_date)
    lookahead_rows_v2 = int(stage_probe["transport_lookahead_flag"].sum())
    lookahead_unique_keys = int(transport_rebuilt["transport_lookahead_flag"].fillna(0).astype(int).sum())

    variants_summary: dict[str, Any] = {}
    for variant_name in [TRANSPORT_CARRY_VARIANT, DISPERSION_PLUS_TRANSPORT_VARIANT]:
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
        stage_coverage["final_after_carry_forward_30d"]["coverage_train"] >= min_coverage
        and stage_coverage["final_after_carry_forward_30d"]["coverage_test"] >= min_coverage
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
        "input_imputation_report": str(imputation_report_path),
        "output_dir": str(output_dir),
        "variants": variants_summary,
        "stage_coverage": stage_coverage,
        "lookahead": {
            "lookahead_unique_keys": lookahead_unique_keys,
            "lookahead_v2_rows": lookahead_rows_v2,
        },
        "gate": {
            "coverage_gate_pass": bool(coverage_gate_pass),
            "leakage_gate_pass": bool(leakage_gate_pass),
            "structural_gate_pass": bool(structural_gate_pass),
            "training_allowed": bool(training_allowed),
            "coverage_train_final": float(stage_coverage["final_after_carry_forward_30d"]["coverage_train"]),
            "coverage_test_final": float(stage_coverage["final_after_carry_forward_30d"]["coverage_test"]),
            "failure_reasons": [
                reason
                for reason, failed in [
                    ("coverage_below_min_threshold", not coverage_gate_pass),
                    ("carry_forward_lookahead_detected", not leakage_gate_pass),
                    ("dataset_contract_mismatch", not structural_gate_pass),
                ]
                if failed
            ],
        },
        "imputation_summary": {
            "backtest": imputation_report.get("backtest", {}),
            "carry_forward_summary": imputation_report.get("carry_forward_summary", {}),
            "day042_reference": imputation_report.get("day042_reference", {}),
        },
        "base_feature_columns_num": fc.get_feature_columns_v2()[0],
        "base_feature_columns_cat": fc.get_feature_columns_v2()[1],
        "target_column": fc.get_feature_columns_v2()[2],
    }
    quality_report_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return quality_report


# SECTION: CLI entrypoint
def main() -> None:
    """Run the CLI entrypoint for the Day 04.3 dataset builder."""
    args = parse_args()
    summary = run(
        v2_input_path=args.v2_input,
        ofertas_typed_input_path=args.ofertas_typed_input,
        transport_rebuilt_input_path=args.transport_rebuilt_input,
        imputation_report_path=args.imputation_report,
        output_dir=args.output_dir,
        quality_report_path=args.quality_report,
        cutoff_date=args.cutoff_date,
        min_coverage=args.min_coverage,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
