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
    from src.etl.marts import build_dataset_modelo_v3_context as day04_v3
    from src.etl.transform.build_ofertas_transport_signals import MULTI_TERMINAL_TOKEN, UNKNOWN_TERMINAL_TOKEN
    from src.ml.shared import functions as fc
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.etl.marts import build_dataset_modelo_v3_context as day04_v3
    from src.etl.transform.build_ofertas_transport_signals import MULTI_TERMINAL_TOKEN, UNKNOWN_TERMINAL_TOKEN
    from src.ml.shared import functions as fc

SOURCE_QUALITY_VARIANT = "source_quality"
DISPERSION_VARIANT = "dispersion"
COMPETITION_VARIANT = "competition"
TRANSPORT_VARIANT = "transport_only"
SELECTED_VARIANT = "selected_signals"

SOURCE_QUALITY_COLUMNS = day04_v3.V3_SIGNAL_FAMILIES[day04_v3.QUALITY_FAMILY]
DISPERSION_COLUMNS = day04_v3.V3_SIGNAL_FAMILIES[day04_v3.DISPERSION_FAMILY]
COMPETITION_COLUMNS = day04_v3.V3_SIGNAL_FAMILIES[day04_v3.COMPETITION_FAMILY]
TRANSPORT_COLUMNS = [
    "v41_transport_cost_min_day_provider",
    "v41_transport_cost_mean_day_provider",
    "v41_transport_cost_range_day_provider",
    "v41_transport_observations",
    "v41_transport_unique_terminal_count",
    "v41_transport_multi_terminal_share",
    "v41_transport_rank_event",
    "v41_transport_gap_vs_min_event",
    "v41_transport_ratio_vs_min_event",
]

VARIANT_OUTPUTS = {
    SOURCE_QUALITY_VARIANT: "dataset_modelo_v2_source_quality.csv",
    DISPERSION_VARIANT: "dataset_modelo_v2_dispersion.csv",
    COMPETITION_VARIANT: "dataset_modelo_v2_competition.csv",
    TRANSPORT_VARIANT: "dataset_modelo_v2_transport_only.csv",
    SELECTED_VARIANT: "dataset_modelo_v3_a2_selected_signals.csv",
}


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 04.1 ablation dataset builder."""
    parser = argparse.ArgumentParser(
        description="Construye datasets Day 04.1 para ablación sobre V2 y señales de transporte."
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
        help="Input staging de ofertas tipadas.",
    )
    parser.add_argument(
        "--transport-input",
        type=Path,
        default=Path("data/public/support/ofertas_transport_signals.csv"),
        help="Input staging con parser de transporte.",
    )
    parser.add_argument(
        "--transport-report",
        type=Path,
        default=Path("artifacts/public/transport_parser_day041.json"),
        help="Reporte del parser de transporte.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/public/day041"),
        help="Directorio de salida para datasets Day 04.1.",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("artifacts/public/data_quality_day041_ablation_matrix.json"),
        help="Reporte JSON de calidad/cobertura para la matriz de ablación.",
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
        help="Cobertura mínima para permitir la variante transport_only.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id opcional.",
    )
    return parser.parse_args()


# SECTION: Shared helpers
def _compute_feature_coverage(
    dataset_df: pd.DataFrame,
    feature_columns: list[str],
    cutoff_date: str,
) -> dict[str, dict[str, float]]:
    """Compute train/test non-null coverage for a list of new feature columns."""
    working = dataset_df.copy()
    working["fecha_evento_dt"] = pd.to_datetime(working["fecha_evento"], errors="coerce")
    cutoff_dt = pd.to_datetime(cutoff_date, errors="raise")
    train_df = working[working["fecha_evento_dt"] <= cutoff_dt].copy()
    test_df = working[working["fecha_evento_dt"] > cutoff_dt].copy()

    coverage: dict[str, dict[str, float]] = {}
    for column in feature_columns:
        coverage[column] = {
            "coverage_train": float(train_df[column].notna().mean()) if not train_df.empty else 0.0,
            "coverage_test": float(test_df[column].notna().mean()) if not test_df.empty else 0.0,
        }
    return coverage


# SECTION: Shared helpers
def _build_transport_day_frame(transport_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transport parser rows to day-product-provider grain."""
    working = transport_df.copy()
    working["fecha_oferta"] = pd.to_datetime(working["fecha_oferta"], errors="coerce").dt.strftime("%Y-%m-%d")
    working["transport_cost_value"] = pd.to_numeric(working["transport_cost_value"], errors="coerce")
    working = working.dropna(subset=["fecha_oferta", "transport_cost_value"]).copy()
    working = working[
        (working["producto_canonico"] != "UNKNOWN")
        & (working["proveedor_candidato"] != "UNKNOWN")
    ].copy()

    if working.empty:
        return pd.DataFrame(
            columns=[
                "fecha_evento",
                "producto_canonico",
                "proveedor_candidato",
                "v41_transport_cost_min_day_provider",
                "v41_transport_cost_mean_day_provider",
                "v41_transport_cost_range_day_provider",
                "v41_transport_observations",
                "v41_transport_unique_terminal_count",
                "v41_transport_multi_terminal_share",
            ]
        )

    grouped = (
        working.groupby(["fecha_oferta", "producto_canonico", "proveedor_candidato"], as_index=False)
        .agg(
            v41_transport_cost_min_day_provider=("transport_cost_value", "min"),
            v41_transport_cost_mean_day_provider=("transport_cost_value", "mean"),
            v41_transport_cost_max_day_provider=("transport_cost_value", "max"),
            v41_transport_observations=("transport_cost_value", "count"),
            v41_transport_unique_terminal_count=(
                "terminal_canonico",
                lambda s: int(
                    s.astype(str)
                    .loc[~s.astype(str).isin([MULTI_TERMINAL_TOKEN, UNKNOWN_TERMINAL_TOKEN])]
                    .nunique()
                ),
            ),
            v41_transport_multi_terminal_share=(
                "parser_status",
                lambda s: float(s.astype(str).eq("parsed_multi_terminal_aggregate").mean()),
            ),
        )
        .rename(columns={"fecha_oferta": "fecha_evento"})
    )
    grouped["v41_transport_cost_range_day_provider"] = (
        grouped["v41_transport_cost_max_day_provider"] - grouped["v41_transport_cost_min_day_provider"]
    )
    grouped = grouped.drop(columns=["v41_transport_cost_max_day_provider"])
    return grouped


# SECTION: Shared helpers
def _build_transport_event_frame(dataset_v2: pd.DataFrame, transport_day_frame: pd.DataFrame) -> pd.DataFrame:
    """Project transport aggregates back to candidate rows and derive event-level transport ranking features."""
    if transport_day_frame.empty:
        columns = ["event_id", "proveedor_candidato"] + TRANSPORT_COLUMNS
        return pd.DataFrame(columns=columns)

    merged = dataset_v2[
        ["event_id", "fecha_evento", "producto_canonico", "proveedor_candidato"]
    ].merge(
        transport_day_frame,
        on=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        how="left",
    )
    merged["v41_transport_cost_min_day_provider"] = pd.to_numeric(
        merged["v41_transport_cost_min_day_provider"], errors="coerce"
    )

    rows: list[dict[str, Any]] = []
    for event_id, group in merged.groupby("event_id", sort=False):
        valid_costs = group["v41_transport_cost_min_day_provider"].dropna().astype(float)
        min_cost = float(valid_costs.min()) if not valid_costs.empty else np.nan
        if not valid_costs.empty:
            ranked = valid_costs.rank(method="dense", ascending=True)
            rank_lookup = ranked.to_dict()
        else:
            rank_lookup = {}
        for row_index, row in group.iterrows():
            current_cost = row.get("v41_transport_cost_min_day_provider", np.nan)
            if pd.isna(current_cost) or pd.isna(min_cost):
                transport_rank = np.nan
                transport_gap = np.nan
                transport_ratio = np.nan
            else:
                transport_rank = float(rank_lookup.get(row_index, np.nan))
                transport_gap = float(current_cost - min_cost)
                transport_ratio = (
                    float(current_cost / min_cost) if min_cost not in {0.0, np.nan} else np.nan
                )
            rows.append(
                {
                    "event_id": str(event_id),
                    "proveedor_candidato": str(row["proveedor_candidato"]),
                    "v41_transport_cost_min_day_provider": row.get("v41_transport_cost_min_day_provider", np.nan),
                    "v41_transport_cost_mean_day_provider": row.get("v41_transport_cost_mean_day_provider", np.nan),
                    "v41_transport_cost_range_day_provider": row.get("v41_transport_cost_range_day_provider", np.nan),
                    "v41_transport_observations": row.get("v41_transport_observations", np.nan),
                    "v41_transport_unique_terminal_count": row.get("v41_transport_unique_terminal_count", np.nan),
                    "v41_transport_multi_terminal_share": row.get("v41_transport_multi_terminal_share", np.nan),
                    "v41_transport_rank_event": transport_rank,
                    "v41_transport_gap_vs_min_event": transport_gap,
                    "v41_transport_ratio_vs_min_event": transport_ratio,
                }
            )
    return pd.DataFrame(rows)


# SECTION: Shared helpers
def _variant_frames(
    dataset_v2: pd.DataFrame,
    ofertas_typed: pd.DataFrame,
    transport_df: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, list[str], list[str]]]:
    """Build all feature-family frames used by the Day 04.1 ablation matrix."""
    source_dispersion = day04_v3._build_source_quality_frame(ofertas_typed=ofertas_typed)
    competition = day04_v3._build_event_competition_frame(dataset_v2=dataset_v2)
    transport_day = _build_transport_day_frame(transport_df=transport_df)
    transport_event = _build_transport_event_frame(dataset_v2=dataset_v2, transport_day_frame=transport_day)

    source_quality_frame = source_dispersion[
        ["fecha_evento", "producto_canonico", "proveedor_candidato"] + SOURCE_QUALITY_COLUMNS
    ].copy()
    dispersion_frame = source_dispersion[
        ["fecha_evento", "producto_canonico", "proveedor_candidato"] + DISPERSION_COLUMNS
    ].copy()
    competition_frame = competition[
        ["event_id", "proveedor_candidato"] + COMPETITION_COLUMNS
    ].copy()
    transport_frame = transport_event[
        ["event_id", "proveedor_candidato"] + TRANSPORT_COLUMNS
    ].copy()

    return {
        SOURCE_QUALITY_VARIANT: (
            source_quality_frame,
            SOURCE_QUALITY_COLUMNS,
            ["fecha_evento", "producto_canonico", "proveedor_candidato"],
        ),
        DISPERSION_VARIANT: (
            dispersion_frame,
            DISPERSION_COLUMNS,
            ["fecha_evento", "producto_canonico", "proveedor_candidato"],
        ),
        COMPETITION_VARIANT: (
            competition_frame,
            COMPETITION_COLUMNS,
            ["event_id", "proveedor_candidato"],
        ),
        TRANSPORT_VARIANT: (
            transport_frame,
            TRANSPORT_COLUMNS,
            ["event_id", "proveedor_candidato"],
        ),
    }


# SECTION: Dataset building
def build_variant_dataset(
    *,
    dataset_v2: pd.DataFrame,
    variant_name: str,
    variant_frame: pd.DataFrame,
    added_columns: list[str],
    join_keys: list[str],
    cutoff_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one Day 04.1 variant dataset and return its quality summary."""
    dataset = dataset_v2.merge(variant_frame, on=join_keys, how="left")
    coverage = _compute_feature_coverage(dataset_df=dataset, feature_columns=added_columns, cutoff_date=cutoff_date)
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
        "feature_coverage": coverage,
    }
    return dataset, summary


# SECTION: Dataset building
def build_selected_signals_dataset(
    *,
    dataset_v2: pd.DataFrame,
    variant_frames: dict[str, tuple[pd.DataFrame, list[str], list[str]]],
    selected_families: list[str],
    output_dir: Path,
    cutoff_date: str,
) -> tuple[Path, dict[str, Any]] | None:
    """Build the optional V3_A2 dataset from selected non-harmful families."""
    if len(selected_families) < 2:
        return None

    dataset = dataset_v2.copy()
    added_columns: list[str] = []
    for family in selected_families:
        frame, family_columns, join_keys = variant_frames[family]
        dataset = dataset.merge(frame, on=join_keys, how="left")
        added_columns.extend(family_columns)

    coverage = _compute_feature_coverage(dataset_df=dataset, feature_columns=added_columns, cutoff_date=cutoff_date)
    missing_flag_columns: list[str] = []
    for column in added_columns:
        if dataset[column].isna().any():
            flag_column = f"{column}_missing_flag"
            dataset[flag_column] = dataset[column].isna().astype(int)
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce").fillna(0.0)
            missing_flag_columns.append(flag_column)
        else:
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    output_path = output_dir / VARIANT_OUTPUTS[SELECTED_VARIANT]
    dataset.to_csv(output_path, index=False, encoding="utf-8")

    positive_per_event = dataset.groupby("event_id")["target_elegido"].sum()
    summary = {
        "variant_name": SELECTED_VARIANT,
        "dataset_name": output_path.name,
        "selected_families": selected_families,
        "rows_output": int(len(dataset)),
        "events_output": int(dataset["event_id"].nunique()),
        "rows_vs_v2_match": int(len(dataset) == len(dataset_v2)),
        "events_vs_v2_match": int(dataset["event_id"].nunique() == dataset_v2["event_id"].nunique()),
        "events_with_invalid_positive_count": int((positive_per_event != 1).sum()) if len(positive_per_event) else 0,
        "added_columns": added_columns,
        "missing_flag_columns": missing_flag_columns,
        "feature_coverage": coverage,
        "dataset_path": str(output_path),
    }
    return output_path, summary


# SECTION: Main pipeline
def run(
    *,
    v2_input_path: Path,
    ofertas_typed_input_path: Path,
    transport_input_path: Path,
    transport_report_path: Path,
    output_dir: Path,
    quality_report_path: Path,
    cutoff_date: str,
    min_coverage: float,
    run_id: str,
) -> dict[str, Any]:
    """Build all Day 04.1 pure ablation datasets and persist the quality matrix report."""
    execution_run_id = run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_ts_utc = datetime.now(timezone.utc).isoformat()

    dataset_v2 = pd.read_csv(v2_input_path, keep_default_na=False)
    ofertas_typed = pd.read_csv(ofertas_typed_input_path, keep_default_na=False)
    transport_df = pd.read_csv(transport_input_path, keep_default_na=False) if transport_input_path.exists() else pd.DataFrame()
    transport_report = (
        json.loads(transport_report_path.read_text(encoding="utf-8")) if transport_report_path.exists() else {}
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)

    variant_frames = _variant_frames(dataset_v2=dataset_v2, ofertas_typed=ofertas_typed, transport_df=transport_df)
    variants_summary: dict[str, Any] = {}

    for variant_name in [SOURCE_QUALITY_VARIANT, DISPERSION_VARIANT, COMPETITION_VARIANT]:
        variant_frame, added_columns, join_keys = variant_frames[variant_name]
        dataset_variant, summary = build_variant_dataset(
            dataset_v2=dataset_v2,
            variant_name=variant_name,
            variant_frame=variant_frame,
            added_columns=added_columns,
            join_keys=join_keys,
            cutoff_date=cutoff_date,
        )
        output_path = output_dir / VARIANT_OUTPUTS[variant_name]
        dataset_variant.to_csv(output_path, index=False, encoding="utf-8")
        summary["dataset_path"] = str(output_path)
        summary["status"] = "built"
        variants_summary[variant_name] = summary

    transport_frame, transport_added_columns, transport_join_keys = variant_frames[TRANSPORT_VARIANT]
    transport_probe = dataset_v2.merge(transport_frame, on=transport_join_keys, how="left")
    transport_coverage = _compute_feature_coverage(
        dataset_df=transport_probe,
        feature_columns=transport_added_columns,
        cutoff_date=cutoff_date,
    )
    transport_min_train = min(
        metrics["coverage_train"] for metrics in transport_coverage.values()
    ) if transport_coverage else 0.0
    transport_min_test = min(
        metrics["coverage_test"] for metrics in transport_coverage.values()
    ) if transport_coverage else 0.0
    transport_gate_pass = transport_min_train >= min_coverage and transport_min_test >= min_coverage

    transport_summary = {
        "variant_name": TRANSPORT_VARIANT,
        "dataset_name": VARIANT_OUTPUTS[TRANSPORT_VARIANT],
        "added_columns": transport_added_columns,
        "feature_coverage": transport_coverage,
        "transport_gate_pass": bool(transport_gate_pass),
        "transport_min_train_coverage": float(transport_min_train),
        "transport_min_test_coverage": float(transport_min_test),
        "transport_parser_report": transport_report,
    }
    if transport_gate_pass:
        dataset_transport, transport_variant_summary = build_variant_dataset(
            dataset_v2=dataset_v2,
            variant_name=TRANSPORT_VARIANT,
            variant_frame=transport_frame,
            added_columns=transport_added_columns,
            join_keys=transport_join_keys,
            cutoff_date=cutoff_date,
        )
        output_path = output_dir / VARIANT_OUTPUTS[TRANSPORT_VARIANT]
        dataset_transport.to_csv(output_path, index=False, encoding="utf-8")
        transport_summary.update(transport_variant_summary)
        transport_summary["dataset_path"] = str(output_path)
        transport_summary["status"] = "built"
    else:
        transport_summary["status"] = "excluded_parser_gate_failed"
    variants_summary[TRANSPORT_VARIANT] = transport_summary

    quality_report = {
        "status": "ok",
        "run_id": execution_run_id,
        "generated_ts_utc": generated_ts_utc,
        "cutoff_date": cutoff_date,
        "min_coverage": float(min_coverage),
        "input_v2_dataset": str(v2_input_path),
        "input_ofertas_typed": str(ofertas_typed_input_path),
        "input_transport_signals": str(transport_input_path),
        "input_transport_report": str(transport_report_path),
        "output_dir": str(output_dir),
        "variants": variants_summary,
        "base_feature_columns_num": fc.get_feature_columns_v2()[0],
        "base_feature_columns_cat": fc.get_feature_columns_v2()[1],
        "target_column": fc.get_feature_columns_v2()[2],
    }
    quality_report_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return quality_report


# SECTION: CLI entrypoint
def main() -> None:
    """Run the CLI entrypoint for the Day 04.1 ablation dataset builder."""
    args = parse_args()
    summary = run(
        v2_input_path=args.v2_input,
        ofertas_typed_input_path=args.ofertas_typed_input,
        transport_input_path=args.transport_input,
        transport_report_path=args.transport_report,
        output_dir=args.output_dir,
        quality_report_path=args.quality_report,
        cutoff_date=args.cutoff_date,
        min_coverage=args.min_coverage,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
