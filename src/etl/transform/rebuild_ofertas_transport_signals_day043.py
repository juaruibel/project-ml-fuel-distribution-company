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
    from src.etl.transform import rebuild_ofertas_transport_signals_day042 as day042_rebuild
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.etl.transform import rebuild_ofertas_transport_signals_day042 as day042_rebuild

CARRY_FORWARD_BUCKET = "carry_forward_30d"
CARRY_FORWARD_KIND = "carry_forward_30d"
MISSING_KIND = day042_rebuild.MISSING_KIND
DONOR_KINDS = {
    day042_rebuild.RAW_EXPLICIT_KIND,
    day042_rebuild.PARSER_FIX_KIND,
    day042_rebuild.DETERMINISTIC_KIND,
}


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 04.3 carry-forward transport rebuild."""
    parser = argparse.ArgumentParser(
        description="Reconstruye senales de transporte Day 04.3 usando ultimo valor previo <= 30 dias."
    )
    parser.add_argument(
        "--transport-input",
        type=Path,
        default=Path("data/public/support/ofertas_transport_signals_day042.csv"),
        help="Input staging Day 04.2 con buckets y provenance flags.",
    )
    parser.add_argument(
        "--v2-input",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v2_candidates.csv"),
        help="Dataset operativo V2 para medir cobertura por filas y splits oficiales.",
    )
    parser.add_argument(
        "--day042-report",
        type=Path,
        default=Path("artifacts/public/transport_missingness_day042.json"),
        help="Reporte Day 04.2 para comparar deltas de cobertura.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/public/support/ofertas_transport_signals_day043.csv"),
        help="Output staging Day 04.3 con carry-forward 30d.",
    )
    parser.add_argument(
        "--imputation-json",
        type=Path,
        default=Path("artifacts/public/transport_imputation_day043.json"),
        help="Reporte JSON de imputacion y cobertura Day 04.3.",
    )
    parser.add_argument(
        "--imputation-csv",
        type=Path,
        default=Path("artifacts/public/transport_imputation_day043.csv"),
        help="CSV tabular con clasificacion Day 04.3 por clave operativa.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2028-02-21",
        help="Cutoff temporal oficial para train/test coverage.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=30,
        help="Ventana maxima hacia atras para el carry-forward.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id opcional para trazabilidad.",
    )
    return parser.parse_args()


# SECTION: Shared helpers
def _safe_float(value: Any) -> float | None:
    """Convert values to Python floats while preserving missing values as None."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


# SECTION: Shared helpers
def _load_json_if_exists(path: Path) -> dict[str, Any]:
    """Load a JSON file from disk when it exists, otherwise return an empty payload."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# SECTION: Shared helpers
def _normalize_day042_staging(staging_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Day 04.2 staging to one row per key with stable dtypes and source-date semantics."""
    working = staging_df.copy()
    working["fecha_oferta"] = pd.to_datetime(working["fecha_oferta"], errors="coerce").dt.strftime("%Y-%m-%d")
    numeric_columns = [
        "transport_cost_value",
        "transport_cost_mean_day_provider",
        "transport_cost_range_day_provider",
        "transport_observations",
        "transport_unique_terminal_count",
        "transport_multi_terminal_share",
        "source_files_count",
        "transport_days_gap",
        "transport_lookahead_flag",
        "transport_imputed_flag",
    ]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    working["transport_source_kind"] = working["transport_source_kind"].astype("string").fillna(MISSING_KIND)
    working["transport_bucket_origin"] = working["transport_bucket_origin"].astype("string").fillna(day042_rebuild.NO_RAW_BUCKET)
    working["no_raw_subreason"] = working.get("no_raw_subreason", pd.Series(dtype="string")).astype("string")
    working["transport_source_date"] = working.get("source_date_for_transport", working["fecha_oferta"]).astype("string")
    working["transport_source_date"] = working["transport_source_date"].fillna(working["fecha_oferta"])
    working["transport_days_gap"] = working["transport_days_gap"].fillna(
        pd.Series(
            np.where(working["transport_source_kind"].isin(list(DONOR_KINDS)), 0.0, np.nan),
            index=working.index,
        )
    )
    working["transport_lookahead_flag"] = working["transport_lookahead_flag"].fillna(0).astype(int)
    working["day042_source_kind"] = working["transport_source_kind"]
    working = working.drop_duplicates(
        subset=["fecha_oferta", "producto_canonico", "proveedor_candidato"],
        keep="first",
    ).reset_index(drop=True)
    return working


# SECTION: Shared helpers
def _build_carry_forward_matches(
    *,
    target_keys: pd.DataFrame,
    donor_pool: pd.DataFrame,
    max_days: int,
) -> pd.DataFrame:
    """Match missing transport keys to the last strictly previous donor within a rolling day window."""
    if target_keys.empty or donor_pool.empty:
        return pd.DataFrame(columns=[
            "fecha_oferta",
            "producto_canonico",
            "proveedor_candidato",
            "transport_cost_value",
            "transport_cost_mean_day_provider",
            "transport_cost_range_day_provider",
            "transport_observations",
            "transport_unique_terminal_count",
            "transport_multi_terminal_share",
            "source_files_count",
            "transport_days_gap",
            "transport_source_date",
        ])

    left = target_keys.copy()
    left["target_dt"] = pd.to_datetime(left["fecha_oferta"], errors="coerce")
    left = left.dropna(subset=["target_dt"]).copy()

    right = donor_pool.copy()
    right["source_dt"] = pd.to_datetime(right["fecha_oferta"], errors="coerce")
    right = right.dropna(subset=["source_dt"]).copy()

    matched_groups: list[pd.DataFrame] = []
    tolerance = pd.Timedelta(days=max_days)
    for (producto_canonico, proveedor_candidato), left_group in left.groupby(
        ["producto_canonico", "proveedor_candidato"],
        sort=False,
    ):
        right_group = right[
            (right["producto_canonico"] == producto_canonico)
            & (right["proveedor_candidato"] == proveedor_candidato)
        ].copy()
        if right_group.empty:
            continue
        merged = pd.merge_asof(
            left=left_group.sort_values("target_dt", kind="stable"),
            right=right_group[
                [
                    "source_dt",
                    "fecha_oferta",
                    "transport_cost_value",
                    "transport_cost_mean_day_provider",
                    "transport_cost_range_day_provider",
                    "transport_observations",
                    "transport_unique_terminal_count",
                    "transport_multi_terminal_share",
                    "source_files_count",
                ]
            ]
            .rename(columns={"fecha_oferta": "donor_fecha_oferta"})
            .sort_values("source_dt", kind="stable"),
            left_on="target_dt",
            right_on="source_dt",
            direction="backward",
            allow_exact_matches=False,
            tolerance=tolerance,
        )
        matched_groups.append(merged)

    if not matched_groups:
        return pd.DataFrame(columns=[
            "fecha_oferta",
            "producto_canonico",
            "proveedor_candidato",
            "transport_cost_value",
            "transport_cost_mean_day_provider",
            "transport_cost_range_day_provider",
            "transport_observations",
            "transport_unique_terminal_count",
            "transport_multi_terminal_share",
            "source_files_count",
            "transport_days_gap",
            "transport_source_date",
        ])

    matched = pd.concat(matched_groups, ignore_index=True)
    matched["transport_days_gap"] = (matched["target_dt"] - matched["source_dt"]).dt.days.astype("Float64")
    matched["transport_source_date"] = matched["donor_fecha_oferta"]
    return matched[
        [
            "fecha_oferta",
            "producto_canonico",
            "proveedor_candidato",
            "transport_cost_value",
            "transport_cost_mean_day_provider",
            "transport_cost_range_day_provider",
            "transport_observations",
            "transport_unique_terminal_count",
            "transport_multi_terminal_share",
            "source_files_count",
            "transport_days_gap",
            "transport_source_date",
        ]
    ].copy()


# SECTION: Shared helpers
def _build_backtest_payload(donor_pool: pd.DataFrame, max_days: int) -> dict[str, Any]:
    """Backtest the carry-forward imputator over explicit historical transport rows."""
    if donor_pool.empty:
        return {
            "coverage_over_explicit_rows": 0.0,
            "matched_rows": 0,
            "total_rows": 0,
            "median_abs_error": None,
            "p90_abs_error": None,
            "median_ape": None,
            "p90_ape": None,
            "breakdown_by_product": [],
        }

    targets = donor_pool[
        [
            "fecha_oferta",
            "producto_canonico",
            "proveedor_candidato",
            "transport_cost_value",
        ]
    ].copy()
    targets["target_dt"] = pd.to_datetime(targets["fecha_oferta"], errors="coerce")
    targets = targets.dropna(subset=["target_dt"]).copy()

    donors = donor_pool[
        [
            "fecha_oferta",
            "producto_canonico",
            "proveedor_candidato",
            "transport_cost_value",
        ]
    ].copy()
    donors["source_dt"] = pd.to_datetime(donors["fecha_oferta"], errors="coerce")
    donors = donors.dropna(subset=["source_dt"]).copy()

    backtest_groups: list[pd.DataFrame] = []
    tolerance = pd.Timedelta(days=max_days)
    for (producto_canonico, proveedor_candidato), target_group in targets.groupby(
        ["producto_canonico", "proveedor_candidato"],
        sort=False,
    ):
        donor_group = donors[
            (donors["producto_canonico"] == producto_canonico)
            & (donors["proveedor_candidato"] == proveedor_candidato)
        ].copy()
        if donor_group.empty:
            continue
        merged = pd.merge_asof(
            left=target_group.rename(columns={"transport_cost_value": "actual_transport_cost"}).sort_values(
                "target_dt",
                kind="stable",
            ),
            right=donor_group[
                [
                    "fecha_oferta",
                    "transport_cost_value",
                    "source_dt",
                ]
            ]
            .rename(columns={"fecha_oferta": "source_fecha_oferta", "transport_cost_value": "donor_transport_cost"})
            .sort_values("source_dt", kind="stable"),
            left_on="target_dt",
            right_on="source_dt",
            direction="backward",
            allow_exact_matches=False,
            tolerance=tolerance,
        )
        backtest_groups.append(merged)

    if backtest_groups:
        backtest = pd.concat(backtest_groups, ignore_index=True)
    else:
        backtest = pd.DataFrame(columns=[
            "fecha_oferta",
            "producto_canonico",
            "proveedor_candidato",
            "actual_transport_cost",
            "target_dt",
            "source_fecha_oferta",
            "donor_transport_cost",
            "source_dt",
        ])
    backtest["matched_flag"] = backtest["donor_transport_cost"].notna().astype(int)
    matched = backtest[backtest["matched_flag"] == 1].copy()
    if not matched.empty:
        matched["abs_error"] = (matched["actual_transport_cost"] - matched["donor_transport_cost"]).abs()
        matched["ape"] = matched["abs_error"] / matched["actual_transport_cost"].abs().replace(0, np.nan)
    else:
        matched["abs_error"] = pd.Series(dtype=float)
        matched["ape"] = pd.Series(dtype=float)

    breakdown_rows: list[dict[str, Any]] = []
    for product, group in backtest.groupby("producto_canonico", dropna=False):
        matched_group = group[group["matched_flag"] == 1].copy()
        if not matched_group.empty:
            matched_group["abs_error"] = (
                matched_group["actual_transport_cost"] - matched_group["donor_transport_cost"]
            ).abs()
            matched_group["ape"] = matched_group["abs_error"] / matched_group["actual_transport_cost"].abs().replace(0, np.nan)
        breakdown_rows.append(
            {
                "producto_canonico": str(product),
                "coverage": float(group["matched_flag"].mean()),
                "matched_rows": int(group["matched_flag"].sum()),
                "total_rows": int(len(group)),
                "median_abs_error": _safe_float(matched_group["abs_error"].median()) if not matched_group.empty else None,
                "p90_abs_error": _safe_float(matched_group["abs_error"].quantile(0.9)) if not matched_group.empty else None,
                "median_ape": _safe_float(matched_group["ape"].median()) if not matched_group.empty else None,
                "p90_ape": _safe_float(matched_group["ape"].quantile(0.9)) if not matched_group.empty else None,
            }
        )

    return {
        "coverage_over_explicit_rows": float(backtest["matched_flag"].mean()),
        "matched_rows": int(backtest["matched_flag"].sum()),
        "total_rows": int(len(backtest)),
        "median_abs_error": _safe_float(matched["abs_error"].median()) if not matched.empty else None,
        "p90_abs_error": _safe_float(matched["abs_error"].quantile(0.9)) if not matched.empty else None,
        "median_ape": _safe_float(matched["ape"].median()) if not matched.empty else None,
        "p90_ape": _safe_float(matched["ape"].quantile(0.9)) if not matched.empty else None,
        "breakdown_by_product": breakdown_rows,
    }


# SECTION: Shared helpers
def _compute_stage_coverage(
    *,
    rows_with_sources: pd.DataFrame,
    cutoff_date: str,
) -> dict[str, dict[str, float]]:
    """Compute train/test coverage for Day 04.3 by provenance stage."""
    working = rows_with_sources.copy()
    working["fecha_evento_dt"] = pd.to_datetime(working["fecha_evento"], errors="coerce")
    cutoff_dt = pd.to_datetime(cutoff_date, errors="raise")
    train_mask = working["fecha_evento_dt"] <= cutoff_dt
    test_mask = working["fecha_evento_dt"] > cutoff_dt

    stage_masks = {
        "raw_explicit": working["transport_source_kind"].eq(day042_rebuild.RAW_EXPLICIT_KIND),
        "parser_fix_or_rebuild": working["transport_source_kind"].isin(list(DONOR_KINDS)),
        "final_after_carry_forward_30d": working["transport_source_kind"].ne(MISSING_KIND),
    }
    return {
        stage_name: {
            "coverage_train": float(mask.loc[train_mask].mean()) if int(train_mask.sum()) > 0 else 0.0,
            "coverage_test": float(mask.loc[test_mask].mean()) if int(test_mask.sum()) > 0 else 0.0,
        }
        for stage_name, mask in stage_masks.items()
    }


# SECTION: Main pipeline
def run(
    *,
    transport_input_path: Path,
    v2_input_path: Path,
    day042_report_path: Path,
    output_csv_path: Path,
    imputation_json_path: Path,
    imputation_csv_path: Path,
    cutoff_date: str,
    max_days: int,
    run_id: str,
) -> dict[str, Any]:
    """Rebuild Day 04.3 transport staging with carry-forward 30d and persist audit artifacts."""
    execution_run_id = run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_ts_utc = datetime.now(timezone.utc).isoformat()

    stage042 = _normalize_day042_staging(pd.read_csv(transport_input_path, keep_default_na=False))
    day042_report = _load_json_if_exists(day042_report_path)
    v2_rows = pd.read_csv(
        v2_input_path,
        usecols=["event_id", "fecha_evento", "producto_canonico", "proveedor_candidato"],
        keep_default_na=False,
    )

    donor_pool = stage042[stage042["transport_source_kind"].isin(list(DONOR_KINDS))].copy()
    donor_pool["transport_imputed_flag"] = donor_pool["transport_source_kind"].isin(
        [day042_rebuild.DETERMINISTIC_KIND]
    ).astype(int)
    donor_pool["transport_source_date"] = donor_pool["transport_source_date"].fillna(donor_pool["fecha_oferta"])
    donor_pool["transport_days_gap"] = donor_pool["transport_days_gap"].fillna(0.0)
    donor_pool["transport_lookahead_flag"] = 0

    carry_targets = stage042[~stage042["transport_source_kind"].isin(list(DONOR_KINDS))][
        ["fecha_oferta", "producto_canonico", "proveedor_candidato"]
    ].drop_duplicates().reset_index(drop=True)
    carry_matches = _build_carry_forward_matches(
        target_keys=carry_targets,
        donor_pool=donor_pool,
        max_days=max_days,
    )

    match_keys = set(
        map(tuple, carry_matches.dropna(subset=["transport_source_date"])[["fecha_oferta", "producto_canonico", "proveedor_candidato"]].itertuples(index=False, name=None))
    )
    final_rows: list[dict[str, Any]] = []
    for row in stage042.to_dict(orient="records"):
        key = (row["fecha_oferta"], row["producto_canonico"], row["proveedor_candidato"])
        record = dict(row)
        record["transport_source_date"] = str(record.get("transport_source_date", record["fecha_oferta"]))
        record["transport_days_gap"] = _safe_float(record.get("transport_days_gap"))
        record["transport_cost_value"] = _safe_float(record.get("transport_cost_value"))
        record["transport_cost_mean_day_provider"] = _safe_float(record.get("transport_cost_mean_day_provider"))
        record["transport_cost_range_day_provider"] = _safe_float(record.get("transport_cost_range_day_provider"))
        record["transport_observations"] = _safe_float(record.get("transport_observations"))
        record["transport_unique_terminal_count"] = _safe_float(record.get("transport_unique_terminal_count"))
        record["transport_multi_terminal_share"] = _safe_float(record.get("transport_multi_terminal_share"))
        record["source_files_count"] = _safe_float(record.get("source_files_count"))
        record["transport_lookahead_flag"] = 0

        if record["transport_source_kind"] not in DONOR_KINDS:
            if key in match_keys:
                matched_row = carry_matches.loc[
                    (carry_matches["fecha_oferta"] == row["fecha_oferta"])
                    & (carry_matches["producto_canonico"] == row["producto_canonico"])
                    & (carry_matches["proveedor_candidato"] == row["proveedor_candidato"])
                ].iloc[0]
                record["transport_cost_value"] = _safe_float(matched_row["transport_cost_value"])
                record["transport_cost_mean_day_provider"] = _safe_float(matched_row["transport_cost_mean_day_provider"])
                record["transport_cost_range_day_provider"] = _safe_float(matched_row["transport_cost_range_day_provider"])
                record["transport_observations"] = _safe_float(matched_row["transport_observations"])
                record["transport_unique_terminal_count"] = _safe_float(matched_row["transport_unique_terminal_count"])
                record["transport_multi_terminal_share"] = _safe_float(matched_row["transport_multi_terminal_share"])
                record["source_files_count"] = _safe_float(matched_row["source_files_count"])
                record["transport_source_kind"] = CARRY_FORWARD_KIND
                record["transport_bucket_origin"] = CARRY_FORWARD_BUCKET
                record["transport_imputed_flag"] = 1
                record["transport_days_gap"] = _safe_float(matched_row["transport_days_gap"])
                record["transport_source_date"] = str(matched_row["transport_source_date"])
            else:
                record["transport_source_kind"] = MISSING_KIND
                record["transport_bucket_origin"] = day042_rebuild.NO_RAW_BUCKET
                record["transport_imputed_flag"] = 0
                record["transport_days_gap"] = None
                record["transport_source_date"] = record["fecha_oferta"]
        final_rows.append(record)

    final_staging = pd.DataFrame(final_rows)
    final_staging["transport_imputed_flag"] = pd.to_numeric(
        final_staging["transport_imputed_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    final_staging["transport_lookahead_flag"] = 0
    final_staging["rebuild_run_id"] = execution_run_id
    final_staging["rebuild_ts_utc"] = generated_ts_utc
    final_staging = final_staging.sort_values(
        ["fecha_oferta", "producto_canonico", "proveedor_candidato"],
        kind="stable",
    ).reset_index(drop=True)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    imputation_json_path.parent.mkdir(parents=True, exist_ok=True)
    imputation_csv_path.parent.mkdir(parents=True, exist_ok=True)

    staging_output_columns = [
        "fecha_oferta",
        "producto_canonico",
        "proveedor_candidato",
        "transport_cost_value",
        "transport_cost_mean_day_provider",
        "transport_cost_range_day_provider",
        "transport_observations",
        "transport_unique_terminal_count",
        "transport_multi_terminal_share",
        "source_files_count",
        "transport_source_kind",
        "transport_imputed_flag",
        "transport_days_gap",
        "transport_lookahead_flag",
        "transport_source_date",
        "transport_bucket_origin",
        "no_raw_subreason",
        "rebuild_run_id",
        "rebuild_ts_utc",
    ]
    final_staging[staging_output_columns].to_csv(output_csv_path, index=False, encoding="utf-8")

    classified_csv = final_staging.merge(
        v2_rows.groupby(["fecha_evento", "producto_canonico", "proveedor_candidato"], as_index=False)
        .size()
        .rename(columns={"fecha_evento": "fecha_oferta", "size": "v2_row_count"}),
        on=["fecha_oferta", "producto_canonico", "proveedor_candidato"],
        how="left",
    )[[
        "fecha_oferta",
        "producto_canonico",
        "proveedor_candidato",
        "day042_source_kind",
        "transport_source_kind",
        "transport_bucket_origin",
        "transport_source_date",
        "transport_days_gap",
        "transport_imputed_flag",
        "transport_lookahead_flag",
        "no_raw_subreason",
        "v2_row_count",
    ]]
    classified_csv.to_csv(imputation_csv_path, index=False, encoding="utf-8")

    rows_with_buckets = v2_rows.merge(
        final_staging[
            [
                "fecha_oferta",
                "producto_canonico",
                "proveedor_candidato",
                "transport_source_kind",
                "transport_bucket_origin",
                "no_raw_subreason",
                "transport_lookahead_flag",
            ]
        ].rename(columns={"fecha_oferta": "fecha_evento"}),
        on=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        how="left",
    )
    rows_with_buckets["transport_source_kind"] = rows_with_buckets["transport_source_kind"].fillna(MISSING_KIND)
    rows_with_buckets["transport_bucket_origin"] = rows_with_buckets["transport_bucket_origin"].fillna(day042_rebuild.NO_RAW_BUCKET)
    rows_with_buckets["transport_lookahead_flag"] = pd.to_numeric(
        rows_with_buckets["transport_lookahead_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    rows_with_buckets["year"] = pd.to_datetime(rows_with_buckets["fecha_evento"], errors="coerce").dt.year.astype("Int64").astype(str)

    stage_coverage = _compute_stage_coverage(rows_with_sources=rows_with_buckets, cutoff_date=cutoff_date)
    lookahead_rows_v2 = int(rows_with_buckets["transport_lookahead_flag"].sum())
    carry_forward_rows_v2 = int(rows_with_buckets["transport_source_kind"].eq(CARRY_FORWARD_KIND).sum())

    backtest_payload = _build_backtest_payload(donor_pool=donor_pool, max_days=max_days)
    report = {
        "status": "ok",
        "run_id": execution_run_id,
        "generated_ts_utc": generated_ts_utc,
        "cutoff_date": cutoff_date,
        "max_days": int(max_days),
        "inputs": {
            "transport_input": str(transport_input_path),
            "v2_input": str(v2_input_path),
            "day042_report": str(day042_report_path),
        },
        "outputs": {
            "staging_output": str(output_csv_path),
            "imputation_json": str(imputation_json_path),
            "imputation_csv": str(imputation_csv_path),
        },
        "bucket_counts_unique_keys": final_staging["transport_bucket_origin"].value_counts(dropna=False).to_dict(),
        "bucket_counts_v2_rows": rows_with_buckets["transport_bucket_origin"].value_counts(dropna=False).to_dict(),
        "bucket_breakdown_by_year_v2_rows": day042_rebuild._row_bucket_breakdown(
            frame=rows_with_buckets,
            bucket_column="transport_bucket_origin",
            group_column="year",
        ),
        "bucket_breakdown_by_product_v2_rows": day042_rebuild._row_bucket_breakdown(
            frame=rows_with_buckets,
            bucket_column="transport_bucket_origin",
            group_column="producto_canonico",
        ),
        "top_provider_bucket_rows_v2": day042_rebuild._top_group_bucket_rows(
            frame=rows_with_buckets,
            group_column="proveedor_candidato",
            bucket_column="transport_bucket_origin",
            top_n=50,
        ),
        "stage_coverage": stage_coverage,
        "lookahead_stats": {
            "lookahead_rows_v2": int(lookahead_rows_v2),
            "lookahead_unique_keys": int(final_staging["transport_lookahead_flag"].sum()),
        },
        "carry_forward_summary": {
            "carry_forward_rows_v2": int(carry_forward_rows_v2),
            "final_train_coverage": float(stage_coverage["final_after_carry_forward_30d"]["coverage_train"]),
            "final_test_coverage": float(stage_coverage["final_after_carry_forward_30d"]["coverage_test"]),
            "delta_train_vs_day042_final": float(
                stage_coverage["final_after_carry_forward_30d"]["coverage_train"]
                - day042_report.get("stage_coverage", {}).get("final_after_heuristic", {}).get("coverage_train", 0.0)
            ),
            "delta_train_vs_day042_no_lookahead": float(
                stage_coverage["final_after_carry_forward_30d"]["coverage_train"]
                - day042_report.get("stage_coverage", {}).get("final_after_heuristic_no_lookahead", {}).get("coverage_train", 0.0)
            ),
        },
        "backtest": backtest_payload,
        "day042_reference": {
            "coverage_train_final": day042_report.get("stage_coverage", {}).get("final_after_heuristic", {}).get("coverage_train"),
            "coverage_train_final_no_lookahead": day042_report.get("stage_coverage", {}).get("final_after_heuristic_no_lookahead", {}).get("coverage_train"),
            "lookahead_rows_v2": day042_report.get("lookahead_stats", {}).get("heuristic_v2_rows_with_lookahead"),
        },
    }
    imputation_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# SECTION: CLI entrypoint
def main() -> None:
    """Run the CLI entrypoint for the Day 04.3 transport carry-forward rebuild."""
    args = parse_args()
    summary = run(
        transport_input_path=args.transport_input,
        v2_input_path=args.v2_input,
        day042_report_path=args.day042_report,
        output_csv_path=args.output_csv,
        imputation_json_path=args.imputation_json,
        imputation_csv_path=args.imputation_csv,
        cutoff_date=args.cutoff_date,
        max_days=args.max_days,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
