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
    from src.etl.marts.build_dataset_modelo_v2_candidates import _build_column_dictionary as build_v2_column_dictionary
    from src.ml.shared import functions as fc
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.etl.marts.build_dataset_modelo_v2_candidates import _build_column_dictionary as build_v2_column_dictionary
    from src.ml.shared import functions as fc

RAW_AUDIT_KEYWORDS = ("transporte", "transport", "porte", "flete", "logistica")

QUALITY_FAMILY = "source_quality"
COMPETITION_FAMILY = "event_competition"
DISPERSION_FAMILY = "multi_terminal_dispersion"
TRANSPORT_AUDIT_FAMILY = "raw_transport_or_unmapped_calcs"

V3_EXTRA_NUM_FEATURE_COLUMNS = [
    "v3_selected_source_calculos_share",
    "v3_cost_source_calculos_share",
    "v3_reconciliation_conflict_share",
    "v3_reconciliation_single_source_share",
    "v3_cost_mean_terminal",
    "v3_cost_std_terminal",
    "v3_cost_range_terminal",
    "v3_cost_cv_terminal",
    "v3_share_terminales_min_cost",
    "v3_coste_segundo_evento",
    "v3_gap_min_vs_second_evento",
    "v3_delta_vs_second_evento",
    "v3_ratio_vs_second_evento",
    "v3_rank_pct_evento",
    "v3_coste_mean_evento",
    "v3_delta_vs_mean_evento",
    "v3_candidatos_min_coste_count",
    "v3_is_unique_min_coste_evento",
]

V3_SIGNAL_FAMILIES = {
    QUALITY_FAMILY: [
        "v3_selected_source_calculos_share",
        "v3_cost_source_calculos_share",
        "v3_reconciliation_conflict_share",
        "v3_reconciliation_single_source_share",
    ],
    DISPERSION_FAMILY: [
        "v3_cost_mean_terminal",
        "v3_cost_std_terminal",
        "v3_cost_range_terminal",
        "v3_cost_cv_terminal",
        "v3_share_terminales_min_cost",
    ],
    COMPETITION_FAMILY: [
        "v3_coste_segundo_evento",
        "v3_gap_min_vs_second_evento",
        "v3_delta_vs_second_evento",
        "v3_ratio_vs_second_evento",
        "v3_rank_pct_evento",
        "v3_coste_mean_evento",
        "v3_delta_vs_mean_evento",
        "v3_candidatos_min_coste_count",
        "v3_is_unique_min_coste_evento",
    ],
}


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the V3_A dataset builder."""
    parser = argparse.ArgumentParser(
        description="Construye dataset_modelo_proveedor_v3_context.csv sin leakage temporal."
    )
    parser.add_argument(
        "--v2-input",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v2_candidates.csv"),
        help="Input base V2 a grano event_id + proveedor_candidato.",
    )
    parser.add_argument(
        "--ofertas-typed-input",
        type=Path,
        default=Path("data/public/support/ofertas_typed.csv"),
        help="Input staging de ofertas tipadas.",
    )
    parser.add_argument(
        "--raw-matrix-input",
        type=Path,
        default=Path("data/public/support/ofertas_raw_matrix_cells.csv"),
        help="Input opcional de celdas raw para auditoría de cálculos no mapeados.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v3_context.csv"),
        help="Output dataset V3_A.",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("artifacts/public/data_quality_v3_context.json"),
        help="Output JSON de calidad/cobertura V3_A.",
    )
    parser.add_argument(
        "--data-dictionary",
        type=Path,
        default=Path("artifacts/public/data_dictionary_v3_context.md"),
        help="Output markdown del diccionario V3_A.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2028-02-21",
        help="Cutoff temporal oficial para coverage train/test.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.80,
        help="Cobertura mínima train/test para considerar una señal usable.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador opcional de ejecución.",
    )
    return parser.parse_args()


# SECTION: Feature contract
def get_feature_columns_v3() -> tuple[list[str], list[str], str]:
    """Return the centralized V3_A feature contract used by training/inference experiments."""
    base_num, base_cat, target_col = fc.get_feature_columns_v2()
    return base_num + V3_EXTRA_NUM_FEATURE_COLUMNS, base_cat, target_col


# SECTION: Shared helpers
def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a stable ratio and avoid divisions by zero."""
    if denominator == 0 or np.isnan(denominator):
        return 0.0
    return float(numerator / denominator)


# SECTION: Raw source aggregation
def _build_source_quality_frame(ofertas_typed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate staging/raw source quality and dispersion signals by day-product-provider."""
    working = ofertas_typed.copy()
    working["fecha_oferta"] = pd.to_datetime(working["fecha_oferta"], errors="coerce").dt.strftime("%Y-%m-%d")
    working["coste_min"] = pd.to_numeric(working["coste_min"], errors="coerce")
    working["selected_source"] = working.get("selected_source", "").fillna("").astype(str).str.strip().str.lower()
    working["cost_source"] = working.get("cost_source", "").fillna("").astype(str).str.strip().str.lower()
    working["reconciliation_status"] = (
        working.get("reconciliation_status", "").fillna("").astype(str).str.strip().str.lower()
    )

    grouped = (
        working.groupby(["fecha_oferta", "producto_canonico", "proveedor_canonico"], as_index=False)
        .agg(
            v3_selected_source_calculos_share=("selected_source", lambda s: float(s.eq("calculos").mean())),
            v3_cost_source_calculos_share=("cost_source", lambda s: float(s.eq("calculos").mean())),
            v3_reconciliation_conflict_share=("reconciliation_status", lambda s: float(s.eq("conflict").mean())),
            v3_reconciliation_single_source_share=(
                "reconciliation_status",
                lambda s: float(s.eq("single_source").mean()),
            ),
            v3_cost_mean_terminal=("coste_min", "mean"),
            v3_cost_std_terminal=("coste_min", lambda s: float(s.std(ddof=0)) if len(s) else 0.0),
            v3_cost_min_terminal=("coste_min", "min"),
            v3_cost_max_terminal=("coste_min", "max"),
            v3_terminal_rows_count=("terminal_canonico", "count"),
            v3_terminal_min_rows_count=("coste_min", lambda s: int(np.isclose(s.astype(float), float(s.min())).sum())),
        )
        .rename(
            columns={
                "fecha_oferta": "fecha_evento",
                "proveedor_canonico": "proveedor_candidato",
            }
        )
    )

    grouped["v3_cost_range_terminal"] = grouped["v3_cost_max_terminal"] - grouped["v3_cost_min_terminal"]
    grouped["v3_cost_cv_terminal"] = grouped.apply(
        lambda row: _safe_ratio(float(row["v3_cost_std_terminal"]), float(row["v3_cost_mean_terminal"])),
        axis=1,
    )
    grouped["v3_share_terminales_min_cost"] = grouped.apply(
        lambda row: _safe_ratio(
            float(row["v3_terminal_min_rows_count"]),
            float(row["v3_terminal_rows_count"]),
        ),
        axis=1,
    )
    grouped = grouped.drop(
        columns=[
            "v3_cost_min_terminal",
            "v3_cost_max_terminal",
            "v3_terminal_rows_count",
            "v3_terminal_min_rows_count",
        ]
    )
    for column in V3_SIGNAL_FAMILIES[QUALITY_FAMILY] + V3_SIGNAL_FAMILIES[DISPERSION_FAMILY]:
        grouped[column] = pd.to_numeric(grouped[column], errors="coerce")
    return grouped


# SECTION: Event competition features
def _build_event_competition_frame(dataset_v2: pd.DataFrame) -> pd.DataFrame:
    """Aggregate competition features at event level and project them back to each candidate row."""
    working = dataset_v2.copy()
    working["coste_min_dia_proveedor"] = pd.to_numeric(working["coste_min_dia_proveedor"], errors="coerce")
    working["rank_coste_dia_producto"] = pd.to_numeric(working["rank_coste_dia_producto"], errors="coerce")
    working["candidatos_evento_count"] = pd.to_numeric(working["candidatos_evento_count"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for event_id, group in working.groupby("event_id", sort=False):
        costs = group["coste_min_dia_proveedor"].dropna().astype(float)
        if costs.empty:
            second_cost = 0.0
            min_cost = 0.0
            mean_cost = 0.0
            min_count = 0
        else:
            min_cost = float(costs.min())
            unique_costs = sorted(set(costs.tolist()))
            second_cost = float(unique_costs[1]) if len(unique_costs) > 1 else float(unique_costs[0])
            mean_cost = float(costs.mean())
            min_count = int(np.isclose(costs, min_cost).sum())

        gap_min_vs_second = float(second_cost - min_cost)
        is_unique_min = int(min_count == 1)

        for _, row in group.iterrows():
            rank_value = float(row.get("rank_coste_dia_producto", 0.0) or 0.0)
            candidate_count = float(row.get("candidatos_evento_count", 0.0) or 0.0)
            cost_value = float(row.get("coste_min_dia_proveedor", 0.0) or 0.0)
            rank_pct = 0.0 if candidate_count <= 1 else float((rank_value - 1.0) / (candidate_count - 1.0))
            rows.append(
                {
                    "event_id": str(event_id),
                    "proveedor_candidato": str(row.get("proveedor_candidato", "")).strip(),
                    "v3_coste_segundo_evento": second_cost,
                    "v3_gap_min_vs_second_evento": gap_min_vs_second,
                    "v3_delta_vs_second_evento": float(cost_value - second_cost),
                    "v3_ratio_vs_second_evento": _safe_ratio(cost_value, second_cost),
                    "v3_rank_pct_evento": rank_pct,
                    "v3_coste_mean_evento": mean_cost,
                    "v3_delta_vs_mean_evento": float(cost_value - mean_cost),
                    "v3_candidatos_min_coste_count": int(min_count),
                    "v3_is_unique_min_coste_evento": int(is_unique_min),
                }
            )

    competition = pd.DataFrame(rows)
    for column in V3_SIGNAL_FAMILIES[COMPETITION_FAMILY]:
        competition[column] = pd.to_numeric(competition[column], errors="coerce")
    return competition


# SECTION: Coverage and audit
def _compute_feature_coverage(dataset_v3: pd.DataFrame, cutoff_date: str) -> dict[str, dict[str, float]]:
    """Compute non-null coverage by feature for train/test using the official cutoff date."""
    coverage: dict[str, dict[str, float]] = {}
    cutoff_dt = pd.to_datetime(cutoff_date, errors="raise")
    working = dataset_v3.copy()
    working["fecha_evento_dt"] = pd.to_datetime(working["fecha_evento"], errors="coerce")
    train_df = working[working["fecha_evento_dt"] <= cutoff_dt].copy()
    test_df = working[working["fecha_evento_dt"] > cutoff_dt].copy()

    for column in V3_EXTRA_NUM_FEATURE_COLUMNS:
        train_cov = float(train_df[column].notna().mean()) if not train_df.empty else 0.0
        test_cov = float(test_df[column].notna().mean()) if not test_df.empty else 0.0
        coverage[column] = {
            "coverage_train": train_cov,
            "coverage_test": test_cov,
        }
    return coverage


# SECTION: Coverage and audit
def _build_transport_audit(raw_matrix_path: Path) -> dict[str, Any]:
    """Audit raw matrix cells for transport/unmapped-calculation hints without changing the parser."""
    if not raw_matrix_path.exists():
        return {
            "source_path": str(raw_matrix_path),
            "keyword_hits": 0,
            "sample_hits": [],
            "decision": "exclude_missing_raw_matrix",
        }

    raw_matrix = pd.read_csv(raw_matrix_path, dtype=str, keep_default_na=False)
    if raw_matrix.empty or "cell_value" not in raw_matrix.columns:
        return {
            "source_path": str(raw_matrix_path),
            "keyword_hits": 0,
            "sample_hits": [],
            "decision": "exclude_empty_raw_matrix",
        }

    normalized = raw_matrix["cell_value"].astype(str).str.lower()
    mask = normalized.apply(lambda value: any(keyword in value for keyword in RAW_AUDIT_KEYWORDS))
    matches = raw_matrix.loc[mask, ["source_file", "sheet_name", "row_idx", "col_idx", "cell_value"]].head(10)
    return {
        "source_path": str(raw_matrix_path),
        "keyword_hits": int(mask.sum()),
        "sample_hits": matches.to_dict(orient="records"),
        "decision": "candidate_parser_extension" if int(mask.sum()) > 0 else "exclude_no_evidence_in_raw",
    }


# SECTION: Signal contract
def _build_signal_decision_table(
    coverage_by_feature: dict[str, dict[str, float]],
    min_coverage: float,
    transport_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the decision-complete signal-family table required by the Day 04 notebook."""
    family_specs = [
        {
            "signal_family": QUALITY_FAMILY,
            "source_table": "data/public/support/ofertas_typed.csv",
            "raw_grain": "fecha_oferta + producto_canonico + proveedor_canonico + terminal_canonico",
            "target_grain": "event_id + proveedor_candidato",
            "requires_parser_change": False,
            "pre_decision": True,
            "leakage_risk": "low",
        },
        {
            "signal_family": DISPERSION_FAMILY,
            "source_table": "data/public/support/ofertas_typed.csv",
            "raw_grain": "fecha_oferta + producto_canonico + proveedor_canonico + terminal_canonico",
            "target_grain": "event_id + proveedor_candidato",
            "requires_parser_change": False,
            "pre_decision": True,
            "leakage_risk": "low",
        },
        {
            "signal_family": COMPETITION_FAMILY,
            "source_table": "data/public/dataset_modelo_proveedor_v2_candidates.csv",
            "raw_grain": "event_id + proveedor_candidato",
            "target_grain": "event_id + proveedor_candidato",
            "requires_parser_change": False,
            "pre_decision": True,
            "leakage_risk": "low",
        },
        {
            "signal_family": TRANSPORT_AUDIT_FAMILY,
            "source_table": "data/public/support/ofertas_raw_matrix_cells.csv",
            "raw_grain": "sheet cell",
            "target_grain": "not_joined",
            "requires_parser_change": True,
            "pre_decision": True,
            "leakage_risk": "unknown_until_parser",
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in family_specs:
        family_name = spec["signal_family"]
        if family_name == TRANSPORT_AUDIT_FAMILY:
            coverage_train = 0.0
            coverage_test = 0.0
            decision = transport_audit["decision"]
        else:
            family_columns = V3_SIGNAL_FAMILIES[family_name]
            train_cov = [coverage_by_feature[column]["coverage_train"] for column in family_columns]
            test_cov = [coverage_by_feature[column]["coverage_test"] for column in family_columns]
            coverage_train = float(min(train_cov)) if train_cov else 0.0
            coverage_test = float(min(test_cov)) if test_cov else 0.0
            decision = (
                "include_v3_a"
                if coverage_train >= min_coverage and coverage_test >= min_coverage
                else "exclude_low_coverage"
            )

        rows.append(
            {
                **spec,
                "coverage_train": coverage_train,
                "coverage_test": coverage_test,
                "decision": decision,
            }
        )
    return rows


# SECTION: Documentation
def _build_v3_column_dictionary() -> dict[str, dict[str, str]]:
    """Extend the V2 dictionary with V3_A columns and missing-flag roles."""
    base = build_v2_column_dictionary()
    base.update(
        {
            "v3_selected_source_calculos_share": {
                "rol": "feature_quality",
                "justificacion": "Share de registros raw donde la oferta viene seleccionada desde la hoja Cálculos.",
            },
            "v3_cost_source_calculos_share": {
                "rol": "feature_quality",
                "justificacion": "Share de costes mínimos cuyo origen trazado es Cálculos.",
            },
            "v3_reconciliation_conflict_share": {
                "rol": "feature_quality",
                "justificacion": "Proporción de observaciones raw con conflicto entre Tabla y Cálculos.",
            },
            "v3_reconciliation_single_source_share": {
                "rol": "feature_quality",
                "justificacion": "Proporción de observaciones raw donde solo existe una fuente válida.",
            },
            "v3_cost_mean_terminal": {
                "rol": "feature_dispersion",
                "justificacion": "Coste medio observado entre terminales para ese día-producto-proveedor.",
            },
            "v3_cost_std_terminal": {
                "rol": "feature_dispersion",
                "justificacion": "Desviación estándar del coste entre terminales del proveedor.",
            },
            "v3_cost_range_terminal": {
                "rol": "feature_dispersion",
                "justificacion": "Rango de coste entre terminales (`max-min`) para ese proveedor en el día.",
            },
            "v3_cost_cv_terminal": {
                "rol": "feature_dispersion",
                "justificacion": "Coeficiente de variación del coste entre terminales.",
            },
            "v3_share_terminales_min_cost": {
                "rol": "feature_dispersion",
                "justificacion": "Share aproximado de terminales que sostienen el mínimo diario del proveedor.",
            },
            "v3_coste_segundo_evento": {
                "rol": "feature_competition",
                "justificacion": "Segundo mejor coste disponible en el evento.",
            },
            "v3_gap_min_vs_second_evento": {
                "rol": "feature_competition",
                "justificacion": "Diferencia entre el mejor coste del evento y el segundo mejor.",
            },
            "v3_delta_vs_second_evento": {
                "rol": "feature_competition",
                "justificacion": "Diferencia del candidato frente al segundo mejor coste del evento.",
            },
            "v3_ratio_vs_second_evento": {
                "rol": "feature_competition",
                "justificacion": "Ratio del coste del candidato frente al segundo mejor coste del evento.",
            },
            "v3_rank_pct_evento": {
                "rol": "feature_competition",
                "justificacion": "Posición relativa del candidato dentro del ranking de coste del evento.",
            },
            "v3_coste_mean_evento": {
                "rol": "feature_competition",
                "justificacion": "Coste medio de todos los candidatos del evento.",
            },
            "v3_delta_vs_mean_evento": {
                "rol": "feature_competition",
                "justificacion": "Diferencia del candidato frente al coste medio del evento.",
            },
            "v3_candidatos_min_coste_count": {
                "rol": "feature_competition",
                "justificacion": "Número de candidatos empatados en el coste mínimo del evento.",
            },
            "v3_is_unique_min_coste_evento": {
                "rol": "feature_competition",
                "justificacion": "Flag 1/0 que indica si existe un único ganador en el coste mínimo del evento.",
            },
        }
    )
    return base


# SECTION: Documentation
def _write_data_dictionary(output_path: Path, dataset_v3: pd.DataFrame) -> None:
    """Persist the markdown data dictionary for the V3_A dataset."""
    metadata = _build_v3_column_dictionary()
    lines = [
        "# Day04 · Data Dictionary V3_A (raw context)",
        "",
        "## Alcance",
        "- Documento generado automáticamente desde el schema final de `dataset_modelo_proveedor_v3_context.csv`.",
        "- Mantiene contrato V2 y añade señales raw/comparativa/dispersión aprobadas para Day 04.",
        "",
        "| columna | dtype | rol | justificación |",
        "|---|---|---|---|",
    ]

    for column in dataset_v3.columns:
        info = metadata.get(column, {"rol": "unknown", "justificacion": "No documentado."})
        lines.append(
            f"| {column} | {dataset_v3[column].dtype} | {info['rol']} | {info['justificacion']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


# SECTION: Main pipeline
def run(
    *,
    v2_input_path: Path,
    ofertas_typed_input_path: Path,
    raw_matrix_input_path: Path,
    output_dataset_path: Path,
    quality_report_path: Path,
    data_dictionary_path: Path,
    cutoff_date: str,
    min_coverage: float,
    run_id: str | None,
) -> dict[str, Any]:
    """Build the V3_A dataset, its quality report, and the signal decision contract."""
    if not v2_input_path.exists():
        raise FileNotFoundError(f"No existe dataset V2: {v2_input_path}")
    if not ofertas_typed_input_path.exists():
        raise FileNotFoundError(f"No existe ofertas_typed: {ofertas_typed_input_path}")

    execution_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_ts_utc = datetime.now(timezone.utc).isoformat()

    dataset_v2 = pd.read_csv(v2_input_path, keep_default_na=False)
    ofertas_typed = pd.read_csv(ofertas_typed_input_path, keep_default_na=False)

    source_quality = _build_source_quality_frame(ofertas_typed=ofertas_typed)
    event_competition = _build_event_competition_frame(dataset_v2=dataset_v2)

    dataset_v3 = dataset_v2.merge(
        source_quality,
        on=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        how="left",
    ).merge(
        event_competition,
        on=["event_id", "proveedor_candidato"],
        how="left",
    )

    for column in V3_EXTRA_NUM_FEATURE_COLUMNS:
        dataset_v3[column] = pd.to_numeric(dataset_v3[column], errors="coerce")

    coverage_by_feature = _compute_feature_coverage(dataset_v3=dataset_v3, cutoff_date=cutoff_date)
    transport_audit = _build_transport_audit(raw_matrix_path=raw_matrix_input_path)
    signal_decision_table = _build_signal_decision_table(
        coverage_by_feature=coverage_by_feature,
        min_coverage=min_coverage,
        transport_audit=transport_audit,
    )

    missing_flag_columns: list[str] = []
    for column in V3_EXTRA_NUM_FEATURE_COLUMNS:
        if dataset_v3[column].isna().any():
            flag_column = f"{column}_missing_flag"
            dataset_v3[flag_column] = dataset_v3[column].isna().astype(int)
            dataset_v3[column] = dataset_v3[column].fillna(0.0)
            missing_flag_columns.append(flag_column)

    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)
    data_dictionary_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_v3.to_csv(output_dataset_path, index=False, encoding="utf-8")
    _write_data_dictionary(output_path=data_dictionary_path, dataset_v3=dataset_v3)

    positive_per_event = dataset_v3.groupby("event_id")["target_elegido"].sum()
    invalid_positive_events = int((positive_per_event != 1).sum()) if len(positive_per_event) else 0
    feature_cols_num, feature_cols_cat, target_col = get_feature_columns_v3()

    quality_report = {
        "status": "ok",
        "run_id": execution_run_id,
        "generated_ts_utc": generated_ts_utc,
        "cutoff_date": cutoff_date,
        "min_coverage": float(min_coverage),
        "input_v2_dataset": str(v2_input_path),
        "input_ofertas_typed": str(ofertas_typed_input_path),
        "input_raw_matrix": str(raw_matrix_input_path),
        "output_dataset": str(output_dataset_path),
        "output_data_dictionary": str(data_dictionary_path),
        "rows_output": int(len(dataset_v3)),
        "events_output": int(dataset_v3["event_id"].nunique()),
        "rows_vs_v2_match": int(len(dataset_v3) == len(dataset_v2)),
        "events_vs_v2_match": int(dataset_v3["event_id"].nunique() == dataset_v2["event_id"].nunique()),
        "events_with_invalid_positive_count": invalid_positive_events,
        "feature_columns_num": feature_cols_num + missing_flag_columns,
        "feature_columns_cat": feature_cols_cat,
        "target_column": target_col,
        "missing_flag_columns": missing_flag_columns,
        "feature_coverage": coverage_by_feature,
        "signal_decision_table": signal_decision_table,
        "transport_audit": transport_audit,
    }
    quality_report_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return quality_report


# SECTION: CLI entrypoint
def main() -> None:
    """Run the CLI entrypoint for the V3_A dataset builder."""
    args = parse_args()
    summary = run(
        v2_input_path=args.v2_input,
        ofertas_typed_input_path=args.ofertas_typed_input,
        raw_matrix_input_path=args.raw_matrix_input,
        output_dataset_path=args.output_dataset,
        quality_report_path=args.quality_report,
        data_dictionary_path=args.data_dictionary,
        cutoff_date=args.cutoff_date,
        min_coverage=args.min_coverage,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
