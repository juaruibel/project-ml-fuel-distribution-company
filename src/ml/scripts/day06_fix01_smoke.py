#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.ml.product.day06_excel_raw import (
        build_excel_candidate_grain,
        build_excel_enrichment_template,
        normalize_excel_enrichment_frame,
        parse_excel_raw_workbook,
        validate_excel_enrichment,
    )
    from src.ml.product.day06_runtime import (
        Day06RunContractError,
        execute_operational_run,
        get_mode_spec,
        inspect_input_mode_availability,
        load_mode_bundle,
        normalize_operational_input,
        prepare_operational_input,
    )
    from src.ml.product.day06_scoring_contracts import (
        build_contract_report,
        build_mode_feature_matrix,
        inspect_mode_contract,
    )
    from src.ml.shared.day05_tabular import V41_TRANSPORT_COLUMNS
    from src.ml.shared.project_paths import DATA_RAW_DIR, REPORTS_DIR, SAMPLE_INPUT_PATH
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.ml.product.day06_excel_raw import (
        build_excel_candidate_grain,
        build_excel_enrichment_template,
        normalize_excel_enrichment_frame,
        parse_excel_raw_workbook,
        validate_excel_enrichment,
    )
    from src.ml.product.day06_runtime import (
        Day06RunContractError,
        execute_operational_run,
        get_mode_spec,
        inspect_input_mode_availability,
        load_mode_bundle,
        normalize_operational_input,
        prepare_operational_input,
    )
    from src.ml.product.day06_scoring_contracts import (
        build_contract_report,
        build_mode_feature_matrix,
        inspect_mode_contract,
    )
    from src.ml.shared.day05_tabular import V41_TRANSPORT_COLUMNS
    from src.ml.shared.project_paths import DATA_RAW_DIR, REPORTS_DIR, SAMPLE_INPUT_PATH


DEFAULT_WORKBOOK_PATH = (
    DATA_RAW_DIR
    / "SUPPLIER_DAILY_COMPARISON"
    / "2015 COMPARATIVA DE PRECIOS"
    / "2015 -10 OCTUBRE"
    / "Comparativa de precios 21-10-2015.xlsx"
)
from datetime import datetime, timezone as _tz

_SMOKE_TS = datetime.now(_tz.utc).strftime("%Y%m%dT%H%M%SZ")
DEFAULT_REPORT_PATH = (
    REPORTS_DIR / "validations" / "day06_fix01" / f"{_SMOKE_TS}_day06_fix01_mode_availability_smoke_report.json"
)


# SECTION: CLI helpers
def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments for the Day 06.fix01 smoke validation."""
    parser = argparse.ArgumentParser(
        description="Ejecuta el smoke contractual Day 06.fix01 sobre un workbook real."
    )
    parser.add_argument(
        "--workbook",
        default=str(DEFAULT_WORKBOOK_PATH),
        help="Workbook `Comparativa de precios` a validar.",
    )
    parser.add_argument(
        "--report-json",
        default=str(DEFAULT_REPORT_PATH),
        help="Ruta del reporte JSON de salida.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k operativo para los runs del smoke.",
    )
    return parser.parse_args()


# SECTION: Assertion helpers
def assert_condition(condition: bool, message: str) -> None:
    """Raise a deterministic assertion error for one smoke requirement."""
    if not condition:
        raise AssertionError(message)


# SECTION: Enrichment helpers
def build_smoke_enrichment(bundle: Any) -> pd.DataFrame:
    """Build a deterministic manual enrichment table for the workbook smoke."""
    enrichment_df = build_excel_enrichment_template(bundle)
    enrichment_df = enrichment_df.copy()
    enrichment_df["albaran_id"] = "FIX01_SMOKE_ALBARAN_001"
    enrichment_df["linea_id"] = [f"{index + 1:03d}" for index in range(len(enrichment_df))]
    enrichment_df["litros_evento"] = [float(10000 + (index * 250)) for index in range(len(enrichment_df))]
    return normalize_excel_enrichment_frame(enrichment_df)


# SECTION: Enrichment helpers
def build_manual_mode_smoke_input() -> pd.DataFrame:
    """Build one conservative manual input sample without transport-only columns."""
    return pd.DataFrame(
        [
            {
                "event_id": "MANUAL_FIX01_MODE_001",
                "fecha_evento": "2029-02-27",
                "albaran_id": "ALB_MANUAL_FIX01_MODE_001",
                "linea_id": "1",
                "proveedor_candidato": "SUPPLIER_009",
                "producto_canonico": "PRODUCT_002",
                "terminal_compra": "TERMINAL_001",
                "coste_min_dia_proveedor": 0.9000,
                "rank_coste_dia_producto": 1,
                "terminales_cubiertos": 6,
                "observaciones_oferta": 8,
                "candidatos_evento_count": 10,
                "coste_min_evento": 0.9000,
                "coste_max_evento": 0.9600,
                "spread_coste_evento": 0.0600,
                "delta_vs_min_evento": 0.0000,
                "ratio_vs_min_evento": 1.0000,
                "litros_evento": 33000,
                "dia_semana": 4,
                "mes": 2,
                "fin_mes": 0,
                "blocked_by_rule_candidate": 0,
            },
            {
                "event_id": "MANUAL_FIX01_MODE_001",
                "fecha_evento": "2029-02-27",
                "albaran_id": "ALB_MANUAL_FIX01_MODE_001",
                "linea_id": "1",
                "proveedor_candidato": "SUPPLIER_050",
                "producto_canonico": "PRODUCT_002",
                "terminal_compra": "TERMINAL_001",
                "coste_min_dia_proveedor": 0.9120,
                "rank_coste_dia_producto": 2,
                "terminales_cubiertos": 6,
                "observaciones_oferta": 8,
                "candidatos_evento_count": 10,
                "coste_min_evento": 0.9000,
                "coste_max_evento": 0.9600,
                "spread_coste_evento": 0.0600,
                "delta_vs_min_evento": 0.0120,
                "ratio_vs_min_evento": 1.0133,
                "litros_evento": 33000,
                "dia_semana": 4,
                "mes": 2,
                "fin_mes": 0,
                "blocked_by_rule_candidate": 0,
            },
        ]
    )


# SECTION: Runtime helpers
def run_operational_mode(
    *,
    input_df: pd.DataFrame,
    workbook_path: Path,
    workbook_bytes: bytes,
    input_mode: str,
    inference_mode: str,
    top_k: int,
) -> dict[str, Any]:
    """Prepare the input, execute one Day 06 operational mode and return artifact presence checks."""
    prepared_input = prepare_operational_input(
        input_df=input_df,
        input_mode=input_mode,
        source_name=workbook_path.name,
        source_suffix=workbook_path.suffix,
        source_bytes=workbook_bytes,
    )
    run_bundle = execute_operational_run(
        prepared_input=prepared_input,
        inference_mode=inference_mode,
        top_k=top_k,
        surface="smoke_fix01",
    )
    manifest_payload = run_bundle["manifest_payload"]
    return {
        "run_id": manifest_payload["run_id"],
        "inference_mode": inference_mode,
        "validation_status": manifest_payload["validation_status"],
        "scoring_status": manifest_payload["scoring_status"],
        "input_original_path": manifest_payload["input_original_path"],
        "input_candidate_grain_path": manifest_payload["input_candidate_grain_path"],
        "input_normalized_path": manifest_payload["input_normalized_path"],
        "detail_csv": manifest_payload["detail_csv"],
        "resumen_evento_csv": manifest_payload["resumen_evento_csv"],
        "resumen_albaran_csv": manifest_payload["resumen_albaran_csv"],
        "feedback_csv": manifest_payload["feedback_csv"],
        "run_manifest_path": str(run_bundle["run_manifest_path"]),
        "scoring_contract_report_path": str(run_bundle["scoring_contract_report_path"]),
        "paths_exist": {
            "input_original": Path(manifest_payload["input_original_path"]).exists(),
            "candidate_grain": Path(manifest_payload["input_candidate_grain_path"]).exists(),
            "normalized": Path(manifest_payload["input_normalized_path"]).exists(),
            "detail": Path(manifest_payload["detail_csv"]).exists(),
            "resumen_evento": Path(manifest_payload["resumen_evento_csv"]).exists(),
            "resumen_albaran": Path(manifest_payload["resumen_albaran_csv"]).exists(),
            "feedback": Path(manifest_payload["feedback_csv"]).exists(),
            "run_manifest": Path(run_bundle["run_manifest_path"]).exists(),
            "scoring_contract_report": Path(run_bundle["scoring_contract_report_path"]).exists(),
        },
    }


# SECTION: Runtime helpers
def run_negative_champion_contract(
    *,
    input_df: pd.DataFrame,
    workbook_path: Path,
    workbook_bytes: bytes,
    top_k: int,
) -> dict[str, Any]:
    """Run the explicit negative case where champion transport columns are removed pre-scoring."""
    prepared_input = prepare_operational_input(
        input_df=input_df,
        input_mode="excel_raw_negative_contract",
        source_name=workbook_path.name,
        source_suffix=workbook_path.suffix,
        source_bytes=workbook_bytes,
    )
    try:
        execute_operational_run(
            prepared_input=prepared_input,
            inference_mode="champion_pure",
            top_k=top_k,
            surface="smoke_fix01_negative",
        )
    except Day06RunContractError as error:
        manifest_payload = error.manifest_payload
        detail_csv = str(manifest_payload.get("detail_csv", ""))
        resumen_evento_csv = str(manifest_payload.get("resumen_evento_csv", ""))
        resumen_albaran_csv = str(manifest_payload.get("resumen_albaran_csv", ""))
        feedback_csv = str(manifest_payload.get("feedback_csv", ""))
        return {
            "status": "EXPECTED_FAIL",
            "message": error.message,
            "run_manifest_path": str(error.run_manifest_path),
            "contract_report_path": str(error.contract_report_path),
            "scoring_status": manifest_payload.get("scoring_status", ""),
            "fallback_modes": manifest_payload.get("fallback_modes", []),
            "detail_csv": detail_csv,
            "resumen_evento_csv": resumen_evento_csv,
            "resumen_albaran_csv": resumen_albaran_csv,
            "feedback_csv": feedback_csv,
            "paths_exist": {
                "input_original": Path(str(manifest_payload["input_original_path"])).exists(),
                "candidate_grain": Path(str(manifest_payload["input_candidate_grain_path"])).exists(),
                "normalized": Path(str(manifest_payload["input_normalized_path"])).exists(),
                "run_manifest": Path(str(error.run_manifest_path)).exists(),
                "contract_report": Path(str(error.contract_report_path)).exists(),
            },
            "paths_empty": {
                "detail_csv": detail_csv == "",
                "resumen_evento_csv": resumen_evento_csv == "",
                "resumen_albaran_csv": resumen_albaran_csv == "",
                "feedback_csv": feedback_csv == "",
            },
        }

    raise AssertionError("El caso negativo del champion debería fallar pre-scoring y no lo hizo.")


# SECTION: Report builders
def build_mode_inventory(normalized_df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Build the contract inventory for baseline, champion and baseline_with_policy."""
    mode_expected_columns = {
        mode_key: load_mode_bundle(
            str(get_mode_spec(mode_key).model_path),
            str(get_mode_spec(mode_key).metadata_path),
        )[2]
        for mode_key in ["baseline", "champion_pure", "baseline_with_policy"]
    }
    contract_report = build_contract_report(
        input_df=normalized_df,
        mode_expected_columns=mode_expected_columns,
    )
    return contract_report, mode_expected_columns


# SECTION: Report builders
def summarize_mode_case(mode_availability: dict[str, Any]) -> dict[str, Any]:
    """Keep only the mode-availability fields required by the Day 06.fix01 reopening smoke."""
    return {
        "input_mode": mode_availability["input_mode"],
        "input_columns_total": mode_availability["input_columns_total"],
        "input_columns": mode_availability["input_columns"],
        "normalized_columns_total": mode_availability["normalized_columns_total"],
        "normalized_columns": mode_availability["normalized_columns"],
        "validation_summary": mode_availability["validation_summary"],
        "policy_grouping_available": mode_availability["policy_grouping_available"],
        "policy_grouping_reason": mode_availability["policy_grouping_reason"],
        "policy_grouped_albaran_count": mode_availability["policy_grouped_albaran_count"],
        "policy_max_events_per_albaran": mode_availability["policy_max_events_per_albaran"],
        "selected_default_mode": mode_availability["selected_default_mode"],
        "enabled_mode_keys": mode_availability["enabled_mode_keys"],
        "mode_catalog": mode_availability["mode_catalog"],
        "contract_report": mode_availability["contract_report"],
    }


# SECTION: Report builders
def write_report(report_path: Path, payload: dict[str, Any]) -> None:
    """Persist the smoke report JSON with deterministic formatting."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# SECTION: Smoke runner
def main() -> None:
    """Execute the Day 06.fix01 smoke and persist a reproducible report."""
    args = parse_args()
    workbook_path = Path(args.workbook).resolve()
    report_path = Path(args.report_json).resolve()
    top_k = int(args.top_k)

    assert_condition(workbook_path.exists(), f"No existe workbook smoke: {workbook_path}")
    workbook_bytes = workbook_path.read_bytes()
    bundle = parse_excel_raw_workbook(
        source_name=workbook_path.name,
        source_suffix=workbook_path.suffix,
        source_bytes=workbook_bytes,
    )
    enrichment_df = build_smoke_enrichment(bundle)
    enrichment_status = validate_excel_enrichment(enrichment_df)
    assert_condition(enrichment_status["status"] == "PASS", enrichment_status["message"])

    candidate_df = build_excel_candidate_grain(
        bundle=bundle,
        enrichment_df=enrichment_df,
    )
    normalized_df = normalize_operational_input(candidate_df)
    contract_report, mode_expected_columns = build_mode_inventory(normalized_df)

    baseline_prepared_df, baseline_matrix, baseline_inspection = build_mode_feature_matrix(
        input_df=normalized_df,
        expected_feature_columns=mode_expected_columns["baseline"],
        mode_key="baseline",
    )
    champion_prepared_df, champion_matrix, champion_inspection = build_mode_feature_matrix(
        input_df=normalized_df,
        expected_feature_columns=mode_expected_columns["champion_pure"],
        mode_key="champion_pure",
    )

    negative_input_df = candidate_df.drop(columns=V41_TRANSPORT_COLUMNS, errors="ignore")
    negative_normalized_df = normalize_operational_input(negative_input_df)
    champion_negative_inspection = inspect_mode_contract(
        input_df=negative_normalized_df,
        expected_feature_columns=mode_expected_columns["champion_pure"],
        mode_key="champion_pure",
    )
    csv_input_df = pd.read_csv(SAMPLE_INPUT_PATH, keep_default_na=False)
    manual_input_df = build_manual_mode_smoke_input()
    no_mode_input_df = manual_input_df.drop(columns=["terminal_compra"])

    excel_mode_availability = inspect_input_mode_availability(
        input_df=candidate_df,
        input_mode="excel_raw",
    )
    csv_mode_availability = inspect_input_mode_availability(
        input_df=csv_input_df,
        input_mode="csv",
    )
    manual_mode_availability = inspect_input_mode_availability(
        input_df=manual_input_df,
        input_mode="manual",
    )
    policy_mode_availability = inspect_input_mode_availability(
        input_df=negative_input_df,
        input_mode="excel_raw_missing_transport",
    )
    no_mode_availability = inspect_input_mode_availability(
        input_df=no_mode_input_df,
        input_mode="manual_broken_contract",
    )

    assert_condition(
        contract_report["modes"]["baseline"]["status"] == "PASS",
        "El baseline debería pasar el contrato Day 06.fix01.",
    )
    assert_condition(
        str(enrichment_df["albaran_id"].dtype) == "string",
        "`albaran_id` del enrichment debería quedar tipado como string nullable.",
    )
    assert_condition(
        str(enrichment_df["linea_id"].dtype) == "string",
        "`linea_id` del enrichment debería quedar tipado como string nullable.",
    )
    assert_condition(
        str(enrichment_df["litros_evento"].dtype) == "Float64",
        "`litros_evento` del enrichment debería quedar tipado como Float64 nullable.",
    )
    assert_condition(
        str(candidate_df["albaran_id"].dtype) == "string",
        "`albaran_id` del candidate-grain debería quedar tipado como string nullable.",
    )
    assert_condition(
        str(candidate_df["linea_id"].dtype) == "string",
        "`linea_id` del candidate-grain debería quedar tipado como string nullable.",
    )
    assert_condition(
        str(candidate_df["litros_evento"].dtype) == "Float64",
        "`litros_evento` del candidate-grain debería quedar tipado como Float64 nullable.",
    )
    assert_condition(
        str(candidate_df["precio_unitario_evento"].dtype) == "Float64",
        "`precio_unitario_evento` placeholder debería quedar tipado como Float64 nullable.",
    )
    assert_condition(
        str(candidate_df["importe_total_evento"].dtype) == "Float64",
        "`importe_total_evento` placeholder debería quedar tipado como Float64 nullable.",
    )
    assert_condition(
        str(candidate_df["target_elegido"].dtype) == "Int64",
        "`target_elegido` placeholder debería quedar tipado como Int64 nullable.",
    )
    assert_condition(
        contract_report["modes"]["champion_pure"]["status"] == "PASS",
        "El champion debería pasar el contrato transport-only en el smoke positivo.",
    )
    assert_condition(
        champion_negative_inspection.status == "FAIL",
        "El caso negativo del champion debería fallar cuando faltan features transport-only.",
    )
    assert_condition(
        excel_mode_availability["selected_default_mode"] == "champion_pure",
        "Excel raw con contrato completo debería recomendar `champion_pure`.",
    )
    assert_condition(
        any(
            entry["mode_key"] == "champion_pure" and entry["enabled"] and entry["recommended"]
            for entry in excel_mode_availability["mode_catalog"]
        ),
        "Excel raw con contrato completo debería dejar `champion_pure` habilitado y recomendado.",
    )
    assert_condition(
        csv_mode_availability["selected_default_mode"] == "baseline",
        "El CSV sample sin transporte debería caer a `baseline` como default conservador.",
    )
    assert_condition(
        any(
            entry["mode_key"] == "champion_pure" and not entry["enabled"]
            for entry in csv_mode_availability["mode_catalog"]
        ),
        "El CSV sample sin transporte debería deshabilitar `champion_pure`.",
    )
    assert_condition(
        manual_mode_availability["selected_default_mode"] == "baseline",
        "El input manual conservador debería recomendar `baseline`.",
    )
    assert_condition(
        any(
            entry["mode_key"] == "champion_pure" and not entry["enabled"]
            for entry in manual_mode_availability["mode_catalog"]
        ),
        "El input manual sin transporte debería deshabilitar `champion_pure`.",
    )
    assert_condition(
        policy_mode_availability["selected_default_mode"] == "baseline_with_policy",
        "Sin champion pero con agrupación útil por albarán, el default debería ser `baseline_with_policy`.",
    )
    assert_condition(
        not any(
            entry["mode_key"] == "champion_pure" and entry["enabled"]
            for entry in policy_mode_availability["mode_catalog"]
        ),
        "El caso policy fallback no debería habilitar `champion_pure`.",
    )
    assert_condition(
        no_mode_availability["selected_default_mode"] is None
        and len(no_mode_availability["enabled_mode_keys"]) == 0,
        "El caso sin contrato debería dejar todos los modos deshabilitados.",
    )

    baseline_run = run_operational_mode(
        input_df=candidate_df,
        workbook_path=workbook_path,
        workbook_bytes=workbook_bytes,
        input_mode="excel_raw",
        inference_mode="baseline",
        top_k=top_k,
    )
    champion_run = run_operational_mode(
        input_df=candidate_df,
        workbook_path=workbook_path,
        workbook_bytes=workbook_bytes,
        input_mode="excel_raw",
        inference_mode="champion_pure",
        top_k=top_k,
    )
    policy_run = run_operational_mode(
        input_df=candidate_df,
        workbook_path=workbook_path,
        workbook_bytes=workbook_bytes,
        input_mode="excel_raw",
        inference_mode="baseline_with_policy",
        top_k=top_k,
    )
    negative_run = run_negative_champion_contract(
        input_df=negative_input_df,
        workbook_path=workbook_path,
        workbook_bytes=workbook_bytes,
        top_k=top_k,
    )

    assert_condition(
        all(baseline_run["paths_exist"].values()),
        "El run baseline no persistió todos los artefactos esperados.",
    )
    assert_condition(
        all(champion_run["paths_exist"].values()),
        "El run champion no persistió todos los artefactos esperados.",
    )
    assert_condition(
        all(policy_run["paths_exist"].values()),
        "El run baseline_with_policy no persistió todos los artefactos esperados.",
    )
    assert_condition(
        all(negative_run["paths_exist"].values()),
        "El run negativo del champion no persistió los artefactos pre-scoring requeridos.",
    )
    assert_condition(
        all(negative_run["paths_empty"].values()),
        "El run negativo del champion no debería publicar outputs ni feedback.",
    )

    report_payload = {
        "smoke_name": "day06_fix01_mode_availability_smoke",
        "workbook_path": str(workbook_path),
        "csv_sample_path": str(SAMPLE_INPUT_PATH),
        "parse_summary": bundle.parse_summary,
        "sheet_names": bundle.sheet_names,
        "selected_table_sheet": bundle.selected_table_sheet,
        "selected_calculos_sheet": bundle.selected_calculos_sheet,
        "enrichment_status": enrichment_status,
        "enrichment_dtypes": {column: str(dtype) for column, dtype in enrichment_df.dtypes.items()},
        "candidate_grain_columns_total": int(len(candidate_df.columns)),
        "candidate_grain_columns": list(candidate_df.columns),
        "candidate_grain_dtypes": {column: str(dtype) for column, dtype in candidate_df.dtypes.items()},
        "normalized_columns_total": int(len(normalized_df.columns)),
        "normalized_columns": list(normalized_df.columns),
        "normalized_dtypes": {column: str(dtype) for column, dtype in normalized_df.dtypes.items()},
        "baseline_contract": {
            "status": baseline_inspection.status,
            "message": baseline_inspection.message,
            "expected_feature_columns_total": len(mode_expected_columns["baseline"]),
            "expected_feature_columns": mode_expected_columns["baseline"],
            "matrix_columns_total": int(len(baseline_matrix.columns)),
        },
        "champion_contract": {
            "status": champion_inspection.status,
            "message": champion_inspection.message,
            "expected_feature_columns_total": len(mode_expected_columns["champion_pure"]),
            "expected_feature_columns": mode_expected_columns["champion_pure"],
            "matrix_columns_total": int(len(champion_matrix.columns)),
            "critical_transport_columns": list(V41_TRANSPORT_COLUMNS),
        },
        "champion_negative_contract": {
            "status": champion_negative_inspection.status,
            "message": champion_negative_inspection.message,
            "missing_raw_columns": champion_negative_inspection.missing_raw_columns,
            "critical_all_null_columns": champion_negative_inspection.critical_all_null_columns,
        },
        "contract_report": contract_report,
        "mode_availability_cases": {
            "excel_raw_complete_contract": summarize_mode_case(excel_mode_availability),
            "csv_without_transport": summarize_mode_case(csv_mode_availability),
            "manual_conservative": summarize_mode_case(manual_mode_availability),
            "policy_recommended_without_champion": summarize_mode_case(policy_mode_availability),
            "no_modes_available": summarize_mode_case(no_mode_availability),
        },
        "matrix_preview": {
            "baseline_rows": int(len(baseline_prepared_df)),
            "baseline_columns": int(len(baseline_matrix.columns)),
            "champion_rows": int(len(champion_prepared_df)),
            "champion_columns": int(len(champion_matrix.columns)),
        },
        "operational_runs": {
            "baseline": baseline_run,
            "champion_pure": champion_run,
            "baseline_with_policy": policy_run,
            "champion_negative": negative_run,
        },
    }
    write_report(report_path, report_payload)
    print(report_path)


if __name__ == "__main__":
    main()
