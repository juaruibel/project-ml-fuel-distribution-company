#!/usr/bin/env python3

# LIBRERIAS

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from src.ml.shared.numeric_parsing import parse_numeric_series_locale
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.ml.shared.numeric_parsing import parse_numeric_series_locale

# CONSTANTES

DEFAULT_CONTRACT_PATH = Path("config/inference_input_contract.yaml")
DEFAULT_REPORT_ROOT = Path("artifacts/public/validations/input_daily")
VALID_STATUSES = {"PASS", "PASS_WITH_WARNINGS", "FAIL"}


# FUNCIÓN DE PARSEO DE ARGUMENTOS
def parse_args() -> argparse.Namespace:
    """
    Define argumentos CLI para validar un CSV diario de inferencia.
    """
    parser = argparse.ArgumentParser(
        description="Valida el input diario de inferencia contra el contrato mínimo."
    )
    parser.add_argument("--input-csv", required=True, help="Ruta del CSV diario a validar.")
    parser.add_argument(
        "--contract",
        default=str(DEFAULT_CONTRACT_PATH),
        help="Ruta del contrato YAML de validación.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Ruta opcional de reporte JSON. Si se omite, se genera ruta por defecto.",
    )
    parser.add_argument(
        "--input-name",
        default="",
        help="Nombre lógico del input para trazabilidad (opcional).",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Si está activo, devuelve exit code 1 cuando status=FAIL.",
    )
    return parser.parse_args()


# FUNCIÓN DE CARGA DE CONTRATO
def load_contract(contract_path: Path) -> dict[str, Any]:
    """
    Carga y valida estructura base del contrato YAML.
    """
    if not contract_path.exists():
        raise FileNotFoundError(f"No existe contrato YAML: {contract_path}")

    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Contrato inválido en {contract_path}: se esperaba un objeto YAML.")

    for key in ["grain", "columns", "domain", "coherence", "report"]:
        if key not in payload:
            raise ValueError(f"Contrato inválido: falta sección `{key}`.")

    return payload


# FUNCIÓN DE NORMALIZACIÓN NUMÉRICA
def parse_numeric_series(series: pd.Series) -> pd.Series:
    """
    Convierte serie a numérico aceptando formato europeo con coma decimal.
    """
    return parse_numeric_series_locale(series)


# FUNCIÓN DE NORMALIZACIÓN DE VACÍOS
def is_blank_series(series: pd.Series) -> pd.Series:
    """
    Devuelve máscara booleana de valores vacíos/no informados.
    """
    text = series.astype(str).str.strip().str.lower()
    return series.isna() | text.isin(["", "nan", "none", "null"])


# FUNCIÓN DE LISTA DE COLUMNAS REQUERIDAS
def get_required_columns(contract: dict[str, Any]) -> list[str]:
    """
    Extrae todas las columnas obligatorias a partir del contrato.
    """
    columns_block = contract.get("columns", {})
    required = []
    for bucket in ["date", "categorical_non_empty", "numeric"]:
        required.extend(columns_block.get(bucket, []) or [])
    # Mantener orden y unicidad.
    return list(dict.fromkeys(required))


# FUNCIÓN DE CONSTRUCCIÓN DE ISSUE
def build_issue(
    *,
    code: str,
    message: str,
    column: str | None = None,
    rows_affected: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Construye un issue estándar para errores o warnings.
    """
    payload: dict[str, Any] = {"code": code, "message": message}
    if column is not None:
        payload["column"] = column
    if rows_affected is not None:
        payload["rows_affected"] = int(rows_affected)
    if details:
        payload["details"] = details
    return payload


# FUNCIÓN DE CONSTRUCCIÓN DE RUTA DE REPORTE
def build_default_report_path(report_root: Path, run_id: str) -> Path:
    """
    Construye ruta por defecto para reporte JSON de validación.
    """
    run_date = run_id[:8]
    report_dir = report_root / run_date
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"input_validation_report_{run_id}.json"


# FUNCIÓN DE VALIDACIÓN DE DATAFRAME
def validate_inference_dataframe(
    dataframe: pd.DataFrame,
    contract_path: Path,
    input_name: str = "",
    report_json: Path | None = None,
) -> dict[str, Any]:
    """
    Valida un dataframe de inferencia diaria contra el contrato YAML.
    """
    contract = load_contract(contract_path)
    required_columns = get_required_columns(contract)

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    missing_required = [column for column in required_columns if column not in dataframe.columns]
    if missing_required:
        errors.append(
            build_issue(
                code="missing_required_columns",
                message="Faltan columnas obligatorias para inferencia diaria.",
                details={"missing_columns": missing_required},
            )
        )

    working = dataframe.copy()
    numeric_cache: dict[str, pd.Series] = {}

    # Validaciones de columnas categóricas obligatorias no vacías.
    for column in contract.get("columns", {}).get("categorical_non_empty", []) or []:
        if column not in working.columns:
            continue
        invalid_mask = is_blank_series(working[column])
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            errors.append(
                build_issue(
                    code="null_or_blank_categorical",
                    message="La columna categórica obligatoria contiene valores vacíos.",
                    column=column,
                    rows_affected=invalid_count,
                )
            )

    # Validaciones de fecha parseable.
    for column in contract.get("columns", {}).get("date", []) or []:
        if column not in working.columns:
            continue
        parsed = pd.to_datetime(working[column], errors="coerce")
        invalid_mask = parsed.isna()
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            errors.append(
                build_issue(
                    code="invalid_date",
                    message="La columna de fecha contiene valores no parseables.",
                    column=column,
                    rows_affected=invalid_count,
                )
            )

    # Validaciones numéricas base.
    for column in contract.get("columns", {}).get("numeric", []) or []:
        if column not in working.columns:
            continue
        parsed_numeric = parse_numeric_series(working[column])
        numeric_cache[column] = parsed_numeric
        invalid_mask = parsed_numeric.isna()
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            errors.append(
                build_issue(
                    code="invalid_numeric",
                    message="La columna numérica obligatoria contiene valores no convertibles.",
                    column=column,
                    rows_affected=invalid_count,
                )
            )

    # Validación de grano (duplicados por clave).
    key_columns = (contract.get("grain", {}) or {}).get("key_columns", []) or []
    if key_columns and all(column in working.columns for column in key_columns):
        duplicate_mask = working.duplicated(subset=key_columns, keep=False)
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count > 0:
            errors.append(
                build_issue(
                    code="duplicate_grain_rows",
                    message="Se detectaron duplicados en el grano obligatorio.",
                    rows_affected=duplicate_count,
                    details={"key_columns": key_columns},
                )
            )

    # Validaciones de dominio.
    for column, rules in (contract.get("domain", {}) or {}).items():
        if column not in working.columns:
            continue
        series_num = numeric_cache.get(column, parse_numeric_series(working[column]))
        numeric_cache[column] = series_num

        if "allowed" in rules:
            allowed_values = {float(value) for value in rules.get("allowed", [])}
            invalid_mask = series_num.notna() & ~series_num.isin(allowed_values)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                errors.append(
                    build_issue(
                        code="domain_allowed_values",
                        message="La columna contiene valores fuera del dominio permitido.",
                        column=column,
                        rows_affected=invalid_count,
                        details={"allowed": sorted(allowed_values)},
                    )
                )

        if "min_inclusive" in rules:
            min_inclusive = float(rules["min_inclusive"])
            invalid_mask = series_num.notna() & (series_num < min_inclusive)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                errors.append(
                    build_issue(
                        code="domain_min_inclusive",
                        message="La columna incumple mínimo inclusivo.",
                        column=column,
                        rows_affected=invalid_count,
                        details={"min_inclusive": min_inclusive},
                    )
                )

        if "max_inclusive" in rules:
            max_inclusive = float(rules["max_inclusive"])
            invalid_mask = series_num.notna() & (series_num > max_inclusive)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                errors.append(
                    build_issue(
                        code="domain_max_inclusive",
                        message="La columna incumple máximo inclusivo.",
                        column=column,
                        rows_affected=invalid_count,
                        details={"max_inclusive": max_inclusive},
                    )
                )

        if "min_exclusive" in rules:
            min_exclusive = float(rules["min_exclusive"])
            invalid_mask = series_num.notna() & (series_num <= min_exclusive)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                errors.append(
                    build_issue(
                        code="domain_min_exclusive",
                        message="La columna incumple mínimo exclusivo.",
                        column=column,
                        rows_affected=invalid_count,
                        details={"min_exclusive": min_exclusive},
                    )
                )

    # Validaciones de coherencia entre columnas.
    for rule in (contract.get("coherence", []) or []):
        rule_id = str(rule.get("rule_id", "coherence_rule"))
        left = str(rule.get("left", "")).strip()
        operator = str(rule.get("operator", "")).strip()

        if left == "" or operator == "":
            continue
        if left not in working.columns:
            continue

        left_series = numeric_cache.get(left, parse_numeric_series(working[left]))
        numeric_cache[left] = left_series

        if "right" in rule:
            right = str(rule.get("right", "")).strip()
            if right == "" or right not in working.columns:
                continue
            right_series = numeric_cache.get(right, parse_numeric_series(working[right]))
            numeric_cache[right] = right_series

            if operator == "<=":
                invalid_mask = left_series.notna() & right_series.notna() & ~(left_series <= right_series)
            elif operator == ">=":
                invalid_mask = left_series.notna() & right_series.notna() & ~(left_series >= right_series)
            else:
                continue
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                errors.append(
                    build_issue(
                        code="coherence_rule_failed",
                        message="Regla de coherencia entre columnas incumplida.",
                        rows_affected=invalid_count,
                        details={"rule_id": rule_id, "left": left, "operator": operator, "right": right},
                    )
                )
            continue

        if "value" in rule:
            value = float(rule.get("value"))
            if operator == ">=":
                invalid_mask = left_series.notna() & ~(left_series >= value)
            elif operator == "<=":
                invalid_mask = left_series.notna() & ~(left_series <= value)
            else:
                continue
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                errors.append(
                    build_issue(
                        code="coherence_rule_failed",
                        message="Regla de coherencia con valor fijo incumplida.",
                        rows_affected=invalid_count,
                        details={"rule_id": rule_id, "left": left, "operator": operator, "value": value},
                    )
                )

    # Warnings por columnas extra.
    warn_on_extra = bool((contract.get("report", {}) or {}).get("warn_on_extra_columns", True))
    if warn_on_extra:
        extra_columns = sorted(list(set(working.columns) - set(required_columns)))
        if extra_columns:
            warnings.append(
                build_issue(
                    code="extra_columns_detected",
                    message="Se detectaron columnas extra no obligatorias en el contrato.",
                    details={"extra_columns": extra_columns},
                )
            )

    if errors:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    if status not in VALID_STATUSES:
        raise ValueError(f"Status de validación inválido: {status}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "status": status,
        "run_id": run_id,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "input_name": input_name,
        "contract_path": str(contract_path.resolve()),
        "rows_total": int(len(working)),
        "columns_total": int(len(working.columns)),
        "required_columns_total": int(len(required_columns)),
        "error_count": int(len(errors)),
        "warning_count": int(len(warnings)),
        "errors": errors,
        "warnings": warnings,
    }

    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_json"] = str(report_json.resolve())

    return report


# FUNCIÓN DE VALIDACIÓN DE CSV
def validate_inference_csv(
    input_csv: Path,
    contract_path: Path,
    input_name: str = "",
    report_json: Path | None = None,
) -> dict[str, Any]:
    """
    Valida un CSV diario cargándolo como dataframe de texto.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe CSV de entrada: {input_csv}")
    frame = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    resolved_name = input_name or str(input_csv.resolve())
    return validate_inference_dataframe(
        dataframe=frame,
        contract_path=contract_path,
        input_name=resolved_name,
        report_json=report_json,
    )


# FUNCIÓN MAIN
def main() -> None:
    """
    Ejecuta la validación de contrato desde CLI y muestra resumen en stdout.
    """
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    contract_path = Path(args.contract).resolve()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_json = (
        Path(args.report_json).resolve()
        if args.report_json
        else build_default_report_path(DEFAULT_REPORT_ROOT.resolve(), run_id=run_id)
    )

    report = validate_inference_csv(
        input_csv=input_csv,
        contract_path=contract_path,
        input_name=args.input_name,
        report_json=report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.fail_on_error and report.get("status") == "FAIL":
        raise SystemExit(1)


# MAIN GUARD
if __name__ == "__main__":
    main()
