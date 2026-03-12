from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.ml.product import recommend_supplier as rs
from src.ml.shared import functions as fc
from src.ml.shared.day05_tabular import V41_TRANSPORT_COLUMNS


BASE_NUMERIC_COLUMNS, BASE_CATEGORICAL_COLUMNS, _ = fc.get_feature_columns_v2()


@dataclass(frozen=True)
class Day06ModeContract:
    """Describe the raw feature contract required to build one scoring matrix."""

    mode_key: str
    raw_numeric_columns: list[str]
    raw_categorical_columns: list[str]
    critical_raw_columns: list[str]


@dataclass(frozen=True)
class Day06ContractInspection:
    """Hold the contract status and diagnostics for one inference mode."""

    mode_key: str
    status: str
    message: str
    user_message: str
    developer_detail: str
    severity: str
    action_hint: str
    expected_feature_columns_total: int
    expected_feature_columns: list[str]
    missing_raw_columns: list[str]
    critical_all_null_columns: list[str]
    matrix_columns_total: int
    matrix_builder: str


class Day06ScoringContractError(ValueError):
    """Raise when one inference mode cannot build its scoring contract safely."""

    def __init__(self, *, mode_key: str, inspection: Day06ContractInspection):
        super().__init__(inspection.message)
        self.mode_key = mode_key
        self.inspection = inspection


def build_contract_user_message(
    *,
    mode_key: str,
    status: str,
    missing_raw_columns: list[str],
    critical_all_null_columns: list[str],
) -> tuple[str, str, str, str]:
    """Translate one technical inspection into a short product-facing message bundle."""
    normalized_status = str(status).upper()
    if normalized_status == "PASS":
        if mode_key == "champion_pure":
            return (
                "El champion está disponible para este input.",
                "Contrato transport-only válido y reconstruible desde el input actual.",
                "success",
                "Puedes seleccionarlo si quieres ejecutar el run con la recomendación principal.",
            )
        return (
            "El modo está disponible para este input.",
            "Contrato V2 base válido para el modo solicitado.",
            "success",
            "Puedes continuar con la validación y la ejecución.",
        )

    if missing_raw_columns:
        if mode_key == "champion_pure":
            return (
                "Faltan datos de transporte para usar el champion.",
                f"Faltan columnas raw requeridas para `{mode_key}`: {missing_raw_columns}",
                "error",
                "Completa el input con esas columnas o usa un modo base disponible.",
            )
        return (
            "Este modo no está disponible para el archivo actual.",
            f"Faltan columnas raw requeridas para `{mode_key}`: {missing_raw_columns}",
            "error",
            "Revisa el input o cambia a un modo habilitado.",
        )

    if critical_all_null_columns:
        if mode_key == "champion_pure":
            return (
                "El champion no puede usarse porque los datos de transporte llegan vacíos.",
                f"Las columnas críticas del champion llegan vacías en todo el input: {critical_all_null_columns}",
                "error",
                "Revisa esos datos o ejecuta un modo base disponible.",
            )
        return (
            "Este modo no tiene señal suficiente para ejecutarse.",
            f"Las columnas críticas del modo llegan vacías en todo el input: {critical_all_null_columns}",
            "error",
            "Completa el input o usa otro modo habilitado.",
        )

    return (
        "No se pudo validar este modo para el input actual.",
        f"El contrato de `{mode_key}` no se pudo validar con el input actual.",
        "warning",
        "Revisa el detalle técnico o cambia a un modo habilitado.",
    )


# SECTION: Mode contracts
def get_mode_contract(mode_key: str) -> Day06ModeContract:
    """Return the raw feature contract for one Day 06 inference mode."""
    if mode_key == "champion_pure":
        return Day06ModeContract(
            mode_key=mode_key,
            raw_numeric_columns=list(BASE_NUMERIC_COLUMNS) + list(V41_TRANSPORT_COLUMNS),
            raw_categorical_columns=list(BASE_CATEGORICAL_COLUMNS),
            critical_raw_columns=list(V41_TRANSPORT_COLUMNS),
        )
    if mode_key in {"baseline", "baseline_with_policy"}:
        return Day06ModeContract(
            mode_key=mode_key,
            raw_numeric_columns=list(BASE_NUMERIC_COLUMNS),
            raw_categorical_columns=list(BASE_CATEGORICAL_COLUMNS),
            critical_raw_columns=[],
        )
    raise ValueError(f"Modo Day 06 no soportado para contrato de scoring: {mode_key}")


# SECTION: Mode inspections
def inspect_mode_contract(
    *,
    input_df: pd.DataFrame,
    expected_feature_columns: list[str],
    mode_key: str,
) -> Day06ContractInspection:
    """Inspect whether one input dataframe can satisfy one mode-specific scoring contract."""
    contract = get_mode_contract(mode_key)
    prepared_df = rs.ensure_event_column(input_df, event_col="event_id")
    incoming_columns = set(prepared_df.columns)
    expected_columns = set(expected_feature_columns)

    if expected_columns.issubset(incoming_columns):
        user_message, developer_detail, severity, action_hint = build_contract_user_message(
            mode_key=mode_key,
            status="PASS",
            missing_raw_columns=[],
            critical_all_null_columns=[],
        )
        return Day06ContractInspection(
            mode_key=mode_key,
            status="PASS",
            message="El input ya contiene exactamente el contrato esperado por metadata.",
            user_message=user_message,
            developer_detail=developer_detail,
            severity=severity,
            action_hint=action_hint,
            expected_feature_columns_total=int(len(expected_feature_columns)),
            expected_feature_columns=list(expected_feature_columns),
            missing_raw_columns=[],
            critical_all_null_columns=[],
            matrix_columns_total=int(len(expected_feature_columns)),
            matrix_builder="expected_feature_columns_direct",
        )

    required_raw_columns = contract.raw_numeric_columns + contract.raw_categorical_columns
    missing_raw_columns = [column for column in required_raw_columns if column not in prepared_df.columns]
    if missing_raw_columns:
        user_message, developer_detail, severity, action_hint = build_contract_user_message(
            mode_key=mode_key,
            status="FAIL",
            missing_raw_columns=missing_raw_columns,
            critical_all_null_columns=[],
        )
        return Day06ContractInspection(
            mode_key=mode_key,
            status="FAIL",
            message=(
                "No se pudo construir la matriz del modo porque faltan columnas raw requeridas: "
                f"{missing_raw_columns}"
            ),
            user_message=user_message,
            developer_detail=developer_detail,
            severity=severity,
            action_hint=action_hint,
            expected_feature_columns_total=int(len(expected_feature_columns)),
            expected_feature_columns=list(expected_feature_columns),
            missing_raw_columns=missing_raw_columns,
            critical_all_null_columns=[],
            matrix_columns_total=0,
            matrix_builder="raw_contract_failed_missing_columns",
        )

    working = prepared_df.copy()
    for column in contract.raw_numeric_columns:
        working[column] = rs.parse_numeric(working[column])
    for column in contract.raw_categorical_columns:
        working[column] = (
            working[column]
            .astype("string")
            .fillna("UNKNOWN")
            .str.strip()
            .replace("", "UNKNOWN")
        )

    critical_all_null_columns: list[str] = []
    for column in contract.critical_raw_columns:
        non_null_count = int(pd.to_numeric(working[column], errors="coerce").notna().sum())
        if non_null_count == 0:
            critical_all_null_columns.append(column)

    if critical_all_null_columns:
        user_message, developer_detail, severity, action_hint = build_contract_user_message(
            mode_key=mode_key,
            status="FAIL",
            missing_raw_columns=[],
            critical_all_null_columns=critical_all_null_columns,
        )
        return Day06ContractInspection(
            mode_key=mode_key,
            status="FAIL",
            message=(
                "El contrato raw existe pero las features críticas del champion llegan vacías en todo el input: "
                f"{critical_all_null_columns}"
            ),
            user_message=user_message,
            developer_detail=developer_detail,
            severity=severity,
            action_hint=action_hint,
            expected_feature_columns_total=int(len(expected_feature_columns)),
            expected_feature_columns=list(expected_feature_columns),
            missing_raw_columns=[],
            critical_all_null_columns=critical_all_null_columns,
            matrix_columns_total=0,
            matrix_builder="raw_contract_failed_all_null_critical_columns",
        )

    matrix = pd.get_dummies(
        working[required_raw_columns],
        drop_first=False,
    ).reindex(columns=expected_feature_columns, fill_value=0.0)
    user_message, developer_detail, severity, action_hint = build_contract_user_message(
        mode_key=mode_key,
        status="PASS",
        missing_raw_columns=[],
        critical_all_null_columns=[],
    )
    return Day06ContractInspection(
        mode_key=mode_key,
        status="PASS",
        message="Contrato raw válido; la matriz se puede reconstruir de forma alineada a metadata.",
        user_message=user_message,
        developer_detail=developer_detail,
        severity=severity,
        action_hint=action_hint,
        expected_feature_columns_total=int(len(expected_feature_columns)),
        expected_feature_columns=list(expected_feature_columns),
        missing_raw_columns=[],
        critical_all_null_columns=[],
        matrix_columns_total=int(len(matrix.columns)),
        matrix_builder="raw_contract_rebuilt_from_candidate_grain",
    )


# SECTION: Matrix builders
def build_mode_feature_matrix(
    *,
    input_df: pd.DataFrame,
    expected_feature_columns: list[str],
    mode_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Day06ContractInspection]:
    """Build one mode-specific feature matrix or raise if the contract is unsafe."""
    inspection = inspect_mode_contract(
        input_df=input_df,
        expected_feature_columns=expected_feature_columns,
        mode_key=mode_key,
    )
    if inspection.status != "PASS":
        raise Day06ScoringContractError(mode_key=mode_key, inspection=inspection)

    contract = get_mode_contract(mode_key)
    prepared_df = rs.ensure_event_column(input_df, event_col="event_id")
    incoming_columns = set(prepared_df.columns)
    expected_columns = set(expected_feature_columns)
    if expected_columns.issubset(incoming_columns):
        matrix = prepared_df[expected_feature_columns].copy()
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return prepared_df, matrix, inspection

    working = prepared_df.copy()
    for column in contract.raw_numeric_columns:
        working[column] = rs.parse_numeric(working[column])
    working[contract.raw_numeric_columns] = working[contract.raw_numeric_columns].fillna(0.0)
    for column in contract.raw_categorical_columns:
        working[column] = (
            working[column]
            .astype("string")
            .fillna("UNKNOWN")
            .str.strip()
            .replace("", "UNKNOWN")
        )

    matrix = pd.get_dummies(
        working[contract.raw_numeric_columns + contract.raw_categorical_columns],
        drop_first=False,
    )
    matrix = matrix.reindex(columns=expected_feature_columns, fill_value=0.0)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return prepared_df, matrix, inspection


# SECTION: Report builders
def build_contract_report(
    *,
    input_df: pd.DataFrame,
    mode_expected_columns: dict[str, list[str]],
) -> dict[str, Any]:
    """Build one reproducible contract report across the active Day 06 modes."""
    report: dict[str, Any] = {
        "input_columns_total": int(len(input_df.columns)),
        "input_columns": list(input_df.columns),
        "modes": {},
    }
    for mode_key, expected_feature_columns in mode_expected_columns.items():
        inspection = inspect_mode_contract(
            input_df=input_df,
            expected_feature_columns=expected_feature_columns,
            mode_key=mode_key,
        )
        report["modes"][mode_key] = {
            "status": inspection.status,
            "message": inspection.message,
            "user_message": inspection.user_message,
            "developer_detail": inspection.developer_detail,
            "severity": inspection.severity,
            "action_hint": inspection.action_hint,
            "expected_feature_columns_total": inspection.expected_feature_columns_total,
            "missing_raw_columns": inspection.missing_raw_columns,
            "critical_all_null_columns": inspection.critical_all_null_columns,
            "matrix_columns_total": inspection.matrix_columns_total,
            "matrix_builder": inspection.matrix_builder,
        }
    return report
