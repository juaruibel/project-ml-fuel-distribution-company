from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.product import recommend_supplier as rs
from src.ml.product import validate_inference_input as vinput
from src.ml.product.day06_normalization import normalize_operational_input
from src.ml.product.day06_scoring_contracts import build_contract_report
from src.ml.shared.project_paths import (
    BASELINE_METADATA_PATH,
    BASELINE_MODEL_PATH,
    CHAMPION_METADATA_PATH,
    CHAMPION_MODEL_PATH,
    INPUT_CONTRACT_PATH,
)


@dataclass(frozen=True)
class InferenceModeSpec:
    """Describe one Day 06 inference mode exposed in the product surface."""

    key: str
    label: str
    description: str
    rollout_status: str
    model_path: Path
    metadata_path: Path
    uses_assist_policy: bool = False


@lru_cache(maxsize=8)
def load_mode_bundle(model_path: str, metadata_path: str) -> tuple[Any, dict[str, Any], list[str]]:
    """Load one model bundle once per process."""
    return rs.load_model_bundle(
        model_path=Path(model_path),
        metadata_path=Path(metadata_path),
    )


MODE_SPECS = {
    "baseline": InferenceModeSpec(
        key="baseline",
        label="Histórico congelado · LR_smote_0.5",
        description="Serving default histórico congelado. Referencia de despliegue actual.",
        rollout_status="historical_default",
        model_path=BASELINE_MODEL_PATH,
        metadata_path=BASELINE_METADATA_PATH,
    ),
    "champion_pure": InferenceModeSpec(
        key="champion_pure",
        label="Recomendado para rollout · Champion puro",
        description="Referencia conservadora recomendada para el siguiente rollout explícito.",
        rollout_status="recommended_rollout",
        model_path=CHAMPION_MODEL_PATH,
        metadata_path=CHAMPION_METADATA_PATH,
    ),
    "baseline_with_policy": InferenceModeSpec(
        key="baseline_with_policy",
        label="Alternativa operativa · LR + PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009",
        description="Alternativa operativa visible con capa determinista vigente.",
        rollout_status="operational_alternative",
        model_path=BASELINE_MODEL_PATH,
        metadata_path=BASELINE_METADATA_PATH,
        uses_assist_policy=True,
    ),
}


def build_validation_user_message(validation_summary: dict[str, Any]) -> dict[str, str]:
    """Translate one validation summary into a short user-facing status bundle."""
    status = str(validation_summary.get("status", "")).upper()
    errors = [
        str(issue.get("message", "")).strip()
        for issue in validation_summary.get("errors", [])
        if str(issue.get("message", "")).strip() != ""
    ]
    warnings = [
        str(issue.get("message", "")).strip()
        for issue in validation_summary.get("warnings", [])
        if str(issue.get("message", "")).strip() != ""
    ]
    if status == "PASS":
        return {
            "user_message": "Input válido. Puedes continuar.",
            "developer_detail": "Contrato mínimo del input validado sin errores ni warnings.",
            "severity": "success",
            "action_hint": "Puedes ejecutar el run con cualquier modo habilitado.",
        }
    if status == "PASS_WITH_WARNINGS":
        warning_preview = warnings[:2]
        detail = " ".join(warning_preview) if warning_preview else "Validación correcta con advertencias no bloqueantes."
        return {
            "user_message": "Input válido con advertencias.",
            "developer_detail": detail,
            "severity": "warning",
            "action_hint": "Puedes continuar, pero conviene revisar las advertencias antes de cerrar el run.",
        }
    error_preview = errors[:2]
    detail = " ".join(error_preview) if error_preview else "El input incumple el contrato mínimo de inferencia."
    return {
        "user_message": "El input no está listo para ejecutar.",
        "developer_detail": detail,
        "severity": "error",
        "action_hint": "Corrige los errores del input antes de lanzar el run operativo.",
    }


def build_mode_availability_message(
    *,
    mode_key: str,
    enabled: bool,
    recommended: bool,
    validation_message: dict[str, str] | None,
    mode_report: dict[str, Any],
    policy_grouping: dict[str, Any],
) -> dict[str, str]:
    """Build the product-facing and technical explanation for one mode entry."""
    if validation_message is not None:
        return {
            "user_reason": str(validation_message["user_message"]),
            "developer_detail": str(validation_message["developer_detail"]),
            "severity": str(validation_message["severity"]),
            "action_hint": str(validation_message["action_hint"]),
        }

    if mode_key == "champion_pure":
        if enabled and recommended:
            return {
                "user_reason": "El champion está disponible y es la opción recomendada para este input.",
                "developer_detail": str(mode_report["message"]),
                "severity": "success",
                "action_hint": "Puedes ejecutarlo directamente como recomendación principal.",
            }
        if enabled:
            return {
                "user_reason": "El champion está disponible como alternativa para este input.",
                "developer_detail": str(mode_report["message"]),
                "severity": "info",
                "action_hint": "Puedes seleccionarlo si prefieres comparar con el modo recomendado.",
            }
        return {
            "user_reason": str(mode_report.get("user_message", "Este modo no está disponible para el input actual.")),
            "developer_detail": str(mode_report.get("developer_detail", mode_report["message"])),
            "severity": str(mode_report.get("severity", "error")),
            "action_hint": str(mode_report.get("action_hint", "Revisa el input o usa un modo habilitado.")),
        }

    if mode_key == "baseline_with_policy":
        if not enabled:
            return {
                "user_reason": str(mode_report.get("user_message", "Este modo no está disponible para el input actual.")),
                "developer_detail": str(mode_report.get("developer_detail", mode_report["message"])),
                "severity": str(mode_report.get("severity", "error")),
                "action_hint": str(mode_report.get("action_hint", "Revisa el input o usa otro modo habilitado.")),
            }
        if recommended:
            return {
                "user_reason": "La alternativa con policy está disponible y recomendada para este input.",
                "developer_detail": str(policy_grouping["reason"]),
                "severity": "success",
                "action_hint": "Puedes usarla si quieres priorizar coherencia operativa por albarán.",
            }
        return {
            "user_reason": "La alternativa con policy está disponible, pero no es la recomendación por defecto para este input.",
            "developer_detail": str(policy_grouping["reason"]),
            "severity": "info",
            "action_hint": "Úsala solo si necesitas esa capa operativa y el input tiene agrupación útil.",
        }

    if enabled and recommended:
        return {
            "user_reason": "El baseline está disponible y actúa como opción conservadora para este input.",
            "developer_detail": str(mode_report["message"]),
            "severity": "success",
            "action_hint": "Puedes ejecutarlo como referencia histórica estable.",
        }
    if enabled:
        return {
            "user_reason": "El baseline está disponible como referencia histórica.",
            "developer_detail": str(mode_report["message"]),
            "severity": "info",
            "action_hint": "Puedes usarlo si quieres una referencia conservadora del serving actual.",
        }
    return {
        "user_reason": str(mode_report.get("user_message", "Este modo no está disponible para el input actual.")),
        "developer_detail": str(mode_report.get("developer_detail", mode_report["message"])),
        "severity": str(mode_report.get("severity", "error")),
        "action_hint": str(mode_report.get("action_hint", "Revisa el input o usa otro modo habilitado.")),
    }


def get_mode_specs() -> list[InferenceModeSpec]:
    """Return the canonical ordered Day 06 product modes."""
    return [
        MODE_SPECS["baseline"],
        MODE_SPECS["champion_pure"],
        MODE_SPECS["baseline_with_policy"],
    ]


def get_mode_spec(mode_key: str) -> InferenceModeSpec:
    """Return one mode spec by key."""
    if mode_key not in MODE_SPECS:
        raise ValueError(f"Modo Day 06 no soportado: {mode_key}")
    return MODE_SPECS[mode_key]


def get_mode_expected_columns() -> dict[str, list[str]]:
    """Load the expected feature columns for the three canonical Day 06 modes."""
    return {
        mode_key: load_mode_bundle(
            str(mode_spec.model_path),
            str(mode_spec.metadata_path),
        )[2]
        for mode_key, mode_spec in MODE_SPECS.items()
    }


def inspect_policy_grouping_availability(input_df: pd.DataFrame) -> dict[str, Any]:
    """Inspect whether one input contains useful multi-event `albaran_id` groups for the assist policy."""
    prepared_df = rs.ensure_event_column(input_df, event_col="event_id")
    if "albaran_id" not in prepared_df.columns:
        return {
            "available": False,
            "reason": "El input no contiene `albaran_id`; no hay agrupación útil para recomendar la policy.",
            "grouped_albaran_count": 0,
            "max_events_per_albaran": 0,
        }

    grouping_frame = prepared_df[["albaran_id", "event_id"]].copy()
    grouping_frame["albaran_id"] = grouping_frame["albaran_id"].astype("string").fillna("").str.strip()
    grouping_frame["event_id"] = grouping_frame["event_id"].astype("string").fillna("").str.strip()
    grouping_frame = grouping_frame.loc[
        (grouping_frame["albaran_id"] != "") & (grouping_frame["event_id"] != "")
    ].drop_duplicates()

    if grouping_frame.empty:
        return {
            "available": False,
            "reason": "El input no contiene pares válidos `albaran_id + event_id` para recomendar la policy.",
            "grouped_albaran_count": 0,
            "max_events_per_albaran": 0,
        }

    event_counts = grouping_frame.groupby("albaran_id")["event_id"].nunique()
    grouped_albaran_count = int((event_counts > 1).sum())
    max_events_per_albaran = int(event_counts.max()) if not event_counts.empty else 0
    if grouped_albaran_count == 0:
        return {
            "available": False,
            "reason": "No hay ningún `albaran_id` con más de un `event_id`; la policy no se recomienda por defecto.",
            "grouped_albaran_count": grouped_albaran_count,
            "max_events_per_albaran": max_events_per_albaran,
        }

    return {
        "available": True,
        "reason": (
            "Hay agrupación útil por `albaran_id`; "
            f"se detectan {grouped_albaran_count} albaranes con más de un `event_id`."
        ),
        "grouped_albaran_count": grouped_albaran_count,
        "max_events_per_albaran": max_events_per_albaran,
    }


def inspect_input_mode_availability(
    *,
    input_df: pd.DataFrame,
    input_mode: str,
) -> dict[str, Any]:
    """Inspect the enabled and recommended Day 06 modes for one concrete product input."""
    normalized_df = normalize_operational_input(input_df=input_df)
    validation_summary = vinput.validate_inference_dataframe(
        dataframe=normalized_df,
        contract_path=INPUT_CONTRACT_PATH,
        input_name=f"preflight::{input_mode}",
        report_json=None,
    )
    mode_expected_columns = get_mode_expected_columns()
    contract_report = build_contract_report(
        input_df=normalized_df,
        mode_expected_columns=mode_expected_columns,
    )
    policy_grouping = inspect_policy_grouping_availability(normalized_df)

    champion_enabled = contract_report["modes"]["champion_pure"]["status"] == "PASS"
    baseline_enabled = contract_report["modes"]["baseline"]["status"] == "PASS"
    policy_enabled = contract_report["modes"]["baseline_with_policy"]["status"] == "PASS"

    selected_default_mode: str | None = None
    if champion_enabled:
        selected_default_mode = "champion_pure"
    elif policy_enabled and bool(policy_grouping["available"]):
        selected_default_mode = "baseline_with_policy"
    elif baseline_enabled:
        selected_default_mode = "baseline"
    elif policy_enabled:
        selected_default_mode = "baseline_with_policy"

    validation_status = str(validation_summary.get("status", "")).upper()
    validation_message: dict[str, str] | None = None
    if validation_status == "FAIL":
        validation_message = build_validation_user_message(validation_summary)
        selected_default_mode = None

    mode_catalog: list[dict[str, Any]] = []
    enabled_mode_keys: list[str] = []
    for mode_spec in get_mode_specs():
        mode_report = contract_report["modes"][mode_spec.key]
        enabled = validation_status != "FAIL" and mode_report["status"] == "PASS"
        recommended = mode_spec.key == selected_default_mode
        default_candidate = recommended

        if enabled:
            enabled_mode_keys.append(mode_spec.key)

        message_bundle = build_mode_availability_message(
            mode_key=mode_spec.key,
            enabled=enabled,
            recommended=recommended,
            validation_message=validation_message,
            mode_report=mode_report,
            policy_grouping=policy_grouping,
        )

        mode_catalog.append(
            {
                "mode_key": mode_spec.key,
                "mode_label": mode_spec.label,
                "rollout_status": mode_spec.rollout_status,
                "enabled": enabled,
                "recommended": recommended,
                "default_candidate": default_candidate,
                "status": "recommended" if recommended else ("available" if enabled else "disabled"),
                "reason": message_bundle["user_reason"],
                "user_reason": message_bundle["user_reason"],
                "developer_detail": message_bundle["developer_detail"],
                "severity": message_bundle["severity"],
                "action_hint": message_bundle["action_hint"],
                "contract_status": str(mode_report["status"]),
                "missing_raw_columns": list(mode_report.get("missing_raw_columns", [])),
                "critical_all_null_columns": list(mode_report.get("critical_all_null_columns", [])),
            }
        )

    return {
        "input_mode": input_mode,
        "input_columns_total": int(len(input_df.columns)),
        "input_columns": list(input_df.columns),
        "normalized_columns_total": int(len(normalized_df.columns)),
        "normalized_columns": list(normalized_df.columns),
        "validation_summary": validation_summary,
        "policy_grouping_available": bool(policy_grouping["available"]),
        "policy_grouping_reason": str(policy_grouping["reason"]),
        "policy_grouped_albaran_count": int(policy_grouping["grouped_albaran_count"]),
        "policy_max_events_per_albaran": int(policy_grouping["max_events_per_albaran"]),
        "selected_default_mode": selected_default_mode,
        "enabled_mode_keys": enabled_mode_keys,
        "mode_catalog": mode_catalog,
        "contract_report": contract_report,
    }
