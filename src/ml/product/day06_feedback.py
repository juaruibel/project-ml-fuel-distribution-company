from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from src.ml.product.day06_scoring_contracts import Day06ScoringContractError
from src.ml.product.day06_sql_mirror import mirror_day06_bundle
from src.ml.product.day06_sql_store import publish_day06_feedback as _publish_feedback_sql, resolve_day06_sql_store_target
from src.ml.shared.helpers import utc_now_iso, write_json

if TYPE_CHECKING:
    from src.ml.product.day06_modes import InferenceModeSpec
    from src.ml.product.day06_runtime import PreparedOperationalInput


FEEDBACK_ACTIONS = ["pending_review", "accepted", "overridden", "rejected"]


def save_feedback(
    *,
    run_bundle: dict[str, Any],
    feedback_df: pd.DataFrame,
) -> dict[str, Any]:
    """Persist edited feedback, update the manifest and rerun the optional SQL mirror."""
    run_id = str(run_bundle["manifest_payload"]["run_id"])
    if "event_id" in feedback_df.columns and feedback_df["event_id"].duplicated().any():
        raise ValueError("feedback.csv no puede contener duplicados de event_id dentro del mismo run_id.")

    writable = feedback_df.copy()
    writable["run_id"] = run_id
    writable.to_csv(run_bundle["feedback_path"], index=False)

    manifest_payload = dict(run_bundle["manifest_payload"])
    manifest_payload["feedback_rows"] = int(len(writable))
    manifest_payload["feedback_saved_at_utc"] = utc_now_iso()
    sql_status = mirror_day06_bundle(
        run_manifest=manifest_payload,
        normalized_df=run_bundle["prepared_input"].normalized_df,
        resumen_df=run_bundle["resumen_df"],
        feedback_df=writable,
    )
    manifest_payload["sql_mirror_status"] = sql_status

    # Publish feedback to SQL store
    sql_fb_result = _publish_feedback_sql(
        run_manifest=manifest_payload,
        feedback_df=writable,
    )
    store_target = resolve_day06_sql_store_target()
    manifest_payload["sql_store_backend"] = "sqlite" if store_target else "none"
    manifest_payload["sql_store_target"] = str(store_target) if store_target else ""
    manifest_payload["sql_store_status"] = sql_fb_result.get("status", sql_status)
    manifest_payload["sql_store_published_at_utc"] = sql_fb_result.get("published_at_utc", "")

    write_json(run_bundle["run_manifest_path"], manifest_payload)
    run_bundle["manifest_payload"] = manifest_payload
    run_bundle["feedback_df"] = writable
    return {
        "feedback_path": run_bundle["feedback_path"],
        "run_manifest_path": run_bundle["run_manifest_path"],
        "sql_mirror_status": sql_status,
        "sql_store_status": sql_fb_result.get("status", sql_status),
    }


def persist_contract_failure_run(
    *,
    prepared_input: PreparedOperationalInput,
    mode_spec: InferenceModeSpec,
    top_k: int,
    surface: str,
    run_paths: dict[str, Any],
    contract_report: dict[str, Any],
    contract_error: Day06ScoringContractError,
) -> dict[str, Any]:
    """Persist one pre-scoring failure manifest when the selected mode cannot satisfy its feature contract."""
    error_message = (
        f"Contrato de scoring inválido para `{contract_error.mode_key}`: "
        f"{contract_error.inspection.message}"
    )
    user_message = str(contract_error.inspection.user_message)
    developer_detail = str(contract_error.inspection.developer_detail)
    action_hint = str(contract_error.inspection.action_hint)
    severity = str(contract_error.inspection.severity)
    manifest_payload = {
        "run_id": prepared_input.run_id,
        "run_date": prepared_input.run_date,
        "surface": surface,
        "input_mode": prepared_input.input_mode,
        "input_original_path": str(prepared_input.input_original_path),
        "input_candidate_grain_path": str(prepared_input.input_candidate_grain_path),
        "input_normalized_path": str(prepared_input.input_normalized_path),
        "validation_report_path": str(prepared_input.validation_report_path),
        "scoring_contract_report_path": str(run_paths["scoring_contract_json"]),
        "inference_mode": mode_spec.key,
        "top_k": int(top_k),
        "model_artifact_path": str(mode_spec.model_path),
        "metadata_path": str(mode_spec.metadata_path),
        "detail_csv": "",
        "resumen_evento_csv": "",
        "resumen_albaran_csv": "",
        "feedback_csv": "",
        "sql_mirror_status": "disabled_failed_pre_scoring",
        "created_at_utc": prepared_input.created_at_utc,
        "mode_label": mode_spec.label,
        "mode_rollout_status": mode_spec.rollout_status,
        "validation_status": str(prepared_input.validation_summary.get("status", "")).upper(),
        "scoring_status": "FAIL_CONTRACT_PRE_SCORING",
        "contract_error_mode": contract_error.mode_key,
        "user_message": user_message,
        "developer_detail": developer_detail,
        "error_severity": severity,
        "action_hint": action_hint,
        "error_message": error_message,
        "fallback_modes": ["baseline", "baseline_with_policy"] if mode_spec.key == "champion_pure" else [],
        "contract_report_modes": sorted(list(contract_report.get("modes", {}).keys())),
    }
    write_json(run_paths["run_manifest_json"], manifest_payload)
    return manifest_payload


def build_feedback_template(
    *,
    resumen_df: pd.DataFrame,
    run_id: str,
    inference_mode: str,
) -> pd.DataFrame:
    """Build the editable Day 06 feedback template at one row per event."""
    template = resumen_df[
        [
            column
            for column in [
                "event_id",
                "recommended_supplier",
                "decision_final",
                "override_reason",
            ]
            if column in resumen_df.columns
        ]
    ].copy()
    template.insert(0, "run_id", run_id)
    template.insert(2, "inference_mode", inference_mode)
    template["feedback_action"] = "pending_review"
    template["feedback_notes"] = ""
    template["reviewed_at_utc"] = ""
    for column in [
        "run_id",
        "event_id",
        "inference_mode",
        "recommended_supplier",
        "decision_final",
        "feedback_action",
        "override_reason",
        "feedback_notes",
        "reviewed_at_utc",
    ]:
        if column not in template.columns:
            template[column] = ""
    return template[
        [
            "run_id",
            "event_id",
            "inference_mode",
            "recommended_supplier",
            "decision_final",
            "feedback_action",
            "override_reason",
            "feedback_notes",
            "reviewed_at_utc",
        ]
    ]
