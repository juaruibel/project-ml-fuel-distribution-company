from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.ml.product import recommend_supplier as rs
from src.ml.product import validate_inference_input as vinput
from src.ml.product.day06_feedback import (
    FEEDBACK_ACTIONS,
    build_feedback_template,
    persist_contract_failure_run,
    save_feedback,
)
from src.ml.product.day06_modes import (
    MODE_SPECS,
    InferenceModeSpec,
    build_mode_availability_message,
    build_validation_user_message,
    get_mode_expected_columns,
    get_mode_spec,
    get_mode_specs,
    inspect_input_mode_availability,
    inspect_policy_grouping_availability,
    load_mode_bundle,
)
from src.ml.product.day06_normalization import OPTIONAL_TEXT_COLUMNS, normalize_operational_input
from src.ml.product.day06_scoring_contracts import (
    Day06ScoringContractError,
    build_contract_report,
    build_mode_feature_matrix,
    inspect_mode_contract,
)
from src.ml.product.day06_sql_mirror import mirror_day06_bundle
from src.ml.product.day06_sql_store import publish_day06_run_bundle, resolve_day06_sql_store_target
from src.ml.product.day06_warnings import (
    build_operational_warning_frame,
    build_top1_frame,
    build_warning_reason_text,
    merge_warning_frame,
)
from src.ml.rules import engine as rengine
from src.ml.rules.albaran_coherence import build_resumen_albaran
from src.ml.shared import functions as fc
from src.ml.shared.helpers import build_postinference_audit_paths, build_run_id, utc_now_iso, write_json
from src.ml.shared.project_paths import (
    BASELINE_METADATA_PATH,
    INPUT_CONTRACT_PATH,
    INPUT_VALIDATION_REPORTS_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    LEGACY_OUTPUTS_DIR,
    REPORTS_DIR,
    RULES_CSV_PATH,
    CHAMPION_METADATA_PATH,
)


RAW_INPUT_ROOT = DATA_RAW_DIR / "inference_inputs"
NORMALIZED_INPUT_ROOT = DATA_PROCESSED_DIR / "inference_inputs"
INFERENCE_RUNS_ROOT = REPORTS_DIR / "inference_runs"
INFERENCE_FEEDBACK_ROOT = REPORTS_DIR / "inference_feedback"
ProgressCallback = Callable[[str, str], None]


@dataclass
class PreparedOperationalInput:
    """Bundle the staged input artifacts and validation result for one operational run."""

    run_id: str
    run_date: str
    created_at_utc: str
    input_mode: str
    source_name: str
    source_suffix: str
    input_original_path: Path
    input_candidate_grain_path: Path
    input_normalized_path: Path
    validation_report_path: Path
    validation_summary: dict[str, Any]
    candidate_grain_df: pd.DataFrame
    normalized_df: pd.DataFrame


class Day06RunContractError(RuntimeError):
    """Describe one run that failed before scoring due to a contract mismatch."""

    def __init__(
        self,
        *,
        message: str,
        run_manifest_path: Path,
        contract_report_path: Path,
        manifest_payload: dict[str, Any],
    ) -> None:
        """Store the pre-scoring failure payload so the UI can offer explicit reruns."""
        super().__init__(message)
        self.message = message
        self.run_manifest_path = run_manifest_path
        self.contract_report_path = contract_report_path
        self.manifest_payload = manifest_payload


def _emit_progress(
    progress_callback: ProgressCallback | None,
    step_key: str,
    user_message: str,
) -> None:
    """Emit one optional runtime progress update without affecting existing callers."""
    if progress_callback is None:
        return
    progress_callback(step_key, user_message)


def prepare_operational_input(
    *,
    input_df: pd.DataFrame,
    input_mode: str,
    source_name: str,
    source_suffix: str,
    source_bytes: bytes,
    progress_callback: ProgressCallback | None = None,
) -> PreparedOperationalInput:
    """Persist the original input, candidate-grain and normalized input for one Day 06 run."""
    run_id = build_unique_run_id()
    run_date = run_id[:8]
    created_at_utc = utc_now_iso()
    paths = build_operational_paths(run_id=run_id, source_suffix=source_suffix)

    paths["input_original"].write_bytes(source_bytes)
    candidate_grain_df = input_df.copy()
    candidate_grain_df.to_csv(paths["input_candidate_grain"], index=False)
    _emit_progress(progress_callback, "normalizing_input", "Normalizando input para validación y scoring.")
    normalized_df = normalize_operational_input(input_df=candidate_grain_df)
    normalized_df.to_csv(paths["input_normalized"], index=False)

    validation_report_path = vinput.build_default_report_path(
        report_root=INPUT_VALIDATION_REPORTS_DIR,
        run_id=run_id,
    )
    _emit_progress(progress_callback, "validating_input", "Validando input diario.")
    validation_summary = vinput.validate_inference_dataframe(
        dataframe=normalized_df,
        contract_path=INPUT_CONTRACT_PATH,
        input_name=source_name,
        report_json=validation_report_path,
    )

    return PreparedOperationalInput(
        run_id=run_id,
        run_date=run_date,
        created_at_utc=created_at_utc,
        input_mode=input_mode,
        source_name=source_name,
        source_suffix=source_suffix,
        input_original_path=paths["input_original"],
        input_candidate_grain_path=paths["input_candidate_grain"],
        input_normalized_path=paths["input_normalized"],
        validation_report_path=validation_report_path,
        validation_summary=validation_summary,
        candidate_grain_df=candidate_grain_df,
        normalized_df=normalized_df,
    )


def execute_operational_run(
    *,
    prepared_input: PreparedOperationalInput,
    inference_mode: str,
    top_k: int,
    surface: str = "product",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute one Day 06 operational run and persist its canonical artifacts."""
    mode_spec = get_mode_spec(inference_mode)
    paths = build_operational_paths(
        run_id=prepared_input.run_id,
        source_suffix=prepared_input.source_suffix,
    )
    mode_expected_columns = get_mode_expected_columns()
    _emit_progress(progress_callback, "validating_contract", "Validando contrato de scoring.")
    contract_report = build_contract_report(
        input_df=prepared_input.normalized_df,
        mode_expected_columns=mode_expected_columns,
    )
    write_json(paths["scoring_contract_json"], contract_report)

    baseline_scored_df = pd.DataFrame()
    champion_scored_df = pd.DataFrame()
    baseline_metadata: dict[str, Any] = {}
    champion_metadata: dict[str, Any] = {}

    try:
        _emit_progress(progress_callback, "running_scoring", "Ejecutando scoring operativo.")
        baseline_scored_df, baseline_metadata = run_scored_model_inference(
            input_df=prepared_input.normalized_df,
            mode_spec=MODE_SPECS["baseline"],
            top_k=top_k,
        )
        baseline_scored_df = ensure_day06_optional_columns(baseline_scored_df)
    except Day06ScoringContractError as error:
        failure_payload = persist_contract_failure_run(
            prepared_input=prepared_input,
            mode_spec=mode_spec,
            top_k=top_k,
            surface=surface,
            run_paths=paths,
            contract_report=contract_report,
            contract_error=error,
        )
        raise Day06RunContractError(
            message=failure_payload["error_message"],
            run_manifest_path=paths["run_manifest_json"],
            contract_report_path=paths["scoring_contract_json"],
            manifest_payload=failure_payload,
        )

    if contract_report["modes"]["champion_pure"]["status"] == "PASS":
        try:
            champion_scored_df, champion_metadata = run_scored_model_inference(
                input_df=prepared_input.normalized_df,
                mode_spec=MODE_SPECS["champion_pure"],
                top_k=top_k,
            )
            champion_scored_df = ensure_day06_optional_columns(champion_scored_df)
        except Day06ScoringContractError as error:
            if mode_spec.key != "champion_pure":
                champion_scored_df = pd.DataFrame()
            else:
                failure_payload = persist_contract_failure_run(
                    prepared_input=prepared_input,
                    mode_spec=mode_spec,
                    top_k=top_k,
                    surface=surface,
                    run_paths=paths,
                    contract_report=contract_report,
                    contract_error=error,
                )
                raise Day06RunContractError(
                    message=failure_payload["error_message"],
                    run_manifest_path=paths["run_manifest_json"],
                    contract_report_path=paths["scoring_contract_json"],
                    manifest_payload=failure_payload,
                )
    elif mode_spec.key == "champion_pure":
        contract_error = Day06ScoringContractError(
            mode_key="champion_pure",
            inspection=inspect_mode_contract(
                input_df=prepared_input.normalized_df,
                expected_feature_columns=mode_expected_columns["champion_pure"],
                mode_key="champion_pure",
            ),
        )
        failure_payload = persist_contract_failure_run(
            prepared_input=prepared_input,
            mode_spec=mode_spec,
            top_k=top_k,
            surface=surface,
            run_paths=paths,
            contract_report=contract_report,
            contract_error=contract_error,
        )
        raise Day06RunContractError(
            message=failure_payload["error_message"],
            run_manifest_path=paths["run_manifest_json"],
            contract_report_path=paths["scoring_contract_json"],
            manifest_payload=failure_payload,
        )

    _emit_progress(progress_callback, "preparing_outputs", "Preparando outputs operativos.")
    if mode_spec.key == "baseline":
        detail_df = baseline_scored_df.copy()
        legacy_raw_output_path = persist_legacy_raw_output(
            detail_df=detail_df,
            prefix="reco_baseline",
        )
        resumen_df = build_base_event_summary(
            detail_df=detail_df,
            run_id=prepared_input.run_id,
            ts_utc=prepared_input.created_at_utc,
            decision_source="historical_default",
        )
        resumen_albaran_df = build_resumen_albaran(
            resumen_df=resumen_df,
            run_id=prepared_input.run_id,
            ts_utc=prepared_input.created_at_utc,
        )
        legacy_summary_path = None
        rules_summary = None
    elif mode_spec.key == "champion_pure":
        detail_df = champion_scored_df.copy()
        legacy_raw_output_path = persist_legacy_raw_output(
            detail_df=detail_df,
            prefix="reco_champion_pure",
        )
        resumen_df = build_base_event_summary(
            detail_df=detail_df,
            run_id=prepared_input.run_id,
            ts_utc=prepared_input.created_at_utc,
            decision_source="recommended_rollout",
        )
        resumen_albaran_df = build_resumen_albaran(
            resumen_df=resumen_df,
            run_id=prepared_input.run_id,
            ts_utc=prepared_input.created_at_utc,
        )
        legacy_summary_path = None
        rules_summary = None
    else:
        assist_run = execute_assist_policy_run(
            scored_df=baseline_scored_df,
            run_id=prepared_input.run_id,
            raw_prefix="reco_baseline_with_policy",
        )
        detail_df = assist_run["detail_df"]
        resumen_df = assist_run["resumen_df"]
        resumen_albaran_df = assist_run["resumen_albaran_df"]
        legacy_raw_output_path = assist_run["raw_output_path"]
        legacy_summary_path = assist_run["summary_path"]
        rules_summary = assist_run["summary_payload"]

    warning_df = (
        build_operational_warning_frame(
            baseline_detail_df=baseline_scored_df,
            champion_detail_df=champion_scored_df,
        )
        if not baseline_scored_df.empty and not champion_scored_df.empty
        else pd.DataFrame(columns=["event_id", "low_confidence_flag", "warning_reasons"])
    )
    detail_df = merge_warning_frame(detail_df=detail_df, warning_df=warning_df)
    resumen_df = merge_warning_frame(detail_df=resumen_df, warning_df=warning_df)
    if "run_id" not in resumen_df.columns:
        resumen_df["run_id"] = prepared_input.run_id
    if "ts_utc" not in resumen_df.columns:
        resumen_df["ts_utc"] = prepared_input.created_at_utc
    if "review_status" not in resumen_df.columns:
        resumen_df["review_status"] = "pending_review"

    _emit_progress(progress_callback, "persisting_artifacts", "Persistiendo artefactos del run.")
    detail_df.to_csv(paths["detail_csv"], index=False)
    resumen_df.to_csv(paths["resumen_evento_csv"], index=False)
    resumen_albaran_df.to_csv(paths["resumen_albaran_csv"], index=False)

    feedback_df = build_feedback_template(
        resumen_df=resumen_df,
        run_id=prepared_input.run_id,
        inference_mode=mode_spec.key,
    )
    feedback_df.to_csv(paths["feedback_csv"], index=False)

    manifest_payload = {
        "run_id": prepared_input.run_id,
        "run_date": prepared_input.run_date,
        "surface": surface,
        "input_mode": prepared_input.input_mode,
        "input_original_path": str(prepared_input.input_original_path),
        "input_candidate_grain_path": str(prepared_input.input_candidate_grain_path),
        "input_normalized_path": str(prepared_input.input_normalized_path),
        "validation_report_path": str(prepared_input.validation_report_path),
        "scoring_contract_report_path": str(paths["scoring_contract_json"]),
        "inference_mode": mode_spec.key,
        "top_k": int(top_k),
        "model_artifact_path": str(mode_spec.model_path),
        "metadata_path": str(mode_spec.metadata_path),
        "detail_csv": str(paths["detail_csv"]),
        "resumen_evento_csv": str(paths["resumen_evento_csv"]),
        "resumen_albaran_csv": str(paths["resumen_albaran_csv"]),
        "feedback_csv": str(paths["feedback_csv"]),
        "sql_mirror_status": "pending",
        "created_at_utc": prepared_input.created_at_utc,
        "mode_label": mode_spec.label,
        "mode_rollout_status": mode_spec.rollout_status,
        "validation_status": str(prepared_input.validation_summary.get("status", "")).upper(),
        "scoring_status": "SUCCESS",
        "warning_events": int(warning_df["low_confidence_flag"].sum()) if not warning_df.empty else 0,
        "legacy_raw_output_path": str(legacy_raw_output_path) if legacy_raw_output_path is not None else "",
        "legacy_summary_path": str(legacy_summary_path) if legacy_summary_path is not None else "",
    }
    sql_status = mirror_day06_bundle(
        run_manifest=manifest_payload,
        normalized_df=prepared_input.normalized_df,
        resumen_df=resumen_df,
        feedback_df=feedback_df,
        warning_df=warning_df,
    )
    manifest_payload["sql_mirror_status"] = sql_status

    # SQL store canonical fields — use clean status from the store, not the mirror wrapper
    store_target = resolve_day06_sql_store_target()
    manifest_payload["sql_store_backend"] = "sqlite" if store_target else "none"
    manifest_payload["sql_store_target"] = str(store_target) if store_target else ""
    manifest_payload["sql_store_status"] = sql_status if sql_status in ("published", "disabled_unconfigured") else "published" if "published" in sql_status else sql_status
    manifest_payload["sql_store_published_at_utc"] = prepared_input.created_at_utc if sql_status == "published" else ""

    write_json(paths["run_manifest_json"], manifest_payload)

    return {
        "prepared_input": prepared_input,
        "mode_spec": mode_spec,
        "detail_df": detail_df,
        "resumen_df": resumen_df,
        "resumen_albaran_df": resumen_albaran_df,
        "warning_df": warning_df,
        "feedback_df": feedback_df,
        "manifest_payload": manifest_payload,
        "run_manifest_path": paths["run_manifest_json"],
        "detail_path": paths["detail_csv"],
        "resumen_evento_path": paths["resumen_evento_csv"],
        "resumen_albaran_path": paths["resumen_albaran_csv"],
        "feedback_path": paths["feedback_csv"],
        "scoring_contract_report_path": paths["scoring_contract_json"],
        "rules_summary": rules_summary,
        "legacy_raw_output_path": legacy_raw_output_path,
        "legacy_summary_path": legacy_summary_path,
        "baseline_metadata": baseline_metadata,
        "champion_metadata": champion_metadata,
    }


def build_operational_paths(*, run_id: str, source_suffix: str) -> dict[str, Path]:
    """Build the canonical Day 06 operational artifact paths."""
    run_date = run_id[:8]
    suffix = normalize_suffix(source_suffix)
    raw_day_dir = RAW_INPUT_ROOT / run_date
    normalized_day_dir = NORMALIZED_INPUT_ROOT / run_date
    run_day_dir = INFERENCE_RUNS_ROOT / run_date
    feedback_day_dir = INFERENCE_FEEDBACK_ROOT / run_date

    raw_day_dir.mkdir(parents=True, exist_ok=True)
    normalized_day_dir.mkdir(parents=True, exist_ok=True)
    run_day_dir.mkdir(parents=True, exist_ok=True)
    feedback_day_dir.mkdir(parents=True, exist_ok=True)

    return {
        "input_original": raw_day_dir / f"{run_id}_input_original{suffix}",
        "input_candidate_grain": normalized_day_dir / f"{run_id}_candidate_grain.csv",
        "input_normalized": normalized_day_dir / f"{run_id}_normalized.csv",
        "detail_csv": run_day_dir / f"{run_id}_detalle.csv",
        "resumen_evento_csv": run_day_dir / f"{run_id}_resumen_evento.csv",
        "resumen_albaran_csv": run_day_dir / f"{run_id}_resumen_albaran.csv",
        "run_manifest_json": run_day_dir / f"{run_id}_run_manifest.json",
        "scoring_contract_json": run_day_dir / f"{run_id}_scoring_contract.json",
        "feedback_csv": feedback_day_dir / f"{run_id}_feedback.csv",
    }


def build_unique_run_id() -> str:
    """Generate one UTC run id that does not collide with existing Day 06 artifacts."""
    candidate_ts = datetime.now(timezone.utc)
    for _ in range(5):
        run_id = build_run_id(candidate_ts)
        paths = build_operational_paths(run_id=run_id, source_suffix=".csv")
        if not any(
            path.exists()
            for path in [
                paths["input_original"],
                paths["input_candidate_grain"],
                paths["input_normalized"],
                paths["run_manifest_json"],
                paths["feedback_csv"],
            ]
        ):
            return run_id
        candidate_ts = candidate_ts + timedelta(seconds=1)
    raise RuntimeError("No se pudo generar un run_id Day 06 único tras varios intentos.")


def normalize_suffix(source_suffix: str) -> str:
    """Normalize a source suffix into one usable file extension."""
    cleaned = str(source_suffix).strip().lower()
    if cleaned == "":
        return ".csv"
    return cleaned if cleaned.startswith(".") else f".{cleaned}"


def run_scored_model_inference(
    *,
    input_df: pd.DataFrame,
    mode_spec: InferenceModeSpec,
    top_k: int,
    event_col: str = "event_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one scored inference dataframe using the mode-specific Day 06 scoring contract."""
    model, metadata, expected_feature_columns = load_mode_bundle(
        str(mode_spec.model_path),
        str(mode_spec.metadata_path),
    )
    prepared_df, matrix, _ = build_mode_feature_matrix(
        input_df=input_df,
        expected_feature_columns=expected_feature_columns,
        mode_key=mode_spec.key,
    )
    scored_df = rs.infer(
        df=prepared_df,
        matrix=matrix,
        model=model,
        event_col=event_col,
        top_k=top_k,
    )
    return scored_df, metadata


def persist_legacy_raw_output(*, detail_df: pd.DataFrame, prefix: str) -> Path:
    """Persist the legacy raw scored CSV for compatibility with historical tracing."""
    return rs.save_inference_output(
        result_df=ensure_day06_optional_columns(detail_df),
        output_dir=LEGACY_OUTPUTS_DIR,
        prefix=prefix,
    )


def ensure_day06_optional_columns(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Fill optional identifiers that legacy Day 03/05 assets still expect."""
    working = detail_df.copy()
    for optional_column in ["albaran_id", "linea_id"]:
        if optional_column not in working.columns:
            working[optional_column] = ""
    return working


def execute_assist_policy_run(
    *,
    scored_df: pd.DataFrame,
    run_id: str,
    raw_prefix: str,
    albaran_policy: str = "PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009",
) -> dict[str, Any]:
    """Apply the existing Day 03 assist policy and keep legacy audit compatibility."""
    raw_output_path = persist_legacy_raw_output(detail_df=scored_df, prefix=raw_prefix)
    legacy_paths = build_postinference_audit_paths(
        report_root=REPORTS_DIR,
        raw_output_path=raw_output_path,
        run_id=run_id,
        mode="assist",
        albaran_policy=albaran_policy,
    )
    summary_payload = rengine.run(
        input_csv=raw_output_path,
        output_csv=legacy_paths["detail"],
        output_resumen_csv=legacy_paths["resumen_evento"],
        output_resumen_albaran_csv=legacy_paths["resumen_albaran"],
        rules_csv=RULES_CSV_PATH,
        mode="assist",
        albaran_policy=albaran_policy,
        summary_json=legacy_paths["summary"],
    )
    return {
        "raw_output_path": raw_output_path,
        "detail_df": pd.read_csv(legacy_paths["detail"], keep_default_na=False),
        "resumen_df": pd.read_csv(legacy_paths["resumen_evento"], keep_default_na=False),
        "resumen_albaran_df": pd.read_csv(legacy_paths["resumen_albaran"], keep_default_na=False),
        "summary_path": legacy_paths["summary"],
        "summary_payload": summary_payload,
    }


def build_base_event_summary(
    *,
    detail_df: pd.DataFrame,
    run_id: str,
    ts_utc: str,
    decision_source: str,
) -> pd.DataFrame:
    """Build one canonical event summary for a pure model mode."""
    working = detail_df.copy()
    working["rank_event_score"] = pd.to_numeric(working["rank_event_score"], errors="coerce").fillna(999).astype(int)
    top_rows = (
        working.sort_values(["event_id", "rank_event_score"], kind="mergesort")
        .groupby("event_id", as_index=False)
        .first()
    )
    summary = pd.DataFrame(
        {
            "event_id": top_rows["event_id"],
            "fecha_evento": top_rows.get("fecha_evento", ""),
            "albaran_id": top_rows.get("albaran_id", ""),
            "recommended_supplier": top_rows.get("proveedor_candidato", ""),
            "decision_pre_policy": top_rows.get("proveedor_candidato", ""),
            "decision_final": top_rows.get("proveedor_candidato", ""),
            "decision_source": decision_source,
            "override_reason": "",
            "blocked_candidates_count": 0,
            "policy_applied_event": 0,
            "policy_rule_id": "",
            "policy_reason_event": "",
            "run_id": run_id,
            "ts_utc": ts_utc,
        }
    )
    return summary


def build_reference_comparison_table() -> pd.DataFrame:
    """Build the Day 06 reference comparison table for the demo surface."""
    baseline_metadata = rs.load_metadata(BASELINE_METADATA_PATH)
    champion_metadata = rs.load_metadata(CHAMPION_METADATA_PATH)
    registry_df = pd.read_csv(REPORTS_DIR / "metrics" / "final_baseline_vs_candidates.csv", keep_default_na=False)
    policy_variant = "V2_TRANSPORT_ONLY_LR_smote_0.5_WITH_DETERMINISTIC_LAYER_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009_v1"
    policy_rows = registry_df.loc[registry_df["model_variant"].astype(str) == policy_variant].copy()
    policy_row = policy_rows.iloc[-1] if not policy_rows.empty else None

    rows = [
        {
            "referencia": "Histórico congelado",
            "mode_key": "baseline",
            "modelo": str(baseline_metadata.get("model_name", "LR_smote_0.5")),
            "top1_hit": float(baseline_metadata.get("metrics", {}).get("top1_hit", 0.0)),
            "top2_hit": float(baseline_metadata.get("metrics", {}).get("top2_hit", 0.0)),
            "balanced_accuracy": float(baseline_metadata.get("metrics", {}).get("balanced_accuracy", 0.0)),
            "estado": "serving_default_historico",
        },
        {
            "referencia": "Champion puro",
            "mode_key": "champion_pure",
            "modelo": str(champion_metadata.get("model_name", "champion_pure")),
            "top1_hit": float(champion_metadata.get("metrics", {}).get("top1_hit", 0.0)),
            "top2_hit": float(champion_metadata.get("metrics", {}).get("top2_hit", 0.0)),
            "balanced_accuracy": float(champion_metadata.get("metrics", {}).get("balanced_accuracy", 0.0)),
            "estado": "default_recomendado_rollout",
        },
    ]
    if policy_row is not None:
        rows.append(
            {
                "referencia": "Alternativa operativa",
                "mode_key": "baseline_with_policy",
                "modelo": policy_variant,
                "top1_hit": float(pd.to_numeric(policy_row.get("top1_hit"), errors="coerce")),
                "top2_hit": float(pd.to_numeric(policy_row.get("top2_hit"), errors="coerce")),
                "balanced_accuracy": float(pd.to_numeric(policy_row.get("balanced_accuracy"), errors="coerce")),
                "estado": "operational_alternative",
            }
        )
    return pd.DataFrame(rows)


def compute_runtime_evaluation(detail_df: pd.DataFrame) -> dict[str, Any] | None:
    """Compute row and event evaluation only when the input carries labels."""
    if "target_elegido" not in detail_df.columns:
        return None
    try:
        y_true = detail_df["target_elegido"].astype(int)
        y_pred = detail_df["pred_label"].astype(int)
        row_metrics = fc.compute_row_metrics(y_true, y_pred)
        eval_frame = detail_df[["event_id", "score_model", "target_elegido", "coste_min_dia_proveedor"]].copy()
        eval_frame["target_elegido"] = eval_frame["target_elegido"].astype(int)
        return {
            "row_metrics": {
                "accuracy": float(row_metrics["accuracy"]),
                "balanced_accuracy": float(row_metrics["balanced_accuracy"]),
                "f1_pos": float(row_metrics["f1_pos"]),
            },
            "event_metrics": {
                "top1_hit": float(fc.topk_hit_by_event(eval_frame, "score_model", k=1)),
                "top2_hit": float(fc.topk_hit_by_event(eval_frame, "score_model", k=2)),
            },
        }
    except Exception:
        return None
