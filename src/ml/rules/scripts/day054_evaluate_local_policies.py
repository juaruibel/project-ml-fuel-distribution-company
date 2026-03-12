#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.ml.metrics import registry
    from src.ml.metrics.day054_policy_evaluation import (
        build_policy_trial_record,
        compute_help_harm_breakdown,
        compute_policy_trial_metrics,
        evaluate_policy_gate,
    )
    from src.ml.rules.albaran_coherence import build_resumen_albaran
    from src.ml.rules.day054_local_policies import build_day054_flag_frame
    from src.ml.rules.day054_policy_catalog import build_day054c_policy_catalog
    from src.ml.rules.day054_policy_strategies import (
        apply_policy_selection_strategy,
        materialize_policy_detail_frame,
        materialize_policy_event_summary,
    )
    from src.ml.shared.day054_policy_helpers import (
        DEFAULT_AUDIT_INPUT_NAME,
        build_candidate_comparison_frame,
        build_day054_invalidation_path,
        build_day054_output_paths,
        build_event_comparison_frame,
        build_event_master_frame,
        build_event_source_frame,
        build_policy_audit_paths,
        compute_event_metrics_from_detail,
        load_json,
        read_registry_row_by_variant,
        resolve_shared_cutoff_date,
        run_existing_albaran_policy,
        score_model_holdout,
    )
    from src.ml.shared.helpers import build_run_id, utc_now_iso, write_json
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.ml.metrics import registry
    from src.ml.metrics.day054_policy_evaluation import (
        build_policy_trial_record,
        compute_help_harm_breakdown,
        compute_policy_trial_metrics,
        evaluate_policy_gate,
    )
    from src.ml.rules.albaran_coherence import build_resumen_albaran
    from src.ml.rules.day054_local_policies import build_day054_flag_frame
    from src.ml.rules.day054_policy_catalog import build_day054c_policy_catalog
    from src.ml.rules.day054_policy_strategies import (
        apply_policy_selection_strategy,
        materialize_policy_detail_frame,
        materialize_policy_event_summary,
    )
    from src.ml.shared.day054_policy_helpers import (
        DEFAULT_AUDIT_INPUT_NAME,
        build_candidate_comparison_frame,
        build_day054_invalidation_path,
        build_day054_output_paths,
        build_event_comparison_frame,
        build_event_master_frame,
        build_event_source_frame,
        build_policy_audit_paths,
        compute_event_metrics_from_detail,
        load_json,
        read_registry_row_by_variant,
        resolve_shared_cutoff_date,
        run_existing_albaran_policy,
        score_model_holdout,
    )
    from src.ml.shared.helpers import build_run_id, utc_now_iso, write_json


CURRENT_OPERATIONAL_POLICY_VARIANT = "V2_TRANSPORT_ONLY_LR_smote_0.5_WITH_DETERMINISTIC_LAYER_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009_v1"
NOTEBOOK_REF = "notebooks/20_day05_4_fallback_flags_and_local_policies.ipynb"
PROMPT_REF = "src/prompts/daily/20260310_day05_4_run04_prompt.md"
INVALIDATED_RUN_ID = "20260310T093320Z"
CLEAN_DAY054B_VALID_RUN_ID = "20260310T132832Z"
REGISTRY_FLOAT_DECIMALS = 6
PREFLIGHT_TOLERANCE = 1e-9


# SECTION: JSON helpers
def _to_jsonable(value: Any) -> Any:
    """Convert nested pandas/numpy/path values into JSON-safe Python scalars."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_to_jsonable(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(key): _to_jsonable(item) for key, item in value.to_dict().items()}
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            return str(value)
    return value


# SECTION: Path helpers
def resolve_registry_artifact_path(project_root: Path, raw_path: str) -> Path:
    """Resolve one registry path whether it is already absolute or repo-relative."""
    candidate = Path(str(raw_path).strip())
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


# SECTION: Metrics helpers
def _metadata_metrics(metadata_payload: dict[str, Any]) -> dict[str, float]:
    """Extract the five preflight metrics from one metadata payload."""
    metrics = dict(metadata_payload.get("metrics") or {})
    return {
        "accuracy": float(metrics.get("accuracy", metrics.get("test_acc", 0.0))),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", metrics.get("test_bal_acc", 0.0))),
        "top1_hit": float(metrics.get("top1_hit", 0.0)),
        "top2_hit": float(metrics.get("top2_hit", 0.0)),
        "coverage": float(metrics.get("coverage", 0.0)),
    }


# SECTION: Metrics helpers
def _registry_metrics(registry_row: dict[str, Any]) -> dict[str, float]:
    """Extract the five preflight metrics from one official registry row."""
    extracted = registry.extract_metrics(registry_row)
    return {
        "accuracy": float(extracted.get("accuracy") or 0.0),
        "balanced_accuracy": float(extracted.get("balanced_accuracy") or 0.0),
        "top1_hit": float(extracted.get("top1_hit") or 0.0),
        "top2_hit": float(extracted.get("top2_hit") or 0.0),
        "coverage": float(extracted.get("coverage") or 0.0),
    }


# SECTION: Metrics helpers
def _compose_reference_metrics(row_metrics: dict[str, float], event_metrics: dict[str, float]) -> dict[str, float]:
    """Compose the canonical five-metric preflight payload from row and event metrics."""
    return {
        "accuracy": float(row_metrics["accuracy"]),
        "balanced_accuracy": float(row_metrics["balanced_accuracy"]),
        "top1_hit": float(event_metrics["top1_hit"]),
        "top2_hit": float(event_metrics["top2_hit"]),
        "coverage": float(event_metrics["coverage"]),
    }


# SECTION: Metrics helpers
def _event_source_metrics(event_source_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate top-k and coverage metrics from one event-level source frame."""
    if event_source_df.empty:
        return {
            "top1_hit": 0.0,
            "top2_hit": 0.0,
            "coverage": 0.0,
            "test_events": 0,
        }
    return {
        "top1_hit": float(event_source_df["top1_hit"].mean()),
        "top2_hit": float(event_source_df["top2_hit"].mean()),
        "coverage": float(event_source_df["decision_final"].astype(str).str.strip().ne("").mean()),
        "test_events": int(len(event_source_df)),
    }


# SECTION: Metrics helpers
def _rounded_registry_metrics(actual_metrics: dict[str, float]) -> dict[str, float]:
    """Round actual metrics to the registry storage precision before comparing against CSV row values."""
    return {
        key: round(float(value), REGISTRY_FLOAT_DECIMALS)
        for key, value in actual_metrics.items()
    }


# SECTION: Metrics helpers
def _compare_metric_sets(
    *,
    actual_metrics: dict[str, float],
    expected_metrics: dict[str, float],
    tolerance: float,
) -> dict[str, dict[str, Any]]:
    """Compare two metric payloads and return per-metric deltas plus pass/fail flags."""
    report: dict[str, dict[str, Any]] = {}
    for metric_name, actual_value in actual_metrics.items():
        expected_value = float(expected_metrics[metric_name])
        delta = abs(float(actual_value) - expected_value)
        report[metric_name] = {
            "actual": float(actual_value),
            "expected": expected_value,
            "abs_delta": delta,
            "ok": delta <= tolerance,
        }
    return report


# SECTION: Metrics helpers
def _assert_preflight_ok(reference_name: str, preflight_payload: dict[str, Any]) -> None:
    """Raise when one reference fails its reproducibility preflight."""
    metadata_ok = all(
        metric_payload["ok"]
        for metric_payload in preflight_payload["metadata_check"].values()
    )
    registry_ok = all(
        metric_payload["ok"]
        for metric_payload in preflight_payload["registry_check"].values()
    )
    if metadata_ok and registry_ok:
        return
    raise ValueError(
        f"Day 05.4 preflight failed for {reference_name}: "
        f"metadata_ok={metadata_ok} registry_ok={registry_ok}"
    )


# SECTION: Metrics helpers
def build_reference_preflight_payload(
    *,
    reference_name: str,
    actual_metrics: dict[str, float],
    metadata_payload: dict[str, Any],
    registry_row: dict[str, Any],
) -> dict[str, Any]:
    """Build one preflight reproducibility payload against metadata and the official registry row."""
    metadata_metrics = _metadata_metrics(metadata_payload=metadata_payload)
    registry_metrics = _registry_metrics(registry_row=registry_row)
    metadata_check = _compare_metric_sets(
        actual_metrics=actual_metrics,
        expected_metrics=metadata_metrics,
        tolerance=PREFLIGHT_TOLERANCE,
    )
    registry_check = _compare_metric_sets(
        actual_metrics=_rounded_registry_metrics(actual_metrics=actual_metrics),
        expected_metrics=registry_metrics,
        tolerance=PREFLIGHT_TOLERANCE,
    )
    payload = {
        "reference_name": reference_name,
        "tolerance": PREFLIGHT_TOLERANCE,
        "actual_metrics": actual_metrics,
        "metadata_metrics": metadata_metrics,
        "registry_metrics": registry_metrics,
        "metadata_check": metadata_check,
        "registry_check": registry_check,
    }
    _assert_preflight_ok(reference_name=reference_name, preflight_payload=payload)
    return payload


# SECTION: Metrics helpers
def build_operational_reference_preflight_payload(
    *,
    actual_model_metrics: dict[str, float],
    actual_policy_metrics: dict[str, float],
    metadata_payload: dict[str, Any],
    registry_row: dict[str, Any],
) -> dict[str, Any]:
    """Build the operational-reference preflight by checking model metadata and policy registry independently."""
    metadata_metrics = _metadata_metrics(metadata_payload=metadata_payload)
    registry_metrics = _registry_metrics(registry_row=registry_row)
    metadata_check = _compare_metric_sets(
        actual_metrics=actual_model_metrics,
        expected_metrics=metadata_metrics,
        tolerance=PREFLIGHT_TOLERANCE,
    )
    registry_check = _compare_metric_sets(
        actual_metrics=_rounded_registry_metrics(actual_metrics=actual_policy_metrics),
        expected_metrics=registry_metrics,
        tolerance=PREFLIGHT_TOLERANCE,
    )
    payload = {
        "reference_name": str(registry_row["model_variant"]),
        "tolerance": PREFLIGHT_TOLERANCE,
        "actual_model_metrics": actual_model_metrics,
        "actual_policy_metrics": actual_policy_metrics,
        "metadata_metrics": metadata_metrics,
        "registry_metrics": registry_metrics,
        "metadata_check": metadata_check,
        "registry_check": registry_check,
    }
    metadata_ok = all(metric_payload["ok"] for metric_payload in metadata_check.values())
    registry_ok = all(metric_payload["ok"] for metric_payload in registry_check.values())
    if not metadata_ok or not registry_ok:
        raise ValueError(
            "Day 05.4 preflight failed for operational reference: "
            f"metadata_ok={metadata_ok} registry_ok={registry_ok}"
        )
    return payload


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 05.4c local policy runner."""
    parser = argparse.ArgumentParser(
        description="Day 05.4c · endurecimiento final del residual/composite antes de Brent."
    )
    parser.add_argument(
        "--baseline-dataset-csv",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v2_candidates.csv"),
        help="Dataset baseline oficial.",
    )
    parser.add_argument(
        "--baseline-model-path",
        type=Path,
        default=Path("models/public/baseline/model.pkl"),
        help="Modelo baseline oficial.",
    )
    parser.add_argument(
        "--baseline-metadata-path",
        type=Path,
        default=Path("models/public/baseline/metadata.json"),
        help="Metadata baseline oficial.",
    )
    parser.add_argument(
        "--champion-dataset-csv",
        type=Path,
        default=Path("data/public/day041/dataset_modelo_v2_transport_only.csv"),
        help="Dataset champion puro vigente.",
    )
    parser.add_argument(
        "--champion-model-path",
        type=Path,
        default=Path("models/public/champion_pure/model.pkl"),
        help="Modelo champion puro vigente.",
    )
    parser.add_argument(
        "--champion-metadata-path",
        type=Path,
        default=Path("models/public/champion_pure/metadata.json"),
        help="Metadata champion puro vigente.",
    )
    parser.add_argument(
        "--operational-variant",
        type=str,
        default=CURRENT_OPERATIONAL_POLICY_VARIANT,
        help="Variant actual de policy operativa vigente en el registry oficial.",
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("artifacts/public/metrics/final_baseline_vs_candidates.csv"),
        help="Registry oficial baseline vs candidates.",
    )
    parser.add_argument(
        "--rules-csv",
        type=Path,
        default=Path("config/business_blocklist_rules.csv"),
        help="CSV de reglas de negocio.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports"),
        help="Directorio raíz de reportes.",
    )
    parser.add_argument(
        "--day-id",
        type=str,
        default="Day 05.4c",
        help="Identificador del día activo.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k operativo.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id opcional. Si vacío, usa el helper global del repo.",
    )
    parser.add_argument(
        "--invalidated-run-id",
        type=str,
        default=INVALIDATED_RUN_ID,
        help="Run id histórico Day 05.4 previamente invalidado y mantenido en cuarentena.",
    )
    parser.add_argument(
        "--base-valid-run-id",
        type=str,
        default=CLEAN_DAY054B_VALID_RUN_ID,
        help="Run id auditado Day 05.4b que sirve como base válida para la microiteración 05.4c.",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="No añadir finalista al registry oficial aunque pase gate.",
    )
    return parser.parse_args()


# SECTION: Paths
def resolve_repo_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve all repo-relative input paths used by the Day 05.4 runner."""
    project_root = Path(__file__).resolve().parents[4]
    return {
        "project_root": project_root,
        "baseline_dataset_csv": (project_root / args.baseline_dataset_csv).resolve(),
        "baseline_model_path": (project_root / args.baseline_model_path).resolve(),
        "baseline_metadata_path": (project_root / args.baseline_metadata_path).resolve(),
        "champion_dataset_csv": (project_root / args.champion_dataset_csv).resolve(),
        "champion_model_path": (project_root / args.champion_model_path).resolve(),
        "champion_metadata_path": (project_root / args.champion_metadata_path).resolve(),
        "registry_csv": (project_root / args.registry_csv).resolve(),
        "rules_csv": (project_root / args.rules_csv).resolve(),
        "reports_root": (project_root / args.reports_root).resolve(),
    }


# SECTION: Trial serialization
def persist_policy_trial_artifacts(
    *,
    report_root: Path,
    run_id: str,
    policy_metadata: dict[str, Any],
    policy_detail_df: pd.DataFrame,
    policy_event_df: pd.DataFrame,
    policy_resumen_albaran_df: pd.DataFrame,
    policy_summary_payload: dict[str, Any],
) -> dict[str, Path]:
    """Persist one full Day 05.4 policy audit package."""
    audit_paths = build_policy_audit_paths(
        report_root=report_root,
        input_name=DEFAULT_AUDIT_INPUT_NAME,
        policy_variant=str(policy_metadata["policy_variant"]),
        run_id=run_id,
    )
    audit_paths["detail"].parent.mkdir(parents=True, exist_ok=True)
    policy_detail_df.to_csv(audit_paths["detail"], index=False)
    policy_event_df.to_csv(audit_paths["resumen_evento"], index=False)
    policy_resumen_albaran_df.to_csv(audit_paths["resumen_albaran"], index=False)
    if policy_summary_payload:
        write_json(audit_paths["summary"], _to_jsonable(policy_summary_payload))
    return audit_paths


# SECTION: Final decision
def resolve_final_run_decision(policy_trials_df: pd.DataFrame) -> tuple[str, dict[str, Any] | None]:
    """Resolve the Day 05.4 final decision and the winning promotable policy, if any."""
    promotable = policy_trials_df.loc[policy_trials_df["decision_label"] == "promotable"].copy()
    if not promotable.empty:
        promotable = promotable.sort_values(
            ["delta_top2_vs_operational", "delta_top1_vs_operational", "selected_share"],
            ascending=[False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        return "promote_day054_policy_candidate", promotable.iloc[0].to_dict()
    return "keep_current_operational_policy", None


# SECTION: Day 05.4c smoke helpers
def build_day054c_smoke_expectations() -> dict[str, dict[str, Any]]:
    """Return the conservative Day 05.4c smoke expectations anchored to the audited Day 05.4b rerun."""
    return {
        "day054c_go_b_residual_SUPPLIER_050_rank2_clean_v1": {
            "expected_selected_events": 19,
            "expected_top1_improves": 19,
            "max_top1_harms": 0,
            "notebook19_alignment_note": (
                "Endurece el residual SUPPLIER_050 con rank transport <= 2 para quedarse solo con el subconjunto limpio "
                "que preserva las ganancias sin harmed detectables."
            ),
        },
        "day054c_go_b_residual_SUPPLIER_019_low_conf_v1": {
            "expected_selected_events": 9,
            "expected_top1_improves": 9,
            "max_top1_harms": 0,
            "notebook19_alignment_note": (
                "Mantiene el residual SUPPLIER_019 de baja confianza tal como quedó limpio en Day 05.4b."
            ),
        },
        "day054c_go_b_residual_outer_terminal_extension_v1": {
            "expected_selected_events": 10,
            "expected_top1_improves": 10,
            "max_top1_harms": 0,
            "notebook19_alignment_note": (
                "Mantiene la extensión outer terminal tal como quedó limpia en Day 05.4b."
            ),
        },
    }


# SECTION: Day 05.4c smoke helpers
def compute_day054c_smoke_rows(clean_event_audit_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute conservative Day 05.4c smoke rows from the audited Day 05.4b event audit."""
    working = clean_event_audit_df.copy()
    dominant_mask = working["go_b_dominant_slice_event"].fillna(0).astype(int) == 1
    baseline_SUPPLIER_009_mask = working["baseline__top1_provider"].astype(str) == "SUPPLIER_009"
    go_b_non_dominant_mask = (working["producto_canonico"].astype(str) == "PRODUCT_003") & (~dominant_mask) & baseline_SUPPLIER_009_mask
    transport_rank = pd.to_numeric(
        working.get("top1_v41_transport_rank_event_pure_champion", working.get("v41_transport_rank_event")),
        errors="coerce",
    )
    rules = {
        "day054c_go_b_residual_SUPPLIER_050_rank2_clean_v1": (
            go_b_non_dominant_mask
            & (working["terminal_compra"].astype(str) == "TERMINAL_001")
            & (working["pure_champion__top1_provider"].astype(str) == "SUPPLIER_050")
            & (transport_rank <= 2)
        ),
        "day054c_go_b_residual_SUPPLIER_019_low_conf_v1": (
            go_b_non_dominant_mask
            & (working["terminal_compra"].astype(str) == "TERMINAL_001")
            & (working["pure_champion__top1_provider"].astype(str) == "SUPPLIER_019")
            & (pd.to_numeric(working["top1_score_pure_champion"], errors="coerce") <= 0.723)
        ),
        "day054c_go_b_residual_outer_terminal_extension_v1": (
            go_b_non_dominant_mask
            & (working["terminal_compra"].astype(str).isin(["TERMINAL_002", "TERMINAL_003"]))
            & (working["pure_champion__top1_provider"].astype(str).isin(["SUPPLIER_011", "SUPPLIER_020"]))
        ),
    }
    expectations = build_day054c_smoke_expectations()
    rows: list[dict[str, Any]] = []
    for policy_variant, mask in rules.items():
        selected = working.loc[mask].copy()
        improves_top1 = int(
            ((selected["baseline__top1_hit"].astype(int) == 1) & (selected["pure_champion__top1_hit"].astype(int) == 0)).sum()
        )
        harms_top1 = int(
            ((selected["baseline__top1_hit"].astype(int) == 0) & (selected["pure_champion__top1_hit"].astype(int) == 1)).sum()
        )
        improves_top2 = int(
            ((selected["baseline__top2_hit"].astype(int) == 1) & (selected["pure_champion__top2_hit"].astype(int) == 0)).sum()
        )
        harms_top2 = int(
            ((selected["baseline__top2_hit"].astype(int) == 0) & (selected["pure_champion__top2_hit"].astype(int) == 1)).sum()
        )
        expectation = expectations[policy_variant]
        rows.append(
            {
                "policy_variant": policy_variant,
                "selected_events": int(len(selected)),
                "improves_top1_vs_champion": improves_top1,
                "harms_top1_vs_champion": harms_top1,
                "improves_top2_vs_champion": improves_top2,
                "harms_top2_vs_champion": harms_top2,
                "expected_selected_events": expectation["expected_selected_events"],
                "expected_top1_improves": expectation["expected_top1_improves"],
                "max_top1_harms": expectation["max_top1_harms"],
                "alignment_ok": (
                    int(len(selected)) >= int(expectation["expected_selected_events"])
                    and improves_top1 >= int(expectation["expected_top1_improves"])
                    and harms_top1 <= int(expectation["max_top1_harms"])
                ),
                "notebook19_alignment_note": expectation["notebook19_alignment_note"],
            }
        )
    return rows


# SECTION: Day 05.4c smoke helpers
def build_day054c_smoke_differences(
    *,
    smoke_rows: list[dict[str, Any]],
    policy_trial_records: list[dict[str, Any]],
) -> list[str]:
    """Build one list of explanation strings whenever the Day 05.4c rerun diverges from its smoke anchor."""
    trial_by_variant = {str(row["policy_variant"]): row for row in policy_trial_records}
    differences: list[str] = []
    for smoke_row in smoke_rows:
        if not bool(smoke_row.get("alignment_ok", False)):
            differences.append(
                    f"{smoke_row['policy_variant']}: el smoke base no reproduce la lectura esperada de Notebook 19 "
                    f"y/o del cierre Day 05.4b; "
                    f"selected_events={smoke_row['selected_events']}, improves_top1_vs_champion={smoke_row['improves_top1_vs_champion']}, "
                    f"harms_top1_vs_champion={smoke_row['harms_top1_vs_champion']}."
                )
        trial_row = trial_by_variant.get(str(smoke_row["policy_variant"]))
        if trial_row is None:
            differences.append(
                f"{smoke_row['policy_variant']}: la variante no aparece en el rerun auditado."
            )
            continue
        selected_delta = int(trial_row["selected_events"]) - int(smoke_row["selected_events"])
        harmed_delta = int(trial_row["harms_top1_vs_pure_champion"]) - int(smoke_row["harms_top1_vs_champion"])
        if selected_delta == 0 and harmed_delta == 0:
            continue
        differences.append(
            f"{smoke_row['policy_variant']}: selected_events {trial_row['selected_events']} vs smoke {smoke_row['selected_events']}; "
            f"harms_top1_vs_champion {trial_row['harms_top1_vs_pure_champion']} vs smoke {smoke_row['harms_top1_vs_champion']}."
        )
    return differences


# SECTION: Registry append
def append_promoted_policy_to_registry(
    *,
    registry_csv: Path,
    day_id: str,
    run_id: str,
    policy_variant: str,
    champion_model_path: Path,
    champion_metadata_path: Path,
    champion_dataset_path: Path,
    policy_metrics_json: Path,
) -> None:
    """Append one promoted Day 05.4 policy candidate to the official registry."""
    registry.append_candidate(
        argparse.Namespace(
            command="append-candidate",
            output=str(registry_csv),
            run_id=run_id,
            day_id=day_id,
            model_variant=policy_variant,
            metadata=str(champion_metadata_path),
            metrics_json=str(policy_metrics_json),
            dataset=str(champion_dataset_path),
            coverage=None,
            test_events=None,
            model_path=str(champion_model_path),
            gate_pass="true",
            promotion_decision="promote",
        )
    )


# SECTION: Registry append
def append_keep_current_operational_policy_to_registry(
    *,
    registry_csv: Path,
    day_id: str,
    run_id: str,
    cutoff_date: str,
    operational_row: dict[str, Any],
    closure_metrics_payload_path: Path,
) -> None:
    """Append one canonical Day 05.4c closure row that reaffirms the operational policy without promotion."""
    rows = registry.read_registry_rows(registry_csv)
    model_variant = "day054c_run_closure_keep_current_operational_policy_v1"
    registry.assert_unique_key(rows, run_id, model_variant, cutoff_date)
    closure_payload = load_json(closure_metrics_payload_path)
    closure_metrics = registry.extract_metrics(closure_payload)
    row = registry.build_base_row(
        run_id=run_id,
        day_id=day_id,
        model_variant=model_variant,
        model_role="candidate",
        dataset_name=str(operational_row["dataset_name"]),
        dataset_snapshot_hash=str(operational_row["dataset_snapshot_hash"]),
        cutoff_date=cutoff_date,
        metrics=closure_metrics,
        selection_rule=(
            "day054c_audited_keep_current_operational_policy_after_SUPPLIER_050_rank2_hardening_and_composite_gate"
        ),
        model_path=str(operational_row["model_path"]),
        metadata_path=str(operational_row["metadata_path"]),
        metrics_source=str(closure_metrics_payload_path),
    )
    row["delta_top2_vs_baseline"] = str(operational_row.get("delta_top2_vs_baseline", ""))
    row["delta_bal_acc_vs_baseline"] = str(operational_row.get("delta_bal_acc_vs_baseline", ""))
    row["delta_coverage_vs_baseline"] = str(operational_row.get("delta_coverage_vs_baseline", ""))
    row["gate_top2_ok"] = str(operational_row.get("gate_top2_ok", ""))
    row["gate_bal_acc_ok"] = str(operational_row.get("gate_bal_acc_ok", ""))
    row["gate_coverage_ok"] = str(operational_row.get("gate_coverage_ok", ""))
    row["gate_pass"] = str(operational_row.get("gate_pass", ""))
    row["promotion_decision"] = "keep_current_operational_policy"
    rows.append(row)
    registry.write_registry_rows(registry_csv, rows)


# SECTION: Payload builders
def build_registry_compatible_policy_summary(
    *,
    run_id: str,
    ts_utc: str,
    day_id: str,
    cutoff_date: str,
    policy_metadata: dict[str, Any],
    trial_record: dict[str, Any],
    source_row_metrics: dict[str, float],
    audit_paths: dict[str, Path],
    source_refs: dict[str, str],
) -> dict[str, Any]:
    """Build one policy summary payload compatible with Day 05.x audit and registry patterns."""
    metrics = {
        "accuracy": source_row_metrics["accuracy"],
        "balanced_accuracy": source_row_metrics["balanced_accuracy"],
        "f1_pos": source_row_metrics["f1_pos"],
        "top1_hit": trial_record["top1_hit"],
        "top2_hit": trial_record["top2_hit"],
        "coverage": trial_record["coverage"],
        "test_events": trial_record["events"],
        "top1_hit_before": trial_record["top1_hit"] - trial_record["delta_top1_vs_champion"],
        "top2_hit_before": trial_record["top2_hit"] - trial_record["delta_top2_vs_champion"],
        "top1_hit_after": trial_record["top1_hit"],
        "top2_hit_after": trial_record["top2_hit"],
        "coherence_before": trial_record["coherence_before"],
        "coherence_after": trial_record["coherence_after"],
        "coherence_delta": trial_record["coherence_delta"],
        "overrides_count": trial_record["overrides_count"],
        "overrides_improved": trial_record["overrides_improved"],
        "overrides_harmed": trial_record["overrides_harmed"],
        "overrides_neutral": trial_record["overrides_neutral"],
        "coverage_before": 1.0,
        "coverage_after": trial_record["coverage"],
    }
    return {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": day_id,
        "scope": "after_policy",
        "eval_scope": "day054_official_holdout",
        "cutoff_applied": cutoff_date,
        "policy": policy_metadata["policy_variant"],
        "model_variant": policy_metadata["policy_variant"],
        "metrics": metrics,
        "gate": {
            key: trial_record[key]
            for key in [
                "gate_pass",
                "decision_label",
                "material_improvement_ok",
                "non_target_ok_vs_operational",
                "coverage_ok_vs_operational",
                "coverage_ok_vs_champion",
                "top2_ok_vs_champion",
                "coherence_ok",
                "overrides_harmed_ok",
                "selected_events_ok",
                "effective_override_ok",
                "failure_reason",
            ]
        },
        "trial_record": trial_record,
        "policy_metadata": policy_metadata,
        "sources": {
            "detail_csv": str(audit_paths["detail"]),
            "resumen_evento_csv": str(audit_paths["resumen_evento"]),
            "resumen_albaran_csv": str(audit_paths["resumen_albaran"]),
            "summary_json": str(audit_paths["summary"]),
            **source_refs,
        },
    }


# SECTION: Runner
def run_day054(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full Day 05.4c runner end-to-end."""
    paths = resolve_repo_paths(args)
    run_id = args.run_id.strip() or build_run_id()
    ts_utc = utc_now_iso()
    output_paths = build_day054_output_paths(reports_root=paths["reports_root"], run_id=run_id)
    output_paths["day054_root"].mkdir(parents=True, exist_ok=True)
    output_paths["invalidations_root"].mkdir(parents=True, exist_ok=True)
    clean_event_audit_path = output_paths["day054_root"] / f"{args.base_valid_run_id}_event_level_audit.csv"
    if not clean_event_audit_path.exists():
        raise FileNotFoundError(
            f"No existe el audit base Day 05.4b para Day 05.4c: {clean_event_audit_path}"
        )
    clean_event_audit_df = pd.read_csv(clean_event_audit_path)
    smoke_rows = compute_day054c_smoke_rows(clean_event_audit_df=clean_event_audit_df)

    cutoff_date = resolve_shared_cutoff_date(
        champion_metadata_path=paths["champion_metadata_path"],
        baseline_metadata_path=paths["baseline_metadata_path"],
    )

    baseline_scored = score_model_holdout(
        dataset_path=paths["baseline_dataset_csv"],
        model_path=paths["baseline_model_path"],
        metadata_path=paths["baseline_metadata_path"],
        cutoff_date=cutoff_date,
        top_k=args.top_k,
    )
    champion_scored = score_model_holdout(
        dataset_path=paths["champion_dataset_csv"],
        model_path=paths["champion_model_path"],
        metadata_path=paths["champion_metadata_path"],
        cutoff_date=cutoff_date,
        top_k=args.top_k,
    )
    champion_registry_row = read_registry_row_by_variant(
        registry_csv=paths["registry_csv"],
        model_variant=str(champion_scored["metadata"]["model_name"]),
    )
    operational_row = read_registry_row_by_variant(
        registry_csv=paths["registry_csv"],
        model_variant=args.operational_variant,
    )
    operational_metrics_source_path = resolve_registry_artifact_path(
        paths["project_root"],
        operational_row["metrics_source"],
    )
    operational_metrics_source_payload = load_json(operational_metrics_source_path)
    operational_dataset_path = resolve_registry_artifact_path(
        paths["project_root"],
        str((operational_metrics_source_payload.get("sources") or {}).get("dataset_csv", paths["champion_dataset_csv"])),
    )
    operational_scored = score_model_holdout(
        dataset_path=operational_dataset_path,
        model_path=resolve_registry_artifact_path(paths["project_root"], operational_row["model_path"]),
        metadata_path=resolve_registry_artifact_path(paths["project_root"], operational_row["metadata_path"]),
        cutoff_date=cutoff_date,
        top_k=args.top_k,
    )

    baseline_with_policy = run_existing_albaran_policy(
        scored_df=baseline_scored["detail_df"],
        input_name="day054_baseline_with_policy_input",
        run_id=run_id,
        day054_root=output_paths["day054_root"],
        report_root=paths["reports_root"],
        rules_csv=paths["rules_csv"],
        albaran_policy="PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009",
    )
    operational_policy_reference = run_existing_albaran_policy(
        scored_df=operational_scored["detail_df"],
        input_name="day054_operational_reference_input",
        run_id=run_id,
        day054_root=output_paths["day054_root"],
        report_root=paths["reports_root"],
        rules_csv=paths["rules_csv"],
        albaran_policy="PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009",
    )

    baseline_event_source = build_event_source_frame(
        detail_df=baseline_scored["detail_df"],
        source_model_variant="LR_smote_0.5",
    )
    champion_event_source = build_event_source_frame(
        detail_df=champion_scored["detail_df"],
        source_model_variant=champion_scored["metadata"]["model_name"],
    )
    baseline_policy_event_source = build_event_source_frame(
        detail_df=baseline_with_policy["detail_df"],
        resumen_evento_df=baseline_with_policy["resumen_evento_df"],
        source_model_variant="LR_smote_0.5_WITH_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009",
    )
    operational_event_source = build_event_source_frame(
        detail_df=operational_policy_reference["detail_df"],
        resumen_evento_df=operational_policy_reference["resumen_evento_df"],
        source_model_variant=args.operational_variant,
    )

    champion_preflight = build_reference_preflight_payload(
        reference_name=str(champion_scored["metadata"]["model_name"]),
        actual_metrics=_compose_reference_metrics(
            row_metrics=champion_scored["row_metrics"],
            event_metrics=champion_scored["event_metrics"],
        ),
        metadata_payload=champion_scored["metadata"],
        registry_row=champion_registry_row,
    )
    operational_preflight = build_operational_reference_preflight_payload(
        actual_model_metrics=_compose_reference_metrics(
            row_metrics=operational_scored["row_metrics"],
            event_metrics=operational_scored["event_metrics"],
        ),
        actual_policy_metrics=_compose_reference_metrics(
            row_metrics=operational_scored["row_metrics"],
            event_metrics=_event_source_metrics(operational_event_source),
        ),
        metadata_payload=operational_scored["metadata"],
        registry_row=operational_row,
    )

    candidate_compare_df = build_candidate_comparison_frame(
        baseline_detail_df=baseline_scored["detail_df"],
        champion_detail_df=champion_scored["detail_df"],
    )
    event_compare_df = build_event_comparison_frame(candidate_compare_df=candidate_compare_df)
    flag_frame_df = build_day054_flag_frame(
        event_compare_df=event_compare_df,
        candidate_compare_df=candidate_compare_df,
    )
    event_master_df = build_event_master_frame(
        event_compare_df=event_compare_df,
        flag_frame_df=flag_frame_df,
        source_frames={
            "baseline": baseline_event_source,
            "pure_champion": champion_event_source,
            "baseline_with_policy": baseline_policy_event_source,
            "operational_policy_reference": operational_event_source,
        },
    )

    policy_catalog = build_day054c_policy_catalog(prompt_ref=PROMPT_REF, notebook_ref=NOTEBOOK_REF)
    policy_trial_records: list[dict[str, Any]] = []
    policy_summaries: list[dict[str, Any]] = []
    policy_audit_path_map: dict[str, dict[str, str]] = {}

    for policy_metadata in policy_catalog:
        selection_df = apply_policy_selection_strategy(
            event_master_df=event_master_df,
            policy_metadata=policy_metadata,
        )
        policy_event_df = materialize_policy_event_summary(
            event_master_df=event_master_df,
            selection_df=selection_df,
            policy_metadata=policy_metadata,
        )
        policy_resumen_albaran_df = build_resumen_albaran(
            resumen_df=policy_event_df,
            run_id=run_id,
            ts_utc=ts_utc,
        )
        policy_detail_df = materialize_policy_detail_frame(
            candidate_compare_df=candidate_compare_df,
            policy_event_df=policy_event_df,
        )
        trial_metrics = compute_policy_trial_metrics(
            policy_event_df=policy_event_df,
            policy_resumen_albaran_df=policy_resumen_albaran_df,
        )
        trial_metrics.update(compute_help_harm_breakdown(policy_event_df=policy_event_df, comparator_prefix="pure_champion"))
        trial_metrics.update(
            compute_help_harm_breakdown(
                policy_event_df=policy_event_df,
                comparator_prefix="operational_policy_reference",
            )
        )
        gate_result = evaluate_policy_gate(
            policy_metadata=policy_metadata,
            trial_metrics=trial_metrics,
            operational_reference_metrics={
                "coverage": float(operational_event_source["decision_final"].astype(str).str.strip().ne("").mean()),
                "top2_hit": float(operational_event_source["top2_hit"].mean()),
            },
            champion_reference_metrics={
                "coverage": float(champion_event_source["decision_final"].astype(str).str.strip().ne("").mean()),
                "top2_hit": float(champion_event_source["top2_hit"].mean()),
            },
        )
        trial_record = build_policy_trial_record(
            run_id=run_id,
            policy_metadata=policy_metadata,
            trial_metrics=trial_metrics,
            gate_result=gate_result,
        )
        audit_paths = persist_policy_trial_artifacts(
            report_root=paths["reports_root"],
            run_id=run_id,
            policy_metadata=policy_metadata,
            policy_detail_df=policy_detail_df,
            policy_event_df=policy_event_df,
            policy_resumen_albaran_df=policy_resumen_albaran_df,
            policy_summary_payload={},
        )
        policy_summary_payload = build_registry_compatible_policy_summary(
            run_id=run_id,
            ts_utc=ts_utc,
            day_id=args.day_id,
            cutoff_date=cutoff_date,
            policy_metadata=policy_metadata,
            trial_record=trial_record,
            source_row_metrics=champion_scored["row_metrics"],
            audit_paths=audit_paths,
            source_refs={
                "prompt_ref": PROMPT_REF,
                "notebook_ref": NOTEBOOK_REF,
                "champion_model_path": str(paths["champion_model_path"]),
                "champion_metadata_path": str(paths["champion_metadata_path"]),
                "champion_dataset_csv": str(paths["champion_dataset_csv"]),
                "baseline_model_path": str(paths["baseline_model_path"]),
                "baseline_metadata_path": str(paths["baseline_metadata_path"]),
                "baseline_dataset_csv": str(paths["baseline_dataset_csv"]),
                "operational_variant": args.operational_variant,
                "operational_model_path": str(resolve_registry_artifact_path(paths["project_root"], operational_row["model_path"])),
                "operational_metadata_path": str(resolve_registry_artifact_path(paths["project_root"], operational_row["metadata_path"])),
                "operational_dataset_csv": str(operational_dataset_path),
                "operational_metrics_source": str(operational_metrics_source_path),
            },
        )
        write_json(audit_paths["summary"], _to_jsonable(policy_summary_payload))
        policy_audit_path_map[str(policy_metadata["policy_variant"])] = {
            name: str(path) for name, path in audit_paths.items()
        }
        policy_trial_records.append(trial_record)
        policy_summaries.append(
            {
                "policy_variant": policy_metadata["policy_variant"],
                "decision_label": trial_record["decision_label"],
                "audit_paths": policy_audit_path_map[str(policy_metadata["policy_variant"])],
            }
        )

    policy_trials_df = pd.DataFrame(policy_trial_records)
    policy_trials_df.to_csv(output_paths["policy_trials_csv"], index=False)
    write_json(output_paths["policy_trials_json"], _to_jsonable({"policy_trials": policy_trial_records}))
    event_master_df.to_csv(output_paths["event_level_audit_csv"], index=False)
    candidate_compare_df.to_csv(output_paths["candidate_level_audit_csv"], index=False)

    final_decision, winning_policy = resolve_final_run_decision(policy_trials_df)
    smoke_differences = build_day054c_smoke_differences(
        smoke_rows=smoke_rows,
        policy_trial_records=policy_trial_records,
    )
    run_summary_payload = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "day_id": args.day_id,
        "cutoff_date": cutoff_date,
        "prompt_ref": PROMPT_REF,
        "notebook_ref": NOTEBOOK_REF,
        "decision": final_decision,
        "winning_policy": winning_policy,
        "historical_invalidated_run_id": args.invalidated_run_id,
        "base_day054b_valid_run_id": args.base_valid_run_id,
        "current_operational_policy_variant": args.operational_variant,
        "champion_pure_variant": champion_scored["metadata"]["model_name"],
        "baseline_variant": "LR_smote_0.5",
        "preflight": {
            "champion_pure": champion_preflight,
            "operational_policy_reference": operational_preflight,
        },
        "pre_rerun_smokes": smoke_rows,
        "notebook19_differences": smoke_differences,
        "stop_rule_applied": final_decision == "keep_current_operational_policy",
        "source_row_metrics": {
            "baseline": baseline_scored["row_metrics"],
            "champion_pure": champion_scored["row_metrics"],
            "operational_policy_reference": operational_scored["row_metrics"],
        },
        "policy_trials_path": str(output_paths["policy_trials_csv"]),
        "policy_audit_paths": policy_audit_path_map,
    }
    write_json(
        output_paths["policy_summary_json"],
        _to_jsonable(
            {
                "run_id": run_id,
                "ts_utc": ts_utc,
                "day_id": args.day_id,
                "notebook_ref": NOTEBOOK_REF,
                "prompt_ref": PROMPT_REF,
                "policies": policy_summaries,
            }
        ),
    )
    write_json(output_paths["run_summary_json"], _to_jsonable(run_summary_payload))

    if winning_policy is not None and not args.skip_registry:
        winning_summary_path = Path(
            policy_audit_path_map[str(winning_policy["policy_variant"])]["summary"]
        )
        append_promoted_policy_to_registry(
            registry_csv=paths["registry_csv"],
            day_id=args.day_id,
            run_id=run_id,
            policy_variant=str(winning_policy["policy_variant"]),
            champion_model_path=paths["champion_model_path"],
            champion_metadata_path=paths["champion_metadata_path"],
            champion_dataset_path=paths["champion_dataset_csv"],
            policy_metrics_json=winning_summary_path,
        )
    if winning_policy is None and not args.skip_registry:
        closure_payload = {
            "run_id": run_id,
            "day_id": args.day_id,
            "scope": "day054c_operational_reaffirmation",
            "decision": "keep_current_operational_policy",
            "reference_variant": args.operational_variant,
            "historical_invalidated_run_id": args.invalidated_run_id,
            "base_day054b_valid_run_id": args.base_valid_run_id,
            "metrics": {
                **operational_scored["row_metrics"],
                "top1_hit": float(operational_event_source["top1_hit"].mean()),
                "top2_hit": float(operational_event_source["top2_hit"].mean()),
                "coverage": float(operational_event_source["decision_final"].astype(str).str.strip().ne("").mean()),
                "test_events": int(len(operational_event_source)),
            },
            "refs": {
                "run_summary": str(output_paths["run_summary_json"]),
                "base_day054b_run_summary": str(output_paths["day054_root"] / f"{args.base_valid_run_id}_run_summary.json"),
                "invalidation": str(build_day054_invalidation_path(reports_root=paths["reports_root"], invalidated_run_id=args.invalidated_run_id)),
                "notebook_ref": NOTEBOOK_REF,
                "prompt_ref": PROMPT_REF,
            },
        }
        write_json(output_paths["registry_reaffirmation_metrics_json"], _to_jsonable(closure_payload))
        append_keep_current_operational_policy_to_registry(
            registry_csv=paths["registry_csv"],
            day_id=args.day_id,
            run_id=run_id,
            cutoff_date=cutoff_date,
            operational_row=operational_row,
            closure_metrics_payload_path=output_paths["registry_reaffirmation_metrics_json"],
        )

    return {
        "run_id": run_id,
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "decision": final_decision,
        "winning_policy": winning_policy,
    }


# SECTION: Entry point
def main() -> None:
    """Execute the Day 05.4 runner from the command line."""
    args = parse_args()
    result = run_day054(args)
    print(json.dumps(_to_jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
