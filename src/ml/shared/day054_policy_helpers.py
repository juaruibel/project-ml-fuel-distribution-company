from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.ml.product import recommend_supplier as rs
    from src.ml.rules import engine as rengine
    from src.ml.shared import functions as fc
    from src.ml.shared.numeric_parsing import parse_numeric_series_locale
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.ml.product import recommend_supplier as rs
    from src.ml.rules import engine as rengine
    from src.ml.shared import functions as fc
    from src.ml.shared.numeric_parsing import parse_numeric_series_locale


DEFAULT_AUDIT_INPUT_NAME = "day054_comparable_holdout"
DEFAULT_DAY054_INVALIDATED_REASON = "day054_scoring_contract_mismatch"
RAW_CATEGORICAL_FEATURE_COLUMNS = [
    "proveedor_candidato",
    "producto_canonico",
    "terminal_compra",
]
EVENT_SOURCE_CORE_COLUMNS = [
    "event_id",
    "recommended_supplier",
    "decision_pre_policy",
    "decision_final",
    "decision_source",
    "override_reason",
    "policy_applied_event",
    "policy_rule_id",
    "policy_reason_event",
    "source_model_variant",
    "rank_real_source",
    "top1_provider",
    "top1_hit",
    "top2_hit",
]


# SECTION: JSON helpers
def load_json(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


# SECTION: Text helpers
def slugify_token(value: str) -> str:
    """Build one filesystem-safe token from a free-text value."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


# SECTION: Registry helpers
def read_registry_row_by_variant(registry_csv: Path, model_variant: str) -> dict[str, Any]:
    """Read one registry row by canonical model variant."""
    registry_df = pd.read_csv(registry_csv, dtype=str, keep_default_na=False)
    mask = registry_df["model_variant"].astype(str).eq(model_variant)
    if not mask.any():
        raise ValueError(f"No existe model_variant={model_variant} en {registry_csv}")
    return registry_df.loc[mask].iloc[0].to_dict()


# SECTION: Metadata helpers
def resolve_shared_cutoff_date(champion_metadata_path: Path, baseline_metadata_path: Path) -> str:
    """Resolve the official cutoff date and verify both models share the same holdout split."""
    champion_metadata = load_json(champion_metadata_path)
    baseline_metadata = load_json(baseline_metadata_path)
    champion_cutoff = str(champion_metadata.get("cutoff_date", "")).strip()
    baseline_cutoff = str(baseline_metadata.get("cutoff_date", "")).strip()
    if champion_cutoff == "" or baseline_cutoff == "":
        raise ValueError("No se pudo resolver cutoff_date en los metadata de champion/baseline.")
    if champion_cutoff != baseline_cutoff:
        raise ValueError(
            "Champion y baseline no comparten el mismo cutoff_date: "
            f"{champion_cutoff} != {baseline_cutoff}"
        )
    return champion_cutoff


# SECTION: Dataset helpers
def filter_holdout_dataframe(dataframe: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """Filter one dataset down to the official post-cutoff holdout."""
    working = dataframe.copy()
    working["fecha_evento_dt"] = pd.to_datetime(working["fecha_evento"], errors="coerce")
    cutoff_dt = pd.Timestamp(cutoff_date)
    holdout = working.loc[working["fecha_evento_dt"] > cutoff_dt].copy()
    holdout = holdout.drop(columns=["fecha_evento_dt"])
    if holdout.empty:
        raise ValueError(f"El holdout quedó vacío al aplicar cutoff_date={cutoff_date}.")
    return holdout


# SECTION: Model scoring helpers
def build_contract_aligned_feature_matrix(
    dataframe: pd.DataFrame,
    expected_feature_columns: list[str],
) -> pd.DataFrame:
    """Build one feature matrix that respects the exact metadata contract of the evaluated reference."""
    working = dataframe.copy()
    incoming_columns = set(working.columns)
    expected_set = set(expected_feature_columns)
    if expected_set.issubset(incoming_columns):
        matrix = working[expected_feature_columns].copy()
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return matrix

    numeric_direct_columns = [
        column
        for column in expected_feature_columns
        if column in incoming_columns and column not in RAW_CATEGORICAL_FEATURE_COLUMNS
    ]
    numeric_frames: list[pd.DataFrame] = []
    if numeric_direct_columns:
        numeric_frame = working[numeric_direct_columns].copy()
        for column in numeric_direct_columns:
            numeric_frame[column] = parse_numeric_series_locale(numeric_frame[column]).fillna(0.0)
        numeric_frames.append(numeric_frame)

    dummy_frames: list[pd.DataFrame] = []
    for base_column in RAW_CATEGORICAL_FEATURE_COLUMNS:
        if base_column not in incoming_columns:
            continue
        normalized = (
            working[base_column]
            .astype("string")
            .fillna("UNKNOWN")
            .str.strip()
            .replace("", "UNKNOWN")
        )
        dummy_frame = pd.get_dummies(normalized, prefix=base_column, prefix_sep="_", dtype=float)
        dummy_frames.append(dummy_frame)

    if numeric_frames or dummy_frames:
        matrix = pd.concat([*numeric_frames, *dummy_frames], axis=1)
    else:
        matrix = pd.DataFrame(index=working.index)
    matrix = matrix.reindex(columns=expected_feature_columns, fill_value=0.0)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return matrix


# SECTION: Model scoring helpers
def run_contract_aware_inference_dataframe(
    *,
    input_df: pd.DataFrame,
    model: Any,
    expected_feature_columns: list[str],
    event_col: str,
    top_k: int,
) -> pd.DataFrame:
    """Run one inference dataframe using the exact metadata feature contract instead of the raw generic fallback."""
    prepared_df = rs.ensure_event_column(input_df, event_col=event_col)
    matrix = build_contract_aligned_feature_matrix(
        dataframe=prepared_df,
        expected_feature_columns=expected_feature_columns,
    )
    return rs.infer(
        df=prepared_df,
        matrix=matrix,
        model=model,
        event_col=event_col,
        top_k=top_k,
    )


# SECTION: Model scoring helpers
def score_model_holdout(
    *,
    dataset_path: Path,
    model_path: Path,
    metadata_path: Path,
    cutoff_date: str,
    top_k: int,
) -> dict[str, Any]:
    """Score one model on the official holdout using its exact metadata contract."""
    source_df = pd.read_csv(dataset_path, keep_default_na=False)
    holdout_df = filter_holdout_dataframe(source_df, cutoff_date=cutoff_date)
    model, metadata, expected_feature_columns = rs.load_model_bundle(
        model_path=model_path,
        metadata_path=metadata_path,
    )
    detail_df = run_contract_aware_inference_dataframe(
        input_df=holdout_df,
        model=model,
        expected_feature_columns=expected_feature_columns,
        event_col="event_id",
        top_k=top_k,
    )
    row_metrics = compute_row_metrics_from_detail(detail_df)
    event_metrics = compute_event_metrics_from_detail(detail_df)
    return {
        "detail_df": detail_df,
        "metadata": metadata,
        "row_metrics": row_metrics,
        "event_metrics": event_metrics,
    }


# SECTION: Metrics helpers
def compute_row_metrics_from_detail(detail_df: pd.DataFrame) -> dict[str, float]:
    """Compute row-level classification metrics from one scored detail dataframe."""
    y_true = pd.to_numeric(detail_df["target_elegido"], errors="coerce").fillna(0).astype(int)
    y_pred = pd.to_numeric(detail_df["pred_label"], errors="coerce").fillna(0).astype(int)
    metrics = fc.compute_row_metrics(y_true, y_pred)
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "f1_pos": float(metrics["f1_pos"]),
    }


# SECTION: Metrics helpers
def compute_event_metrics_from_detail(detail_df: pd.DataFrame) -> dict[str, float]:
    """Compute event-level top-k metrics from one scored detail dataframe."""
    if detail_df.empty:
        return {
            "top1_hit": 0.0,
            "top2_hit": 0.0,
            "coverage": 0.0,
            "test_events": 0,
        }
    working = detail_df.copy()
    working["event_id"] = working["event_id"].astype(str).str.strip()
    working["target_elegido"] = pd.to_numeric(working["target_elegido"], errors="coerce").fillna(0).astype(int)
    working["score_model"] = pd.to_numeric(working["score_model"], errors="coerce")
    events = int(working["event_id"].nunique())
    return {
        "top1_hit": float(fc.topk_hit_by_event(working, "score_model", k=1)),
        "top2_hit": float(fc.topk_hit_by_event(working, "score_model", k=2)),
        "coverage": 1.0 if events else 0.0,
        "test_events": events,
    }


# SECTION: Output helpers
def build_day054_output_paths(reports_root: Path, run_id: str) -> dict[str, Path]:
    """Build the canonical Day 05.4 output paths for one run id."""
    day054_root = reports_root / "metrics" / "day05_4"
    return {
        "day054_root": day054_root,
        "invalidations_root": day054_root / "invalidations",
        "policy_trials_csv": day054_root / f"{run_id}_policy_trials.csv",
        "policy_trials_json": day054_root / f"{run_id}_policy_trials.json",
        "policy_summary_json": day054_root / f"{run_id}_policy_summary.json",
        "run_summary_json": day054_root / f"{run_id}_run_summary.json",
        "registry_reaffirmation_metrics_json": day054_root / f"{run_id}_registry_reaffirmation_metrics.json",
        "event_level_audit_csv": day054_root / f"{run_id}_event_level_audit.csv",
        "candidate_level_audit_csv": day054_root / f"{run_id}_candidate_level_audit.csv",
        "artifact_inventory_md": reports_root / "day05_4_artifact_inventory.md",
    }


# SECTION: Output helpers
def build_source_input_path(day054_root: Path, run_id: str, input_name: str) -> Path:
    """Build the canonical stored input path for one intermediate scored source."""
    return day054_root / f"{run_id}_{slugify_token(input_name)}.csv"


# SECTION: Output helpers
def build_policy_audit_paths(report_root: Path, input_name: str, policy_variant: str, run_id: str) -> dict[str, Path]:
    """Build canonical per-policy audit paths under artifacts/public/postinferencia/audits/<YYYYMMDD>/."""
    run_date_utc = datetime.now(timezone.utc).strftime("%Y%m%d")
    audit_root = report_root / "postinferencia" / "audits" / run_date_utc
    safe_input = slugify_token(input_name)
    safe_policy = slugify_token(policy_variant)
    return {
        "detail": audit_root / f"{safe_input}_detalle_{safe_policy}_{run_id}.csv",
        "resumen_evento": audit_root / f"{safe_input}_resumen_evento_{safe_policy}_{run_id}.csv",
        "resumen_albaran": audit_root / f"{safe_input}_resumen_albaran_{safe_policy}_{run_id}.csv",
        "summary": audit_root / f"{safe_input}_summary_{safe_policy}_{run_id}.json",
    }


# SECTION: Invalidation helpers
def build_day054_invalidation_path(reports_root: Path, invalidated_run_id: str) -> Path:
    """Build the canonical invalidation JSON path for one invalidated Day 05.4 run."""
    return reports_root / "metrics" / "day05_4" / "invalidations" / f"{invalidated_run_id}_invalidated.json"


# SECTION: Invalidation helpers
def load_day054_invalidations(reports_root: Path) -> list[dict[str, Any]]:
    """Load every persisted Day 05.4 invalidation artifact under the canonical invalidations directory."""
    invalidations_root = reports_root / "metrics" / "day05_4" / "invalidations"
    if not invalidations_root.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for invalidation_path in sorted(invalidations_root.glob("*_invalidated.json")):
        payloads.append(load_json(invalidation_path))
    return payloads


# SECTION: Invalidation helpers
def load_invalidated_day054_run_ids(reports_root: Path) -> set[str]:
    """Return the set of Day 05.4 run ids that are currently quarantined as invalidated."""
    return {
        str(payload.get("invalidated_run_id", "")).strip()
        for payload in load_day054_invalidations(reports_root=reports_root)
        if str(payload.get("invalidated_run_id", "")).strip()
    }


# SECTION: Policy helpers
def run_existing_albaran_policy(
    *,
    scored_df: pd.DataFrame,
    input_name: str,
    run_id: str,
    day054_root: Path,
    report_root: Path,
    rules_csv: Path,
    albaran_policy: str,
) -> dict[str, Any]:
    """Apply the existing Day 03 albaran policy on one scored source and load its outputs."""
    raw_input_path = build_source_input_path(day054_root=day054_root, run_id=run_id, input_name=input_name)
    raw_input_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(raw_input_path, index=False)
    audit_paths = build_policy_audit_paths(
        report_root=report_root,
        input_name=input_name,
        policy_variant=albaran_policy,
        run_id=run_id,
    )
    rengine.run(
        input_csv=raw_input_path,
        output_csv=audit_paths["detail"],
        output_resumen_csv=audit_paths["resumen_evento"],
        output_resumen_albaran_csv=audit_paths["resumen_albaran"],
        rules_csv=rules_csv,
        mode="assist",
        albaran_policy=albaran_policy,
        summary_json=audit_paths["summary"],
    )
    return {
        "input_csv": raw_input_path,
        "paths": audit_paths,
        "detail_df": pd.read_csv(audit_paths["detail"], keep_default_na=False),
        "resumen_evento_df": pd.read_csv(audit_paths["resumen_evento"], keep_default_na=False),
        "resumen_albaran_df": pd.read_csv(audit_paths["resumen_albaran"], keep_default_na=False),
        "summary_json": load_json(audit_paths["summary"]),
    }


# SECTION: Ranking helpers
def _clean_text(value: Any) -> str:
    """Normalize one scalar into a stripped string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


# SECTION: Ranking helpers
def _prepare_detail_for_ranking(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize one scored detail dataframe for ranking-based event metrics."""
    working = detail_df.copy()
    working["event_id"] = working["event_id"].astype(str).str.strip()
    working["proveedor_candidato"] = working["proveedor_candidato"].astype(str).str.strip()
    working["rank_event_score"] = pd.to_numeric(working["rank_event_score"], errors="coerce")
    working["score_model"] = pd.to_numeric(working["score_model"], errors="coerce")
    working["target_elegido"] = pd.to_numeric(working["target_elegido"], errors="coerce").fillna(0).astype(int)
    working["linea_id"] = working.get("linea_id", "").astype(str).str.strip()
    if working["rank_event_score"].isna().any():
        working["rank_event_score"] = (
            working.groupby("event_id")["score_model"].rank(method="first", ascending=False)
        )
    working["rank_event_score"] = working["rank_event_score"].astype(int)
    return working


# SECTION: Ranking helpers
def _build_true_provider_map(detail_df: pd.DataFrame) -> dict[str, str]:
    """Build the event -> real provider map from the positive candidate rows."""
    positive = detail_df.loc[detail_df["target_elegido"] == 1].copy()
    positive = positive.sort_values(["event_id", "rank_event_score"], kind="mergesort")
    first_positive = positive.groupby("event_id", as_index=False).first()
    return dict(zip(first_positive["event_id"].astype(str), first_positive["proveedor_candidato"].astype(str)))


# SECTION: Ranking helpers
def _build_ranked_candidates_map(detail_df: pd.DataFrame) -> dict[str, list[str]]:
    """Build ranked candidate lists per event from one scored detail dataframe."""
    ranked = detail_df.sort_values(
        ["event_id", "rank_event_score", "score_model"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    mapping: dict[str, list[str]] = {}
    for event_id, group in ranked.groupby("event_id", sort=False):
        ordered = [provider for provider in group["proveedor_candidato"].astype(str).tolist() if provider]
        mapping[str(event_id)] = list(dict.fromkeys(ordered))
    return mapping


# SECTION: Ranking helpers
def _build_event_meta_map(detail_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Build one per-event metadata map from one scored detail dataframe."""
    top_rows = detail_df.sort_values(["event_id", "rank_event_score"], kind="mergesort").groupby("event_id", as_index=False).first()
    meta_cols = [
        "event_id",
        "fecha_evento",
        "albaran_id",
        "linea_id",
        "producto_canonico",
        "terminal_compra",
    ]
    return {
        row["event_id"]: {column: _clean_text(row.get(column, "")) for column in meta_cols if column != "event_id"}
        for row in top_rows[meta_cols].to_dict(orient="records")
    }


# SECTION: Event-source helpers
def build_event_source_frame(
    *,
    detail_df: pd.DataFrame,
    source_model_variant: str,
    resumen_evento_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one event-level source frame with comparable top1/top2 metrics."""
    detail = _prepare_detail_for_ranking(detail_df)
    true_provider_by_event = _build_true_provider_map(detail)
    ranked_candidates_by_event = _build_ranked_candidates_map(detail)
    event_meta_by_event = _build_event_meta_map(detail)
    if resumen_evento_df is not None:
        resumen_evento_df = resumen_evento_df.copy()
        resumen_evento_df["event_id"] = resumen_evento_df["event_id"].astype(str).str.strip()
        resumen_lookup = resumen_evento_df.set_index("event_id")
    else:
        resumen_lookup = None

    records: list[dict[str, Any]] = []
    for event_id, ranked_candidates in ranked_candidates_by_event.items():
        true_provider = _clean_text(true_provider_by_event.get(event_id, ""))
        if true_provider == "":
            continue
        meta = event_meta_by_event[event_id]
        recommended_supplier = ranked_candidates[0]
        decision_pre_policy = recommended_supplier
        decision_final = recommended_supplier
        decision_source = "model"
        override_reason = ""
        policy_applied_event = 0
        policy_rule_id = ""
        policy_reason_event = ""

        if resumen_lookup is not None and event_id in resumen_lookup.index:
            resumen_row = resumen_lookup.loc[event_id]
            decision_pre_policy = _clean_text(resumen_row.get("decision_pre_policy", recommended_supplier)) or recommended_supplier
            decision_final = _clean_text(resumen_row.get("decision_final", recommended_supplier)) or recommended_supplier
            decision_source = _clean_text(resumen_row.get("decision_source", "policy")) or "policy"
            override_reason = _clean_text(resumen_row.get("override_reason", ""))
            policy_applied_event = int(pd.to_numeric(resumen_row.get("policy_applied_event", 0), errors="coerce") or 0)
            policy_rule_id = _clean_text(resumen_row.get("policy_rule_id", ""))
            policy_reason_event = _clean_text(resumen_row.get("policy_reason_event", ""))

        ordered_after = [decision_final] + [provider for provider in ranked_candidates if provider != decision_final]
        rank_real_source = ordered_after.index(true_provider) + 1 if true_provider in ordered_after else len(ordered_after) + 1
        records.append(
            {
                "event_id": event_id,
                "fecha_evento": meta["fecha_evento"],
                "albaran_id": meta["albaran_id"],
                "linea_id": meta["linea_id"],
                "producto_canonico": meta["producto_canonico"],
                "terminal_compra": meta["terminal_compra"],
                "proveedor_real": true_provider,
                "recommended_supplier": recommended_supplier,
                "decision_pre_policy": decision_pre_policy,
                "decision_final": decision_final,
                "decision_source": decision_source,
                "override_reason": override_reason,
                "policy_applied_event": policy_applied_event,
                "policy_rule_id": policy_rule_id,
                "policy_reason_event": policy_reason_event,
                "source_model_variant": source_model_variant,
                "rank_real_source": rank_real_source,
                "top1_provider": decision_final,
                "top1_hit": int(decision_final == true_provider),
                "top2_hit": int(true_provider in ordered_after[:2]),
            }
        )
    return pd.DataFrame(records).sort_values(["fecha_evento", "event_id"], kind="mergesort").reset_index(drop=True)


# SECTION: Event-source helpers
def prefix_event_source_frame(source_df: pd.DataFrame, source_key: str) -> pd.DataFrame:
    """Prefix the core comparable columns of one source event frame."""
    prefixed = source_df[EVENT_SOURCE_CORE_COLUMNS].copy()
    rename_map = {
        column: f"{source_key}__{column}"
        for column in prefixed.columns
        if column != "event_id"
    }
    return prefixed.rename(columns=rename_map)


# SECTION: Event-source helpers
def build_event_master_frame(
    *,
    event_compare_df: pd.DataFrame,
    flag_frame_df: pd.DataFrame,
    source_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge comparable event features, flags and prefixed source summaries into one master table."""
    event_master = event_compare_df.merge(
        flag_frame_df,
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    for source_key, source_df in source_frames.items():
        event_master = event_master.merge(
            prefix_event_source_frame(source_df=source_df, source_key=source_key),
            on="event_id",
            how="left",
            validate="one_to_one",
        )
    for column in [
        "go_c_fallback_flag",
        "go_b_dominant_low_conf_flag",
        "go_b_dominant_flagged_candidates",
        "go_b_dominant_slice_event",
        "go_b_residual_contextual_selector_flag",
        "day054b_go_b_residual_SUPPLIER_050_clean_flag",
        "day054b_go_b_residual_SUPPLIER_019_low_conf_flag",
        "day054b_go_b_residual_outer_terminal_extension_flag",
        "day054b_go_b_residual_precision_flag",
        "day054c_go_b_residual_SUPPLIER_050_rank2_clean_flag",
        "day054c_go_b_residual_SUPPLIER_019_low_conf_flag",
        "day054c_go_b_residual_outer_terminal_extension_flag",
        "day054c_go_b_residual_rank2_precision_flag",
    ]:
        event_master[column] = pd.to_numeric(event_master.get(column, 0), errors="coerce").fillna(0).astype(int)
    return event_master


# SECTION: Candidate comparison helpers
def build_candidate_comparison_frame(
    *,
    baseline_detail_df: pd.DataFrame,
    champion_detail_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the comparable candidate universe baseline vs champion at one row per event/candidate."""
    champion_base = champion_detail_df.copy().rename(
        columns={
            "score_model": "score_pure_champion",
            "pred_label": "pred_label_pure_champion",
            "rank_event_score": "rank_pure_champion",
            "is_top1": "is_top1_pure_champion",
            "is_topk": "is_topk_pure_champion",
        }
    )
    baseline_view = baseline_detail_df[
        [
            "event_id",
            "proveedor_candidato",
            "score_model",
            "pred_label",
            "rank_event_score",
            "is_top1",
            "is_topk",
        ]
    ].copy().rename(
        columns={
            "score_model": "score_baseline",
            "pred_label": "pred_label_baseline",
            "rank_event_score": "rank_baseline",
            "is_top1": "is_top1_baseline",
            "is_topk": "is_topk_baseline",
        }
    )
    candidate_compare = champion_base.merge(
        baseline_view,
        on=["event_id", "proveedor_candidato"],
        how="inner",
        validate="one_to_one",
    )
    return candidate_compare


# SECTION: Event comparison helpers
def build_event_comparison_frame(candidate_compare_df: pd.DataFrame) -> pd.DataFrame:
    """Build the Day 05.3-style comparable event universe from candidate-level scores."""
    working = candidate_compare_df.copy()
    for numeric_column in [
        "target_elegido",
        "rank_baseline",
        "rank_pure_champion",
        "score_baseline",
        "score_pure_champion",
        "candidatos_evento_count",
        "coste_min_dia_proveedor",
        "ratio_vs_min_evento",
        "v41_transport_rank_event",
    ]:
        if numeric_column in working.columns:
            working[numeric_column] = pd.to_numeric(working[numeric_column], errors="coerce")

    positive_rows = working.loc[
        working["target_elegido"].fillna(0).astype(int) == 1,
        [
            "event_id",
            "proveedor_candidato",
            "rank_baseline",
            "rank_pure_champion",
            "score_baseline",
            "score_pure_champion",
        ],
    ].copy().rename(
        columns={
            "proveedor_candidato": "proveedor_real",
            "rank_baseline": "rank_real_baseline",
            "rank_pure_champion": "rank_real_pure_champion",
            "score_baseline": "score_real_baseline",
            "score_pure_champion": "score_real_pure_champion",
        }
    )

    baseline_top1 = working.loc[
        working["rank_baseline"].fillna(0).astype(int) == 1,
        ["event_id", "proveedor_candidato", "score_baseline"],
    ].copy().rename(
        columns={
            "proveedor_candidato": "top1_provider_baseline",
            "score_baseline": "top1_score_baseline",
        }
    )

    champion_top1 = working.loc[
        working["rank_pure_champion"].fillna(0).astype(int) == 1,
        ["event_id", "proveedor_candidato", "score_pure_champion"],
    ].copy().rename(
        columns={
            "proveedor_candidato": "top1_provider_pure_champion",
            "score_pure_champion": "top1_score_pure_champion",
        }
    )

    event_features = working.groupby("event_id", as_index=False).agg(
        fecha_evento=("fecha_evento", "first"),
        albaran_id=("albaran_id", "first"),
        linea_id=("linea_id", "first"),
        producto_canonico=("producto_canonico", "first"),
        terminal_compra=("terminal_compra", "first"),
        candidatos_evento_count=("candidatos_evento_count", "first"),
        min_coste_evento=("coste_min_dia_proveedor", "min"),
        max_coste_evento=("coste_min_dia_proveedor", "max"),
        mean_coste_evento=("coste_min_dia_proveedor", "mean"),
    )
    event_features["spread_coste_holdout_evento"] = (
        pd.to_numeric(event_features["max_coste_evento"], errors="coerce")
        - pd.to_numeric(event_features["min_coste_evento"], errors="coerce")
    )

    event_compare = (
        event_features
        .merge(positive_rows, on="event_id", how="inner", validate="one_to_one")
        .merge(baseline_top1, on="event_id", how="inner", validate="one_to_one")
        .merge(champion_top1, on="event_id", how="inner", validate="one_to_one")
    )
    event_compare["top1_hit_baseline"] = (
        event_compare["top1_provider_baseline"] == event_compare["proveedor_real"]
    ).astype(int)
    event_compare["top1_hit_pure_champion"] = (
        event_compare["top1_provider_pure_champion"] == event_compare["proveedor_real"]
    ).astype(int)
    event_compare["top2_hit_baseline"] = (
        pd.to_numeric(event_compare["rank_real_baseline"], errors="coerce") <= 2
    ).astype(int)
    event_compare["top2_hit_pure_champion"] = (
        pd.to_numeric(event_compare["rank_real_pure_champion"], errors="coerce") <= 2
    ).astype(int)
    event_compare["top1_changed_vs_baseline"] = (
        event_compare["top1_provider_baseline"] != event_compare["top1_provider_pure_champion"]
    ).astype(int)
    event_compare["delta_rank_real_champion_vs_baseline"] = (
        pd.to_numeric(event_compare["rank_real_pure_champion"], errors="coerce")
        - pd.to_numeric(event_compare["rank_real_baseline"], errors="coerce")
    )
    event_compare["top1_result_vs_baseline"] = pd.Series(
        pd.NA,
        index=event_compare.index,
        dtype="object",
    )
    event_compare["top2_result_vs_baseline"] = pd.Series(
        pd.NA,
        index=event_compare.index,
        dtype="object",
    )
    for target_col in ["top1_result_vs_baseline", "top2_result_vs_baseline"]:
        left_hit = "top1_hit_baseline" if target_col.startswith("top1") else "top2_hit_baseline"
        right_hit = "top1_hit_pure_champion" if target_col.startswith("top1") else "top2_hit_pure_champion"
        event_compare.loc[
            (event_compare[left_hit] == 0) & (event_compare[right_hit] == 1),
            target_col,
        ] = "champion_improves"
        event_compare.loc[
            (event_compare[left_hit] == 1) & (event_compare[right_hit] == 0),
            target_col,
        ] = "champion_worsens"
        event_compare.loc[
            (event_compare[left_hit] == 1) & (event_compare[right_hit] == 1),
            target_col,
        ] = "both_hit"
        event_compare.loc[
            (event_compare[left_hit] == 0) & (event_compare[right_hit] == 0),
            target_col,
        ] = "both_fail"

    event_compare["ranking_real_result_vs_baseline"] = "unknown"
    event_compare.loc[
        pd.to_numeric(event_compare["rank_real_pure_champion"], errors="coerce")
        < pd.to_numeric(event_compare["rank_real_baseline"], errors="coerce"),
        "ranking_real_result_vs_baseline",
    ] = "champion_ranks_real_better"
    event_compare.loc[
        pd.to_numeric(event_compare["rank_real_pure_champion"], errors="coerce")
        > pd.to_numeric(event_compare["rank_real_baseline"], errors="coerce"),
        "ranking_real_result_vs_baseline",
    ] = "baseline_ranks_real_better"
    event_compare.loc[
        pd.to_numeric(event_compare["rank_real_pure_champion"], errors="coerce")
        == pd.to_numeric(event_compare["rank_real_baseline"], errors="coerce"),
        "ranking_real_result_vs_baseline",
    ] = "same_rank"
    event_compare["both_fail_top1"] = (
        (event_compare["top1_hit_baseline"] == 0)
        & (event_compare["top1_hit_pure_champion"] == 0)
    ).astype(int)
    event_compare["both_fail_top2"] = (
        (event_compare["top2_hit_baseline"] == 0)
        & (event_compare["top2_hit_pure_champion"] == 0)
    ).astype(int)
    return event_compare.sort_values(["fecha_evento", "event_id"], kind="mergesort").reset_index(drop=True)
