from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# SECTION: Shared builders
def _build_selection_frame(event_master_df: pd.DataFrame) -> pd.DataFrame:
    """Start one selection frame aligned 1:1 with the event master table."""
    return event_master_df[["event_id"]].copy()


# SECTION: Shared builders
def _merge_reason(primary_reason: str, secondary_reason: str) -> str:
    """Merge one strategy reason with one source-side override reason."""
    primary = str(primary_reason or "").strip()
    secondary = str(secondary_reason or "").strip()
    if primary and secondary:
        return f"{primary}|{secondary}"
    return primary or secondary


# SECTION: Champion pass-through
def apply_keep_champion_strategy(event_master_df: pd.DataFrame, policy_variant: str) -> pd.DataFrame:
    """Keep the champion decisions unchanged for all events."""
    selection = _build_selection_frame(event_master_df)
    selection["selected_source"] = "pure_champion"
    selection["policy_applied_event"] = 0
    selection["policy_reason_event"] = ""
    selection["policy_variant"] = policy_variant
    return selection


# SECTION: Diagnostic flagging
def apply_flag_only_strategy(
    event_master_df: pd.DataFrame,
    policy_variant: str,
    flag_column: str,
) -> pd.DataFrame:
    """Mark one diagnostic flag while keeping the champion as the active source."""
    selection = _build_selection_frame(event_master_df)
    selection[flag_column] = pd.to_numeric(event_master_df[flag_column], errors="coerce").fillna(0).astype(int)
    selection["selected_source"] = "pure_champion"
    selection["policy_applied_event"] = selection[flag_column]
    selection["policy_reason_event"] = np.where(
        selection[flag_column] == 1,
        flag_column,
        "",
    )
    selection["policy_variant"] = policy_variant
    return selection


# SECTION: Single-flag fallback
def apply_fallback_to_source_strategy(
    event_master_df: pd.DataFrame,
    policy_variant: str,
    flag_column: str,
    fallback_source: str,
) -> pd.DataFrame:
    """Fallback from champion to one alternate source whenever one flag is active."""
    selection = _build_selection_frame(event_master_df)
    selection[flag_column] = pd.to_numeric(event_master_df[flag_column], errors="coerce").fillna(0).astype(int)
    selection["selected_source"] = np.where(
        selection[flag_column] == 1,
        fallback_source,
        "pure_champion",
    )
    selection["policy_applied_event"] = selection[flag_column]
    selection["policy_reason_event"] = np.where(
        selection[flag_column] == 1,
        f"{flag_column}_to_{fallback_source}",
        "",
    )
    selection["policy_variant"] = policy_variant
    return selection


# SECTION: Composite fallback
def apply_composite_fallback_strategy(
    event_master_df: pd.DataFrame,
    policy_variant: str,
    flag_columns: list[str],
    fallback_source: str,
) -> pd.DataFrame:
    """Fallback to one alternate source when any of the configured flags is active."""
    selection = _build_selection_frame(event_master_df)
    for flag_column in flag_columns:
        selection[flag_column] = pd.to_numeric(event_master_df[flag_column], errors="coerce").fillna(0).astype(int)
    selection["policy_applied_event"] = (
        selection[flag_columns].sum(axis=1) > 0
    ).astype(int)
    selection["selected_source"] = np.where(
        selection["policy_applied_event"] == 1,
        fallback_source,
        "pure_champion",
    )
    selection["policy_reason_event"] = selection.apply(
        lambda row: "|".join(
            [f"{flag_column}_to_{fallback_source}" for flag_column in flag_columns if int(row[flag_column]) == 1]
        ),
        axis=1,
    )
    selection["policy_variant"] = policy_variant
    return selection


# SECTION: Research selector
def apply_research_contextual_selector_strategy(
    event_master_df: pd.DataFrame,
    policy_variant: str,
    flag_column: str,
    fallback_source: str,
) -> pd.DataFrame:
    """Run one research-only contextual selector to measure possible ensemble value."""
    return apply_fallback_to_source_strategy(
        event_master_df=event_master_df,
        policy_variant=policy_variant,
        flag_column=flag_column,
        fallback_source=fallback_source,
    )


# SECTION: Strategy dispatch
def apply_policy_selection_strategy(
    event_master_df: pd.DataFrame,
    policy_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Dispatch one Day 05.4 catalog entry to the corresponding strategy implementation."""
    strategy_name = str(policy_metadata["strategy_name"])
    if strategy_name == "flag_only":
        return apply_flag_only_strategy(
            event_master_df=event_master_df,
            policy_variant=str(policy_metadata["policy_variant"]),
            flag_column=str(policy_metadata["flag_column"]),
        )
    if strategy_name == "fallback":
        return apply_fallback_to_source_strategy(
            event_master_df=event_master_df,
            policy_variant=str(policy_metadata["policy_variant"]),
            flag_column=str(policy_metadata["flag_column"]),
            fallback_source=str(policy_metadata["fallback_source"]),
        )
    if strategy_name == "composite_fallback":
        return apply_composite_fallback_strategy(
            event_master_df=event_master_df,
            policy_variant=str(policy_metadata["policy_variant"]),
            flag_columns=list(policy_metadata["flag_columns"]),
            fallback_source=str(policy_metadata["fallback_source"]),
        )
    if strategy_name == "research_contextual_selector":
        return apply_research_contextual_selector_strategy(
            event_master_df=event_master_df,
            policy_variant=str(policy_metadata["policy_variant"]),
            flag_column=str(policy_metadata["flag_column"]),
            fallback_source=str(policy_metadata["fallback_source"]),
        )
    if strategy_name == "keep_champion":
        return apply_keep_champion_strategy(
            event_master_df=event_master_df,
            policy_variant=str(policy_metadata["policy_variant"]),
        )
    raise ValueError(f"Estrategia Day 05.4 no soportada: {strategy_name}")


# SECTION: Materialization
def materialize_policy_event_summary(
    event_master_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    policy_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Materialize one final event-level policy summary from selected source assignments."""
    merged = event_master_df.merge(selection_df, on="event_id", how="inner", validate="one_to_one")
    records: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        selected_source = str(row["selected_source"]).strip()
        selected_prefix = f"{selected_source}__"
        champion_prefix = "pure_champion__"
        baseline_prefix = "baseline__"
        op_prefix = "operational_policy_reference__"
        baseline_policy_prefix = "baseline_with_policy__"
        policy_applied = int(row.get("policy_applied_event", 0) or 0)
        source_override_reason = row.get(f"{selected_prefix}override_reason", "")
        policy_reason = row.get("policy_reason_event", "")
        records.append(
            {
                "event_id": row["event_id"],
                "fecha_evento": row["fecha_evento"],
                "albaran_id": row["albaran_id"],
                "linea_id": row.get("linea_id", ""),
                "producto_canonico": row["producto_canonico"],
                "terminal_compra": row["terminal_compra"],
                "proveedor_real": row["proveedor_real"],
                "recommended_supplier": row.get(f"{champion_prefix}recommended_supplier", ""),
                "decision_pre_policy": row.get(f"{champion_prefix}decision_final", ""),
                "decision_final": row.get(f"{selected_prefix}decision_final", ""),
                "decision_source": selected_source,
                "override_reason": _merge_reason(policy_reason, source_override_reason),
                "policy_applied_event": policy_applied,
                "policy_rule_id": policy_metadata["policy_variant"] if policy_applied == 1 else "",
                "policy_reason_event": policy_reason if policy_applied == 1 else "",
                "policy_variant": policy_metadata["policy_variant"],
                "family": policy_metadata["family"],
                "target_metric": policy_metadata["target_metric"],
                "promotion_eligible": int(policy_metadata["promotion_eligible"]),
                "fallback_source": policy_metadata["fallback_source"],
                "selected_source_model_variant": row.get(f"{selected_prefix}source_model_variant", ""),
                "selected_top1_provider": row.get(f"{selected_prefix}top1_provider", ""),
                "selected_top1_hit": int(row.get(f"{selected_prefix}top1_hit", 0) or 0),
                "selected_top2_hit": int(row.get(f"{selected_prefix}top2_hit", 0) or 0),
                "selected_rank_real": int(row.get(f"{selected_prefix}rank_real_source", 999) or 999),
                "baseline_top1_hit": int(row.get(f"{baseline_prefix}top1_hit", 0) or 0),
                "baseline_top2_hit": int(row.get(f"{baseline_prefix}top2_hit", 0) or 0),
                "baseline_with_policy_top1_hit": int(row.get(f"{baseline_policy_prefix}top1_hit", 0) or 0),
                "baseline_with_policy_top2_hit": int(row.get(f"{baseline_policy_prefix}top2_hit", 0) or 0),
                "pure_champion_top1_hit": int(row.get(f"{champion_prefix}top1_hit", 0) or 0),
                "pure_champion_top2_hit": int(row.get(f"{champion_prefix}top2_hit", 0) or 0),
                "operational_policy_reference_top1_hit": int(row.get(f"{op_prefix}top1_hit", 0) or 0),
                "operational_policy_reference_top2_hit": int(row.get(f"{op_prefix}top2_hit", 0) or 0),
                "go_c_fallback_flag": int(row.get("go_c_fallback_flag", 0) or 0),
                "go_b_dominant_low_conf_flag": int(row.get("go_b_dominant_low_conf_flag", 0) or 0),
                "go_b_dominant_flagged_candidates": int(row.get("go_b_dominant_flagged_candidates", 0) or 0),
                "go_b_dominant_slice_event": int(row.get("go_b_dominant_slice_event", 0) or 0),
                "go_b_residual_contextual_selector_flag": int(row.get("go_b_residual_contextual_selector_flag", 0) or 0),
                "day054b_go_b_residual_SUPPLIER_050_clean_flag": int(row.get("day054b_go_b_residual_SUPPLIER_050_clean_flag", 0) or 0),
                "day054b_go_b_residual_SUPPLIER_019_low_conf_flag": int(row.get("day054b_go_b_residual_SUPPLIER_019_low_conf_flag", 0) or 0),
                "day054b_go_b_residual_outer_terminal_extension_flag": int(
                    row.get("day054b_go_b_residual_outer_terminal_extension_flag", 0) or 0
                ),
                "day054b_go_b_residual_precision_flag": int(row.get("day054b_go_b_residual_precision_flag", 0) or 0),
                "day054c_go_b_residual_SUPPLIER_050_rank2_clean_flag": int(
                    row.get("day054c_go_b_residual_SUPPLIER_050_rank2_clean_flag", 0) or 0
                ),
                "day054c_go_b_residual_SUPPLIER_019_low_conf_flag": int(
                    row.get("day054c_go_b_residual_SUPPLIER_019_low_conf_flag", 0) or 0
                ),
                "day054c_go_b_residual_outer_terminal_extension_flag": int(
                    row.get("day054c_go_b_residual_outer_terminal_extension_flag", 0) or 0
                ),
                "day054c_go_b_residual_rank2_precision_flag": int(
                    row.get("day054c_go_b_residual_rank2_precision_flag", 0) or 0
                ),
            }
        )
    return pd.DataFrame(records).sort_values(["fecha_evento", "event_id"], kind="mergesort").reset_index(drop=True)


# SECTION: Materialization
def materialize_policy_detail_frame(candidate_compare_df: pd.DataFrame, policy_event_df: pd.DataFrame) -> pd.DataFrame:
    """Attach event-level policy outcomes back to the comparable candidate universe."""
    event_view = policy_event_df[
        [
            "event_id",
            "policy_variant",
            "family",
            "target_metric",
            "promotion_eligible",
            "fallback_source",
            "selected_source_model_variant",
            "decision_pre_policy",
            "decision_final",
            "decision_source",
            "override_reason",
            "policy_applied_event",
            "policy_rule_id",
            "policy_reason_event",
            "selected_top1_provider",
            "selected_top1_hit",
            "selected_top2_hit",
            "selected_rank_real",
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
        ]
    ].copy()
    return candidate_compare_df.merge(
        event_view,
        on="event_id",
        how="inner",
        validate="many_to_one",
    )


# SECTION: Materialization
def materialize_selected_source_detail_frame(
    source_detail_frames: dict[str, pd.DataFrame],
    policy_event_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compose one candidate-level detail frame by taking each event from its selected source."""
    source_selection = policy_event_df[
        [
            "event_id",
            "policy_variant",
            "decision_pre_policy",
            "decision_final",
            "decision_source",
            "override_reason",
            "policy_applied_event",
            "policy_rule_id",
            "policy_reason_event",
        ]
    ].copy()
    source_selection["event_id"] = source_selection["event_id"].astype(str).str.strip()
    source_selection["decision_source"] = source_selection["decision_source"].astype(str).str.strip()

    detail_frames: list[pd.DataFrame] = []
    for source_key, detail_df in source_detail_frames.items():
        chosen_events = source_selection.loc[source_selection["decision_source"] == str(source_key)].copy()
        if chosen_events.empty:
            continue
        working = detail_df.copy()
        working["event_id"] = working["event_id"].astype(str).str.strip()
        selected = working.merge(
            chosen_events,
            on="event_id",
            how="inner",
            validate="many_to_one",
        )
        selected["selected_source"] = str(source_key)
        detail_frames.append(selected)

    if not detail_frames:
        return pd.DataFrame()

    output = pd.concat(detail_frames, ignore_index=True)
    output = output.sort_values(["event_id", "rank_event_score"], kind="mergesort").reset_index(drop=True)
    return output
