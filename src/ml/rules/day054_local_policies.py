from __future__ import annotations

import pandas as pd


GO_B_DOMINANT_RATIO_THRESHOLD = 1.035
GO_B_DOMINANT_TRANSPORT_RANK_THRESHOLD = 5
DAY054B_SUPPLIER_019_LOW_CONF_SCORE_THRESHOLD = 0.723
DAY054C_SUPPLIER_050_RANK2_MAX = 2


# SECTION: Shared PRODUCT_003 helpers
def resolve_go_b_dominant_event_ids(event_compare_df: pd.DataFrame) -> set[str]:
    """Return the canonical dominant PRODUCT_003 event ids discovered in Notebook 19."""
    dominant_slice_mask = (
        (event_compare_df["producto_canonico"].astype(str) == "PRODUCT_003")
        & (event_compare_df["terminal_compra"].astype(str) == "TERMINAL_001")
        & (event_compare_df["proveedor_real"].astype(str) == "SUPPLIER_050")
        & (event_compare_df["top1_provider_baseline"].astype(str) == "SUPPLIER_009")
        & (event_compare_df["top1_provider_pure_champion"].astype(str) == "SUPPLIER_009")
        & (event_compare_df["top2_result_vs_baseline"].astype(str) == "champion_worsens")
    )
    return set(event_compare_df.loc[dominant_slice_mask, "event_id"].astype(str))


# SECTION: PRODUCT_005 flags
def compute_go_c_fallback_flags(event_compare_df: pd.DataFrame) -> pd.DataFrame:
    """Build the event-level PRODUCT_005 fallback flag from the notebook 19 operational slice."""
    flags = event_compare_df[["event_id"]].copy()
    flags["go_c_fallback_flag"] = (
        (event_compare_df["producto_canonico"].astype(str) == "PRODUCT_005")
        & (event_compare_df["top1_provider_baseline"].astype(str) == "SUPPLIER_009")
        & (event_compare_df["top1_provider_pure_champion"].astype(str) == "SUPPLIER_050")
    ).astype(int)
    return flags


# SECTION: PRODUCT_003 dominant flags
def compute_go_b_dominant_flags(event_compare_df: pd.DataFrame, candidate_compare_df: pd.DataFrame) -> pd.DataFrame:
    """Build the dominant PRODUCT_003 slice flag and its observable low-confidence fallback trigger."""
    dominant_event_ids = resolve_go_b_dominant_event_ids(event_compare_df=event_compare_df)
    candidate_flags = candidate_compare_df.merge(
        event_compare_df[
            [
                "event_id",
                "top1_provider_pure_champion",
                "top2_result_vs_baseline",
            ]
        ],
        on="event_id",
        how="inner",
        validate="many_to_one",
    ).copy()
    ratio_series = pd.to_numeric(
        candidate_flags.get("ratio_vs_min_evento", pd.Series(index=candidate_flags.index, dtype="float64")),
        errors="coerce",
    )
    transport_rank_series = pd.to_numeric(
        candidate_flags.get("v41_transport_rank_event", pd.Series(index=candidate_flags.index, dtype="float64")),
        errors="coerce",
    )
    candidate_flags["go_b_dominant_candidate_flag"] = (
        (candidate_flags["producto_canonico"].astype(str) == "PRODUCT_003")
        & (candidate_flags["terminal_compra"].astype(str) == "TERMINAL_001")
        & (candidate_flags["proveedor_candidato"].astype(str) == "SUPPLIER_050")
        & (candidate_flags["top1_provider_pure_champion"].astype(str) == "SUPPLIER_009")
        & (ratio_series >= GO_B_DOMINANT_RATIO_THRESHOLD)
        & (transport_rank_series >= GO_B_DOMINANT_TRANSPORT_RANK_THRESHOLD)
    ).astype(int)
    event_flags = candidate_flags.groupby("event_id", as_index=False).agg(
        go_b_dominant_low_conf_flag=("go_b_dominant_candidate_flag", "max"),
        go_b_dominant_flagged_candidates=("go_b_dominant_candidate_flag", "sum"),
    )
    event_flags["go_b_dominant_slice_event"] = (
        event_flags["event_id"].astype(str).isin(dominant_event_ids)
    ).astype(int)
    return event_flags


# SECTION: PRODUCT_003 residual flags
def compute_go_b_residual_research_flags(event_compare_df: pd.DataFrame) -> pd.DataFrame:
    """Build the notebook 19 PRODUCT_003 residual research-only contextual selector flag."""
    dominant_event_ids = resolve_go_b_dominant_event_ids(event_compare_df=event_compare_df)
    flags = event_compare_df[["event_id"]].copy()
    flags["go_b_residual_contextual_selector_flag"] = (
        (event_compare_df["producto_canonico"].astype(str) == "PRODUCT_003")
        & (~event_compare_df["event_id"].astype(str).isin(dominant_event_ids))
        & (event_compare_df["terminal_compra"].astype(str) == "TERMINAL_001")
        & (event_compare_df["top1_provider_baseline"].astype(str) == "SUPPLIER_009")
        & (
            event_compare_df["top1_provider_pure_champion"].astype(str).isin(["SUPPLIER_019", "SUPPLIER_050"])
        )
    ).astype(int)
    return flags


# SECTION: Day 05.4b residual precision flags
def compute_day054b_go_b_residual_precision_flags(event_compare_df: pd.DataFrame) -> pd.DataFrame:
    """Build the strict Day 05.4b PRODUCT_003 residual precision flags from observable event-level signals."""
    dominant_event_ids = resolve_go_b_dominant_event_ids(event_compare_df=event_compare_df)
    champion_score = pd.to_numeric(
        event_compare_df.get("top1_score_pure_champion", pd.Series(index=event_compare_df.index, dtype="float64")),
        errors="coerce",
    )
    base_mask = (
        (event_compare_df["producto_canonico"].astype(str) == "PRODUCT_003")
        & (~event_compare_df["event_id"].astype(str).isin(dominant_event_ids))
        & (event_compare_df["top1_provider_baseline"].astype(str) == "SUPPLIER_009")
    )
    flags = event_compare_df[["event_id"]].copy()
    flags["day054b_go_b_residual_SUPPLIER_050_clean_flag"] = (
        base_mask
        & (event_compare_df["terminal_compra"].astype(str) == "TERMINAL_001")
        & (event_compare_df["top1_provider_pure_champion"].astype(str) == "SUPPLIER_050")
    ).astype(int)
    flags["day054b_go_b_residual_SUPPLIER_019_low_conf_flag"] = (
        base_mask
        & (event_compare_df["terminal_compra"].astype(str) == "TERMINAL_001")
        & (event_compare_df["top1_provider_pure_champion"].astype(str) == "SUPPLIER_019")
        & (champion_score <= DAY054B_SUPPLIER_019_LOW_CONF_SCORE_THRESHOLD)
    ).astype(int)
    flags["day054b_go_b_residual_outer_terminal_extension_flag"] = (
        base_mask
        & (event_compare_df["terminal_compra"].astype(str).isin(["TERMINAL_002", "TERMINAL_003"]))
        & (event_compare_df["top1_provider_pure_champion"].astype(str).isin(["SUPPLIER_011", "SUPPLIER_020"]))
    ).astype(int)
    precision_columns = [
        "day054b_go_b_residual_SUPPLIER_050_clean_flag",
        "day054b_go_b_residual_SUPPLIER_019_low_conf_flag",
        "day054b_go_b_residual_outer_terminal_extension_flag",
    ]
    flags["day054b_go_b_residual_precision_flag"] = (
        flags[precision_columns].sum(axis=1) > 0
    ).astype(int)
    return flags


# SECTION: Day 05.4c residual precision flags
def compute_day054c_go_b_residual_precision_flags(event_compare_df: pd.DataFrame) -> pd.DataFrame:
    """Build the Day 05.4c residual precision flags using the audited SUPPLIER_050 rank<=2 hardening."""
    dominant_event_ids = resolve_go_b_dominant_event_ids(event_compare_df=event_compare_df)
    champion_score = pd.to_numeric(
        event_compare_df.get("top1_score_pure_champion", pd.Series(index=event_compare_df.index, dtype="float64")),
        errors="coerce",
    )
    transport_rank = pd.to_numeric(
        event_compare_df.get("top1_v41_transport_rank_event_pure_champion", event_compare_df.get("v41_transport_rank_event")),
        errors="coerce",
    )
    base_mask = (
        (event_compare_df["producto_canonico"].astype(str) == "PRODUCT_003")
        & (~event_compare_df["event_id"].astype(str).isin(dominant_event_ids))
        & (event_compare_df["top1_provider_baseline"].astype(str) == "SUPPLIER_009")
    )
    flags = event_compare_df[["event_id"]].copy()
    flags["day054c_go_b_residual_SUPPLIER_050_rank2_clean_flag"] = (
        base_mask
        & (event_compare_df["terminal_compra"].astype(str) == "TERMINAL_001")
        & (event_compare_df["top1_provider_pure_champion"].astype(str) == "SUPPLIER_050")
        & (transport_rank <= DAY054C_SUPPLIER_050_RANK2_MAX)
    ).astype(int)
    flags["day054c_go_b_residual_SUPPLIER_019_low_conf_flag"] = (
        base_mask
        & (event_compare_df["terminal_compra"].astype(str) == "TERMINAL_001")
        & (event_compare_df["top1_provider_pure_champion"].astype(str) == "SUPPLIER_019")
        & (champion_score <= DAY054B_SUPPLIER_019_LOW_CONF_SCORE_THRESHOLD)
    ).astype(int)
    flags["day054c_go_b_residual_outer_terminal_extension_flag"] = (
        base_mask
        & (event_compare_df["terminal_compra"].astype(str).isin(["TERMINAL_002", "TERMINAL_003"]))
        & (event_compare_df["top1_provider_pure_champion"].astype(str).isin(["SUPPLIER_011", "SUPPLIER_020"]))
    ).astype(int)
    precision_columns = [
        "day054c_go_b_residual_SUPPLIER_050_rank2_clean_flag",
        "day054c_go_b_residual_SUPPLIER_019_low_conf_flag",
        "day054c_go_b_residual_outer_terminal_extension_flag",
    ]
    flags["day054c_go_b_residual_rank2_precision_flag"] = (
        flags[precision_columns].sum(axis=1) > 0
    ).astype(int)
    return flags


# SECTION: Flag assembly
def build_day054_flag_frame(event_compare_df: pd.DataFrame, candidate_compare_df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full Day 05.4 flag frame at event level."""
    flag_frame = compute_go_c_fallback_flags(event_compare_df=event_compare_df)
    flag_frame = flag_frame.merge(
        compute_go_b_dominant_flags(
            event_compare_df=event_compare_df,
            candidate_compare_df=candidate_compare_df,
        ),
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    flag_frame = flag_frame.merge(
        compute_go_b_residual_research_flags(event_compare_df=event_compare_df),
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    flag_frame = flag_frame.merge(
        compute_day054b_go_b_residual_precision_flags(event_compare_df=event_compare_df),
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    flag_frame = flag_frame.merge(
        compute_day054c_go_b_residual_precision_flags(event_compare_df=event_compare_df),
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
        flag_frame[column] = pd.to_numeric(flag_frame.get(column, 0), errors="coerce").fillna(0).astype(int)
    return flag_frame
