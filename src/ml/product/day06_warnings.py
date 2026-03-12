from __future__ import annotations

import pandas as pd

from src.ml.rules.day054_local_policies import (
    DAY054B_SUPPLIER_019_LOW_CONF_SCORE_THRESHOLD,
    DAY054C_SUPPLIER_050_RANK2_MAX,
    GO_B_DOMINANT_RATIO_THRESHOLD,
    GO_B_DOMINANT_TRANSPORT_RANK_THRESHOLD,
)


def build_operational_warning_frame(
    *,
    baseline_detail_df: pd.DataFrame,
    champion_detail_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build observable low-confidence warnings derived from the Day 05.3/05.4 slices."""
    candidate_compare_df = champion_detail_df.copy().rename(
        columns={
            "score_model": "score_pure_champion",
            "pred_label": "pred_label_pure_champion",
            "rank_event_score": "rank_pure_champion",
            "is_top1": "is_top1_pure_champion",
            "is_topk": "is_topk_pure_champion",
        }
    ).merge(
        baseline_detail_df[
            [
                "event_id",
                "proveedor_candidato",
                "score_model",
                "pred_label",
                "rank_event_score",
                "is_top1",
                "is_topk",
            ]
        ].rename(
            columns={
                "score_model": "score_baseline",
                "pred_label": "pred_label_baseline",
                "rank_event_score": "rank_baseline",
                "is_top1": "is_top1_baseline",
                "is_topk": "is_topk_baseline",
            }
        ),
        on=["event_id", "proveedor_candidato"],
        how="inner",
        validate="one_to_one",
    )
    for optional_column in ["albaran_id", "linea_id"]:
        if optional_column not in candidate_compare_df.columns:
            candidate_compare_df[optional_column] = ""

    baseline_top1 = build_top1_frame(detail_df=baseline_detail_df, source_key="baseline")
    champion_top1 = build_top1_frame(detail_df=champion_detail_df, source_key="pure_champion")
    event_features = candidate_compare_df.groupby("event_id", as_index=False).agg(
        fecha_evento=("fecha_evento", "first"),
        albaran_id=("albaran_id", "first"),
        linea_id=("linea_id", "first"),
        producto_canonico=("producto_canonico", "first"),
        terminal_compra=("terminal_compra", "first"),
    )
    event_compare_df = (
        event_features
        .merge(baseline_top1, on="event_id", how="left", validate="one_to_one")
        .merge(champion_top1, on="event_id", how="left", validate="one_to_one")
    )
    event_compare_df["baseline_champion_disagreement_flag"] = (
        event_compare_df["top1_provider_baseline"].astype(str)
        != event_compare_df["top1_provider_pure_champion"].astype(str)
    ).astype(int)
    event_compare_df["go_c_review_flag"] = (
        (event_compare_df["producto_canonico"].astype(str) == "PRODUCT_005")
        & (event_compare_df["top1_provider_baseline"].astype(str) == "SUPPLIER_009")
        & (event_compare_df["top1_provider_pure_champion"].astype(str) == "SUPPLIER_050")
    ).astype(int)

    candidate_flags = candidate_compare_df.merge(
        event_compare_df[
            [
                "event_id",
                "top1_provider_baseline",
                "top1_provider_pure_champion",
                "top1_score_pure_champion",
                "top1_v41_transport_rank_event_pure_champion",
            ]
        ],
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    ratio_series = pd.to_numeric(candidate_flags.get("ratio_vs_min_evento"), errors="coerce")
    champion_transport_rank = pd.to_numeric(
        candidate_flags.get("top1_v41_transport_rank_event_pure_champion"),
        errors="coerce",
    )
    champion_score = pd.to_numeric(
        candidate_flags.get("top1_score_pure_champion"),
        errors="coerce",
    )
    candidate_flags["go_b_dominant_review_candidate_flag"] = (
        (candidate_flags["producto_canonico"].astype(str) == "PRODUCT_003")
        & (candidate_flags["terminal_compra"].astype(str) == "TERMINAL_001")
        & (candidate_flags["proveedor_candidato"].astype(str) == "SUPPLIER_050")
        & (candidate_flags["top1_provider_pure_champion"].astype(str) == "SUPPLIER_009")
        & (ratio_series >= GO_B_DOMINANT_RATIO_THRESHOLD)
        & (pd.to_numeric(candidate_flags.get("v41_transport_rank_event"), errors="coerce") >= GO_B_DOMINANT_TRANSPORT_RANK_THRESHOLD)
    ).astype(int)
    candidate_flags["go_b_residual_SUPPLIER_050_rank2_flag"] = (
        (candidate_flags["producto_canonico"].astype(str) == "PRODUCT_003")
        & (candidate_flags["terminal_compra"].astype(str) == "TERMINAL_001")
        & (candidate_flags["top1_provider_pure_champion"].astype(str) == "SUPPLIER_050")
        & (champion_transport_rank <= DAY054C_SUPPLIER_050_RANK2_MAX)
    ).astype(int)
    candidate_flags["go_b_residual_SUPPLIER_019_low_conf_flag"] = (
        (candidate_flags["producto_canonico"].astype(str) == "PRODUCT_003")
        & (candidate_flags["terminal_compra"].astype(str) == "TERMINAL_001")
        & (candidate_flags["top1_provider_pure_champion"].astype(str) == "SUPPLIER_019")
        & (champion_score <= DAY054B_SUPPLIER_019_LOW_CONF_SCORE_THRESHOLD)
    ).astype(int)
    candidate_flags["go_b_residual_outer_terminal_flag"] = (
        (candidate_flags["producto_canonico"].astype(str) == "PRODUCT_003")
        & (candidate_flags["terminal_compra"].astype(str).isin(["TERMINAL_002", "TERMINAL_003"]))
        & (candidate_flags["top1_provider_pure_champion"].astype(str).isin(["SUPPLIER_011", "SUPPLIER_020"]))
    ).astype(int)

    event_flags = candidate_flags.groupby("event_id", as_index=False).agg(
        go_b_dominant_review_flag=("go_b_dominant_review_candidate_flag", "max"),
        go_b_residual_SUPPLIER_050_rank2_flag=("go_b_residual_SUPPLIER_050_rank2_flag", "max"),
        go_b_residual_SUPPLIER_019_low_conf_flag=("go_b_residual_SUPPLIER_019_low_conf_flag", "max"),
        go_b_residual_outer_terminal_flag=("go_b_residual_outer_terminal_flag", "max"),
    )
    warning_df = event_compare_df.merge(event_flags, on="event_id", how="left", validate="one_to_one")
    numeric_flag_columns = [
        "go_b_dominant_review_flag",
        "go_b_residual_SUPPLIER_050_rank2_flag",
        "go_b_residual_SUPPLIER_019_low_conf_flag",
        "go_b_residual_outer_terminal_flag",
    ]
    for column in numeric_flag_columns:
        if column in warning_df.columns:
            warning_df[column] = pd.to_numeric(warning_df[column], errors="coerce").fillna(0).astype(int)
    warning_df["low_confidence_flag"] = (
        warning_df[
            [
                "baseline_champion_disagreement_flag",
                "go_c_review_flag",
                "go_b_dominant_review_flag",
                "go_b_residual_SUPPLIER_050_rank2_flag",
                "go_b_residual_SUPPLIER_019_low_conf_flag",
                "go_b_residual_outer_terminal_flag",
            ]
        ].sum(axis=1)
        > 0
    ).astype(int)
    warning_df["warning_reasons"] = warning_df.apply(build_warning_reason_text, axis=1)
    return warning_df[
        [
            "event_id",
            "low_confidence_flag",
            "warning_reasons",
            "baseline_champion_disagreement_flag",
            "go_c_review_flag",
            "go_b_dominant_review_flag",
            "go_b_residual_SUPPLIER_050_rank2_flag",
            "go_b_residual_SUPPLIER_019_low_conf_flag",
            "go_b_residual_outer_terminal_flag",
        ]
    ]


def build_top1_frame(*, detail_df: pd.DataFrame, source_key: str) -> pd.DataFrame:
    """Build one event-level top1 frame from one scored detail dataframe."""
    working = detail_df.copy()
    if "v41_transport_rank_event" not in working.columns:
        working["v41_transport_rank_event"] = pd.NA
    working["rank_event_score"] = pd.to_numeric(working["rank_event_score"], errors="coerce").fillna(999).astype(int)
    top_rows = (
        working.sort_values(["event_id", "rank_event_score"], kind="mergesort")
        .groupby("event_id", as_index=False)
        .first()
    )
    return top_rows.rename(
        columns={
            "proveedor_candidato": f"top1_provider_{source_key}",
            "score_model": f"top1_score_{source_key}",
            "v41_transport_rank_event": f"top1_v41_transport_rank_event_{source_key}",
        }
    )[
        [
            "event_id",
            f"top1_provider_{source_key}",
            f"top1_score_{source_key}",
            f"top1_v41_transport_rank_event_{source_key}",
        ]
    ]


def build_warning_reason_text(row: pd.Series) -> str:
    """Convert one warning flag row into one human-readable warning string."""
    reason_map = [
        ("baseline_champion_disagreement_flag", "baseline_vs_champion_disagreement"),
        ("go_c_review_flag", "go_c_slice_review"),
        ("go_b_dominant_review_flag", "go_b_dominant_review"),
        ("go_b_residual_SUPPLIER_050_rank2_flag", "go_b_residual_SUPPLIER_050_rank2_review"),
        ("go_b_residual_SUPPLIER_019_low_conf_flag", "go_b_residual_SUPPLIER_019_low_conf_review"),
        ("go_b_residual_outer_terminal_flag", "go_b_residual_outer_terminal_review"),
    ]
    return "|".join(
        label
        for flag_column, label in reason_map
        if int(pd.to_numeric(row.get(flag_column, 0), errors="coerce") or 0) == 1
    )


def merge_warning_frame(*, detail_df: pd.DataFrame, warning_df: pd.DataFrame) -> pd.DataFrame:
    """Merge one event-level warning frame into one dataframe that carries `event_id`."""
    if detail_df.empty or warning_df.empty or "event_id" not in detail_df.columns:
        return detail_df.copy()
    merged = detail_df.merge(warning_df, on="event_id", how="left", validate="many_to_one")
    for column in [
        "low_confidence_flag",
        "baseline_champion_disagreement_flag",
        "go_c_review_flag",
        "go_b_dominant_review_flag",
        "go_b_residual_SUPPLIER_050_rank2_flag",
        "go_b_residual_SUPPLIER_019_low_conf_flag",
        "go_b_residual_outer_terminal_flag",
    ]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0).astype(int)
    if "warning_reasons" in merged.columns:
        merged["warning_reasons"] = merged["warning_reasons"].fillna("").astype(str)
    return merged
