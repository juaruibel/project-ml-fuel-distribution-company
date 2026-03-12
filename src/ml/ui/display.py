from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st


def format_display_value(value: Any) -> str:
    """Convert one mixed scalar/list/dict into one stable text value for presentation tables."""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(format_display_value(item) for item in value if format_display_value(item) != "")
    try:
        if value is None or pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def build_display_summary_dataframe(
    rows: list[dict[str, Any]],
    *,
    value_columns: list[str] | tuple[str, ...] = ("valor",),
) -> pd.DataFrame:
    """Build one summary dataframe whose display columns are explicitly coerced to string."""
    display_df = pd.DataFrame(rows)
    for column in value_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(format_display_value).astype("string")
    return display_df


def normalize_display_dataframe(
    dataframe: pd.DataFrame,
    *,
    stringify_columns: list[str] | tuple[str, ...] | None = None,
    all_columns: bool = False,
) -> pd.DataFrame:
    """Convert one display-only dataframe to stable string columns without mutating operational data."""
    display_df = dataframe.copy()
    target_columns = list(display_df.columns) if all_columns else list(stringify_columns or [])
    for column in target_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(format_display_value).astype("string")
    return display_df


def render_message_block(
    *,
    message: str,
    severity: str,
    action_hint: str = "",
) -> None:
    """Render one short user-facing status block with an optional next-step hint."""
    normalized_severity = severity.lower().strip()
    display_message = message.strip()
    if action_hint.strip():
        display_message = f"{display_message} {action_hint.strip()}"

    if normalized_severity == "success":
        st.success(display_message)
    elif normalized_severity == "warning":
        st.warning(display_message)
    elif normalized_severity == "error":
        st.error(display_message)
    else:
        st.info(display_message)


def render_technical_detail_expander(
    *,
    title: str = "Detalle técnico",
    detail: str = "",
    kv_pairs: list[tuple[str, Any]] | None = None,
    tables: list[tuple[str, pd.DataFrame]] | None = None,
) -> None:
    """Render one compact expander with technical context for support/debugging."""
    has_detail = detail.strip() != ""
    has_kv_pairs = bool(kv_pairs)
    has_tables = bool(tables)
    if not has_detail and not has_kv_pairs and not has_tables:
        return

    with st.expander(title):
        if has_detail:
            st.code(detail.strip(), language="text")
        for label, value in kv_pairs or []:
            if value in (None, "", [], {}):
                continue
            st.markdown(f"**{label}:** `{value}`")
        for table_title, table_df in tables or []:
            if table_df.empty:
                continue
            st.markdown(f"**{table_title}**")
            st.dataframe(
                normalize_display_dataframe(table_df, all_columns=True),
                hide_index=True,
                use_container_width=True,
            )
