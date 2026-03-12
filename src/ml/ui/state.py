from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st


CONTEXTUAL_STATE_KEYS = [
    "day06_last_run",
    "day06_run_progress",
    "day06_last_validation",
    "day06_last_validation_path",
    "day06_last_failed_contract",
    "day06_feedback_editor_df",
    "day06_post_action_notice",
]


def initialize_product_state(*, manual_template_factory: Callable[[], pd.DataFrame]) -> None:
    """Initialize the Streamlit session state for the product surface."""
    if "day06_manual_df" not in st.session_state:
        st.session_state.day06_manual_df = manual_template_factory()
    if "day06_uploaded_df" not in st.session_state:
        st.session_state.day06_uploaded_df = None
    if "day06_input_payload" not in st.session_state:
        st.session_state.day06_input_payload = None
    if "day06_last_validation" not in st.session_state:
        st.session_state.day06_last_validation = None
    if "day06_last_validation_path" not in st.session_state:
        st.session_state.day06_last_validation_path = None
    if "day06_last_run" not in st.session_state:
        st.session_state.day06_last_run = None
    if "day06_last_failed_contract" not in st.session_state:
        st.session_state.day06_last_failed_contract = None
    if "day06_feedback_editor_df" not in st.session_state:
        st.session_state.day06_feedback_editor_df = None
    if "day06_post_action_notice" not in st.session_state:
        st.session_state.day06_post_action_notice = None
    if "day06_user_selected_event_id" not in st.session_state:
        st.session_state.day06_user_selected_event_id = None
    if "day06_user_selected_proposal_id" not in st.session_state:
        st.session_state.day06_user_selected_proposal_id = None
    if "day06_user_feedback_draft_notice" not in st.session_state:
        st.session_state.day06_user_feedback_draft_notice = None
    if "day06_excel_bundle" not in st.session_state:
        st.session_state.day06_excel_bundle = None
    if "day06_excel_enrichment_df" not in st.session_state:
        st.session_state.day06_excel_enrichment_df = None
    if "day06_excel_enrichment_saved_df" not in st.session_state:
        st.session_state.day06_excel_enrichment_saved_df = None
    if "day06_excel_source_fingerprint" not in st.session_state:
        st.session_state.day06_excel_source_fingerprint = ""
    if "day06_excel_parse_progress" not in st.session_state:
        st.session_state.day06_excel_parse_progress = None
    if "day06_excel_parse_progress_source_fingerprint" not in st.session_state:
        st.session_state.day06_excel_parse_progress_source_fingerprint = ""
    if "day06_mode_availability" not in st.session_state:
        st.session_state.day06_mode_availability = None
    if "day06_mode_selector" not in st.session_state:
        st.session_state.day06_mode_selector = None
    if "day06_last_run_context_fingerprint" not in st.session_state:
        st.session_state.day06_last_run_context_fingerprint = None
    if "day06_last_validation_context_fingerprint" not in st.session_state:
        st.session_state.day06_last_validation_context_fingerprint = None
    if "day06_last_validation_path_context_fingerprint" not in st.session_state:
        st.session_state.day06_last_validation_path_context_fingerprint = None
    if "day06_last_failed_contract_context_fingerprint" not in st.session_state:
        st.session_state.day06_last_failed_contract_context_fingerprint = None
    if "day06_feedback_editor_context_fingerprint" not in st.session_state:
        st.session_state.day06_feedback_editor_context_fingerprint = None
    if "day06_post_action_notice_context_fingerprint" not in st.session_state:
        st.session_state.day06_post_action_notice_context_fingerprint = None


def canonicalize_input_payload(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Convert one session payload into a JSON-serializable structure for fingerprinting."""
    serialized: dict[str, Any] = {}
    for key, value in input_payload.items():
        if key == "source_bytes":
            serialized["source_bytes_sha1"] = hashlib.sha1(bytes(value)).hexdigest()
            continue
        if isinstance(value, Path):
            serialized[key] = str(value)
            continue
        serialized[key] = value
    return serialized


def _serialize_scalar(value: Any) -> str:
    """Convert one scalar/list/dict into a fingerprint-safe string representation."""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_serialize_scalar(item) for item in value if _serialize_scalar(item) != "")
    try:
        if value is None or pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def serialize_input_dataframe(input_df: pd.DataFrame | None) -> str:
    """Serialize the current input dataframe into a stable CSV string for context tracking."""
    if input_df is None:
        return ""
    normalized_df = input_df.copy()
    for column in normalized_df.columns:
        normalized_df[column] = normalized_df[column].map(_serialize_scalar).astype("string")
    return normalized_df.to_csv(index=False)


def build_input_context_fingerprint(
    *,
    input_df: pd.DataFrame | None,
    input_payload: dict[str, Any] | None,
) -> str | None:
    """Build one stable fingerprint for the currently visible product input context."""
    if input_payload is None:
        return None

    payload_serialized = canonicalize_input_payload(input_payload)
    dataframe_csv = serialize_input_dataframe(input_df=input_df)
    fingerprint_payload = {
        "payload": payload_serialized,
        "dataframe_csv": dataframe_csv,
    }
    serialized_payload = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha1(serialized_payload.encode("utf-8")).hexdigest()


def set_contextual_state(
    key: str,
    value: Any,
    *,
    context_fingerprint: str | None,
) -> None:
    """Persist one contextual UI state together with the fingerprint that owns its visibility."""
    st.session_state[key] = value
    st.session_state[f"{key}_context_fingerprint"] = context_fingerprint


def get_visible_contextual_state(
    key: str,
    *,
    current_context_fingerprint: str | None,
) -> Any:
    """Return one contextual state only when it belongs to the currently visible input context."""
    if current_context_fingerprint is None:
        return None
    owner_fingerprint = st.session_state.get(f"{key}_context_fingerprint")
    if owner_fingerprint != current_context_fingerprint:
        return None
    return st.session_state.get(key)


def clear_contextual_visibility() -> None:
    """Hide contextual panels when the current input stops being evaluable."""
    for key in CONTEXTUAL_STATE_KEYS:
        st.session_state[f"{key}_context_fingerprint"] = None
