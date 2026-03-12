from __future__ import annotations

from typing import Any

import streamlit as st


PROGRESS_HISTORY_LIMIT = 4


def build_progress_state() -> dict[str, Any]:
    """Create one empty progress state for parse or run operations."""
    return {
        "status": "running",
        "current_step_key": "",
        "current_message": "",
        "recent_steps": [],
    }


def advance_progress_state(
    progress_state: dict[str, Any],
    *,
    step_key: str,
    user_message: str,
) -> dict[str, Any]:
    """Move one progress state to a new active phase and keep a short history of completed steps."""
    updated_state = dict(progress_state)
    recent_steps = list(updated_state.get("recent_steps", []))
    previous_step_key = str(updated_state.get("current_step_key", "")).strip()
    previous_message = str(updated_state.get("current_message", "")).strip()

    if previous_step_key and previous_step_key != step_key and previous_message:
        recent_steps.append({"step_key": previous_step_key, "message": previous_message})

    updated_state["status"] = "running"
    updated_state["current_step_key"] = step_key
    updated_state["current_message"] = user_message.strip()
    updated_state["recent_steps"] = recent_steps[-PROGRESS_HISTORY_LIMIT:]
    return updated_state


def finish_progress_state(
    progress_state: dict[str, Any],
    *,
    status: str,
    final_message: str,
    include_current_in_history: bool,
) -> dict[str, Any]:
    """Close one progress state in `success` or `error`, optionally persisting the last active phase."""
    updated_state = dict(progress_state)
    recent_steps = list(updated_state.get("recent_steps", []))
    if include_current_in_history:
        current_step_key = str(updated_state.get("current_step_key", "")).strip()
        current_message = str(updated_state.get("current_message", "")).strip()
        if current_step_key and current_message:
            recent_steps.append({"step_key": current_step_key, "message": current_message})

    updated_state["status"] = status
    updated_state["current_message"] = final_message.strip()
    updated_state["recent_steps"] = recent_steps[-PROGRESS_HISTORY_LIMIT:]
    return updated_state


def render_progress_block(progress_state: dict[str, Any]) -> None:
    """Render one sober progress block with the active phase plus a short completed history."""
    if not isinstance(progress_state, dict):
        return

    current_message = str(progress_state.get("current_message", "")).strip()
    recent_steps = [
        str(entry.get("message", "")).strip()
        for entry in progress_state.get("recent_steps", [])
        if isinstance(entry, dict) and str(entry.get("message", "")).strip() != ""
    ]
    if current_message == "" and not recent_steps:
        return

    status = str(progress_state.get("status", "running")).strip().lower()
    status_label = {
        "running": "En curso",
        "success": "Completado",
        "error": "Fallido",
    }.get(status, "Estado")

    with st.container(border=True):
        st.markdown("#### Estado actual")
        st.markdown(f"**{status_label}** · {current_message or 'Procesando.'}")
        if recent_steps:
            st.caption("Fases completadas")
            st.markdown("\n".join(f"- {message}" for message in recent_steps))


def render_progress_slot(progress_slot: Any, progress_state: dict[str, Any] | None) -> None:
    """Render or clear one dedicated placeholder for progress visibility."""
    if progress_slot is None:
        return
    progress_slot.empty()
    if not isinstance(progress_state, dict):
        return
    with progress_slot.container():
        render_progress_block(progress_state)


def set_excel_parse_progress_state(
    progress_state: dict[str, Any] | None,
    *,
    source_fingerprint: str,
) -> None:
    """Persist the visible Excel parse progress for the currently uploaded workbook."""
    st.session_state.day06_excel_parse_progress = progress_state
    st.session_state.day06_excel_parse_progress_source_fingerprint = source_fingerprint


def get_visible_excel_parse_progress(source_fingerprint: str) -> dict[str, Any] | None:
    """Return the Excel parse progress only for the currently visible workbook."""
    owner_fingerprint = str(st.session_state.get("day06_excel_parse_progress_source_fingerprint", ""))
    if owner_fingerprint != source_fingerprint:
        return None
    progress_state = st.session_state.get("day06_excel_parse_progress")
    return progress_state if isinstance(progress_state, dict) else None
