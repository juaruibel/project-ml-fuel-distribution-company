from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.ml.product.day06_runtime import FEEDBACK_ACTIONS, build_feedback_template, save_feedback
from src.ml.ui.state import get_visible_contextual_state, set_contextual_state


def render_feedback_editor(
    run_bundle: dict[str, Any],
    *,
    current_context_fingerprint: str | None,
) -> None:
    """Render the editable feedback editor and persist the Day 06 review file."""
    st.markdown("### Feedback humano y decisión final")
    feedback_df = get_visible_contextual_state(
        "day06_feedback_editor_df",
        current_context_fingerprint=current_context_fingerprint,
    )
    if not isinstance(feedback_df, pd.DataFrame):
        feedback_df = build_feedback_template(
            resumen_df=run_bundle["resumen_df"],
            run_id=run_bundle["manifest_payload"]["run_id"],
            inference_mode=run_bundle["mode_spec"].key,
        )

    edited_feedback_df = st.data_editor(
        feedback_df,
        hide_index=True,
        use_container_width=True,
        key="day06_feedback_editor",
        column_config={
            "feedback_action": st.column_config.SelectboxColumn(
                "feedback_action",
                options=FEEDBACK_ACTIONS,
                required=True,
            ),
        },
    )
    set_contextual_state(
        "day06_feedback_editor_df",
        edited_feedback_df,
        context_fingerprint=current_context_fingerprint,
    )

    if st.button("Guardar feedback operativo"):
        writable = edited_feedback_df.copy()
        reviewed_mask = writable["feedback_action"].astype(str).ne("pending_review")
        writable.loc[reviewed_mask, "reviewed_at_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        result = save_feedback(run_bundle=run_bundle, feedback_df=writable)
        set_contextual_state(
            "day06_feedback_editor_df",
            writable,
            context_fingerprint=current_context_fingerprint,
        )
        sql_display = result.get("sql_store_status", result.get("sql_mirror_status", ""))
        set_contextual_state(
            "day06_last_run",
            run_bundle,
            context_fingerprint=current_context_fingerprint,
        )
        set_contextual_state(
            "day06_post_action_notice",
            {
                "message": "Feedback guardado correctamente.",
                "severity": "success",
                "action_hint": "La decisión final y el estado SQL ya quedan sincronizados para seguimiento operativo.",
                "detail": f"SQL reporting: {sql_display}",
                "kv_pairs": [
                    ("feedback.csv", str(result["feedback_path"])),
                    ("run manifest", str(result["run_manifest_path"])),
                ],
            },
            context_fingerprint=current_context_fingerprint,
        )
        st.rerun()

    feedback_path = run_bundle["feedback_path"]
    if feedback_path.exists():
        st.download_button(
            label="Descargar feedback CSV",
            data=feedback_path.read_bytes(),
            file_name=feedback_path.name,
            mime="text/csv",
        )
