from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.ml.product.day06_excel_raw import (
    ExcelRawParseBundle,
    build_excel_candidate_grain,
    build_excel_enrichment_template,
    normalize_excel_enrichment_frame,
    parse_excel_raw_workbook,
    summarize_excel_enrichment_pending,
    validate_excel_enrichment,
)
from src.ml.product.day06_feedback import build_feedback_template, save_feedback
from src.ml.product.day06_runtime import inspect_input_mode_availability
from src.ml.shared.day06_demo_examples import get_day06_demo_workbooks
from src.ml.shared.project_paths import CHAMPION_METADATA_PATH
from src.ml.ui.display import normalize_display_dataframe, render_message_block
from src.ml.ui.product import manual_template, run_product_inference
from src.ml.ui.product_user_frames import (
    build_event_review_frame,
    build_input_plan_frame,
    build_purchase_proposal_frame,
    expand_input_plan_to_enrichment_frame,
    fan_out_proposal_feedback,
    infer_shared_albaran_id,
)
from src.ml.ui.progress import (
    advance_progress_state,
    build_progress_state,
    finish_progress_state,
    get_visible_excel_parse_progress,
    render_progress_slot,
    set_excel_parse_progress_state,
)
from src.ml.ui.state import (
    build_input_context_fingerprint,
    get_visible_contextual_state,
    initialize_product_state,
    set_contextual_state,
)


USER_ACTION_ACCEPT = "Aceptar recomendacion"
USER_ACTION_TOP2 = "Elegir alternativa 2"
USER_ACTION_OTHER = "Otro proveedor"
USER_ACTION_REVIEW = "Marcar para revision"

USER_WARNING_REASON_MAP = {
    "baseline_vs_champion_disagreement": "Comparar con la referencia historica.",
    "go_c_slice_review": "Caso PRODUCT_005 para revisar.",
    "go_b_dominant_review": "Caso PRODUCT_003 dominante para revisar.",
    "go_b_residual_SUPPLIER_050_rank2_review": "SUPPLIER_050 aparece como alternativa relevante.",
    "go_b_residual_SUPPLIER_019_low_conf_review": "Diferencia corta entre opciones.",
    "go_b_residual_outer_terminal_review": "Terminal con patron residual a revisar.",
}


def _is_excel_bundle(value: Any) -> bool:
    """Return whether one session object behaves like the parsed Excel bundle."""
    return isinstance(value, ExcelRawParseBundle) or (
        value is not None
        and hasattr(value, "parse_summary")
        and hasattr(value, "sheet_names")
        and hasattr(value, "source_name")
    )


@lru_cache(maxsize=1)
def _load_champion_user_metrics() -> dict[str, float]:
    """Load the official champion holdout metrics once for the user-facing header."""
    metadata_payload = json.loads(CHAMPION_METADATA_PATH.read_text(encoding="utf-8"))
    metrics = metadata_payload.get("metrics", {})
    return {
        "top1_hit": float(metrics.get("top1_hit", 0.0)),
        "top2_hit": float(metrics.get("top2_hit", 0.0)),
    }


def render_product_user_surface() -> None:
    """Render the redesigned daily user experience for the Day 06 product."""
    initialize_product_state(manual_template_factory=manual_template)

    header_candidate_df = _build_header_candidate_preview()
    header_input_payload = st.session_state.get("day06_input_payload")
    header_context_fingerprint = build_input_context_fingerprint(
        input_df=header_candidate_df,
        input_payload=header_input_payload if isinstance(header_input_payload, dict) else None,
    )
    header_last_run = get_visible_contextual_state(
        "day06_last_run",
        current_context_fingerprint=header_context_fingerprint,
    )
    header_feedback_df = get_visible_contextual_state(
        "day06_feedback_editor_df",
        current_context_fingerprint=header_context_fingerprint,
    )
    render_user_header(
        has_bundle=_is_excel_bundle(st.session_state.get("day06_excel_bundle")),
        candidate_ready=isinstance(header_candidate_df, pd.DataFrame),
        has_run=isinstance(header_last_run, dict),
        feedback_df=header_feedback_df if isinstance(header_feedback_df, pd.DataFrame) else None,
    )

    excel_bundle = render_user_upload_section()
    if not _is_excel_bundle(excel_bundle):
        excel_bundle = st.session_state.get("day06_excel_bundle")

    candidate_df: pd.DataFrame | None = None
    if _is_excel_bundle(excel_bundle):
        candidate_df = render_user_enrichment_section(excel_bundle)

    input_payload = st.session_state.get("day06_input_payload")
    current_context_fingerprint = build_input_context_fingerprint(
        input_df=candidate_df,
        input_payload=input_payload,
    )

    render_user_run_section(
        candidate_df=candidate_df,
        input_payload=input_payload if isinstance(input_payload, dict) else None,
        current_context_fingerprint=current_context_fingerprint,
    )
    render_user_post_action_notice(current_context_fingerprint=current_context_fingerprint)

    last_run = get_visible_contextual_state(
        "day06_last_run",
        current_context_fingerprint=current_context_fingerprint,
    )
    if isinstance(last_run, dict):
        render_user_review_section(
            run_bundle=last_run,
            current_context_fingerprint=current_context_fingerprint,
        )


def _build_header_candidate_preview() -> pd.DataFrame | None:
    """Best-effort candidate preview used only to compute the visible status rail at page load."""
    excel_bundle = st.session_state.get("day06_excel_bundle")
    if not _is_excel_bundle(excel_bundle):
        return None

    saved_enrichment_df = st.session_state.get("day06_excel_enrichment_saved_df")
    if not isinstance(saved_enrichment_df, pd.DataFrame):
        return None
    saved_enrichment_df = normalize_excel_enrichment_frame(saved_enrichment_df)
    if validate_excel_enrichment(saved_enrichment_df).get("status") != "PASS":
        return None

    try:
        return build_excel_candidate_grain(
            bundle=excel_bundle,
            enrichment_df=saved_enrichment_df,
        )
    except Exception:
        return None


def render_user_header(
    *,
    has_bundle: bool,
    candidate_ready: bool,
    has_run: bool,
    feedback_df: pd.DataFrame | None,
) -> None:
    """Render the user-facing header, official metrics and the four-step rail."""
    review_done = False
    if isinstance(feedback_df, pd.DataFrame) and not feedback_df.empty:
        review_done = bool(feedback_df["feedback_action"].astype(str).ne("pending_review").any())

    metrics = _load_champion_user_metrics()
    st.markdown(
        """
        <div class="user-header">
          <h2>Propuesta diaria de compra</h2>
          <p>Carga la comparativa, completa el pedido y cierra una propuesta de compra lista para revisar en vivo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="user-metric-rail">
          <div class="user-metric">
            <span>Top 1 historico del champion</span>
            <strong>{metrics["top1_hit"]:.2%}</strong>
          </div>
          <div class="user-metric">
            <span>Top 2 historico del champion</span>
            <strong>{metrics["top2_hit"]:.2%}</strong>
          </div>
        </div>
        <div class="user-footer-note">
          Estas metricas son del holdout oficial del champion a nivel evento. La propuesta diaria agrupa resultados solo cuando el caso es limpio.
        </div>
        """,
        unsafe_allow_html=True,
    )

    step_states = [
        ("1. Cargar comparativa", "Archivo listo." if has_bundle else "Sube o carga el ejemplo.", has_bundle, not has_bundle),
        (
            "2. Completar datos del pedido",
            "Pedido listo para calcular." if candidate_ready else "Rellena albaran y litros.",
            candidate_ready,
            has_bundle and not candidate_ready,
        ),
        (
            "3. Obtener recomendacion",
            "Propuesta calculada." if has_run else "Lanza la propuesta champion.",
            has_run,
            candidate_ready and not has_run,
        ),
        (
            "4. Revisar y guardar",
            "Decision cerrada." if review_done else "Revisa y guarda.",
            review_done,
            has_run and not review_done,
        ),
    ]
    step_blocks: list[str] = []
    for title, detail, is_done, is_active in step_states:
        css_class = "user-step"
        if is_done:
            css_class += " is-done"
        elif is_active:
            css_class += " is-active"
        step_blocks.append(f"<div class='{css_class}'><strong>{title}</strong><span>{detail}</span></div>")
    st.markdown(
        f"<div class='user-step-list'>{''.join(step_blocks)}</div>",
        unsafe_allow_html=True,
    )


def _prepare_excel_bundle_from_source(
    *,
    source_name: str,
    source_suffix: str,
    raw_bytes: bytes,
    progress_slot: Any,
) -> ExcelRawParseBundle | None:
    """Parse one selected workbook source and persist the shared user-flow session state."""
    source_fingerprint = hashlib.sha1(raw_bytes).hexdigest()
    render_progress_slot(progress_slot, get_visible_excel_parse_progress(source_fingerprint))

    if source_fingerprint != st.session_state.get("day06_excel_source_fingerprint", ""):
        progress_state = build_progress_state()
        set_excel_parse_progress_state(progress_state, source_fingerprint=source_fingerprint)

        def update_parse_progress(step_key: str, user_message: str) -> None:
            nonlocal progress_state
            progress_state = advance_progress_state(
                progress_state,
                step_key=step_key,
                user_message=user_message,
            )
            set_excel_parse_progress_state(progress_state, source_fingerprint=source_fingerprint)
            render_progress_slot(progress_slot, progress_state)

        try:
            excel_bundle = parse_excel_raw_workbook(
                source_name=source_name,
                source_suffix=source_suffix,
                source_bytes=raw_bytes,
                progress_callback=update_parse_progress,
            )
        except Exception as error:
            progress_state = finish_progress_state(
                progress_state,
                status="error",
                final_message=f"No se pudo preparar el archivo: {error}",
                include_current_in_history=False,
            )
            set_excel_parse_progress_state(progress_state, source_fingerprint=source_fingerprint)
            render_progress_slot(progress_slot, progress_state)
            st.session_state.day06_excel_bundle = None
            st.session_state.day06_excel_enrichment_df = None
            st.session_state.day06_excel_enrichment_saved_df = None
            st.session_state.day06_excel_source_fingerprint = ""
            st.session_state.day06_user_selected_proposal_id = None
            st.error(f"No se pudo leer la comparativa: {error}")
            return None

        enrichment_template = build_excel_enrichment_template(excel_bundle)
        progress_state = finish_progress_state(
            progress_state,
            status="success",
            final_message="Archivo preparado. Ya puedes completar el pedido.",
            include_current_in_history=True,
        )
        set_excel_parse_progress_state(progress_state, source_fingerprint=source_fingerprint)
        render_progress_slot(progress_slot, progress_state)
        st.session_state.day06_excel_bundle = excel_bundle
        st.session_state.day06_excel_enrichment_df = enrichment_template.copy()
        st.session_state.day06_excel_enrichment_saved_df = enrichment_template.copy()
        st.session_state.day06_excel_source_fingerprint = source_fingerprint
        st.session_state.day06_input_payload = {
            "input_mode": "excel_raw",
            "source_name": source_name,
            "source_suffix": source_suffix,
            "source_bytes": raw_bytes,
            "sheet_names": excel_bundle.sheet_names,
        }
        st.session_state.day06_user_selected_proposal_id = None
        st.success("Comparativa lista.")

    excel_bundle = st.session_state.get("day06_excel_bundle")
    if not _is_excel_bundle(excel_bundle):
        return None
    return excel_bundle


def render_user_upload_section() -> ExcelRawParseBundle | None:
    """Render the workbook upload step for the daily user flow."""
    with st.container(border=True):
        st.markdown("### 1. Cargar comparativa")
        st.caption("Usa el Excel raw del dia o abre uno de los dos casos ya preparados para demo.")
        demo_examples = get_day06_demo_workbooks()
        example_columns = st.columns([1, 1, 1.7])
        selected_example = None
        for column, demo_example in zip(example_columns[: len(demo_examples)], demo_examples):
            with column:
                example_clicked = st.button(
                    demo_example.button_label,
                    key=f"day06_user_{demo_example.key}",
                    use_container_width=True,
                    disabled=not demo_example.asset_path.exists(),
                )
                st.caption(demo_example.note)
                if example_clicked:
                    selected_example = demo_example
        with example_columns[-1]:
            uploaded = st.file_uploader(
                "Comparativa del dia",
                type=["xlsx", "xls"],
                key="day06_user_excel_uploader",
            )
        progress_slot = st.empty()
        if selected_example is not None:
            return _prepare_excel_bundle_from_source(
                source_name=selected_example.source_name,
                source_suffix=selected_example.asset_path.suffix,
                raw_bytes=selected_example.asset_path.read_bytes(),
                progress_slot=progress_slot,
            )

        if uploaded is None:
            existing_bundle = st.session_state.get("day06_excel_bundle")
            existing_source_fingerprint = str(st.session_state.get("day06_excel_source_fingerprint", ""))
            if _is_excel_bundle(existing_bundle):
                render_progress_slot(
                    progress_slot,
                    get_visible_excel_parse_progress(existing_source_fingerprint),
                )
                st.info("Ya hay una comparativa cargada. Puedes seguir con el pedido o sustituir el archivo.")
                summary = existing_bundle.parse_summary
                st.markdown(
                    f"""
                    <div class="user-footer-note">
                      Archivo: <strong>{existing_bundle.source_name}</strong> ·
                      Eventos detectados: <strong>{summary.get("base_events_total", 0)}</strong> ·
                      Hojas: <strong>{", ".join(existing_bundle.sheet_names)}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return existing_bundle

            render_progress_slot(progress_slot, None)
            st.info("Carga Ejemplo 1, Ejemplo 2 o sube el workbook para comenzar.")
            return None

        excel_bundle = _prepare_excel_bundle_from_source(
            source_name=uploaded.name,
            source_suffix=Path(uploaded.name).suffix,
            raw_bytes=uploaded.getvalue(),
            progress_slot=progress_slot,
        )
        if not _is_excel_bundle(excel_bundle):
            return None
        summary = excel_bundle.parse_summary
        st.markdown(
            f"""
            <div class="user-footer-note">
              Archivo: <strong>{excel_bundle.source_name}</strong> ·
              Eventos detectados: <strong>{summary.get("base_events_total", 0)}</strong> ·
              Hojas: <strong>{", ".join(excel_bundle.sheet_names)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return excel_bundle


def render_user_enrichment_section(excel_bundle: ExcelRawParseBundle) -> pd.DataFrame | None:
    """Render the user enrichment step at proposal-input grain and return one candidate frame when ready."""
    with st.container(border=True):
        st.markdown("### 2. Completar datos del pedido")
        st.caption(
            "El albaran se aplica a todo el pedido. Los litros se rellenan por producto o por terminal si hace falta separar."
        )

        current_enrichment_df = st.session_state.get("day06_excel_enrichment_df")
        if not isinstance(current_enrichment_df, pd.DataFrame):
            current_enrichment_df = build_excel_enrichment_template(excel_bundle)
        current_enrichment_df = normalize_excel_enrichment_frame(current_enrichment_df)
        input_plan_df = build_input_plan_frame(current_enrichment_df)
        shared_albaran_id = infer_shared_albaran_id(current_enrichment_df)

        pending_summary = summarize_excel_enrichment_pending(current_enrichment_df)
        split_rows = int(input_plan_df["input_grain"].astype(str).eq("product_terminal").sum()) if not input_plan_df.empty else 0
        st.markdown(
            f"""
            <div class="user-footer-note">
              Lineas visibles de pedido: <strong>{len(input_plan_df)}</strong> ·
              Productos: <strong>{int(current_enrichment_df["producto_canonico"].astype(str).nunique())}</strong> ·
              Filas desdobladas por terminal: <strong>{split_rows}</strong> ·
              Pendiente: <strong>{pending_summary["pending_litros_evento"]}</strong> litros.
            </div>
            """,
            unsafe_allow_html=True,
        )

        source_fingerprint = str(st.session_state.get("day06_excel_source_fingerprint", "")) or "default"
        albaran_key = f"day06_user_order_albaran_{source_fingerprint}"
        same_litros_key = f"day06_user_same_litros_{source_fingerprint}"
        same_litros_value_key = f"day06_user_same_litros_value_{source_fingerprint}"
        editor_key = f"day06_user_input_plan_editor_{source_fingerprint}"

        default_litros_value = _infer_shared_litros_value(input_plan_df)
        if albaran_key not in st.session_state:
            st.session_state[albaran_key] = shared_albaran_id
        if same_litros_key not in st.session_state:
            st.session_state[same_litros_key] = False
        if same_litros_value_key not in st.session_state:
            st.session_state[same_litros_value_key] = default_litros_value if default_litros_value is not None else 1000.0

        plan_display_df = input_plan_df.rename(
            columns={
                "fecha_evento": "fecha",
                "producto_canonico": "producto",
                "terminal_compra": "terminal",
                "litros_evento": "litros",
            }
        )[["fecha", "producto", "terminal", "litros"]].copy()

        order_left, order_right = st.columns([1.2, 0.9])
        with order_left:
            albaran_value = st.text_input(
                "Albaran del pedido",
                key=albaran_key,
                placeholder="ALB_001",
            )
        with order_right:
            same_litros_enabled = st.checkbox(
                "Usar los mismos litros para todo el pedido",
                key=same_litros_key,
            )

        common_litros = None
        if same_litros_enabled:
            common_litros = st.number_input(
                "Litros comunes",
                key=same_litros_value_key,
                min_value=1.0,
                step=1.0,
            )

        with st.form("day06_user_enrichment_form", clear_on_submit=False):
            if same_litros_enabled:
                st.dataframe(
                    normalize_display_dataframe(plan_display_df.drop(columns=["litros"]), all_columns=True),
                    hide_index=True,
                    use_container_width=True,
                )
                edited_plan_display = plan_display_df.copy()
                edited_plan_display["litros"] = float(common_litros) if common_litros is not None else pd.NA
            else:
                edited_plan_display = st.data_editor(
                    plan_display_df,
                    hide_index=True,
                    use_container_width=True,
                    key=editor_key,
                    disabled=["fecha", "producto", "terminal"],
                    column_config={
                        "litros": st.column_config.NumberColumn(
                            "litros",
                            min_value=1.0,
                            step=1.0,
                            required=True,
                        ),
                    },
                )

            enrichment_submitted = st.form_submit_button(
                "Guardar datos del pedido",
                type="primary",
            )

        if enrichment_submitted:
            if str(albaran_value).strip() == "":
                st.error("Escribe un albaran antes de guardar el pedido.")
            else:
                committed_plan_df = input_plan_df.copy()
                committed_plan_df["litros_evento"] = edited_plan_display["litros"]
                committed_enrichment_df = expand_input_plan_to_enrichment_frame(
                    base_enrichment_df=current_enrichment_df,
                    input_plan_df=committed_plan_df,
                    albaran_id=str(albaran_value),
                )
                enrichment_status = validate_excel_enrichment(committed_enrichment_df)
                if enrichment_status["status"] != "PASS":
                    st.warning(str(enrichment_status["message"]))
                else:
                    st.session_state.day06_excel_enrichment_df = committed_enrichment_df.copy()
                    st.session_state.day06_excel_enrichment_saved_df = committed_enrichment_df.copy()
                    st.success("Datos del pedido guardados.")

        committed_enrichment_df = st.session_state.get("day06_excel_enrichment_saved_df")
        if not isinstance(committed_enrichment_df, pd.DataFrame):
            committed_enrichment_df = current_enrichment_df.copy()
        committed_enrichment_df = normalize_excel_enrichment_frame(committed_enrichment_df)
        enrichment_status = validate_excel_enrichment(committed_enrichment_df)
        if enrichment_status["status"] != "PASS":
            st.info("Completa albaran y litros para poder calcular la recomendacion.")
            return None

        try:
            candidate_df = build_excel_candidate_grain(
                bundle=excel_bundle,
                enrichment_df=committed_enrichment_df,
            )
        except Exception as error:
            st.warning(f"Todavia no se puede construir la recomendacion: {error}")
            return None

        committed_plan_df = build_input_plan_frame(committed_enrichment_df)
        st.markdown(
            f"""
            <div class="user-footer-note">
              Pedido listo: <strong>{len(committed_plan_df)}</strong> linea(s) visibles de pedido ·
              <strong>{candidate_df["event_id"].astype(str).nunique()}</strong> evento(s) tecnicos.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return candidate_df


def render_user_run_section(
    *,
    candidate_df: pd.DataFrame | None,
    input_payload: dict[str, Any] | None,
    current_context_fingerprint: str | None,
) -> None:
    """Render the champion-only execution step for the user surface."""
    with st.container(border=True):
        st.markdown("### 3. Obtener recomendacion")
        st.caption("Esta superficie trabaja con el champion y ensena solo la propuesta necesaria para decidir.")
        run_progress_slot = st.empty()
        render_progress_slot(
            run_progress_slot,
            get_visible_contextual_state(
                "day06_run_progress",
                current_context_fingerprint=current_context_fingerprint,
            ),
        )

        if input_payload is None or candidate_df is None:
            st.info("Primero prepara la comparativa y completa los datos del pedido.")
            return

        mode_availability = inspect_input_mode_availability(
            input_df=candidate_df,
            input_mode=str(input_payload.get("input_mode", "")),
        )
        champion_entry = next(
            (
                entry
                for entry in mode_availability.get("mode_catalog", [])
                if str(entry.get("mode_key", "")) == "champion_pure"
            ),
            None,
        )
        if not isinstance(champion_entry, dict) or not bool(champion_entry.get("enabled")):
            user_reason = "No se puede calcular la recomendacion champion con el archivo actual."
            if isinstance(champion_entry, dict):
                user_reason = str(champion_entry.get("user_reason", user_reason))
            render_message_block(
                message=user_reason,
                severity=str(champion_entry.get("severity", "warning")) if isinstance(champion_entry, dict) else "warning",
                action_hint="Corrige el archivo o usa Producto-Dev si necesitas diagnostico tecnico.",
            )
            return

        st.markdown(
            "<div class='user-footer-note'>Modo fijo: <strong>champion</strong> · Alternativa visible: <strong>top 2</strong></div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "Calcular recomendacion",
            type="primary",
            disabled=candidate_df is None,
        ):
            run_product_inference(
                input_df=candidate_df,
                input_payload=input_payload,
                inference_mode="champion_pure",
                top_k=2,
                context_fingerprint=current_context_fingerprint,
                progress_slot=run_progress_slot,
            )


def render_user_post_action_notice(*, current_context_fingerprint: str | None) -> None:
    """Render one short user-facing post action notice and then clear it."""
    post_action_notice = get_visible_contextual_state(
        "day06_post_action_notice",
        current_context_fingerprint=current_context_fingerprint,
    )
    if not isinstance(post_action_notice, dict):
        return

    render_message_block(
        message=str(post_action_notice.get("message", "")).strip(),
        severity=str(post_action_notice.get("severity", "info")),
        action_hint=str(post_action_notice.get("action_hint", "")),
    )
    set_contextual_state(
        "day06_post_action_notice",
        None,
        context_fingerprint=current_context_fingerprint,
    )


def render_user_review_section(
    *,
    run_bundle: dict[str, Any],
    current_context_fingerprint: str | None,
) -> None:
    """Render the user-facing purchase proposal plus guided feedback actions."""
    with st.container(border=True):
        st.markdown("### 4. Revisar y guardar")

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
            set_contextual_state(
                "day06_feedback_editor_df",
                feedback_df.copy(),
                context_fingerprint=current_context_fingerprint,
            )

        event_review_df = build_event_review_frame(
            detail_df=run_bundle["detail_df"],
            resumen_df=run_bundle["resumen_df"],
            feedback_df=feedback_df,
        )
        if event_review_df.empty:
            st.info("No hay propuestas disponibles para revisar.")
            return

        event_review_df["motivo_revision"] = event_review_df["motivo_revision"].map(_humanize_warning_reasons)
        proposal_df = build_purchase_proposal_frame(event_review_df)
        if proposal_df.empty:
            st.info("No hay propuestas disponibles para revisar.")
            return

        review_rows = int(proposal_df["confianza"].astype(str).eq("Revisar").sum())
        if review_rows > 0:
            st.markdown(
                f"<div class='user-review-note'>{review_rows} propuesta(s) necesitan una revision mas atenta antes de cerrar la decision.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='user-footer-note'>La propuesta ya esta lista para validarse y guardarse.</div>",
                unsafe_allow_html=True,
            )

        reviewed_rows = int(proposal_df["feedback_action"].astype(str).ne("pending_review").sum())
        split_rows = int(proposal_df["proposal_grain"].astype(str).ne("product").sum())
        summary_cards = [
            ("Pedido", _summarize_proposal_value(proposal_df["albaran"], empty_label="Sin albaran")),
            ("Lineas visibles", str(len(proposal_df))),
            (
                "Proveedor propuesto",
                _summarize_proposal_value(
                    proposal_df["proveedor_recomendado"],
                    empty_label="Sin propuesta",
                    mixed_label="Segun producto",
                ),
            ),
            (
                "Alternativa",
                _summarize_proposal_value(
                    proposal_df["alternativa"],
                    empty_label="Sin alternativa",
                    mixed_label="Segun producto",
                ),
            ),
            ("Casos a revisar", str(review_rows)),
        ]
        st.markdown(
            "<div class='user-summary-grid'>"
            + "".join(
                f"<div class='user-summary-card'><span>{label}</span><strong>{value}</strong></div>"
                for label, value in summary_cards
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="user-footer-note">
              Propuestas: <strong>{len(proposal_df)}</strong> ·
              Revisar: <strong>{review_rows}</strong> ·
              Desdobladas: <strong>{split_rows}</strong> ·
              Decisiones preparadas: <strong>{reviewed_rows}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        proposal_display = normalize_display_dataframe(
            proposal_df[
                [
                    "fecha",
                    "albaran",
                    "producto",
                    "litros",
                    "proveedor_recomendado",
                    "alternativa",
                    "confianza",
                    "nota",
                    "decision_final",
                ]
            ].rename(
                columns={
                    "proveedor_recomendado": "proveedor recomendado",
                    "decision_final": "decision final",
                }
            ),
            all_columns=True,
        )
        st.dataframe(proposal_display, hide_index=True, use_container_width=True)

        proposal_options = proposal_df["proposal_id"].astype(str).tolist()
        if st.session_state.get("day06_user_selected_proposal_id") not in proposal_options:
            st.session_state.day06_user_selected_proposal_id = proposal_options[0]
        option_labels = {
            row["proposal_id"]: _build_proposal_option_label(row)
            for _, row in proposal_df.iterrows()
        }
        selected_proposal_id = st.selectbox(
            "Propuesta a revisar",
            options=proposal_options,
            key="day06_user_selected_proposal_id",
            format_func=lambda proposal_id: option_labels.get(proposal_id, proposal_id),
        )
        selected_row = proposal_df[proposal_df["proposal_id"].astype(str) == str(selected_proposal_id)].iloc[0]

        choice_left, choice_right = st.columns(2)
        with choice_left:
            st.markdown(
                _build_choice_card_html(
                    title="Proveedor recomendado",
                    supplier=str(selected_row["proveedor_recomendado"]),
                    meta_pairs=[
                        ("Confianza", str(selected_row["confianza"])),
                        ("Producto", str(selected_row["producto"])),
                        ("Agrupacion", _build_proposal_grain_label(str(selected_row["proposal_grain"]))),
                    ],
                ),
                unsafe_allow_html=True,
            )
        with choice_right:
            alternative_label = str(selected_row["alternativa"]).strip() or "Sin alternativa visible"
            st.markdown(
                _build_choice_card_html(
                    title="Alternativa visible",
                    supplier=alternative_label,
                    meta_pairs=[
                        ("Albaran", str(selected_row["albaran"])),
                        ("Litros", _format_liters_label(selected_row["litros"])),
                        ("Nota", str(selected_row["nota"]) or "Sin nota adicional."),
                    ],
                ),
                unsafe_allow_html=True,
            )

        action_options = [USER_ACTION_ACCEPT, USER_ACTION_OTHER, USER_ACTION_REVIEW]
        if str(selected_row["alternativa"]).strip():
            action_options.insert(1, USER_ACTION_TOP2)
        default_action = _infer_user_action_label(
            feedback_action=str(selected_row.get("feedback_action", "")),
            decision_final=str(selected_row.get("decision_final", "")),
            recommended_supplier=str(selected_row["proveedor_recomendado"]),
            top2_supplier=str(selected_row["alternativa"]),
            confidence_label=str(selected_row["confianza"]),
        )
        if default_action not in action_options:
            default_action = USER_ACTION_ACCEPT

        action_label = st.radio(
            "Decision para esta propuesta",
            options=action_options,
            index=action_options.index(default_action),
            horizontal=True,
            key=f"day06_user_action_{selected_proposal_id}",
        )

        manual_supplier = ""
        if action_label == USER_ACTION_OTHER:
            manual_supplier = st.text_input(
                "Proveedor final",
                value=_build_manual_supplier_default(
                    decision_final=str(selected_row.get("decision_final", "")),
                    top1_supplier=str(selected_row["proveedor_recomendado"]),
                    top2_supplier=str(selected_row["alternativa"]),
                ),
                key=f"day06_user_manual_supplier_{selected_proposal_id}",
                placeholder="Escribe el proveedor final",
            )

        requires_reason = action_label in (USER_ACTION_TOP2, USER_ACTION_OTHER, USER_ACTION_REVIEW)
        override_reason = ""
        feedback_notes = ""
        if requires_reason:
            override_reason = st.text_input(
                "Motivo",
                value=str(selected_row.get("override_reason", "")),
                key=f"day06_user_override_reason_{selected_proposal_id}",
                placeholder="Explica por que ajustas la decision",
            )
            feedback_notes = st.text_area(
                "Notas",
                value=str(selected_row.get("feedback_notes", "")),
                key=f"day06_user_feedback_notes_{selected_proposal_id}",
                placeholder="Contexto operativo opcional",
            )

        apply_left, save_right = st.columns([1, 1])
        with apply_left:
            if st.button("Aplicar decision a la propuesta", key=f"day06_user_apply_{selected_proposal_id}"):
                decision_final, feedback_action = _resolve_feedback_payload(
                    action_label=action_label,
                    recommended_supplier=str(selected_row["proveedor_recomendado"]),
                    top2_supplier=str(selected_row["alternativa"]),
                    manual_supplier=manual_supplier,
                    current_decision=str(selected_row.get("decision_final", "")),
                )
                if action_label == USER_ACTION_OTHER and decision_final.strip() == "":
                    st.error("Escribe el proveedor final antes de aplicar la decision.")
                else:
                    updated_feedback_df = fan_out_proposal_feedback(
                        feedback_df=feedback_df,
                        proposal_row=selected_row,
                        decision_final=decision_final,
                        feedback_action=feedback_action,
                        override_reason=override_reason if requires_reason else "",
                        feedback_notes=feedback_notes if requires_reason else "",
                    )
                    set_contextual_state(
                        "day06_feedback_editor_df",
                        updated_feedback_df,
                        context_fingerprint=current_context_fingerprint,
                    )
                    st.session_state.day06_user_feedback_draft_notice = (
                        f"Decision preparada para {option_labels[str(selected_proposal_id)]}."
                    )
                    st.rerun()

        with save_right:
            if st.button("Guardar revision", type="primary", key="day06_user_save_feedback"):
                writable = feedback_df.copy()
                reviewed_mask = writable["feedback_action"].astype(str).ne("pending_review")
                writable.loc[reviewed_mask, "reviewed_at_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                result = save_feedback(run_bundle=run_bundle, feedback_df=writable)
                set_contextual_state(
                    "day06_feedback_editor_df",
                    writable,
                    context_fingerprint=current_context_fingerprint,
                )
                set_contextual_state(
                    "day06_last_run",
                    run_bundle,
                    context_fingerprint=current_context_fingerprint,
                )
                set_contextual_state(
                    "day06_post_action_notice",
                    {
                        "message": "Revision guardada.",
                        "severity": "success",
                        "action_hint": "La propuesta final ya queda cerrada para la demo y el seguimiento operativo.",
                        "detail": f"feedback.csv: {result['feedback_path']}",
                    },
                    context_fingerprint=current_context_fingerprint,
                )
                st.rerun()

        draft_notice = st.session_state.get("day06_user_feedback_draft_notice")
        if isinstance(draft_notice, str) and draft_notice.strip():
            st.info(draft_notice)
            st.session_state.day06_user_feedback_draft_notice = None


def _infer_shared_litros_value(input_plan_df: pd.DataFrame) -> float | None:
    """Return one shared litros value when the visible plan already carries one."""
    if input_plan_df.empty or "litros_evento" not in input_plan_df.columns:
        return None
    litros_values = pd.to_numeric(input_plan_df["litros_evento"], errors="coerce").dropna()
    if litros_values.empty:
        return None
    unique_values = list(dict.fromkeys(float(value) for value in litros_values.tolist()))
    if len(unique_values) == 1:
        return unique_values[0]
    return None


def _humanize_warning_reasons(raw_reason_text: Any) -> str:
    """Convert one technical warning label set into concise user copy."""
    if raw_reason_text is None or pd.isna(raw_reason_text):
        return ""
    labels = [label.strip() for label in str(raw_reason_text).split("|") if label.strip()]
    messages = [USER_WARNING_REASON_MAP.get(label, label.replace("_", " ")) for label in labels]
    return " · ".join(messages)


def _summarize_proposal_value(
    values: pd.Series,
    *,
    empty_label: str,
    mixed_label: str = "Mixto",
) -> str:
    """Return one compact summary label from one proposal column."""
    unique_values = [
        value
        for value in dict.fromkeys(str(raw_value).strip() for raw_value in values.tolist())
        if value and value.lower() != "nan"
    ]
    if not unique_values:
        return empty_label
    if len(unique_values) == 1:
        return unique_values[0]
    return mixed_label


def _build_proposal_option_label(row: pd.Series) -> str:
    """Build one compact, human-readable label for one proposal row."""
    albaran_id = str(row.get("albaran", "")).strip() or "Sin albaran"
    producto = str(row.get("producto", "")).strip() or "Sin producto"
    return f"{albaran_id} · {producto}"


def _build_proposal_grain_label(proposal_grain: str) -> str:
    """Translate one internal proposal grain into short user-facing copy."""
    if proposal_grain == "product":
        return "Por producto"
    if proposal_grain == "product_terminal":
        return "Producto desdoblado por terminal"
    return "Linea separada"


def _build_choice_card_html(
    *,
    title: str,
    supplier: str,
    meta_pairs: list[tuple[str, str]],
) -> str:
    """Render one simple comparison block for one selected proposal."""
    meta_html = "".join(f"<p><strong>{label}:</strong> {value}</p>" for label, value in meta_pairs if value.strip())
    return (
        "<div class='user-choice-card'>"
        f"<h4>{title}</h4>"
        f"<p><strong>{supplier}</strong></p>"
        f"{meta_html}"
        "</div>"
    )


def _format_liters_label(value: Any) -> str:
    """Format one litros value for the user-facing cards."""
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return ""
    return f"{float(numeric_value):,.1f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _infer_user_action_label(
    *,
    feedback_action: str,
    decision_final: str,
    recommended_supplier: str,
    top2_supplier: str,
    confidence_label: str,
) -> str:
    """Translate one persisted feedback row into one user-facing review action."""
    if feedback_action == "accepted":
        return USER_ACTION_ACCEPT
    if feedback_action == "overridden" and decision_final.strip() == top2_supplier.strip() and top2_supplier.strip():
        return USER_ACTION_TOP2
    if feedback_action in ("overridden", "rejected") and decision_final.strip():
        return USER_ACTION_OTHER
    if feedback_action == "pending_review" and confidence_label == "Revisar":
        return USER_ACTION_REVIEW
    if decision_final.strip() == recommended_supplier.strip():
        return USER_ACTION_ACCEPT
    return USER_ACTION_REVIEW


def _build_manual_supplier_default(
    *,
    decision_final: str,
    top1_supplier: str,
    top2_supplier: str,
) -> str:
    """Pre-fill the manual supplier field only when the saved decision is not top1/top2."""
    if decision_final.strip() in ("", top1_supplier.strip(), top2_supplier.strip()):
        return ""
    return decision_final.strip()


def _resolve_feedback_payload(
    *,
    action_label: str,
    recommended_supplier: str,
    top2_supplier: str,
    manual_supplier: str,
    current_decision: str,
) -> tuple[str, str]:
    """Map one user-facing action into the persisted Day 06 feedback contract."""
    if action_label == USER_ACTION_ACCEPT:
        return recommended_supplier, "accepted"
    if action_label == USER_ACTION_TOP2:
        return top2_supplier, "overridden"
    if action_label == USER_ACTION_OTHER:
        return manual_supplier.strip(), "overridden"
    decision_final = current_decision.strip() or recommended_supplier
    return decision_final, "pending_review"
