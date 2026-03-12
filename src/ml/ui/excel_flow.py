from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from src.ml.product.day06_excel_raw import (
    ExcelRawParseBundle,
    apply_linea_sequence_to_blank_rows,
    apply_value_to_blank_enrichment_rows,
    build_excel_candidate_grain,
    build_excel_enrichment_template,
    normalize_excel_enrichment_frame,
    parse_excel_raw_workbook,
    summarize_excel_enrichment_pending,
    validate_excel_enrichment,
)
from src.ml.product.day06_runtime import get_mode_specs, inspect_input_mode_availability
from src.ml.ui.display import build_display_summary_dataframe
from src.ml.ui.progress import (
    advance_progress_state,
    build_progress_state,
    finish_progress_state,
    get_visible_excel_parse_progress,
    render_progress_slot,
    set_excel_parse_progress_state,
)
from src.ml.ui.state import get_visible_contextual_state


def apply_editor_state_patch(
    *,
    base_df: pd.DataFrame,
    editor_state: Any,
) -> pd.DataFrame:
    """Apply the serialized `st.data_editor` patch state over the canonical enrichment dataframe."""
    if not isinstance(editor_state, dict):
        return base_df.copy()

    edited_rows = editor_state.get("edited_rows", {})
    if not isinstance(edited_rows, dict):
        return base_df.copy()

    working = base_df.copy()
    for row_position, edited_columns in edited_rows.items():
        try:
            row_index = int(row_position)
        except (TypeError, ValueError):
            continue
        if row_index < 0 or row_index >= len(working) or not isinstance(edited_columns, dict):
            continue
        for column_name, value in edited_columns.items():
            if column_name not in working.columns:
                continue
            if column_name == "litros_evento":
                working.at[row_index, column_name] = pd.NA if value in (None, "") else value
            else:
                working.at[row_index, column_name] = "" if value is None else value
    return normalize_excel_enrichment_frame(working)


def build_excel_raw_candidate_context(excel_bundle: ExcelRawParseBundle) -> dict[str, Any]:
    """Build the current enrichment and candidate-grain context for the guided Excel flow."""
    editable_df = hydrate_excel_enrichment_dataframe(excel_bundle)
    current_enrichment_df = st.session_state.get("day06_excel_enrichment_df")
    if not isinstance(current_enrichment_df, pd.DataFrame):
        current_enrichment_df = editable_df.copy()
    current_enrichment_df = normalize_excel_enrichment_frame(current_enrichment_df)
    committed_enrichment_df = st.session_state.get("day06_excel_enrichment_saved_df")
    if not isinstance(committed_enrichment_df, pd.DataFrame):
        committed_enrichment_df = build_excel_enrichment_template(excel_bundle)
    committed_enrichment_df = normalize_excel_enrichment_frame(committed_enrichment_df)

    enrichment_status = validate_excel_enrichment(committed_enrichment_df)
    candidate_df: pd.DataFrame | None = None
    candidate_error: str | None = None
    if enrichment_status["status"] == "PASS":
        try:
            candidate_df = build_excel_candidate_grain(
                bundle=excel_bundle,
                enrichment_df=committed_enrichment_df,
            )
        except Exception as error:
            candidate_error = str(error)

    return {
        "editable_df": editable_df,
        "current_enrichment_df": current_enrichment_df,
        "committed_enrichment_df": committed_enrichment_df,
        "enrichment_status": enrichment_status,
        "candidate_df": candidate_df,
        "candidate_error": candidate_error,
    }


def build_excel_raw_step_states(
    *,
    excel_bundle: ExcelRawParseBundle | None,
    candidate_ready: bool,
    mode_availability: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Describe the current five-step state for the guided Excel flow."""
    has_bundle = isinstance(excel_bundle, ExcelRawParseBundle)
    validation_ready = candidate_ready
    execution_ready = bool(mode_availability and mode_availability.get("selected_default_mode"))
    return [
        {"title": "Paso 1 · Sube archivo", "status": "completado" if has_bundle else "actual"},
        {"title": "Paso 2 · Revisa parseo", "status": "completado" if has_bundle else "pendiente"},
        {
            "title": "Paso 3 · Completa enrichment",
            "status": "completado" if candidate_ready else ("actual" if has_bundle else "pendiente"),
        },
        {
            "title": "Paso 4 · Valida contrato",
            "status": "actual" if validation_ready else "pendiente",
        },
        {
            "title": "Paso 5 · Ejecuta",
            "status": "actual" if execution_ready else "pendiente",
        },
    ]


def render_excel_raw_step_summary(step_states: list[dict[str, str]]) -> None:
    """Render one simple vertical summary of the guided Excel flow."""
    status_labels = {
        "actual": "Actual",
        "completado": "Completado",
        "pendiente": "Pendiente",
    }
    st.markdown("### Flujo Excel raw")
    st.caption("Sigue los pasos en orden. Los pasos posteriores se activan cuando el anterior deja el input listo.")
    summary_lines = [
        f"{position}. {step['title'].split('·', 1)[1].strip()} - {status_labels.get(step['status'], step['status'])}"
        for position, step in enumerate(step_states, start=1)
    ]
    st.markdown("\n".join(f"- {line}" for line in summary_lines))


def render_excel_raw_blocked_step(*, title: str, message: str) -> None:
    """Render one visible-but-blocked step in the guided Excel flow."""
    st.markdown(f"### {title}")
    st.caption("Pendiente")
    st.info(message)


def render_excel_raw_upload_step() -> None:
    """Render the upload step and update session state when a workbook is parsed successfully."""
    st.markdown("### Paso 1 · Sube archivo")
    upload_completed = isinstance(st.session_state.get("day06_excel_bundle"), ExcelRawParseBundle)
    st.caption("Completado" if upload_completed else "Actual")
    uploaded = st.file_uploader("Sube Excel raw", type=["xlsx", "xls"], key="day06_excel_uploader")
    progress_slot = st.empty()
    if uploaded is None:
        render_progress_slot(progress_slot, None)
        st.info("Sube un workbook `Comparativa de precios` para comenzar el flujo guiado de producto.")
        return

    raw_bytes = uploaded.getvalue()
    source_fingerprint = hashlib.sha1(raw_bytes).hexdigest()
    render_progress_slot(progress_slot, get_visible_excel_parse_progress(source_fingerprint))
    if source_fingerprint == st.session_state.day06_excel_source_fingerprint:
        excel_bundle = st.session_state.get("day06_excel_bundle")
        if isinstance(excel_bundle, ExcelRawParseBundle):
            st.success("Archivo cargado. La familia soportada ya está detectada para este workbook.")
        return

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
            source_name=uploaded.name,
            source_suffix=Path(uploaded.name).suffix,
            source_bytes=raw_bytes,
            progress_callback=update_parse_progress,
        )
    except Exception as error:
        progress_state = finish_progress_state(
            progress_state,
            status="error",
            final_message=f"No se pudo completar el parseo: {error}",
            include_current_in_history=False,
        )
        set_excel_parse_progress_state(progress_state, source_fingerprint=source_fingerprint)
        render_progress_slot(progress_slot, progress_state)
        st.session_state.day06_excel_bundle = None
        st.session_state.day06_excel_enrichment_df = None
        st.session_state.day06_excel_enrichment_saved_df = None
        st.session_state.day06_excel_source_fingerprint = ""
        st.session_state.pop("day06_excel_enrichment_editor", None)
        st.error(f"No se pudo parsear `Excel raw`: {error}")
        return

    enrichment_template = build_excel_enrichment_template(excel_bundle)
    progress_state = finish_progress_state(
        progress_state,
        status="success",
        final_message="Parseo completado. Workbook listo para revisar y completar enrichment.",
        include_current_in_history=True,
    )
    set_excel_parse_progress_state(progress_state, source_fingerprint=source_fingerprint)
    render_progress_slot(progress_slot, progress_state)
    st.session_state.day06_excel_bundle = excel_bundle
    st.session_state.day06_excel_enrichment_df = enrichment_template.copy()
    st.session_state.day06_excel_enrichment_saved_df = enrichment_template.copy()
    st.session_state.day06_excel_source_fingerprint = source_fingerprint
    st.session_state.pop("day06_excel_enrichment_editor", None)
    st.session_state.day06_input_payload = {
        "input_mode": "excel_raw",
        "source_name": uploaded.name,
        "source_suffix": Path(uploaded.name).suffix,
        "source_bytes": raw_bytes,
        "sheet_names": excel_bundle.sheet_names,
    }
    st.session_state.day06_last_failed_contract = None
    st.success("Familia soportada detectada. El workbook ya está listo para revisar parseo y completar enrichment.")


def render_excel_raw_parse_step(excel_bundle: ExcelRawParseBundle) -> None:
    """Render a compact parse summary once the workbook has been accepted."""
    summary = excel_bundle.parse_summary
    st.markdown("### Paso 2 · Revisa parseo")
    st.caption("Completado")
    st.success(
        "Se detectó una familia soportada `Comparativa de precios` con hojas `Tabla` y `Cálculos` listas para el flujo."
    )
    summary_df = build_display_summary_dataframe(
        [
            {"métrica": "archivo", "valor": excel_bundle.source_name},
            {"métrica": "hojas_detectadas", "valor": ", ".join(excel_bundle.sheet_names)},
            {"métrica": "tabla_seleccionada", "valor": summary.get("selected_table_sheet", "")},
            {"métrica": "calculos_seleccionada", "valor": summary.get("selected_calculos_sheet", "")},
            {"métrica": "eventos_base", "valor": summary.get("base_events_total", 0)},
            {"métrica": "filas_candidate_seed", "valor": summary.get("candidate_seed_rows_total", 0)},
            {"métrica": "rows_transport_signal", "valor": summary.get("transport_rows_with_signal", 0)},
        ]
    )
    st.dataframe(summary_df, hide_index=True, use_container_width=True)


def set_excel_enrichment_working_copy(enrichment_df: pd.DataFrame) -> None:
    """Replace the current enrichment working copy and reset pending editor patches."""
    st.session_state.day06_excel_enrichment_df = enrichment_df.copy()
    st.session_state.pop("day06_excel_enrichment_editor", None)


def render_excel_raw_bulk_enrichment_tools(
    *,
    enrichment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Render explicit batch-assist actions for the Excel enrichment step."""
    st.markdown("#### Ayuda masiva")
    st.caption("Aplica valores solo sobre filas vacías. El resultado queda visible en la tabla antes de guardar.")

    pending_summary = summarize_excel_enrichment_pending(enrichment_df)
    summary_df = build_display_summary_dataframe(
        [
            {"campo": "albaran_id", "filas_pendientes": pending_summary["pending_albaran_id"]},
            {"campo": "linea_id", "filas_pendientes": pending_summary["pending_linea_id"]},
            {"campo": "litros_evento", "filas_pendientes": pending_summary["pending_litros_evento"]},
        ],
        value_columns=("filas_pendientes",),
    )
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    working_df = enrichment_df.copy()

    albaran_left, albaran_right = st.columns([2, 1])
    with albaran_left:
        bulk_albaran_id = st.text_input(
            "Aplicar `albaran_id` a filas vacías",
            key="day06_bulk_albaran_id",
            placeholder="ALB_001",
        )
    with albaran_right:
        st.caption(f"Se aplicarían {pending_summary['pending_albaran_id']} fila(s).")
        apply_albaran = st.button("Aplicar albarán", key="day06_bulk_apply_albaran")
    if apply_albaran:
        try:
            working_df, affected_rows = apply_value_to_blank_enrichment_rows(
                enrichment_df=working_df,
                column_name="albaran_id",
                value=bulk_albaran_id,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            set_excel_enrichment_working_copy(working_df)
            st.success(
                "Sin cambios: no había filas vacías para `albaran_id`."
                if affected_rows == 0
                else f"`albaran_id` aplicado a {affected_rows} fila(s) vacías con valor `{bulk_albaran_id.strip()}`."
            )
            working_df = st.session_state.day06_excel_enrichment_df.copy()

    litros_left, litros_right = st.columns([2, 1])
    with litros_left:
        bulk_litros = st.number_input(
            "Aplicar `litros_evento` a filas vacías",
            key="day06_bulk_litros_evento",
            min_value=1.0,
            step=1.0,
            value=1000.0,
        )
    with litros_right:
        st.caption(f"Se aplicarían {pending_summary['pending_litros_evento']} fila(s).")
        apply_litros = st.button("Aplicar litros", key="day06_bulk_apply_litros")
    if apply_litros:
        try:
            working_df, affected_rows = apply_value_to_blank_enrichment_rows(
                enrichment_df=working_df,
                column_name="litros_evento",
                value=bulk_litros,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            set_excel_enrichment_working_copy(working_df)
            st.success(
                "Sin cambios: no había filas vacías para `litros_evento`."
                if affected_rows == 0
                else f"`litros_evento` aplicado a {affected_rows} fila(s) vacías con valor `{bulk_litros:.0f}`."
            )
            working_df = st.session_state.day06_excel_enrichment_df.copy()

    sequence_pending = summarize_excel_enrichment_pending(working_df)["pending_linea_id"]
    st.caption(
        "La secuencia de `linea_id` se genera en orden visible, empezando en `1`, y solo rellena huecos."
    )
    if st.button("Generar `linea_id` secuencial", key="day06_bulk_apply_linea"):
        try:
            working_df, affected_rows, sequence_range = apply_linea_sequence_to_blank_rows(
                enrichment_df=working_df,
                start_value=1,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            set_excel_enrichment_working_copy(working_df)
            st.success(
                "Sin cambios: no había filas vacías para `linea_id`."
                if affected_rows == 0
                else f"`linea_id` generado para {affected_rows} fila(s) vacías con rango `{sequence_range}`."
            )
            working_df = st.session_state.day06_excel_enrichment_df.copy()
    else:
        st.caption(f"Se aplicarían {sequence_pending} fila(s) vacías.")

    return working_df


def render_excel_raw_enrichment_step(
    excel_bundle: ExcelRawParseBundle,
    *,
    candidate_context: dict[str, Any],
) -> pd.DataFrame | None:
    """Render the enrichment step and return the candidate-grain dataframe when the input is ready."""
    st.markdown("### Paso 3 · Completa enrichment")
    candidate_df = candidate_context["candidate_df"]
    step_status_container = st.empty()
    step_status_container.caption("Completado" if isinstance(candidate_df, pd.DataFrame) else "Actual")
    st.caption(
        "Completa `litros_evento`, `albaran_id` y `linea_id`. Los cambios del editor solo se aplican al pulsar "
        "`Guardar enrichment y continuar`."
    )
    working_enrichment_df = render_excel_raw_bulk_enrichment_tools(
        enrichment_df=candidate_context["current_enrichment_df"],
    )
    candidate_context["current_enrichment_df"] = working_enrichment_df.copy()
    candidate_context["editable_df"] = working_enrichment_df.copy()

    with st.form("day06_excel_enrichment_form", clear_on_submit=False):
        edited_df = st.data_editor(
            working_enrichment_df,
            hide_index=True,
            use_container_width=True,
            key="day06_excel_enrichment_editor",
            disabled=[
                "event_seed_id",
                "fecha_evento",
                "producto_canonico",
                "terminal_compra",
            ],
            column_config={
                "albaran_id": st.column_config.TextColumn(
                    "albaran_id",
                    required=True,
                ),
                "linea_id": st.column_config.TextColumn(
                    "linea_id",
                    required=True,
                ),
                "litros_evento": st.column_config.NumberColumn(
                    "litros_evento",
                    min_value=1.0,
                    step=1.0,
                    required=True,
                ),
            },
        )
        enrichment_submitted = st.form_submit_button(
            "Guardar enrichment y continuar",
            type="primary",
        )

    if enrichment_submitted:
        normalized_edited_df = normalize_excel_enrichment_frame(edited_df)
        st.session_state.day06_excel_enrichment_df = normalized_edited_df.copy()
        st.session_state.day06_excel_enrichment_saved_df = normalized_edited_df.copy()
        st.success("Enrichment guardado. El flujo ya puede intentar construir el candidate-grain.")
        candidate_context = build_excel_raw_candidate_context(excel_bundle)
        candidate_df = candidate_context["candidate_df"]
        step_status_container.caption("Completado" if isinstance(candidate_df, pd.DataFrame) else "Actual")

    enrichment_status = candidate_context["enrichment_status"]
    if enrichment_status["status"] != "PASS":
        st.info(enrichment_status["message"])
        return None

    candidate_df = candidate_context["candidate_df"]
    if not isinstance(candidate_df, pd.DataFrame):
        candidate_error = str(candidate_context.get("candidate_error", "")).strip()
        st.warning(
            f"No se puede construir `candidate_grain` todavía: {candidate_error}"
            if candidate_error
            else "No se puede construir `candidate_grain` todavía."
        )
        return None

    st.success("Enrichment completo. El input ya está listo para validar contrato y elegir modo.")
    return candidate_df


def render_excel_raw_guided_flow() -> tuple[pd.DataFrame | None, dict[str, Any] | None, dict[str, Any]]:
    """Render the guided Day 06 Excel flow up to the point where candidate-grain becomes evaluable."""
    summary_container = st.container()
    render_excel_raw_upload_step()

    excel_bundle = st.session_state.get("day06_excel_bundle")
    candidate_context: dict[str, Any] = {
        "editable_df": None,
        "current_enrichment_df": None,
        "enrichment_status": {"status": "PENDING", "message": ""},
        "candidate_df": None,
        "candidate_error": None,
    }
    mode_availability: dict[str, Any] | None = None
    if not isinstance(excel_bundle, ExcelRawParseBundle):
        with summary_container:
            render_excel_raw_step_summary(
                build_excel_raw_step_states(
                    excel_bundle=None,
                    candidate_ready=False,
                    mode_availability=None,
                )
            )
        render_excel_raw_blocked_step(
            title="Paso 2 · Revisa parseo",
            message="Primero carga un workbook soportado para poder revisar qué hojas y qué universo candidato se detectaron.",
        )
        render_excel_raw_blocked_step(
            title="Paso 3 · Completa enrichment",
            message="El enrichment manual se habilita cuando el parseo detecta correctamente la familia `Comparativa de precios`.",
        )
        return None, st.session_state.get("day06_input_payload"), {"mode_availability": None}

    candidate_context = build_excel_raw_candidate_context(excel_bundle)
    render_excel_raw_parse_step(excel_bundle)
    candidate_df = render_excel_raw_enrichment_step(
        excel_bundle,
        candidate_context=candidate_context,
    )
    if isinstance(candidate_df, pd.DataFrame):
        mode_availability = inspect_input_mode_availability(
            input_df=candidate_df,
            input_mode="excel_raw",
        )
    with summary_container:
        render_excel_raw_step_summary(
            build_excel_raw_step_states(
                excel_bundle=excel_bundle,
                candidate_ready=isinstance(candidate_df, pd.DataFrame),
                mode_availability=mode_availability,
            )
        )
    return candidate_df, st.session_state.get("day06_input_payload"), {"mode_availability": mode_availability}


def render_excel_raw_candidate_preview(candidate_df: pd.DataFrame) -> None:
    """Render one compact candidate-grain summary plus a focused preview."""
    preview_cols = [
        column
        for column in [
            "event_id",
            "fecha_evento",
            "albaran_id",
            "linea_id",
            "producto_canonico",
            "terminal_compra",
            "proveedor_candidato",
            "coste_min_dia_proveedor",
            "v41_transport_cost_min_day_provider",
        ]
        if column in candidate_df.columns
    ]
    summary_df = build_display_summary_dataframe(
        [
            {"métrica": "rows_candidate_grain", "valor": int(len(candidate_df))},
            {
                "métrica": "eventos_unicos",
                "valor": int(candidate_df["event_id"].nunique()) if "event_id" in candidate_df.columns else 0,
            },
            {
                "métrica": "proveedores_unicos",
                "valor": int(candidate_df["proveedor_candidato"].nunique())
                if "proveedor_candidato" in candidate_df.columns
                else 0,
            },
        ]
    )
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    if preview_cols:
        st.dataframe(candidate_df[preview_cols], hide_index=True, use_container_width=True)


def render_excel_raw_contract_and_execution_steps(
    *,
    input_df: pd.DataFrame | None,
    input_payload: dict[str, Any] | None,
    flow_context: dict[str, Any] | None,
    current_context_fingerprint: str | None,
    render_mode_availability_panel_callback: Callable[..., tuple[dict[str, Any] | None, str | None]],
    run_product_inference_callback: Callable[..., None],
) -> int:
    """Render steps 4 and 5 for the guided Excel flow and execute the run when the input is ready."""
    del flow_context

    st.markdown("### Paso 4 · Valida contrato")
    if input_payload is None:
        st.caption("Pendiente")
        st.info("Sube un workbook soportado para poder validar contrato y disponibilidad de modos.")
        selected_mode = None
    else:
        if input_df is None:
            st.caption("Pendiente")
            st.info("Completa el enrichment manual para construir el `candidate_grain` y validar el contrato real.")
            selected_mode = None
        else:
            st.caption("Actual")
            _, selected_mode = render_mode_availability_panel_callback(
                input_df=input_df,
                input_payload=input_payload,
            )
            render_excel_raw_candidate_preview(input_df)

    st.markdown("### Paso 5 · Ejecuta")
    top_k = int(st.session_state.get("day06_excel_top_k", 2))
    run_progress_slot = st.empty()
    render_progress_slot(
        run_progress_slot,
        get_visible_contextual_state(
            "day06_run_progress",
            current_context_fingerprint=current_context_fingerprint,
        ),
    )
    if input_payload is None:
        st.caption("Pendiente")
        st.info("La ejecución se habilita después de cargar y reconocer un workbook soportado.")
        return top_k
    if input_df is None:
        st.caption("Pendiente")
        st.info("La ejecución se habilita cuando el enrichment deja el input listo para validación y scoring.")
        return top_k
    if selected_mode is None:
        st.caption("Pendiente")
        st.info("Necesitas al menos un modo viable por contrato para ejecutar el run operativo.")
        return top_k

    st.caption("Actual")
    st.caption("Selecciona el modo viable, confirma el Top-k y lanza el run operativo desde este último paso.")
    top_k = st.slider("Top-k objetivo", min_value=1, max_value=3, value=top_k, key="day06_excel_top_k")
    if st.button("Ejecutar run operativo", type="primary", key="day06_excel_execute_button"):
        run_product_inference_callback(
            input_df=input_df,
            input_payload=input_payload,
            inference_mode=selected_mode,
            top_k=top_k,
            context_fingerprint=current_context_fingerprint,
            progress_slot=run_progress_slot,
        )
    return top_k


def hydrate_excel_enrichment_dataframe(excel_bundle: ExcelRawParseBundle) -> pd.DataFrame:
    """Hydrate the enrichment dataframe from session state plus pending editor deltas."""
    editable_df = st.session_state.get("day06_excel_enrichment_df")
    if not isinstance(editable_df, pd.DataFrame):
        editable_df = build_excel_enrichment_template(excel_bundle)
    editable_df = normalize_excel_enrichment_frame(editable_df)

    editor_state = st.session_state.get("day06_excel_enrichment_editor")
    hydrated_df = apply_editor_state_patch(
        base_df=editable_df,
        editor_state=editor_state,
    )
    st.session_state.day06_excel_enrichment_df = hydrated_df.copy()
    return hydrated_df
