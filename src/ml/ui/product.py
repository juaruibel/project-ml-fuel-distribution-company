from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.ml.product.day06_runtime import (
    Day06RunContractError,
    build_validation_user_message,
    execute_operational_run,
    get_mode_specs,
    inspect_input_mode_availability,
    prepare_operational_input,
)
from src.ml.shared.project_paths import SAMPLE_INPUT_PATH
from src.ml.ui.common import load_csv, render_section_badge
from src.ml.ui.display import render_message_block, render_technical_detail_expander
from src.ml.ui.excel_flow import (
    render_excel_raw_contract_and_execution_steps,
    render_excel_raw_guided_flow,
)
from src.ml.ui.feedback import render_feedback_editor
from src.ml.ui.progress import (
    advance_progress_state,
    build_progress_state,
    finish_progress_state,
    render_progress_slot,
)
from src.ml.ui.results import (
    render_contract_failure_panel,
    render_detail_results,
    render_last_validation,
    render_run_overview,
    render_runtime_evaluation,
    render_sql_infrastructure_panel,
    render_sql_publish_action,
)
from src.ml.ui.state import (
    build_input_context_fingerprint,
    get_visible_contextual_state,
    initialize_product_state,
    set_contextual_state,
)


def render_product_dev_surface() -> None:
    """Render the advanced Day 06 product surface for support and technical operations."""
    render_section_badge("Superficie Producto-Dev")
    st.markdown("## Producto-Dev", unsafe_allow_html=False)
    st.markdown(
        "Superficie avanzada para soporte, validacion contractual, SQL reporting y revision tecnica del run.",
        unsafe_allow_html=False,
    )
    st.info(
        "El serving default histórico sigue siendo `LR_smote_0.5`. "
        "La recomendación de `champion_pure` y `baseline_with_policy` depende del contrato real del input actual."
    )

    initialize_product_state(manual_template_factory=manual_template)

    input_df, input_payload, input_mode, flow_context = render_input_panel()
    current_context_fingerprint = build_input_context_fingerprint(
        input_df=input_df,
        input_payload=input_payload,
    )

    if input_mode == "Excel raw":
        top_k = render_excel_raw_contract_and_execution_steps(
            input_df=input_df,
            input_payload=input_payload,
            flow_context=flow_context,
            current_context_fingerprint=current_context_fingerprint,
            render_mode_availability_panel_callback=render_mode_availability_panel,
            run_product_inference_callback=run_product_inference,
        )
    else:
        top_k = st.slider("Top-k objetivo", min_value=1, max_value=3, value=2)
        _, selected_mode = render_mode_availability_panel(
            input_df=input_df,
            input_payload=input_payload,
        )
        run_progress_slot = st.empty()
        render_progress_slot(
            run_progress_slot,
            get_visible_contextual_state(
                "day06_run_progress",
                current_context_fingerprint=current_context_fingerprint,
            ),
        )

        if st.button(
            "Ejecutar run operativo",
            type="primary",
            disabled=input_df is None or selected_mode is None,
        ):
            run_product_inference(
                input_df=input_df,
                input_payload=input_payload,
                inference_mode=selected_mode,
                top_k=top_k,
                context_fingerprint=current_context_fingerprint,
                progress_slot=run_progress_slot,
            )

    render_last_validation(current_context_fingerprint=current_context_fingerprint)
    render_contract_failure_panel(
        input_df=input_df,
        input_payload=input_payload,
        top_k=top_k,
        current_context_fingerprint=current_context_fingerprint,
        run_product_inference_callback=run_product_inference,
    )

    # SQL infrastructure panel — always visible, even without a run
    render_sql_infrastructure_panel()

    post_action_notice = get_visible_contextual_state(
        "day06_post_action_notice",
        current_context_fingerprint=current_context_fingerprint,
    )
    if isinstance(post_action_notice, dict):
        render_message_block(
            message=str(post_action_notice.get("message", "")).strip(),
            severity=str(post_action_notice.get("severity", "info")),
            action_hint=str(post_action_notice.get("action_hint", "")),
        )
        render_technical_detail_expander(
            detail=str(post_action_notice.get("detail", "")).strip(),
            kv_pairs=list(post_action_notice.get("kv_pairs", [])),
        )
        set_contextual_state(
            "day06_post_action_notice",
            None,
            context_fingerprint=current_context_fingerprint,
        )

    last_run = get_visible_contextual_state(
        "day06_last_run",
        current_context_fingerprint=current_context_fingerprint,
    )
    if not isinstance(last_run, dict):
        return

    render_run_overview(last_run)
    render_detail_results(last_run, default_top_k=top_k)
    render_runtime_evaluation(last_run)
    render_sql_publish_action(last_run, current_context_fingerprint=current_context_fingerprint)
    render_feedback_editor(last_run, current_context_fingerprint=current_context_fingerprint)


def render_product_surface() -> None:
    """Backward-compatible alias for the advanced product surface."""
    render_product_dev_surface()


def build_mode_catalog_dataframe(mode_availability: dict[str, Any]) -> pd.DataFrame:
    """Build the user-facing mode catalog dataframe."""
    return pd.DataFrame(
        [
            {
                "modo": entry["mode_label"],
                "estado": str(entry["status"]).upper(),
                "contrato": entry["contract_status"],
                "recomendado": "Sí" if entry["recommended"] else "No",
                "motivo": entry["user_reason"],
            }
            for entry in mode_availability["mode_catalog"]
        ]
    )


def render_mode_detail(
    *,
    mode_entry: dict[str, Any],
    title: str = "Detalle técnico del modo",
) -> None:
    """Render one short visible reason plus technical support details for one mode."""
    action_hint = str(mode_entry.get("action_hint", "")).strip()
    if action_hint:
        st.caption(action_hint)
    render_technical_detail_expander(
        title=title,
        detail=str(mode_entry.get("developer_detail", "")).strip(),
        kv_pairs=[
            ("Severidad", mode_entry.get("severity", "")),
            ("Contrato", mode_entry.get("contract_status", "")),
            ("Columnas raw ausentes", ", ".join(mode_entry.get("missing_raw_columns", []))),
            ("Columnas críticas vacías", ", ".join(mode_entry.get("critical_all_null_columns", []))),
        ],
    )


def render_input_panel() -> tuple[pd.DataFrame | None, dict[str, Any] | None, str, dict[str, Any] | None]:
    """Render the Day 06 input panel and return the edited dataframe plus input payload."""
    input_mode = st.radio(
        "Modo de entrada",
        ["Excel raw", "CSV", "Formulario manual"],
        horizontal=True,
    )
    input_df: pd.DataFrame | None = None
    payload: dict[str, Any] | None = None
    flow_context: dict[str, Any] | None = None

    if input_mode == "CSV":
        left, right = st.columns([1, 2])
        with left:
            if st.button("Cargar ejemplo real (2028-05-28)"):
                sample_bytes = SAMPLE_INPUT_PATH.read_bytes()
                st.session_state.day06_uploaded_df = load_csv(SAMPLE_INPUT_PATH)
                st.session_state.day06_input_payload = {
                    "input_mode": "csv",
                    "source_name": SAMPLE_INPUT_PATH.name,
                    "source_suffix": SAMPLE_INPUT_PATH.suffix,
                    "source_bytes": sample_bytes,
                }
        with right:
            uploaded = st.file_uploader("Sube CSV para inferencia", type=["csv"], key="day06_csv_uploader")
            if uploaded is not None:
                payload = {
                    "input_mode": "csv",
                    "source_name": uploaded.name,
                    "source_suffix": Path(uploaded.name).suffix,
                    "source_bytes": uploaded.getvalue(),
                }
                st.session_state.day06_uploaded_df = pd.read_csv(BytesIO(payload["source_bytes"]))
                st.session_state.day06_input_payload = payload

        if st.session_state.day06_uploaded_df is not None:
            st.session_state.day06_uploaded_df = st.data_editor(
                st.session_state.day06_uploaded_df,
                num_rows="dynamic",
                use_container_width=True,
                key="day06_csv_editor",
            )
            input_df = st.session_state.day06_uploaded_df.copy()
            payload = st.session_state.day06_input_payload

    elif input_mode == "Excel raw":
        input_df, payload, flow_context = render_excel_raw_guided_flow()

    else:
        left, right = st.columns([1, 3])
        with left:
            if st.button("Añadir fila candidato"):
                new_row = st.session_state.day06_manual_df.iloc[[-1]].copy()
                st.session_state.day06_manual_df = pd.concat(
                    [st.session_state.day06_manual_df, new_row],
                    ignore_index=True,
                )
        with right:
            st.caption("Edita el input manual y luego ejecuta el run operativo.")

        st.session_state.day06_manual_df = st.data_editor(
            st.session_state.day06_manual_df,
            num_rows="dynamic",
            use_container_width=True,
            key="day06_manual_editor",
        )
        input_df = st.session_state.day06_manual_df.copy()
        payload = {
            "input_mode": "manual",
            "source_name": "manual_input.csv",
            "source_suffix": ".csv",
            "source_bytes": input_df.to_csv(index=False).encode("utf-8"),
        }

    return input_df, payload, input_mode, flow_context


def build_pending_mode_catalog() -> pd.DataFrame:
    """Build the placeholder mode catalog shown before an `Excel raw` input becomes contract-evaluable."""
    return pd.DataFrame(
        [
            {
                "modo": spec.label,
                "estado": "PENDING",
                "contrato": "PENDING",
                "recomendado": "No",
                "motivo": "Completa enrichment manual y construye `candidate_grain` para evaluar el contrato real.",
            }
            for spec in get_mode_specs()
        ]
    )


def render_disabled_champion_notice(mode_availability: dict[str, Any]) -> None:
    """Render one short explanation block when `champion_pure` exists but is unavailable for the current input."""
    champion_entry = next(
        (entry for entry in mode_availability["mode_catalog"] if entry["mode_key"] == "champion_pure"),
        None,
    )
    if not isinstance(champion_entry, dict) or bool(champion_entry.get("enabled")):
        return

    render_message_block(
        message="`champion_pure` existe en el catálogo, pero no está disponible para este input.",
        severity=str(champion_entry.get("severity", "warning")),
        action_hint=str(champion_entry.get("action_hint", "")),
    )
    render_technical_detail_expander(
        title="Detalle técnico de `champion_pure`",
        detail=str(champion_entry.get("developer_detail", "")).strip(),
        kv_pairs=[
            ("Columnas raw ausentes", ", ".join(champion_entry.get("missing_raw_columns", []))),
            ("Columnas críticas vacías", ", ".join(champion_entry.get("critical_all_null_columns", []))),
        ],
    )


def _render_mode_selection_from_availability(mode_availability: dict[str, Any]) -> str | None:
    """Render the selector for already computed mode availability."""
    st.dataframe(build_mode_catalog_dataframe(mode_availability), hide_index=True, use_container_width=True)
    render_disabled_champion_notice(mode_availability)

    enabled_mode_keys = list(mode_availability["enabled_mode_keys"])
    selected_default_mode = mode_availability["selected_default_mode"]
    if not enabled_mode_keys or selected_default_mode is None:
        st.error("No hay ningún modo viable para el input actual. Revisa el contrato del input antes de ejecutar.")
        st.session_state.day06_mode_selector = None
        return None

    if st.session_state.day06_mode_selector not in enabled_mode_keys:
        st.session_state.day06_mode_selector = selected_default_mode

    mode_labels = {spec.key: spec.label for spec in get_mode_specs()}
    selected_mode = st.selectbox(
        "Modo de inferencia",
        options=enabled_mode_keys,
        key="day06_mode_selector",
        format_func=lambda key: (
            f"{mode_labels[key]} · recomendado"
            if key == selected_default_mode
            else mode_labels[key]
        ),
        help="El selector solo muestra modos habilitados; el catálogo superior enseña también los deshabilitados y el motivo.",
    )
    selected_entry = next(
        entry for entry in mode_availability["mode_catalog"] if entry["mode_key"] == selected_mode
    )
    render_mode_detail(mode_entry=selected_entry)
    return selected_mode


def render_mode_availability_panel(
    *,
    input_df: pd.DataFrame | None,
    input_payload: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Render the dynamic Day 06 mode catalog and return the current availability plus selected mode."""
    st.markdown("### Disponibilidad de modos")
    if input_payload is None:
        st.info("Carga un input para evaluar qué modos están disponibles y cuál se recomienda por contrato.")
        st.session_state.day06_mode_availability = None
        return None, None

    input_mode = str(input_payload.get("input_mode", ""))
    if input_df is None:
        if input_mode == "excel_raw":
            st.info(
                "`Excel raw` es el flujo prioritario del producto, pero antes hay que cerrar el enrichment manual "
                "para evaluar el contrato real de los modos."
            )
            st.dataframe(build_pending_mode_catalog(), hide_index=True, use_container_width=True)
        else:
            st.warning("El input actual todavía no es evaluable para seleccionar modo.")
        st.session_state.day06_mode_availability = None
        return None, None

    mode_availability = inspect_input_mode_availability(
        input_df=input_df,
        input_mode=input_mode,
    )
    st.session_state.day06_mode_availability = mode_availability
    return mode_availability, _render_mode_selection_from_availability(mode_availability)


def run_product_inference(
    *,
    input_df: pd.DataFrame | None,
    input_payload: dict[str, Any] | None,
    inference_mode: str,
    top_k: int,
    context_fingerprint: str | None,
    progress_slot: Any | None = None,
) -> None:
    """Stage the input, validate it and execute one operational Day 06 run."""
    if input_df is None or input_payload is None:
        return

    mode_availability = inspect_input_mode_availability(
        input_df=input_df,
        input_mode=str(input_payload["input_mode"]),
    )
    mode_entry = next(
        (entry for entry in mode_availability["mode_catalog"] if entry["mode_key"] == inference_mode),
        None,
    )
    if mode_entry is None or not bool(mode_entry["enabled"]):
        set_contextual_state("day06_run_progress", None, context_fingerprint=context_fingerprint)
        render_progress_slot(progress_slot, None)
        set_contextual_state("day06_last_run", None, context_fingerprint=context_fingerprint)
        set_contextual_state("day06_last_failed_contract", None, context_fingerprint=context_fingerprint)
        if isinstance(mode_entry, dict):
            render_message_block(
                message=str(mode_entry.get("user_reason", "El modo seleccionado no está disponible.")),
                severity=str(mode_entry.get("severity", "error")),
                action_hint=str(mode_entry.get("action_hint", "")),
            )
            render_mode_detail(mode_entry=mode_entry)
        else:
            st.error("El modo seleccionado no está disponible.")
        return

    progress_state = build_progress_state()

    def update_run_progress(step_key: str, user_message: str) -> None:
        nonlocal progress_state
        progress_state = advance_progress_state(
            progress_state,
            step_key=step_key,
            user_message=user_message,
        )
        set_contextual_state("day06_run_progress", progress_state, context_fingerprint=context_fingerprint)
        render_progress_slot(progress_slot, progress_state)

    prepared_input = prepare_operational_input(
        input_df=input_df,
        input_mode=str(input_payload["input_mode"]),
        source_name=str(input_payload["source_name"]),
        source_suffix=str(input_payload["source_suffix"]),
        source_bytes=bytes(input_payload["source_bytes"]),
        progress_callback=update_run_progress,
    )
    set_contextual_state(
        "day06_last_validation",
        prepared_input.validation_summary,
        context_fingerprint=context_fingerprint,
    )
    set_contextual_state(
        "day06_last_validation_path",
        prepared_input.validation_report_path,
        context_fingerprint=context_fingerprint,
    )

    validation_status = str(prepared_input.validation_summary.get("status", "")).upper()
    if validation_status == "FAIL":
        validation_message = build_validation_user_message(prepared_input.validation_summary)
        progress_state = finish_progress_state(
            progress_state,
            status="error",
            final_message=validation_message["user_message"],
            include_current_in_history=False,
        )
        set_contextual_state("day06_run_progress", progress_state, context_fingerprint=context_fingerprint)
        render_progress_slot(progress_slot, progress_state)
        set_contextual_state("day06_last_run", None, context_fingerprint=context_fingerprint)
        set_contextual_state("day06_last_failed_contract", None, context_fingerprint=context_fingerprint)
        render_message_block(
            message=validation_message["user_message"],
            severity=validation_message["severity"],
            action_hint=validation_message["action_hint"],
        )
        return

    if validation_status == "PASS_WITH_WARNINGS":
        validation_message = build_validation_user_message(prepared_input.validation_summary)
        render_message_block(
            message=validation_message["user_message"],
            severity=validation_message["severity"],
            action_hint=validation_message["action_hint"],
        )

    try:
        run_bundle = execute_operational_run(
            prepared_input=prepared_input,
            inference_mode=inference_mode,
            top_k=top_k,
            surface="product",
            progress_callback=update_run_progress,
        )
    except Day06RunContractError as error:
        progress_state = finish_progress_state(
            progress_state,
            status="error",
            final_message=str(error.manifest_payload.get("user_message", error.message)),
            include_current_in_history=False,
        )
        set_contextual_state("day06_run_progress", progress_state, context_fingerprint=context_fingerprint)
        render_progress_slot(progress_slot, progress_state)
        set_contextual_state("day06_last_run", None, context_fingerprint=context_fingerprint)
        set_contextual_state(
            "day06_last_failed_contract",
            {
                "message": error.message,
                "user_message": str(error.manifest_payload.get("user_message", error.message)),
                "developer_detail": str(error.manifest_payload.get("developer_detail", error.message)),
                "severity": str(error.manifest_payload.get("error_severity", "error")),
                "action_hint": str(error.manifest_payload.get("action_hint", "")),
                "run_manifest_path": str(error.run_manifest_path),
                "contract_report_path": str(error.contract_report_path),
                "manifest_payload": error.manifest_payload,
            },
            context_fingerprint=context_fingerprint,
        )
        render_message_block(
            message=str(error.manifest_payload.get("user_message", error.message)),
            severity=str(error.manifest_payload.get("error_severity", "error")),
            action_hint=str(error.manifest_payload.get("action_hint", "")),
        )
        return

    progress_state = finish_progress_state(
        progress_state,
        status="success",
        final_message="Run operativo completado. Resultados y artefactos listos.",
        include_current_in_history=True,
    )
    set_contextual_state("day06_run_progress", progress_state, context_fingerprint=context_fingerprint)
    render_progress_slot(progress_slot, progress_state)
    set_contextual_state("day06_last_run", run_bundle, context_fingerprint=context_fingerprint)
    set_contextual_state("day06_last_failed_contract", None, context_fingerprint=context_fingerprint)
    set_contextual_state(
        "day06_feedback_editor_df",
        run_bundle["feedback_df"].copy(),
        context_fingerprint=context_fingerprint,
    )
    render_message_block(
        message="Run operativo completado.",
        severity="success",
        action_hint="Ya puedes revisar resultados, advertencias y feedback humano.",
    )


def manual_template() -> pd.DataFrame:
    """Return the default manual Day 06 input template."""
    return pd.DataFrame(
        [
            {
                "event_id": "MANUAL_001",
                "fecha_evento": "2029-02-27",
                "albaran_id": "ALB_MANUAL_001",
                "linea_id": "1",
                "proveedor_candidato": "SUPPLIER_009",
                "producto_canonico": "PRODUCT_002",
                "terminal_compra": "TERMINAL_001",
                "coste_min_dia_proveedor": 0.9000,
                "rank_coste_dia_producto": 1,
                "terminales_cubiertos": 6,
                "observaciones_oferta": 8,
                "candidatos_evento_count": 10,
                "coste_min_evento": 0.9000,
                "coste_max_evento": 0.9600,
                "spread_coste_evento": 0.0600,
                "delta_vs_min_evento": 0.0000,
                "ratio_vs_min_evento": 0.0000,
                "litros_evento": 33000,
                "dia_semana": 4,
                "mes": 2,
                "fin_mes": 0,
                "blocked_by_rule_candidate": 0,
            },
            {
                "event_id": "MANUAL_001",
                "fecha_evento": "2029-02-27",
                "albaran_id": "ALB_MANUAL_001",
                "linea_id": "1",
                "proveedor_candidato": "SUPPLIER_050",
                "producto_canonico": "PRODUCT_002",
                "terminal_compra": "TERMINAL_001",
                "coste_min_dia_proveedor": 0.9120,
                "rank_coste_dia_producto": 2,
                "terminales_cubiertos": 6,
                "observaciones_oferta": 8,
                "candidatos_evento_count": 10,
                "coste_min_evento": 0.9000,
                "coste_max_evento": 0.9600,
                "spread_coste_evento": 0.0600,
                "delta_vs_min_evento": 0.0120,
                "ratio_vs_min_evento": 0.0133,
                "litros_evento": 33000,
                "dia_semana": 4,
                "mes": 2,
                "fin_mes": 0,
                "blocked_by_rule_candidate": 0,
            },
        ]
    )
