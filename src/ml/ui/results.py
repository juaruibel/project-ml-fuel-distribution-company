from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from src.ml.product.day06_runtime import build_validation_user_message, compute_runtime_evaluation
from src.ml.product.day06_sql_store import (
    get_day06_sql_store_status,
    initialize_day06_sql_store,
    publish_day06_run_bundle,
)
from src.ml.shared import functions as fc
from src.ml.shared.project_paths import V2_PATH
from src.ml.ui.common import load_csv, render_metric_card
from src.ml.ui.display import (
    build_display_summary_dataframe,
    normalize_display_dataframe,
    render_message_block,
    render_technical_detail_expander,
)
from src.ml.ui.state import get_visible_contextual_state, set_contextual_state


def render_contract_failure_panel(
    *,
    input_df: pd.DataFrame | None,
    input_payload: dict[str, Any] | None,
    top_k: int,
    current_context_fingerprint: str | None,
    run_product_inference_callback: Callable[..., None],
) -> None:
    """Render the explicit fallback actions when one selected mode fails pre-scoring."""
    failure_payload = get_visible_contextual_state(
        "day06_last_failed_contract",
        current_context_fingerprint=current_context_fingerprint,
    )
    if not isinstance(failure_payload, dict):
        return

    st.markdown("### Contrato de scoring")
    render_message_block(
        message=str(failure_payload.get("user_message", "El run falló antes del scoring.")),
        severity=str(failure_payload.get("severity", "error")),
        action_hint=str(failure_payload.get("action_hint", "")),
    )
    render_technical_detail_expander(
        detail=str(failure_payload.get("developer_detail", failure_payload.get("message", ""))).strip(),
        kv_pairs=[
            ("Manifest", failure_payload.get("run_manifest_path", "")),
            ("Reporte contractual", failure_payload.get("contract_report_path", "")),
        ],
    )
    fallback_modes = list(failure_payload.get("manifest_payload", {}).get("fallback_modes", []))
    if input_df is None or input_payload is None or not fallback_modes:
        return

    left, right = st.columns(2)
    with left:
        if "baseline" in fallback_modes and st.button("Relanzar como baseline", key="day06_rerun_baseline"):
            run_product_inference_callback(
                input_df=input_df,
                input_payload=input_payload,
                inference_mode="baseline",
                top_k=top_k,
                context_fingerprint=current_context_fingerprint,
            )
    with right:
        if "baseline_with_policy" in fallback_modes and st.button(
            "Relanzar como baseline_with_policy",
            key="day06_rerun_baseline_with_policy",
        ):
            run_product_inference_callback(
                input_df=input_df,
                input_payload=input_payload,
                inference_mode="baseline_with_policy",
                top_k=top_k,
                context_fingerprint=current_context_fingerprint,
            )


def render_last_validation(*, current_context_fingerprint: str | None) -> None:
    """Render the latest validation summary if present."""
    validation_summary = get_visible_contextual_state(
        "day06_last_validation",
        current_context_fingerprint=current_context_fingerprint,
    )
    validation_path = get_visible_contextual_state(
        "day06_last_validation_path",
        current_context_fingerprint=current_context_fingerprint,
    )
    if not isinstance(validation_summary, dict):
        return

    st.markdown("### Validación de input diario")
    validation_message = build_validation_user_message(validation_summary)
    render_message_block(
        message=validation_message["user_message"],
        severity=validation_message["severity"],
        action_hint=validation_message["action_hint"],
    )

    summary_df = build_display_summary_dataframe(
        [
            {"métrica": "rows_total", "valor": validation_summary.get("rows_total")},
            {"métrica": "columns_total", "valor": validation_summary.get("columns_total")},
            {"métrica": "error_count", "valor": validation_summary.get("error_count")},
            {"métrica": "warning_count", "valor": validation_summary.get("warning_count")},
        ]
    )
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    technical_tables: list[tuple[str, pd.DataFrame]] = []
    if validation_summary.get("errors"):
        technical_tables.append(("Errores", pd.DataFrame(validation_summary["errors"])))
    if validation_summary.get("warnings"):
        technical_tables.append(("Warnings", pd.DataFrame(validation_summary["warnings"])))
    render_technical_detail_expander(
        detail=validation_message["developer_detail"],
        kv_pairs=[("Reporte JSON de validación", validation_path)],
        tables=technical_tables,
    )


def render_run_overview(run_bundle: dict[str, Any]) -> None:
    """Render the high-level overview of the latest Day 06 run."""
    manifest = run_bundle["manifest_payload"]
    warning_events = int(manifest["warning_events"])
    if warning_events > 0:
        render_message_block(
            message="Run completado con advertencias operativas.",
            severity="warning",
            action_hint="Puedes revisar los eventos marcados antes de cerrar el feedback.",
        )
    else:
        render_message_block(
            message="Run completado correctamente.",
            severity="success",
            action_hint="Ya puedes revisar el detalle, validar la decisión y guardar feedback.",
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Run id", str(manifest["run_id"]))
    with c2:
        render_metric_card("Modo", str(run_bundle["mode_spec"].key))
    with c3:
        sql_label = str(manifest.get("sql_store_status", manifest.get("sql_mirror_status", "")))
        render_metric_card("SQL reporting", sql_label)
    with c4:
        render_metric_card("Warnings", str(manifest["warning_events"]))

    manifest_path = run_bundle["run_manifest_path"]
    if manifest_path.exists():
        st.download_button(
            label="Descargar run manifest",
            data=manifest_path.read_bytes(),
            file_name=manifest_path.name,
            mime="application/json",
        )
    contract_report_path = run_bundle.get("scoring_contract_report_path")
    if contract_report_path is not None and contract_report_path.exists():
        st.download_button(
            label="Descargar reporte contractual",
            data=contract_report_path.read_bytes(),
            file_name=contract_report_path.name,
            mime="application/json",
        )
    render_technical_detail_expander(
        kv_pairs=[
            ("Input original", manifest["input_original_path"]),
            ("Candidate grain", manifest["input_candidate_grain_path"]),
            ("Input normalizado", manifest["input_normalized_path"]),
            ("Contrato scoring", manifest["scoring_contract_report_path"]),
            ("Detalle", manifest["detail_csv"]),
        ],
    )


def render_sql_infrastructure_panel() -> None:
    """Render SQL infrastructure actions: initialize and check status. Always visible."""
    st.markdown("### SQL reporting")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Inicializar base de datos", key="day06_sql_init"):
            result = initialize_day06_sql_store()
            status = result.get("status", "error")
            if status in ("initialized", "ready"):
                render_message_block(
                    message=f"Store SQL inicializada ({result.get('tables_created', 0)} tablas, {result.get('views_created', 0)} vistas).",
                    severity="success",
                    action_hint=f"Backend: {result.get('backend', '')} · Target: {result.get('target', '')}",
                )
            elif status == "disabled_unconfigured":
                render_message_block(
                    message="SQL store no configurada. Exporta DAY06_SQL_STORE_PATH para activar.",
                    severity="warning",
                    action_hint="export DAY06_SQL_STORE_PATH=/path/to/day06_store.db",
                )
            else:
                render_message_block(message=f"Estado inesperado: {status}", severity="error")
    with c2:
        if st.button("Comprobar estado SQL", key="day06_sql_status"):
            store_status = get_day06_sql_store_status()
            status_val = store_status.get("status", "unknown")
            severity = "success" if status_val == "ready" else ("warning" if "configured" in status_val else "info")
            msg_parts = [
                f"Estado: **{status_val}**",
                f"Backend: {store_status.get('backend', 'none')}",
            ]
            if store_status.get("runs_total") is not None:
                msg_parts.append(f"Runs publicados: {store_status['runs_total']}")
            render_message_block(
                message=" · ".join(msg_parts),
                severity=severity,
                action_hint=f"Target: {store_status.get('target', 'N/A')}",
            )
            row_counts = store_status.get("row_counts")
            if isinstance(row_counts, dict) and row_counts:
                st.dataframe(
                    build_display_summary_dataframe(
                        [{"tabla": k, "filas": v} for k, v in row_counts.items()]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )


def render_sql_publish_action(
    run_bundle: dict[str, Any],
    *,
    current_context_fingerprint: str | None,
) -> None:
    """Render the manual SQL re-publish action. Only visible after a run."""
    st.caption("El run ya se publica automáticamente. Usa esta acción solo para republicar o reintentar.")
    if st.button("Republicar/Reintentar publicación SQL", key="day06_sql_publish"):
        manifest = run_bundle["manifest_payload"]
        pub_result = publish_day06_run_bundle(
            run_manifest=manifest,
            normalized_df=run_bundle["prepared_input"].normalized_df,
            resumen_df=run_bundle["resumen_df"],
            feedback_df=run_bundle["feedback_df"],
            warning_df=run_bundle.get("warning_df"),
        )
        pub_status = pub_result.get("status", "error")
        if pub_status == "published":
            manifest["sql_store_backend"] = str(pub_result.get("backend", manifest.get("sql_store_backend", "")))
            manifest["sql_store_target"] = str(pub_result.get("target", manifest.get("sql_store_target", "")))
            manifest["sql_store_status"] = pub_status
            manifest["sql_store_published_at_utc"] = str(
                pub_result.get("published_at_utc", manifest.get("sql_store_published_at_utc", ""))
            )
            run_bundle["manifest_payload"] = manifest
            set_contextual_state(
                "day06_last_run",
                run_bundle,
                context_fingerprint=current_context_fingerprint,
            )
            set_contextual_state(
                "day06_post_action_notice",
                {
                    "message": f"Run {pub_result.get('run_id', '')} republicado en SQL reporting.",
                    "severity": "success",
                    "action_hint": (
                        f"{pub_result.get('rows_event_decisions', 0)} decisiones · "
                        f"{pub_result.get('rows_feedback', 0)} feedback rows"
                    ),
                    "detail": f"SQL reporting: {pub_status}",
                    "kv_pairs": [
                        ("Target", str(pub_result.get("target", ""))),
                        ("Publicado en", str(pub_result.get("published_at_utc", ""))),
                    ],
                },
                context_fingerprint=current_context_fingerprint,
            )
            st.rerun()
        elif pub_status == "disabled_unconfigured":
            render_message_block(
                message="SQL store no configurada.",
                severity="warning",
                action_hint="export DAY06_SQL_STORE_PATH=/path/to/day06_store.db",
            )
        else:
            render_message_block(message=f"Publicación: {pub_status}", severity="error")


def render_detail_results(run_bundle: dict[str, Any], default_top_k: int) -> None:
    """Render detail, event summary, albaran summary and warning tables."""
    detail_df = run_bundle["detail_df"]
    resumen_df = run_bundle["resumen_df"]
    resumen_albaran_df = run_bundle["resumen_albaran_df"]
    warning_df = run_bundle["warning_df"]

    st.markdown("### Resultado de inferencia")
    event_options = sorted(detail_df["event_id"].astype(str).unique().tolist()) if "event_id" in detail_df.columns else []
    selected_events = st.multiselect("Filtrar por evento", options=event_options)
    display_top_k = st.slider("Mostrar hasta rank", min_value=1, max_value=8, value=default_top_k, key="day06_display_topk")

    filtered = detail_df.copy()
    if selected_events:
        filtered = filtered[filtered["event_id"].astype(str).isin(selected_events)]
    filtered["rank_event_score"] = pd.to_numeric(filtered["rank_event_score"], errors="coerce").fillna(999).astype(int)
    filtered = filtered[filtered["rank_event_score"] <= int(display_top_k)]
    display_cols = [
        column for column in [
            "event_id",
            "proveedor_candidato",
            "score_model",
            "rank_event_score",
            "is_top1",
            "is_topk",
            "low_confidence_flag",
            "warning_reasons",
            "target_elegido",
        ] if column in filtered.columns
    ]
    st.dataframe(filtered[display_cols], use_container_width=True)
    st.download_button(
        label="Descargar CSV detalle",
        data=run_bundle["detail_path"].read_bytes(),
        file_name=run_bundle["detail_path"].name,
        mime="text/csv",
    )

    st.markdown("### Resumen por evento")
    event_cols = [
        column for column in [
            "event_id",
            "recommended_supplier",
            "decision_final",
            "decision_source",
            "low_confidence_flag",
            "warning_reasons",
            "override_reason",
            "run_id",
        ] if column in resumen_df.columns
    ]
    st.dataframe(resumen_df[event_cols], hide_index=True, use_container_width=True)
    st.download_button(
        label="Descargar CSV resumen por evento",
        data=run_bundle["resumen_evento_path"].read_bytes(),
        file_name=run_bundle["resumen_evento_path"].name,
        mime="text/csv",
    )

    st.markdown("### Resumen por albarán")
    albaran_cols = [
        column for column in [
            "fecha_evento",
            "albaran_id",
            "eventos_total",
            "eventos_override",
            "coherence_before",
            "coherence_after",
            "run_id",
        ] if column in resumen_albaran_df.columns
    ]
    st.dataframe(resumen_albaran_df[albaran_cols], hide_index=True, use_container_width=True)
    st.download_button(
        label="Descargar CSV resumen por albarán",
        data=run_bundle["resumen_albaran_path"].read_bytes(),
        file_name=run_bundle["resumen_albaran_path"].name,
        mime="text/csv",
    )

    if not warning_df.empty:
        flagged = warning_df[warning_df["low_confidence_flag"] == 1].copy()
        if not flagged.empty:
            st.markdown("### Warnings operativos")
            st.dataframe(
                normalize_display_dataframe(flagged[["event_id", "warning_reasons"]], all_columns=True),
                hide_index=True,
                use_container_width=True,
            )


def render_runtime_evaluation(run_bundle: dict[str, Any]) -> None:
    """Render optional evaluation and business baselines when labels are present."""
    detail_df = run_bundle["detail_df"]
    evaluation = compute_runtime_evaluation(detail_df)
    st.markdown("### Contexto de decisión")

    p1, p2 = fc.get_top_providers_from_history(load_csv(V2_PATH))
    if p1:
        text = f"Proveedor histórico top-1 en train: **{p1}**"
        if p2:
            text += f" · top-2: **{p2}**"
        st.markdown(text)

    if all(column in detail_df.columns for column in ["event_id", "coste_min_dia_proveedor", "proveedor_candidato"]):
        cheapest = detail_df.copy()
        cheapest["coste_min_dia_proveedor"] = pd.to_numeric(
            cheapest["coste_min_dia_proveedor"], errors="coerce"
        )
        cheapest = cheapest.sort_values(["event_id", "coste_min_dia_proveedor"], ascending=[True, True])
        cheapest_top1 = cheapest.groupby("event_id", sort=False).head(1)
        st.dataframe(
            cheapest_top1[["event_id", "proveedor_candidato", "coste_min_dia_proveedor"]],
            hide_index=True,
            use_container_width=True,
        )

    if evaluation is None:
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Accuracy", f"{evaluation['row_metrics']['accuracy']:.4f}")
    with c2:
        render_metric_card("Balanced accuracy", f"{evaluation['row_metrics']['balanced_accuracy']:.4f}")
    with c3:
        render_metric_card("F1 clase positiva", f"{evaluation['row_metrics']['f1_pos']:.4f}")

    event_metrics_df = build_display_summary_dataframe(
        [
            {"métrica": "top1_hit", "valor": evaluation["event_metrics"]["top1_hit"]},
            {"métrica": "top2_hit", "valor": evaluation["event_metrics"]["top2_hit"]},
        ]
    )
    st.dataframe(event_metrics_df, hide_index=True, use_container_width=True)
