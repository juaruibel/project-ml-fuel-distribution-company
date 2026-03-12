from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.ml.product.day06_runtime import build_reference_comparison_table
from src.ml.shared import functions as fc
from src.ml.shared.project_paths import (
    ASSETS_DIR,
    BASELINE_METADATA_PATH,
    CHAMPION_METADATA_PATH,
    DAY01_METRICS_PATH,
    DAY02_QUALITY_PATH,
    DAY04_RESULTS_PATH,
    DAY06_SLIDES_DIR,
    V1_PATH,
    V2_PATH,
)
from src.ml.ui.common import load_csv, load_json, render_metric_card, render_section_badge
from src.ml.ui.demo_story import DEMO_OVERVIEW_BULLETS, HOW_TO_USE_EXPANDER, NOTEBOOK_EXPANDERS


def render_demo_surface() -> None:
    """Render the Day 06 demo surface for presentation and model defense."""
    render_section_badge("Superficie Demo")
    st.markdown(
        """
        <div class="demo-hero">
          <h2>Demo del proyecto</h2>
          <p>Esta superficie explica el valor del producto, la defensa del champion y el recorrido de uso sin entrar en detalles tecnicos innecesarios.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, references_tab, evidence_tab, defense_tab = st.tabs(
        ["Historia", "Referencias activas", "Evidencia tecnica", "Defensa y uso"]
    )
    with overview_tab:
        render_demo_overview()
    with references_tab:
        render_reference_comparison()
    with evidence_tab:
        render_demo_evidence()
    with defense_tab:
        render_demo_defense()


def render_demo_overview() -> None:
    """Render the project story and the presentation framing."""
    intro_img = ASSETS_DIR / "intro_placeholder.jpg"
    if intro_img.exists():
        st.image(str(intro_img), use_container_width=True)

    for bullet in DEMO_OVERVIEW_BULLETS:
        st.markdown(f"- {bullet}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Serving historico", "LR_smote_0.5")
    with c2:
        render_metric_card("Champion puro", "Day05.1")
    with c3:
        render_metric_card("Policy visible", "LR + PRODUCT_002/PRODUCT_003")
    with c4:
        render_metric_card("Objetivo Day 06", "Producto + presentacion")

    st.markdown(
        """
        <div class="demo-runbook">
          <h4>Que se demuestra hoy</h4>
          <ul>
            <li>Que el producto reduce friccion diaria con un flujo corto y trazable.</li>
            <li>Que el champion es defendible frente al baseline historico.</li>
            <li>Que las reglas locales se auditaron con honestidad y no se sobreprometieron.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reference_comparison() -> None:
    """Render the active reference comparison table for Day 06."""
    reference_df = build_reference_comparison_table()
    st.markdown("### Referencias activas del producto")
    st.dataframe(reference_df, hide_index=True, use_container_width=True)
    st.markdown(
        "<p class='small-note'>`champion_pure` es la recomendacion conservadora de rollout. "
        "El serving default historico sigue congelado hasta una decision explicita de despliegue.</p>",
        unsafe_allow_html=True,
    )


def render_demo_evidence() -> None:
    """Render summarized notebook-derived evidence without using notebooks as navigation entries."""
    with st.expander("Notebook 01 · KNN baseline", expanded=False):
        render_nb01_summary()
    with st.expander("Notebook 02 · Feature Engineering", expanded=False):
        render_nb02_summary()
    with st.expander("Notebook 03 · Ensemble / baselines negocio", expanded=False):
        render_nb03_summary()
    with st.expander("Notebook 04 · Tuning y baseline Day04", expanded=False):
        render_nb04_summary()


def render_demo_defense() -> None:
    """Render the Day 05.3/05.4 defense narrative plus app usage guidance."""
    for notebook_story in NOTEBOOK_EXPANDERS:
        render_story_expander(notebook_story)
    render_story_expander(HOW_TO_USE_EXPANDER)


def render_story_expander(story_payload: dict[str, object]) -> None:
    """Render one curated story expander with cards and slide placeholders."""
    title = str(story_payload["title"])
    intro = str(story_payload["intro"])
    cards = list(story_payload.get("cards", []))
    slides = list(story_payload.get("slides", []))

    with st.expander(title, expanded=False):
        st.markdown(
            f"""
            <div class="demo-panel">
              <h3>{title}</h3>
              <p>{intro}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        card_blocks = "".join(
            f"<div class='demo-story-card'><strong>{card['title']}</strong><p>{card['body']}</p></div>"
            for card in cards
        )
        st.markdown(f"<div class='demo-story-grid'>{card_blocks}</div>", unsafe_allow_html=True)
        for slide in slides:
            render_slide_slot(slide)


def render_slide_slot(slide_payload: dict[str, str]) -> None:
    """Render one PNG slot, or one clean placeholder when the slide is missing."""
    slide_path = DAY06_SLIDES_DIR / slide_payload["filename"]
    is_pending = not slide_path.exists()
    css_class = "slide-slot is-pending" if is_pending else "slide-slot"
    status_label = "pendiente de slide" if is_pending else "PNG cargado"
    st.markdown(
        f"""
        <div class="{css_class}">
          <h4>{slide_payload['title']}</h4>
          <span class="slide-status">{status_label}</span>
          <p>{slide_path}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if slide_path.exists():
        st.image(str(slide_path), use_container_width=True)
    st.markdown(
        """
        <div class="slide-copy">
          <strong>Texto exacto para la slide</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(str(slide_payload["prompt_text"]), language="text")


def render_nb01_summary() -> None:
    """Render a compact summary of notebook 01."""
    df = load_csv(V1_PATH).copy()
    df["fecha_compra"] = pd.to_datetime(df["fecha_compra"], errors="coerce")
    df = df.dropna(subset=["fecha_compra", "proveedor_elegido"]).sort_values("fecha_compra").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    train_ratio = train["proveedor_elegido"].value_counts(normalize=True)
    test_ratio = test["proveedor_elegido"].value_counts(normalize=True)
    providers = train["proveedor_elegido"].value_counts().head(10).index
    dist_df = pd.DataFrame(
        {
            "proveedor": providers,
            "train_ratio": train_ratio.reindex(providers).fillna(0).values,
            "test_ratio": test_ratio.reindex(providers).fillna(0).values,
        }
    )

    left, right = st.columns([1.2, 1])
    with left:
        melted = dist_df.melt(id_vars="proveedor", var_name="split", value_name="ratio")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=melted, x="proveedor", y="ratio", hue="split", ax=ax)
        ax.set_xlabel("Proveedor")
        ax.set_ylabel("Ratio")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        metrics = fc.parse_day01_metrics_markdown(DAY01_METRICS_PATH)
        metric_table = pd.DataFrame(
            [
                {"metrica": "Dummy accuracy", "valor": metrics.get("dummy_accuracy")},
                {"metrica": "KNN k=5 accuracy", "valor": metrics.get("knn_k5_accuracy")},
                {"metrica": "KNN k=5 macro_f1", "valor": metrics.get("knn_k5_macro_f1")},
                {"metrica": "KNN k=5 balanced_accuracy", "valor": metrics.get("knn_k5_bal_acc")},
            ]
        )
        st.dataframe(metric_table, hide_index=True, use_container_width=True)


def render_nb02_summary() -> None:
    """Render a compact summary of notebook 02."""
    df = load_csv(V2_PATH).copy()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        render_metric_card("Clase positiva", f"{float(df['target_elegido'].astype(int).mean()):.2%}")
    with col_b:
        render_metric_card("Eventos", f"{int(df['event_id'].nunique()):,}")
    with col_c:
        render_metric_card(
            "Media candidatos/evento",
            f"{float(df.groupby('event_id')['proveedor_candidato'].nunique().mean()):.2f}",
        )

    quality = load_json(DAY02_QUALITY_PATH)
    quality_df = pd.DataFrame(
        [
            {"metrica": key, "valor": quality.get(key)}
            for key in [
                "events_total",
                "events_included",
                "events_excluded_no_positive",
                "avg_candidates_per_event",
                "p50_candidates_per_event",
                "p90_candidates_per_event",
                "rows_output",
            ]
        ]
    )
    st.dataframe(quality_df, hide_index=True, use_container_width=True)


def render_nb03_summary() -> None:
    """Render a compact summary of notebook 03 and business baselines."""
    results = load_csv(DAY04_RESULTS_PATH)
    selected = results[results["model_variant"].isin(["LogisticRegression (grid)", "GradientBoosting (grid)"])].copy()
    if not selected.empty:
        view = selected[["model_variant", "test_bal_acc", "top1_hit", "top2_hit", "test_f1_pos"]].copy()
        st.dataframe(view, hide_index=True, use_container_width=True)

    v2_df = load_csv(V2_PATH)
    df_model = fc.df_model_knn_feature(v2_df)
    train_df, test_df, _ = fc.split_temporal_feature(df_model)
    baseline_info = fc.compute_business_baselines(train_df, test_df)
    st.dataframe(baseline_info["baseline_table"], hide_index=True, use_container_width=True)


def render_nb04_summary() -> None:
    """Render a compact summary of notebook 04 and current model artifacts."""
    results = load_csv(DAY04_RESULTS_PATH).copy()
    leaderboard = results.sort_values(["top2_hit", "test_bal_acc", "test_f1_pos"], ascending=False).reset_index(drop=True)
    st.dataframe(
        leaderboard[
            [
                "model_variant",
                "search_type",
                "test_acc",
                "test_bal_acc",
                "test_f1_pos",
                "top1_hit",
                "top2_hit",
                "is_champion",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.markdown(
        f"<p class='small-note'>Baseline metadata: <code>{BASELINE_METADATA_PATH}</code><br>"
        f"Champion puro metadata: <code>{CHAMPION_METADATA_PATH}</code></p>",
        unsafe_allow_html=True,
    )
