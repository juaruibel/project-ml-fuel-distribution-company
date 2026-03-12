from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.product.recommend_supplier import load_model_bundle, run_inference_dataframe, save_inference_output

st.set_page_config(page_title="Recommend Supplier · Public Portfolio", page_icon="⛽", layout="wide")

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "public"
DATA_DIR = PROJECT_ROOT / "data" / "public"
EXAMPLE_PATH = DATA_DIR / "inference_inputs" / "example_real_day_2024-05-28.csv"
TRAINING_SUMMARY_PATH = ARTIFACTS_DIR / "models" / "public_training_summary.json"
NOTEBOOK_SUMMARY_PATH = ARTIFACTS_DIR / "notebooks" / "notebook_execution_summary.json"
OUTPUT_DIR = DATA_DIR / "inference_outputs"


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_example() -> pd.DataFrame:
    return pd.read_csv(EXAMPLE_PATH)


def model_paths() -> tuple[Path, Path]:
    model_dir = PROJECT_ROOT / "models" / "public" / "champion_pure"
    return model_dir / "model.pkl", model_dir / "metadata.json"


def render_overview() -> None:
    st.title("Recommend Supplier · Public Portfolio")
    st.markdown(
        "- Repositorio público pseudonimizado y auditado.\n"
        "- Incluye los 21 notebooks, `src` relevante, training público de baseline y champion, y artefactos auditables.\n"
        "- No contiene `raw`, `staging`, `curated`, `reports`, `docs` ni ficheros Office/PDF."
    )

    training = load_json(TRAINING_SUMMARY_PATH)
    notebooks = load_json(NOTEBOOK_SUMMARY_PATH)
    left, right = st.columns(2)
    left.metric("Model parity", training["status"])
    right.metric("Notebook execution", notebooks["status"])

    rows = []
    for model_payload in training["models"]:
        row = {"role": model_payload["role"], "variant": model_payload["model_variant"]}
        row.update(model_payload["metrics"])
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_notebooks() -> None:
    st.header("Notebook inventory")
    notebook_paths = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    catalog = pd.DataFrame(
        [
            {
                "notebook": path.name,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
            }
            for path in notebook_paths
        ]
    )
    st.dataframe(catalog, hide_index=True, use_container_width=True)


def render_inference() -> None:
    st.header("Inference demo")
    if "public_input_df" not in st.session_state:
        st.session_state.public_input_df = None

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Load public example", type="primary"):
            st.session_state.public_input_df = load_example()
    with col2:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            st.session_state.public_input_df = pd.read_csv(uploaded)

    if st.session_state.public_input_df is None:
        st.info("Load the public example or upload a CSV with the public contract.")
        return

    edited = st.data_editor(st.session_state.public_input_df, num_rows="dynamic", use_container_width=True)
    top_k = st.slider("Top-k", min_value=1, max_value=3, value=2)

    if st.button("Run public inference"):
        model_path, metadata_path = model_paths()
        model, metadata, expected_feature_columns = load_model_bundle(model_path, metadata_path)
        result = run_inference_dataframe(
            input_df=edited,
            model=model,
            expected_feature_columns=expected_feature_columns,
            event_col="event_id",
            top_k=top_k,
        )
        output_path = save_inference_output(result, OUTPUT_DIR, prefix="public_reco")
        st.success(f"Saved output to {output_path.relative_to(PROJECT_ROOT)}")
        visible_cols = [
            column
            for column in [
                "event_id",
                "proveedor_candidato",
                "score_model",
                "rank_event_score",
                "is_top1",
                "is_topk",
                "target_elegido",
            ]
            if column in result.columns
        ]
        st.dataframe(result[visible_cols], use_container_width=True)
        st.caption(f"Model: {metadata.get('model_name', 'public_champion_pure')}")


section = st.sidebar.radio(
    "Section",
    ["Overview", "Notebook inventory", "Inference demo"],
)

if section == "Overview":
    render_overview()
elif section == "Notebook inventory":
    render_notebooks()
else:
    render_inference()
