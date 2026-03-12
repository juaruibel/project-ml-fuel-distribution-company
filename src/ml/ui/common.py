from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Load one CSV file from disk with a stable Streamlit cache."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    """Load one JSON file from disk with a stable Streamlit cache."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def render_metric_card(title: str, value: str) -> None:
    """Render one compact metric card."""
    st.markdown(
        f"""
        <div class=\"card\">
            <p class=\"card-title\">{title}</p>
            <p class=\"card-value\">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_badge(label: str) -> None:
    """Render one compact badge for section context."""
    st.markdown(f"<span class='surface-chip'>{label}</span>", unsafe_allow_html=True)
