from __future__ import annotations

import json
import os

import streamlit.components.v1 as components


APP_TITLE = "Recommend Supplier · Day 06"
APP_ICON = "⛽"
APP_PROFILE_FULL = "full"
APP_PROFILE_CLOUD_DEMO = "cloud_demo"
APP_PROFILE_OPTIONS = (
    APP_PROFILE_FULL,
    APP_PROFILE_CLOUD_DEMO,
)

SURFACE_MODE_CLASSES = (
    "surface-demo-mode",
    "surface-product-dev-mode",
    "surface-product-user-mode",
)


def get_app_profile() -> str:
    """Return the supported Day 06 app profile, defaulting to `full`."""
    raw_value = os.getenv("DAY06_APP_PROFILE", APP_PROFILE_FULL).strip().lower()
    if raw_value in APP_PROFILE_OPTIONS:
        return raw_value
    return APP_PROFILE_FULL


def is_cloud_demo_profile() -> bool:
    """Return whether the current profile is the reduced cloud demo mode."""
    return get_app_profile() == APP_PROFILE_CLOUD_DEMO


def get_cloud_demo_access_code() -> str:
    """Return the shared passphrase required by the private cloud demo, if configured."""
    return os.getenv("DAY06_DEMO_ACCESS_CODE", "").strip()


def apply_surface_mode(surface_mode: str) -> None:
    """Toggle one scoped body class so each surface can own its visual language."""
    if surface_mode not in SURFACE_MODE_CLASSES:
        raise ValueError(f"surface_mode no soportado: {surface_mode}")

    mode_classes_json = json.dumps(list(SURFACE_MODE_CLASSES))
    selected_mode_json = json.dumps(surface_mode)
    components.html(
        f"""
        <script>
        const modeClasses = {mode_classes_json};
        const selectedMode = {selected_mode_json};
        const body = window.parent.document.body;
        body.classList.remove(...modeClasses);
        body.classList.add(selectedMode);
        </script>
        """,
        height=0,
        width=0,
    )


STYLE = """
<style>
:root {
  --bg-soft: #f2ece5;
  --card-bg: #fffdf9;
  --panel-bg: #ffffff;
  --text-main: #23190f;
  --text-muted: #6c5a4a;
  --accent: #8a3c18;
  --accent-soft: #f0dfcf;
  --border: #d9c4af;
  --demo-bg: #f7f2ea;
  --demo-panel: #fffdf9;
  --demo-border: #ddcdbd;
  --demo-accent: #6e4024;
}
.main .block-container {
  padding-top: 1.1rem;
  padding-bottom: 2rem;
  max-width: 1280px;
}
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.85rem;
  box-shadow: 0 4px 14px rgba(31, 19, 8, 0.04);
}
.card-title {
  margin: 0;
  font-size: 0.84rem;
  color: var(--text-muted);
}
.card-value {
  margin: 0.35rem 0 0 0;
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--text-main);
}
.surface-chip {
  display: inline-block;
  padding: 0.28rem 0.58rem;
  border-radius: 8px;
  background: var(--accent-soft);
  color: var(--text-main);
  font-size: 0.82rem;
  margin-bottom: 0.6rem;
  border: 1px solid var(--border);
}
.small-note {
  color: var(--text-muted);
  font-size: 0.92rem;
}
body.surface-demo-mode .stApp {
  background: var(--demo-bg);
  color: var(--text-main);
  font-family: "Avenir Next", "Helvetica Neue", "Nimbus Sans", sans-serif;
}
body.surface-demo-mode [data-testid="stSidebar"] {
  background: #efe6db;
  border-right: 1px solid var(--demo-border);
}
body.surface-demo-mode [data-testid="stSidebar"] * {
  color: var(--text-main);
}
body.surface-demo-mode .demo-page {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
body.surface-demo-mode .demo-hero,
body.surface-demo-mode .demo-panel,
body.surface-demo-mode .slide-slot,
body.surface-demo-mode .demo-story-card,
body.surface-demo-mode .demo-runbook {
  border: 1px solid var(--demo-border);
  border-radius: 10px;
  background: var(--demo-panel);
}
body.surface-demo-mode .demo-hero {
  padding: 1.2rem 1.35rem;
}
body.surface-demo-mode .demo-hero h2 {
  margin: 0;
  font-size: 1.7rem;
}
body.surface-demo-mode .demo-hero p {
  margin: 0.45rem 0 0 0;
  max-width: 820px;
  color: var(--text-muted);
}
body.surface-demo-mode .demo-grid,
body.surface-demo-mode .demo-story-grid {
  display: grid;
  gap: 0.75rem;
}
body.surface-demo-mode .demo-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}
body.surface-demo-mode .demo-story-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 0.8rem;
}
body.surface-demo-mode .demo-panel {
  padding: 1rem 1.05rem;
}
body.surface-demo-mode .demo-panel h3,
body.surface-demo-mode .demo-story-card h4,
body.surface-demo-mode .slide-slot h4,
body.surface-demo-mode .demo-runbook h4 {
  margin: 0 0 0.45rem 0;
  color: var(--text-main);
}
body.surface-demo-mode .demo-panel p,
body.surface-demo-mode .demo-story-card p,
body.surface-demo-mode .slide-slot p,
body.surface-demo-mode .demo-runbook p,
body.surface-demo-mode .demo-runbook li {
  color: var(--text-muted);
}
body.surface-demo-mode .demo-story-card {
  padding: 0.95rem 1rem;
}
body.surface-demo-mode .demo-story-card strong {
  display: block;
  margin-bottom: 0.22rem;
  color: var(--text-main);
}
body.surface-demo-mode .slide-slot {
  padding: 0.95rem 1rem;
  margin-top: 0.8rem;
}
body.surface-demo-mode .slide-slot.is-pending {
  background: #fcf7f0;
}
body.surface-demo-mode .slide-slot img {
  border-radius: 8px;
}
body.surface-demo-mode .slide-status {
  display: inline-block;
  padding: 0.2rem 0.48rem;
  border-radius: 7px;
  border: 1px solid var(--demo-border);
  font-size: 0.8rem;
  color: var(--demo-accent);
  background: #f6eadc;
}
body.surface-demo-mode .slide-copy {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid #eadfd3;
}
body.surface-demo-mode .slide-copy code {
  white-space: pre-wrap;
}
body.surface-demo-mode .demo-runbook {
  padding: 1rem 1.05rem;
}
body.surface-demo-mode .demo-runbook ul {
  padding-left: 1rem;
  margin: 0.4rem 0 0 0;
}

body.surface-product-user-mode .stApp {
  background: #f1f1ed;
  color: #1f2320;
  font-family: "Avenir Next", "Helvetica Neue", "Nimbus Sans", sans-serif;
}
body.surface-product-user-mode [data-testid="stSidebar"] {
  background: #ebece6;
  border-right: 1px solid #d7d9d2;
}
body.surface-product-user-mode [data-testid="stSidebar"] * {
  color: #1f2320;
}
body.surface-product-user-mode .main .block-container {
  max-width: 1160px;
  padding-top: 1.25rem;
  padding-bottom: 2.25rem;
}
body.surface-product-user-mode h1,
body.surface-product-user-mode h2,
body.surface-product-user-mode h3,
body.surface-product-user-mode h4 {
  color: #1f2320;
  letter-spacing: -0.01em;
}
body.surface-product-user-mode p,
body.surface-product-user-mode li,
body.surface-product-user-mode label,
body.surface-product-user-mode .stMarkdown,
body.surface-product-user-mode .stCaption {
  color: #404742;
}
body.surface-product-user-mode div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid #d7d9d2;
  border-radius: 10px;
  background: #ffffff;
  box-shadow: 0 8px 24px rgba(33, 38, 35, 0.04);
}
body.surface-product-user-mode div[data-baseweb="input"] > div,
body.surface-product-user-mode div[data-baseweb="select"] > div,
body.surface-product-user-mode textarea,
body.surface-product-user-mode .stNumberInput input {
  border-radius: 10px !important;
  border-color: #cfd3ca !important;
  background: #fbfbf8 !important;
}
body.surface-product-user-mode div[data-baseweb="input"] > div:focus-within,
body.surface-product-user-mode div[data-baseweb="select"] > div:focus-within,
body.surface-product-user-mode textarea:focus,
body.surface-product-user-mode .stNumberInput input:focus {
  border-color: #35594a !important;
  box-shadow: 0 0 0 1px #35594a !important;
}
body.surface-product-user-mode .stButton > button,
body.surface-product-user-mode .stDownloadButton > button {
  border-radius: 10px;
  border: 1px solid #c6cbc2;
  background: #ffffff;
  color: #21322a;
  min-height: 2.75rem;
  font-weight: 600;
  transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease;
}
body.surface-product-user-mode .stButton > button *,
body.surface-product-user-mode .stDownloadButton > button * {
  color: inherit !important;
}
body.surface-product-user-mode .stButton > button[kind="primary"] {
  background: #35594a;
  border-color: #35594a;
  color: #f7f8f4 !important;
}
body.surface-product-user-mode .stButton > button:hover,
body.surface-product-user-mode .stDownloadButton > button:hover {
  border-color: #aab1a5;
  color: #1f2320;
}
body.surface-product-user-mode .stButton > button[kind="primary"]:hover,
body.surface-product-user-mode .stButton > button[kind="primary"]:focus,
body.surface-product-user-mode .stButton > button[kind="primary"]:focus-visible,
body.surface-product-user-mode .stButton > button[kind="primary"]:active {
  background: #2c4c3f;
  border-color: #2c4c3f;
  color: #f7f8f4 !important;
}
body.surface-product-user-mode .stAlert {
  border-radius: 10px;
}
body.surface-product-user-mode [data-testid="stDataFrame"] {
  border: 1px solid #d7d9d2;
  border-radius: 10px;
  overflow: hidden;
  background: #ffffff;
}
body.surface-product-user-mode .user-page {
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
}
body.surface-product-user-mode .user-header {
  padding: 1.25rem 1.35rem;
  border: 1px solid #d7d9d2;
  border-radius: 10px;
  background: #ffffff;
}
body.surface-product-user-mode .user-header h2 {
  margin: 0;
  font-size: 1.65rem;
}
body.surface-product-user-mode .user-header p {
  margin: 0.45rem 0 0 0;
  max-width: 760px;
  color: #55605a;
}
body.surface-product-user-mode .user-quick-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  margin-top: 0.65rem;
}
body.surface-product-user-mode .user-summary-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.75rem;
  margin: 0.8rem 0 0.4rem 0;
}
body.surface-product-user-mode .user-summary-card {
  border: 1px solid #d7d9d2;
  border-radius: 10px;
  background: #fbfbf8;
  padding: 0.85rem 0.95rem;
}
body.surface-product-user-mode .user-summary-card span {
  display: block;
  color: #5d645f;
  font-size: 0.86rem;
  margin-bottom: 0.2rem;
}
body.surface-product-user-mode .user-summary-card strong {
  color: #1f2320;
  font-size: 1rem;
}
body.surface-product-user-mode .user-metric-rail {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.75rem;
  margin-top: 0.8rem;
}
body.surface-product-user-mode .user-metric {
  border: 1px solid #d7d9d2;
  border-radius: 10px;
  background: #ffffff;
  padding: 0.95rem 1rem;
}
body.surface-product-user-mode .user-metric span {
  display: block;
  color: #58625c;
  font-size: 0.9rem;
  margin-bottom: 0.2rem;
}
body.surface-product-user-mode .user-metric strong {
  color: #1f2320;
  font-size: 1.35rem;
  font-weight: 700;
}
body.surface-product-user-mode .user-step-list {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
}
body.surface-product-user-mode .user-step {
  border: 1px solid #d7d9d2;
  border-radius: 10px;
  background: #ffffff;
  padding: 0.9rem 1rem;
}
body.surface-product-user-mode .user-step strong {
  display: block;
  color: #1f2320;
  margin-bottom: 0.25rem;
}
body.surface-product-user-mode .user-step span {
  color: #5d645f;
  font-size: 0.93rem;
}
body.surface-product-user-mode .user-step.is-active {
  border-color: #35594a;
  box-shadow: inset 0 0 0 1px #35594a;
}
body.surface-product-user-mode .user-step.is-done {
  border-color: #b8c2ba;
  background: #f8f8f4;
}
body.surface-product-user-mode .user-subtle {
  color: #5d645f;
  font-size: 0.94rem;
}
body.surface-product-user-mode .user-surface {
  border: 1px solid #d7d9d2;
  border-radius: 10px;
  background: #ffffff;
  padding: 1.05rem 1.1rem;
}
body.surface-product-user-mode .user-surface h3 {
  margin: 0 0 0.35rem 0;
}
body.surface-product-user-mode .user-surface p {
  margin: 0;
}
body.surface-product-user-mode .user-choice-card {
  border: 1px solid #d7d9d2;
  border-radius: 10px;
  background: #fafaf7;
  padding: 0.95rem 1rem;
  min-height: 138px;
}
body.surface-product-user-mode .user-choice-card h4 {
  margin: 0 0 0.45rem 0;
  font-size: 1rem;
}
body.surface-product-user-mode .user-choice-card p {
  margin: 0.15rem 0;
}
body.surface-product-user-mode .user-review-note {
  border: 1px solid #e1d2ba;
  background: #f5ead7;
  color: #6f5227;
  border-radius: 10px;
  padding: 0.8rem 0.9rem;
}
body.surface-product-user-mode .user-footer-note {
  color: #68706a;
  font-size: 0.92rem;
  padding-top: 0.1rem;
  margin-top: 0.15rem;
}
@media (max-width: 980px) {
  body.surface-demo-mode .demo-grid,
  body.surface-demo-mode .demo-story-grid,
  body.surface-product-user-mode .user-summary-grid,
  body.surface-product-user-mode .user-metric-rail,
  body.surface-product-user-mode .user-step-list {
    grid-template-columns: 1fr 1fr;
  }
}
@media (max-width: 640px) {
  body.surface-demo-mode .demo-grid,
  body.surface-demo-mode .demo-story-grid,
  body.surface-product-user-mode .user-summary-grid,
  body.surface-product-user-mode .user-metric-rail,
  body.surface-product-user-mode .user-step-list {
    grid-template-columns: 1fr;
  }
}
</style>
"""
