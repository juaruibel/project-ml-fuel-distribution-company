"""Day 06 SQL Mirror · Thin compatibility wrapper over day06_sql_store.

Historical callers that import `mirror_day06_bundle` or `get_sql_mirror_target`
continue to work. New code should import from `day06_sql_store` directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.product.day06_sql_store import (
    publish_day06_run_bundle,
    resolve_day06_sql_store_target,
)


def get_sql_mirror_target() -> Path | None:
    """Resolve the optional SQLite target (compatibility alias)."""
    return resolve_day06_sql_store_target()


def mirror_day06_bundle(
    *,
    run_manifest: dict[str, Any],
    normalized_df: pd.DataFrame,
    resumen_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    warning_df: pd.DataFrame | None = None,
) -> str:
    """Mirror one Day 06 operational bundle into the SQL store.

    Returns a short status string for the manifest ``sql_mirror_status`` field.
    """
    result = publish_day06_run_bundle(
        run_manifest=run_manifest,
        normalized_df=normalized_df,
        resumen_df=resumen_df,
        feedback_df=feedback_df,
        warning_df=warning_df,
    )
    return result.get("status", "disabled_unconfigured")
