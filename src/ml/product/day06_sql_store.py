"""Day 06 SQL Store · Persistencia canónica opcional para reporting.

Fuente de verdad primaria = run bundle local (CSV/JSON).
Esta store SQLite local es un backend opcional, estable y BI-ready,
pensado para Tableau u otras herramientas de reporting.

Configuración:
    export DAY06_SQL_STORE_PATH=/path/to/day06_store.db
    # Compatibilidad: DAY06_SQL_MIRROR_PATH sigue funcionando.
    # Si ambos existen, DAY06_SQL_STORE_PATH tiene prioridad.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


_ENV_STORE = "DAY06_SQL_STORE_PATH"
_ENV_MIRROR = "DAY06_SQL_MIRROR_PATH"


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_TABLE_DDL: dict[str, str] = {
    "day06_runs": """
        CREATE TABLE IF NOT EXISTS day06_runs (
            run_id                TEXT PRIMARY KEY,
            run_date              TEXT,
            surface               TEXT,
            input_mode            TEXT,
            inference_mode        TEXT,
            mode_label            TEXT,
            mode_rollout_status   TEXT,
            top_k                 INTEGER,
            validation_status     TEXT,
            scoring_status        TEXT,
            warning_events        INTEGER DEFAULT 0,
            sql_mirror_status     TEXT,
            created_at_utc        TEXT,
            published_at_utc      TEXT
        )
    """,
    "day06_input_rows": """
        CREATE TABLE IF NOT EXISTS day06_input_rows (
            run_id                TEXT NOT NULL,
            row_index             INTEGER NOT NULL,
            event_id              TEXT,
            fecha_evento          TEXT,
            proveedor_candidato   TEXT,
            producto_canonico     TEXT,
            terminal_compra       TEXT,
            coste_min_dia_proveedor REAL,
            rank_coste_dia_producto INTEGER,
            litros_evento         REAL,
            PRIMARY KEY (run_id, row_index)
        )
    """,
    "day06_event_decisions": """
        CREATE TABLE IF NOT EXISTS day06_event_decisions (
            run_id                TEXT NOT NULL,
            event_id              TEXT NOT NULL,
            fecha_evento          TEXT,
            albaran_id            TEXT,
            recommended_supplier  TEXT,
            decision_final        TEXT,
            decision_source       TEXT,
            override_reason       TEXT,
            low_confidence_flag   INTEGER DEFAULT 0,
            warning_reasons       TEXT,
            review_status         TEXT,
            published_at_utc      TEXT,
            PRIMARY KEY (run_id, event_id)
        )
    """,
    "day06_feedback": """
        CREATE TABLE IF NOT EXISTS day06_feedback (
            run_id                TEXT NOT NULL,
            event_id              TEXT NOT NULL,
            inference_mode        TEXT,
            recommended_supplier  TEXT,
            decision_final        TEXT,
            feedback_action       TEXT,
            override_reason       TEXT,
            feedback_notes        TEXT,
            reviewed_at_utc       TEXT,
            published_at_utc      TEXT,
            PRIMARY KEY (run_id, event_id)
        )
    """,
    "day06_low_confidence_queue": """
        CREATE TABLE IF NOT EXISTS day06_low_confidence_queue (
            run_id                TEXT NOT NULL,
            event_id              TEXT NOT NULL,
            fecha_evento          TEXT,
            albaran_id            TEXT,
            recommended_supplier  TEXT,
            decision_final        TEXT,
            warning_reasons       TEXT,
            review_status         TEXT,
            published_at_utc      TEXT,
            PRIMARY KEY (run_id, event_id)
        )
    """,
    "day06_kpis_by_mode": """
        CREATE TABLE IF NOT EXISTS day06_kpis_by_mode (
            run_id                TEXT PRIMARY KEY,
            inference_mode        TEXT,
            warning_events        INTEGER DEFAULT 0,
            rows_input            INTEGER DEFAULT 0,
            rows_event_decisions  INTEGER DEFAULT 0,
            rows_feedback         INTEGER DEFAULT 0,
            published_at_utc      TEXT
        )
    """,
}

_VIEW_DDL: dict[str, str] = {
    "vw_day06_runs": """
        CREATE VIEW IF NOT EXISTS vw_day06_runs AS
        SELECT
            run_id,
            run_date,
            surface,
            input_mode,
            inference_mode,
            mode_label,
            top_k,
            validation_status,
            scoring_status,
            warning_events,
            created_at_utc,
            published_at_utc
        FROM day06_runs
        ORDER BY created_at_utc DESC
    """,
    "vw_day06_event_decisions": """
        CREATE VIEW IF NOT EXISTS vw_day06_event_decisions AS
        SELECT
            d.run_id,
            r.run_date,
            r.inference_mode,
            d.event_id,
            d.fecha_evento,
            d.albaran_id,
            d.recommended_supplier,
            d.decision_final,
            d.decision_source,
            d.override_reason,
            d.low_confidence_flag,
            d.warning_reasons,
            d.review_status
        FROM day06_event_decisions d
        LEFT JOIN day06_runs r ON d.run_id = r.run_id
        ORDER BY r.run_date DESC, d.event_id
    """,
    "vw_day06_feedback": """
        CREATE VIEW IF NOT EXISTS vw_day06_feedback AS
        SELECT
            f.run_id,
            r.run_date,
            f.event_id,
            f.inference_mode,
            f.recommended_supplier,
            f.decision_final,
            f.feedback_action,
            f.override_reason,
            f.feedback_notes,
            f.reviewed_at_utc
        FROM day06_feedback f
        LEFT JOIN day06_runs r ON f.run_id = r.run_id
        ORDER BY r.run_date DESC, f.event_id
    """,
    "vw_day06_low_confidence_queue": """
        CREATE VIEW IF NOT EXISTS vw_day06_low_confidence_queue AS
        SELECT
            q.run_id,
            r.run_date,
            r.inference_mode,
            q.event_id,
            q.fecha_evento,
            q.albaran_id,
            q.recommended_supplier,
            q.decision_final,
            q.warning_reasons,
            q.review_status
        FROM day06_low_confidence_queue q
        LEFT JOIN day06_runs r ON q.run_id = r.run_id
        ORDER BY r.run_date DESC, q.event_id
    """,
    "vw_day06_kpis_by_mode": """
        CREATE VIEW IF NOT EXISTS vw_day06_kpis_by_mode AS
        SELECT
            k.run_id,
            r.run_date,
            k.inference_mode,
            k.warning_events,
            k.rows_input,
            k.rows_event_decisions,
            k.rows_feedback,
            k.published_at_utc
        FROM day06_kpis_by_mode k
        LEFT JOIN day06_runs r ON k.run_id = r.run_id
        ORDER BY r.run_date DESC
    """,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_day06_sql_store_target() -> Path | None:
    """Resolve the canonical SQLite store path from the environment.

    Priority: DAY06_SQL_STORE_PATH > DAY06_SQL_MIRROR_PATH.
    Returns None when neither is configured.
    """
    for env_var in (_ENV_STORE, _ENV_MIRROR):
        raw = os.getenv(env_var, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    return None


def initialize_day06_sql_store(*, target_path: Path | None = None) -> dict[str, Any]:
    """Create or upgrade the Day 06 SQL store with all tables and views.

    Safe to call multiple times (idempotent CREATE IF NOT EXISTS).
    Returns a status dict with backend info and table/view counts.
    """
    resolved = target_path or resolve_day06_sql_store_target()
    if resolved is None:
        return {
            "status": "disabled_unconfigured",
            "backend": "none",
            "target": None,
            "tables_created": 0,
            "views_created": 0,
        }

    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    try:
        cursor = conn.cursor()
        for ddl in _TABLE_DDL.values():
            cursor.execute(ddl)
        # Drop and recreate views to pick up schema changes
        for view_name, view_ddl in _VIEW_DDL.items():
            cursor.execute(f"DROP VIEW IF EXISTS {view_name}")
            cursor.execute(view_ddl)
        conn.commit()
    finally:
        conn.close()

    return {
        "status": "initialized",
        "backend": "sqlite",
        "target": str(resolved),
        "tables_created": len(_TABLE_DDL),
        "views_created": len(_VIEW_DDL),
    }


def publish_day06_run_bundle(
    *,
    run_manifest: dict[str, Any],
    normalized_df: pd.DataFrame,
    resumen_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    warning_df: pd.DataFrame | None = None,
    target_path: Path | None = None,
) -> dict[str, Any]:
    """Publish one complete Day 06 run bundle to the SQL store.

    Idempotent: re-publishing the same run_id replaces all rows.
    Returns a status dict with backend, target and row counts.
    """
    resolved = target_path or resolve_day06_sql_store_target()
    if resolved is None:
        return _disabled_result()

    if not resolved.exists():
        initialize_day06_sql_store(target_path=resolved)

    run_id = str(run_manifest.get("run_id", ""))
    now_utc = _utc_now()
    conn = sqlite3.connect(resolved)
    try:
        # 1. day06_runs
        _upsert_run(conn, run_manifest, now_utc)

        # 2. day06_input_rows
        _replace_input_rows(conn, run_id, normalized_df, now_utc)

        # 3. day06_event_decisions
        _replace_event_decisions(conn, run_id, resumen_df, now_utc)

        # 4. day06_feedback
        _replace_feedback(conn, run_id, feedback_df, now_utc)

        # 5. day06_low_confidence_queue
        _replace_low_confidence_queue(conn, run_id, resumen_df, warning_df, now_utc)

        # 6. day06_kpis_by_mode
        _replace_kpis(conn, run_id, run_manifest, normalized_df, resumen_df, feedback_df, now_utc)

        conn.commit()
    finally:
        conn.close()

    return {
        "status": "published",
        "backend": "sqlite",
        "target": str(resolved),
        "run_id": run_id,
        "published_at_utc": now_utc,
        "rows_input": len(normalized_df),
        "rows_event_decisions": len(resumen_df),
        "rows_feedback": len(feedback_df),
    }


def publish_day06_feedback(
    *,
    run_manifest: dict[str, Any],
    feedback_df: pd.DataFrame,
    target_path: Path | None = None,
) -> dict[str, Any]:
    """Re-publish feedback rows for an existing run (after human review).

    Also propagates overrides into event_decisions and low_confidence_queue
    so that BI views stay in sync, and updates the KPI feedback count.
    """
    resolved = target_path or resolve_day06_sql_store_target()
    if resolved is None:
        return _disabled_result()

    if not resolved.exists():
        initialize_day06_sql_store(target_path=resolved)

    run_id = str(run_manifest.get("run_id", ""))
    now_utc = _utc_now()
    conn = sqlite3.connect(resolved)
    try:
        _replace_feedback(conn, run_id, feedback_df, now_utc)

        # Propagate overrides to event_decisions and low_confidence_queue
        _sync_overrides_from_feedback(conn, run_id, feedback_df, now_utc)

        # Update KPI feedback count
        conn.execute(
            "UPDATE day06_kpis_by_mode SET rows_feedback = ?, published_at_utc = ? WHERE run_id = ?",
            (len(feedback_df), now_utc, run_id),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "status": "feedback_published",
        "backend": "sqlite",
        "target": str(resolved),
        "run_id": run_id,
        "published_at_utc": now_utc,
        "rows_feedback": len(feedback_df),
    }


def get_day06_sql_store_status(*, target_path: Path | None = None) -> dict[str, Any]:
    """Return the current status of the SQL store: backend, tables, row counts."""
    resolved = target_path or resolve_day06_sql_store_target()
    if resolved is None:
        return {
            "status": "disabled_unconfigured",
            "backend": "none",
            "target": None,
            "configured_env_var": _detect_configured_env_var(),
        }

    if not resolved.exists():
        return {
            "status": "configured_not_initialized",
            "backend": "sqlite",
            "target": str(resolved),
            "configured_env_var": _detect_configured_env_var(),
        }

    conn = sqlite3.connect(resolved)
    try:
        cursor = conn.cursor()
        tables = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        views = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='view'"
            ).fetchall()
        }
        expected_tables = set(_TABLE_DDL.keys())
        expected_views = set(_VIEW_DDL.keys())
        missing_tables = sorted(expected_tables - tables)
        missing_views = sorted(expected_views - views)

        row_counts: dict[str, int] = {}
        for table_name in sorted(expected_tables & tables):
            count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            row_counts[table_name] = int(count)

        runs_total = row_counts.get("day06_runs", 0)
    finally:
        conn.close()

    schema_ok = not missing_tables and not missing_views
    return {
        "status": "ready" if schema_ok else "schema_incomplete",
        "backend": "sqlite",
        "target": str(resolved),
        "configured_env_var": _detect_configured_env_var(),
        "tables_found": len(expected_tables & tables),
        "tables_expected": len(expected_tables),
        "views_found": len(expected_views & views),
        "views_expected": len(expected_views),
        "missing_tables": missing_tables,
        "missing_views": missing_views,
        "row_counts": row_counts,
        "runs_total": runs_total,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _detect_configured_env_var() -> str:
    if os.getenv(_ENV_STORE, "").strip():
        return _ENV_STORE
    if os.getenv(_ENV_MIRROR, "").strip():
        return _ENV_MIRROR
    return "none"


def _disabled_result() -> dict[str, Any]:
    return {
        "status": "disabled_unconfigured",
        "backend": "none",
        "target": None,
    }


def _serialize(value: object) -> object:
    """Normalize scalars before writing them to SQLite."""
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _upsert_run(conn: sqlite3.Connection, manifest: dict[str, Any], now_utc: str) -> None:
    run_id = str(manifest.get("run_id", ""))
    conn.execute("DELETE FROM day06_runs WHERE run_id = ?", (run_id,))
    conn.execute(
        """INSERT INTO day06_runs (
            run_id, run_date, surface, input_mode, inference_mode,
            mode_label, mode_rollout_status, top_k,
            validation_status, scoring_status, warning_events,
            sql_mirror_status, created_at_utc, published_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            str(manifest.get("run_date", "")),
            str(manifest.get("surface", "")),
            str(manifest.get("input_mode", "")),
            str(manifest.get("inference_mode", "")),
            str(manifest.get("mode_label", "")),
            str(manifest.get("mode_rollout_status", "")),
            int(manifest.get("top_k", 0)),
            str(manifest.get("validation_status", "")),
            str(manifest.get("scoring_status", "")),
            int(manifest.get("warning_events", 0)),
            str(manifest.get("sql_mirror_status", "")),
            str(manifest.get("created_at_utc", "")),
            now_utc,
        ),
    )


def _replace_input_rows(
    conn: sqlite3.Connection,
    run_id: str,
    normalized_df: pd.DataFrame,
    now_utc: str,
) -> None:
    conn.execute("DELETE FROM day06_input_rows WHERE run_id = ?", (run_id,))
    if normalized_df.empty:
        return
    target_cols = [
        "event_id", "fecha_evento", "proveedor_candidato", "producto_canonico",
        "terminal_compra", "coste_min_dia_proveedor", "rank_coste_dia_producto",
        "litros_evento",
    ]
    for idx, row in normalized_df.iterrows():
        values = [run_id, int(idx)]
        for col in target_cols:
            values.append(_serialize(row.get(col)))
        conn.execute(
            f"INSERT INTO day06_input_rows (run_id, row_index, {', '.join(target_cols)}) "
            f"VALUES ({', '.join(['?'] * (2 + len(target_cols)))})",
            tuple(values),
        )


def _replace_event_decisions(
    conn: sqlite3.Connection,
    run_id: str,
    resumen_df: pd.DataFrame,
    now_utc: str,
) -> None:
    conn.execute("DELETE FROM day06_event_decisions WHERE run_id = ?", (run_id,))
    if resumen_df.empty:
        return
    for _, row in resumen_df.iterrows():
        conn.execute(
            """INSERT INTO day06_event_decisions (
                run_id, event_id, fecha_evento, albaran_id,
                recommended_supplier, decision_final, decision_source,
                override_reason, low_confidence_flag, warning_reasons,
                review_status, published_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                _serialize(row.get("event_id")),
                _serialize(row.get("fecha_evento")),
                _serialize(row.get("albaran_id")),
                _serialize(row.get("recommended_supplier")),
                _serialize(row.get("decision_final")),
                _serialize(row.get("decision_source")),
                _serialize(row.get("override_reason")),
                int(pd.to_numeric(row.get("low_confidence_flag", 0), errors="coerce") or 0),
                _serialize(row.get("warning_reasons")),
                _serialize(row.get("review_status")),
                now_utc,
            ),
        )


def _replace_feedback(
    conn: sqlite3.Connection,
    run_id: str,
    feedback_df: pd.DataFrame,
    now_utc: str,
) -> None:
    conn.execute("DELETE FROM day06_feedback WHERE run_id = ?", (run_id,))
    if feedback_df.empty:
        return
    for _, row in feedback_df.iterrows():
        conn.execute(
            """INSERT INTO day06_feedback (
                run_id, event_id, inference_mode,
                recommended_supplier, decision_final,
                feedback_action, override_reason,
                feedback_notes, reviewed_at_utc, published_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                _serialize(row.get("event_id")),
                _serialize(row.get("inference_mode")),
                _serialize(row.get("recommended_supplier")),
                _serialize(row.get("decision_final")),
                _serialize(row.get("feedback_action")),
                _serialize(row.get("override_reason")),
                _serialize(row.get("feedback_notes")),
                _serialize(row.get("reviewed_at_utc")),
                now_utc,
            ),
        )


def _replace_low_confidence_queue(
    conn: sqlite3.Connection,
    run_id: str,
    resumen_df: pd.DataFrame,
    warning_df: pd.DataFrame | None,
    now_utc: str,
) -> None:
    conn.execute("DELETE FROM day06_low_confidence_queue WHERE run_id = ?", (run_id,))

    # Derive low-confidence events from resumen + warning
    source_df = resumen_df.copy()
    if "low_confidence_flag" not in source_df.columns and isinstance(warning_df, pd.DataFrame) and not warning_df.empty:
        if "event_id" in source_df.columns and "event_id" in warning_df.columns:
            flag_cols = warning_df[["event_id", "low_confidence_flag", "warning_reasons"]].copy()
            source_df = source_df.merge(flag_cols, on="event_id", how="left")

    if "low_confidence_flag" not in source_df.columns:
        return

    source_df["low_confidence_flag"] = pd.to_numeric(
        source_df["low_confidence_flag"], errors="coerce"
    ).fillna(0).astype(int)
    flagged = source_df[source_df["low_confidence_flag"] == 1].copy()
    if flagged.empty:
        return

    for _, row in flagged.iterrows():
        conn.execute(
            """INSERT INTO day06_low_confidence_queue (
                run_id, event_id, fecha_evento, albaran_id,
                recommended_supplier, decision_final,
                warning_reasons, review_status, published_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                _serialize(row.get("event_id")),
                _serialize(row.get("fecha_evento")),
                _serialize(row.get("albaran_id")),
                _serialize(row.get("recommended_supplier")),
                _serialize(row.get("decision_final")),
                _serialize(row.get("warning_reasons")),
                _serialize(row.get("review_status")),
                now_utc,
            ),
        )


def _replace_kpis(
    conn: sqlite3.Connection,
    run_id: str,
    manifest: dict[str, Any],
    normalized_df: pd.DataFrame,
    resumen_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    now_utc: str,
) -> None:
    conn.execute("DELETE FROM day06_kpis_by_mode WHERE run_id = ?", (run_id,))
    conn.execute(
        """INSERT INTO day06_kpis_by_mode (
            run_id, inference_mode, warning_events,
            rows_input, rows_event_decisions, rows_feedback,
            published_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            str(manifest.get("inference_mode", "")),
            int(manifest.get("warning_events", 0)),
            len(normalized_df),
            len(resumen_df),
            len(feedback_df),
            now_utc,
        ),
    )


def _sync_overrides_from_feedback(
    conn: sqlite3.Connection,
    run_id: str,
    feedback_df: pd.DataFrame,
    now_utc: str,
) -> None:
    """Propagate feedback overrides into event_decisions and low_confidence_queue.

    Only updates decision_final, override_reason, and review_status —
    preserves original scoring fields (recommended_supplier, decision_source, etc.).
    """
    if feedback_df.empty:
        return
    for _, row in feedback_df.iterrows():
        event_id = _serialize(row.get("event_id"))
        if event_id is None:
            continue
        decision_final = _serialize(row.get("decision_final"))
        override_reason = _serialize(row.get("override_reason"))
        feedback_action = _serialize(row.get("feedback_action"))
        # Map feedback_action to review_status
        review_status = feedback_action if feedback_action else "pending_review"

        # Update event_decisions
        conn.execute(
            """UPDATE day06_event_decisions
               SET decision_final = ?,
                   override_reason = ?,
                   review_status = ?,
                   published_at_utc = ?
               WHERE run_id = ? AND event_id = ?""",
            (decision_final, override_reason, review_status, now_utc, run_id, event_id),
        )
        # Update low_confidence_queue (if event exists there)
        conn.execute(
            """UPDATE day06_low_confidence_queue
               SET decision_final = ?,
                   review_status = ?,
                   published_at_utc = ?
               WHERE run_id = ? AND event_id = ?""",
            (decision_final, review_status, now_utc, run_id, event_id),
        )

