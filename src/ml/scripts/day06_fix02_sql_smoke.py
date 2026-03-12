"""Day 06.fix02 · SQL Store Smoke Test.

Validates the Day 06 SQL store service:
  1. Initialize store
  2. Publish a synthetic run bundle
  3. Publish feedback
  4. Check table existence
  5. Check view existence
  6. Verify coherent row counts

Output: artifacts/public/validations/day06_fix02/<date>_day06_fix02_sql_smoke_report.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.product.day06_sql_store import (
    get_day06_sql_store_status,
    initialize_day06_sql_store,
    publish_day06_feedback,
    publish_day06_run_bundle,
)

REPORT_DIR = PROJECT_ROOT / "reports" / "validations" / "day06_fix02"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TABLES = {
    "day06_runs",
    "day06_input_rows",
    "day06_event_decisions",
    "day06_feedback",
    "day06_low_confidence_queue",
    "day06_kpis_by_mode",
}

EXPECTED_VIEWS = {
    "vw_day06_runs",
    "vw_day06_event_decisions",
    "vw_day06_feedback",
    "vw_day06_low_confidence_queue",
    "vw_day06_kpis_by_mode",
}


def build_synthetic_bundle(run_id: str = "20260311T170000Z") -> dict:
    """Build a minimal synthetic Day 06 run bundle for testing."""
    normalized_df = pd.DataFrame(
        [
            {
                "event_id": "EVT_001",
                "fecha_evento": "2030-03-11",
                "proveedor_candidato": "SUPPLIER_009",
                "producto_canonico": "PRODUCT_002",
                "terminal_compra": "TERMINAL_001",
                "coste_min_dia_proveedor": 0.90,
                "rank_coste_dia_producto": 1,
                "litros_evento": 33000,
            },
            {
                "event_id": "EVT_001",
                "fecha_evento": "2030-03-11",
                "proveedor_candidato": "SUPPLIER_050",
                "producto_canonico": "PRODUCT_002",
                "terminal_compra": "TERMINAL_001",
                "coste_min_dia_proveedor": 0.92,
                "rank_coste_dia_producto": 2,
                "litros_evento": 33000,
            },
        ]
    )
    resumen_df = pd.DataFrame(
        [
            {
                "event_id": "EVT_001",
                "fecha_evento": "2030-03-11",
                "albaran_id": "ALB_001",
                "recommended_supplier": "SUPPLIER_009",
                "decision_final": "SUPPLIER_009",
                "decision_source": "recommended_rollout",
                "override_reason": "",
                "low_confidence_flag": 0,
                "warning_reasons": "",
                "review_status": "pending_review",
            },
        ]
    )
    warning_df = pd.DataFrame(
        [
            {
                "event_id": "EVT_001",
                "low_confidence_flag": 0,
                "warning_reasons": "",
            },
        ]
    )
    feedback_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "event_id": "EVT_001",
                "inference_mode": "champion_pure",
                "recommended_supplier": "SUPPLIER_009",
                "decision_final": "SUPPLIER_009",
                "feedback_action": "pending_review",
                "override_reason": "",
                "feedback_notes": "",
                "reviewed_at_utc": "",
            },
        ]
    )
    manifest = {
        "run_id": run_id,
        "run_date": "20260311",
        "surface": "smoke_test",
        "input_mode": "csv",
        "inference_mode": "champion_pure",
        "mode_label": "Champion puro (V2_TRANSPORT_ONLY)",
        "mode_rollout_status": "default_recomendado_rollout",
        "top_k": 2,
        "validation_status": "PASS",
        "scoring_status": "SUCCESS",
        "warning_events": 0,
        "sql_mirror_status": "pending",
        "created_at_utc": "2026-03-11T17:00:00Z",
    }
    return {
        "manifest": manifest,
        "normalized_df": normalized_df,
        "resumen_df": resumen_df,
        "warning_df": warning_df,
        "feedback_df": feedback_df,
    }


def main() -> None:
    results: dict = {
        "test": "day06_fix02_sql_smoke",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": {},
    }
    all_passed = True

    # Use a temp file for isolated testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        store_path = Path(tmp.name)

    try:
        # 1. Initialize
        init_result = initialize_day06_sql_store(target_path=store_path)
        init_ok = init_result.get("status") == "initialized"
        results["checks"]["init_store"] = {
            "passed": init_ok,
            "status": init_result.get("status"),
            "tables_created": init_result.get("tables_created"),
            "views_created": init_result.get("views_created"),
        }
        if not init_ok:
            all_passed = False

        # 2. Publish a synthetic run
        bundle = build_synthetic_bundle()
        pub_result = publish_day06_run_bundle(
            run_manifest=bundle["manifest"],
            normalized_df=bundle["normalized_df"],
            resumen_df=bundle["resumen_df"],
            feedback_df=bundle["feedback_df"],
            warning_df=bundle["warning_df"],
            target_path=store_path,
        )
        pub_ok = pub_result.get("status") == "published"
        results["checks"]["publish_run"] = {
            "passed": pub_ok,
            "status": pub_result.get("status"),
            "rows_input": pub_result.get("rows_input"),
            "rows_event_decisions": pub_result.get("rows_event_decisions"),
            "rows_feedback": pub_result.get("rows_feedback"),
        }
        if not pub_ok:
            all_passed = False

        # 3. Publish feedback with override (SUPPLIER_009 → SUPPLIER_050)
        updated_feedback = bundle["feedback_df"].copy()
        updated_feedback["feedback_action"] = "overridden"
        updated_feedback["decision_final"] = "SUPPLIER_050"
        updated_feedback["override_reason"] = "smoke_override_test"
        updated_feedback["reviewed_at_utc"] = "2026-03-11T18:00:00Z"
        fb_result = publish_day06_feedback(
            run_manifest=bundle["manifest"],
            feedback_df=updated_feedback,
            target_path=store_path,
        )
        fb_ok = fb_result.get("status") == "feedback_published"
        results["checks"]["publish_feedback"] = {
            "passed": fb_ok,
            "status": fb_result.get("status"),
            "rows_feedback": fb_result.get("rows_feedback"),
        }
        if not fb_ok:
            all_passed = False

        # 4. Check tables
        status = get_day06_sql_store_status(target_path=store_path)
        missing_tables = set(status.get("missing_tables", []))
        tables_ok = len(missing_tables) == 0
        results["checks"]["tables_exist"] = {
            "passed": tables_ok,
            "tables_found": status.get("tables_found"),
            "tables_expected": status.get("tables_expected"),
            "missing": sorted(missing_tables),
        }
        if not tables_ok:
            all_passed = False

        # 5. Check views
        missing_views = set(status.get("missing_views", []))
        views_ok = len(missing_views) == 0
        results["checks"]["views_exist"] = {
            "passed": views_ok,
            "views_found": status.get("views_found"),
            "views_expected": status.get("views_expected"),
            "missing": sorted(missing_views),
        }
        if not views_ok:
            all_passed = False

        # 6. Row count coherence
        row_counts = status.get("row_counts", {})
        counts_ok = (
            row_counts.get("day06_runs", 0) == 1
            and row_counts.get("day06_input_rows", 0) == 2
            and row_counts.get("day06_event_decisions", 0) == 1
            and row_counts.get("day06_feedback", 0) == 1
            and row_counts.get("day06_kpis_by_mode", 0) == 1
        )
        results["checks"]["row_counts_coherent"] = {
            "passed": counts_ok,
            "row_counts": row_counts,
        }
        if not counts_ok:
            all_passed = False

        # 7. Feedback sync coherence — verify override propagated to event_decisions
        import sqlite3

        conn = sqlite3.connect(store_path)
        try:
            cursor = conn.cursor()
            # Check event_decisions reflects the override
            ed_row = cursor.execute(
                "SELECT decision_final, review_status FROM day06_event_decisions WHERE run_id = ? AND event_id = ?",
                (bundle["manifest"]["run_id"], "EVT_001"),
            ).fetchone()
            ed_decision = ed_row[0] if ed_row else None
            ed_review = ed_row[1] if ed_row else None

            # Check feedback table has the correct data
            fb_row = cursor.execute(
                "SELECT decision_final, feedback_action FROM day06_feedback WHERE run_id = ? AND event_id = ?",
                (bundle["manifest"]["run_id"], "EVT_001"),
            ).fetchone()
            fb_decision = fb_row[0] if fb_row else None
            fb_action = fb_row[1] if fb_row else None

            sync_ok = (
                ed_decision == "SUPPLIER_050"
                and ed_review == "overridden"
                and fb_decision == "SUPPLIER_050"
                and fb_action == "overridden"
            )
            results["checks"]["feedback_sync_coherent"] = {
                "passed": sync_ok,
                "event_decisions_decision_final": ed_decision,
                "event_decisions_review_status": ed_review,
                "feedback_decision_final": fb_decision,
                "feedback_action": fb_action,
                "expected_decision_final": "SUPPLIER_050",
                "expected_review_status": "overridden",
            }
            if not sync_ok:
                all_passed = False
        finally:
            conn.close()

    finally:
        store_path.unlink(missing_ok=True)

    results["overall"] = "PASS" if all_passed else "FAIL"

    # Write report with timestamp to avoid overwriting evidence
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORT_DIR / f"{ts}_day06_fix02_sql_smoke_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(report_path)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
