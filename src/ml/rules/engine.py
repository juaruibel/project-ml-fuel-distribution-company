#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ml.metrics.postinference_metrics import compute_postinference_metrics
from src.ml.rules.albaran_coherence import apply_albaran_policy, build_resumen_albaran
from src.ml.rules.blocklist import apply_blocklist_candidates, load_blocklist_rules
from src.ml.shared.helpers import build_run_id, ensure_parent, utc_now_iso, write_json

REQUIRED_COLUMNS = (
    "event_id",
    "fecha_evento",
    "producto_canonico",
    "terminal_compra",
    "proveedor_candidato",
    "score_model",
    "rank_event_score",
)

ALBARAN_POLICY_CHOICES = ("none", "PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009")


def parse_args() -> argparse.Namespace:
    """Parsea CLI para motor de reglas post-inferencia."""
    parser = argparse.ArgumentParser(
        description="Aplica reglas deterministas post-inferencia sobre CSV de recomendación."
    )
    parser.add_argument("--input-csv", required=True, help="Ruta CSV de inferencia de entrada.")
    parser.add_argument("--output-csv", required=True, help="Ruta CSV de salida detalle (grano candidato).")
    parser.add_argument(
        "--output-resumen-csv",
        default="",
        help="Ruta CSV opcional de resumen por evento.",
    )
    parser.add_argument(
        "--output-resumen-albaran-csv",
        default="",
        help="Ruta CSV opcional de resumen por albarán.",
    )
    parser.add_argument(
        "--rules-csv",
        default="config/business_blocklist_rules.csv",
        help="Ruta CSV de reglas de negocio.",
    )
    parser.add_argument(
        "--mode",
        default="shadow",
        choices=["shadow", "assist"],
        help="Modo de ejecución: shadow (sin cambiar decisión) o assist (ajusta decisión).",
    )
    parser.add_argument(
        "--albaran-policy",
        default="none",
        choices=list(ALBARAN_POLICY_CHOICES),
        help="Política de coherencia por albarán (solo aplica en mode=assist).",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Ruta opcional para guardar resumen de ejecución en JSON.",
    )
    return parser.parse_args()


def _validate_input_schema(df: pd.DataFrame) -> None:
    """Valida columnas mínimas requeridas del input."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "CSV de inferencia incompatible para rules engine. "
            f"Faltan columnas requeridas: {missing}"
        )


def _prepare_detail_output(output_df: pd.DataFrame, run_id: str, ts_utc: str) -> pd.DataFrame:
    """Normaliza salida detalle y añade trazabilidad de corrida."""
    detail_df = output_df.copy()
    detail_df["blocked_by_rule_candidate"] = pd.to_numeric(
        detail_df.get("blocked_by_rule_candidate", 0), errors="coerce"
    ).fillna(0).astype(int)
    detail_df["block_reason_candidate"] = (
        detail_df.get("block_reason_candidate", "")
        .fillna("")
        .astype(str)
        .str.strip()
    )
    detail_df["blocked_by_rule"] = detail_df["blocked_by_rule_candidate"].astype(int)
    detail_df["block_reason"] = detail_df["block_reason_candidate"].astype(str)
    detail_df["run_id"] = run_id
    detail_df["ts_utc"] = ts_utc
    return detail_df


def _coerce_numeric_rank_and_score(working: pd.DataFrame) -> pd.DataFrame:
    working = working.copy()
    working["score_model"] = pd.to_numeric(working["score_model"], errors="coerce")
    working["rank_event_score"] = pd.to_numeric(working["rank_event_score"], errors="coerce")
    missing_rank = working["rank_event_score"].isna()
    if missing_rank.any():
        working.loc[missing_rank, "rank_event_score"] = (
            working.groupby("event_id")["score_model"].rank(method="first", ascending=False)
        )
    working["rank_event_score"] = working["rank_event_score"].astype(int)
    return working


def _build_event_summary(detail_df: pd.DataFrame, mode: str, run_id: str, ts_utc: str) -> pd.DataFrame:
    """Construye resumen a grano evento con trazabilidad pre/post política."""
    required_cols = {
        "event_id",
        "proveedor_candidato",
        "rank_event_score",
        "score_model",
        "blocked_by_rule",
        "fecha_evento",
        "albaran_id",
        "producto_canonico",
        "terminal_compra",
    }
    missing = required_cols - set(detail_df.columns)
    if missing:
        raise ValueError(f"No se puede construir resumen por evento. Faltan columnas: {sorted(missing)}")

    working = detail_df.copy()
    for column in ["event_id", "proveedor_candidato", "fecha_evento", "albaran_id", "producto_canonico", "terminal_compra"]:
        working[column] = working[column].astype(str).str.strip()

    working["blocked_by_rule"] = pd.to_numeric(working["blocked_by_rule"], errors="coerce").fillna(0).astype(int)
    working = _coerce_numeric_rank_and_score(working)

    records: list[dict[str, object]] = []
    for event_id, group in working.groupby("event_id", sort=False):
        g = group.sort_values(["rank_event_score", "score_model"], ascending=[True, False], kind="mergesort")
        top_model_row = g.iloc[0]
        recommended_supplier = str(top_model_row["proveedor_candidato"]).strip()
        blocked_candidates_count = int(g["blocked_by_rule"].sum())

        decision_pre_policy = recommended_supplier
        decision_final = recommended_supplier
        decision_source = "model"
        override_reason = ""

        if mode == "assist":
            unblocked = g[g["blocked_by_rule"] == 0]
            if unblocked.empty:
                decision_source = "model_fallback"
                override_reason = "all_candidates_blocked_fallback_model"
            else:
                best_unblocked = str(unblocked.iloc[0]["proveedor_candidato"]).strip()
                decision_pre_policy = best_unblocked
                decision_final = best_unblocked
                if best_unblocked != recommended_supplier:
                    decision_source = "rules_blocklist"
                    override_reason = "rule_block_top1"

        records.append(
            {
                "event_id": str(event_id),
                "fecha_evento": str(top_model_row.get("fecha_evento", "")).strip(),
                "albaran_id": str(top_model_row.get("albaran_id", "")).strip(),
                "linea_id": str(top_model_row.get("linea_id", "")).strip(),
                "producto_canonico": str(top_model_row.get("producto_canonico", "")).strip(),
                "terminal_compra": str(top_model_row.get("terminal_compra", "")).strip(),
                "recommended_supplier": recommended_supplier,
                "decision_pre_policy": decision_pre_policy,
                "decision_final": decision_final,
                "decision_source": decision_source,
                "override_reason": override_reason,
                "policy_applied_event": 0,
                "policy_rule_id": "",
                "policy_reason_event": "",
                "blocked_candidates_count": blocked_candidates_count,
                "run_id": run_id,
                "ts_utc": ts_utc,
            }
        )

    output = pd.DataFrame(records)
    if output.empty:
        return output

    output["policy_applied_event"] = pd.to_numeric(output["policy_applied_event"], errors="coerce").fillna(0).astype(int)
    return output


def _build_summary(
    detail_df: pd.DataFrame,
    resumen_df: pd.DataFrame,
    resumen_albaran_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    input_path: Path,
    output_path: Path,
    output_resumen_path: Path | None,
    output_resumen_albaran_path: Path | None,
    rules_path: Path,
    mode: str,
    albaran_policy: str,
    run_id: str,
    ts_utc: str,
) -> dict:
    """Construye resumen operativo de la corrida."""
    blocked = pd.to_numeric(detail_df.get("blocked_by_rule_candidate", 0), errors="coerce").fillna(0)
    rows_total = int(len(detail_df))
    rows_blocked = int(blocked.sum())

    events_total = int(detail_df["event_id"].astype(str).nunique()) if "event_id" in detail_df.columns else None
    events_blocked = (
        int(detail_df.loc[blocked > 0, "event_id"].astype(str).nunique())
        if "event_id" in detail_df.columns
        else None
    )

    reason_counts = (
        detail_df.loc[blocked > 0, "block_reason_candidate"].fillna("").astype(str).value_counts().to_dict()
        if "block_reason_candidate" in detail_df.columns
        else {}
    )

    decision_source_counts = (
        resumen_df["decision_source"].astype(str).value_counts().to_dict()
        if not resumen_df.empty and "decision_source" in resumen_df.columns
        else {}
    )

    overrides_count = (
        int((resumen_df["decision_final"].astype(str).str.strip() != resumen_df["recommended_supplier"].astype(str).str.strip()).sum())
        if not resumen_df.empty and {"decision_final", "recommended_supplier"}.issubset(resumen_df.columns)
        else 0
    )

    overrides_policy_count = (
        int((resumen_df["decision_final"].astype(str).str.strip() != resumen_df["decision_pre_policy"].astype(str).str.strip()).sum())
        if not resumen_df.empty and {"decision_final", "decision_pre_policy"}.issubset(resumen_df.columns)
        else 0
    )

    summary: dict[str, object] = {
        "run_id": run_id,
        "mode": mode,
        "albaran_policy": albaran_policy,
        "ts_utc": ts_utc,
        "input_csv": str(input_path),
        "output_detalle_csv": str(output_path),
        "output_resumen_csv": str(output_resumen_path) if output_resumen_path is not None else None,
        "output_resumen_albaran_csv": (
            str(output_resumen_albaran_path) if output_resumen_albaran_path is not None else None
        ),
        "rules_csv": str(rules_path),
        "rules_active": int(len(rules_df)),
        "rows_total": rows_total,
        "rows_blocked": rows_blocked,
        "rows_blocked_pct": float(rows_blocked / rows_total) if rows_total else 0.0,
        "events_total": events_total,
        "events_blocked": events_blocked,
        "events_blocked_pct": (
            float(events_blocked / events_total)
            if events_total not in (None, 0) and events_blocked is not None
            else None
        ),
        "reason_counts": reason_counts,
        "decision_source_counts": decision_source_counts,
        "overrides_count": overrides_count,
        "overrides_policy_count": overrides_policy_count,
        "policy_applied_events": (
            int(pd.to_numeric(resumen_df.get("policy_applied_event", 0), errors="coerce").fillna(0).astype(int).sum())
            if not resumen_df.empty
            else 0
        ),
        "policy_applied_albaranes": (
            int(pd.to_numeric(resumen_albaran_df.get("policy_applied", 0), errors="coerce").fillna(0).astype(int).sum())
            if not resumen_albaran_df.empty
            else 0
        ),
    }

    if not resumen_albaran_df.empty:
        cb = pd.to_numeric(resumen_albaran_df.get("coherence_before"), errors="coerce")
        ca = pd.to_numeric(resumen_albaran_df.get("coherence_after"), errors="coerce")
        valid = cb.notna() & ca.notna()
        summary["pair_groups_PRODUCT_002_PRODUCT_003"] = int(valid.sum())
        summary["coherence_before"] = float(cb[valid].mean()) if valid.any() else None
        summary["coherence_after"] = float(ca[valid].mean()) if valid.any() else None
        if summary.get("coherence_before") is not None and summary.get("coherence_after") is not None:
            summary["coherence_delta"] = float(summary["coherence_after"] - summary["coherence_before"])
        else:
            summary["coherence_delta"] = None
    else:
        summary["pair_groups_PRODUCT_002_PRODUCT_003"] = 0
        summary["coherence_before"] = None
        summary["coherence_after"] = None
        summary["coherence_delta"] = None

    has_targets = "target_elegido" in detail_df.columns
    if has_targets:
        try:
            day03_metrics = compute_postinference_metrics(
                detail_df=detail_df,
                resumen_df=resumen_df,
                resumen_albaran_df=resumen_albaran_df,
            )
            summary.update(day03_metrics)
        except Exception as metrics_exc:
            summary["metrics_warning"] = f"No se pudieron calcular métricas Day03: {metrics_exc}"

    return summary


def run(
    input_csv: Path,
    output_csv: Path,
    rules_csv: Path = Path("config/business_blocklist_rules.csv"),
    mode: str = "shadow",
    albaran_policy: str = "none",
    summary_json: Path | None = None,
    output_resumen_csv: Path | None = None,
    output_resumen_albaran_csv: Path | None = None,
) -> dict:
    """Ejecuta reglas post-inferencia y persiste salidas de auditoría."""
    if mode not in {"shadow", "assist"}:
        raise ValueError(f"Modo no soportado: {mode}")
    if albaran_policy not in ALBARAN_POLICY_CHOICES:
        raise ValueError(f"Política de albarán no soportada: {albaran_policy}")

    if not input_csv.exists():
        raise FileNotFoundError(f"No existe input CSV: {input_csv}")
    if not rules_csv.exists():
        raise FileNotFoundError(f"No existe rules CSV: {rules_csv}")

    input_df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    _validate_input_schema(input_df)

    run_id = build_run_id()
    ts_utc = utc_now_iso()

    rules_df = load_blocklist_rules(rules_csv)
    output_df = apply_blocklist_candidates(input_df, rules_df)
    detail_df = _prepare_detail_output(output_df=output_df, run_id=run_id, ts_utc=ts_utc)

    resumen_df = _build_event_summary(detail_df=detail_df, mode=mode, run_id=run_id, ts_utc=ts_utc)
    if mode == "assist" and albaran_policy != "none" and not resumen_df.empty:
        resumen_df = apply_albaran_policy(
            resumen_df=resumen_df,
            detail_df=detail_df,
            albaran_policy=albaran_policy,
        )

    resumen_albaran_df = build_resumen_albaran(resumen_df=resumen_df, run_id=run_id, ts_utc=ts_utc)

    ensure_parent(output_csv)
    detail_df.to_csv(output_csv, index=False)

    if output_resumen_csv is not None:
        ensure_parent(output_resumen_csv)
        resumen_df.to_csv(output_resumen_csv, index=False)

    if output_resumen_albaran_csv is not None:
        ensure_parent(output_resumen_albaran_csv)
        resumen_albaran_df.to_csv(output_resumen_albaran_csv, index=False)

    summary = _build_summary(
        detail_df=detail_df,
        resumen_df=resumen_df,
        resumen_albaran_df=resumen_albaran_df,
        rules_df=rules_df,
        input_path=input_csv,
        output_path=output_csv,
        output_resumen_path=output_resumen_csv,
        output_resumen_albaran_path=output_resumen_albaran_csv,
        rules_path=rules_csv,
        mode=mode,
        albaran_policy=albaran_policy,
        run_id=run_id,
        ts_utc=ts_utc,
    )

    if summary_json is not None:
        write_json(summary_json, summary)

    return summary


def main() -> None:
    """Entrada CLI para `python -m src.ml.rules.engine`."""
    args = parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_resumen_csv = Path(args.output_resumen_csv).resolve() if args.output_resumen_csv else None
    output_resumen_albaran_csv = (
        Path(args.output_resumen_albaran_csv).resolve() if args.output_resumen_albaran_csv else None
    )
    rules_csv = Path(args.rules_csv).resolve()
    summary_json = Path(args.summary_json).resolve() if args.summary_json else None

    summary = run(
        input_csv=input_csv,
        output_csv=output_csv,
        output_resumen_csv=output_resumen_csv,
        output_resumen_albaran_csv=output_resumen_albaran_csv,
        rules_csv=rules_csv,
        mode=args.mode,
        albaran_policy=args.albaran_policy,
        summary_json=summary_json,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
