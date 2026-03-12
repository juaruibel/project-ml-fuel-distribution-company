from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _load_inputs(fact_compras_path: Path, ofertas_typed_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not fact_compras_path.exists():
        raise FileNotFoundError(f"No existe fact compras: {fact_compras_path}")
    if not ofertas_typed_path.exists():
        raise FileNotFoundError(f"No existe ofertas typed: {ofertas_typed_path}")

    compras = pd.read_csv(fact_compras_path, dtype=str, keep_default_na=False)
    ofertas = pd.read_csv(ofertas_typed_path, dtype=str, keep_default_na=False)
    if "dq_status" in ofertas.columns:
        ofertas = ofertas[ofertas["dq_status"] == "valid"].copy()
    return compras, ofertas


def _build_bridge_candidates(compras: pd.DataFrame, ofertas: pd.DataFrame) -> pd.DataFrame:
    compras = compras.copy().reset_index(drop=True)
    compras["row_id"] = compras.index.astype(int)
    compras["precio_unitario"] = pd.to_numeric(compras.get("precio_unitario", ""), errors="coerce")
    compras["precio_x1000"] = compras["precio_unitario"] * 1000

    ofertas = ofertas.copy()
    ofertas["coste_min"] = pd.to_numeric(ofertas.get("coste_min", ""), errors="coerce")

    compras_base = compras.rename(columns={"fecha_compra": "fecha_oferta"})[
        [
            "row_id",
            "fecha_oferta",
            "producto_canonico",
            "proveedor_canonico",
            "terminal_canonico",
            "precio_x1000",
        ]
    ].rename(columns={"terminal_canonico": "terminal_compra"})

    ofertas_base = ofertas[
        [
            "fecha_oferta",
            "producto_canonico",
            "proveedor_canonico",
            "terminal_canonico",
            "coste_min",
        ]
    ].rename(columns={"terminal_canonico": "terminal_oferta"})

    merged = compras_base.merge(
        ofertas_base,
        on=["fecha_oferta", "producto_canonico", "proveedor_canonico"],
        how="left",
    )
    merged["delta_abs"] = (merged["coste_min"] - merged["precio_x1000"]).abs()
    merged["delta_sort"] = merged["delta_abs"].fillna(10**12)
    return merged


def _summarize_terminals(best_match: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    rows: list[dict] = []
    matched = best_match[best_match["terminal_oferta"].notna()].copy()

    for terminal_compra, frame in matched.groupby("terminal_compra", dropna=False):
        shares = frame["terminal_oferta"].value_counts(normalize=True)
        counts = frame["terminal_oferta"].value_counts()

        top1_terminal = shares.index[0] if len(shares) >= 1 else ""
        top1_share = float(shares.iloc[0]) if len(shares) >= 1 else 0.0
        top1_count = int(counts.iloc[0]) if len(counts) >= 1 else 0

        top2_terminal = shares.index[1] if len(shares) >= 2 else ""
        top2_share = float(shares.iloc[1]) if len(shares) >= 2 else 0.0
        top2_count = int(counts.iloc[1]) if len(counts) >= 2 else 0

        suggested_terminal = top1_terminal if top1_share >= min_confidence else ""
        suggestion_status = "high_confidence" if suggested_terminal else "ambiguous"

        rows.append(
            {
                "terminal_compra": terminal_compra,
                "matched_rows": int(len(frame)),
                "top1_terminal_oferta": top1_terminal,
                "top1_share": top1_share,
                "top1_count": top1_count,
                "top2_terminal_oferta": top2_terminal,
                "top2_share": top2_share,
                "top2_count": top2_count,
                "suggested_terminal_oferta": suggested_terminal,
                "suggestion_status": suggestion_status,
                "median_delta_abs": float(frame["delta_abs"].median()),
                "p90_delta_abs": float(frame["delta_abs"].quantile(0.90)),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["terminal_compra"]).reset_index(drop=True)


def _to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Sin datos para tabla de mapeo._"

    lines = [
        "| terminal_compra | matched_rows | top1_terminal_oferta | top1_share | top2_terminal_oferta | top2_share | suggested_terminal_oferta | suggestion_status |",
        "|---|---:|---|---:|---|---:|---|---|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| {terminal_compra} | {matched_rows} | {top1_terminal_oferta} | {top1_share:.4f} | {top2_terminal_oferta} | {top2_share:.4f} | {suggested_terminal_oferta} | {suggestion_status} |".format(
                terminal_compra=row["terminal_compra"],
                matched_rows=int(row["matched_rows"]),
                top1_terminal_oferta=row["top1_terminal_oferta"],
                top1_share=float(row["top1_share"]),
                top2_terminal_oferta=row["top2_terminal_oferta"],
                top2_share=float(row["top2_share"]),
                suggested_terminal_oferta=row["suggested_terminal_oferta"] or "-",
                suggestion_status=row["suggestion_status"],
            )
        )
    return "\n".join(lines)


def run(
    fact_compras_path: Path,
    ofertas_typed_path: Path,
    output_md_path: Path,
    output_csv_path: Path,
    min_confidence: float = 0.65,
    run_id: str | None = None,
) -> dict:
    execution_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_ts_utc = datetime.now(timezone.utc).isoformat()

    compras, ofertas = _load_inputs(fact_compras_path=fact_compras_path, ofertas_typed_path=ofertas_typed_path)
    merged = _build_bridge_candidates(compras=compras, ofertas=ofertas)
    best_match = merged.sort_values(["row_id", "delta_sort"]).drop_duplicates("row_id", keep="first")
    coverage_ratio = float(best_match["terminal_oferta"].notna().mean()) if len(best_match) else 0.0

    terminal_summary = _summarize_terminals(best_match=best_match, min_confidence=min_confidence)
    confident_rows = terminal_summary[terminal_summary["suggestion_status"] == "high_confidence"]

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_summary.to_csv(output_csv_path, index=False, encoding="utf-8")

    md_content = "\n".join(
        [
            "# Day02 · Diagnóstico terminal bridge (TERM -> CLH)",
            "",
            f"- run_id: `{execution_run_id}`",
            f"- generated_ts_utc: `{generated_ts_utc}`",
            f"- input_fact_compras: `{fact_compras_path}`",
            f"- input_ofertas_typed: `{ofertas_typed_path}`",
            f"- rows_compras: `{len(compras)}`",
            f"- rows_ofertas_valid: `{len(ofertas)}`",
            f"- row_coverage_with_terminal_candidate: `{coverage_ratio:.6f}`",
            f"- min_confidence_for_suggestion: `{min_confidence:.2f}`",
            f"- suggested_mappings_high_confidence: `{len(confident_rows)}`",
            "",
            "## Nota metodológica",
            "- Se compara `precio_unitario_compra * 1000` contra `coste_min` de ofertas para elegir la terminal CLH más cercana por cada línea.",
            "- Este diagnóstico es informativo y no se usa automáticamente para construir features.",
            "",
            "## Resumen por terminal de compra",
            _to_markdown_table(terminal_summary),
            "",
        ]
    )
    output_md_path.write_text(md_content, encoding="utf-8")

    summary = {
        "status": "ok",
        "run_id": execution_run_id,
        "generated_ts_utc": generated_ts_utc,
        "input_fact_compras": str(fact_compras_path),
        "input_ofertas_typed": str(ofertas_typed_path),
        "output_md": str(output_md_path),
        "output_csv": str(output_csv_path),
        "rows_compras": int(len(compras)),
        "rows_ofertas_valid": int(len(ofertas)),
        "row_coverage_with_terminal_candidate": coverage_ratio,
        "min_confidence_for_suggestion": float(min_confidence),
        "suggested_mappings_high_confidence": int(len(confident_rows)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera diagnóstico de puente terminal TERM->CLH para Day02 (informativo)."
    )
    parser.add_argument(
        "--fact-compras",
        type=Path,
        default=Path("data/public/support/fact_compras.csv"),
        help="Input de compras curated.",
    )
    parser.add_argument(
        "--ofertas-typed",
        type=Path,
        default=Path("data/public/support/ofertas_typed.csv"),
        help="Input de ofertas tipadas.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/public/data_terminal_bridge_diagnostic.md"),
        help="Output markdown del diagnóstico.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/public/data_terminal_bridge_stats.csv"),
        help="Output tabular del diagnóstico.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Umbral mínimo de confianza para sugerir mapping.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador opcional de ejecución.",
    )
    args = parser.parse_args()

    summary = run(
        fact_compras_path=args.fact_compras,
        ofertas_typed_path=args.ofertas_typed,
        output_md_path=args.output_md,
        output_csv_path=args.output_csv,
        min_confidence=args.min_confidence,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
