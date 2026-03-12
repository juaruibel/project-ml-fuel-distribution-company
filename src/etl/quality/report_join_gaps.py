from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _top_counts(series: pd.Series, top_n: int = 20) -> list[dict]:
    counts = series.fillna("").astype(str).str.strip()
    counts = counts[counts != ""].value_counts().head(top_n)
    return [{"value": idx, "count": int(count)} for idx, count in counts.items()]


def run(
    join_input_path: Path,
    rejects_input_path: Path,
    report_output_path: Path,
    aliases_backlog_output_path: Path,
    top_n: int = 30,
) -> dict:
    if not join_input_path.exists():
        raise FileNotFoundError(f"No existe join_diagnostico: {join_input_path}")

    join_frame = pd.read_csv(join_input_path, dtype=str, keep_default_na=False)
    join_frame["year"] = join_frame.get("fecha_oferta", "").astype(str).str[:4]

    right_only = join_frame[join_frame.get("join_status", "") == "right_only"].copy()
    left_only = join_frame[join_frame.get("join_status", "") == "left_only"].copy()

    alias_backlog = pd.DataFrame(
        columns=["raw_value", "count", "suggested_canonico", "status", "notes"]
    )
    unknown_provider_rows = 0
    if rejects_input_path.exists():
        rejects = pd.read_csv(rejects_input_path, dtype=str, keep_default_na=False)
        mask_unknown = rejects.get("dq_reason", "").str.contains(
            "unknown_proveedor_canonico", case=False, regex=False
        )
        unknown_provider = rejects[mask_unknown].copy()
        unknown_provider_rows = int(len(unknown_provider))
        counts = (
            unknown_provider.get("proveedor_min_raw", pd.Series([], dtype="string"))
            .fillna("")
            .astype(str)
            .str.strip()
        )
        counts = counts[counts != ""].value_counts().head(top_n)
        alias_backlog = pd.DataFrame(
            {
                "raw_value": counts.index,
                "count": counts.values.astype(int),
                "suggested_canonico": "",
                "status": "pending",
                "notes": "Auto-generado desde ofertas_rejects_typed",
            }
        )

    aliases_backlog_output_path.parent.mkdir(parents=True, exist_ok=True)
    alias_backlog.to_csv(aliases_backlog_output_path, index=False, encoding="utf-8")

    status_counts = join_frame.get("join_status", pd.Series([], dtype="string")).value_counts()
    by_year = (
        join_frame.groupby(["year", "join_status"]).size().unstack(fill_value=0).reset_index()
        if not join_frame.empty
        else pd.DataFrame()
    )
    by_year_records = by_year.to_dict(orient="records")

    report = {
        "status": "ok",
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_ts_utc": datetime.now(timezone.utc).isoformat(),
        "input_join": str(join_input_path),
        "input_rejects_typed": str(rejects_input_path),
        "rows_join_total": int(len(join_frame)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "right_only_top_proveedores": _top_counts(right_only.get("proveedor_canonico", pd.Series([], dtype="string")), top_n=top_n),
        "right_only_top_productos": _top_counts(right_only.get("producto_canonico", pd.Series([], dtype="string")), top_n=top_n),
        "left_only_top_proveedores": _top_counts(left_only.get("proveedor_canonico", pd.Series([], dtype="string")), top_n=top_n),
        "left_only_top_productos": _top_counts(left_only.get("producto_canonico", pd.Series([], dtype="string")), top_n=top_n),
        "right_only_top_years": _top_counts(right_only.get("year", pd.Series([], dtype="string")), top_n=10),
        "left_only_top_years": _top_counts(left_only.get("year", pd.Series([], dtype="string")), top_n=10),
        "coverage_by_year": by_year_records,
        "unknown_provider_rows_in_rejects": unknown_provider_rows,
        "aliases_backlog_rows": int(len(alias_backlog)),
        "aliases_backlog_output": str(aliases_backlog_output_path),
    }

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera análisis de gaps de join y backlog de aliases de proveedor."
    )
    parser.add_argument(
        "--join-input",
        type=Path,
        default=Path("data/public/support/join_diagnostico.csv"),
        help="Ruta de join_diagnostico.csv.",
    )
    parser.add_argument(
        "--rejects-input",
        type=Path,
        default=Path("data/public/support/ofertas_rejects_typed.csv"),
        help="Ruta de ofertas_rejects_typed.csv.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/public/etl_join_gap_analysis_v1.json"),
        help="Ruta del reporte de gaps.",
    )
    parser.add_argument(
        "--aliases-backlog-output",
        type=Path,
        default=Path("config/provider_aliases_backlog.csv"),
        help="Ruta del backlog de aliases de proveedores.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Número de elementos top por dimensión.",
    )
    args = parser.parse_args()

    report = run(
        join_input_path=args.join_input,
        rejects_input_path=args.rejects_input,
        report_output_path=args.report_output,
        aliases_backlog_output_path=args.aliases_backlog_output,
        top_n=args.top_n,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
