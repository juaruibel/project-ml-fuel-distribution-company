from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _to_snake(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"[^\w]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _normalize_columns(columns: list[str]) -> list[str]:
    normalized = [_to_snake(str(column)) for column in columns]
    fixed: list[str] = []
    counts: dict[str, int] = {}
    for col in normalized:
        if col not in counts:
            counts[col] = 0
            fixed.append(col or "unnamed")
            continue
        counts[col] += 1
        base = col or "unnamed"
        fixed.append(f"{base}_{counts[col]}")
    return fixed


def _write_output(df: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8")
        return
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        return
    raise ValueError(
        "Formato de salida no soportado. Usa extensión `.csv` (recomendado) o `.parquet`."
    )


def run(input_path: Path, output_path: Path, run_id: str | None = None) -> dict:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ingestion_ts = datetime.now(timezone.utc).isoformat()
    try:
        xls = pd.ExcelFile(input_path, engine="xlrd")
    except ImportError as exc:
        raise RuntimeError(
            "Falta la dependencia 'xlrd' para leer archivos .xls. Instala `xlrd>=2.0,<3.0`."
        ) from exc
    frames: list[pd.DataFrame] = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
        if df.empty:
            continue
        df.columns = _normalize_columns([str(column) for column in df.columns])
        df = df.reset_index(drop=True)
        df["row_idx"] = df.index + 2
        df["source_file"] = str(input_path)
        df["sheet_name"] = sheet_name
        df["run_id"] = run_id
        df["ingestion_ts_utc"] = ingestion_ts
        frames.append(df)
    if not frames:
        raise RuntimeError("No se encontraron filas útiles en el XLS de compras.")
    full = pd.concat(frames, axis=0, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_output(full, output_path)
    summary = {
        "status": "ok",
        "run_id": run_id,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "sheets_read": len(xls.sheet_names),
        "rows_written": int(len(full)),
        "columns_written": int(full.shape[1]),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extracción inicial de compras desde XLS legacy a staging CSV/Parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/SUPPLIER_DAILY_COMPARISON/compras totales.xls"),
        help="Ruta del XLS de compras.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public/support/compras_raw_xls.csv"),
        help="Ruta de salida de staging (`.csv` recomendado en bootcamp, `.parquet` opcional).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador de ejecución ETL.",
    )
    args = parser.parse_args()
    summary = run(input_path=args.input, output_path=args.output, run_id=args.run_id)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
