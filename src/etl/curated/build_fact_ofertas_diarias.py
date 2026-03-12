from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def run(input_path: Path, output_path: Path, run_id: str | None = None) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el input requerido: {input_path}")

    execution_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    frame = pd.read_csv(input_path, dtype={"fecha_oferta": "string"})
    if frame.empty:
        raise RuntimeError("`ofertas_typed.csv` está vacío; no se puede construir fact_ofertas_diarias.")

    frame["coste_min"] = pd.to_numeric(frame["coste_min"], errors="coerce")
    frame = frame.dropna(subset=["fecha_oferta", "producto_canonico", "proveedor_canonico", "coste_min"])

    fact = (
        frame.groupby(["fecha_oferta", "producto_canonico", "proveedor_canonico"], as_index=False)
        .agg(
            coste_min_dia_proveedor=("coste_min", "min"),
            terminales_cubiertos=("terminal_canonico", "nunique"),
            observaciones_oferta=("record_hash", "nunique"),
        )
        .sort_values(["fecha_oferta", "producto_canonico", "coste_min_dia_proveedor", "proveedor_canonico"])
        .reset_index(drop=True)
    )

    fact["rank_coste_dia_producto"] = (
        fact.groupby(["fecha_oferta", "producto_canonico"])["coste_min_dia_proveedor"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    fact["curated_run_id"] = execution_run_id
    fact["curated_ts_utc"] = datetime.now(timezone.utc).isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path, index=False, encoding="utf-8")

    summary = {
        "status": "ok",
        "curated_run_id": execution_run_id,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "rows_output": int(len(fact)),
        "distinct_fechas": int(fact["fecha_oferta"].nunique()),
        "distinct_productos": int(fact["producto_canonico"].nunique()),
        "distinct_proveedores": int(fact["proveedor_canonico"].nunique()),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye fact_ofertas_diarias.csv desde ofertas_typed.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/public/support/ofertas_typed.csv"),
        help="Input de ofertas tipadas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public/support/fact_ofertas_diarias.csv"),
        help="Output curated de ofertas diarias.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador de ejecución opcional.",
    )
    args = parser.parse_args()

    summary = run(input_path=args.input, output_path=args.output, run_id=args.run_id)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
