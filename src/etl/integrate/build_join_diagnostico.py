from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )


def _normalize_key(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    text = _strip_accents(text).upper()
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _load_mapping(mapping_path: Path, raw_column: str, canonical_column: str) -> dict[str, str]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"No existe mapping requerido: {mapping_path}")
    mapping: dict[str, str] = {}
    with mapping_path.open("r", encoding="utf-8") as mapping_file:
        reader = csv.reader(mapping_file)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Mapping vacío: {mapping_path}")
        if len(header) < 2:
            raise ValueError(f"Mapping inválido en {mapping_path}: header incompleto.")
        if header[0].strip() != raw_column or header[-1].strip() != canonical_column:
            raise ValueError(
                f"Mapping inválido en {mapping_path}. Se esperaban `{raw_column}` y `{canonical_column}`."
            )
        for row in reader:
            if len(row) < 2:
                continue
            raw_value_text = ",".join(row[:-1]).strip()
            canonical_value_text = row[-1].strip()
            raw_value = _normalize_key(raw_value_text)
            canonical_value = _clean_text(canonical_value_text)
            if raw_value and canonical_value:
                mapping[raw_value] = canonical_value
    if not mapping:
        raise ValueError(f"Mapping sin filas válidas: {mapping_path}")
    return mapping


def _map_to_canonical(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return series.astype("string").map(lambda value: mapping.get(_normalize_key(value), "UNKNOWN"))


def _get_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name in frame.columns:
        return frame[column_name]
    return pd.Series([""] * len(frame), index=frame.index, dtype="string")


def _build_fact_compras(
    compras_input_path: Path,
    fact_compras_output_path: Path,
    productos_mapping_path: Path,
    proveedores_mapping_path: Path,
    terminales_mapping_path: Path,
    run_id: str,
) -> pd.DataFrame:
    if not compras_input_path.exists():
        raise FileNotFoundError(f"No existe el input de compras tipadas: {compras_input_path}")

    compras_frame = pd.read_csv(compras_input_path, dtype=str, keep_default_na=False)
    if "dq_status" in compras_frame.columns:
        compras_frame = compras_frame[compras_frame["dq_status"] == "valid"].copy()

    compras_frame["fecha_compra"] = pd.to_datetime(
        _get_series(compras_frame, "fecha_compra"), errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    compras_frame["litros"] = pd.to_numeric(_get_series(compras_frame, "litros"), errors="coerce")
    compras_frame["precio_unitario"] = pd.to_numeric(
        _get_series(compras_frame, "precio_unitario"), errors="coerce"
    )
    compras_frame["importe_total"] = pd.to_numeric(
        _get_series(compras_frame, "importe_total"), errors="coerce"
    )

    productos_mapping = _load_mapping(productos_mapping_path, "raw_value", "producto_canonico")
    proveedores_mapping = _load_mapping(proveedores_mapping_path, "raw_value", "proveedor_canonico")
    terminales_mapping = _load_mapping(terminales_mapping_path, "raw_value", "terminal_canonico")

    fact_compras = pd.DataFrame(index=compras_frame.index)
    fact_compras["fecha_compra"] = _get_series(compras_frame, "fecha_compra").astype("string")
    fact_compras["albaran_id"] = _get_series(compras_frame, "albaran_id").map(_clean_text)
    fact_compras["linea_id"] = _get_series(compras_frame, "linea_id").map(_clean_text)
    fact_compras["producto_canonico"] = _map_to_canonical(
        _get_series(compras_frame, "producto_raw"), productos_mapping
    )
    fact_compras["proveedor_canonico"] = _map_to_canonical(
        _get_series(compras_frame, "proveedor_raw"), proveedores_mapping
    )
    fact_compras["terminal_canonico"] = _map_to_canonical(
        _get_series(compras_frame, "terminal_raw"), terminales_mapping
    )
    fact_compras["litros"] = pd.to_numeric(_get_series(compras_frame, "litros"), errors="coerce")
    fact_compras["precio_unitario"] = pd.to_numeric(
        _get_series(compras_frame, "precio_unitario"), errors="coerce"
    )
    fact_compras["importe_total"] = pd.to_numeric(
        _get_series(compras_frame, "importe_total"), errors="coerce"
    )
    fact_compras["source_file"] = _get_series(compras_frame, "source_file").map(_clean_text)
    fact_compras["source_transform_run_id"] = _get_series(compras_frame, "transform_run_id").map(_clean_text)
    fact_compras["curated_run_id"] = run_id
    fact_compras["curated_ts_utc"] = datetime.now(timezone.utc).isoformat()

    fact_compras = fact_compras.dropna(subset=["fecha_compra"])
    fact_compras_output_path.parent.mkdir(parents=True, exist_ok=True)
    fact_compras.to_csv(fact_compras_output_path, index=False, encoding="utf-8")
    return fact_compras


def _infer_failure_reason(row: pd.Series) -> str:
    status = row["join_status"]
    if status == "matched":
        return ""

    has_unknown = any(
        str(row[column]).strip().upper() == "UNKNOWN"
        for column in ["producto_canonico", "proveedor_canonico"]
    )
    if status == "left_only":
        return "oferta_con_dimension_unknown" if has_unknown else "oferta_sin_compra"
    return "compra_con_dimension_unknown" if has_unknown else "compra_sin_oferta_mapeada"


def run(
    ofertas_fact_path: Path,
    compras_typed_path: Path,
    fact_compras_output_path: Path,
    join_output_path: Path,
    report_output_path: Path,
    productos_mapping_path: Path,
    proveedores_mapping_path: Path,
    terminales_mapping_path: Path,
    run_id: str | None = None,
) -> dict:
    if not ofertas_fact_path.exists():
        raise FileNotFoundError(f"No existe el input de fact ofertas: {ofertas_fact_path}")

    execution_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fact_compras = _build_fact_compras(
        compras_input_path=compras_typed_path,
        fact_compras_output_path=fact_compras_output_path,
        productos_mapping_path=productos_mapping_path,
        proveedores_mapping_path=proveedores_mapping_path,
        terminales_mapping_path=terminales_mapping_path,
        run_id=execution_run_id,
    )

    ofertas_fact = pd.read_csv(ofertas_fact_path, dtype=str, keep_default_na=False)
    ofertas_fact["coste_min_dia_proveedor"] = pd.to_numeric(
        ofertas_fact.get("coste_min_dia_proveedor", ""), errors="coerce"
    )
    ofertas_fact["terminales_cubiertos"] = pd.to_numeric(
        ofertas_fact.get("terminales_cubiertos", ""), errors="coerce"
    )
    ofertas_fact["rank_coste_dia_producto"] = pd.to_numeric(
        ofertas_fact.get("rank_coste_dia_producto", ""), errors="coerce"
    )

    offers_keys = (
        ofertas_fact.groupby(["fecha_oferta", "producto_canonico", "proveedor_canonico"], as_index=False)
        .agg(
            offers_rows=("coste_min_dia_proveedor", "size"),
            coste_min_dia_proveedor=("coste_min_dia_proveedor", "min"),
            terminales_cubiertos=("terminales_cubiertos", "max"),
            rank_coste_dia_producto=("rank_coste_dia_producto", "min"),
        )
        .copy()
    )

    compras_fact_for_join = fact_compras.rename(columns={"fecha_compra": "fecha_oferta"}).copy()
    compras_keys = (
        compras_fact_for_join.groupby(["fecha_oferta", "producto_canonico", "proveedor_canonico"], as_index=False)
        .agg(
            compras_rows=("albaran_id", "size"),
            litros_totales=("litros", "sum"),
            importe_total=("importe_total", "sum"),
            precio_medio_compra=("precio_unitario", "mean"),
        )
        .copy()
    )

    join_frame = offers_keys.merge(
        compras_keys,
        on=["fecha_oferta", "producto_canonico", "proveedor_canonico"],
        how="outer",
        indicator=True,
    )
    join_frame["join_status"] = join_frame["_merge"].map(
        {
            "both": "matched",
            "left_only": "left_only",
            "right_only": "right_only",
        }
    )
    join_frame["motivo_fallo"] = join_frame.apply(_infer_failure_reason, axis=1)
    join_frame["join_run_id"] = execution_run_id
    join_frame["join_ts_utc"] = datetime.now(timezone.utc).isoformat()

    join_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    join_frame.to_csv(join_output_path, index=False, encoding="utf-8")

    status_counter = Counter(join_frame["join_status"])
    failure_counter = Counter(reason for reason in join_frame["motivo_fallo"] if reason)
    matched_count = status_counter.get("matched", 0)
    report = {
        "status": "ok",
        "join_run_id": execution_run_id,
        "input_fact_ofertas": str(ofertas_fact_path),
        "input_compras_typed": str(compras_typed_path),
        "output_fact_compras": str(fact_compras_output_path),
        "output_join_diagnostico": str(join_output_path),
        "rows_fact_ofertas_keys": int(len(offers_keys)),
        "rows_fact_compras_keys": int(len(compras_keys)),
        "rows_join_output": int(len(join_frame)),
        "status_counts": dict(status_counter),
        "failure_reason_counts": dict(failure_counter),
        "coverage_over_compras_keys": float(matched_count / len(compras_keys)) if len(compras_keys) else 0.0,
        "coverage_over_ofertas_keys": float(matched_count / len(offers_keys)) if len(offers_keys) else 0.0,
    }
    report_output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye fact_compras.csv y join_diagnostico.csv (ofertas vs compras)."
    )
    parser.add_argument(
        "--fact-ofertas",
        type=Path,
        default=Path("data/public/support/fact_ofertas_diarias.csv"),
        help="Input de ofertas curated.",
    )
    parser.add_argument(
        "--compras-typed",
        type=Path,
        default=Path("data/public/support/compras_typed.csv"),
        help="Input de compras tipadas.",
    )
    parser.add_argument(
        "--fact-compras-output",
        type=Path,
        default=Path("data/public/support/fact_compras.csv"),
        help="Output de fact compras curated.",
    )
    parser.add_argument(
        "--join-output",
        type=Path,
        default=Path("data/public/support/join_diagnostico.csv"),
        help="Output del join diagnóstico.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/public/etl_join_coverage_v1.json"),
        help="Reporte resumen de cobertura del join.",
    )
    parser.add_argument(
        "--productos-mapping",
        type=Path,
        default=Path("config/productos_mapping_v1.csv"),
        help="Mapping producto raw -> canónico.",
    )
    parser.add_argument(
        "--proveedores-mapping",
        type=Path,
        default=Path("config/proveedores_mapping_v1.csv"),
        help="Mapping proveedor raw -> canónico.",
    )
    parser.add_argument(
        "--terminales-mapping",
        type=Path,
        default=Path("config/terminales_mapping_v1.csv"),
        help="Mapping terminal raw -> canónico.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador opcional de la ejecución.",
    )
    args = parser.parse_args()

    summary = run(
        ofertas_fact_path=args.fact_ofertas,
        compras_typed_path=args.compras_typed,
        fact_compras_output_path=args.fact_compras_output,
        join_output_path=args.join_output,
        report_output_path=args.report_output,
        productos_mapping_path=args.productos_mapping,
        proveedores_mapping_path=args.proveedores_mapping,
        terminales_mapping_path=args.terminales_mapping,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
