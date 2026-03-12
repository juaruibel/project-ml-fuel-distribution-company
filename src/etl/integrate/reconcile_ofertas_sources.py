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


KEY_COLUMNS = ["fecha_oferta", "terminal_canonico", "producto_canonico"]


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


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


def _parse_number(value: Any) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    compact = text.replace(" ", "")
    has_dot = "." in compact
    has_comma = "," in compact
    if has_dot and has_comma:
        if compact.rfind(",") > compact.rfind("."):
            compact = compact.replace(".", "").replace(",", ".")
        else:
            compact = compact.replace(",", "")
    elif has_comma and not has_dot:
        compact = compact.replace(",", ".")
    compact = re.sub(r"[^0-9\.-]", "", compact)
    if compact in {"", "-", ".", "-."}:
        return None
    parsed = pd.to_numeric(compact, errors="coerce")
    if pd.isna(parsed):
        return None
    numeric = float(parsed)
    if numeric <= 0:
        return None
    return numeric


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
            raw_norm = _normalize_key(raw_value_text)
            canonical_value = _clean_text(canonical_value_text)
            if raw_norm and canonical_value:
                mapping[raw_norm] = canonical_value
    if not mapping:
        raise ValueError(f"Mapping sin filas válidas: {mapping_path}")
    return mapping


def _map_to_canonical(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return series.astype("string").map(lambda value: mapping.get(_normalize_key(value), "UNKNOWN"))


def _get_series(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return pd.Series([""] * len(frame), index=frame.index, dtype="string")


def _prepare_table_rows(
    table_input_path: Path,
    productos_mapping: dict[str, str],
    proveedores_mapping: dict[str, str],
    terminales_mapping: dict[str, str],
) -> pd.DataFrame:
    table_frame = pd.read_csv(table_input_path, dtype=str, keep_default_na=False)
    table_frame = table_frame.copy()
    table_frame["fecha_oferta"] = pd.to_datetime(
        _get_series(table_frame, ["fecha_raw"]), errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    table_frame["terminal_raw"] = _get_series(table_frame, ["terminal_raw"]).map(_clean_text)
    table_frame["producto_raw"] = _get_series(table_frame, ["producto_raw"]).map(_clean_text)
    table_frame["proveedor_raw"] = _get_series(table_frame, ["proveedor_min_raw", "proveedor_raw"]).map(_clean_text)
    table_frame["coste_raw"] = _get_series(table_frame, ["coste_min_raw", "coste_raw"]).map(_clean_text)
    table_frame["coste_num"] = table_frame["coste_raw"].map(_parse_number)

    table_frame["terminal_canonico"] = _map_to_canonical(table_frame["terminal_raw"], terminales_mapping)
    table_frame["producto_canonico"] = _map_to_canonical(table_frame["producto_raw"], productos_mapping)
    table_frame["proveedor_canonico"] = _map_to_canonical(table_frame["proveedor_raw"], proveedores_mapping)
    table_frame["source_type"] = "tabla"

    table_frame = table_frame.dropna(subset=["fecha_oferta"])
    table_frame = table_frame[
        (table_frame["terminal_canonico"] != "UNKNOWN") & (table_frame["producto_canonico"] != "UNKNOWN")
    ].copy()
    if table_frame.empty:
        return table_frame

    selected_rows: list[pd.Series] = []
    for _, group in table_frame.groupby(KEY_COLUMNS, dropna=False):
        valid = group[group["coste_num"].notna()].copy()
        if not valid.empty:
            selected_rows.append(valid.loc[valid["coste_num"].idxmin()])
        else:
            selected_rows.append(group.iloc[0])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _prepare_calculos_rows(
    calculos_input_path: Path,
    productos_mapping: dict[str, str],
    proveedores_mapping: dict[str, str],
    terminales_mapping: dict[str, str],
) -> pd.DataFrame:
    calculos_frame = pd.read_csv(calculos_input_path, dtype=str, keep_default_na=False)
    calculos_frame = calculos_frame.copy()
    calculos_frame["fecha_oferta"] = pd.to_datetime(
        _get_series(calculos_frame, ["fecha_raw"]), errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    calculos_frame["terminal_raw"] = _get_series(calculos_frame, ["terminal_raw"]).map(_clean_text)
    calculos_frame["producto_raw"] = _get_series(calculos_frame, ["producto_raw"]).map(_clean_text)
    calculos_frame["proveedor_raw"] = _get_series(calculos_frame, ["proveedor_raw"]).map(_clean_text)
    calculos_frame["coste_raw"] = _get_series(calculos_frame, ["coste_raw"]).map(_clean_text)
    calculos_frame["coste_num"] = calculos_frame["coste_raw"].map(_parse_number)

    calculos_frame["terminal_canonico"] = _map_to_canonical(calculos_frame["terminal_raw"], terminales_mapping)
    calculos_frame["producto_canonico"] = _map_to_canonical(calculos_frame["producto_raw"], productos_mapping)
    calculos_frame["proveedor_canonico"] = _map_to_canonical(calculos_frame["proveedor_raw"], proveedores_mapping)
    calculos_frame["source_type"] = "calculos"
    calculos_frame = calculos_frame.dropna(subset=["fecha_oferta"])
    calculos_frame = calculos_frame[
        (calculos_frame["terminal_canonico"] != "UNKNOWN") & (calculos_frame["producto_canonico"] != "UNKNOWN")
    ].copy()
    return calculos_frame


def _build_key(row: pd.Series) -> tuple[str, str, str]:
    return (
        str(row["fecha_oferta"]),
        str(row["terminal_canonico"]),
        str(row["producto_canonico"]),
    )


def _status_for_key(
    table_row: pd.Series | None,
    calculos_min_row: pd.Series | None,
    tolerance: float = 0.01,
) -> str:
    if table_row is None and calculos_min_row is None:
        return "single_source"
    if table_row is None or calculos_min_row is None:
        return "single_source"

    table_provider = str(table_row.get("proveedor_canonico", ""))
    calc_provider = str(calculos_min_row.get("proveedor_canonico", ""))
    table_cost = table_row.get("coste_num", None)
    calc_cost = calculos_min_row.get("coste_num", None)
    if pd.notna(table_cost) and pd.notna(calc_cost):
        cost_close = abs(float(table_cost) - float(calc_cost)) <= tolerance
    else:
        cost_close = False

    if table_provider == calc_provider and cost_close:
        return "agree"
    return "conflict"


def run(
    table_input_path: Path,
    calculos_input_path: Path,
    output_path: Path,
    report_path: Path,
    productos_mapping_path: Path,
    proveedores_mapping_path: Path,
    terminales_mapping_path: Path,
    run_id: str | None = None,
) -> dict:
    if not table_input_path.exists():
        raise FileNotFoundError(f"No existe input Tabla: {table_input_path}")
    if not calculos_input_path.exists():
        raise FileNotFoundError(f"No existe input Cálculos: {calculos_input_path}")

    execution_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ingestion_ts_utc = datetime.now(timezone.utc).isoformat()

    productos_mapping = _load_mapping(productos_mapping_path, "raw_value", "producto_canonico")
    proveedores_mapping = _load_mapping(proveedores_mapping_path, "raw_value", "proveedor_canonico")
    terminales_mapping = _load_mapping(terminales_mapping_path, "raw_value", "terminal_canonico")

    table_rows = _prepare_table_rows(
        table_input_path=table_input_path,
        productos_mapping=productos_mapping,
        proveedores_mapping=proveedores_mapping,
        terminales_mapping=terminales_mapping,
    )
    calculos_rows = _prepare_calculos_rows(
        calculos_input_path=calculos_input_path,
        productos_mapping=productos_mapping,
        proveedores_mapping=proveedores_mapping,
        terminales_mapping=terminales_mapping,
    )

    table_by_key: dict[tuple[str, str, str], pd.Series] = {}
    for _, row in table_rows.iterrows():
        table_by_key[_build_key(row)] = row

    calculos_groups: dict[tuple[str, str, str], pd.DataFrame] = {}
    if not calculos_rows.empty:
        for key, group in calculos_rows.groupby(KEY_COLUMNS, dropna=False):
            calculos_groups[(str(key[0]), str(key[1]), str(key[2]))] = group.copy()

    all_keys = sorted(set(table_by_key.keys()) | set(calculos_groups.keys()))
    reconciled_rows: list[dict] = []

    empty_calculos_group = pd.DataFrame(columns=calculos_rows.columns)

    for key in all_keys:
        fecha_oferta, terminal_canonico, producto_canonico = key
        table_row = table_by_key.get(key)
        calculos_group = calculos_groups.get(key, empty_calculos_group)
        calculos_valid = calculos_group[calculos_group["coste_num"].notna()].copy()
        calculos_min_row = None
        if not calculos_valid.empty:
            calculos_min_row = calculos_valid.loc[calculos_valid["coste_num"].idxmin()]

        reconciliation_status = _status_for_key(table_row, calculos_min_row)

        if calculos_min_row is not None:
            selected_source = "calculos"
            selected_group = calculos_valid.copy()
        elif table_row is not None:
            selected_source = "tabla"
            selected_group = pd.DataFrame([table_row])
        else:
            continue

        tabla_provider_raw = _clean_text(table_row.get("proveedor_raw", "")) if table_row is not None else ""
        tabla_cost_raw = _clean_text(table_row.get("coste_raw", "")) if table_row is not None else ""
        tabla_source_file = _clean_text(table_row.get("source_file", "")) if table_row is not None else ""

        calculos_provider_raw = (
            _clean_text(calculos_min_row.get("proveedor_raw", "")) if calculos_min_row is not None else ""
        )
        calculos_cost_raw = _clean_text(calculos_min_row.get("coste_raw", "")) if calculos_min_row is not None else ""
        calculos_source_file = (
            _clean_text(calculos_min_row.get("source_file", "")) if calculos_min_row is not None else ""
        )

        for _, selected_row in selected_group.iterrows():
            row_terminal = selected_row.get("row_terminal", selected_row.get("row_idx", ""))
            col_producto = selected_row.get("col_producto", selected_row.get("col_idx", ""))
            reconciled_rows.append(
                {
                    "fecha_raw": fecha_oferta,
                    "terminal_raw": _clean_text(selected_row.get("terminal_raw", "")),
                    "producto_raw": _clean_text(selected_row.get("producto_raw", "")),
                    "proveedor_min_raw": _clean_text(selected_row.get("proveedor_raw", "")),
                    "coste_min_raw": _clean_text(selected_row.get("coste_raw", "")),
                    "source_file": _clean_text(selected_row.get("source_file", "")),
                    "sheet_name": _clean_text(selected_row.get("sheet_name", "")),
                    "row_terminal": row_terminal,
                    "col_producto": col_producto,
                    "row_cost_anchor": selected_row.get("row_cost_anchor", ""),
                    "row_product_header": selected_row.get("row_product_header", ""),
                    "provider_header_row": selected_row.get("provider_header_row", ""),
                    "cost_lookup_col": selected_row.get("cost_lookup_col", ""),
                    "cost_source": _clean_text(selected_row.get("cost_source", selected_source)),
                    "selected_source": selected_source,
                    "reconciliation_status": reconciliation_status,
                    "reconciliation_key": f"{fecha_oferta}|{terminal_canonico}|{producto_canonico}",
                    "key_fecha_oferta": fecha_oferta,
                    "key_terminal_canonico": terminal_canonico,
                    "key_producto_canonico": producto_canonico,
                    "tabla_proveedor_min_raw": tabla_provider_raw,
                    "tabla_coste_min_raw": tabla_cost_raw,
                    "tabla_source_file": tabla_source_file,
                    "calculos_proveedor_min_raw": calculos_provider_raw,
                    "calculos_coste_min_raw": calculos_cost_raw,
                    "calculos_source_file": calculos_source_file,
                    "run_id": execution_run_id,
                    "ingestion_ts_utc": ingestion_ts_utc,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    reconciled_frame = pd.DataFrame(reconciled_rows)
    reconciled_frame.to_csv(output_path, index=False, encoding="utf-8")

    selected_source_counts = Counter(reconciled_frame["selected_source"]) if not reconciled_frame.empty else Counter()
    status_counts = Counter(reconciled_frame["reconciliation_status"]) if not reconciled_frame.empty else Counter()
    summary = {
        "status": "ok",
        "run_id": execution_run_id,
        "input_tabla_rows": int(len(table_rows)),
        "input_calculos_rows": int(len(calculos_rows)),
        "reconciled_rows": int(len(reconciled_frame)),
        "keys_total": int(len(all_keys)),
        "selected_source_counts": dict(selected_source_counts),
        "reconciliation_status_counts": dict(status_counts),
        "output_file": str(output_path),
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcilia ofertas de Tabla y Cálculos con estrategia dual + comparar."
    )
    parser.add_argument(
        "--tabla-input",
        type=Path,
        default=Path("data/public/support/ofertas_raw_mincost.csv"),
        help="Input crudo de ofertas extraído desde Tabla.",
    )
    parser.add_argument(
        "--calculos-input",
        type=Path,
        default=Path("data/public/support/ofertas_calculos_raw_long.csv"),
        help="Input crudo de ofertas extraído desde Cálculos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public/support/ofertas_reconciled_raw.csv"),
        help="Output reconciliado para transform typed.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/public/etl_ofertas_source_reconciliation_v1.json"),
        help="Reporte resumen de reconciliación.",
    )
    parser.add_argument(
        "--productos-mapping",
        type=Path,
        default=Path("config/productos_mapping_v1.csv"),
        help="Mapping de productos.",
    )
    parser.add_argument(
        "--proveedores-mapping",
        type=Path,
        default=Path("config/proveedores_mapping_v1.csv"),
        help="Mapping de proveedores.",
    )
    parser.add_argument(
        "--terminales-mapping",
        type=Path,
        default=Path("config/terminales_mapping_v1.csv"),
        help="Mapping de terminales.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador opcional de ejecución.",
    )
    args = parser.parse_args()

    summary = run(
        table_input_path=args.tabla_input,
        calculos_input_path=args.calculos_input,
        output_path=args.output,
        report_path=args.report,
        productos_mapping_path=args.productos_mapping,
        proveedores_mapping_path=args.proveedores_mapping,
        terminales_mapping_path=args.terminales_mapping,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
