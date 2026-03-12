from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for column_name in candidates:
        if column_name in frame.columns:
            return frame[column_name]
    return pd.Series([""] * len(frame), index=frame.index, dtype="string")


def _parse_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return parsed.dt.strftime("%Y-%m-%d")


def _parse_number(series: pd.Series, invalid_cost_tokens_norm: set[str]) -> pd.Series:
    def _normalize_number_token(raw_value: Any) -> str | None:
        text_value = _clean_text(raw_value)
        if not text_value:
            return None

        token_norm = _normalize_key(text_value)
        if token_norm in invalid_cost_tokens_norm:
            return None

        compact = text_value.replace(" ", "")
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
        return compact

    normalized = series.astype("string").map(_normalize_number_token)
    return pd.to_numeric(normalized, errors="coerce")


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
            if not raw_value or not canonical_value:
                continue
            mapping[raw_value] = canonical_value

    if not mapping:
        raise ValueError(f"Mapping sin filas válidas: {mapping_path}")
    return mapping


def _map_to_canonical(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return series.astype("string").map(lambda value: mapping.get(_normalize_key(value), pd.NA))


def _append_reason(reason_series: pd.Series, mask: pd.Series, reason_name: str) -> pd.Series:
    output = reason_series.copy()
    apply_mask = mask.fillna(False)
    without_reason = apply_mask & output.isna()
    with_reason = apply_mask & output.notna()
    output.loc[without_reason] = reason_name
    output.loc[with_reason] = output.loc[with_reason] + "|" + reason_name
    return output


def _build_record_hash(valid_frame: pd.DataFrame) -> pd.Series:
    hash_base = (
        valid_frame[
            [
                "fecha_oferta",
                "terminal_canonico",
                "producto_canonico",
                "proveedor_canonico",
                "coste_min",
                "source_file",
                "row_terminal",
                "col_producto",
            ]
        ]
        .fillna("")
        .astype(str)
        .agg("|".join, axis=1)
    )
    return hash_base.apply(lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest())


def run(
    input_path: Path,
    output_path: Path,
    rejects_path: Path,
    report_path: Path,
    productos_mapping_path: Path,
    proveedores_mapping_path: Path,
    terminales_mapping_path: Path,
    layout_rules_path: Path,
    run_id: str | None = None,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el input requerido: {input_path}")

    layout_rules = _load_json(layout_rules_path)
    invalid_cost_tokens_norm = {
        _normalize_key(token) for token in layout_rules.get("invalid_cost_tokens", []) if str(token).strip()
    }

    productos_mapping = _load_mapping(productos_mapping_path, "raw_value", "producto_canonico")
    proveedores_mapping = _load_mapping(proveedores_mapping_path, "raw_value", "proveedor_canonico")
    terminales_mapping = _load_mapping(terminales_mapping_path, "raw_value", "terminal_canonico")

    raw_frame = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    execution_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    transform_ts_utc = datetime.now(timezone.utc).isoformat()

    typed_frame = pd.DataFrame(index=raw_frame.index)
    typed_frame["fecha_oferta"] = _parse_date(_first_existing_column(raw_frame, ["fecha_raw"]))
    typed_frame["terminal_raw"] = _first_existing_column(raw_frame, ["terminal_raw"]).map(_clean_text)
    typed_frame["producto_raw"] = _first_existing_column(raw_frame, ["producto_raw"]).map(_clean_text)
    typed_frame["proveedor_min_raw"] = _first_existing_column(
        raw_frame, ["proveedor_min_raw", "proveedor_raw"]
    ).map(_clean_text)
    typed_frame["coste_min_raw"] = _first_existing_column(
        raw_frame, ["coste_min_raw", "coste_raw"]
    ).map(_clean_text)
    typed_frame["coste_min"] = _parse_number(
        _first_existing_column(raw_frame, ["coste_min_raw", "coste_raw"]),
        invalid_cost_tokens_norm=invalid_cost_tokens_norm,
    )

    typed_frame["terminal_canonico"] = _map_to_canonical(typed_frame["terminal_raw"], terminales_mapping)
    typed_frame["producto_canonico"] = _map_to_canonical(typed_frame["producto_raw"], productos_mapping)
    typed_frame["proveedor_canonico"] = _map_to_canonical(
        typed_frame["proveedor_min_raw"], proveedores_mapping
    )

    typed_frame["source_file"] = _first_existing_column(raw_frame, ["source_file"]).map(_clean_text)
    typed_frame["sheet_name"] = _first_existing_column(raw_frame, ["sheet_name"]).map(_clean_text)
    typed_frame["row_terminal"] = pd.to_numeric(
        _first_existing_column(raw_frame, ["row_terminal", "row_idx"]), errors="coerce"
    )
    typed_frame["col_producto"] = pd.to_numeric(
        _first_existing_column(raw_frame, ["col_producto", "col_idx"]), errors="coerce"
    )
    typed_frame["row_cost_anchor"] = pd.to_numeric(
        _first_existing_column(raw_frame, ["row_cost_anchor"]), errors="coerce"
    )
    typed_frame["row_product_header"] = pd.to_numeric(
        _first_existing_column(raw_frame, ["row_product_header"]), errors="coerce"
    )
    typed_frame["provider_header_row"] = pd.to_numeric(
        _first_existing_column(raw_frame, ["provider_header_row"]), errors="coerce"
    )
    typed_frame["cost_lookup_col"] = pd.to_numeric(
        _first_existing_column(raw_frame, ["cost_lookup_col"]), errors="coerce"
    )
    typed_frame["cost_source"] = _first_existing_column(raw_frame, ["cost_source"]).map(_clean_text)
    typed_frame["selected_source"] = _first_existing_column(raw_frame, ["selected_source"]).map(_clean_text)
    typed_frame["reconciliation_status"] = _first_existing_column(
        raw_frame, ["reconciliation_status"]
    ).map(_clean_text)
    typed_frame["reconciliation_key"] = _first_existing_column(raw_frame, ["reconciliation_key"]).map(_clean_text)
    typed_frame["source_run_id"] = _first_existing_column(raw_frame, ["run_id"]).map(_clean_text)
    typed_frame["source_ingestion_ts_utc"] = _first_existing_column(
        raw_frame, ["ingestion_ts_utc"]
    ).map(_clean_text)
    typed_frame["transform_run_id"] = execution_run_id
    typed_frame["transform_ts_utc"] = transform_ts_utc

    dq_reason = pd.Series([pd.NA] * len(typed_frame), index=typed_frame.index, dtype="string")
    dq_reason = _append_reason(dq_reason, typed_frame["fecha_oferta"].isna(), "invalid_fecha_oferta")
    dq_reason = _append_reason(dq_reason, typed_frame["terminal_raw"].eq(""), "missing_terminal_raw")
    dq_reason = _append_reason(dq_reason, typed_frame["producto_raw"].eq(""), "missing_producto_raw")
    dq_reason = _append_reason(dq_reason, typed_frame["proveedor_min_raw"].eq(""), "missing_proveedor_raw")
    dq_reason = _append_reason(dq_reason, typed_frame["coste_min"].isna(), "invalid_coste_min")
    dq_reason = _append_reason(
        dq_reason,
        typed_frame["coste_min"].fillna(0) <= 0,
        "non_positive_coste_min",
    )
    dq_reason = _append_reason(
        dq_reason,
        typed_frame["terminal_canonico"].isna(),
        "unknown_terminal_canonico",
    )
    dq_reason = _append_reason(
        dq_reason,
        typed_frame["producto_canonico"].isna(),
        "unknown_producto_canonico",
    )
    dq_reason = _append_reason(
        dq_reason,
        typed_frame["proveedor_canonico"].isna(),
        "unknown_proveedor_canonico",
    )

    is_valid_row = dq_reason.isna()
    valid_frame = typed_frame.loc[is_valid_row].copy()
    valid_frame["record_hash"] = _build_record_hash(valid_frame)
    valid_frame["dq_status"] = "valid"

    rejects_frame = typed_frame.loc[~is_valid_row].copy()
    rejects_frame["dq_status"] = "reject"
    rejects_frame["dq_reason"] = dq_reason.loc[~is_valid_row].astype("string")
    rejects_frame["dq_reason"] = rejects_frame.apply(
        lambda row: (
            f"{row['dq_reason']}|source_selected_{row['selected_source']}"
            if _clean_text(row.get("selected_source", ""))
            else row["dq_reason"]
        ),
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    valid_frame.to_csv(output_path, index=False, encoding="utf-8")
    rejects_frame.to_csv(rejects_path, index=False, encoding="utf-8")

    reject_counter = Counter(reason for reason_list in rejects_frame["dq_reason"] for reason in str(reason_list).split("|"))
    summary = {
        "status": "ok",
        "transform_run_id": execution_run_id,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "rejects_file": str(rejects_path),
        "rows_input": int(len(raw_frame)),
        "rows_valid": int(len(valid_frame)),
        "rows_rejected": int(len(rejects_frame)),
        "reject_rate": float(len(rejects_frame) / len(raw_frame)) if len(raw_frame) else 0.0,
        "distinct_productos_valid": int(valid_frame["producto_canonico"].nunique(dropna=True)),
        "distinct_proveedores_valid": int(valid_frame["proveedor_canonico"].nunique(dropna=True)),
        "distinct_terminales_valid": int(valid_frame["terminal_canonico"].nunique(dropna=True)),
        "top_reject_reasons": dict(reject_counter.most_common(10)),
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transforma ofertas_raw_mincost.csv a ofertas_typed.csv con tipado + DQ."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/public/support/ofertas_reconciled_raw.csv"),
        help="Input crudo reconciliado de ofertas (Tabla + Cálculos).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public/support/ofertas_typed.csv"),
        help="Output de registros válidos de ofertas.",
    )
    parser.add_argument(
        "--rejects",
        type=Path,
        default=Path("data/public/support/ofertas_rejects_typed.csv"),
        help="Output de rechazos DQ en ofertas.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/public/etl_dq_ofertas_typed.json"),
        help="Reporte JSON de calidad de ofertas tipadas.",
    )
    parser.add_argument(
        "--productos-mapping",
        type=Path,
        default=Path("config/productos_mapping_v1.csv"),
        help="Mapping producto raw -> producto canónico.",
    )
    parser.add_argument(
        "--proveedores-mapping",
        type=Path,
        default=Path("config/proveedores_mapping_v1.csv"),
        help="Mapping proveedor raw -> proveedor canónico.",
    )
    parser.add_argument(
        "--terminales-mapping",
        type=Path,
        default=Path("config/terminales_mapping_v1.csv"),
        help="Mapping terminal raw -> terminal canónico.",
    )
    parser.add_argument(
        "--layout-rules",
        type=Path,
        default=Path("config/ofertas_layout_rules.json"),
        help="Reglas de layout y sentinelas inválidos.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador de ejecución ETL opcional.",
    )
    args = parser.parse_args()

    summary = run(
        input_path=args.input,
        output_path=args.output,
        rejects_path=args.rejects,
        report_path=args.report,
        productos_mapping_path=args.productos_mapping,
        proveedores_mapping_path=args.proveedores_mapping,
        terminales_mapping_path=args.terminales_mapping,
        layout_rules_path=args.layout_rules,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
