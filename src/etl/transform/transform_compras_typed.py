from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _to_snake(column_name: str) -> str:
    """Convierte nombres de columna a snake_case simple."""
    normalized = column_name.strip().lower()
    normalized = re.sub(r"[^\w]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _normalize_schema(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas para no depender de mayúsculas/espacios."""
    output_frame = raw_frame.copy()
    output_frame.columns = [_to_snake(str(column_name)) for column_name in output_frame.columns]
    return output_frame


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """
    Devuelve la primera columna existente entre varios candidatos.
    Si no existe ninguna, devuelve una serie vacía para mantener el pipeline estable.
    """
    for column_name in candidates:
        if column_name in frame.columns:
            return frame[column_name]
    return pd.Series([""] * len(frame), index=frame.index, dtype="string")


def _clean_text(series: pd.Series) -> pd.Series:
    """Limpieza básica de strings para reducir ruido en joins posteriores."""
    text_series = series.astype("string").str.strip()
    return (
        text_series.replace(
            {
                "": pd.NA,
                "-": pd.NA,
                "--": pd.NA,
                "nan": pd.NA,
                "none": pd.NA,
                "null": pd.NA,
                "  -   -": pd.NA,
            },
            regex=False,
        )
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _parse_number(series: pd.Series) -> pd.Series:
    """
    Parsea números permitiendo formato europeo/mixto.
    Ejemplos soportados: '1.234,56', '1234.56', '1234'.
    """
    def _normalize_token(raw_value: object) -> str | None:
        if raw_value is None:
            return None
        text_value = str(raw_value).strip()
        if text_value == "" or text_value.lower() in {"nan", "none", "null"}:
            return None
        text_value = text_value.replace(" ", "")
        has_dot = "." in text_value
        has_comma = "," in text_value
        if has_dot and has_comma:
            # Caso europeo clásico: 1.234,56
            if text_value.rfind(",") > text_value.rfind("."):
                text_value = text_value.replace(".", "").replace(",", ".")
            else:
                # Caso raro inverso: 1,234.56
                text_value = text_value.replace(",", "")
        elif has_comma and not has_dot:
            # Decimal con coma: 1234,56
            text_value = text_value.replace(",", ".")
        # Si solo hay punto, se conserva para no romper decimales tipo 0.79428.
        text_value = re.sub(r"[^0-9\.-]", "", text_value)
        if text_value in {"", "-", ".", "-."}:
            return None
        return text_value

    normalized_series = series.astype("string").map(_normalize_token)
    return pd.to_numeric(normalized_series, errors="coerce")


def _parse_date(series: pd.Series) -> pd.Series:
    """Parsea fechas y devuelve formato ISO `YYYY-MM-DD` para consistencia."""
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return parsed.dt.strftime("%Y-%m-%d")


def _append_reason(current_reason: pd.Series, mask: pd.Series, reason_name: str) -> pd.Series:
    """Añade razón de rechazo manteniendo histórico de errores por fila."""
    reason_series = current_reason.copy()
    rows_with_reason = mask.fillna(False)
    rows_without_previous_reason = rows_with_reason & reason_series.isna()
    rows_with_previous_reason = rows_with_reason & reason_series.notna()
    reason_series.loc[rows_without_previous_reason] = reason_name
    reason_series.loc[rows_with_previous_reason] = (
        reason_series.loc[rows_with_previous_reason] + "|" + reason_name
    )
    return reason_series


def _build_record_hash(typed_frame: pd.DataFrame) -> pd.Series:
    """Genera hash técnico estable para deduplicación y trazabilidad."""
    hash_base = (
        typed_frame[
            [
                "fecha_compra",
                "albaran_id",
                "linea_id",
                "proveedor_raw",
                "producto_raw",
                "litros",
                "precio_unitario",
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
    run_id: str | None = None,
) -> dict:
    """
    Flujo principal:
    1) lee compras_raw_xls.csv (staging crudo)
    2) selecciona/renombra columnas canónicas
    3) tipa datos (texto/fecha/numérico)
    4) aplica reglas DQ mínimas
    5) escribe compras_typed.csv + compras_rejects.csv + reporte JSON
    """
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    raw_frame = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[])
    raw_frame = _normalize_schema(raw_frame)

    execution_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    transform_timestamp_utc = datetime.now(timezone.utc).isoformat()

    # Selección de columnas con nombres legibles y estables para negocio/ML.
    typed_frame = pd.DataFrame(index=raw_frame.index)
    typed_frame["albaran_id"] = _clean_text(_first_existing_column(raw_frame, ["albaran"]))
    typed_frame["linea_id"] = _clean_text(_first_existing_column(raw_frame, ["linea"]))
    typed_frame["fecha_compra"] = _parse_date(_first_existing_column(raw_frame, ["fecha"]))
    typed_frame["fecha_despacho"] = _parse_date(
        _first_existing_column(raw_frame, ["fechadesp", "fecha_despacho"])
    )
    typed_frame["proveedor_raw"] = _clean_text(_first_existing_column(raw_frame, ["proveedor"]))
    typed_frame["proveedor_nombre_raw"] = _clean_text(_first_existing_column(raw_frame, ["nombre"]))
    typed_frame["producto_raw"] = _clean_text(_first_existing_column(raw_frame, ["articulo"]))
    typed_frame["producto_nombre_raw"] = _clean_text(_first_existing_column(raw_frame, ["descri"]))
    typed_frame["almacen_raw"] = _clean_text(_first_existing_column(raw_frame, ["almacen"]))
    typed_frame["terminal_raw"] = _clean_text(_first_existing_column(raw_frame, ["pcarga", "pca"]))
    typed_frame["ref_proveedor_raw"] = _clean_text(_first_existing_column(raw_frame, ["refprov"]))
    typed_frame["litros"] = _parse_number(_first_existing_column(raw_frame, ["litros", "litrosreal"]))
    typed_frame["precio_unitario"] = _parse_number(_first_existing_column(raw_frame, ["coste", "pvp"]))
    typed_frame["importe_total"] = _parse_number(_first_existing_column(raw_frame, ["total", "importept"]))

    # Metadata técnica para trazabilidad.
    typed_frame["source_file"] = _clean_text(_first_existing_column(raw_frame, ["source_file"]))
    typed_frame["sheet_name"] = _clean_text(_first_existing_column(raw_frame, ["sheet_name"]))
    typed_frame["row_idx"] = _parse_number(_first_existing_column(raw_frame, ["row_idx"]))
    typed_frame["source_run_id"] = _clean_text(_first_existing_column(raw_frame, ["run_id"]))
    typed_frame["source_ingestion_ts_utc"] = _clean_text(
        _first_existing_column(raw_frame, ["ingestion_ts_utc"])
    )
    typed_frame["transform_run_id"] = execution_run_id
    typed_frame["transform_ts_utc"] = transform_timestamp_utc

    # Reglas DQ mínimas obligatorias para no contaminar capas posteriores.
    dq_reason = pd.Series([pd.NA] * len(typed_frame), index=typed_frame.index, dtype="string")
    dq_reason = _append_reason(dq_reason, typed_frame["albaran_id"].isna(), "missing_albaran_id")
    dq_reason = _append_reason(dq_reason, typed_frame["linea_id"].isna(), "missing_linea_id")
    dq_reason = _append_reason(dq_reason, typed_frame["fecha_compra"].isna(), "invalid_fecha_compra")
    dq_reason = _append_reason(dq_reason, typed_frame["proveedor_raw"].isna(), "missing_proveedor_raw")
    dq_reason = _append_reason(dq_reason, typed_frame["producto_raw"].isna(), "missing_producto_raw")
    dq_reason = _append_reason(dq_reason, typed_frame["litros"].isna(), "invalid_litros")
    dq_reason = _append_reason(dq_reason, typed_frame["litros"].fillna(0) <= 0, "non_positive_litros")
    dq_reason = _append_reason(
        dq_reason, typed_frame["precio_unitario"].isna(), "invalid_precio_unitario"
    )
    dq_reason = _append_reason(
        dq_reason,
        typed_frame["precio_unitario"].fillna(0) <= 0,
        "non_positive_precio_unitario",
    )
    dq_reason = _append_reason(dq_reason, typed_frame["importe_total"].isna(), "invalid_importe_total")

    is_valid_row = dq_reason.isna()
    valid_frame = typed_frame.loc[is_valid_row].copy()
    valid_frame["record_hash"] = _build_record_hash(valid_frame)
    valid_frame["dq_status"] = "valid"

    rejects_frame = typed_frame.loc[~is_valid_row].copy()
    rejects_frame["dq_status"] = "reject"
    rejects_frame["dq_reason"] = dq_reason.loc[~is_valid_row].astype("string")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    valid_frame.to_csv(output_path, index=False, encoding="utf-8")
    rejects_frame.to_csv(rejects_path, index=False, encoding="utf-8")

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
        "distinct_proveedores_raw_valid": int(valid_frame["proveedor_raw"].nunique(dropna=True)),
        "distinct_productos_raw_valid": int(valid_frame["producto_raw"].nunique(dropna=True)),
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transforma compras_raw_xls.csv a compras_typed.csv con tipado y reglas DQ."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/public/support/compras_raw_xls.csv"),
        help="Ruta del CSV crudo generado desde compras totales.xls.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public/support/compras_typed.csv"),
        help="Ruta de salida del CSV tipado (registros válidos).",
    )
    parser.add_argument(
        "--rejects",
        type=Path,
        default=Path("data/public/support/compras_rejects.csv"),
        help="Ruta de salida del CSV de rechazos DQ.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/public/etl_dq_compras_typed.json"),
        help="Ruta del reporte de calidad/resumen de transformación.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador de ejecución (opcional).",
    )
    args = parser.parse_args()
    summary = run(
        input_path=args.input,
        output_path=args.output,
        rejects_path=args.rejects,
        report_path=args.report,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
