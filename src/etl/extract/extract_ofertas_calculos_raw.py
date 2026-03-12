from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )


def _normalize_text(value: Any) -> str:
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


def _parse_cost(value: Any, invalid_tokens: set[str]) -> float | None:
    text = _clean_text(value)
    if not text:
        return None

    text_norm = _normalize_text(text)
    if text_norm in invalid_tokens:
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


def _parse_date_from_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")

    text = _clean_text(value)
    if not text:
        return None
    if not re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text):
        return None
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _parse_date_from_filename(file_path: Path, patterns: list[str]) -> str | None:
    file_name = file_path.name
    for pattern in patterns:
        match = re.search(pattern, file_name)
        if not match:
            continue
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        if year < 100:
            year += 2000 if year < 70 else 1900
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _extract_fecha(raw_frame: pd.DataFrame, file_path: Path, rules: dict) -> str | None:
    for row_idx in range(min(20, len(raw_frame))):
        for value in raw_frame.iloc[row_idx].tolist():
            parsed = _parse_date_from_value(value)
            if parsed:
                return parsed
    return _parse_date_from_filename(file_path, rules.get("filename_date_patterns", []))


def _find_row_contains(normalized_frame: pd.DataFrame, token: str, start_row: int = 0) -> int | None:
    token_norm = _normalize_text(token)
    for row_index in range(start_row, len(normalized_frame)):
        if normalized_frame.iloc[row_index].str.contains(re.escape(token_norm), regex=True).any():
            return row_index
    return None


def _find_best_product_row(
    normalized_frame: pd.DataFrame,
    product_tokens: list[str],
    start_row: int,
    end_row: int,
) -> int | None:
    token_set = {_normalize_text(token) for token in product_tokens}
    best_row = None
    best_hits = 0
    for row_index in range(max(0, start_row), min(end_row, len(normalized_frame) - 1) + 1):
        row_values = normalized_frame.iloc[row_index]
        hits = int(row_values.isin(token_set).sum())
        if hits > best_hits:
            best_hits = hits
            best_row = row_index
    if best_row is None or best_hits < 2:
        return None
    return best_row


def _detect_product_columns(
    raw_frame: pd.DataFrame,
    product_row_index: int,
    product_tokens: list[str],
) -> list[dict]:
    token_lookup = {_normalize_text(token): token for token in product_tokens}
    product_columns: list[dict] = []
    for col_idx, value in raw_frame.iloc[product_row_index].items():
        product_norm = _normalize_text(value)
        if product_norm in token_lookup:
            product_columns.append(
                {
                    "col_idx": int(col_idx),
                    "producto_raw": token_lookup[product_norm],
                }
            )
    return product_columns


def _detect_provider_columns(
    raw_frame: pd.DataFrame,
    normalized_frame: pd.DataFrame,
    provider_row_index: int,
    min_product_col: int,
    product_tokens: list[str],
) -> list[dict]:
    token_set = {_normalize_text(token) for token in product_tokens}
    provider_columns: list[dict] = []
    for col_idx, value in raw_frame.iloc[provider_row_index].items():
        if int(col_idx) < min_product_col:
            continue
        provider_raw = _clean_text(value)
        provider_norm = _normalize_text(value)
        if not provider_raw:
            continue
        if provider_norm in token_set:
            continue
        provider_columns.append(
            {
                "col_idx": int(col_idx),
                "proveedor_raw": provider_raw,
            }
        )
    return provider_columns


def _find_provider_row(
    raw_frame: pd.DataFrame,
    normalized_frame: pd.DataFrame,
    product_row_index: int,
    min_product_col: int,
    product_tokens: list[str],
    search_window: int,
) -> int | None:
    token_set = {_normalize_text(token) for token in product_tokens}
    start_row = max(0, product_row_index - search_window)
    end_row = max(0, product_row_index - 1)
    best_row = None
    best_score = 0
    for row_index in range(start_row, end_row + 1):
        row_values = normalized_frame.iloc[row_index]
        if int(row_values.isin(token_set).sum()) >= 2:
            continue
        row_raw = raw_frame.iloc[row_index]
        non_empty = 0
        for col_idx, value in row_raw.items():
            if int(col_idx) < min_product_col:
                continue
            if _clean_text(value):
                non_empty += 1
        if non_empty > best_score:
            best_score = non_empty
            best_row = row_index
    return best_row


def _match_provider_for_product(product_col: int, provider_columns: list[dict]) -> dict | None:
    eligible = [provider for provider in provider_columns if provider["col_idx"] <= product_col]
    if not eligible:
        return None
    return max(eligible, key=lambda provider: provider["col_idx"])


def _find_terminal_in_row(raw_frame: pd.DataFrame, row_index: int, terminal_pattern: re.Pattern[str]) -> tuple[int | None, str]:
    for col_idx in range(min(8, raw_frame.shape[1])):
        terminal_raw = _clean_text(raw_frame.iloc[row_index, col_idx])
        if terminal_raw and terminal_pattern.search(terminal_raw):
            return col_idx, terminal_raw
    return None, ""


def _extract_rows_from_file(
    file_path: Path,
    sheet_name: str,
    raw_frame: pd.DataFrame,
    normalized_frame: pd.DataFrame,
    rules: dict,
    run_id: str,
    ingestion_ts_utc: str,
) -> tuple[list[dict], list[dict]]:
    output_rows: list[dict] = []
    reject_rows: list[dict] = []
    invalid_cost_count = 0
    invalid_cost_examples = 0

    fecha_raw = _extract_fecha(raw_frame, file_path, rules)
    if not fecha_raw:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "row_idx": None,
                "col_idx": None,
                "terminal_raw": "",
                "producto_raw": "",
                "proveedor_raw": "",
                "coste_raw": "",
                "reject_reason": "missing_fecha",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return output_rows, reject_rows

    anchors = rules["anchors"]
    price_row = _find_row_contains(normalized_frame, anchors["price_with_transport_token"])
    if price_row is None:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "row_idx": None,
                "col_idx": None,
                "terminal_raw": "",
                "producto_raw": "",
                "proveedor_raw": "",
                "coste_raw": "",
                "reject_reason": "missing_price_anchor",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return output_rows, reject_rows

    min_price_row = _find_row_contains(
        normalized_frame,
        anchors["min_price_token"],
        start_row=price_row,
    )
    if min_price_row is None:
        min_price_row = min(price_row + 12, len(raw_frame) - 1)

    product_row = _find_best_product_row(
        normalized_frame=normalized_frame,
        product_tokens=rules["product_tokens"],
        start_row=max(0, price_row - rules.get("product_row_search_window", 4)),
        end_row=price_row,
    )
    if product_row is None:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "row_idx": None,
                "col_idx": None,
                "terminal_raw": "",
                "producto_raw": "",
                "proveedor_raw": "",
                "coste_raw": "",
                "reject_reason": "missing_product_row",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return output_rows, reject_rows

    product_columns = _detect_product_columns(
        raw_frame=raw_frame,
        product_row_index=product_row,
        product_tokens=rules["product_tokens"],
    )
    if not product_columns:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "row_idx": None,
                "col_idx": None,
                "terminal_raw": "",
                "producto_raw": "",
                "proveedor_raw": "",
                "coste_raw": "",
                "reject_reason": "missing_product_columns",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return output_rows, reject_rows

    min_product_col = min(item["col_idx"] for item in product_columns)
    provider_row = _find_provider_row(
        raw_frame=raw_frame,
        normalized_frame=normalized_frame,
        product_row_index=product_row,
        min_product_col=min_product_col,
        product_tokens=rules["product_tokens"],
        search_window=rules.get("provider_row_search_window", 4),
    )
    if provider_row is None:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "row_idx": None,
                "col_idx": None,
                "terminal_raw": "",
                "producto_raw": "",
                "proveedor_raw": "",
                "coste_raw": "",
                "reject_reason": "missing_provider_row",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return output_rows, reject_rows

    provider_columns = _detect_provider_columns(
        raw_frame=raw_frame,
        normalized_frame=normalized_frame,
        provider_row_index=provider_row,
        min_product_col=min_product_col,
        product_tokens=rules["product_tokens"],
    )
    if not provider_columns:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "row_idx": None,
                "col_idx": None,
                "terminal_raw": "",
                "producto_raw": "",
                "proveedor_raw": "",
                "coste_raw": "",
                "reject_reason": "missing_provider_columns",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return output_rows, reject_rows

    invalid_cost_tokens = {_normalize_text(token) for token in rules.get("invalid_cost_tokens", [])}
    terminal_pattern = re.compile(rules["terminal_regex"], flags=re.IGNORECASE)

    for row_index in range(price_row, min_price_row):
        terminal_col, terminal_raw = _find_terminal_in_row(
            raw_frame=raw_frame,
            row_index=row_index,
            terminal_pattern=terminal_pattern,
        )
        if terminal_col is None:
            continue

        for product_column in product_columns:
            provider_match = _match_provider_for_product(
                product_col=product_column["col_idx"],
                provider_columns=provider_columns,
            )
            if provider_match is None:
                reject_rows.append(
                    {
                        "source_file": str(file_path),
                        "sheet_name": sheet_name,
                        "row_idx": int(row_index),
                        "col_idx": int(product_column["col_idx"]),
                        "terminal_raw": terminal_raw,
                        "producto_raw": product_column["producto_raw"],
                        "proveedor_raw": "",
                        "coste_raw": _clean_text(raw_frame.iloc[row_index, product_column["col_idx"]]),
                        "reject_reason": "missing_provider_for_product_column",
                        "run_id": run_id,
                        "ingestion_ts_utc": ingestion_ts_utc,
                    }
                )
                continue

            proveedor_raw = provider_match["proveedor_raw"]
            coste_cell_value = raw_frame.iloc[row_index, product_column["col_idx"]]
            coste_raw = _clean_text(coste_cell_value)
            coste_numeric = _parse_cost(coste_cell_value, invalid_tokens=invalid_cost_tokens)
            if coste_numeric is None:
                invalid_cost_count += 1
                if invalid_cost_examples < 5:
                    reject_rows.append(
                        {
                            "source_file": str(file_path),
                            "sheet_name": sheet_name,
                            "row_idx": int(row_index),
                            "col_idx": int(product_column["col_idx"]),
                            "terminal_raw": terminal_raw,
                            "producto_raw": product_column["producto_raw"],
                            "proveedor_raw": proveedor_raw,
                            "coste_raw": coste_raw,
                            "reject_reason": "invalid_cost",
                            "run_id": run_id,
                            "ingestion_ts_utc": ingestion_ts_utc,
                        }
                    )
                    invalid_cost_examples += 1
                continue

            output_rows.append(
                {
                    "fecha_raw": fecha_raw,
                    "terminal_raw": terminal_raw,
                    "producto_raw": product_column["producto_raw"],
                    "proveedor_raw": proveedor_raw,
                    "coste_raw": str(coste_numeric),
                    "source_file": str(file_path),
                    "sheet_name": sheet_name,
                    "row_idx": int(row_index),
                    "col_idx": int(product_column["col_idx"]),
                    "row_provider_header": int(provider_row),
                    "row_product_header": int(product_row),
                    "row_price_anchor": int(price_row),
                    "run_id": run_id,
                    "ingestion_ts_utc": ingestion_ts_utc,
                }
            )

    if not output_rows:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "row_idx": None,
                "col_idx": None,
                "terminal_raw": "",
                "producto_raw": "",
                "proveedor_raw": "",
                "coste_raw": "",
                "reject_reason": "no_valid_cost_rows",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
    if invalid_cost_count > 0:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "row_idx": None,
                "col_idx": None,
                "terminal_raw": "",
                "producto_raw": "",
                "proveedor_raw": "",
                "coste_raw": "",
                "reject_reason": "invalid_cost_aggregated",
                "invalid_cost_count": int(invalid_cost_count),
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
    return output_rows, reject_rows


def _read_calculos_sheet(file_path: Path, preferred_sheet_names: list[str]) -> tuple[pd.DataFrame | None, str, str | None]:
    engine = "xlrd" if file_path.suffix.lower() == ".xls" else "openpyxl"
    workbook = pd.ExcelFile(file_path, engine=engine)
    selected_sheet = None
    for candidate in preferred_sheet_names:
        if candidate in workbook.sheet_names:
            selected_sheet = candidate
            break
    if selected_sheet is None:
        return None, engine, None
    frame = pd.read_excel(workbook, sheet_name=selected_sheet, header=None, dtype=object)
    return frame, engine, selected_sheet


def run(
    sources_config_path: Path,
    layout_rules_path: Path,
    output_raw_long_path: Path,
    output_rejects_path: Path,
    report_path: Path,
    run_id: str | None = None,
) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    sources_config = _load_json(sources_config_path)
    rules = _load_json(layout_rules_path)

    offers_config = sources_config["sources"]["ofertas"]
    raw_root = repo_root / sources_config["raw_root"]
    excluded_prefixes = offers_config.get("exclude_name_prefixes", [])

    run_id_value = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ingestion_ts_utc = datetime.now(timezone.utc).isoformat()

    source_files: list[Path] = []
    for glob_pattern in offers_config.get("globs", []):
        for candidate in raw_root.glob(glob_pattern):
            if not candidate.is_file():
                continue
            if any(candidate.name.startswith(prefix) for prefix in excluded_prefixes):
                continue
            source_files.append(candidate)
    source_files = sorted(set(source_files))

    all_rows: list[dict] = []
    all_rejects: list[dict] = []
    files_ok = 0
    files_rejected = 0

    for file_path in source_files:
        try:
            raw_frame, engine_used, selected_sheet = _read_calculos_sheet(
                file_path=file_path,
                preferred_sheet_names=rules.get("preferred_sheet_names", ["Cálculos", "Calculos"]),
            )
            if raw_frame is None or selected_sheet is None:
                files_rejected += 1
                all_rejects.append(
                    {
                        "source_file": str(file_path),
                        "sheet_name": None,
                        "row_idx": None,
                        "col_idx": None,
                        "terminal_raw": "",
                        "producto_raw": "",
                        "proveedor_raw": "",
                        "coste_raw": "",
                        "reject_reason": "no_calculos_sheet",
                        "engine": engine_used,
                        "run_id": run_id_value,
                        "ingestion_ts_utc": ingestion_ts_utc,
                    }
                )
                continue

            normalized_frame = raw_frame.fillna("").astype(str).apply(
                lambda column: column.map(_normalize_text)
            )
            extracted_rows, reject_rows = _extract_rows_from_file(
                file_path=file_path,
                sheet_name=selected_sheet,
                raw_frame=raw_frame,
                normalized_frame=normalized_frame,
                rules=rules,
                run_id=run_id_value,
                ingestion_ts_utc=ingestion_ts_utc,
            )
            all_rows.extend(extracted_rows)
            all_rejects.extend(reject_rows)
            if extracted_rows:
                files_ok += 1
            else:
                files_rejected += 1
        except Exception as error:
            files_rejected += 1
            all_rejects.append(
                {
                    "source_file": str(file_path),
                    "sheet_name": None,
                    "row_idx": None,
                    "col_idx": None,
                    "terminal_raw": "",
                    "producto_raw": "",
                    "proveedor_raw": "",
                    "coste_raw": "",
                    "reject_reason": "file_processing_error",
                    "error_message": str(error)[:500],
                    "run_id": run_id_value,
                    "ingestion_ts_utc": ingestion_ts_utc,
                }
            )

    output_raw_long_path.parent.mkdir(parents=True, exist_ok=True)
    output_rejects_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_rows).to_csv(output_raw_long_path, index=False, encoding="utf-8")
    pd.DataFrame(all_rejects).to_csv(output_rejects_path, index=False, encoding="utf-8")

    reject_counter = Counter(row.get("reject_reason", "unknown") for row in all_rejects)
    report = {
        "status": "ok",
        "run_id": run_id_value,
        "files_total": len(source_files),
        "files_ok": files_ok,
        "files_rejected": files_rejected,
        "coverage_ok_ratio": (files_ok / len(source_files)) if source_files else 0.0,
        "rows_raw_long": len(all_rows),
        "rows_rejects": len(all_rejects),
        "top_reject_reasons": dict(reject_counter.most_common(10)),
        "output_raw_long": str(output_raw_long_path),
        "output_rejects": str(output_rejects_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae ofertas desde hoja Cálculos/Calculos en formato largo."
    )
    parser.add_argument(
        "--sources-config",
        type=Path,
        default=Path("config/etl_sources.json"),
        help="Ruta de config de fuentes ETL.",
    )
    parser.add_argument(
        "--layout-rules",
        type=Path,
        default=Path("config/ofertas_calculos_layout_rules.json"),
        help="Ruta de reglas de layout de Cálculos.",
    )
    parser.add_argument(
        "--output-raw-long",
        type=Path,
        default=Path("data/public/support/ofertas_calculos_raw_long.csv"),
        help="Salida staging de ofertas largas desde Cálculos.",
    )
    parser.add_argument(
        "--output-rejects",
        type=Path,
        default=Path("data/public/support/ofertas_calculos_rejects.csv"),
        help="Salida staging de rechazos del extractor Cálculos.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/public/etl_dq_ofertas_calculos_extract.json"),
        help="Reporte DQ del extractor Cálculos.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador opcional de ejecución.",
    )
    args = parser.parse_args()

    report = run(
        sources_config_path=args.sources_config,
        layout_rules_path=args.layout_rules,
        output_raw_long_path=args.output_raw_long,
        output_rejects_path=args.output_rejects,
        report_path=args.report,
        run_id=args.run_id,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
