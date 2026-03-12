from __future__ import annotations

import argparse
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


def _terminal_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", _normalize_text(value))


def _read_sheet(file_path: Path, preferred_sheet_names: list[str]) -> tuple[pd.DataFrame | None, str, str | None]:
    engine = "xlrd" if file_path.suffix.lower() == ".xls" else "openpyxl"
    workbook = pd.ExcelFile(file_path, engine=engine)
    selected_sheet = None
    for candidate in preferred_sheet_names:
        if candidate in workbook.sheet_names:
            selected_sheet = candidate
            break
    if selected_sheet is None:
        return None, engine, None
    frame = pd.read_excel(workbook, sheet_name=selected_sheet, header=None, dtype=str)
    return frame, engine, selected_sheet


def _normalize_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    text_frame = raw_frame.fillna("").astype(str)
    return text_frame.apply(lambda column: column.map(_normalize_text))


def _find_row_contains(normalized_frame: pd.DataFrame, token: str) -> int | None:
    token_norm = _normalize_text(token)
    for row_index in range(len(normalized_frame)):
        if normalized_frame.iloc[row_index].str.contains(re.escape(token_norm), regex=True).any():
            return row_index
    return None


def _find_best_product_row(
    normalized_frame: pd.DataFrame,
    product_tokens: list[str],
    start_row: int = 0,
    end_row: int | None = None,
) -> int | None:
    end_limit = end_row if end_row is not None else len(normalized_frame) - 1
    token_set = {_normalize_text(token) for token in product_tokens}
    best_row = None
    best_hits = 0
    for row_index in range(start_row, end_limit + 1):
        row_values = normalized_frame.iloc[row_index]
        hits = int(row_values.isin(token_set).sum())
        if hits > best_hits:
            best_hits = hits
            best_row = row_index
    if best_row is None or best_hits < 2:
        return None
    return best_row


# SECTION: Date resolution
def _parse_candidate_date(value: Any) -> pd.Timestamp | None:
    """Parse a candidate date value from the workbook row while preserving invalids as null."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


# SECTION: Date resolution
def _extract_path_year_hint(file_path: Path) -> int | None:
    """Extract the last explicit 20xx year from the file path for audit purposes."""
    matches = re.findall(r"(20\d{2})", str(file_path))
    if not matches:
        return None
    return int(matches[-1])


# SECTION: Date resolution
def _extract_file_date_hint(file_path: Path, path_year_hint: int | None) -> pd.Timestamp | None:
    """Extract a deterministic date hint from the file name using dd-mm-yy/dd-mm-yyyy patterns."""
    match = re.search(r"(?<!\d)(\d{1,2})[-_/.](\d{1,2})[-_/.](\d{2,4})(?!\d)", file_path.stem)
    if not match:
        return None

    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))
    if year < 100:
        if path_year_hint is not None and path_year_hint % 100 == year:
            year = path_year_hint
        else:
            year = 2000 + year if year <= 25 else 1900 + year

    try:
        return pd.Timestamp(datetime(year, month, day)).normalize()
    except ValueError:
        return None


# SECTION: Date resolution
def _extract_date_from_row(raw_frame: pd.DataFrame, date_row_index: int) -> tuple[str | None, str]:
    """Return the first parseable row date candidate plus the raw token that produced it."""
    for value in raw_frame.iloc[date_row_index].tolist():
        parsed = _parse_candidate_date(value)
        if parsed is not None:
            return parsed.strftime("%Y-%m-%d"), _clean_text(value)
    return None, ""


# SECTION: Date resolution
def _resolve_fecha_oferta(file_path: Path, raw_frame: pd.DataFrame, date_row_index: int) -> dict[str, Any]:
    """Resolve the final offer date using row date and file-name hint without changing day/month conflicts."""
    row_date, row_date_raw_value = _extract_date_from_row(raw_frame=raw_frame, date_row_index=date_row_index)
    path_year_hint = _extract_path_year_hint(file_path=file_path)
    file_date_hint_ts = _extract_file_date_hint(file_path=file_path, path_year_hint=path_year_hint)
    file_date_hint = file_date_hint_ts.strftime("%Y-%m-%d") if file_date_hint_ts is not None else None

    row_date_ts = pd.to_datetime(row_date, errors="coerce") if row_date else pd.NaT
    row_date_ts = row_date_ts.normalize() if pd.notna(row_date_ts) else pd.NaT

    selected_date = None
    resolution_status = "missing_date"
    date_autocorrected_flag = False
    day_month_conflict_flag = False

    if pd.notna(row_date_ts) and file_date_hint_ts is not None:
        same_day_month = (
            int(row_date_ts.day) == int(file_date_hint_ts.day)
            and int(row_date_ts.month) == int(file_date_hint_ts.month)
        )
        if same_day_month and int(row_date_ts.year) != int(file_date_hint_ts.year):
            selected_date = file_date_hint
            resolution_status = "auto_correct_year_from_file_name"
            date_autocorrected_flag = True
        elif same_day_month:
            selected_date = row_date
            resolution_status = "row_date_matches_file_name"
        else:
            selected_date = row_date
            resolution_status = "row_date_kept_day_month_conflict"
            day_month_conflict_flag = True
    elif pd.notna(row_date_ts):
        selected_date = row_date
        resolution_status = "row_date_only"
    elif file_date_hint is not None:
        selected_date = file_date_hint
        resolution_status = "file_name_fallback_missing_row_date"

    selected_year = int(selected_date[:4]) if selected_date else None
    row_year = int(row_date[:4]) if row_date else None

    return {
        "fecha_raw": selected_date,
        "fecha_row_candidate": row_date,
        "fecha_file_hint": file_date_hint,
        "fecha_row_raw_value": row_date_raw_value,
        "path_year_hint": path_year_hint,
        "row_year": row_year,
        "selected_year": selected_year,
        "date_resolution_status": resolution_status,
        "date_autocorrected_flag": date_autocorrected_flag,
        "day_month_conflict_flag": day_month_conflict_flag,
    }


def _detect_product_columns(
    raw_frame: pd.DataFrame,
    product_row_index: int,
    product_tokens: list[str],
) -> list[tuple[int, str]]:
    token_lookup = {_normalize_text(token): token for token in product_tokens}
    product_columns: list[tuple[int, str]] = []
    for column_index, raw_value in raw_frame.iloc[product_row_index].items():
        token_norm = _normalize_text(raw_value)
        if token_norm in token_lookup:
            product_columns.append((int(column_index), token_lookup[token_norm]))
    return product_columns


def _find_provider_header_row(
    raw_frame: pd.DataFrame,
    normalized_frame: pd.DataFrame,
    date_row_index: int,
    global_product_row_index: int,
    product_tokens: list[str],
) -> int:
    token_set = {_normalize_text(token) for token in product_tokens}
    start_row = max(0, date_row_index - 4)
    end_row = min(global_product_row_index, date_row_index + 2)

    best_row = date_row_index
    best_score = -1
    for row_index in range(start_row, end_row + 1):
        row_norm = normalized_frame.iloc[row_index]
        product_hits = int(row_norm.isin(token_set).sum())
        if product_hits >= 2:
            continue
        row_raw = raw_frame.iloc[row_index]
        non_empty_score = sum(
            1 for col_index, value in row_raw.items() if int(col_index) >= 8 and _clean_text(value)
        )
        if non_empty_score > best_score:
            best_score = non_empty_score
            best_row = row_index
    return best_row


def _build_provider_blocks(
    raw_frame: pd.DataFrame,
    provider_header_row: int,
    global_product_row: int,
    product_tokens: list[str],
) -> list[dict]:
    token_set = {_normalize_text(token) for token in product_tokens}
    provider_cells: list[dict] = []
    row_raw = raw_frame.iloc[provider_header_row]

    for col_index, value in row_raw.items():
        col_value = _clean_text(value)
        if not col_value or int(col_index) < 8:
            continue
        value_norm = _normalize_text(col_value)
        if value_norm == "FECHA":
            continue
        if value_norm in token_set:
            continue
        provider_cells.append(
            {
                "col_index": int(col_index),
                "provider_raw": col_value,
                "provider_norm": value_norm,
            }
        )

    if not provider_cells:
        return []

    blocks: list[dict] = []
    max_col = raw_frame.shape[1]
    for idx, provider_cell in enumerate(provider_cells):
        start_col = provider_cell["col_index"]
        end_col = provider_cells[idx + 1]["col_index"] if idx + 1 < len(provider_cells) else max_col
        product_to_col: dict[str, int] = {}
        for col_index in range(start_col, end_col):
            product_norm = _normalize_text(raw_frame.iloc[global_product_row, col_index])
            if product_norm in token_set and product_norm not in product_to_col:
                product_to_col[product_norm] = int(col_index)
        if not product_to_col:
            continue
        blocks.append(
            {
                "provider_raw": provider_cell["provider_raw"],
                "provider_norm": provider_cell["provider_norm"],
                "start_col": start_col,
                "end_col": end_col,
                "product_to_col": product_to_col,
            }
        )
    return blocks


def _find_terminal_label(raw_frame: pd.DataFrame, row_index: int, max_col: int) -> tuple[int | None, str]:
    terminal_col = None
    terminal_raw = ""
    for col_index in range(max_col):
        value = _clean_text(raw_frame.iloc[row_index, col_index])
        if value:
            terminal_col = col_index
            terminal_raw = value
    return terminal_col, terminal_raw


def _extract_summary_terminal_rows(
    raw_frame: pd.DataFrame,
    product_columns: list[tuple[int, str]],
    start_row: int,
    terminal_pattern: re.Pattern[str],
    product_tokens: list[str],
) -> list[dict]:
    rows: list[dict] = []
    if not product_columns:
        return rows

    first_product_col = min(column for column, _ in product_columns)
    product_norm_set = {_normalize_text(token) for token in product_tokens}
    for row_index in range(start_row, len(raw_frame)):
        terminal_col, terminal_raw = _find_terminal_label(raw_frame, row_index, first_product_col)
        if terminal_col is None:
            continue
        if not terminal_pattern.search(terminal_raw):
            continue

        provider_cells: list[dict] = []
        for product_col, product_raw in product_columns:
            provider_raw = _clean_text(raw_frame.iloc[row_index, product_col])
            provider_norm = _normalize_text(provider_raw)
            if not provider_raw:
                continue
            if provider_norm in product_norm_set:
                continue
            provider_cells.append(
                {
                    "col_producto": int(product_col),
                    "producto_raw": product_raw,
                    "proveedor_min_raw": provider_raw,
                }
            )

        if provider_cells:
            rows.append(
                {
                    "row_terminal": int(row_index),
                    "terminal_col": int(terminal_col),
                    "terminal_raw": terminal_raw,
                    "provider_cells": provider_cells,
                    "terminal_norm": _normalize_text(terminal_raw),
                    "terminal_key": _terminal_key(terminal_raw),
                }
            )
    return rows


def _build_upper_terminal_rows(
    raw_frame: pd.DataFrame,
    start_row: int,
    end_row: int,
    max_terminal_col: int,
    terminal_pattern: re.Pattern[str],
) -> list[dict]:
    rows: list[dict] = []
    for row_index in range(start_row, end_row + 1):
        terminal_col, terminal_raw = _find_terminal_label(raw_frame, row_index, max_terminal_col)
        if terminal_col is None:
            continue
        if not terminal_pattern.search(terminal_raw):
            continue
        rows.append(
            {
                "row_idx": int(row_index),
                "terminal_col": int(terminal_col),
                "terminal_raw": terminal_raw,
                "terminal_norm": _normalize_text(terminal_raw),
                "terminal_key": _terminal_key(terminal_raw),
            }
        )
    return rows


def _match_provider_block(provider_raw: str, provider_blocks: list[dict]) -> dict | None:
    provider_norm = _normalize_text(provider_raw)
    if not provider_norm:
        return None

    for block in provider_blocks:
        if provider_norm == block["provider_norm"]:
            return block
    for block in provider_blocks:
        block_norm = block["provider_norm"]
        if provider_norm in block_norm or block_norm in provider_norm:
            return block
    provider_first_token = provider_norm.split(" ")[0]
    if provider_first_token:
        for block in provider_blocks:
            if block["provider_norm"].split(" ")[0] == provider_first_token:
                return block
    return None


def _match_terminal_row(summary_terminal: dict, upper_terminal_rows: list[dict]) -> dict | None:
    key = summary_terminal["terminal_key"]
    norm = summary_terminal["terminal_norm"]

    for row in upper_terminal_rows:
        if row["terminal_key"] == key:
            return row
    for row in upper_terminal_rows:
        if row["terminal_norm"] == norm:
            return row
    for row in upper_terminal_rows:
        if key and (key in row["terminal_key"] or row["terminal_key"] in key):
            return row
    return None


def _extract_cost_value(
    raw_frame: pd.DataFrame,
    summary_terminal: dict,
    product_raw: str,
    provider_raw: str,
    provider_blocks: list[dict],
    upper_terminal_rows: list[dict],
    cost_row_index: int | None,
) -> tuple[str, str, int | None]:
    product_norm = _normalize_text(product_raw)
    provider_block = _match_provider_block(provider_raw, provider_blocks)
    if provider_block is None:
        return "", "missing_provider_block", None

    product_col = provider_block["product_to_col"].get(product_norm)
    if product_col is None:
        return "", "missing_product_col_in_provider_block", None

    matched_terminal = _match_terminal_row(summary_terminal, upper_terminal_rows)
    if matched_terminal is not None:
        cost_raw = _clean_text(raw_frame.iloc[matched_terminal["row_idx"], product_col])
        if cost_raw:
            return cost_raw, "upper_terminal_cost", int(product_col)

    if cost_row_index is not None:
        cost_raw = _clean_text(raw_frame.iloc[cost_row_index, product_col])
        if cost_raw:
            return cost_raw, "min_cost_anchor_fallback", int(product_col)

    return "", "missing_cost_value", int(product_col)


# SECTION: File extraction
def _build_rows_for_file(
    file_path: Path,
    sheet_name: str,
    raw_frame: pd.DataFrame,
    normalized_frame: pd.DataFrame,
    rules: dict,
    run_id: str,
    ingestion_ts_utc: str,
) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    """Extract min-cost rows, traceable matrix cells, rejects, and date audit for a single workbook."""
    mincost_rows: list[dict] = []
    matrix_rows: list[dict] = []
    reject_rows: list[dict] = []
    date_audit: dict[str, Any] = {
        "source_file": str(file_path),
        "sheet_name": sheet_name,
        "fecha_raw": None,
        "fecha_row_candidate": None,
        "fecha_file_hint": None,
        "fecha_row_raw_value": "",
        "path_year_hint": None,
        "row_year": None,
        "selected_year": None,
        "date_resolution_status": "missing_date_anchor",
        "date_autocorrected_flag": False,
        "day_month_conflict_flag": False,
        "run_id": run_id,
        "ingestion_ts_utc": ingestion_ts_utc,
    }

    anchors = rules["anchors"]
    date_row_index = _find_row_contains(normalized_frame, rules["date_anchor_token"])
    if date_row_index is None:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "reject_reason": "missing_date_anchor",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return mincost_rows, matrix_rows, reject_rows, date_audit

    date_audit = {
        **date_audit,
        **_resolve_fecha_oferta(file_path=file_path, raw_frame=raw_frame, date_row_index=date_row_index),
        "date_anchor_row": int(date_row_index),
    }
    fecha_raw = date_audit["fecha_raw"]
    if fecha_raw is None:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "reject_reason": "invalid_date_value",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return mincost_rows, matrix_rows, reject_rows, date_audit

    section_row_index = _find_row_contains(normalized_frame, anchors["terminal_section_token"])
    if section_row_index is None:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "reject_reason": "missing_terminal_section_anchor",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return mincost_rows, matrix_rows, reject_rows, date_audit

    summary_product_row = _find_best_product_row(
        normalized_frame=normalized_frame,
        product_tokens=rules["product_tokens"],
        start_row=section_row_index + 1,
        end_row=min(section_row_index + 8, len(normalized_frame) - 1),
    )
    if summary_product_row is None:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "reject_reason": "missing_summary_product_row",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return mincost_rows, matrix_rows, reject_rows, date_audit

    summary_product_columns = _detect_product_columns(
        raw_frame=raw_frame,
        product_row_index=summary_product_row,
        product_tokens=rules["product_tokens"],
    )
    if not summary_product_columns:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "reject_reason": "missing_summary_product_columns",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return mincost_rows, matrix_rows, reject_rows, date_audit

    global_product_row = _find_best_product_row(
        normalized_frame=normalized_frame,
        product_tokens=rules["product_tokens"],
    )
    if global_product_row is None:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "reject_reason": "missing_global_product_row",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return mincost_rows, matrix_rows, reject_rows, date_audit

    provider_header_row = _find_provider_header_row(
        raw_frame=raw_frame,
        normalized_frame=normalized_frame,
        date_row_index=date_row_index,
        global_product_row_index=global_product_row,
        product_tokens=rules["product_tokens"],
    )
    provider_blocks = _build_provider_blocks(
        raw_frame=raw_frame,
        provider_header_row=provider_header_row,
        global_product_row=global_product_row,
        product_tokens=rules["product_tokens"],
    )

    cost_row_index = _find_row_contains(normalized_frame, anchors["min_cost_row_token"])
    provider_anchor_row = _find_row_contains(normalized_frame, anchors["provider_row_token"])

    terminal_pattern = re.compile(rules["terminal_regex"], flags=re.IGNORECASE)
    summary_terminal_rows = _extract_summary_terminal_rows(
        raw_frame=raw_frame,
        product_columns=summary_product_columns,
        start_row=summary_product_row + 1,
        terminal_pattern=terminal_pattern,
        product_tokens=rules["product_tokens"],
    )
    if not summary_terminal_rows:
        reject_rows.append(
            {
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "reject_reason": "missing_summary_terminal_rows",
                "run_id": run_id,
                "ingestion_ts_utc": ingestion_ts_utc,
            }
        )
        return mincost_rows, matrix_rows, reject_rows, date_audit

    global_product_columns = _detect_product_columns(
        raw_frame=raw_frame,
        product_row_index=global_product_row,
        product_tokens=rules["product_tokens"],
    )
    first_global_product_col = min(column for column, _ in global_product_columns) if global_product_columns else 8

    upper_end_row = (cost_row_index - 1) if cost_row_index is not None else (section_row_index - 1)
    upper_terminal_rows = _build_upper_terminal_rows(
        raw_frame=raw_frame,
        start_row=global_product_row + 1,
        end_row=max(global_product_row + 1, upper_end_row),
        max_terminal_col=max(first_global_product_col, 8),
        terminal_pattern=terminal_pattern,
    )

    for terminal_info in summary_terminal_rows:
        for provider_cell in terminal_info["provider_cells"]:
            cost_raw, cost_source, cost_lookup_col = _extract_cost_value(
                raw_frame=raw_frame,
                summary_terminal=terminal_info,
                product_raw=provider_cell["producto_raw"],
                provider_raw=provider_cell["proveedor_min_raw"],
                provider_blocks=provider_blocks,
                upper_terminal_rows=upper_terminal_rows,
                cost_row_index=cost_row_index,
            )
            mincost_rows.append(
                {
                    "fecha_raw": fecha_raw,
                    "terminal_raw": terminal_info["terminal_raw"],
                    "producto_raw": provider_cell["producto_raw"],
                    "proveedor_min_raw": provider_cell["proveedor_min_raw"],
                    "coste_min_raw": cost_raw,
                    "source_file": str(file_path),
                    "sheet_name": sheet_name,
                    "row_terminal": terminal_info["row_terminal"],
                    "col_producto": provider_cell["col_producto"],
                    "row_cost_anchor": int(cost_row_index) if cost_row_index is not None else None,
                    "row_product_header": int(summary_product_row),
                    "provider_header_row": int(provider_header_row),
                    "cost_lookup_col": cost_lookup_col,
                    "cost_source": cost_source,
                    "fecha_row_candidate": date_audit["fecha_row_candidate"],
                    "fecha_file_hint": date_audit["fecha_file_hint"],
                    "fecha_row_raw_value": date_audit["fecha_row_raw_value"],
                    "path_year_hint": date_audit["path_year_hint"],
                    "date_resolution_status": date_audit["date_resolution_status"],
                    "date_autocorrected_flag": date_audit["date_autocorrected_flag"],
                    "day_month_conflict_flag": date_audit["day_month_conflict_flag"],
                    "run_id": run_id,
                    "ingestion_ts_utc": ingestion_ts_utc,
                }
            )

    summary_product_col_set = {column for column, _ in summary_product_columns}
    global_product_col_set = {column for column, _ in global_product_columns}

    summary_terminal_map = {row["row_terminal"]: row["terminal_col"] for row in summary_terminal_rows}
    upper_terminal_map = {row["row_idx"]: row["terminal_col"] for row in upper_terminal_rows}

    selected_rows = {date_row_index, provider_header_row, global_product_row, summary_product_row, section_row_index}
    if provider_anchor_row is not None:
        selected_rows.add(provider_anchor_row)
    if cost_row_index is not None:
        selected_rows.add(cost_row_index)
    selected_rows.update(summary_terminal_map.keys())
    selected_rows.update(upper_terminal_map.keys())

    for row_index in sorted(selected_rows):
        row_data = raw_frame.iloc[row_index]
        for col_index, cell_value in row_data.items():
            col_idx = int(col_index)
            clean_value = _clean_text(cell_value)
            if not clean_value:
                continue
            context_tag = "other"
            if row_index == date_row_index:
                context_tag = "date_anchor"
            elif row_index == provider_header_row:
                context_tag = "provider_header"
            elif row_index == global_product_row and col_idx in global_product_col_set:
                context_tag = "product_header_global"
            elif row_index == summary_product_row and col_idx in summary_product_col_set:
                context_tag = "product_header_summary"
            elif cost_row_index is not None and row_index == cost_row_index:
                context_tag = "cost_min_anchor"
            elif provider_anchor_row is not None and row_index == provider_anchor_row:
                context_tag = "provider_anchor"
            elif row_index in upper_terminal_map and col_idx == upper_terminal_map[row_index]:
                context_tag = "terminal_label_upper"
            elif row_index in upper_terminal_map and col_idx in global_product_col_set:
                context_tag = "cost_terminal_provider"
            elif row_index in summary_terminal_map and col_idx == summary_terminal_map[row_index]:
                context_tag = "terminal_label_summary"
            elif row_index in summary_terminal_map and col_idx in summary_product_col_set:
                context_tag = "provider_min"

            matrix_rows.append(
                {
                    "source_file": str(file_path),
                    "sheet_name": sheet_name,
                    "row_idx": row_index,
                    "col_idx": col_idx,
                    "cell_value": clean_value,
                    "context_tag": context_tag,
                    "resolved_fecha_oferta": date_audit["fecha_raw"],
                    "fecha_row_candidate": date_audit["fecha_row_candidate"],
                    "fecha_file_hint": date_audit["fecha_file_hint"],
                    "date_resolution_status": date_audit["date_resolution_status"],
                    "date_autocorrected_flag": date_audit["date_autocorrected_flag"],
                    "run_id": run_id,
                    "ingestion_ts_utc": ingestion_ts_utc,
                }
            )

    return mincost_rows, matrix_rows, reject_rows, date_audit


# SECTION: Pipeline execution
def run(
    sources_config_path: Path,
    layout_rules_path: Path,
    output_mincost_path: Path,
    output_matrix_path: Path,
    output_rejects_path: Path,
    report_path: Path,
    run_id: str | None = None,
) -> dict:
    """Run the raw extractor over all offer workbooks and persist outputs plus date-resolution audit."""
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

    all_mincost_rows: list[dict] = []
    all_matrix_rows: list[dict] = []
    all_reject_rows: list[dict] = []
    all_date_audit_rows: list[dict[str, Any]] = []
    files_ok = 0
    files_rejected = 0

    for file_path in source_files:
        try:
            raw_frame, engine_used, selected_sheet = _read_sheet(
                file_path=file_path,
                preferred_sheet_names=rules["preferred_sheet_names"],
            )
            if raw_frame is None or selected_sheet is None:
                files_rejected += 1
                all_reject_rows.append(
                    {
                        "source_file": str(file_path),
                        "sheet_name": None,
                        "reject_reason": "no_tabla_sheet",
                        "engine": engine_used,
                        "run_id": run_id_value,
                        "ingestion_ts_utc": ingestion_ts_utc,
                    }
                )
                continue

            normalized_frame = _normalize_frame(raw_frame)
            mincost_rows, matrix_rows, reject_rows, date_audit = _build_rows_for_file(
                file_path=file_path,
                sheet_name=selected_sheet,
                raw_frame=raw_frame,
                normalized_frame=normalized_frame,
                rules=rules,
                run_id=run_id_value,
                ingestion_ts_utc=ingestion_ts_utc,
            )

            all_mincost_rows.extend(mincost_rows)
            all_matrix_rows.extend(matrix_rows)
            all_date_audit_rows.append(date_audit)
            if reject_rows:
                files_rejected += 1
                all_reject_rows.extend(reject_rows)
            else:
                files_ok += 1
        except Exception as error:
            files_rejected += 1
            all_reject_rows.append(
                {
                    "source_file": str(file_path),
                    "sheet_name": None,
                    "reject_reason": "file_processing_error",
                    "error_message": str(error)[:500],
                    "run_id": run_id_value,
                    "ingestion_ts_utc": ingestion_ts_utc,
                }
            )
            all_date_audit_rows.append(
                {
                    "source_file": str(file_path),
                    "sheet_name": None,
                    "fecha_raw": None,
                    "fecha_row_candidate": None,
                    "fecha_file_hint": None,
                    "fecha_row_raw_value": "",
                    "path_year_hint": _extract_path_year_hint(file_path=file_path),
                    "row_year": None,
                    "selected_year": None,
                    "date_resolution_status": "file_processing_error",
                    "date_autocorrected_flag": False,
                    "day_month_conflict_flag": False,
                    "run_id": run_id_value,
                    "ingestion_ts_utc": ingestion_ts_utc,
                }
            )

    output_mincost_path.parent.mkdir(parents=True, exist_ok=True)
    output_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    output_rejects_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_mincost_rows).to_csv(output_mincost_path, index=False, encoding="utf-8")
    pd.DataFrame(all_matrix_rows).to_csv(output_matrix_path, index=False, encoding="utf-8")
    pd.DataFrame(all_reject_rows).to_csv(output_rejects_path, index=False, encoding="utf-8")

    reject_counter = Counter(row["reject_reason"] for row in all_reject_rows)
    date_audit_df = pd.DataFrame(all_date_audit_rows)
    corrected_df = date_audit_df[date_audit_df["date_autocorrected_flag"].fillna(False)].copy()
    unresolved_df = date_audit_df[date_audit_df["day_month_conflict_flag"].fillna(False)].copy()

    def _year_breakdown(from_col: str, to_col: str) -> dict[str, int]:
        if date_audit_df.empty or from_col not in date_audit_df or to_col not in date_audit_df:
            return {}
        valid = date_audit_df[date_audit_df[from_col].notna() & date_audit_df[to_col].notna()].copy()
        if valid.empty:
            return {}
        labels = (
            valid[from_col].astype(int).astype(str)
            + "->"
            + valid[to_col].astype(int).astype(str)
        )
        return dict(labels.value_counts().to_dict())

    report = {
        "status": "ok",
        "run_id": run_id_value,
        "files_total": len(source_files),
        "files_ok": files_ok,
        "files_rejected": files_rejected,
        "coverage_ok_ratio": (files_ok / len(source_files)) if source_files else 0.0,
        "rows_mincost": len(all_mincost_rows),
        "rows_matrix_cells": len(all_matrix_rows),
        "rows_rejects": len(all_reject_rows),
        "top_reject_reasons": dict(reject_counter.most_common(10)),
        "output_mincost": str(output_mincost_path),
        "output_matrix": str(output_matrix_path),
        "output_rejects": str(output_rejects_path),
        "date_resolution_audit": {
            "files_with_date_audit": int(len(date_audit_df)),
            "autocorrected_files": int(corrected_df["source_file"].nunique()) if not corrected_df.empty else 0,
            "unresolved_day_month_conflicts": int(unresolved_df["source_file"].nunique()) if not unresolved_df.empty else 0,
            "resolution_status_counts": (
                date_audit_df["date_resolution_status"].value_counts(dropna=False).to_dict()
                if not date_audit_df.empty
                else {}
            ),
            "path_year_to_row_year": _year_breakdown("path_year_hint", "row_year"),
            "path_year_to_selected_year": _year_breakdown("path_year_hint", "selected_year"),
            "corrected_files": corrected_df[
                [
                    "source_file",
                    "fecha_row_candidate",
                    "fecha_file_hint",
                    "fecha_raw",
                    "path_year_hint",
                    "date_resolution_status",
                ]
            ].to_dict(orient="records")
            if not corrected_df.empty
            else [],
            "unresolved_conflicts": unresolved_df[
                [
                    "source_file",
                    "fecha_row_candidate",
                    "fecha_file_hint",
                    "fecha_raw",
                    "path_year_hint",
                    "date_resolution_status",
                ]
            ].to_dict(orient="records")
            if not unresolved_df.empty
            else [],
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# SECTION: CLI
def main() -> None:
    """Parse CLI args and run the raw offers extractor."""
    parser = argparse.ArgumentParser(
        description="Extractor V0 híbrido de ofertas: señales mínimas + matriz cruda trazable."
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
        default=Path("config/ofertas_layout_rules.json"),
        help="Ruta del layout de ofertas.",
    )
    parser.add_argument(
        "--output-mincost",
        type=Path,
        default=Path("data/public/support/ofertas_raw_mincost.csv"),
        help="Salida principal de mincost por fecha-terminal-producto.",
    )
    parser.add_argument(
        "--output-matrix",
        type=Path,
        default=Path("data/public/support/ofertas_raw_matrix_cells.csv"),
        help="Salida raw por celdas para trazabilidad.",
    )
    parser.add_argument(
        "--output-rejects",
        type=Path,
        default=Path("data/public/support/ofertas_rejects_extract.csv"),
        help="Salida de rechazos de extracción.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/public/etl_dq_ofertas_tabla_extract.json"),
        help="Reporte DQ de extracción.",
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
        output_mincost_path=args.output_mincost,
        output_matrix_path=args.output_matrix,
        output_rejects_path=args.output_rejects,
        report_path=args.report,
        run_id=args.run_id,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
