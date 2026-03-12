#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

MULTI_TERMINAL_TOKEN = "MULTI_TERMINAL_AGGREGATE"
UNKNOWN_TERMINAL_TOKEN = "UNKNOWN_TERMINAL_CONTEXT"


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 04.1 transport parser."""
    parser = argparse.ArgumentParser(
        description="Parsea señales de transporte desde ofertas_raw_matrix_cells.csv."
    )
    parser.add_argument(
        "--matrix-input",
        type=Path,
        default=Path("data/public/support/ofertas_raw_matrix_cells.csv"),
        help="Input staging con celdas seleccionadas del raw de comparativas.",
    )
    parser.add_argument(
        "--layout-rules",
        type=Path,
        default=Path("config/ofertas_layout_rules.json"),
        help="Reglas de layout de la hoja Tabla.",
    )
    parser.add_argument(
        "--productos-mapping",
        type=Path,
        default=Path("config/productos_mapping_v1.csv"),
        help="Mapping de producto raw a producto canónico.",
    )
    parser.add_argument(
        "--proveedores-mapping",
        type=Path,
        default=Path("config/proveedores_mapping_v1.csv"),
        help="Mapping de proveedor raw a proveedor canónico.",
    )
    parser.add_argument(
        "--terminales-mapping",
        type=Path,
        default=Path("config/terminales_mapping_v1.csv"),
        help="Mapping de terminal raw a terminal canónico.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/public/support/ofertas_transport_signals.csv"),
        help="Output CSV con señales de transporte parseadas.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("artifacts/public/transport_parser_day041.json"),
        help="Reporte JSON del parser de transporte.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id opcional para trazabilidad.",
    )
    return parser.parse_args()


# SECTION: Shared helpers
def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file from disk using UTF-8 encoding."""
    return json.loads(path.read_text(encoding="utf-8"))


# SECTION: Shared helpers
def _clean_text(value: Any) -> str:
    """Trim raw values and collapse repeated whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


# SECTION: Shared helpers
def _strip_accents(text: str) -> str:
    """Remove accents so raw comparisons are layout-stable."""
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )


# SECTION: Shared helpers
def _normalize_text(value: Any) -> str:
    """Normalize text for matching across historical spreadsheet variants."""
    return re.sub(r"\s+", " ", _strip_accents(_clean_text(value)).upper())


# SECTION: Shared helpers
def _parse_number(value: Any, invalid_tokens: set[str]) -> float | None:
    """Parse positive numeric values while filtering layout sentinels like 999999."""
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


# SECTION: Shared helpers
def _load_mapping(mapping_path: Path, raw_column: str, canonical_column: str) -> dict[str, str]:
    """Load a canonical mapping CSV and return a normalized lookup dict."""
    mapping: dict[str, str] = {}
    with mapping_path.open("r", encoding="utf-8") as mapping_file:
        reader = csv.reader(mapping_file)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Mapping vacío: {mapping_path}")
        if header[0].strip() != raw_column or header[-1].strip() != canonical_column:
            raise ValueError(
                f"Mapping inválido en {mapping_path}. Se esperaban columnas `{raw_column}` y `{canonical_column}`."
            )
        for row in reader:
            if len(row) < 2:
                continue
            raw_value = ",".join(row[:-1]).strip()
            canonical_value = row[-1].strip()
            raw_norm = _normalize_text(raw_value)
            canonical_clean = _clean_text(canonical_value)
            if raw_norm and canonical_clean:
                mapping[raw_norm] = canonical_clean
    return mapping


# SECTION: Shared helpers
def _parse_date_from_row(date_cells: list[str]) -> str | None:
    """Parse the operational date from the selected FECHA anchor row."""
    for value in date_cells:
        text = _clean_text(value)
        if not text:
            continue
        if re.match(r"^\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?$", text):
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
        else:
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
    return None


# SECTION: Matrix parsing
def _resolve_group_fecha_oferta(group: pd.DataFrame, row_values: dict[int, dict[int, str]], date_row_idx: int) -> str | None:
    """Resolve the operational date using the corrected extractor output when available."""
    if "resolved_fecha_oferta" in group.columns:
        resolved_series = group["resolved_fecha_oferta"].astype("string").dropna()
        resolved_series = resolved_series[resolved_series.str.strip() != ""]
        if not resolved_series.empty:
            return str(resolved_series.iloc[0]).strip()
    return _parse_date_from_row(date_cells=list(row_values[date_row_idx].values()))


# SECTION: Matrix parsing
def _build_row_map(group: pd.DataFrame) -> tuple[dict[int, dict[int, str]], dict[int, dict[int, str]]]:
    """Build sparse row maps for fast cell lookup and context inspection."""
    row_values: dict[int, dict[int, str]] = {}
    row_contexts: dict[int, dict[int, str]] = {}
    for row in group.itertuples(index=False):
        row_idx = int(row.row_idx)
        col_idx = int(row.col_idx)
        row_values.setdefault(row_idx, {})[col_idx] = _clean_text(row.cell_value)
        row_contexts.setdefault(row_idx, {})[col_idx] = _clean_text(row.context_tag)
    return row_values, row_contexts


# SECTION: Matrix parsing
def _find_row_by_context(row_contexts: dict[int, dict[int, str]], context_tag: str) -> int | None:
    """Return the first row index that contains a given context tag."""
    for row_idx in sorted(row_contexts):
        if context_tag in row_contexts[row_idx].values():
            return row_idx
    return None


# SECTION: Matrix parsing
def _resolve_provider_header_row(
    row_contexts: dict[int, dict[int, str]],
    date_row_idx: int | None,
) -> int | None:
    """Resolve the effective provider header row for layouts stored in the sparse matrix artifact."""
    provider_row_idx = _find_row_by_context(row_contexts=row_contexts, context_tag="provider_header")
    if provider_row_idx is not None:
        return provider_row_idx
    if date_row_idx is not None:
        return date_row_idx
    return _find_row_by_context(row_contexts=row_contexts, context_tag="provider_anchor")


# SECTION: Matrix parsing
def _build_provider_blocks(
    provider_row_values: dict[int, str],
    product_row_values: dict[int, str],
    product_tokens: list[str],
) -> list[dict[str, Any]]:
    """Build provider blocks by combining provider headers with repeated product headers."""
    token_set = {_normalize_text(token) for token in product_tokens}
    provider_cells: list[dict[str, Any]] = []
    for col_idx, raw_value in sorted(provider_row_values.items()):
        if col_idx < 8:
            continue
        value_norm = _normalize_text(raw_value)
        if value_norm in {"", "FECHA"} or value_norm in token_set:
            continue
        provider_cells.append(
            {
                "col_idx": int(col_idx),
                "provider_raw": _clean_text(raw_value),
            }
        )

    if not provider_cells:
        return []

    max_col = max(product_row_values) + 1
    product_token_lookup = {_normalize_text(token): token for token in product_tokens}
    blocks: list[dict[str, Any]] = []
    for index, provider_cell in enumerate(provider_cells):
        start_col = provider_cell["col_idx"]
        end_col = provider_cells[index + 1]["col_idx"] if index + 1 < len(provider_cells) else max_col
        product_to_col: dict[str, dict[str, Any]] = {}
        for col_idx in range(start_col, end_col):
            product_norm = _normalize_text(product_row_values.get(col_idx, ""))
            if product_norm in product_token_lookup:
                product_to_col[product_norm] = {
                    "col_idx": int(col_idx),
                    "producto_raw": product_token_lookup[product_norm],
                }
        if product_to_col:
            blocks.append(
                {
                    "provider_raw": provider_cell["provider_raw"],
                    "start_col": int(start_col),
                    "end_col": int(end_col),
                    "product_to_col": product_to_col,
                }
            )
    return blocks


# SECTION: Matrix parsing
def _extract_terminal_rows(row_contexts: dict[int, dict[int, str]], row_values: dict[int, dict[int, str]]) -> list[dict[str, Any]]:
    """Extract upper terminal labels selected by the raw matrix extractor."""
    terminal_rows: list[dict[str, Any]] = []
    for row_idx in sorted(row_contexts):
        for col_idx, context_tag in row_contexts[row_idx].items():
            if context_tag != "terminal_label_upper":
                continue
            terminal_rows.append(
                {
                    "row_idx": int(row_idx),
                    "col_idx": int(col_idx),
                    "terminal_raw": _clean_text(row_values[row_idx].get(col_idx, "")),
                }
            )
    return terminal_rows


# SECTION: Matrix parsing
def _resolve_terminal_context(
    anchor_row_idx: int,
    previous_anchor_row_idx: int,
    terminal_rows: list[dict[str, Any]],
    terminal_mapping: dict[str, str],
) -> tuple[str, str, str]:
    """Resolve whether a transport row maps to one terminal or to a multi-terminal aggregate."""
    candidates = [
        row for row in terminal_rows if previous_anchor_row_idx < row["row_idx"] < anchor_row_idx
    ]
    candidate_terminals = sorted({_clean_text(row["terminal_raw"]) for row in candidates if _clean_text(row["terminal_raw"])})
    if len(candidate_terminals) == 1:
        terminal_raw = candidate_terminals[0]
        terminal_canonico = terminal_mapping.get(_normalize_text(terminal_raw), "UNKNOWN")
        return terminal_raw, terminal_canonico, "parsed_unique_terminal"
    if len(candidate_terminals) > 1:
        return MULTI_TERMINAL_TOKEN, MULTI_TERMINAL_TOKEN, "parsed_multi_terminal_aggregate"
    return UNKNOWN_TERMINAL_TOKEN, UNKNOWN_TERMINAL_TOKEN, "parsed_missing_terminal_context"


# SECTION: Matrix parsing
def _extract_transport_rows_for_group(
    group: pd.DataFrame,
    invalid_tokens: set[str],
    product_tokens: list[str],
    product_mapping: dict[str, str],
    provider_mapping: dict[str, str],
    terminal_mapping: dict[str, str],
    parser_run_id: str,
    parser_ts_utc: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse transport costs for one raw file/sheet group using the sparse matrix artifact."""
    row_values, row_contexts = _build_row_map(group=group)
    date_row_idx = _find_row_by_context(row_contexts=row_contexts, context_tag="date_anchor")
    provider_row_idx = _resolve_provider_header_row(row_contexts=row_contexts, date_row_idx=date_row_idx)
    global_product_row_idx = _find_row_by_context(row_contexts=row_contexts, context_tag="product_header_global")
    anchor_rows = sorted(
        row_idx for row_idx, contexts in row_contexts.items() if "cost_min_anchor" in contexts.values()
    )
    terminal_rows = _extract_terminal_rows(row_contexts=row_contexts, row_values=row_values)

    profile = {
        "anchors_detected": anchor_rows,
        "provider_row_idx": provider_row_idx,
        "global_product_row_idx": global_product_row_idx,
        "date_row_idx": date_row_idx,
        "upper_terminal_rows": [row["row_idx"] for row in terminal_rows],
        "rows_extracted": 0,
    }

    if date_row_idx is None or provider_row_idx is None or global_product_row_idx is None or not anchor_rows:
        return [], profile

    fecha_oferta = _resolve_group_fecha_oferta(group=group, row_values=row_values, date_row_idx=date_row_idx)
    if fecha_oferta is None:
        return [], profile

    provider_blocks = _build_provider_blocks(
        provider_row_values=row_values[provider_row_idx],
        product_row_values=row_values[global_product_row_idx],
        product_tokens=product_tokens,
    )
    if not provider_blocks:
        return [], profile

    rows: list[dict[str, Any]] = []
    previous_anchor_row_idx = global_product_row_idx
    file_path = str(group["source_file"].iloc[0])
    sheet_name = str(group["sheet_name"].iloc[0])
    file_suffix = Path(file_path).suffix.lower().replace(".", "")

    for anchor_row_idx in anchor_rows:
        terminal_raw, terminal_canonico, parser_status = _resolve_terminal_context(
            anchor_row_idx=anchor_row_idx,
            previous_anchor_row_idx=previous_anchor_row_idx,
            terminal_rows=terminal_rows,
            terminal_mapping=terminal_mapping,
        )
        layout_variant = f"tabla_anchor_row_{anchor_row_idx}"

        for provider_block in provider_blocks:
            provider_raw = provider_block["provider_raw"]
            proveedor_candidato = provider_mapping.get(_normalize_text(provider_raw), "UNKNOWN")
            for product_meta in provider_block["product_to_col"].values():
                col_idx = int(product_meta["col_idx"])
                producto_raw = product_meta["producto_raw"]
                producto_canonico = product_mapping.get(_normalize_text(producto_raw), "UNKNOWN")
                raw_value = _clean_text(row_values.get(anchor_row_idx, {}).get(col_idx, ""))
                numeric_value = _parse_number(value=raw_value, invalid_tokens=invalid_tokens)
                if numeric_value is None:
                    continue
                rows.append(
                    {
                        "source_file": file_path,
                        "sheet_name": sheet_name,
                        "source_file_type": file_suffix,
                        "fecha_oferta": fecha_oferta,
                        "layout_variant": layout_variant,
                        "anchor_row_idx": int(anchor_row_idx),
                        "provider_header_row_idx": int(provider_row_idx),
                        "product_header_row_idx": int(global_product_row_idx),
                        "terminal_raw": terminal_raw,
                        "terminal_canonico": terminal_canonico,
                        "producto_raw": producto_raw,
                        "producto_canonico": producto_canonico,
                        "proveedor_raw": provider_raw,
                        "proveedor_candidato": proveedor_candidato,
                        "transport_cost_value": float(numeric_value),
                        "parser_status": parser_status,
                        "parser_run_id": parser_run_id,
                        "parser_ts_utc": parser_ts_utc,
                    }
                )

        previous_anchor_row_idx = anchor_row_idx

    profile["rows_extracted"] = int(len(rows))
    return rows, profile


# SECTION: Reporting
def _build_parser_report(
    *,
    parsed_rows: pd.DataFrame,
    file_profiles: list[dict[str, Any]],
    matrix_input_path: Path,
    output_csv_path: Path,
    layout_rules_path: Path,
    parser_run_id: str,
    parser_ts_utc: str,
) -> dict[str, Any]:
    """Build the transport parser JSON report used by notebook 12 and Day 04.1 quality checks."""
    if parsed_rows.empty:
        status_counts: dict[str, int] = {}
        layout_counts: dict[str, int] = {}
    else:
        status_counts = parsed_rows["parser_status"].value_counts(dropna=False).to_dict()
        layout_counts = parsed_rows["layout_variant"].value_counts(dropna=False).to_dict()

    anchor_profile_rows: list[dict[str, Any]] = []
    for profile in file_profiles:
        for anchor_row_idx in profile["anchors_detected"]:
            anchor_profile_rows.append(
                {
                    "anchor_row_idx": int(anchor_row_idx),
                    "source_file": profile["source_file"],
                    "sheet_name": profile["sheet_name"],
                    "rows_extracted": int(profile["rows_extracted"]),
                }
            )

    return {
        "status": "ok",
        "parser_run_id": parser_run_id,
        "parser_ts_utc": parser_ts_utc,
        "matrix_input": str(matrix_input_path),
        "layout_rules": str(layout_rules_path),
        "output_csv": str(output_csv_path),
        "rows_output": int(len(parsed_rows)),
        "files_parsed": int(len(file_profiles)),
        "distinct_source_files_with_output": int(parsed_rows["source_file"].nunique()) if not parsed_rows.empty else 0,
        "status_counts": status_counts,
        "layout_variant_counts": layout_counts,
        "xls_rows": int(parsed_rows["source_file_type"].eq("xls").sum()) if not parsed_rows.empty else 0,
        "xlsx_rows": int(parsed_rows["source_file_type"].eq("xlsx").sum()) if not parsed_rows.empty else 0,
        "rows_with_unknown_provider": int(parsed_rows["proveedor_candidato"].eq("UNKNOWN").sum()) if not parsed_rows.empty else 0,
        "rows_with_unknown_product": int(parsed_rows["producto_canonico"].eq("UNKNOWN").sum()) if not parsed_rows.empty else 0,
        "rows_with_unknown_terminal": int(parsed_rows["terminal_canonico"].eq(UNKNOWN_TERMINAL_TOKEN).sum()) if not parsed_rows.empty else 0,
        "rows_with_multi_terminal_aggregate": int(
            parsed_rows["terminal_canonico"].eq(MULTI_TERMINAL_TOKEN).sum()
        ) if not parsed_rows.empty else 0,
        "anchor_profiles": anchor_profile_rows[:200],
        "sample_rows": parsed_rows.head(20).to_dict(orient="records") if not parsed_rows.empty else [],
    }


# SECTION: Main pipeline
def run(
    *,
    matrix_input_path: Path,
    layout_rules_path: Path,
    productos_mapping_path: Path,
    proveedores_mapping_path: Path,
    terminales_mapping_path: Path,
    output_csv_path: Path,
    report_json_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Parse transport rows from the raw matrix artifact and persist a structured staging output."""
    if not matrix_input_path.exists():
        raise FileNotFoundError(f"No existe input matrix: {matrix_input_path}")

    layout_rules = _load_json(layout_rules_path)
    product_tokens = [str(token) for token in layout_rules["product_tokens"]]
    invalid_tokens = {_normalize_text(token) for token in layout_rules["invalid_cost_tokens"]}
    product_mapping = _load_mapping(productos_mapping_path, "raw_value", "producto_canonico")
    provider_mapping = _load_mapping(proveedores_mapping_path, "raw_value", "proveedor_canonico")
    terminal_mapping = _load_mapping(terminales_mapping_path, "raw_value", "terminal_canonico")

    parser_run_id = run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser_ts_utc = datetime.now(timezone.utc).isoformat()

    matrix_df = pd.read_csv(matrix_input_path, dtype=str, keep_default_na=False)
    matrix_df["row_idx"] = pd.to_numeric(matrix_df["row_idx"], errors="coerce").fillna(-1).astype(int)
    matrix_df["col_idx"] = pd.to_numeric(matrix_df["col_idx"], errors="coerce").fillna(-1).astype(int)

    parsed_rows: list[dict[str, Any]] = []
    file_profiles: list[dict[str, Any]] = []
    for (source_file, sheet_name), group in matrix_df.groupby(["source_file", "sheet_name"], sort=False):
        group_rows, profile = _extract_transport_rows_for_group(
            group=group.copy(),
            invalid_tokens=invalid_tokens,
            product_tokens=product_tokens,
            product_mapping=product_mapping,
            provider_mapping=provider_mapping,
            terminal_mapping=terminal_mapping,
            parser_run_id=parser_run_id,
            parser_ts_utc=parser_ts_utc,
        )
        profile.update({"source_file": str(source_file), "sheet_name": str(sheet_name)})
        file_profiles.append(profile)
        parsed_rows.extend(group_rows)

    parsed_df = pd.DataFrame(parsed_rows)
    if parsed_df.empty:
        parsed_df = pd.DataFrame(
            columns=[
                "source_file",
                "sheet_name",
                "source_file_type",
                "fecha_oferta",
                "layout_variant",
                "anchor_row_idx",
                "provider_header_row_idx",
                "product_header_row_idx",
                "terminal_raw",
                "terminal_canonico",
                "producto_raw",
                "producto_canonico",
                "proveedor_raw",
                "proveedor_candidato",
                "transport_cost_value",
                "parser_status",
                "parser_run_id",
                "parser_ts_utc",
            ]
        )
    else:
        parsed_df = parsed_df.sort_values(
            ["fecha_oferta", "producto_canonico", "proveedor_candidato", "layout_variant", "source_file"]
        ).reset_index(drop=True)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_df.to_csv(output_csv_path, index=False, encoding="utf-8")

    report_payload = _build_parser_report(
        parsed_rows=parsed_df,
        file_profiles=file_profiles,
        matrix_input_path=matrix_input_path,
        output_csv_path=output_csv_path,
        layout_rules_path=layout_rules_path,
        parser_run_id=parser_run_id,
        parser_ts_utc=parser_ts_utc,
    )
    report_json_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_payload


# SECTION: CLI entrypoint
def main() -> None:
    """Run the CLI entrypoint for the Day 04.1 transport parser."""
    args = parse_args()
    summary = run(
        matrix_input_path=args.matrix_input,
        layout_rules_path=args.layout_rules,
        productos_mapping_path=args.productos_mapping,
        proveedores_mapping_path=args.proveedores_mapping,
        terminales_mapping_path=args.terminales_mapping,
        output_csv_path=args.output_csv,
        report_json_path=args.report_json,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
