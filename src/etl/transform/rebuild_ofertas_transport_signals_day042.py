#!/usr/bin/env python3

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.etl.transform import build_ofertas_transport_signals as day041_parser
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.etl.transform import build_ofertas_transport_signals as day041_parser

RAW_EXPLICIT_BUCKET = "exact_raw_match"
PARSER_FIX_BUCKET = "parser_fix_possible"
DETERMINISTIC_BUCKET = "deterministic_rebuild_possible"
HEURISTIC_BUCKET = "heuristic_same_month_only"
NO_RAW_BUCKET = "no_raw_source"

RAW_EXPLICIT_KIND = "raw_explicit"
PARSER_FIX_KIND = "parser_fix"
DETERMINISTIC_KIND = "deterministic_rebuild"
HEURISTIC_KIND = "heuristic_same_month"
MISSING_KIND = "missing"

PROVIDER_SUFFIX_TOKENS = [
    "PETROLEUM",
    "ENERGY",
    "ENERGIA",
    "IBERIA",
    "SPAIN",
    "PREPAGO",
    "OFERTA",
    "FIXPRICEVOLU",
    "B2B",
    "OIL",
]


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 04.2 transport rebuild pipeline."""
    parser = argparse.ArgumentParser(
        description="Reconstruye staging de transporte Day 04.2 con buckets y heuristica same-month."
    )
    parser.add_argument(
        "--transport-input",
        type=Path,
        default=Path("data/public/support/ofertas_transport_signals.csv"),
        help="Input Day 04.1 con senales de transporte parseadas.",
    )
    parser.add_argument(
        "--matrix-input",
        type=Path,
        default=Path("data/public/support/ofertas_raw_matrix_cells.csv"),
        help="Input matrix raw extraido desde comparativas.",
    )
    parser.add_argument(
        "--v2-input",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v2_candidates.csv"),
        help="Dataset operativo V2 para definir el universo Day 04.2.",
    )
    parser.add_argument(
        "--layout-rules",
        type=Path,
        default=Path("config/ofertas_layout_rules.json"),
        help="Reglas de layout usadas por el parser de transporte.",
    )
    parser.add_argument(
        "--productos-mapping",
        type=Path,
        default=Path("config/productos_mapping_v1.csv"),
        help="Mapping de producto raw a producto canonico.",
    )
    parser.add_argument(
        "--proveedores-mapping",
        type=Path,
        default=Path("config/proveedores_mapping_v1.csv"),
        help="Mapping de proveedor raw a proveedor canonico.",
    )
    parser.add_argument(
        "--terminales-mapping",
        type=Path,
        default=Path("config/terminales_mapping_v1.csv"),
        help="Mapping de terminal raw a terminal canonico.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/public/support/ofertas_transport_signals_day042.csv"),
        help="Output staging Day 04.2 reconstruido a grano dia-producto-proveedor.",
    )
    parser.add_argument(
        "--missingness-json",
        type=Path,
        default=Path("artifacts/public/transport_missingness_day042.json"),
        help="Reporte JSON de missingness y buckets Day 04.2.",
    )
    parser.add_argument(
        "--missingness-csv",
        type=Path,
        default=Path("artifacts/public/transport_missingness_day042.csv"),
        help="CSV tabular con bucketizacion Day 04.2.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2028-02-21",
        help="Cutoff temporal oficial para train/test coverage.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id opcional para trazabilidad.",
    )
    return parser.parse_args()


# SECTION: Shared helpers
def _clean_text(value: Any) -> str:
    """Return trimmed text without duplicated whitespace."""
    return day041_parser._clean_text(value)


# SECTION: Shared helpers
def _normalize_text(value: Any) -> str:
    """Return the Day 04.1 normalized token representation."""
    return day041_parser._normalize_text(value)


# SECTION: Shared helpers
def _squash_alnum(value: Any) -> str:
    """Remove separators so provider aliases can match across spacing variants."""
    normalized = unicodedata.normalize("NFKD", _clean_text(value))
    ascii_only = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^A-Z0-9]+", "", ascii_only.upper())


# SECTION: Shared helpers
def _strip_provider_suffixes(token: str) -> str:
    """Strip common commercial suffixes from provider aliases conservatively."""
    result = token
    changed = True
    while changed:
        changed = False
        for suffix in PROVIDER_SUFFIX_TOKENS:
            if result.endswith(suffix) and len(result) > len(suffix) + 1:
                result = result[: -len(suffix)]
                changed = True
    return result


# SECTION: Shared helpers
def _build_provider_alias_lookup(v2_universe: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """Build canonical provider lookup from the operational V2 universe."""
    canonical_providers = sorted(v2_universe["proveedor_candidato"].dropna().astype(str).unique().tolist())
    squash_lookup = {_squash_alnum(provider): provider for provider in canonical_providers}
    return squash_lookup, canonical_providers


# SECTION: Shared helpers
def _resolve_provider_alias(
    raw_provider: Any,
    canonical_lookup: dict[str, str],
    canonical_providers: list[str],
) -> tuple[str, str]:
    """Resolve UNKNOWN provider aliases with deterministic normalization and conservative fuzzy matching."""
    squashed = _squash_alnum(raw_provider)
    if squashed == "":
        return "UNKNOWN", "empty_raw"
    if squashed in canonical_lookup:
        return canonical_lookup[squashed], "exact_squash"

    stripped = _strip_provider_suffixes(squashed)
    if stripped in canonical_lookup:
        return canonical_lookup[stripped], "suffix_strip"

    prefix_matches = [
        provider
        for provider in canonical_providers
        if _squash_alnum(provider).startswith(stripped) or stripped.startswith(_squash_alnum(provider))
    ]
    if stripped != "" and len(prefix_matches) == 1:
        return prefix_matches[0], "prefix_unique"

    ratios = sorted(
        (
            difflib.SequenceMatcher(None, stripped, _squash_alnum(provider)).ratio(),
            provider,
        )
        for provider in canonical_providers
    )
    ratios = list(reversed(ratios))
    if ratios and ratios[0][0] >= 0.94:
        if len(ratios) == 1 or (ratios[0][0] - ratios[1][0]) >= 0.05:
            return ratios[0][1], f"fuzzy_{ratios[0][0]:.3f}"
    return "UNKNOWN", "unresolved"


# SECTION: Shared helpers
def _aggregate_transport_rows(
    frame: pd.DataFrame,
    *,
    date_column: str,
) -> pd.DataFrame:
    """Aggregate transport rows to one day-product-provider record with stable summary stats."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "fecha_evento",
                "producto_canonico",
                "proveedor_candidato",
                "transport_cost_value",
                "transport_cost_mean_day_provider",
                "transport_cost_range_day_provider",
                "transport_observations",
                "transport_unique_terminal_count",
                "transport_multi_terminal_share",
                "source_files_count",
            ]
        )

    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
    working["transport_cost_value"] = pd.to_numeric(working["transport_cost_value"], errors="coerce")
    working = working.dropna(subset=[date_column, "transport_cost_value"]).copy()
    working = working[
        (working["producto_canonico"].astype(str) != "UNKNOWN")
        & (working["proveedor_candidato"].astype(str) != "UNKNOWN")
    ].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "fecha_evento",
                "producto_canonico",
                "proveedor_candidato",
                "transport_cost_value",
                "transport_cost_mean_day_provider",
                "transport_cost_range_day_provider",
                "transport_observations",
                "transport_unique_terminal_count",
                "transport_multi_terminal_share",
                "source_files_count",
            ]
        )

    grouped = (
        working.groupby([date_column, "producto_canonico", "proveedor_candidato"], as_index=False)
        .agg(
            transport_cost_value=("transport_cost_value", "min"),
            transport_cost_mean_day_provider=("transport_cost_value", "mean"),
            transport_cost_max_day_provider=("transport_cost_value", "max"),
            transport_observations=("transport_cost_value", "count"),
            transport_unique_terminal_count=(
                "terminal_canonico",
                lambda s: int(
                    pd.Series(s, dtype="string")
                    .loc[
                        ~pd.Series(s, dtype="string").isin(
                            [day041_parser.MULTI_TERMINAL_TOKEN, day041_parser.UNKNOWN_TERMINAL_TOKEN, "UNKNOWN"]
                        )
                    ]
                    .nunique()
                ),
            ),
            transport_multi_terminal_share=(
                "parser_status",
                lambda s: float(pd.Series(s, dtype="string").eq("parsed_multi_terminal_aggregate").mean()),
            ),
            source_files_count=("source_file", lambda s: int(pd.Series(s, dtype="string").nunique())),
        )
        .rename(columns={date_column: "fecha_evento"})
    )
    grouped["transport_cost_range_day_provider"] = (
        grouped["transport_cost_max_day_provider"] - grouped["transport_cost_value"]
    )
    return grouped.drop(columns=["transport_cost_max_day_provider"])


# SECTION: Shared helpers
def _load_explicit_frames(
    transport_input_path: Path,
    canonical_lookup: dict[str, str],
    canonical_providers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Load explicit transport rows and split them into raw-explicit and parser-fix pools."""
    transport_df = pd.read_csv(transport_input_path, keep_default_na=False)
    explicit_known = transport_df[transport_df["proveedor_candidato"].astype(str) != "UNKNOWN"].copy()
    explicit_known["provider_alias_method"] = "existing_mapping"

    unknown = transport_df[transport_df["proveedor_candidato"].astype(str) == "UNKNOWN"].copy()
    alias_audit_rows: list[dict[str, Any]] = []
    if not unknown.empty:
        alias_resolution = unknown["proveedor_raw"].drop_duplicates().to_frame(name="proveedor_raw")
        alias_resolution[["resolved_provider", "alias_method"]] = alias_resolution["proveedor_raw"].apply(
            lambda value: pd.Series(_resolve_provider_alias(value, canonical_lookup, canonical_providers))
        )
        alias_map = dict(zip(alias_resolution["proveedor_raw"], alias_resolution["resolved_provider"]))
        method_map = dict(zip(alias_resolution["proveedor_raw"], alias_resolution["alias_method"]))
        unknown["proveedor_candidato"] = unknown["proveedor_raw"].map(alias_map).fillna("UNKNOWN")
        unknown["provider_alias_method"] = unknown["proveedor_raw"].map(method_map).fillna("unresolved")
        alias_audit_rows = alias_resolution.to_dict(orient="records")
        explicit_fixed = unknown[unknown["proveedor_candidato"].astype(str) != "UNKNOWN"].copy()
    else:
        explicit_fixed = pd.DataFrame(columns=list(transport_df.columns) + ["provider_alias_method"])

    known_day = _aggregate_transport_rows(frame=explicit_known, date_column="fecha_oferta")
    known_day["transport_source_kind"] = RAW_EXPLICIT_KIND
    known_day["transport_bucket_origin"] = RAW_EXPLICIT_BUCKET

    fixed_day = _aggregate_transport_rows(frame=explicit_fixed, date_column="fecha_oferta")
    fixed_day["transport_source_kind"] = PARSER_FIX_KIND
    fixed_day["transport_bucket_origin"] = PARSER_FIX_BUCKET
    return known_day, fixed_day, alias_audit_rows


# SECTION: Detail reconstruction
def _extract_detail_rows(
    *,
    matrix_input_path: Path,
    layout_rules_path: Path,
    productos_mapping_path: Path,
    proveedores_mapping_path: Path,
    terminales_mapping_path: Path,
    canonical_lookup: dict[str, str],
    canonical_providers: list[str],
    run_id: str,
    ts_utc: str,
) -> pd.DataFrame:
    """Parse raw matrix detail rows by terminal to measure deterministic rebuild opportunities."""
    layout_rules = json.loads(layout_rules_path.read_text(encoding="utf-8"))
    product_tokens = [str(token) for token in layout_rules["product_tokens"]]
    invalid_tokens = {_normalize_text(token) for token in layout_rules["invalid_cost_tokens"]}
    product_mapping = day041_parser._load_mapping(productos_mapping_path, "raw_value", "producto_canonico")
    provider_mapping = day041_parser._load_mapping(proveedores_mapping_path, "raw_value", "proveedor_canonico")
    terminal_mapping = day041_parser._load_mapping(terminales_mapping_path, "raw_value", "terminal_canonico")

    matrix_df = pd.read_csv(matrix_input_path, dtype=str, keep_default_na=False)
    matrix_df["row_idx"] = pd.to_numeric(matrix_df["row_idx"], errors="coerce").fillna(-1).astype(int)
    matrix_df["col_idx"] = pd.to_numeric(matrix_df["col_idx"], errors="coerce").fillna(-1).astype(int)

    detail_rows: list[dict[str, Any]] = []
    for (source_file, sheet_name), group in matrix_df.groupby(["source_file", "sheet_name"], sort=False):
        row_values, row_contexts = day041_parser._build_row_map(group=group.copy())
        date_row_idx = day041_parser._find_row_by_context(row_contexts=row_contexts, context_tag="date_anchor")
        provider_row_idx = day041_parser._resolve_provider_header_row(row_contexts=row_contexts, date_row_idx=date_row_idx)
        global_product_row_idx = day041_parser._find_row_by_context(
            row_contexts=row_contexts,
            context_tag="product_header_global",
        )
        if date_row_idx is None or provider_row_idx is None or global_product_row_idx is None:
            continue
        fecha_oferta = day041_parser._parse_date_from_row(date_cells=list(row_values[date_row_idx].values()))
        if fecha_oferta is None:
            continue

        provider_blocks = day041_parser._build_provider_blocks(
            provider_row_values=row_values[provider_row_idx],
            product_row_values=row_values[global_product_row_idx],
            product_tokens=product_tokens,
        )
        if not provider_blocks:
            continue

        terminal_rows = day041_parser._extract_terminal_rows(row_contexts=row_contexts, row_values=row_values)
        for terminal_meta in terminal_rows:
            row_idx = int(terminal_meta["row_idx"])
            terminal_raw = terminal_meta["terminal_raw"]
            terminal_canonico = terminal_mapping.get(_normalize_text(terminal_raw), "UNKNOWN")
            for provider_block in provider_blocks:
                provider_raw = provider_block["provider_raw"]
                provider_candidato = provider_mapping.get(_normalize_text(provider_raw), "UNKNOWN")
                alias_method = "existing_mapping"
                if provider_candidato == "UNKNOWN":
                    provider_candidato, alias_method = _resolve_provider_alias(
                        provider_raw,
                        canonical_lookup,
                        canonical_providers,
                    )
                for product_meta in provider_block["product_to_col"].values():
                    col_idx = int(product_meta["col_idx"])
                    raw_value = _clean_text(row_values.get(row_idx, {}).get(col_idx, ""))
                    numeric_value = day041_parser._parse_number(value=raw_value, invalid_tokens=invalid_tokens)
                    if numeric_value is None:
                        continue
                    producto_canonico = product_mapping.get(_normalize_text(product_meta["producto_raw"]), "UNKNOWN")
                    detail_rows.append(
                        {
                            "source_file": str(source_file),
                            "sheet_name": str(sheet_name),
                            "fecha_oferta": fecha_oferta,
                            "producto_canonico": producto_canonico,
                            "proveedor_candidato": provider_candidato,
                            "terminal_raw": terminal_raw,
                            "terminal_canonico": terminal_canonico,
                            "parser_status": "detail_terminal_rebuild",
                            "provider_alias_method": alias_method,
                            "transport_cost_value": float(numeric_value),
                            "detail_run_id": run_id,
                            "detail_ts_utc": ts_utc,
                        }
                    )
    return pd.DataFrame(detail_rows)


# SECTION: Shared helpers
def _pick_heuristic_rows(
    universe_keys: pd.DataFrame,
    pool_day_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Pick same-month nearest transport values for keys still missing after exact-date stages."""
    if universe_keys.empty or pool_day_frame.empty:
        return pd.DataFrame(columns=list(universe_keys.columns) + list(pool_day_frame.columns))

    missing = universe_keys.copy()
    missing["target_dt"] = pd.to_datetime(missing["fecha_evento"], errors="coerce")
    missing["target_ym"] = missing["target_dt"].dt.to_period("M").astype("string")

    pool = pool_day_frame.copy()
    pool["source_dt"] = pd.to_datetime(pool["fecha_evento"], errors="coerce")
    pool["source_ym"] = pool["source_dt"].dt.to_period("M").astype("string")
    pool = pool.dropna(subset=["source_dt"]).copy()

    candidates = missing.merge(
        pool,
        left_on=["target_ym", "producto_canonico", "proveedor_candidato"],
        right_on=["source_ym", "producto_canonico", "proveedor_candidato"],
        how="left",
        suffixes=("", "_candidate"),
    )
    candidates = candidates.dropna(subset=["source_dt"]).copy()
    if candidates.empty:
        return pd.DataFrame(columns=list(universe_keys.columns) + list(pool_day_frame.columns))

    candidates["transport_days_gap"] = (
        candidates["target_dt"] - candidates["source_dt"]
    ).abs().dt.days.astype("Int64")
    candidates["transport_lookahead_flag"] = (candidates["source_dt"] > candidates["target_dt"]).astype(int)
    candidates["source_date_for_transport"] = candidates["fecha_evento_candidate"]
    candidates = candidates.sort_values(
        [
            "fecha_evento",
            "producto_canonico",
            "proveedor_candidato",
            "transport_days_gap",
            "transport_lookahead_flag",
            "source_dt",
        ]
    )
    selected = candidates.groupby(
        ["fecha_evento", "producto_canonico", "proveedor_candidato"],
        as_index=False,
    ).first()
    selected["transport_source_kind"] = HEURISTIC_KIND
    selected["transport_bucket_origin"] = HEURISTIC_BUCKET
    return selected


# SECTION: Shared helpers
def _safe_float(value: Any) -> float | None:
    """Convert values to Python floats while preserving missing values as None."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


# SECTION: Shared helpers
def _row_bucket_breakdown(frame: pd.DataFrame, bucket_column: str, group_column: str) -> dict[str, dict[str, int]]:
    """Build nested bucket counts for one categorical slice."""
    if frame.empty:
        return {}
    breakdown = (
        frame.groupby([group_column, bucket_column], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values([group_column, bucket_column], kind="stable")
    )
    output: dict[str, dict[str, int]] = {}
    for row in breakdown.itertuples(index=False):
        output.setdefault(str(getattr(row, group_column)), {})[str(getattr(row, bucket_column))] = int(row.rows)
    return output


# SECTION: Shared helpers
def _top_group_bucket_rows(
    frame: pd.DataFrame,
    *,
    group_column: str,
    bucket_column: str,
    top_n: int,
) -> list[dict[str, Any]]:
    """Build a compact top-N breakdown for high-cardinality groupings like provider."""
    if frame.empty:
        return []
    grouped = (
        frame.groupby([group_column, bucket_column], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", group_column], ascending=[False, True], kind="stable")
        .head(top_n)
    )
    return grouped.to_dict(orient="records")


# SECTION: Main pipeline
def run(
    *,
    transport_input_path: Path,
    matrix_input_path: Path,
    v2_input_path: Path,
    layout_rules_path: Path,
    productos_mapping_path: Path,
    proveedores_mapping_path: Path,
    terminales_mapping_path: Path,
    output_csv_path: Path,
    missingness_json_path: Path,
    missingness_csv_path: Path,
    cutoff_date: str,
    run_id: str,
) -> dict[str, Any]:
    """Rebuild Day 04.2 transport staging, classify missingness buckets, and persist all artifacts."""
    execution_run_id = run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_ts_utc = datetime.now(timezone.utc).isoformat()

    v2_rows = pd.read_csv(
        v2_input_path,
        usecols=["event_id", "fecha_evento", "producto_canonico", "proveedor_candidato"],
        keep_default_na=False,
    )
    universe_keys = (
        v2_rows[["fecha_evento", "producto_canonico", "proveedor_candidato"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    universe_keys["year"] = pd.to_datetime(universe_keys["fecha_evento"], errors="coerce").dt.year.astype("Int64")
    universe_keys["month"] = pd.to_datetime(universe_keys["fecha_evento"], errors="coerce").dt.month.astype("Int64")

    canonical_lookup, canonical_providers = _build_provider_alias_lookup(v2_universe=v2_rows)
    explicit_known_day, explicit_fixed_day, alias_audit_rows = _load_explicit_frames(
        transport_input_path=transport_input_path,
        canonical_lookup=canonical_lookup,
        canonical_providers=canonical_providers,
    )

    detail_rows = _extract_detail_rows(
        matrix_input_path=matrix_input_path,
        layout_rules_path=layout_rules_path,
        productos_mapping_path=productos_mapping_path,
        proveedores_mapping_path=proveedores_mapping_path,
        terminales_mapping_path=terminales_mapping_path,
        canonical_lookup=canonical_lookup,
        canonical_providers=canonical_providers,
        run_id=execution_run_id,
        ts_utc=generated_ts_utc,
    )
    detail_day = _aggregate_transport_rows(frame=detail_rows, date_column="fecha_oferta")
    detail_day["transport_source_kind"] = DETERMINISTIC_KIND
    detail_day["transport_bucket_origin"] = DETERMINISTIC_BUCKET

    explicit_keys = set(map(tuple, explicit_known_day[["fecha_evento", "producto_canonico", "proveedor_candidato"]].itertuples(index=False, name=None)))
    parser_fix_day = explicit_fixed_day[
        ~explicit_fixed_day[["fecha_evento", "producto_canonico", "proveedor_candidato"]]
        .apply(tuple, axis=1)
        .isin(explicit_keys)
    ].reset_index(drop=True)
    parser_fix_keys = set(map(tuple, parser_fix_day[["fecha_evento", "producto_canonico", "proveedor_candidato"]].itertuples(index=False, name=None)))

    deterministic_day = detail_day[
        ~detail_day[["fecha_evento", "producto_canonico", "proveedor_candidato"]]
        .apply(tuple, axis=1)
        .isin(explicit_keys | parser_fix_keys)
    ].reset_index(drop=True)
    deterministic_keys = set(
        map(tuple, deterministic_day[["fecha_evento", "producto_canonico", "proveedor_candidato"]].itertuples(index=False, name=None))
    )

    combined_exact_pool = pd.concat(
        [explicit_known_day, parser_fix_day, deterministic_day],
        ignore_index=True,
    )
    exact_date_pool = set(combined_exact_pool["fecha_evento"].astype(str).tolist())
    exact_date_product_pool = set(
        map(tuple, combined_exact_pool[["fecha_evento", "producto_canonico"]].itertuples(index=False, name=None))
    )

    classified_rows: list[dict[str, Any]] = []
    for record in universe_keys.to_dict(orient="records"):
        key = (record["fecha_evento"], record["producto_canonico"], record["proveedor_candidato"])
        if key in explicit_keys:
            record["transport_bucket_origin"] = RAW_EXPLICIT_BUCKET
        elif key in parser_fix_keys:
            record["transport_bucket_origin"] = PARSER_FIX_BUCKET
        elif key in deterministic_keys:
            record["transport_bucket_origin"] = DETERMINISTIC_BUCKET
        else:
            record["transport_bucket_origin"] = ""
        classified_rows.append(record)
    classified = pd.DataFrame(classified_rows)

    missing_after_exact = classified[classified["transport_bucket_origin"] == ""].copy()
    heuristic_selected = _pick_heuristic_rows(
        universe_keys=missing_after_exact[["fecha_evento", "producto_canonico", "proveedor_candidato"]],
        pool_day_frame=combined_exact_pool[
            [
                "fecha_evento",
                "producto_canonico",
                "proveedor_candidato",
                "transport_cost_value",
                "transport_cost_mean_day_provider",
                "transport_cost_range_day_provider",
                "transport_observations",
                "transport_unique_terminal_count",
                "transport_multi_terminal_share",
                "source_files_count",
                "transport_source_kind",
                "transport_bucket_origin",
            ]
        ],
    )
    heuristic_keys = set(
        map(tuple, heuristic_selected[["fecha_evento", "producto_canonico", "proveedor_candidato"]].itertuples(index=False, name=None))
    )
    if heuristic_keys:
        classified.loc[
            classified[["fecha_evento", "producto_canonico", "proveedor_candidato"]].apply(tuple, axis=1).isin(heuristic_keys),
            "transport_bucket_origin",
        ] = HEURISTIC_BUCKET

    classified["no_raw_subreason"] = ""
    remaining_missing_mask = classified["transport_bucket_origin"].eq("")
    classified.loc[remaining_missing_mask, "transport_bucket_origin"] = NO_RAW_BUCKET
    for idx, row in classified.loc[classified["transport_bucket_origin"] == NO_RAW_BUCKET].iterrows():
        date_key = str(row["fecha_evento"])
        date_product_key = (str(row["fecha_evento"]), str(row["producto_canonico"]))
        if date_key not in exact_date_pool:
            subreason = "no_date_raw"
        elif date_product_key not in exact_date_product_pool:
            subreason = "same_date_other_product_only"
        else:
            subreason = "same_date_product_other_provider_only"
        classified.at[idx, "no_raw_subreason"] = subreason

    final_sources = pd.concat(
        [
            explicit_known_day,
            parser_fix_day,
            deterministic_day,
            heuristic_selected[
                [
                    "fecha_evento",
                    "producto_canonico",
                    "proveedor_candidato",
                    "transport_cost_value",
                    "transport_cost_mean_day_provider",
                    "transport_cost_range_day_provider",
                    "transport_observations",
                    "transport_unique_terminal_count",
                    "transport_multi_terminal_share",
                    "source_files_count",
                    "transport_source_kind",
                    "transport_bucket_origin",
                    "transport_days_gap",
                    "transport_lookahead_flag",
                    "source_date_for_transport",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    final_sources = final_sources.drop_duplicates(
        subset=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        keep="first",
    ).copy()

    staging = classified.merge(
        final_sources,
        on=["fecha_evento", "producto_canonico", "proveedor_candidato", "transport_bucket_origin"],
        how="left",
    )
    staging["transport_source_kind"] = staging["transport_source_kind"].fillna(MISSING_KIND)
    staging["transport_imputed_flag"] = staging["transport_source_kind"].isin([DETERMINISTIC_KIND, HEURISTIC_KIND]).astype(int)
    staging["transport_days_gap"] = pd.to_numeric(staging["transport_days_gap"], errors="coerce")
    staging["transport_lookahead_flag"] = pd.to_numeric(staging["transport_lookahead_flag"], errors="coerce").fillna(0).astype(int)
    staging["source_date_for_transport"] = staging["source_date_for_transport"].fillna(staging["fecha_evento"])
    staging["fecha_oferta"] = staging["fecha_evento"]
    staging["transport_bucket_origin"] = staging["transport_bucket_origin"].fillna(NO_RAW_BUCKET)
    staging["rebuild_run_id"] = execution_run_id
    staging["rebuild_ts_utc"] = generated_ts_utc

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    missingness_json_path.parent.mkdir(parents=True, exist_ok=True)
    missingness_csv_path.parent.mkdir(parents=True, exist_ok=True)

    staging_output_columns = [
        "fecha_oferta",
        "producto_canonico",
        "proveedor_candidato",
        "transport_cost_value",
        "transport_cost_mean_day_provider",
        "transport_cost_range_day_provider",
        "transport_observations",
        "transport_unique_terminal_count",
        "transport_multi_terminal_share",
        "source_files_count",
        "transport_source_kind",
        "transport_imputed_flag",
        "transport_days_gap",
        "transport_lookahead_flag",
        "source_date_for_transport",
        "transport_bucket_origin",
        "no_raw_subreason",
        "rebuild_run_id",
        "rebuild_ts_utc",
    ]
    staging[staging_output_columns].to_csv(output_csv_path, index=False, encoding="utf-8")

    classified_csv = classified.merge(
        v2_rows.groupby(["fecha_evento", "producto_canonico", "proveedor_candidato"], as_index=False).size().rename(columns={"size": "v2_row_count"}),
        on=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        how="left",
    ).merge(
        staging[
            [
                "fecha_evento",
                "producto_canonico",
                "proveedor_candidato",
                "transport_source_kind",
                "transport_imputed_flag",
                "transport_days_gap",
                "transport_lookahead_flag",
                "source_date_for_transport",
                "no_raw_subreason",
            ]
        ],
        on=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        how="left",
    )
    classified_csv.to_csv(missingness_csv_path, index=False, encoding="utf-8")

    rows_with_buckets = v2_rows.merge(
        classified[
            [
                "fecha_evento",
                "producto_canonico",
                "proveedor_candidato",
                "transport_bucket_origin",
                "no_raw_subreason",
            ]
        ],
        on=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        how="left",
    )
    rows_with_buckets["transport_source_kind"] = rows_with_buckets["transport_bucket_origin"].map(
        {
            RAW_EXPLICIT_BUCKET: RAW_EXPLICIT_KIND,
            PARSER_FIX_BUCKET: PARSER_FIX_KIND,
            DETERMINISTIC_BUCKET: DETERMINISTIC_KIND,
            HEURISTIC_BUCKET: HEURISTIC_KIND,
            NO_RAW_BUCKET: MISSING_KIND,
        }
    )
    rows_with_buckets["year"] = pd.to_datetime(rows_with_buckets["fecha_evento"], errors="coerce").dt.year.astype("Int64").astype(str)

    staging_rows = v2_rows.merge(
        staging[
            [
                "fecha_evento",
                "producto_canonico",
                "proveedor_candidato",
                "transport_source_kind",
                "transport_lookahead_flag",
            ]
        ],
        on=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        how="left",
    )
    cutoff_dt = pd.to_datetime(cutoff_date, errors="raise")
    event_dates = pd.to_datetime(staging_rows["fecha_evento"], errors="coerce")
    train_mask = event_dates <= cutoff_dt
    test_mask = event_dates > cutoff_dt
    stage_masks = {
        "raw_explicit": staging_rows["transport_source_kind"].eq(RAW_EXPLICIT_KIND),
        "parser_fix_or_rebuild": staging_rows["transport_source_kind"].isin(
            [RAW_EXPLICIT_KIND, PARSER_FIX_KIND, DETERMINISTIC_KIND]
        ),
        "final_after_heuristic": staging_rows["transport_source_kind"].ne(MISSING_KIND),
        "final_after_heuristic_no_lookahead": staging_rows["transport_source_kind"].isin(
            [RAW_EXPLICIT_KIND, PARSER_FIX_KIND, DETERMINISTIC_KIND]
        )
        | (
            staging_rows["transport_source_kind"].eq(HEURISTIC_KIND)
            & staging_rows["transport_lookahead_flag"].fillna(0).eq(0)
        ),
    }
    stage_coverage = {
        stage_name: {
            "coverage_train": float(mask.loc[train_mask].mean()) if int(train_mask.sum()) > 0 else 0.0,
            "coverage_test": float(mask.loc[test_mask].mean()) if int(test_mask.sum()) > 0 else 0.0,
        }
        for stage_name, mask in stage_masks.items()
    }

    parser_fix_rows = rows_with_buckets["transport_bucket_origin"].eq(PARSER_FIX_BUCKET).sum()
    deterministic_rows = rows_with_buckets["transport_bucket_origin"].eq(DETERMINISTIC_BUCKET).sum()
    heuristic_rows = rows_with_buckets["transport_bucket_origin"].eq(HEURISTIC_BUCKET).sum()
    report = {
        "status": "ok",
        "run_id": execution_run_id,
        "generated_ts_utc": generated_ts_utc,
        "cutoff_date": cutoff_date,
        "inputs": {
            "transport_input": str(transport_input_path),
            "matrix_input": str(matrix_input_path),
            "v2_input": str(v2_input_path),
        },
        "outputs": {
            "staging_output": str(output_csv_path),
            "missingness_json": str(missingness_json_path),
            "missingness_csv": str(missingness_csv_path),
        },
        "bucket_counts_unique_keys": classified["transport_bucket_origin"].value_counts(dropna=False).to_dict(),
        "bucket_counts_v2_rows": rows_with_buckets["transport_bucket_origin"].value_counts(dropna=False).to_dict(),
        "bucket_breakdown_by_year_v2_rows": _row_bucket_breakdown(
            frame=rows_with_buckets,
            bucket_column="transport_bucket_origin",
            group_column="year",
        ),
        "bucket_breakdown_by_product_v2_rows": _row_bucket_breakdown(
            frame=rows_with_buckets,
            bucket_column="transport_bucket_origin",
            group_column="producto_canonico",
        ),
        "top_provider_bucket_rows_v2": _top_group_bucket_rows(
            frame=rows_with_buckets,
            group_column="proveedor_candidato",
            bucket_column="transport_bucket_origin",
            top_n=50,
        ),
        "no_raw_subreason_counts_v2_rows": rows_with_buckets.loc[
            rows_with_buckets["transport_bucket_origin"].eq(NO_RAW_BUCKET),
            "no_raw_subreason",
        ].value_counts(dropna=False).to_dict(),
        "stage_coverage": stage_coverage,
        "lookahead_stats": {
            "heuristic_unique_keys": int(staging["transport_source_kind"].eq(HEURISTIC_KIND).sum()),
            "heuristic_unique_keys_with_lookahead": int(
                staging.loc[staging["transport_source_kind"].eq(HEURISTIC_KIND), "transport_lookahead_flag"].sum()
            ),
            "heuristic_v2_rows_with_lookahead": int(
                staging_rows.loc[staging_rows["transport_source_kind"].eq(HEURISTIC_KIND), "transport_lookahead_flag"].sum()
            ),
        },
        "recovery_summary": {
            "parser_fix_rows_v2": int(parser_fix_rows),
            "deterministic_rebuild_rows_v2": int(deterministic_rows),
            "heuristic_rows_v2": int(heuristic_rows),
            "explicit_train_coverage": float(stage_coverage["raw_explicit"]["coverage_train"]),
            "final_train_coverage": float(stage_coverage["final_after_heuristic"]["coverage_train"]),
            "final_train_coverage_no_lookahead": float(stage_coverage["final_after_heuristic_no_lookahead"]["coverage_train"]),
        },
        "provider_alias_resolution": {
            "resolved_aliases": [row for row in alias_audit_rows if row["resolved_provider"] != "UNKNOWN"],
            "resolution_method_counts": pd.DataFrame(alias_audit_rows)["alias_method"].value_counts().to_dict()
            if alias_audit_rows
            else {},
        },
        "detail_rebuild_summary": {
            "detail_rows_output": int(len(detail_rows)),
            "detail_unique_day_keys": int(len(detail_day)),
            "detail_keys_used_for_v2_rebuild": int(len(deterministic_day)),
        },
    }
    missingness_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# SECTION: CLI entrypoint
def main() -> None:
    """Run the CLI entrypoint for the Day 04.2 transport rebuild pipeline."""
    args = parse_args()
    summary = run(
        transport_input_path=args.transport_input,
        matrix_input_path=args.matrix_input,
        v2_input_path=args.v2_input,
        layout_rules_path=args.layout_rules,
        productos_mapping_path=args.productos_mapping,
        proveedores_mapping_path=args.proveedores_mapping,
        terminales_mapping_path=args.terminales_mapping,
        output_csv_path=args.output_csv,
        missingness_json_path=args.missingness_json,
        missingness_csv_path=args.missingness_csv,
        cutoff_date=args.cutoff_date,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
