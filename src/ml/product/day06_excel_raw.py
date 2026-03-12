from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import pandas as pd

from src.etl.extract import extract_ofertas_calculos_raw as extract_calculos
from src.etl.extract import extract_ofertas_raw as extract_tabla
from src.etl.integrate import reconcile_ofertas_sources as reconcile_sources
from src.etl.transform import build_ofertas_transport_signals as transport_signals
from src.etl.marts import build_dataset_modelo_day041_ablation as day041_ablation
from src.ml.rules.blocklist import apply_blocklist_candidates, load_blocklist_rules
from src.ml.shared.day05_tabular import V41_TRANSPORT_COLUMNS
from src.ml.shared.project_paths import (
    OFERTAS_CALCULOS_LAYOUT_RULES_PATH,
    OFERTAS_LAYOUT_RULES_PATH,
    PRODUCTOS_MAPPING_PATH,
    PROVEEDORES_MAPPING_PATH,
    RULES_CSV_PATH,
    TERMINALES_MAPPING_PATH,
)


@dataclass(frozen=True)
class ExcelRawParseBundle:
    """Hold the parsed workbook summary plus the seed candidate universe for Day 06."""

    parse_run_id: str
    source_name: str
    source_suffix: str
    sheet_names: list[str]
    selected_table_sheet: str
    selected_calculos_sheet: str
    event_seed_df: pd.DataFrame
    candidate_seed_df: pd.DataFrame
    parse_summary: dict[str, Any]


ProgressCallback = Callable[[str, str], None]


# SECTION: Shared helpers
def _emit_progress(
    progress_callback: ProgressCallback | None,
    step_key: str,
    user_message: str,
) -> None:
    """Notify one optional UI callback about the current parse step."""
    if progress_callback is None:
        return
    progress_callback(step_key, user_message)


# SECTION: Shared helpers
def _coerce_string_series(series: pd.Series) -> pd.Series:
    """Cast one series to nullable string while preserving blanks as empty text."""
    return series.astype("string").fillna("").str.strip()


# SECTION: Shared helpers
def _coerce_float_series(series: pd.Series) -> pd.Series:
    """Cast one series to nullable float for product-facing dataframes."""
    return pd.to_numeric(series, errors="coerce").astype("Float64")


# SECTION: Shared helpers
def _coerce_int_series(series: pd.Series) -> pd.Series:
    """Cast one series to nullable integer for product-facing dataframes."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


# SECTION: Shared helpers
def _normalize_frame_dtypes(
    dataframe: pd.DataFrame,
    *,
    string_columns: list[str] | tuple[str, ...],
    float_columns: list[str] | tuple[str, ...] = (),
    int_columns: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """Apply one explicit dtype map to the subset of columns present in the dataframe."""
    working = dataframe.copy()
    for column in string_columns:
        if column in working.columns:
            working[column] = _coerce_string_series(working[column])
    for column in float_columns:
        if column in working.columns:
            working[column] = _coerce_float_series(working[column])
    for column in int_columns:
        if column in working.columns:
            working[column] = _coerce_int_series(working[column])
    return working


# SECTION: Shared helpers
def normalize_excel_enrichment_frame(enrichment_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the editable Excel enrichment dataframe to stable nullable dtypes."""
    return _normalize_frame_dtypes(
        enrichment_df,
        string_columns=[
            "event_seed_id",
            "fecha_evento",
            "producto_canonico",
            "terminal_compra",
            "albaran_id",
            "linea_id",
        ],
        float_columns=["litros_evento"],
    )


# SECTION: Shared helpers
def _normalize_candidate_seed_frame(candidate_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the parsed candidate seed before manual enrichment is added."""
    return _normalize_frame_dtypes(
        candidate_seed_df,
        string_columns=[
            "event_seed_id",
            "fecha_evento",
            "producto_canonico",
            "terminal_compra",
            "proveedor_candidato",
        ],
        float_columns=[
            "coste_min_dia_proveedor",
            "coste_min_evento",
            "coste_max_evento",
            "spread_coste_evento",
            "delta_vs_min_evento",
            "ratio_vs_min_evento",
            "v41_transport_cost_min_day_provider",
            "v41_transport_cost_mean_day_provider",
            "v41_transport_cost_range_day_provider",
            "v41_transport_multi_terminal_share",
            "v41_transport_gap_vs_min_event",
            "v41_transport_ratio_vs_min_event",
        ],
        int_columns=[
            "rank_coste_dia_producto",
            "terminales_cubiertos",
            "observaciones_oferta",
            "candidatos_evento_count",
            "v41_transport_observations",
            "v41_transport_unique_terminal_count",
            "v41_transport_rank_event",
        ],
    )


# SECTION: Shared helpers
def _normalize_candidate_grain_frame(candidate_grain_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the final candidate-grain dataframe consumed by UI and scoring."""
    return _normalize_frame_dtypes(
        candidate_grain_df,
        string_columns=[
            "event_id",
            "fecha_evento",
            "albaran_id",
            "linea_id",
            "producto_canonico",
            "terminal_compra",
            "proveedor_elegido_real",
            "proveedor_candidato",
            "block_reason_candidate",
            "v2_run_id",
            "v2_ts_utc",
            "event_seed_id",
            "excel_parser_run_id",
            "excel_source_name",
        ],
        float_columns=[
            "coste_min_dia_proveedor",
            "coste_min_evento",
            "coste_max_evento",
            "spread_coste_evento",
            "delta_vs_min_evento",
            "ratio_vs_min_evento",
            "litros_evento",
            "precio_unitario_evento",
            "importe_total_evento",
            "v41_transport_cost_min_day_provider",
            "v41_transport_cost_mean_day_provider",
            "v41_transport_cost_range_day_provider",
            "v41_transport_multi_terminal_share",
            "v41_transport_gap_vs_min_event",
            "v41_transport_ratio_vs_min_event",
        ],
        int_columns=[
            "rank_coste_dia_producto",
            "terminales_cubiertos",
            "observaciones_oferta",
            "candidatos_evento_count",
            "dia_semana",
            "mes",
            "fin_mes",
            "blocked_by_rule_candidate",
            "target_elegido",
            "v41_transport_observations",
            "v41_transport_unique_terminal_count",
            "v41_transport_rank_event",
        ],
    )


# SECTION: Shared helpers
def _clean_text(value: object) -> str:
    """Normalize one scalar into a stripped string while preserving blanks as empty text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


# SECTION: Shared helpers
def _build_event_seed_id(row: pd.Series) -> str:
    """Build one deterministic base-event identifier before manual enrichment is added."""
    key = "|".join(
        [
            _clean_text(row.get("fecha_evento", "")),
            _clean_text(row.get("producto_canonico", "")),
            _clean_text(row.get("terminal_compra", "")),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# SECTION: Shared helpers
def _build_final_event_id(row: pd.Series) -> str:
    """Build the final Day 06 event identifier once manual albaran/linea enrichment exists."""
    key = "|".join(
        [
            _clean_text(row.get("fecha_evento", "")),
            _clean_text(row.get("albaran_id", "")),
            _clean_text(row.get("linea_id", "")),
            _clean_text(row.get("producto_canonico", "")),
            _clean_text(row.get("terminal_compra", "")),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# SECTION: Shared helpers
def _build_record_hash(row: pd.Series) -> str:
    """Build one deterministic row hash to count unique raw observations in the offer fact."""
    key = "|".join(
        [
            _clean_text(row.get("source_file", "")),
            _clean_text(row.get("sheet_name", "")),
            _clean_text(row.get("row_idx", "")),
            _clean_text(row.get("col_idx", "")),
            _clean_text(row.get("terminal_canonico", "")),
            _clean_text(row.get("producto_canonico", "")),
            _clean_text(row.get("proveedor_canonico", "")),
            _clean_text(row.get("coste_num", "")),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# SECTION: Workbook parsing
def parse_excel_raw_workbook(
    *,
    source_name: str,
    source_suffix: str,
    source_bytes: bytes,
    progress_callback: ProgressCallback | None = None,
) -> ExcelRawParseBundle:
    """Parse one uploaded `Comparativa de precios` workbook into a seed candidate universe."""
    parse_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with TemporaryDirectory(prefix="day06_excel_raw_") as temp_dir:
        temp_root = Path(temp_dir)
        workbook_name = Path(source_name).name or f"comparativa{source_suffix or '.xlsx'}"
        workbook_path = temp_root / workbook_name
        _emit_progress(progress_callback, "reading_workbook", "Leyendo workbook.")
        workbook_path.write_bytes(source_bytes)

        excel_file = pd.ExcelFile(workbook_path)
        sheet_names = [str(sheet) for sheet in excel_file.sheet_names]

        _emit_progress(progress_callback, "extracting_sheets", "Extrayendo hojas `Tabla` y `Cálculos`.")
        table_extract = _extract_table_artifacts(workbook_path=workbook_path, parse_run_id=parse_run_id)
        calculos_extract = _extract_calculos_artifacts(workbook_path=workbook_path, parse_run_id=parse_run_id)
        _emit_progress(progress_callback, "building_candidate_seed", "Construyendo candidate seed.")
        canonical_calculos = _build_canonical_calculos_frame(
            calculos_rows_df=calculos_extract["rows_df"],
            temp_root=temp_root,
        )
        if canonical_calculos.empty:
            raise ValueError(
                "El workbook raw no generó filas canónicas de Cálculos; no se puede construir candidate-grain."
            )

        offer_fact = _build_offer_fact_frame(canonical_calculos)
        event_seed_df = _build_event_seed_frame(canonical_calculos)
        candidate_seed_df = _build_candidate_seed_frame(
            event_seed_df=event_seed_df,
            offer_fact_df=offer_fact,
        )
        candidate_seed_df = _merge_transport_features(
            candidate_seed_df=candidate_seed_df,
            matrix_rows_df=table_extract["matrix_df"],
            parse_run_id=parse_run_id,
            temp_root=temp_root,
        )
        _emit_progress(progress_callback, "preparing_enrichment", "Preparando enrichment manual.")

        parse_summary = {
            "parse_run_id": parse_run_id,
            "source_name": source_name,
            "sheet_names": sheet_names,
            "selected_table_sheet": table_extract["selected_sheet"],
            "selected_calculos_sheet": calculos_extract["selected_sheet"],
            "table_rows_output": int(len(table_extract["table_df"])),
            "table_matrix_rows_output": int(len(table_extract["matrix_df"])),
            "table_reject_rows": int(len(table_extract["rejects_df"])),
            "calculos_rows_output": int(len(calculos_extract["rows_df"])),
            "calculos_reject_rows": int(len(calculos_extract["rejects_df"])),
            "base_events_total": int(len(event_seed_df)),
            "candidate_seed_rows_total": int(len(candidate_seed_df)),
            "transport_rows_with_signal": int(
                pd.to_numeric(
                    candidate_seed_df.get("v41_transport_cost_min_day_provider"),
                    errors="coerce",
                ).notna().sum()
            ),
        }

    return ExcelRawParseBundle(
        parse_run_id=parse_run_id,
        source_name=source_name,
        source_suffix=source_suffix,
        sheet_names=sheet_names,
        selected_table_sheet=str(table_extract["selected_sheet"]),
        selected_calculos_sheet=str(calculos_extract["selected_sheet"]),
        event_seed_df=event_seed_df.copy(),
        candidate_seed_df=candidate_seed_df.copy(),
        parse_summary=parse_summary,
    )


# SECTION: Workbook parsing
def _extract_table_artifacts(*, workbook_path: Path, parse_run_id: str) -> dict[str, Any]:
    """Extract the `Tabla` sheet into raw mincost and matrix artifacts for one workbook."""
    layout_rules = json.loads(OFERTAS_LAYOUT_RULES_PATH.read_text(encoding="utf-8"))
    raw_frame, _, selected_sheet = extract_tabla._read_sheet(
        file_path=workbook_path,
        preferred_sheet_names=layout_rules["preferred_sheet_names"],
    )
    if raw_frame is None or selected_sheet is None:
        raise ValueError("El workbook raw no contiene hoja `Tabla`; Day 06.fix01 requiere `Tabla + Cálculos`.")

    normalized_frame = extract_tabla._normalize_frame(raw_frame)
    mincost_rows, matrix_rows, reject_rows, _ = extract_tabla._build_rows_for_file(
        file_path=workbook_path,
        sheet_name=selected_sheet,
        raw_frame=raw_frame,
        normalized_frame=normalized_frame,
        rules=layout_rules,
        run_id=parse_run_id,
        ingestion_ts_utc=datetime.now(timezone.utc).isoformat(),
    )
    return {
        "selected_sheet": selected_sheet,
        "table_df": pd.DataFrame(mincost_rows),
        "matrix_df": pd.DataFrame(matrix_rows),
        "rejects_df": pd.DataFrame(reject_rows),
    }


# SECTION: Workbook parsing
def _extract_calculos_artifacts(*, workbook_path: Path, parse_run_id: str) -> dict[str, Any]:
    """Extract the `Cálculos` sheet into raw rows for one workbook."""
    layout_rules = json.loads(OFERTAS_CALCULOS_LAYOUT_RULES_PATH.read_text(encoding="utf-8"))
    raw_frame, _, selected_sheet = extract_calculos._read_calculos_sheet(
        file_path=workbook_path,
        preferred_sheet_names=layout_rules.get("preferred_sheet_names", ["Cálculos", "Calculos"]),
    )
    if raw_frame is None or selected_sheet is None:
        raise ValueError("El workbook raw no contiene hoja `Cálculos`; Day 06.fix01 requiere `Tabla + Cálculos`.")

    normalized_frame = raw_frame.fillna("").astype(str).apply(lambda column: column.map(extract_calculos._normalize_text))
    extracted_rows, reject_rows = extract_calculos._extract_rows_from_file(
        file_path=workbook_path,
        sheet_name=selected_sheet,
        raw_frame=raw_frame,
        normalized_frame=normalized_frame,
        rules=layout_rules,
        run_id=parse_run_id,
        ingestion_ts_utc=datetime.now(timezone.utc).isoformat(),
    )
    rows_df = pd.DataFrame(extracted_rows)
    if rows_df.empty:
        raise ValueError("La hoja `Cálculos` no generó filas válidas de oferta; no se puede continuar.")
    return {
        "selected_sheet": selected_sheet,
        "rows_df": rows_df,
        "rejects_df": pd.DataFrame(reject_rows),
    }


# SECTION: Canonical builders
def _build_canonical_calculos_frame(*, calculos_rows_df: pd.DataFrame, temp_root: Path) -> pd.DataFrame:
    """Map the extracted `Cálculos` rows into canonical terminal/product/provider values."""
    calculos_csv_path = temp_root / "calculos_rows.csv"
    calculos_rows_df.to_csv(calculos_csv_path, index=False, encoding="utf-8")

    productos_mapping = reconcile_sources._load_mapping(PRODUCTOS_MAPPING_PATH, "raw_value", "producto_canonico")
    proveedores_mapping = reconcile_sources._load_mapping(PROVEEDORES_MAPPING_PATH, "raw_value", "proveedor_canonico")
    terminales_mapping = reconcile_sources._load_mapping(TERMINALES_MAPPING_PATH, "raw_value", "terminal_canonico")
    canonical_df = reconcile_sources._prepare_calculos_rows(
        calculos_input_path=calculos_csv_path,
        productos_mapping=productos_mapping,
        proveedores_mapping=proveedores_mapping,
        terminales_mapping=terminales_mapping,
    )
    return canonical_df.copy()


# SECTION: Canonical builders
def _build_offer_fact_frame(canonical_calculos_df: pd.DataFrame) -> pd.DataFrame:
    """Build the V2-style offer fact at day-product-provider grain from canonical `Cálculos` rows."""
    working = canonical_calculos_df.copy()
    working["record_hash"] = working.apply(_build_record_hash, axis=1)
    fact = (
        working.groupby(["fecha_oferta", "producto_canonico", "proveedor_canonico"], as_index=False)
        .agg(
            coste_min_dia_proveedor=("coste_num", "min"),
            terminales_cubiertos=("terminal_canonico", "nunique"),
            observaciones_oferta=("record_hash", "nunique"),
        )
        .sort_values(
            ["fecha_oferta", "producto_canonico", "coste_min_dia_proveedor", "proveedor_canonico"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    fact["rank_coste_dia_producto"] = (
        fact.groupby(["fecha_oferta", "producto_canonico"])["coste_min_dia_proveedor"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    return _normalize_frame_dtypes(
        fact.rename(
            columns={
                "fecha_oferta": "fecha_evento",
                "proveedor_canonico": "proveedor_candidato",
            }
        ),
        string_columns=["fecha_evento", "producto_canonico", "proveedor_candidato"],
        float_columns=["coste_min_dia_proveedor"],
        int_columns=["terminales_cubiertos", "observaciones_oferta", "rank_coste_dia_producto"],
    )


# SECTION: Canonical builders
def _build_event_seed_frame(canonical_calculos_df: pd.DataFrame) -> pd.DataFrame:
    """Build the unique event seed table at one row per `fecha + producto + terminal`."""
    events = (
        canonical_calculos_df[["fecha_oferta", "producto_canonico", "terminal_canonico"]]
        .drop_duplicates()
        .rename(
            columns={
                "fecha_oferta": "fecha_evento",
                "terminal_canonico": "terminal_compra",
            }
        )
        .sort_values(["fecha_evento", "producto_canonico", "terminal_compra"], kind="mergesort")
        .reset_index(drop=True)
    )
    events["event_seed_id"] = events.apply(_build_event_seed_id, axis=1)
    return _normalize_frame_dtypes(
        events[["event_seed_id", "fecha_evento", "producto_canonico", "terminal_compra"]].copy(),
        string_columns=["event_seed_id", "fecha_evento", "producto_canonico", "terminal_compra"],
    )


# SECTION: Canonical builders
def _build_candidate_seed_frame(
    *,
    event_seed_df: pd.DataFrame,
    offer_fact_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join base events with the daily offer fact to build the candidate seed universe."""
    candidates = event_seed_df.merge(
        offer_fact_df,
        on=["fecha_evento", "producto_canonico"],
        how="left",
        validate="many_to_many",
    )
    candidates = candidates[
        candidates["proveedor_candidato"].astype("string").fillna("").str.strip().ne("")
    ].copy()
    if candidates.empty:
        raise ValueError("El workbook raw no generó universo candidato tras unir eventos y ofertas.")

    event_stats = (
        candidates.groupby("event_seed_id", as_index=False)
        .agg(
            candidatos_evento_count=("proveedor_candidato", "nunique"),
            coste_min_evento=("coste_min_dia_proveedor", "min"),
            coste_max_evento=("coste_min_dia_proveedor", "max"),
        )
        .copy()
    )
    event_stats["spread_coste_evento"] = event_stats["coste_max_evento"] - event_stats["coste_min_evento"]
    candidates = candidates.merge(event_stats, on="event_seed_id", how="left", validate="many_to_one")
    candidates["delta_vs_min_evento"] = (
        candidates["coste_min_dia_proveedor"] - candidates["coste_min_evento"]
    )
    candidates["ratio_vs_min_evento"] = (
        candidates["coste_min_dia_proveedor"] / candidates["coste_min_evento"]
    )
    return _normalize_candidate_seed_frame(candidates.reset_index(drop=True))


# SECTION: Transport builders
def _merge_transport_features(
    *,
    candidate_seed_df: pd.DataFrame,
    matrix_rows_df: pd.DataFrame,
    parse_run_id: str,
    temp_root: Path,
) -> pd.DataFrame:
    """Parse and merge the `v41_transport_*` features into the seed candidate universe."""
    working = candidate_seed_df.copy()
    if matrix_rows_df.empty:
        for column in V41_TRANSPORT_COLUMNS:
            working[column] = pd.NA
        return _normalize_candidate_seed_frame(working)

    matrix_csv_path = temp_root / "tabla_matrix_rows.csv"
    transport_csv_path = temp_root / "transport_signals.csv"
    transport_report_path = temp_root / "transport_report.json"
    matrix_rows_df.to_csv(matrix_csv_path, index=False, encoding="utf-8")
    transport_signals.run(
        matrix_input_path=matrix_csv_path,
        layout_rules_path=OFERTAS_LAYOUT_RULES_PATH,
        productos_mapping_path=PRODUCTOS_MAPPING_PATH,
        proveedores_mapping_path=PROVEEDORES_MAPPING_PATH,
        terminales_mapping_path=TERMINALES_MAPPING_PATH,
        output_csv_path=transport_csv_path,
        report_json_path=transport_report_path,
        run_id=parse_run_id,
    )
    if not transport_csv_path.exists():
        for column in V41_TRANSPORT_COLUMNS:
            working[column] = pd.NA
        return _normalize_candidate_seed_frame(working)

    transport_df = pd.read_csv(transport_csv_path, dtype=str, keep_default_na=False)
    if transport_df.empty:
        for column in V41_TRANSPORT_COLUMNS:
            working[column] = pd.NA
        return _normalize_candidate_seed_frame(working)

    transport_day_frame = day041_ablation._build_transport_day_frame(transport_df)
    event_contract_input = working[
        ["event_seed_id", "fecha_evento", "producto_canonico", "proveedor_candidato"]
    ].rename(columns={"event_seed_id": "event_id"})
    transport_event_frame = day041_ablation._build_transport_event_frame(
        dataset_v2=event_contract_input,
        transport_day_frame=transport_day_frame,
    ).rename(columns={"event_id": "event_seed_id"})
    merged = working.merge(
        transport_event_frame,
        on=["event_seed_id", "proveedor_candidato"],
        how="left",
        validate="one_to_one",
    )
    for column in V41_TRANSPORT_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    return _normalize_candidate_seed_frame(merged)


# SECTION: Enrichment builders
def build_excel_enrichment_template(bundle: ExcelRawParseBundle) -> pd.DataFrame:
    """Build the editable event-enrichment table that the UI must complete before scoring."""
    template = bundle.event_seed_df.copy()
    template["albaran_id"] = ""
    template["linea_id"] = ""
    template["litros_evento"] = pd.NA
    return normalize_excel_enrichment_frame(
        template[
            [
                "event_seed_id",
                "fecha_evento",
                "producto_canonico",
                "terminal_compra",
                "albaran_id",
                "linea_id",
                "litros_evento",
            ]
        ].copy()
    )


# SECTION: Enrichment builders
def summarize_excel_enrichment_pending(enrichment_df: pd.DataFrame) -> dict[str, int]:
    """Summarize how many rows still need each mandatory enrichment field."""
    working = normalize_excel_enrichment_frame(enrichment_df)
    return {
        "rows_total": int(len(working)),
        "pending_albaran_id": int(working["albaran_id"].astype("string").fillna("").str.strip().eq("").sum()),
        "pending_linea_id": int(working["linea_id"].astype("string").fillna("").str.strip().eq("").sum()),
        "pending_litros_evento": int(
            (pd.to_numeric(working["litros_evento"], errors="coerce").isna()).sum()
        ),
    }


# SECTION: Enrichment builders
def apply_value_to_blank_enrichment_rows(
    *,
    enrichment_df: pd.DataFrame,
    column_name: str,
    value: object,
) -> tuple[pd.DataFrame, int]:
    """Apply one explicit value only to blank enrichment cells in the requested column."""
    if column_name not in enrichment_df.columns:
        raise ValueError(f"La columna `{column_name}` no existe en el enrichment.")

    working = normalize_excel_enrichment_frame(enrichment_df)
    if column_name == "litros_evento":
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value) or float(numeric_value) <= 0:
            raise ValueError("`litros_evento` masivo debe ser numérico y mayor que 0.")
        blank_mask = pd.to_numeric(working[column_name], errors="coerce").isna()
        affected_rows = int(blank_mask.sum())
        if affected_rows:
            working.loc[blank_mask, column_name] = float(numeric_value)
        return normalize_excel_enrichment_frame(working), affected_rows

    normalized_value = _clean_text(value)
    if normalized_value == "":
        raise ValueError(f"`{column_name}` masivo no puede estar vacío.")
    blank_mask = working[column_name].astype("string").fillna("").str.strip().eq("")
    affected_rows = int(blank_mask.sum())
    if affected_rows:
        working.loc[blank_mask, column_name] = normalized_value
    return normalize_excel_enrichment_frame(working), affected_rows


# SECTION: Enrichment builders
def apply_linea_sequence_to_blank_rows(
    *,
    enrichment_df: pd.DataFrame,
    start_value: int = 1,
) -> tuple[pd.DataFrame, int, str]:
    """Fill blank `linea_id` cells with one visible sequential range."""
    if "linea_id" not in enrichment_df.columns:
        raise ValueError("La columna `linea_id` no existe en el enrichment.")
    if start_value < 1:
        raise ValueError("La secuencia de `linea_id` debe empezar en 1 o más.")

    working = normalize_excel_enrichment_frame(enrichment_df)
    blank_mask = working["linea_id"].astype("string").fillna("").str.strip().eq("")
    blank_positions = list(working.index[blank_mask])
    affected_rows = len(blank_positions)
    if affected_rows == 0:
        return normalize_excel_enrichment_frame(working), 0, ""

    sequence_values = [str(value) for value in range(start_value, start_value + affected_rows)]
    for row_index, linea_value in zip(blank_positions, sequence_values, strict=False):
        working.at[row_index, "linea_id"] = linea_value
    return normalize_excel_enrichment_frame(working), affected_rows, f"{sequence_values[0]}..{sequence_values[-1]}"


# SECTION: Enrichment builders
def validate_excel_enrichment(enrichment_df: pd.DataFrame) -> dict[str, Any]:
    """Validate the mandatory manual enrichment required before scoring an uploaded workbook."""
    required_columns = ["event_seed_id", "albaran_id", "linea_id", "litros_evento"]
    missing_columns = [column for column in required_columns if column not in enrichment_df.columns]
    if missing_columns:
        return {
            "status": "FAIL",
            "message": f"Faltan columnas de enrichment obligatorias: {missing_columns}",
        }

    working = normalize_excel_enrichment_frame(enrichment_df)
    blank_albaran = int(working["albaran_id"].astype("string").fillna("").str.strip().eq("").sum())
    blank_linea = int(working["linea_id"].astype("string").fillna("").str.strip().eq("").sum())
    litros_numeric = pd.to_numeric(working["litros_evento"], errors="coerce")
    invalid_litros = int((litros_numeric.isna() | (litros_numeric <= 0)).sum())
    duplicate_event_seed = int(working["event_seed_id"].astype(str).duplicated().sum())
    if blank_albaran or blank_linea or invalid_litros or duplicate_event_seed:
        issues: list[str] = []
        if blank_albaran:
            issues.append(f"albaran_id vacío en {blank_albaran} evento(s)")
        if blank_linea:
            issues.append(f"linea_id vacío en {blank_linea} evento(s)")
        if invalid_litros:
            issues.append(f"litros_evento inválido en {invalid_litros} evento(s)")
        if duplicate_event_seed:
            issues.append(f"event_seed_id duplicado en {duplicate_event_seed} fila(s)")
        return {
            "status": "FAIL",
            "message": " | ".join(issues),
        }
    return {
        "status": "PASS",
        "message": "El enrichment manual está completo y listo para construir candidate-grain.",
    }


# SECTION: Enrichment builders
def build_excel_candidate_grain(
    *,
    bundle: ExcelRawParseBundle,
    enrichment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the parsed workbook with manual enrichment and return the final candidate-grain input."""
    enrichment_df = normalize_excel_enrichment_frame(enrichment_df)
    enrichment_status = validate_excel_enrichment(enrichment_df)
    if enrichment_status["status"] != "PASS":
        raise ValueError(str(enrichment_status["message"]))

    enrichment_base = bundle.event_seed_df.merge(
        enrichment_df[["event_seed_id", "albaran_id", "linea_id", "litros_evento"]],
        on="event_seed_id",
        how="left",
        validate="one_to_one",
    )
    enrichment_base["event_id"] = enrichment_base.apply(_build_final_event_id, axis=1)
    if enrichment_base["event_id"].duplicated().any():
        raise ValueError("El enrichment manual genera `event_id` duplicados; revisa `albaran_id` y `linea_id`.")

    candidates = bundle.candidate_seed_df.merge(
        enrichment_base[["event_seed_id", "event_id", "albaran_id", "linea_id", "litros_evento"]],
        on="event_seed_id",
        how="left",
        validate="many_to_one",
    )
    fechas = pd.to_datetime(candidates["fecha_evento"], errors="coerce")
    candidates["dia_semana"] = fechas.dt.dayofweek
    candidates["mes"] = fechas.dt.month
    candidates["fin_mes"] = fechas.dt.is_month_end.astype(int)
    candidates["proveedor_elegido_real"] = ""
    candidates["precio_unitario_evento"] = pd.NA
    candidates["importe_total_evento"] = pd.NA
    candidates["target_elegido"] = pd.NA
    candidates["v2_run_id"] = ""
    candidates["v2_ts_utc"] = ""

    rules = load_blocklist_rules(RULES_CSV_PATH)
    candidates = apply_blocklist_candidates(candidates, rules)
    for column in V41_TRANSPORT_COLUMNS:
        if column not in candidates.columns:
            candidates[column] = pd.NA

    output_columns = [
        "event_id",
        "fecha_evento",
        "albaran_id",
        "linea_id",
        "producto_canonico",
        "terminal_compra",
        "proveedor_elegido_real",
        "proveedor_candidato",
        "coste_min_dia_proveedor",
        "rank_coste_dia_producto",
        "terminales_cubiertos",
        "observaciones_oferta",
        "candidatos_evento_count",
        "coste_min_evento",
        "coste_max_evento",
        "spread_coste_evento",
        "delta_vs_min_evento",
        "ratio_vs_min_evento",
        "litros_evento",
        "precio_unitario_evento",
        "importe_total_evento",
        "dia_semana",
        "mes",
        "fin_mes",
        "blocked_by_rule_candidate",
        "block_reason_candidate",
        "target_elegido",
        "v2_run_id",
        "v2_ts_utc",
        *V41_TRANSPORT_COLUMNS,
        "event_seed_id",
        "excel_parser_run_id",
        "excel_source_name",
    ]
    candidates["excel_parser_run_id"] = bundle.parse_run_id
    candidates["excel_source_name"] = bundle.source_name
    return _normalize_candidate_grain_frame(candidates[output_columns].copy())
