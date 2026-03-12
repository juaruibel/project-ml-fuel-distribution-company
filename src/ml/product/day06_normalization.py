from __future__ import annotations

import pandas as pd

from src.ml.product import validate_inference_input as vinput
from src.ml.shared.day05_tabular import V41_TRANSPORT_COLUMNS
from src.ml.shared.project_paths import INPUT_CONTRACT_PATH


OPTIONAL_TEXT_COLUMNS = [
    "albaran_id",
    "linea_id",
    "proveedor_elegido_real",
    "block_reason_candidate",
    "v2_run_id",
    "v2_ts_utc",
    "event_seed_id",
    "excel_parser_run_id",
    "excel_source_name",
]


def normalize_operational_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the Day 06 product input into one CSV-ready inference dataframe."""
    contract = vinput.load_contract(INPUT_CONTRACT_PATH)
    normalized = input_df.copy()

    for column in contract.get("columns", {}).get("categorical_non_empty", []) or []:
        if column not in normalized.columns:
            continue
        normalized[column] = (
            normalized[column]
            .astype("string")
            .fillna("")
            .str.strip()
        )

    for column in contract.get("columns", {}).get("date", []) or []:
        if column not in normalized.columns:
            continue
        parsed = pd.to_datetime(normalized[column], errors="coerce")
        normalized[column] = (
            parsed.dt.strftime("%Y-%m-%d")
            .fillna(normalized[column].astype(str).str.strip())
            .astype("string")
        )

    for column in OPTIONAL_TEXT_COLUMNS:
        if column not in normalized.columns:
            continue
        normalized[column] = normalized[column].astype("string").fillna("").str.strip()

    for column in contract.get("columns", {}).get("numeric", []) or []:
        if column not in normalized.columns:
            continue
        normalized[column] = vinput.parse_numeric_series(normalized[column])

    for column in V41_TRANSPORT_COLUMNS:
        if column not in normalized.columns:
            continue
        normalized[column] = vinput.parse_numeric_series(normalized[column])

    ordered_columns = vinput.get_required_columns(contract)
    for column in ordered_columns:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    trailing_columns = [column for column in normalized.columns if column not in ordered_columns]
    return normalized[ordered_columns + trailing_columns].copy()
