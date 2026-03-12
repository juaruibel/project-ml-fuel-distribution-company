from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.shared import functions as fc

V41_TRANSPORT_COLUMNS = [
    "v41_transport_cost_min_day_provider",
    "v41_transport_cost_mean_day_provider",
    "v41_transport_cost_range_day_provider",
    "v41_transport_observations",
    "v41_transport_unique_terminal_count",
    "v41_transport_multi_terminal_share",
    "v41_transport_rank_event",
    "v41_transport_gap_vs_min_event",
    "v41_transport_ratio_vs_min_event",
]

V3_DISPERSION_COLUMNS = [
    "v3_cost_mean_terminal",
    "v3_cost_std_terminal",
    "v3_cost_range_terminal",
    "v3_cost_cv_terminal",
    "v3_share_terminales_min_cost",
]

V3_COMPETITION_COLUMNS = [
    "v3_coste_segundo_evento",
    "v3_gap_min_vs_second_evento",
    "v3_delta_vs_second_evento",
    "v3_ratio_vs_second_evento",
    "v3_rank_pct_evento",
    "v3_coste_mean_evento",
    "v3_delta_vs_mean_evento",
    "v3_candidatos_min_coste_count",
    "v3_is_unique_min_coste_evento",
]

V43_TRANSPORT_COLUMNS = [
    "v43_transport_cost_min_day_provider",
    "v43_transport_cost_mean_day_provider",
    "v43_transport_cost_range_day_provider",
    "v43_transport_observations",
    "v43_transport_unique_terminal_count",
    "v43_transport_multi_terminal_share",
    "v43_transport_rank_event",
    "v43_transport_gap_vs_min_event",
    "v43_transport_ratio_vs_min_event",
    "v43_transport_source_raw_explicit",
    "v43_transport_source_parser_fix",
    "v43_transport_source_deterministic_rebuild",
    "v43_transport_source_carry_forward_30d",
    "v43_transport_source_missing",
    "v43_transport_imputed_flag",
    "v43_transport_days_since_last_transport",
]

V43_TRANSPORT_MISSING_FLAG_COLUMNS = [
    "v43_transport_cost_min_day_provider_missing_flag",
    "v43_transport_cost_mean_day_provider_missing_flag",
    "v43_transport_cost_range_day_provider_missing_flag",
    "v43_transport_observations_missing_flag",
    "v43_transport_unique_terminal_count_missing_flag",
    "v43_transport_multi_terminal_share_missing_flag",
    "v43_transport_rank_event_missing_flag",
    "v43_transport_gap_vs_min_event_missing_flag",
    "v43_transport_ratio_vs_min_event_missing_flag",
    "v43_transport_days_since_last_transport_missing_flag",
]


# SECTION: Dataset catalog
def get_day05_dataset_catalog(project_root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the fixed Day 05 dataset catalog with feature contracts and LR references."""
    feature_cols_num, feature_cols_cat, target_col = fc.get_feature_columns_v2()
    root = project_root.resolve() if project_root is not None else None

    def _resolve(relative_path: str) -> Path:
        relative = Path(relative_path)
        return (root / relative).resolve() if root is not None else relative

    return {
        "V2": {
            "dataset_alias": "V2",
            "dataset_path": _resolve("data/public/dataset_modelo_proveedor_v2_candidates.csv"),
            "feature_cols_num": list(feature_cols_num),
            "feature_cols_cat": list(feature_cols_cat),
            "target_col": target_col,
            "lr_equivalent_variant": "LR_smote_0.5",
        },
        "V2_TRANSPORT_ONLY": {
            "dataset_alias": "V2_TRANSPORT_ONLY",
            "dataset_path": _resolve("data/public/day041/dataset_modelo_v2_transport_only.csv"),
            "feature_cols_num": list(feature_cols_num) + V41_TRANSPORT_COLUMNS,
            "feature_cols_cat": list(feature_cols_cat),
            "target_col": target_col,
            "lr_equivalent_variant": "V2_TRANSPORT_ONLY_LR_smote_0.5_v1",
        },
        "V2_DISPERSION": {
            "dataset_alias": "V2_DISPERSION",
            "dataset_path": _resolve("data/public/day041/dataset_modelo_v2_dispersion.csv"),
            "feature_cols_num": list(feature_cols_num) + V3_DISPERSION_COLUMNS,
            "feature_cols_cat": list(feature_cols_cat),
            "target_col": target_col,
            "lr_equivalent_variant": "V2_DISPERSION_LR_smote_0.5_v1",
        },
        "V2_COMPETITION": {
            "dataset_alias": "V2_COMPETITION",
            "dataset_path": _resolve("data/public/day041/dataset_modelo_v2_competition.csv"),
            "feature_cols_num": list(feature_cols_num) + V3_COMPETITION_COLUMNS,
            "feature_cols_cat": list(feature_cols_cat),
            "target_col": target_col,
            "lr_equivalent_variant": "V2_COMPETITION_LR_smote_0.5_v1",
        },
        "V2_TRANSPORT_CARRY30D_ONLY": {
            "dataset_alias": "V2_TRANSPORT_CARRY30D_ONLY",
            "dataset_path": _resolve("data/public/day043/dataset_modelo_v2_transport_carry30d_only.csv"),
            "feature_cols_num": list(feature_cols_num) + V43_TRANSPORT_COLUMNS + V43_TRANSPORT_MISSING_FLAG_COLUMNS,
            "feature_cols_cat": list(feature_cols_cat),
            "target_col": target_col,
            "lr_equivalent_variant": "V2_TRANSPORT_CARRY30D_ONLY_LR_smote_0.5_v1",
        },
    }


# SECTION: Dataset preparation
def prepare_day05_model_frame(
    dataset_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Prepare one Day 05 dataset while preserving event-level audit columns."""
    required = {"event_id", "fecha_evento", target_col, "proveedor_candidato", "producto_canonico", "terminal_compra"}
    missing = required - set(dataset_df.columns)
    if missing:
        raise ValueError(f"Dataset Day 05 incompatible. Faltan columnas requeridas: {sorted(missing)}")

    working = dataset_df.copy()
    for column in feature_cols_num:
        if column not in working.columns:
            raise ValueError(f"Falta feature numérica esperada en Day 05: {column}")
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
    for column in feature_cols_cat:
        if column not in working.columns:
            raise ValueError(f"Falta feature categórica esperada en Day 05: {column}")
        working[column] = working[column].astype("string").fillna("UNKNOWN").str.strip().replace("", "UNKNOWN")
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce").fillna(0).astype(int)
    working["fecha_evento"] = pd.to_datetime(working["fecha_evento"], errors="coerce")
    working = working.dropna(subset=["fecha_evento"]).reset_index(drop=True)
    return working


# SECTION: Dataset preparation
def split_day05_by_cutoff(dataset_df: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one Day 05 dataset with the fixed official cutoff date."""
    cutoff_dt = pd.to_datetime(cutoff_date, errors="raise")
    train_df = dataset_df[dataset_df["fecha_evento"] <= cutoff_dt].copy()
    test_df = dataset_df[dataset_df["fecha_evento"] > cutoff_dt].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Split temporal inválido para Day 05. "
            f"train_rows={len(train_df)} test_rows={len(test_df)} cutoff={cutoff_date}"
        )
    return train_df, test_df


# SECTION: Matrix building
def build_one_hot_train_test_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build aligned one-hot train/test matrices for LightGBM and XGBoost."""
    X_train, X_test, y_train, y_test = fc.dummificar_train_test(
        train=train_df,
        test=test_df,
        feature_cols_num=feature_cols_num,
        feature_cols_cat=feature_cols_cat,
        target_col=target_col,
    )
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return X_train, X_test, y_train.astype(int), y_test.astype(int)


# SECTION: Matrix building
def build_catboost_train_test_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols_num: list[str],
    feature_cols_cat: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build native categorical matrices for CatBoost without one-hot encoding."""
    columns = feature_cols_num + feature_cols_cat
    X_train = train_df[columns].copy()
    X_test = test_df[columns].copy()
    return X_train, X_test, train_df[target_col].astype(int), test_df[target_col].astype(int)


# SECTION: Temporal CV
def build_temporal_event_folds(
    train_df: pd.DataFrame,
    n_splits: int = 3,
    date_col: str = "fecha_evento",
    event_col: str = "event_id",
) -> list[dict[str, Any]]:
    """Build expanding temporal folds using contiguous date blocks and event-level projection."""
    required = {date_col, event_col}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"No se pueden construir folds temporales. Faltan columnas: {sorted(missing)}")
    if n_splits < 1:
        raise ValueError("n_splits debe ser >= 1.")

    working = train_df[[date_col, event_col]].copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working[event_col] = working[event_col].astype(str).str.strip()
    working = working.dropna(subset=[date_col])

    unique_dates = sorted(pd.Index(working[date_col].dt.normalize().unique()).tolist())
    if len(unique_dates) < n_splits + 1:
        raise ValueError(
            "No hay suficientes fechas únicas para construir folds expansivos. "
            f"unique_dates={len(unique_dates)} n_splits={n_splits}"
        )

    blocks = [list(block) for block in np.array_split(unique_dates, n_splits + 1)]
    folds: list[dict[str, Any]] = []
    accumulated_train_dates: list[pd.Timestamp] = []

    for fold_idx in range(n_splits):
        accumulated_train_dates.extend(blocks[fold_idx])
        valid_dates = blocks[fold_idx + 1]
        train_dates = list(accumulated_train_dates)
        if not train_dates or not valid_dates:
            raise ValueError(f"Fold temporal inválido en Day 05. fold_idx={fold_idx}")

        train_mask = working[date_col].dt.normalize().isin(train_dates)
        valid_mask = working[date_col].dt.normalize().isin(valid_dates)

        train_event_ids = sorted(working.loc[train_mask, event_col].unique().tolist())
        valid_event_ids = sorted(working.loc[valid_mask, event_col].unique().tolist())
        overlap = sorted(set(train_event_ids) & set(valid_event_ids))
        if overlap:
            raise ValueError(
                "Fuga detectada en folds temporales Day 05. "
                f"event_ids solapados entre train y validación: {overlap[:5]}"
            )

        folds.append(
            {
                "fold_idx": fold_idx + 1,
                "train_dates": train_dates,
                "valid_dates": valid_dates,
                "train_event_ids": train_event_ids,
                "valid_event_ids": valid_event_ids,
            }
        )

    return folds
