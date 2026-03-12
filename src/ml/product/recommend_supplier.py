#!/usr/bin/env python3

# Librerías

from pathlib import Path
from datetime import datetime
import argparse  # -> Nos sirve para utilizar herramientas de línea de comandos. Así no tenemos que hardcodear
import json
import joblib
import numpy as np
import pandas as pd

try:
    from src.ml.shared import functions as fc
    from src.ml.shared.numeric_parsing import parse_numeric_series_locale
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.ml.shared import functions as fc
    from src.ml.shared.numeric_parsing import parse_numeric_series_locale

# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera recomendaciones Top-k de proveedor."
    )
    parser.add_argument("--input-csv", required=True, help="CSV de entrada")
    parser.add_argument("--output-csv", required=True, help="CSV de salida")

    # Estas líneas de ruta definen en CLI la ruta/path de los archivos. Se los pasa desde main()    
    parser.add_argument("--model-path", required=True, help="Ruta model.pkl")
    parser.add_argument("--metadata-path", required=True, help="Ruta metadata.json") 
    parser.add_argument("--top-k", type=int, default=2, help="Top-k por event_id")
    parser.add_argument("--event-col", default="event_id", help="Columna de evento")
    return parser.parse_args()

# CARGAR METADATOS COMO CONTRATO DEL CSV

def load_metadata(metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = metadata.get("feature_columns")

    if not isinstance(feature_columns, list) or len(feature_columns) == 0:
        raise ValueError("metadata.json no contiene `feature_columns` válido.")

    return metadata


def load_model_bundle(model_path: Path, metadata_path: Path):
    """
    Carga modelo y metadata, devolviendo también el contrato de columnas esperado.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe metadata: {metadata_path}")

    metadata = load_metadata(metadata_path)
    model = joblib.load(model_path)
    expected_feature_columns = metadata["feature_columns"]
    return model, metadata, expected_feature_columns

# FUNCIÓN PARA PROCESAR Y TRANSFORMAR COLUMNAS NUMÉRICAS

def parse_numeric(series: pd.Series) -> pd.Series:
    return parse_numeric_series_locale(series)


def ensure_event_column(df: pd.DataFrame, event_col: str = "event_id", prefix: str = "EVT_") -> pd.DataFrame:
    """
    Garantiza columna de evento; si no existe o está vacía, la genera.
    """
    working = df.copy()
    if event_col not in working.columns:
        working[event_col] = [f"{prefix}{index:04d}" for index in range(len(working))]
        return working

    raw_series = working[event_col]
    event_series = raw_series.astype(str).str.strip()
    missing_mask = raw_series.isna() | event_series.eq("") | event_series.str.lower().eq("nan")
    if missing_mask.any():
        generated = [f"{prefix}{index:04d}" for index in range(len(working))]
        event_series = event_series.mask(missing_mask, pd.Series(generated, index=working.index))
        working[event_col] = event_series
    return working

# MATRIX

def build_feature_matrix(df: pd.DataFrame, expected_feature_columns: list[str]) -> pd.DataFrame:
    # Definimos constantes llamando a la función de functions.py:
    RAW_FEATURE_NUM_COLS, RAW_FEATURE_CAT_COLS, _ = fc.get_feature_columns_v2()
    # Comprobación de que están todas las columnas presentes
    expected_set = set(expected_feature_columns)
    incoming_set = set(df.columns)
    # Si todas las columnas esperadas están, entonces se selecciona y crea la matriz
    # y luego se convierte todo a numérico, lo que no pasa a NaN y se convierte a 0.0.
    if expected_set.issubset(incoming_set):
        matrix = df[expected_feature_columns].copy()
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return matrix
    # Si missing_raw es True entonces salta el error: cuando no todas las columnas están
    required_raw = RAW_FEATURE_NUM_COLS + RAW_FEATURE_CAT_COLS
    missing_raw = [column for column in required_raw if column not in df.columns]
    if missing_raw:
        raise ValueError(
            "No se pudo construir matriz de features. "
            f"Faltan columnas raw requeridas: {missing_raw}"
        )
    # Si el input no trae todas las columnas, hay que reconstruir la matriz desde raw:
    working = df.copy()
    for column in RAW_FEATURE_NUM_COLS:
        working[column] = parse_numeric(working[column])
    working[RAW_FEATURE_NUM_COLS] = working[RAW_FEATURE_NUM_COLS].fillna(0.0)

    for column in RAW_FEATURE_CAT_COLS:
        working[column] = (
            working[column]
            .astype("string")
            .fillna("UNKNOWN")
            .str.strip()
            .replace("", "UNKNOWN")
        )

    matrix = pd.get_dummies(
        working[RAW_FEATURE_NUM_COLS + RAW_FEATURE_CAT_COLS],
        drop_first=False,
    )
    matrix = matrix.reindex(columns=expected_feature_columns, fill_value=0.0)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return matrix

# FUNCIÓN PARA SCORE

def get_score_vector(model, matrix: pd.DataFrame) -> np.ndarray:
    """
    Normaliza el score porque depende del tipo de modelo viene con distintas y devuelve np.ndarray
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()
    if hasattr(model, "decision_function"):
        decision = model.decision_function(matrix)
        if np.ndim(decision) == 1:
            return decision
        return decision[:, 1]
    predictions = model.predict(matrix)
    return np.asarray(predictions, dtype=float)

# FUNCIÓN QUE TOMA PREDICCIONES DEL MODELO Y GENERA RANKING

def infer(df: pd.DataFrame, matrix: pd.DataFrame, model, event_col: str, top_k: int) -> pd.DataFrame:
    """
    Genera scores y predicciones del modelo y produce un ranking Top‑k.

    - Añade columnas `score_model` y `pred_label`.
    - Calcula ranking por `event_col` si existe, o global si no.
    - Marca `is_top1` y `is_topk`.
    - Devuelve el dataframe ordenado por evento y ranking.

    Params:
        df: DataFrame original con las filas a rankear.
        matrix: Matriz de features alineada con el modelo.
        model: Modelo entrenado con `predict` y/o `predict_proba`.
        event_col: Nombre de la columna de evento para ranking por grupo.
        top_k: K máximo para marcar `is_topk`.
    """
    result = df.copy()
    result["score_model"] = get_score_vector(model, matrix)
    result["pred_label"] = model.predict(matrix).astype(int)

    if event_col in result.columns:
        result["rank_event_score"] = (
            result.groupby(event_col)["score_model"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    else:
        result["rank_event_score"] = (
            result["score_model"].rank(method="first", ascending=False).astype(int)
        )

    result["is_top1"] = (result["rank_event_score"] == 1).astype(int)
    result["is_topk"] = (result["rank_event_score"] <= top_k).astype(int)

    sort_columns = [event_col, "rank_event_score"] if event_col in result.columns else ["rank_event_score"]
    result = result.sort_values(sort_columns).reset_index(drop=True)
    return result


def run_inference_dataframe(
    input_df: pd.DataFrame,
    model,
    expected_feature_columns: list[str],
    event_col: str = "event_id",
    top_k: int = 2,
) -> pd.DataFrame:
    """
    Ejecuta inferencia end-to-end sobre un dataframe.
    """
    prepared_df = ensure_event_column(input_df, event_col=event_col)
    matrix = build_feature_matrix(prepared_df, expected_feature_columns)
    return infer(
        df=prepared_df,
        matrix=matrix,
        model=model,
        event_col=event_col,
        top_k=top_k,
    )


def save_inference_output(
    result_df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "reco",
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
) -> Path:
    """
    Guarda CSV de inferencia en disco y devuelve la ruta del archivo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime(timestamp_fmt)
    output_path = output_dir / f"{prefix}_{timestamp}.csv"
    result_df.to_csv(output_path, index=False)
    return output_path


# FUNCIÓN MAIN

def main() -> None:
    """
    Orquestación del script
    """
    args = parse_args()
    input_path = Path(args.input_csv).resolve()
    output_path = Path(args.output_csv).resolve()
    model_path = Path(args.model_path).resolve()
    metadata_path = Path(args.metadata_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"No existe input CSV: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe metadata: {metadata_path}")
    if args.top_k < 1:
        raise ValueError("--top-k debe ser >= 1.")

    source_df = pd.read_csv(input_path)
    model, metadata, expected_feature_columns = load_model_bundle(
        model_path=model_path,
        metadata_path=metadata_path,
    )
    output_df = run_inference_dataframe(
        input_df=source_df,
        model=model,
        expected_feature_columns=expected_feature_columns,
        event_col=args.event_col,
        top_k=args.top_k,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Input rows: {len(source_df)}")
    print(f"Output rows: {len(output_df)}")
    print(f"Model: {metadata.get('model_name')}")
    print(f"Output saved: {output_path}")

# MAIN GUARD

if __name__ == "__main__":
    main()
