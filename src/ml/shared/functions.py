# Importamos librerías

import json
import re
from pathlib import Path
from datetime import datetime, timezone
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from src.ml.shared.numeric_parsing import parse_numeric_series_locale

# Encuentra la raíz del proyecto usando __file__

BASE_DIR = Path(__file__).resolve().parents[3]

# Define la carpeta data_clean/

DATA_CLEAN_DIR = BASE_DIR / "data" / "public"


def load_data(file_merge):
    """
    Carga el .csv limpio del ETL para analizar pero manteniendo el tipo:
    c.p como string dado que pandas lo convierte automáticamente a int
    :param file_merge: nombre del archivo para cargar
    """
    path = DATA_CLEAN_DIR / file_merge
    return pd.read_csv(path)


def parse_day01_metrics_markdown(report_path: Path) -> dict:
    """
    Extrae métricas clave de Day01 desde el reporte markdown.
    """
    if not report_path.exists():
        return {}

    content = report_path.read_text(encoding="utf-8")
    metrics = {}

    dummy_match = re.search(r"Dummy \(most_frequent\) accuracy:\s*([0-9.]+)", content)
    if dummy_match:
        metrics["dummy_accuracy"] = float(dummy_match.group(1))

    for line in content.splitlines():
        if line.strip().startswith("| 5 |"):
            parts = [part.strip() for part in line.split("|") if part.strip()]
            if len(parts) >= 4:
                metrics["knn_k5_accuracy"] = float(parts[1])
                metrics["knn_k5_macro_f1"] = float(parts[2])
                metrics["knn_k5_bal_acc"] = float(parts[3])
    return metrics


def get_top_providers_from_history(
    v2_df: pd.DataFrame,
    target_col: str = "target_elegido",
    provider_col: str = "proveedor_candidato",
) -> tuple[str | None, str | None]:
    """
    Devuelve top-1 y top-2 de proveedores históricos sobre positivos del dataset.
    """
    required_cols = {target_col, provider_col}
    missing_cols = required_cols - set(v2_df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas para top proveedores históricos: {sorted(missing_cols)}")

    positives = v2_df[v2_df[target_col].astype(int) == 1][provider_col].astype(str)
    provider_counts = positives.value_counts()
    if provider_counts.empty:
        return None, None
    top_provider = provider_counts.index[0]
    second_provider = provider_counts.index[1] if len(provider_counts) > 1 else None
    return top_provider, second_provider


def compute_cheapest_topk_hits(
    df: pd.DataFrame,
    event_col: str = "event_id",
    cost_col: str = "coste_min_dia_proveedor",
    target_col: str = "target_elegido",
) -> tuple[float, float]:
    """
    Calcula Top-1 y Top-2 hit del baseline de proveedor más barato.
    """
    required_cols = {event_col, cost_col, target_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas para baseline más barato: {sorted(missing_cols)}")

    eval_df = df[[event_col, cost_col, target_col]].copy()
    eval_df[event_col] = eval_df[event_col].astype(str)
    eval_df[target_col] = pd.to_numeric(eval_df[target_col], errors="coerce").fillna(0).astype(int)
    eval_df[cost_col] = parse_numeric_series_locale(eval_df[cost_col])

    eval_df["score_cheapest"] = -eval_df[cost_col].fillna(10**9)
    eval_df = eval_df.rename(columns={event_col: "event_id", target_col: "target_elegido"})

    top1_hit = topk_hit_by_event(eval_df, "score_cheapest", k=1)
    top2_hit = topk_hit_by_event(eval_df, "score_cheapest", k=2)
    return float(top1_hit), float(top2_hit)


def export_real_day_example_csv(
    v2_path: Path,
    output_path: Path,
    event_id: str,
    include_target: bool = True,
) -> Path:
    """
    Exporta un evento real de V2 a CSV para demo de inferencia.
    """
    if not v2_path.exists():
        raise FileNotFoundError(f"No existe dataset V2: {v2_path}")

    v2_frame = pd.read_csv(v2_path)
    example = v2_frame[v2_frame["event_id"].astype(str) == str(event_id)].copy()
    if example.empty:
        raise ValueError(f"No se encontró event_id en V2: {event_id}")

    if not include_target and "target_elegido" in example.columns:
        example = example.drop(columns=["target_elegido"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    example.to_csv(output_path, index=False, encoding="utf-8")
    return output_path

def df_model_knn(df):
    """
    Toma un dataframe y lo transforma para prepararlo para KNN.
    :param df: dataframe
    """
    # Columnas de feature: variables importantes numéricas para el modelo
    feature_cols = [
        "coste_min_dia_proveedor", 
        "rank_coste_dia_producto", 
        "terminales_cubiertos",
        "dia_semana", 
        "mes", 
        "fin_mes", 
        "blocked_by_rule"
    ]
    # Copia del df
    df_model = df.copy()

    df_model = df_model[df_model["feature_oferta_disponible"] == 1].copy()
    df_model[feature_cols] = df_model[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_model = df_model.dropna(subset=["fecha_compra", "proveedor_elegido"] + feature_cols)
    return df_model

def seleccionar_feature_cols(df):
    """
    Toma dataframe y selecciona solo feature cols para el entrenamiento del modelo.
    """
    feature_cols = [
        "coste_min_dia_proveedor", 
        "rank_coste_dia_producto", 
        "terminales_cubiertos",
        "dia_semana", 
        "mes", 
        "fin_mes", 
        "blocked_by_rule"
    ]
    X = df[feature_cols]
    y = df["proveedor_elegido"]
    return X, y

def split_temporal(df):
    """
    Toma df y utiliza seleccionar_feature_cols para entrenar modelo.
    """
    df_model = df.sort_values("fecha_compra").reset_index(drop=True)
    split_idx = int(len(df_model) * 0.8)

    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:]

    X_train, y_train = seleccionar_feature_cols(train)
    X_test, y_test = seleccionar_feature_cols(test)

    return X_train, X_test, y_train, y_test 

def entrenar_knn(X_train, X_test, y_train, y_test, k=5):
    """
    Entrena kkn y devuelve el score. K tiene como valor default 5.
    """
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k, n_jobs=-1))])
    pipe.fit(X_train, y_train)
    return pipe.score(X_test, y_test)

def split_temporal_feature(df, train_frac=0.8):
    """
    Split temporal por fecha_evento (sin mezclar pasado/futuro).
    """
    df_model = df.copy()
    df_model["fecha_evento"] = pd.to_datetime(df_model["fecha_evento"], errors="coerce")
    df_model = df_model.dropna(subset=["fecha_evento"])

    fechas_ordenadas = sorted(df_model["fecha_evento"].unique())
    cut_idx = max(1, int(len(fechas_ordenadas) * train_frac))
    cutoff_date = fechas_ordenadas[cut_idx - 1]

    train = df_model[df_model["fecha_evento"] <= cutoff_date].copy()
    test = df_model[df_model["fecha_evento"] > cutoff_date].copy()

    return train, test, cutoff_date


def get_feature_columns_v2():
    """
    Devuelve la definición centralizada de columnas para el dataset V2.
    """
    feature_cols_num = [
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
        "dia_semana", 
        "mes", 
        "fin_mes"
    ]
    feature_cols_cat = [
        "proveedor_candidato", 
        "producto_canonico", 
        "terminal_compra"
    ]
    target_col = "target_elegido"
    return feature_cols_num, feature_cols_cat, target_col


def df_model_knn_feature(df):
    """
    Devuelve dataframe base para luego hacer split temporal y dummificación.
    """
    feature_cols_num, feature_cols_cat, target_col = get_feature_columns_v2()

    df_model = df[["event_id", "fecha_evento"] + feature_cols_num + feature_cols_cat + [target_col]].copy()
    df_model = df_model.dropna(subset=feature_cols_num + feature_cols_cat + [target_col])

    return df_model


def split_temporal_v2(df_model, feature_cols_num, feature_cols_cat, target_col="target_elegido", train_frac=0.8):
    df_tmp = df_model.copy()
    df_tmp["fecha_evento"] = pd.to_datetime(df_tmp["fecha_evento"], errors="coerce")
    df_tmp = df_tmp.dropna(subset=["fecha_evento"])

    fechas = sorted(df_tmp["fecha_evento"].unique())
    cut_idx = max(1, int(len(fechas) * train_frac))
    cutoff_date = fechas[cut_idx - 1]

    train_df = df_tmp[df_tmp["fecha_evento"] <= cutoff_date].copy()
    test_df  = df_tmp[df_tmp["fecha_evento"] > cutoff_date].copy()

    X_train = pd.get_dummies(train_df[feature_cols_num + feature_cols_cat], drop_first=False)
    X_test  = pd.get_dummies(test_df[feature_cols_num + feature_cols_cat], drop_first=False)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df[target_col].astype(int)
    y_test  = test_df[target_col].astype(int)

    return X_train, X_test, y_train, y_test, train_df, test_df, cutoff_date

def dummificar_train_test(train, test, feature_cols_num, feature_cols_cat, target_col="target_elegido"):
    """
    Dummifica train y test por separado y alinea columnas para evitar mismatch.
    """
    cols = feature_cols_num + feature_cols_cat

    X_train = pd.get_dummies(train[cols], drop_first=False)
    X_test = pd.get_dummies(test[cols], drop_first=False)

    # Alinear esquema de test al de train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train[target_col].astype(int)
    y_test = test[target_col].astype(int)

    return X_train, X_test, y_train, y_test


def compute_row_metrics(y_true, y_pred):
    """
    Calcula métricas por fila para clasificación binaria.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "f1_pos": float(f1_score(y_true_arr, y_pred_arr, pos_label=1))
    }


def class_balance_summary(y_train, y_test):
    """
    Resume el desbalance de clases para train y test.
    """
    train_pos_rate = float(pd.Series(y_train).mean())
    test_pos_rate = float(pd.Series(y_test).mean())

    return pd.DataFrame({
        "split": ["train", "test"],
        "pos_rate": [train_pos_rate, test_pos_rate],
        "neg_rate": [1 - train_pos_rate, 1 - test_pos_rate],
        "n_rows": [int(len(y_train)), int(len(y_test))]
    })


def plot_class_balance(y_train, y_test, title="Desbalance de clases (train vs test)"):
    """
    Dibuja barras de proporción positiva/negativa para train y test.
    """
    summary_df = class_balance_summary(y_train, y_test)
    plot_df = summary_df.melt(
        id_vars="split",
        value_vars=["pos_rate", "neg_rate"],
        var_name="class",
        value_name="ratio"
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(data=plot_df, x="split", y="ratio", hue="class", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    fig.tight_layout()

    return summary_df, fig, ax


def topk_hit_by_event(df_eval, score_col, k=1):
    """
    Top-k hit rate por event_id.
    """
    required_cols = {"event_id", "target_elegido", score_col}
    missing_cols = required_cols - set(df_eval.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas para top-k: {sorted(missing_cols)}")

    ranked = df_eval.sort_values(
        ["event_id", score_col],
        ascending=[True, False],
        kind="mergesort"
    )
    topk = ranked.groupby("event_id", sort=False).head(k)
    if topk.empty:
        return np.nan

    return float(topk.groupby("event_id")["target_elegido"].max().mean())


def _cheapest_topk_hit_by_event(df_eval, k=1):
    required_cols = {"event_id", "target_elegido", "coste_min_dia_proveedor"}
    missing_cols = required_cols - set(df_eval.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas para baseline más barato: {sorted(missing_cols)}")

    ranked = df_eval.sort_values(
        ["event_id", "coste_min_dia_proveedor"],
        ascending=[True, True],
        kind="mergesort"
    )
    topk = ranked.groupby("event_id", sort=False).head(k)
    if topk.empty:
        return np.nan

    return float(topk.groupby("event_id")["target_elegido"].max().mean())


def compute_business_baselines(train_df, test_df):
    """
    Construye baselines de negocio (más barato e histórico de proveedores).
    """
    required_cols = {"event_id", "target_elegido", "proveedor_candidato", "coste_min_dia_proveedor"}
    missing_train = required_cols - set(train_df.columns)
    missing_test = required_cols - set(test_df.columns)
    if missing_train:
        raise ValueError(f"Faltan columnas en train_df para baselines: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Faltan columnas en test_df para baselines: {sorted(missing_test)}")

    top1_cheapest = _cheapest_topk_hit_by_event(test_df, k=1)
    top2_cheapest = _cheapest_topk_hit_by_event(test_df, k=2)

    chosen_train = train_df.loc[train_df["target_elegido"] == 1, "proveedor_candidato"]
    chosen_test = test_df.loc[test_df["target_elegido"] == 1, "proveedor_candidato"]

    provider_counts = chosen_train.value_counts()
    if provider_counts.empty:
        top_provider_train = None
        second_provider_train = None
        top1_historical = np.nan
        top2_historical = np.nan
        hist_top1_label = "Siempre proveedor_top1_train"
        hist_top2_label = "Top-2 proveedores_train"
    else:
        top_provider_train = provider_counts.index[0]
        second_provider_train = provider_counts.index[1] if len(provider_counts) > 1 else None

        top1_historical = float((chosen_test == top_provider_train).mean()) if not chosen_test.empty else np.nan

        if second_provider_train is None:
            top2_historical = top1_historical
        else:
            top2_historical = float(chosen_test.isin([top_provider_train, second_provider_train]).mean()) if not chosen_test.empty else np.nan

        hist_top1_label = f"Siempre {top_provider_train}"
        hist_top2_label = f"{top_provider_train}+{second_provider_train}" if second_provider_train else f"{top_provider_train}"

    baseline_table = pd.DataFrame([
        {"baseline": "Más barato", "top1_hit": top1_cheapest, "top2_hit": top2_cheapest},
        {"baseline": hist_top1_label, "top1_hit": top1_historical, "top2_hit": top1_historical},
        {"baseline": hist_top2_label, "top1_hit": top1_historical, "top2_hit": top2_historical}
    ])

    return {
        "baseline_table": baseline_table,
        "top_provider_train": top_provider_train,
        "second_provider_train": second_provider_train
    }


def build_eval_frame(test_df, y_true, y_pred, y_score=None):
    """
    Construye dataframe de evaluación para métricas por evento.
    """
    required_cols = {"event_id", "proveedor_candidato", "coste_min_dia_proveedor"}
    missing_cols = required_cols - set(test_df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas en test_df para evaluación: {sorted(missing_cols)}")

    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    if len(test_df) != len(y_true_arr) or len(y_true_arr) != len(y_pred_arr):
        raise ValueError("Longitudes incompatibles entre test_df, y_true y y_pred.")

    eval_df = test_df[["event_id", "proveedor_candidato", "coste_min_dia_proveedor"]].copy()
    eval_df["target_elegido"] = y_true_arr
    eval_df["pred_label"] = y_pred_arr

    if y_score is None:
        eval_df["score_model"] = eval_df["pred_label"].astype(float)
    else:
        y_score_arr = np.asarray(y_score)
        if len(y_score_arr) != len(eval_df):
            raise ValueError("Longitud de y_score incompatible con test_df.")
        eval_df["score_model"] = y_score_arr.astype(float)

    return eval_df


def evaluate_model_vs_baselines(model, X_train, y_train, X_test, y_test, train_df, test_df):
    """
    Entrena un modelo y devuelve métricas por fila + comparación por evento contra baselines.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test)
        if np.ndim(raw_scores) > 1:
            y_score = raw_scores[:, 1]
        else:
            y_score = raw_scores
    else:
        y_score = np.asarray(y_pred).astype(float)

    row_metrics = compute_row_metrics(y_test, y_pred)
    eval_df = build_eval_frame(test_df, y_test, y_pred, y_score)

    top1_model = topk_hit_by_event(eval_df, "score_model", k=1)
    top2_model = topk_hit_by_event(eval_df, "score_model", k=2)

    baselines_info = compute_business_baselines(train_df, test_df)
    baselines_table = baselines_info["baseline_table"]

    model_row = pd.DataFrame([
        {"baseline": "Modelo", "top1_hit": top1_model, "top2_hit": top2_model}
    ])
    comparison_table = pd.concat([model_row, baselines_table], ignore_index=True)

    return {
        "model": model,
        "row_metrics": row_metrics,
        "event_metrics": {"top1_hit": float(top1_model), "top2_hit": float(top2_model)},
        "comparison_table": comparison_table,
        "eval_frame": eval_df,
        "top_provider_train": baselines_info["top_provider_train"],
        "second_provider_train": baselines_info["second_provider_train"]
    }


def evaluate_balance_variant(name, model, X_train, y_train, X_test, y_test, train_df, test_df, sampler=None):
    """
    Evalúa una variante de balanceo sobre train y devuelve resumen + artefactos.

    Si hay sampler, se aplica únicamente a train para evitar fuga al test.
    """
    if sampler is not None:
        X_train_eval, y_train_eval = sampler.fit_resample(X_train, y_train)
    else:
        X_train_eval, y_train_eval = X_train, y_train

    result = evaluate_model_vs_baselines(
        model=model,
        X_train=X_train_eval, y_train=y_train_eval,
        X_test=X_test, y_test=y_test,
        train_df=train_df, test_df=test_df
    )

    row = {
        "variante": name,
        "train_pos_rate_after": float(pd.Series(y_train_eval).mean()),
        "acc": result["row_metrics"]["accuracy"],
        "bal_acc": result["row_metrics"]["balanced_accuracy"],
        "f1_pos": result["row_metrics"]["f1_pos"],
        "top1_hit": result["event_metrics"]["top1_hit"],
        "top2_hit": result["event_metrics"]["top2_hit"],
    }
    return row, result


def run_balance_variants(variants, X_train, y_train, X_test, y_test, train_df, test_df):
    """
    Ejecuta un conjunto de variantes de balanceo y devuelve tabla ordenada.

    `variants` debe ser una lista de tuplas `(name, model, sampler)`.
    """
    rows = []
    artifacts = {}

    for name, model, sampler in variants:
        model_eval = clone(model)
        sampler_eval = clone(sampler) if sampler is not None else None

        row, result = evaluate_balance_variant(
            name=name,
            model=model_eval,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            train_df=train_df, test_df=test_df,
            sampler=sampler_eval
        )
        rows.append(row)
        artifacts[name] = result

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["top2_hit", "bal_acc"], ascending=False).reset_index(drop=True)

    return table, artifacts


def build_default_lr_balance_variants(c=1.0, max_iter=4000, random_state=0, sampling_strategy=0.5):
    """
    Construye el set estándar de variantes de balanceo para Logistic Regression.
    """
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler, SMOTE

    base_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=c, max_iter=max_iter, random_state=random_state))
    ])

    cw_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state
        ))
    ])

    return [
        ("LR_base", base_lr, None),
        ("LR_class_weight_balanced", cw_lr, None),
        ("LR_under_0.5", base_lr, RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)),
        ("LR_over_0.5", base_lr, RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)),
        ("LR_smote_0.5", base_lr, SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)),
    ]


def _safe_json_value(obj):
    """
    Convierte objetos no serializables a tipos seguros para JSON.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(key): _safe_json_value(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe_json_value(value) for value in obj]
    return str(obj)


def build_day04_leaderboard(summary_df=None, balance_df=None, grid_params_map=None, run_prefix="day04"):
    """
    Construye y ordena el leaderboard de variantes evaluadas en Day 04.
    """
    rows = []
    grid_params_map = grid_params_map or {}

    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        for _, row in summary_df.iterrows():
            variant_name = str(row.get("model"))
            rows.append({
                "model_variant": variant_name,
                "search_type": "grid",
                "best_params_json": json.dumps(_safe_json_value(grid_params_map.get(variant_name, {})), ensure_ascii=False),
                "cv_bal_acc": float(row.get("cv_bal_acc", np.nan)),
                "test_acc": float(row.get("test_acc", np.nan)),
                "test_bal_acc": float(row.get("test_bal_acc", np.nan)),
                "test_f1_pos": float(row.get("test_f1_pos", np.nan)),
                "top1_hit": float(row.get("top1_hit", np.nan)),
                "top2_hit": float(row.get("top2_hit", np.nan)),
                "notes": "from grid search summary",
            })

    if isinstance(balance_df, pd.DataFrame) and not balance_df.empty:
        for _, row in balance_df.iterrows():
            rows.append({
                "model_variant": str(row.get("variante")),
                "search_type": "imbalance_variant",
                "best_params_json": json.dumps({}, ensure_ascii=False),
                "cv_bal_acc": np.nan,
                "test_acc": float(row.get("acc", np.nan)),
                "test_bal_acc": float(row.get("bal_acc", np.nan)),
                "test_f1_pos": float(row.get("f1_pos", np.nan)),
                "top1_hit": float(row.get("top1_hit", np.nan)),
                "top2_hit": float(row.get("top2_hit", np.nan)),
                "notes": "from imbalance comparison table",
            })

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        raise ValueError("No hay resultados en summary_df/balance_df para seleccionar champion.")

    leaderboard = leaderboard.sort_values(
        ["top2_hit", "test_bal_acc", "test_f1_pos"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    run_id = f"{run_prefix}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    leaderboard["run_id"] = run_id
    leaderboard["is_champion"] = 0
    leaderboard.loc[0, "is_champion"] = 1

    return leaderboard


def refit_day04_champion_model(
    model_variant,
    grid_lr_estimator,
    grid_gb_estimator,
    X_train_hyper,
    y_train_hyper,
    X_train,
    y_train,
    c=1.0,
    max_iter=4000,
    random_state=0,
    sampling_strategy=0.5,
):
    """
    Reentrena la variante champion usando el split correspondiente.
    """
    if model_variant == "LogisticRegression (grid)":
        model = clone(grid_lr_estimator)
        X_fit, y_fit = X_train_hyper, y_train_hyper
    elif model_variant == "GradientBoosting (grid)":
        model = clone(grid_gb_estimator)
        X_fit, y_fit = X_train_hyper, y_train_hyper
    else:
        variants = build_default_lr_balance_variants(
            c=c,
            max_iter=max_iter,
            random_state=random_state,
            sampling_strategy=sampling_strategy,
        )
        variant_map = {name: (mdl, sampler) for name, mdl, sampler in variants}
        if model_variant not in variant_map:
            raise ValueError(f"Variant no soportada para reentreno: {model_variant}")

        base_model, sampler = variant_map[model_variant]
        model = clone(base_model)

        if sampler is not None:
            sampler_fit = clone(sampler)
            X_fit, y_fit = sampler_fit.fit_resample(X_train, y_train)
        else:
            X_fit, y_fit = X_train, y_train

    model.fit(X_fit, y_fit)
    return model, X_fit


def save_champion_artifacts(
    model,
    model_dir,
    model_name,
    metrics,
    cutoff_date,
    dataset_name,
    feature_columns,
    selection_rule,
):
    """
    Guarda el modelo champion y su metadata en disco.
    """
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    model_path = model_dir_path / "model.pkl"
    metadata_path = model_dir_path / "metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "params": _safe_json_value(model.get_params()),
        "selection_rule": selection_rule,
        "cutoff_date": str(pd.to_datetime(cutoff_date).date()) if cutoff_date is not None else None,
        "dataset": dataset_name,
        "metrics": _safe_json_value(metrics),
        "feature_columns": [str(column) for column in feature_columns],
        "timestamp_utc": f"{datetime.utcnow().isoformat()}Z",
    }

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return model_path, metadata_path, metadata

# FUNCIÓN DE TABLA RESUMEN PARA REGLAS DETERMINISTAS

def tabla_resumen(df):
    """
    Toma un dataframe y devuelve una tabla resumen para análisis.
    """
    # Se define la tabla
    combo = (
        df.groupby(["producto_canonico", "terminal_compra", "proveedor_candidato"], as_index=False)
        .agg(
            rows=("event_id", "size"),
            events=("event_id", "nunique"),
            wins=("target_elegido", "sum"),
        )
    )
    combo["win_rate_event"] = combo["wins"] / combo["events"]
    # Se devuelve la tabla ordenada en orden descendente para los candidatos
    return combo.sort_values(["events", "rows"], ascending=False).head(20)

# FUNCIÓN PARA VALIDAR REGLAS CANDIDATAS

def check_rule_candidate(combo_df, producto, terminal, proveedor, min_events=100):
    """
    Toma la tabla resumen de la función tabla_resumen, el producto, la terminal y el proveedor
    con un mínimo de eventos = 100; devuelve un dict.
    """
    hit = combo_df[
        (combo_df["producto_canonico"] == producto) &
        (combo_df["terminal_compra"] == terminal) &
        (combo_df["proveedor_candidato"] == proveedor)
    ].copy()

    if hit.empty:
        return {
            "producto": producto, "terminal": terminal, "proveedor": proveedor,
            "events": 0, "wins": 0, "win_rate_event": np.nan,
            "pass_soporte": False, "pass_zero_wins": False, "pre_aprobada_shadow": False
        }

    row = hit.iloc[0]
    pass_soporte = int(row["events"]) >= min_events
    pass_zero_wins = int(row["wins"]) == 0

    return {
        "producto": producto,
        "terminal": terminal,
        "proveedor": proveedor,
        "events": int(row["events"]),
        "wins": int(row["wins"]),
        "win_rate_event": float(row["win_rate_event"]),
        "pass_soporte": pass_soporte,
        "pass_zero_wins": pass_zero_wins,
        "pre_aprobada_shadow": bool(pass_soporte and pass_zero_wins),
    }

# FUNCIÓN PARA CONSTRUIR DIRECTORIO DE TRAZAS FECHADO

def build_dated_trace_dir(
    root: Path,
    trace_rel_dir: str,
    run_ts_utc: str,
) -> Path:
    """
    Construye directorio de trazas con partición diaria en UTC.

    El formato final queda como `artifacts/public/<clase>/<YYYYMMDD>/`.
    """
    run_date_utc = run_ts_utc[:8]
    trace_dir = root / trace_rel_dir / run_date_utc
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


# FUNCIÓN PARA GUARDAR REGLAS DE NEGOCIO EN CSV EN CONFIG
def guardar_reglas_de_negocio(
    checks_df: pd.DataFrame,
    root_dir: Path | None = None,
    output_rel_path: str = "config/business_blocklist_rules.csv",
    trace_rel_dir: str = "artifacts/public/rules",
    reason_prefix: str = "shadow_min_v1_historico_0_wins_eventos_ge_",
    include_template_row: bool = True,
) -> dict:
    """
    Exporta reglas de negocio aprobadas a CSV operativo + copia trazable.

    Espera un DataFrame `checks_df` con columnas:
    - producto, terminal, proveedor, events, pre_aprobada_shadow

    Salidas:
    - Activo vigente: `config/business_blocklist_rules.csv`
    - Trazabilidad: `artifacts/public/rules/<YYYYMMDD>/business_blocklist_rules_shadow_v1_<timestamp>.csv`
    """
    root = Path(root_dir) if root_dir is not None else BASE_DIR
    rules_out_path = root / output_rel_path
    rules_out_path.parent.mkdir(parents=True, exist_ok=True)

    required_cols = {"producto", "terminal", "proveedor", "events", "pre_aprobada_shadow"}
    missing = required_cols - set(checks_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en `checks_df`: {sorted(missing)}")

    pre_mask = (
        checks_df["pre_aprobada_shadow"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes", "si", "sí"])
    )

    aprobadas = checks_df.loc[
        pre_mask, ["producto", "terminal", "proveedor", "events"]
    ].copy()

    if aprobadas.empty:
        raise ValueError("No hay reglas pre_aprobadas_shadow=True. Revisa criterios antes de guardar.")

    aprobadas["events"] = pd.to_numeric(aprobadas["events"], errors="coerce").fillna(0).astype(int)
    aprobadas = aprobadas.sort_values(
        ["producto", "terminal", "proveedor", "events"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    aprobadas["rule_id"] = [f"R{i:03d}" for i in range(1, len(aprobadas) + 1)]
    aprobadas["active"] = "1"
    aprobadas["fecha_inicio"] = "2020-01-01"
    aprobadas["fecha_fin"] = "2034-12-31"
    aprobadas["motivo"] = aprobadas["events"].map(
        lambda n: f"{reason_prefix}{int(n)}"
    )

    rules_df = aprobadas.rename(
        columns={
            "producto": "producto_canonico",
            "terminal": "terminal_canonico",
            "proveedor": "proveedor_canonico",
        }
    )[
        [
            "rule_id",
            "active",
            "fecha_inicio",
            "fecha_fin",
            "producto_canonico",
            "terminal_canonico",
            "proveedor_canonico",
            "motivo",
        ]
    ]

    if include_template_row:
        template_row = pd.DataFrame(
            [
                {
                    "rule_id": "R000",
                    "active": "0",
                    "fecha_inicio": "2020-01-01",
                    "fecha_fin": "2034-12-31",
                    "producto_canonico": "*",
                    "terminal_canonico": "*",
                    "proveedor_canonico": "DUMMY",
                    "motivo": "Plantilla de ejemplo desactivada",
                }
            ]
        )
        rules_export = pd.concat([rules_df, template_row], ignore_index=True)
    else:
        rules_export = rules_df.copy()

    rules_export.to_csv(rules_out_path, index=False)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_dir = build_dated_trace_dir(root=root, trace_rel_dir=trace_rel_dir, run_ts_utc=run_ts)
    trace_path = trace_dir / f"business_blocklist_rules_shadow_v1_{run_ts}.csv"
    rules_export.to_csv(trace_path, index=False)

    return {
        "rules_out_path": rules_out_path,
        "trace_path": trace_path,
        "n_active_rules": int(len(rules_df)),
        "rules_export": rules_export,
    }
