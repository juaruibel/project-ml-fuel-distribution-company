#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml import functions as fc

DATA_PATH = PROJECT_ROOT / 'data' / 'synthetic' / 'dataset_modelo_proveedor_v2_candidates_synthetic.csv'
MODEL_DIR = PROJECT_ROOT / 'models' / 'public_champion'
MODEL_PATH = MODEL_DIR / 'model.pkl'
METADATA_PATH = MODEL_DIR / 'metadata.json'

def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    df_model = fc.df_model_knn_feature(frame)
    train_df, test_df, cutoff_date = fc.split_temporal_feature(df_model)
    feature_cols_num, feature_cols_cat, target_col = fc.get_feature_columns_v2()
    X_train, X_test, y_train, y_test = fc.dummificar_train_test(
        train_df, test_df, feature_cols_num, feature_cols_cat, target_col=target_col
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(C=1.0, max_iter=4000, random_state=42, class_weight='balanced')),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        'model_name': 'LogisticRegression_public_baseline',
        'dataset': DATA_PATH.name,
        'cutoff_date': str(pd.to_datetime(cutoff_date).date()),
        'feature_columns': [str(column) for column in X_train.columns],
        'metrics': {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
            'f1_pos': float(f1_score(y_test, y_pred, pos_label=1)),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'Model saved to: {MODEL_PATH}')
    print(f'Metadata saved to: {METADATA_PATH}')
    print('Metrics:', metadata['metrics'])

if __name__ == '__main__':
    main()
