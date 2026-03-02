---
document_type: prompt_pack
prompt_pack_id: u07_ml_model_public_003
project: project-ml-fuel-distribution-company
repository: project-ml-fuel-distribution-company
language: es
status: active
version: 2.0.0
created_at: "2026-03-02"
last_updated_at: "2026-03-02"
llm:
  provider: OpenAI
  model: "GPT-5.3 Codex Extra-high"
  modality: code_agent
traceability:
  scope: "Modelado, evaluación y producto de inferencia"
---

# Prompt Pack 03 · ML execution

## Inventario de prompts ejecutados

| Prompt ID | Objetivo | Artefactos públicos vinculados |
|---|---|---|
| `P08_DAY01_BASELINE` | Baseline inicial KNN con split temporal | `notebooks/01_ml_knn.ipynb`, `src/ml/functions.py` |
| `P09_DAY02_FEATURE_ENG` | Feature engineering y preparación V2 | `notebooks/02_ml_feature_eng.ipynb`, `src/ml/functions.py` |
| `P10_DAY03_MODEL_SELECTION` | Comparativa de modelos y lectura por evento | `notebooks/03_ml_ensemble.ipynb` |
| `P11_DAY04_TUNING` | Tuning + desbalanceo y export de champion público | `notebooks/04_ml_hyperparameter_tunning.ipynb`, `scripts/train_public_model.py` |
| `P12_INFERENCE_APP` | Flujo de inferencia CLI + app Streamlit | `src/ml/recommend_supplier.py`, `app.py` |

## Resumen operativo

- Evaluación alineada con métricas por fila y por evento (`Top-1`/`Top-2`).
- Modelo público entrenable localmente sobre datos sintéticos.
- Inferencia reproducible desde CSV en CLI y Streamlit.
