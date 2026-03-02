---
document_type: prompt_pack
prompt_pack_id: u07_ml_etl_public_002
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
  scope: "Decisiones ETL y publicación de datasets sintéticos"
---

# Prompt Pack 02 · ETL (resumen público)

## Inventario de prompts ejecutados

| Prompt ID | Objetivo | Artefactos públicos vinculados |
|---|---|---|
| `P05_ETL_ARCHITECTURE` | Definir arquitectura por capas (`raw/staging/curated/marts`) | `README.md` |
| `P06_ETL_DATASETS` | Cerrar contrato de datasets de modelado (`V1` y `V2`) | `data/synthetic/dataset_modelo_proveedor_v1_synthetic.csv`, `data/synthetic/dataset_modelo_proveedor_v2_candidates_synthetic.csv` |
| `P07_ETL_PUBLICATION_POLICY` | Establecer qué no se publica (ETL privado, datos reales, modelos reales) | `docs/privacy_checklist.md`, `README.md` |

## Resumen operativo

- El ETL real se mantiene fuera del repositorio público.
- Se publica solo capa de consumo ML con datos sintéticos reproducibles.
- El flujo público se centra en entrenamiento local (`scripts/train_public_model.py`) e inferencia demo (`app.py`).
