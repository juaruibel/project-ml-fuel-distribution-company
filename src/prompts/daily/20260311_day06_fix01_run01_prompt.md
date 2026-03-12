---
document_type: daily_prompt
prompt_id: P20_DAILY_PROMPT_REFRESH
date_utc: "2030-03-11"
day_id: "Day 06.fix01"
base_prompt_ref: "docs/context/proyecto_final.md"
version: run01
status: active
---

# Day 06.fix01 · Prompt operativo (run01)

## Objetivo
Cerrar el núcleo técnico `raw workbook -> candidate_grain -> normalized -> scoring` antes de cualquier iteración visual, demo final o deploy.

## Contrato fijado
- `Excel raw` soportado solo para la familia `Comparativa de precios` (`.xls/.xlsx`) con hojas `Tabla` y `Cálculos`.
- El parser debe construir un `candidate_grain.csv` auditable a una fila por `event_id + proveedor_candidato`.
- El enrichment faltante del workbook (`litros_evento`, `albaran_id`, `linea_id`) se captura manualmente en UI por evento antes del scoring.
- `baseline` y `baseline_with_policy` usan contrato V2 base.
- `champion_pure` usa ruta contract-aware `V2_TRANSPORT_ONLY` y falla pre-scoring si falta cualquier `v41_transport_*`.
- No hay fallback automático del champion; la UI solo ofrece rerun explícito a `baseline` o `baseline_with_policy`.

## Artefactos obligatorios
- `data/raw/inference_inputs/<YYYYMMDD>/<run_id>_input_original.<xls/xlsx>`
- `data/processed/inference_inputs/<YYYYMMDD>/<run_id>_candidate_grain.csv`
- `data/processed/inference_inputs/<YYYYMMDD>/<run_id>_normalized.csv`
- `artifacts/public/inference_runs/<YYYYMMDD>/<run_id>_scoring_contract.json`
- `artifacts/public/inference_runs/<YYYYMMDD>/<run_id>_run_manifest.json`
- `artifacts/public/inference_feedback/<YYYYMMDD>/<run_id>_feedback.csv`

## Validación mínima
- Smoke reproducible con workbook real `Comparativa de precios 21-10-2015.xlsx`.
- Inventario explícito de columnas:
  - raw transformado / `candidate_grain`
  - `normalized.csv`
  - baseline expected features
  - champion expected features
- Prueba negativa donde el champion falle si faltan columnas `v41_transport_*`.

## Restricciones
- No tocar SQL/Tableau salvo para declarar explícitamente que quedan fuera.
- No mover lógica pesada a `app.py`.
- No reentrenar ni cambiar serving default efectivo.
- Mantener trazabilidad completa en `docs/context/*` y `src/prompts/prompt_artifact_manifest.yaml`.
