---
document_type: daily_prompt
prompt_id: P20_DAILY_PROMPT_REFRESH
date_utc: "2030-03-11"
day_id: "Day 06.fix01"
base_prompt_ref: "docs/context/proyecto_final.md"
version: run08
status: active
---

# Day 06.fix01 · Progreso visible + normalización de tablas · run08

## Objetivo de la sesión

Cerrar dos mejoras coordinadas en `Producto` sin reabrir scoring ni contrato: progreso visible y honesto durante parseo/run, y normalización mínima de dtypes/display para reducir fragilidad UI y warnings evitables.

## Decisiones cerradas

- El progreso visible se resuelve con `Estado actual + historial corto`, sin progress bars falsas ni stepper ornamental.
- `Excel raw` expone fases de parseo reales:
  - `Leyendo workbook`
  - `Extrayendo hojas`
  - `Construyendo candidate seed`
  - `Preparando enrichment manual`
- El run operativo expone fases reales vía callback opcional en runtime:
  - `normalizing_input`
  - `validating_input`
  - `validating_contract`
  - `running_scoring`
  - `preparing_outputs`
  - `persisting_artifacts`
- La normalización se reparte entre origen y display:
  - `albaran_id` y `linea_id` como `string`
  - `litros_evento` y placeholders numéricos relevantes como nullable (`Float64` / `Int64`)
  - tablas resumen/técnicas de pura presentación stringificadas explícitamente
- La beta real se hace con `playwright-cli` sobre la app viva y deja artefactos en `output/playwright/day06_fix01_feature7_8/`.

## Validación mínima esperada

- `py_compile` e import smoke en verde para `product.py`, `day06_runtime.py` y `day06_excel_raw.py`
- smoke funcional Day 06.fix01 en verde con evidencia de dtypes normalizados
- beta Playwright en verde para:
  - estado inicial
  - parseo con historial corto visible
  - run terminado con historial corto visible
  - captura de tabla relevante normalizada
- sin regresión del contrato `raw -> candidate_grain -> normalized -> scoring`
