---
document_type: daily_prompt
prompt_id: P20_DAILY_PROMPT_REFRESH
date_utc: "2030-03-11"
day_id: "Day 06.fix01"
base_prompt_ref: "docs/context/proyecto_final.md"
version: run06
status: active
---

# Day 06.fix01 · Enrichment masivo en Excel raw · run06

## Objetivo de la sesión

Reducir el coste operativo real del enrichment en `Excel raw` añadiendo ayuda masiva explícita y auditable, sin tocar scoring, runtime, SQL, demo ni deploy.

## Decisiones cerradas

- La mejora se concentra en `src/ml/ui/product.py` con helpers puros en `src/ml/product/day06_excel_raw.py`.
- La ayuda masiva solo aplica sobre filas vacías.
- La primera versión soporta:
  - `albaran_id` por lote
  - `litros_evento` por lote
  - `linea_id` secuencial `1..N`
- No hay sobrescritura silenciosa ni automatismo opaco.
- La tabla sigue siendo el espacio de revisión fina antes de guardar.

## Validación mínima esperada

- smoke de compilación/imports
- flujo `Excel raw` funcional
- ayuda masiva usable en `Paso 3`
- beta test con Playwright y artefactos en `output/playwright/`
- sin regresión del contrato `raw -> candidate_grain -> normalized -> scoring`
