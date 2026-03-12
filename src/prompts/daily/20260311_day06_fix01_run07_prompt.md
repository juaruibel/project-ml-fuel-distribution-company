---
document_type: daily_prompt
prompt_id: P20_DAILY_PROMPT_REFRESH
date_utc: "2030-03-11"
day_id: "Day 06.fix01"
base_prompt_ref: "docs/context/proyecto_final.md"
version: run07
status: active
---

# Day 06.fix01 · Separación de mensajes de usuario y detalle técnico · run07

## Objetivo de la sesión

Separar en `Producto` el mensaje operativo visible para usuario del diagnóstico técnico persistido para soporte, auditoría y debugging, sin tocar scoring ni logging global.

## Decisiones cerradas

- La mejora se reparte entre `src/ml/product/day06_scoring_contracts.py`, `src/ml/product/day06_runtime.py` y `src/ml/ui/product.py`.
- El backend sigue generando y persistiendo el detalle técnico completo.
- Se añade una capa semántica mínima con:
  - `user_message`
  - `developer_detail`
  - `severity`
  - `action_hint`
- La UI muestra por defecto solo mensaje corto y accionable.
- El detalle técnico queda en `expander` secundario o en artefactos descargables.

## Validación mínima esperada

- smoke de compilación/imports
- mensajes de usuario comprensibles en modos, validación y fallos pre-scoring
- detalle técnico preservado en manifests/reportes y visible en segundo plano
- beta test con Playwright y artefactos en `output/playwright/`
- sin regresión del flujo `Excel raw` ni del contrato `raw -> candidate_grain -> normalized -> scoring`
