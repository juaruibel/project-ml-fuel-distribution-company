---
document_type: daily_prompt
prompt_id: P20_DAILY_PROMPT_REFRESH
date_utc: "2030-03-11"
day_id: "Day 06.fix01"
base_prompt_ref: "docs/context/proyecto_final.md"
version: run05
status: active
---

# Day 06.fix01 · Excel raw guiado por pasos · run05

## Objetivo de la sesión

Convertir `Excel raw` en un flujo secuencial y entendible de producto, evitando la sensación de pantalla técnica de staging sin tocar scoring, runtime, SQL, demo ni deploy.

## Decisiones cerradas

- La mejora se concentra en `src/ml/ui/product.py`.
- `Excel raw` se reordena en cinco pasos visibles:
  - `Sube archivo`
  - `Revisa parseo`
  - `Completa enrichment`
  - `Valida contrato`
  - `Ejecuta`
- Los pasos futuros siguen visibles pero atenuados con copy breve de bloqueo.
- No se crea una wizard ornamental ni una navegación compleja.
- El contrato `raw -> candidate_grain -> normalized -> scoring` se conserva intacto.
- La UI debe seguir el criterio de `uncodixfy`: normal, sobria y creíble, sin “AI slop”.

## Validación mínima esperada

- smoke de compilación/imports
- flujo `Excel raw` funcional con workbook real
- pasos visibles y en orden
- selector/CTA reservados al paso 5
- beta test final con Playwright y artefactos en `output/playwright/`
