---
document_type: daily_prompt
prompt_id: P20_DAILY_PROMPT_REFRESH
date_utc: "2030-03-11"
day_id: "Day 06.fix01"
base_prompt_ref: "docs/context/proyecto_final.md"
version: run04
status: active
---

# Day 06.fix01 · Claridad de modos deshabilitados · run04

## Objetivo de la sesión

Reducir la desconfianza de producto cuando `champion_pure` no está disponible por contrato para el input actual, sin ocultar el modo ni rediseñar el selector de modos.

## Decisiones cerradas

- El runtime ya devuelve suficiente información; la mejora se resuelve en `src/ml/ui/product.py`.
- El catálogo de modos sigue enseñando todos los modos, incluidos los deshabilitados.
- El selector real sigue mostrando solo modos habilitados.
- Cuando `champion_pure` esté deshabilitado, la UI debe mostrar un bloque corto y explícito que diga:
  - el modo existe;
  - hoy no está disponible para este input;
  - y cuál es el motivo exacto.
- Si aporta claridad real, el aviso puede resumir columnas faltantes o críticas a null, sin convertirse en ficha técnica larga.

## Restricciones

- No tocar scoring, runtime, SQL, demo ni deploy.
- No hacer rediseño visual grande.
- Mantener una UI sobria y funcional, siguiendo el criterio de `uncodixfy`.
- Actualizar trazabilidad en `docs/context/proyecto_final.md`, `docs/context/RESUME.md` y `src/prompts/prompt_artifact_manifest.yaml`.

## Validación mínima esperada

- `CSV` o `Formulario manual` sin contrato transport-only: `champion_pure` visible en catálogo, ausente del selector y explicado con aviso corto.
- Input válido para `V2_TRANSPORT_ONLY`: `champion_pure` habilitado y sin aviso de indisponibilidad.
- Caso sin modos viables: el error global no contradice el catálogo ni el motivo por modo.
