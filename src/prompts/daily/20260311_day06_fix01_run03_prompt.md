---
document_type: daily_prompt
prompt_id: P20_DAILY_PROMPT_REFRESH
date_utc: "2030-03-11"
day_id: "Day 06.fix01"
base_prompt_ref: "docs/context/proyecto_final.md"
version: run03
status: active
---

# Day 06.fix01 · Limpieza lógica contextual UI · run03

## Objetivo de la sesión

Corregir la incoherencia de producto por la que la superficie `Producto` podía seguir mostrando el último run ejecutado aunque el usuario hubiera cambiado de `Excel raw`, `CSV` o `Formulario manual`.

## Decisiones cerradas

- La trazabilidad del backend se conserva: `run_manifest`, artefactos y feedback siguen persistiendo por `run_id`.
- La visibilidad de UI pasa a ser contextual al input visible, no global a toda la sesión.
- El control se resuelve en `src/ml/ui/product.py`, no en `src/ml/product/day06_runtime.py`, porque el problema es de presentación y estado visible.
- El criterio de equivalencia es un `input_context_fingerprint` exacto construido desde:
  - `input_mode`
  - `source_name`
  - `source_bytes` cuando existan
  - el contenido visible del `input_df` o del `candidate_grain` actual
- Si el contexto cambia, se ocultan `last_run`, validación, fallo contractual y feedback.
- Si el usuario vuelve exactamente al mismo contexto, ese estado reaparece.

## Restricciones

- No rediseñar la UI ni mover lógica pesada a `app.py`.
- No tocar scoring, runtime, SQL ni reglas de negocio previas salvo necesidad incidental no prevista.
- Mantener trazabilidad en `docs/context/proyecto_final.md`, `docs/context/RESUME.md` y `src/prompts/prompt_artifact_manifest.yaml`.

## Validación mínima esperada

- `CSV -> run -> cambio a manual -> ocultación -> vuelta al mismo CSV -> reaparición`
- `Excel raw` incompleto no hereda estado previo
- `Excel raw` completado con run propio sí muestra su estado
- feedback y fallo contractual no deben reaparecer en contextos ajenos
