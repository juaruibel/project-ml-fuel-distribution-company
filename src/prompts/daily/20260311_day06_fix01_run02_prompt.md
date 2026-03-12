# Day 06.fix01 · Reapertura acotada del contrato de modos · run02

## Objetivo de la sesión

Corregir la selección, disponibilidad y recomendación de modos en la superficie `Producto` para que:

- `Excel raw` sea el flujo principal del producto;
- `champion_pure` solo quede habilitado y recomendado cuando el input actual satisfaga de verdad el contrato `V2_TRANSPORT_ONLY`;
- `CSV` y `Formulario manual` no se comporten como si el champion fuera siempre viable;
- `baseline` y `baseline_with_policy` sigan visibles como alternativas válidas según contrato;
- y la UX nominal deje de ser `fallar primero y rerun después`.

## Decisiones cerradas

- La disponibilidad de modos no sale de texto estático ni de la pestaña activa; sale de una verificación real del contrato del input.
- El preflight debe reutilizar la misma normalización Day 06 y la misma lógica contractual de scoring.
- Si el champion no está disponible:
  - `baseline_with_policy` será el default solo cuando exista agrupación útil por `albaran_id`;
  - en otro caso el default será `baseline`.
- Los modos deshabilitados siguen visibles, pero no seleccionables.
- No se rediseña la UI en profundidad; solo lógica mínima de disponibilidad/recomendación y copy asociado.

## Trazabilidad mínima esperada

- Backend: `src/ml/product/day06_runtime.py`
- UI: `src/ml/ui/product.py`
- Smoke: `src/ml/scripts/day06_fix01_smoke.py`
- Reporte: `artifacts/public/validations/day06_fix01/20260311_day06_fix01_mode_availability_smoke_report.json`
- Docs: `docs/context/proyecto_final.md`, `docs/context/RESUME.md`
