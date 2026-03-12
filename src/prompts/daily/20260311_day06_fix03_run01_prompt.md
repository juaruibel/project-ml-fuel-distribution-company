---
document_type: daily_prompt
prompt_id: P20_DAILY_PROMPT_REFRESH
date_utc: "2030-03-11"
day_id: "Day 06.fix03"
base_prompt_ref: "docs/context/proyecto_final.md"
version: run01
status: active
---

# Day 06.fix03 · Higiene estructural preparatoria · run01

## Objetivo de la sesión

Partir `src/ml/ui/product.py` y `src/ml/product/day06_runtime.py` por dominios funcionales para reducir riesgo de cambios posteriores, manteniendo la misma lógica de producto, scoring y persistencia.

## Decisiones cerradas

- La sesión se registra como `Day 06.fix03 · run01`, pero no cierra todavía la simplificación funcional del producto.
- `fix02` sigue pendiente como bloque SQL/reporting; este run solo prepara la base técnica para `fix02` y para el resto de `fix03`.
- La compatibilidad se conserva con fachada estable en `product.py` y thin re-exports en `day06_runtime.py`.
- Si aparece un ciclo de imports, se permite un helper puro adicional; la primera extracción válida es `src/ml/product/day06_normalization.py`.
- No se toca `app.py`, no se cambia el contrato `raw -> candidate_grain -> normalized -> scoring` y no se rediseña la UI.

## Delta vs run08

- `run08` cerró visibilidad honesta de progreso y normalización mínima de tablas dentro de archivos grandes.
- `run01` de `fix03` no añade comportamiento nuevo visible: reorganiza el código para que esos cambios y los siguientes se mantengan modificables sin reabrir monolitos.

## Validación mínima esperada

- `py_compile` en verde para fachadas y módulos nuevos.
- import smoke en verde para UI y runtime sin ciclos.
- smoke funcional reutilizando `src/ml/scripts/day06_fix01_smoke.py` con salida nueva en `artifacts/public/validations/day06_fix03/`.
- beta Playwright en la app real con artefactos en `output/playwright/day06_fix03_run01_hygiene/`.
