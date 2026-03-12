# Day 06.fix03 · run02 · Propuesta diaria de compra para administrativo

## Resumen
- Convertir `Producto-Usuario` en una superficie de propuesta diaria de compra, no de inspección del universo candidato.
- Mantener el backend técnico a grano evento y construir encima una capa UX que colapse a `producto` cuando el caso sea limpio.
- Preservar scoring, SQL store, `run_bundle`, `feedback.csv` y demás contratos backend sin redefinir el target histórico del modelo.

## Decisiones cerradas
- La salida visible de `Producto-Usuario` pasa a ser una tabla final de compra con:
  - `fecha`
  - `albaran`
  - `producto`
  - `litros`
  - `proveedor recomendado`
  - `alternativa`
  - `confianza`
  - `nota`
  - `decision final`
- El colapso visible intenta agrupar por `fecha_evento + albaran_id + producto_canonico`.
- Si la propuesta no es homogénea, el producto se desdobla por terminal y se explica; si tampoco basta, se deja una fila por evento sin enseñar `event_id`.
- El feedback ya no se aplica sobre una fila visible de evento, sino sobre la propuesta visible y luego hace fan-out a los `event_id` subyacentes.
- Las métricas visibles para usuario quedan limitadas a `Top 1` + `Top 2` oficiales del champion a nivel evento.

## Archivos a tocar
- `src/ml/ui/product_user.py`
- `src/ml/ui/product_user_frames.py` (nuevo)
- `src/ml/ui/state.py`
- `src/ml/ui/settings.py`
- `docs/context/proyecto_final.md`
- `docs/context/RESUME.md`
- `src/prompts/prompt_artifact_manifest.yaml`

## Validación obligatoria
- `py_compile` de `app.py` y los módulos UI tocados.
- Import smoke de `render_product_user_surface`, `build_input_plan_frame` y `build_purchase_proposal_frame`.
- Reejecutar:
  - `src/ml/scripts/day06_fix01_smoke.py`
  - `src/ml/scripts/day06_fix02_sql_smoke.py`
- Caso de aceptación con `Comparativa de precios 08-04-2024.xlsx`:
  - runtime técnico sigue generando `20` eventos;
  - `Producto-Usuario` colapsa a `6` propuestas visibles;
  - `PRODUCT_002` y `PRODUCT_003` quedan en `Revisar`;
  - feedback sobre una propuesta colapsada replica correctamente a todos los `event_id` subyacentes.
