# Day 06.fix03 · run01 · Producto-Usuario rediseñado + Producto-Dev separado

## Resumen
- Replantear `fix03` como split por audiencia y por lenguaje visual.
- Crear `Producto-Usuario` como experiencia operativa principal para administrativo.
- Mantener la superficie actual como `Producto-Dev`.
- Aplicar rediseño completo solo a `Producto-Usuario` siguiendo `uncodixfy`, sin tocar scoring, SQL store ni contratos.

## Decisiones cerradas
- `Producto-Usuario` queda como superficie por defecto al abrir la app.
- `Producto-Usuario` solo trabaja con `Excel raw`.
- El modo queda fijado a `champion_pure`.
- `top_k` queda fijado a `2`.
- Si el champion no queda disponible, la superficie bloquea y no hace fallback a baseline.
- `linea_id` y `event_id` permanecen internos; no se enseñan en la vista diaria.
- `SQL reporting`, métricas runtime, manifiestos y tablas técnicas quedan fuera del flujo diario.
- `Tableau` y analítica visual no entran en `fix03`; se dejan fuera del flujo de usuario.

## Archivos a tocar
- `app.py`
- `src/ml/ui/settings.py`
- `src/ml/ui/state.py`
- `src/ml/ui/product.py`
- `src/ml/ui/product_user.py` (nuevo)
- `docs/context/proyecto_final.md`
- `docs/context/RESUME.md`
- `src/prompts/prompt_artifact_manifest.yaml`

## Implementación
- En `app.py`, pasar a tres superficies:
  - `Producto-Usuario`
  - `Demo`
  - `Producto-Dev`
- En `settings.py`, añadir theming scoped por superficie para aislar el rediseño visual de `Producto-Usuario`.
- En `product.py`, renombrar visualmente la superficie existente a `Producto-Dev` y mantenerla como fachada avanzada.
- Crear `product_user.py` con:
  - carga de `Excel raw`;
  - enrichment solo de `albaran_id` y `litros_evento`;
  - generación automática de `linea_id`;
  - run fijo `champion_pure`, `top_k=2`;
  - worklist a una fila por evento con `top1`, `top2`, `confianza`, `motivo revision` y `decision final`;
  - feedback guiado con acciones:
    - `Aceptar recomendacion`
    - `Elegir alternativa 2`
    - `Otro proveedor`
    - `Marcar para revision`
- Mantener la persistencia actual de `feedback.csv`, `run_manifest.json` y SQL store.

## Validación obligatoria
- `py_compile` de `app.py` y los módulos UI tocados.
- Import smoke de `render_product_user_surface`, `render_product_dev_surface` y `build_user_review_frame`.
- Reejecutar:
  - `src/ml/scripts/day06_fix01_smoke.py`
  - `src/ml/scripts/day06_fix02_sql_smoke.py`
- Beta real con Playwright en `output/playwright/day06_fix03_run01_user_mode/` verificando:
  - default en `Producto-Usuario`;
  - superficie visual distinta a `Producto-Dev`;
  - ausencia de `CSV`, `manual`, selector de modo, `Top-k`, SQL, métricas runtime, `event_id` y `linea_id` en modo usuario;
  - flujo completo `Excel raw -> enrichment -> run -> review -> feedback`;
  - supervivencia de `Producto-Dev` como superficie avanzada.
