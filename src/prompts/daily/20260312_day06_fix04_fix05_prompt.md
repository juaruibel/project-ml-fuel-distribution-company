# Day 06.fix04 + Day 06.fix05 · Prompt diario

## Contexto operativo
- Commit de partida: `09366e1`
- `fix01`, `fix02` y `fix03` ya cerrados.
- `Producto-Usuario` es la superficie principal.
- `Producto-Dev` debe quedar separado y secundario.
- No reabrir scoring, training, schema SQL ni lógica core salvo compatibilidad menor para demo/deploy.

## Objetivo del bloque
1. Cerrar `fix04` como bloque de presentación:
   - reducir fricción en `Producto-Usuario`
   - añadir acceso rápido de ejemplo
   - reforzar el resumen final de propuesta
   - convertir `Demo` en superficie oficial de defensa del proyecto
2. Cerrar `fix05` como bloque de factibilidad cloud mínima:
   - introducir perfil `cloud_demo`
   - preparar bundle reducido de deploy
   - validar arranque local/headless y dejar claro qué queda fuera

## Puntos no negociables
- `Demo` debe mantener lo que ya mostraba y añadir expanders para `Notebook 19`, `Notebook 20`, `Notebook 21` y `Cómo usar la app`.
- Cada expander nuevo debe mostrar narrativa curada, hallazgos y placeholders PNG con el texto exacto que luego se convertirá en slide.
- `cloud_demo` debe mantener visibles `Producto-Usuario` y `Demo`, ocultando `Producto-Dev`.
- Persistencia cloud aceptada solo como efímera.

## Artefactos esperados
- UI Day 06 ajustada para presentación.
- `DAY06_APP_PROFILE=full|cloud_demo`.
- `requirements-cloud.txt`.
- `scripts/build_day06_cloud_demo_bundle.py`.
- bundle generado en `dist/day06_cloud_demo/`.
- `README.md`, `docs/context/proyecto_final.md`, `docs/context/RESUME.md` y `prompt_artifact_manifest.yaml` alineados.

## Validación obligatoria
- `py_compile` de módulos tocados.
- `src/ml/scripts/day06_fix01_smoke.py`
- `src/ml/scripts/day06_fix02_sql_smoke.py`
- Streamlit headless en repo y en bundle cloud.
- Beta Playwright real sobre `Producto-Usuario`, `Demo` y perfil `cloud_demo`.
