# Day 06.fix05 · Render deploy first · Prompt diario

## Contexto operativo
- Commit de partida: `09366e1`.
- `fix01`, `fix02`, `fix03` y `fix04` ya están cerrados.
- `Day 07` sale completo del plan activo y pasa a backlog postbootcamp.
- El foco inmediato deja de ser la demo local y pasa a `deploy privado en Render`.

## Objetivo del bloque
1. Dejar lista una exportación privada root-ready para GitHub + Render.
2. Proteger `cloud_demo` con `DAY06_DEMO_ACCESS_CODE`.
3. Fijar dos workbooks reales de demo:
   - `Ejemplo 1`: `Comparativa de precios 21-10-2015.xlsx`
   - `Ejemplo 2`: `Comparativa de precios 13-11-2015.xlsx`
4. Cerrar runbook, coste objetivo y checklist de despliegue de 48 horas.

## Contrato no negociable
- Proveedor objetivo: `Render`.
- Cuenta: personal, con login GitHub.
- Workspace: `Hobby`.
- Servicio: `Web Service` público, plan `Starter`, subdominio `onrender.com`.
- Perfil remoto: `DAY06_APP_PROFILE=cloud_demo`.
- Seguridad mínima: clave compartida vía `DAY06_DEMO_ACCESS_CODE`.
- Persistencia: efímera, sin `DAY06_SQL_STORE_PATH` activo por defecto.

## Artefactos esperados
- `scripts/build_day06_render_private_repo.py`
- `dist/day06_render_private_repo/`
- `README.md` + `RUNBOOK.md` dentro del repo exportado
- `assets/day06_demo_comparativa.xlsx`
- `assets/day06_demo_comparativa_2.xlsx`
- `src/prompts/prompt_artifact_manifest.yaml` sincronizado

## Validación obligatoria
- `py_compile` de módulos tocados.
- `src/ml/scripts/day06_fix01_smoke.py`
- `src/ml/scripts/day06_fix02_sql_smoke.py`
- Streamlit headless en repo completo y en `dist/day06_render_private_repo/`.
- Beta Playwright real con passphrase y con `Ejemplo 1` + `Ejemplo 2`.
