# Day 06.fix02 · Run 01 · Prompt diario

**Fecha:** 2030-03-11
**Run ID:** 20260311_day06_fix02_run01
**Sesión:** Implementación de `Day 06.fix02 · Persistencia canónica + SQL/reporting listo para Tableau`

## Decisiones del operador

1. Fuente de verdad primaria = run bundle local (CSV/JSON). Sin cambio.
2. Store SQL oficial = SQLite local explícito y estable, no MySQL.
3. Variable canónica: `DAY06_SQL_STORE_PATH` (compatibilidad con `DAY06_SQL_MIRROR_PATH`).
4. No acoplar `src/sql/db_init.sql` ni `src/sql/src/sql/public_db_views_serving.sql` en esta iteración.
5. No tocar scoring, contratos, `app.py` ni el flujo `raw -> candidate_grain -> normalized -> scoring`.

## Artefactos generados

### Código
- `NEW` `src/ml/product/day06_sql_store.py` — servicio puro (6 tablas, 5 vistas, 5 funciones públicas)
- `MODIFY` `src/ml/product/day06_sql_mirror.py` — thin wrapper de compatibilidad
- `MODIFY` `src/ml/product/day06_runtime.py` — publica a store al final de run exitoso + campos `sql_store_*`
- `MODIFY` `src/ml/product/day06_feedback.py` — republica feedback a store + campos `sql_store_*`
- `MODIFY` `src/ml/ui/results.py` — panel SQL reporting (init/status + republicación manual) + métrica card
- `MODIFY` `src/ml/ui/feedback.py` — sincroniza `sql_store_status` al guardar con rerender controlado
- `MODIFY` `src/ml/ui/product.py` — wired `render_sql_reporting_panel`
- `NEW` `src/ml/scripts/day06_fix02_sql_smoke.py` — smoke test (6 checks)

### Validación
- `artifacts/public/validations/day06_fix02/20260312T082135Z_day06_fix02_sql_smoke_report.json` — 6/6 PASS + `feedback_sync_coherent=true`
- `day06_fix01_smoke.py` — exit 0
- Streamlit headless launch — OK

### Documentación
- `docs/context/proyecto_final.md` — fix02 checkboxes ✅
- `docs/context/RESUME.md` — fix02 closure bullets
- `src/prompts/daily/20260311_day06_fix02_run01_prompt.md` — este archivo
