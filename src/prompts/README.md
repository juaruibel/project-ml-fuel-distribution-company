# Prompt Traceability

Carpeta de trazabilidad de prompts utilizados durante el proyecto.

## Convención

- Guardar solo prompts ejecutados con artefactos verificables.
- Mantener metadata YAML en cada archivo.
- Excluir borradores, duplicados y variantes no utilizadas.

## Inventario activo

- `1_prompt.md` · Prompt pack inicial de Unit 07 (contexto, viabilidad, EDA, roadmap y arquitectura de repos).
- `2_prompt_etl_execution.md` · Prompt pack de ejecución ETL (V0, dual, train-ready, V2 candidates).
- `3_prompt_ml_execution.md` · Prompt pack de ejecución ML (Day 01 a Day 04 + inferencia + documentación).
- `4_prompt_session_protocol.md` · Protocolo de arranque de sesión IA (prompt base literal + plantilla de refresco diario).
- `daily/` · Histórico oficial de prompts diarios (`<YYYYMMDD>_dayNN_prompt.md`).
- `prompt_artifact_manifest.yaml` · Mapa de trazabilidad prompt -> artefactos verificados.

## Mantenimiento del protocolo diario

- Cada cambio de prompt diario debe actualizar `prompt_artifact_manifest.yaml`.
- El cierre de cada jornada debe dejar un archivo en `src/prompts/daily/` con `delta` explícito.
- No arrastrar contexto conversacional largo: la sesión siguiente arranca desde prompt base + último prompt diario trazado.
