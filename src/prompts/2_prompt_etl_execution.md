---
document_type: prompt_pack
prompt_pack_id: u07_ml_etl_execution_002
project: recommend_supplier
repository: proyecto-ml-source
language: es
status: active
version: 1.0.0
created_at: "2030-02-27"
last_updated_at: "2030-02-27"
llm:
  provider: OpenAI
  model: "GPT-5.3 Codex Extra-high"
  modality: code_agent
traceability:
  scope: "Ejecución ETL (V0 -> V1 -> V2)"
  inclusion_rule: "Solo prompts con artefactos confirmados en repo"
---

# Prompt Pack 02 · ETL Execution (Trazabilidad)

## Inventario de prompts ejecutados

| Prompt ID | Objetivo | Artefactos trazables |
|---|---|---|
| `P08_ETL_V0_HYBRID` | Implementar ETL de ofertas V0 (híbrido) + integración con compras typed | `src/etl/extract/extract_ofertas_raw.py`, `src/etl/transform/transform_ofertas_typed.py`, `src/etl/curated/build_fact_ofertas_diarias.py`, `src/etl/integrate/build_join_diagnostico.py`, `src/etl/marts/build_dataset_modelo_v1.py`, `artifacts/public/etl_dq_ofertas_tabla_extract.json`, `artifacts/public/etl_dq_ofertas_typed.json`, `artifacts/public/etl_join_coverage_v1.json` |
| `P09_ETL_V1_DUAL` | Extender a fuente dual (`Tabla` + `Cálculos`) con reconciliación | `src/etl/extract/extract_ofertas_calculos_raw.py`, `src/etl/integrate/reconcile_ofertas_sources.py`, `src/etl/quality/report_join_gaps.py`, `artifacts/public/etl_ofertas_source_reconciliation_v1.json`, `artifacts/public/etl_join_gap_analysis_v1.json` |
| `P10_ETL_TRAIN_READY_V1` | Cierre train-ready V1 y mejora de cobertura | `config/provider_aliases_backlog.csv`, `config/proveedores_mapping_v1.csv`, `data/public/dataset_modelo_proveedor_v1.csv`, `artifacts/public/archive/etl/etl_train_ready_v1_execution_notes.md` |
| `P11_ETL_V2_CANDIDATES` | Construir dataset V2 de candidatos (sin ML) | `src/etl/marts/build_dataset_modelo_v2_candidates.py`, `src/etl/quality/report_terminal_bridge_diagnostic.py`, `artifacts/public/data_dictionary_v2_candidates.md`, `artifacts/public/data_quality_v2_candidates.json`, `artifacts/public/data_terminal_bridge_diagnostic.md` |
| `P12_ETL_GOVERNANCE` | Formalizar capas de datos, pre-flight y flujo robusto | `src/etl/extract/discover_sources.py`, `config/etl_sources.json`, `../../../plan_etl.md`, `../../../RESUME.md` |

---

## P08_ETL_V0_HYBRID

### Objetivo
Implementar el tramo sensible de ofertas con estrategia híbrida y generar un primer dataset de entrenamiento.

### Entradas
- Fuentes `raw` de ofertas y compras.
- Reglas iniciales de layout/mapping.
- Requisitos `CSV-first` para bootcamp.

### Instrucciones normalizadas
1. Extraer señales fiables de ofertas (`coste/proveedor` mínimo) y matriz cruda de trazabilidad.
2. Tipar ofertas con reglas DQ explícitas.
3. Integrar con compras tipadas para diagnóstico de join.
4. Generar mart V1 para baseline.

### Salidas esperadas
- Pipeline V0 funcional de extremo a extremo (extract -> transform -> curated -> marts).

### Criterios de aceptación
- Reportes DQ y cobertura no vacíos.
- Dataset V1 generado y usable.

---

## P09_ETL_V1_DUAL

### Objetivo
Aumentar cobertura incorporando hoja `Cálculos` y reconciliación dual.

### Entradas
- Extractor V0 (`Tabla`).
- Nueva fuente `Cálculos`.
- Reglas de reconciliación y priorización.

### Instrucciones normalizadas
1. Extraer `Cálculos` en formato largo.
2. Reconciliar `Tabla` + `Cálculos` con estados `agree/conflict/single_source`.
3. Regenerar tipado y reportes de gaps.
4. Mantener retrocompatibilidad con pipeline existente.

### Salidas esperadas
- `ofertas_reconciled_raw` y reportes de reconciliación/gaps actualizados.

### Criterios de aceptación
- Cobertura de extracción sostenida.
- Sin ruptura de contratos previos.

---

## P10_ETL_TRAIN_READY_V1

### Objetivo
Cerrar versión train-ready V1 para iniciar modelado de Day 01.

### Entradas
- Backlog de aliases de proveedor.
- Reportes de gaps y cobertura.

### Instrucciones normalizadas
1. Resolver aliases de mayor impacto.
2. Reejecutar cadena parcial E2E.
3. Validar gates de calidad.
4. Congelar snapshot y contrato de features inicial.

### Salidas esperadas
- Dataset V1 estable para baseline.

### Criterios de aceptación
- Métricas de cobertura dentro de objetivo.
- Snapshot y manifest de control disponibles.

---

## P11_ETL_V2_CANDIDATES

### Objetivo
Construir dataset candidato por evento-proveedor para modelado más robusto.

### Entradas
- `fact_compras.csv`
- `fact_ofertas_diarias.csv`
- reglas de bloqueo de negocio

### Instrucciones normalizadas
1. Expandir candidatos por evento de compra.
2. Crear `target_elegido` binario.
3. Derivar features de competencia por evento.
4. Separar eventos excluidos con motivo.

### Salidas esperadas
- `dataset_modelo_proveedor_v2_candidates.csv`
- `dataset_modelo_proveedor_v2_excluded_events.csv`
- diccionario y reporte de calidad V2

### Criterios de aceptación
- Un único positivo por evento incluido.
- Reportes de calidad y diccionario completos.

---

## P12_ETL_GOVERNANCE

### Objetivo
Asegurar gobernanza de datos y robustez pre-ejecución.

### Entradas
- Nuevas fuentes (`compras totales.xls`).
- Política de descarte de `Compras.pdf`.
- No negociables de escalabilidad.

### Instrucciones normalizadas
1. Definir responsabilidades de capas `raw/staging/curated/marts`.
2. Ejecutar pre-flight de fuentes.
3. Confirmar arquitectura lista para ejecución inmediata.

### Salidas esperadas
- Flujo ETL gobernado y documentado.

### Criterios de aceptación
- Estructura coherente con crecimiento a producción.
