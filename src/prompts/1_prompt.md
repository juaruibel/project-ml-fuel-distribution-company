---
document_type: prompt_pack
prompt_pack_id: u07_ml_bootstrap_001
project: recommend_supplier
repository: proyecto-ml-source
owner: Project Author
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
  policy: "Conservar únicamente prompts ejecutados con outputs verificables"
  storage_location: "src/prompts/"
  exclusions:
    - prompts duplicados
    - variantes coloquiales no ejecutadas
    - borradores sin artefactos de salida
---

# Prompt Pack 01 · Unit 07 ML (Trazabilidad)

Este archivo consolida los prompts **ejecutados y útiles** en la fase inicial del proyecto.  
Formato normalizado: objetivo, entradas, instrucciones, salidas esperadas y criterios de aceptación.

## Inventario de prompts ejecutados

| Prompt ID | Objetivo | Output trazable principal |
|---|---|---|
| `P01_CONTEXT_U07` | Cargar contexto bootcamp + preparar Unit 07 | `../../../RESUME.md`, `../../../day01_ml.md` |
| `P02_FEASIBILITY_ML` | Evaluar viabilidad del proyecto de clasificación | `../../../plan_etl.md` |
| `P03_EDA_RAW` | EDA técnico inicial de fuentes raw | `../../../eda_inicial_comparativa_petroleras.md` |
| `P04_ETL_STRATEGY` | Diseñar ETL robusto y escalable | `../../../plan_etl.md` |
| `P05_PRIVACY_POLICY` | Definir política de anonimización/publicación | `../../../RESUME.md`, `../../../plan_etl.md` |
| `P06_DAY01_ROADMAP` | Plan operativo para Day 01 (4h de aplicación) | `../../../day01_ml.md` |
| `P07_REPO_DEVOPS` | Arquitectura repo privado/público y publish seguro | `../../scripts/publish_to_bootcamp.py`, `../../config/publish_manifest.json` |

---

## P01_CONTEXT_U07

### Objetivo
Cargar contexto académico-técnico del alumno y preparar entorno de trabajo de Unit 07.

### Entradas
- `AGENTS.md` (protocolo de estructura/nomenclatura)
- Unidades previas (`Unit 01` a `Unit 06`)
- Material de `Unit 07` (lessons/labs)

### Instrucciones normalizadas
1. Leer y cumplir `AGENTS.md`.
2. Revisar progresión histórica del alumno (estilo, nivel, fricciones).
3. Evaluar complejidad técnica de labs de Unit 07.
4. Preparar estructura de carpetas Day/lessons/README.
5. Confirmar readiness sin proponer todavía proyecto final.

### Salidas esperadas
- Estructura preparada para Unit 07.
- Confirmación de contexto cargado y nivel técnico estimado.

### Criterios de aceptación
- Organización alineada con `AGENTS.md`.
- Contexto histórico y técnico explícito antes de pasar a definición de proyecto.

---

## P02_FEASIBILITY_ML

### Objetivo
Emitir informe de viabilidad para proyecto ML de recomendación de proveedor.

### Entradas
- Backlog de ideas del alumno.
- Restricciones de alcance semanal (Unit 07).
- Disponibilidad de datos históricos (ofertas + compras).

### Instrucciones normalizadas
1. Evaluar la idea principal (clasificación de proveedor) con foco en esfuerzo ETL.
2. Evaluar alternativas (forecast precio, stock, rutas) con criterio realista.
3. Proponer opción exacta de proyecto viable para cumplir la unidad.

### Salidas esperadas
- Informe de viabilidad técnico-negocio.
- Recomendación de alcance final para la semana.

### Criterios de aceptación
- Honestidad de riesgo/tiempo.
- Priorización orientada a entrega en plazo.

---

## P03_EDA_RAW

### Objetivo
Realizar EDA inicial técnico sobre fuentes reales en `data/raw`.

### Entradas
- Carpeta raw de comparativas (`.xls/.xlsx`)
- Archivo documental `Compras.pdf` (solo revisión inicial)

### Instrucciones normalizadas
1. Inspeccionar cobertura, formatos, calidad estructural y anomalías.
2. Identificar riesgos de parseo y necesidad de tipado/mapeos.
3. Determinar viabilidad de extracción para construir dataset unificado.

### Salidas esperadas
- Diagnóstico EDA inicial y riesgos ETL.

### Criterios de aceptación
- Cobertura de calidad de datos, no solo descriptiva.
- Recomendaciones accionables para pipeline.

---

## P04_ETL_STRATEGY

### Objetivo
Diseñar plan ETL robusto, incremental y escalable.

### Entradas
- Hallazgos de EDA
- Restricciones de privacidad/publicación
- Requerimientos de trazabilidad y reproducibilidad

### Instrucciones normalizadas
1. Definir capas `raw/staging/curated/marts`.
2. Separar extracción, tipado, DQ, reconciliación e integración.
3. Incluir backlog ejecutable (`- [ ]`) con hitos y DoD.
4. Definir outputs para entrenamiento (`V1`, `V2 candidates`) y reportes DQ.

### Salidas esperadas
- Plan ETL completo (`plan_etl.md`) con fases y controles.

### Criterios de aceptación
- Pipeline reproducible.
- Escalabilidad prevista sin refactor total.

---

## P05_PRIVACY_POLICY

### Objetivo
Asegurar privacidad para demo bootcamp y publicación en portfolio.

### Entradas
- Estructura de datos empresariales.
- Arquitectura privada/pública de repos.

### Instrucciones normalizadas
1. Determinar si basta con excluir raw o si exige anonimización adicional.
2. Definir política de publicación por artefactos.
3. Establecer controles mínimos para evitar exposición de información sensible.

### Salidas esperadas
- Criterio de anonimización y alcance de publicación.

### Criterios de aceptación
- Riesgo de exposición reducido y documentado.

---

## P06_DAY01_ROADMAP

### Objetivo
Crear backlog operativo para Day 01 (enfoque práctico de 4h tras teoría/lab).

### Entradas
- Lección/lab de Day 31 (`7.1_intro_to_ml.ipynb`)
- Estado del proyecto (ETL completado)

### Instrucciones normalizadas
1. Traducir objetivos de la lección a tareas aplicadas al proyecto.
2. Estructurar checklist con dependencias y validaciones.
3. Priorizar tareas de valor para baseline inicial.

### Salidas esperadas
- `day01_ml.md` con checklist ejecutable.

### Criterios de aceptación
- Plan accionable en ventana de 4 horas.
- Tareas alineadas con lab y dataset disponible.

---

## P07_REPO_DEVOPS

### Objetivo
Definir arquitectura operativa privada/pública con automatización segura.

### Entradas
- Repos `proyecto-ml-source` y `proyecto-ml-public`
- Requisitos de no publicar ETL ni datos sensibles

### Instrucciones normalizadas
1. Configurar flujo Git independiente por repo.
2. Definir estrategia de publicación controlada.
3. Implementar script de publicación y manifiesto de archivos permitidos.
4. Documentar requisitos de entorno y reproducibilidad.

### Salidas esperadas
- `scripts/publish_to_bootcamp.py`
- `config/publish_manifest.json`
- Reglas de operación privada/pública

### Criterios de aceptación
- Repo público sin ETL ni datos sensibles.
- Publicación reproducible con comando único.

---

## Política de mantenimiento del prompt pack

1. Añadir solo prompts que hayan generado artefactos verificables.
2. Versionar cambios con fecha y motivo.
3. No incluir conversaciones exploratorias sin impacto real.
4. Mantener lenguaje técnico y eliminar formulaciones coloquiales.
