---
document_type: prompt_pack
prompt_pack_id: u07_ml_model_execution_003
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
  scope: "Ejecución ML (Day 01 -> Day 04 + inferencia)"
  inclusion_rule: "Solo prompts vinculados a notebooks/scripts con evidencia en repo"
---

# Prompt Pack 03 · ML Execution (Trazabilidad)

## Inventario de prompts ejecutados

| Prompt ID | Objetivo | Artefactos trazables |
|---|---|---|
| `P13_DAY01_BASELINE` | Construir baseline KNN y validación temporal inicial | `notebooks/01_ml_knn.ipynb`, `src/ml/shared/functions.py` |
| `P14_DAY02_FEATURE_ENG` | Aplicar feature engineering con dataset V2 y evaluación top-k | `notebooks/02_ml_feature_eng.ipynb`, `src/ml/shared/functions.py`, `artifacts/public/data_dictionary_v2_candidates.md` |
| `P15_DAY03_MODEL_SELECTION` | Comparar modelos (ensemble + referencias) y decidir champion provisional | `notebooks/03_ml_ensemble.ipynb`, `src/ml/shared/functions.py` |
| `P16_DAY04_TUNING_IMB` | Tuning + desbalanceo + selección de champion con regla de negocio | `notebooks/04_ml_hyperparameter_tunning.ipynb`, `artifacts/public/ml_day04_tuning_results.csv`, `artifacts/public/ml_validation_protocol.md`, `artifacts/public/ml_day04_runbook.md`, `models/public/baseline/model.pkl`, `models/public/baseline/metadata.json` |
| `P17_INFERENCE_MVP` | Implementar inferencia CLI para recomendación de proveedor | `src/ml/product/recommend_supplier.py` |
| `P18_DOC_ALIGNMENT` | Alinear documentación final para presentación/reproducibilidad | `README.md`, `../../../day04_ml.md`, `../../../BACKLOG.md` |

---

## P13_DAY01_BASELINE

### Objetivo
Construir baseline técnico con KNN y establecer métricas mínimas de comparación.

### Entradas
- `dataset_modelo_proveedor_v1.csv`
- split temporal y funciones de preparación

### Instrucciones normalizadas
1. Preparar subset reproducible sin leakage.
2. Entrenar baseline KNN con pipeline de escalado.
3. Comparar con baseline `Dummy`.
4. Documentar lectura correcta de métricas en contexto desbalanceado.

### Salidas esperadas
- Notebook Day 01 consistente y ejecutable `Run All`.

### Criterios de aceptación
- Split temporal correcto.
- Métricas reportadas sin warning crítico.

---

## P14_DAY02_FEATURE_ENG

### Objetivo
Mejorar señal con `V2 candidates` y validar impacto en métricas operativas.

### Entradas
- `dataset_modelo_proveedor_v2_candidates.csv`
- contrato de features V2

### Instrucciones normalizadas
1. Construir dataframe de modelado para tarea binaria por candidato.
2. Aplicar codificación categórica y checks de calidad.
3. Calcular métricas por fila y por evento (`Top-1/Top-2`).
4. Mantener Day 02 sin tuning exhaustivo (evitar duplicidad con Day 04).

### Salidas esperadas
- Notebook Day 02 limpio, enfocado a FE y evaluación base.

### Criterios de aceptación
- Coherencia entre objetivo del día y alcance implementado.

---

## P15_DAY03_MODEL_SELECTION

### Objetivo
Comparar familias de modelos y elegir champion provisional para negocio.

### Entradas
- mismo split temporal fijo
- features y preprocesado de Day 02

### Instrucciones normalizadas
1. Evaluar modelos ensemble y referencias adicionales.
2. Comparar contra baselines de negocio.
3. Priorizar métricas por evento para recomendación.
4. Dejar conclusiones sin contradicciones entre celdas.

### Salidas esperadas
- Notebook Day 03 con decisión explícita de champion provisional.

### Criterios de aceptación
- Narrativa técnica y narrativa de negocio alineadas.

---

## P16_DAY04_TUNING_IMB

### Objetivo
Cerrar validación robusta con tuning y técnicas de balanceo; exportar champion.

### Entradas
- champion provisional Day 03
- protocolo de validación temporal

### Instrucciones normalizadas
1. Ejecutar `GridSearchCV` en modelos objetivo.
2. Evaluar variantes `class_weight`, under/over-sampling y SMOTE.
3. Seleccionar champion con regla determinista (`Top-2` -> `bal_acc` -> `f1_pos`).
4. Exportar artefactos (`model.pkl`, `metadata.json`) y documentación operativa.

### Salidas esperadas
- Day 04 cerrado con artefactos listos para inferencia/demo.

### Criterios de aceptación
- Reproducibilidad de selección de champion.
- Reportes y runbook disponibles.

---

## P17_INFERENCE_MVP

### Objetivo
Implementar inferencia reproducible para consumo en CLI/Streamlit.

### Entradas
- champion exportado de Day 04
- contrato de features de inferencia

### Instrucciones normalizadas
1. Cargar metadata y validar esquema de entrada.
2. Preparar matriz de features compatible con entrenamiento.
3. Generar score, ranking y top-k por evento.
4. Persistir salida con trazabilidad temporal.

### Salidas esperadas
- Script de inferencia funcional en `src/ml/product/recommend_supplier.py`.

### Criterios de aceptación
- Ejecución CLI sin ruptura de columnas.
- Salida utilizable en capa de presentación.

---

## P18_DOC_ALIGNMENT

### Objetivo
Alinear documentación técnica para handoff, demo y portfolio.

### Entradas
- estado final de notebooks/scripts/reportes
- requisitos de transparencia (privacidad + uso de IA)

### Instrucciones normalizadas
1. Unificar estructura del README.
2. Incluir reproducibilidad, límites y roadmap.
3. Documentar política de publicación privada/pública.
4. Registrar trazabilidad de prompts utilizados.

### Salidas esperadas
- Documentación consistente con el estado real del proyecto.

### Criterios de aceptación
- README navegable por índice.
- Sin claims no soportados por artefactos.
