> ⚠️ Esta es una copia anonimizada del README del entorno privado.
> Se eliminaron datos reales, ETL operativo y artefactos internos.
> Branding público: `Fuel Distribution Company, S.L.`.

## Snapshot público reproducible (sintético)

- `seed`: `42`
- `generated_utc`: `2026-03-02T07:54:08.411272+00:00`
- `rows_v1`: `1802`
- `rows_v2`: `14392`
- `events_v2`: `1802`
- `V1 Dummy accuracy`: `0.6011`
- `V1 KNN (k=5) accuracy`: `0.5429`
- `V2 Logistic accuracy`: `0.9045`
- `V2 Logistic balanced_accuracy`: `0.9263`
- `V2 Logistic f1_pos`: `0.7140`
- `V2 Logistic Top-1 hit`: `0.8128`
- `V2 Logistic Top-2 hit`: `0.9916`
- `Baseline más barato Top-1`: `0.8128`
- `Baseline más barato Top-2`: `0.9916`

> Estos valores se regeneran en cada publicación pública con datos sintéticos.

# Project · Unit 07 · Machine Learning · Recommend Supplier (Public Anonymized)

*MVP de recomendación de proveedor para compra de combustibles (ETL + ML)*

**Project Author**

---

## Índice

1. [Objetivo](#objetivo)
2. [Transparencia sobre uso de IA](#uso-ia)
3. [Trazabilidad de prompts](#trazabilidad-prompts)
4. [Instalación](#instalacion)
5. [Cómo replicar el proyecto](#como-replicar)
6. [Run app (Streamlit)](#run-app)
7. [Estructura del proyecto](#estructura-proyecto)
8. [Contexto de negocio](#contexto-negocio)
9. [Dataset](#dataset)
10. [Pipeline ETL](#pipeline-etl)
11. [Notebook 01 · KNN baseline](#nb01)
12. [Notebook 02 · Feature Engineering](#nb02)
13. [Notebook 03 · Ensemble](#nb03)
14. [Notebook 04 · Hyperparameter Tuning + Imbalanced](#nb04)
15. [MVP de inferencia y demo](#mvp-inferencia-demo)
16. [Resultados / Insights](#resultados-insights)
17. [Limitaciones y próximos pasos](#limitaciones-proximos-pasos)
18. [Licencia](#licencia)

---

<a id="objetivo"></a>
## Objetivo

El objetivo de este proyecto de Machine Learning es construir un MVP de recomendación de proveedor para una empresa de logística/venta de hidrocarburos, utilizando un histórico operativo de aproximadamente 10 años.

El problema de negocio es concreto: cada día se reciben ofertas de precios de distintos proveedores y hay que decidir con quién comprar. Esa decisión depende de patrones históricos y de reglas de negocio que no siempre están completamente explícitas en los datos.

A nivel técnico, el proyecto plantea una tarea de clasificación para estimar qué proveedor tiene mayor probabilidad de ser el elegido en cada evento de compra. Para validar utilidad real, el modelo se compara contra baselines operativos (`más barato`, `clase mayoritaria histórica`, `top-2 histórico`) y no solo contra métricas de laboratorio.

El objetivo operativo del MVP no es sustituir al decisor humano, sino ofrecer una recomendación asistida que reduzca tiempo de decisión, estandarice criterios y mejore trazabilidad. Si el modelo demuestra señal estable frente a baselines, se usará como base para iterar con reglas de negocio más sofisticadas y nuevas variables.

---

<a id="uso-ia"></a>
## Transparencia sobre uso de IA

Durante el proyecto se ha utilizado asistencia por IA en varias fases para acelerar la ejecución y guiar la consecución de objetivos que, sin esta ayuda, habrían tenido un coste temporal significativamente mayor (especialmente en ETL).

Para mantener trazabilidad metodológica, los prompts utilizados y el contexto de trabajo con el agente de código se documentan en `src/prompts`.

---

<a id="trazabilidad-prompts"></a>
## Trazabilidad de prompts

La trazabilidad de interacciones con IA se mantiene en `src/prompts/` con un esquema versionado y orientado a auditoría:

- `src/prompts/1_prompt.md`: bootstrap del proyecto (contexto, viabilidad, roadmap y arquitectura).
- `src/prompts/2_prompt_etl_execution.md`: ejecución ETL (V0, dual, cierre train-ready, V2).
- `src/prompts/3_prompt_ml_execution.md`: ejecución ML (Day 01 a Day 04, inferencia y cierre documental).
- `src/prompts/prompt_artifact_manifest.yaml`: mapa `prompt_id -> artefactos` verificados en repo.
- `src/prompts/README.md`: política de mantenimiento de la trazabilidad.

Reglas aplicadas:
- solo prompts ejecutados y con outputs verificables;
- exclusión de duplicados, borradores y variantes no usadas;
- metadata YAML normalizada (modelo, versión, alcance y criterios de inclusión).

---

<a id="instalacion"></a>
## Instalación

Requisitos:
- `Python 3.11.x`
- `git`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Cómo replicar el proyecto

Orden recomendado en público:

1. Instalar dependencias.
2. Entrenar modelo público sintético:
   - `python scripts/train_public_model.py`
3. Ejecutar notebooks (sin outputs) si quieres reproducir análisis.
4. Levantar app:
   - `streamlit run app.py`

Nota:
- El ETL real no se publica.
- Este repositorio se limita a capa ML + demo con datos sintéticos.

## Run app (Streamlit)

```bash
source .venv/bin/activate
python scripts/train_public_model.py
streamlit run app.py
```

Alcance público:
- datasets sintéticos en `data/synthetic/`;
- inferencia con `models/public_champion/model.pkl` tras entrenar localmente;
- ejemplo de entrada en `data/synthetic/inference_input_example_synthetic.csv`;
- salida de inferencia guardada en `data/synthetic/inference_outputs/`.

## Estructura del proyecto

```text
project-public/
  README.md
  LICENSE
  app.py
  requirements.txt
  requirements-dev.txt
  data/
    synthetic/
  notebooks/
    01_ml_knn.ipynb
    02_ml_feature_eng.ipynb
    03_ml_ensemble.ipynb
    04_ml_hyperparameter_tunning.ipynb
  src/
    ml/
    prompts/
  scripts/
    train_public_model.py
  docs/
    privacy_checklist.md
    public_publish_manifest.json
```

## Contexto de negocio

Actualmente, el administrativo recibe diariamente por email las ofertas de entre 12 y 28 proveedores. La operativa consiste en cargar esa información en Excel, ejecutar cálculos y seleccionar proveedor aplicando reglas de negocio. El proceso completo de preparación (copiar/pegar entradas y salidas, consolidar datos y revisar) puede superar 1 hora diaria, mientras que la decisión final de análisis/elección suele concentrarse en unos 10 minutos.

El MVP de este proyecto busca reducir esa fricción con una herramienta ejecutable en `app.py` (Streamlit) que entregue recomendaciones `Top-2` con enfoque **human in the loop**. La utilidad del sistema se evalúa contra baselines operativos (`clase mayoritaria`, `más barato`, `top-2 histórico`) para validar si realmente aporta señal de mejora frente a reglas simples.

La principal fricción técnica para escalar es la ingesta automática desde email, que puede convertirse en un punto sensible del pipeline por heterogeneidad de formatos y robustez operativa. En el estado actual, el proceso de decisión real incluye variables no completamente capturadas en los datos históricos: precio mínimo del día, línea de crédito/liquidez por proveedor, contexto de cierre de mes y señales externas como Brent. Estas variables se contemplan como fase de evolución del modelo.

El objetivo de producción no es una automatización ciega, sino un modo asistido con validación humana continua, con meta de alcanzar una cobertura operativa alta y una precisión estable en decisión diaria.


---

<a id="dataset"></a>
## Dataset

**Fuentes principales**

| Fuente | Tipo | Qué aporta | Estado |
|--------|------|-----------|--------|
| `SUPPLIER_DAILY_COMPARISON/*.xls*` | Excel | Ofertas diarias por terminal/producto/proveedor | ✅ |
| `compras totales.xls` | Excel | Compras reales históricas (ground truth) | ✅ |
| `Compras.pdf` | PDF | Histórico documental (no procesado en ETL) | ✅ (no pipeline) |

**Capas de datos**
- `raw`: ingesta original sin tocar.
- `staging`: tipado, limpieza y rechazos con trazabilidad.
- `curated`: tablas normalizadas de negocio (`fact_compras`, `fact_ofertas_diarias`, `join_diagnostico`).
- `marts`: datasets para entrenamiento/inferencia.

**Resumen de datasets públicos (sintéticos)**

| Dataset | Grano | Filas |
|---|---|---:|
| `dataset_modelo_proveedor_v1_synthetic.csv` | 1 línea de compra real | 1802 |
| `dataset_modelo_proveedor_v2_candidates_synthetic.csv` | 1 par (evento, proveedor candidato) | 14392 |
| `inference_input_example_synthetic.csv` | muestra de inferencia | variable |

## Pipeline ETL

- Esta sección se ejecuta solo en entorno privado y no se publica en este repositorio.
- En público se mantiene únicamente trazabilidad anonimizada en `src/prompts/`.

## Notebook 01 · KNN baseline

### Cierre Day 01 (Baseline sintético)

- Dataset: `data/synthetic/dataset_modelo_proveedor_v1_synthetic.csv`
- Split temporal reproducible (80/20).
- Dummy accuracy: `0.6011`
- KNN (`k=5`) accuracy: `0.5429`
- Lectura: baseline inicial de referencia para validar que el pipeline de evaluación está correcto.

## Notebook 02 · Feature Engineering

### Conclusiones (Day 02 · Feature Engineering sobre V2)

- Se trabajó sobre `data/synthetic/dataset_modelo_proveedor_v2_candidates_synthetic.csv`, cambiando de enfoque a clasificación binaria por candidato (`target_elegido`).
- El `feature engineering` aplicado en V2 mejora el comportamiento de KNN respecto al baseline inicial de Day 01.
- En `accuracy`, KNN queda muy cerca de `Dummy` (o lo supera levemente según `k`), sin una ventaja concluyente si se mira solo esa métrica.
- Esto es coherente con el desbalance de clases: un baseline mayoritario puede mantener accuracy alta sin capturar la lógica real de selección.
- Para este problema, las métricas prioritarias son `balanced_accuracy`, `f1` de clase positiva (`target_elegido=1`) y métricas por evento (`Top-1` / `Top-2 hit`).
- Conclusión operativa: KNN + FE aporta valor como baseline diagnóstico, pero el siguiente paso natural son modelos más robustos (LogReg y ensembles).

---

<a id="nb03"></a>
## Notebook 03 · Ensemble

### Cierre Day 03 (comparación operativa)

- Modelo de referencia público: `LogisticRegression` (binaria por candidato, V2).
- Métricas por fila:
  - `accuracy`: `0.9045`
  - `balanced_accuracy`: `0.9263`
  - `f1_pos`: `0.7140`
- Métricas por evento:
  - `Top-1 hit`: `0.8128`
  - `Top-2 hit`: `0.9916`
- Baseline más barato:
  - `Top-1`: `0.8128`
  - `Top-2`: `0.9916`

## Notebook 04 · Hyperparameter Tuning + Imbalanced

### Cierre Day 04 (modelo exportado en público)

- Champion público exportado: `models/public_champion/model.pkl`.
- Entrenamiento reproducible: `python scripts/train_public_model.py`.
- Métricas del champion (reproducibles con seed `42`):
  - `accuracy`: `0.9045`
  - `balanced_accuracy`: `0.9263`
  - `f1_pos`: `0.7140`
  - `Top-1`: `0.8128`
  - `Top-2`: `0.9916`

## MVP de inferencia y demo

El MVP público permite:

- entrenar un modelo reproducible sobre datos sintéticos;
- ejecutar inferencia por `event_id` y ranking `Top-k`;
- comparar métricas por fila y por evento en la demo;
- validar flujo de producto sin exponer activos privados.

## Resultados / Insights

### Métricas versionadas (public release)

- `rows_v1`: `1802`
- `rows_v2`: `14392`
- `events_v2`: `1802`
- V1:
  - `Dummy accuracy`: `0.6011`
  - `KNN k=5 accuracy`: `0.5429`
- V2 (`LogisticRegression`):
  - `accuracy`: `0.9045`
  - `balanced_accuracy`: `0.9263`
  - `f1_pos`: `0.7140`
  - `Top-1`: `0.8128`
  - `Top-2`: `0.9916`
- Baseline más barato:
  - `Top-1`: `0.8128`
  - `Top-2`: `0.9916`

Nota: todas estas métricas se recalculan durante `publish_to_bootcamp.py`.

## Limitaciones y próximos pasos

**Limitaciones (MVP)**
- Falta incorporar variables externas de negocio (p. ej. crédito/cupo/stock/mercado).
- Pipeline aún semimanual para operación diaria completa.
- Riesgo de sesgo por distribución histórica dominante de proveedores.
- En el estado actual, la recomendación debe mantenerse en modo **human-in-the-loop**.

**Próximos pasos**
- [ ] Integrar validación humana + feedback loop de nuevas decisiones.
- [ ] Mejorar comparativa de negocio y cobertura operativa.
- [ ] Incorporar nuevas señales de negocio (crédito, cupo, stock y variables externas) para robustecer la decisión.
- [ ] Definir criterios de promoción a operación diaria (métricas mínimas, cobertura y control de riesgo).

**Cierre**
- El MVP ya entrega valor operativo inicial y una base técnica reutilizable para iteraciones progresivas hacia producción.

---

<a id="licencia"></a>
## Licencia

Este repositorio se distribuye bajo licencia MIT.  
Consulta el archivo `LICENSE` para el texto completo.
