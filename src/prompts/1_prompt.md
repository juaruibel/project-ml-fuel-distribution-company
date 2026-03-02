---
document_type: prompt_pack
prompt_pack_id: u07_ml_bootstrap_public_001
project: project-ml-fuel-distribution-company
repository: project-ml-fuel-distribution-company
language: es
status: active
version: 2.0.0
created_at: "2026-03-02"
last_updated_at: "2026-03-02"
llm:
  provider: OpenAI
  model: "GPT-5.3 Codex Extra-high"
  modality: code_agent
traceability:
  policy: "Conservar solo prompts con evidencia en el repositorio público"
---

# Prompt Pack 01 · Contexto y setup inicial

## Inventario de prompts ejecutados

| Prompt ID | Objetivo | Artefactos públicos vinculados |
|---|---|---|
| `P01_CONTEXT_SETUP` | Preparar contexto técnico del proyecto y estructura de trabajo | `README.md`, `src/prompts/README.md` |
| `P02_FEASIBILITY` | Evaluar viabilidad de clasificación de proveedor y alcance de la unidad | `README.md` |
| `P03_PRIVACY_RELEASE` | Definir política de anonimización y publicación pública | `docs/privacy_checklist.md`, `README.md` |
| `P04_DAY01_ROADMAP` | Definir enfoque de baseline inicial para Day 01 | `notebooks/01_ml_knn.ipynb` |

## Resumen operativo

- Se consolidó el objetivo de negocio: recomendación `Top-k` asistida, no autopilot ciego.
- Se fijó una estrategia de publicación pública con datos sintéticos y sin ETL privado.
- Se estableció trazabilidad de prompts en `src/prompts/`.
