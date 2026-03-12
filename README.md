# Fuel Distribution Company · Supplier Recommendation

Public, pseudonymized, portfolio-ready version of a full machine learning project for daily supplier recommendation in fuel purchasing.

## What problem this project solves

In the original business flow, a buyer receives daily price comparisons from multiple suppliers and must decide who to buy from for each product and delivery context. That decision is repetitive, time-sensitive, and often driven by a mix of price, transport conditions, historical behavior, and local business rules.

This project turns that process into a reproducible decision-support system:

- ETL to build auditable training datasets from operational sources.
- ML models to rank supplier candidates.
- post-inference policy logic to reflect operational constraints.
- a Streamlit product layer to make the system usable by non-technical users.

The goal is not autopilot. The goal is a defendable human-in-the-loop assistant with traceability.

## What this public repo demonstrates

This repository is not a toy excerpt. It is the public portfolio version of the full project, prepared to show the actual work end to end while protecting sensitive information.

It demonstrates:

- a complete ETL and feature engineering pipeline;
- baseline and champion model training;
- notebook-driven analysis across the full project lifecycle;
- explainability work with error analysis and SHAP;
- operational policy analysis and product-oriented closure;
- a reproducible public audit process over the published artifacts.

## Project story

### 1. Data and ETL

The project starts by converting operational purchasing and offer data into structured marts for modeling. The ETL layer creates progressively richer datasets, moving from a baseline candidate dataset to transport-aware and context-enriched versions.

The public repo keeps the ETL code and the model-ready public datasets so the feature engineering story is visible, not hidden.

### 2. Baseline model

The initial control model is a logistic-regression baseline over the main candidate dataset. That baseline remains important throughout the project because every later candidate is evaluated against it using the same temporal logic and the same operational metrics.

### 3. Day 04 and Day 05 model iterations

The project then explores stronger candidates:

- richer context datasets built from the ETL pipeline;
- transport-related rebuilds and carry-forward strategies;
- tabular model families;
- a final pure champion based on the transport-only dataset.

The final pure champion promoted in the public repo is:

- `V2_TRANSPORT_ONLY_LIGHTGBM_CLASS_WEIGHT_BALANCED_v1`

### 4. Error analysis, SHAP, and operational policies

The work does not stop at raw model metrics. The later notebooks analyze:

- where the champion wins and where it still fails;
- which slices are operationally meaningful;
- whether local fallback rules or flags are defendable;
- why some apparent improvements should not be promoted.

That is why notebooks `19`, `20`, and `21` are especially important in this repo: they show the explanation, the restraint, and the decision logic behind the final project story.

### 5. Product layer

The original private project also includes a Streamlit product and a cloud demo profile. This public repo keeps a lightweight public app entrypoint so the product direction remains visible alongside the analytical work.

## Public publication strategy

This repo was built with a pseudonymization strategy designed to preserve the technical value of the project without exposing private business information.

What is preserved:

- the real project structure;
- the real notebooks;
- the base public datasets and support layers needed to rebuild the heavier derived marts;
- compressed public support files when the uncompressed staging would be too heavy for Git;
- public model retraining;
- explainability and policy analysis;
- reproducible public audit artifacts.

What is removed or excluded:

- raw source files;
- staging and curated business layers;
- operational documents and internal reports;
- direct business identifiers;
- Excel, PDF, and other sensitive source artifacts.

Why the repo contains `data/public`, `models/public`, and `artifacts/public`:

- `data/public` holds the pseudonymized base datasets plus the public support layers required to rebuild the heavier derived marts.
- `models/public` holds public models retrained from those public datasets.
- `artifacts/public` holds the auditable outputs needed to validate the public repo, including metric parity, notebook execution, and publication manifest files.

### Why some large CSVs are intentionally not versioned

The first public push proved that GitHub would accept the repository as-is, but it also showed that several derived mart CSVs were unnecessarily heavy for a portfolio-first repo. Those files are now treated as rebuildable local cache instead of tracked source.

What stays in Git:

- the base operational datasets needed for public retraining;
- the public support tables required to rebuild the derived marts;
- the executed notebooks, public models, and audit artifacts.

What is rebuilt locally on demand:

- `data/public/dataset_modelo_proveedor_v3_context.csv`
- `data/public/day041/*.csv`
- `data/public/day042/*.csv`
- `data/public/day043/*.csv`

This keeps the repo lighter to clone and review while preserving a defensible path to full reproduction.

One example is `data/public/support/ofertas_typed_full.csv.gz`: the compressed artifact is versioned, while `scripts/rebuild_public_datasets.py` materializes the local `data/public/support/ofertas_typed.csv` cache only when needed.

## Validation status

The current public state was validated locally before publication.

- public training: `PASS`
- public notebook execution: `21/21 PASS`
- strict public audit: `PASS`
- dataset metric parity: `PASS`
- sensitive token hits: `0`
- forbidden extensions in repo tree: `0`

Current public training snapshot:

| role | accuracy | balanced_accuracy | top2_hit |
|---|---:|---:|---:|
| baseline | `0.8874` | `0.8651` | `0.8583` |
| champion_pure | `0.9007` | `0.8814` | `0.8826` |

Audit evidence lives in:

- `artifacts/public/public_audit_summary.json`
- `artifacts/public/public_publish_manifest.json`
- `artifacts/public/models/public_training_summary.json`
- `artifacts/public/notebooks/notebook_execution_summary.json`

## Repository contents

```text
.
├── app.py
├── artifacts/public/
├── config/
├── data/public/
├── models/public/
├── notebooks/
├── scripts/
└── src/
```

Highlights:

- `notebooks/`: the 21 real notebooks of the project, sanitized and executable.
- `src/etl`, `src/ml`, `src/sql`, `src/prompts`: the public code and traceability layer.
- `scripts/rebuild_public_datasets.py`: rebuilds the large derived marts that are intentionally omitted from Git.
- `scripts/train_public_models.py`: retrains the public baseline and champion.
- `scripts/run_public_notebooks.py`: executes and sanitizes notebook outputs.
- `scripts/audit_public_repo.py`: runs the strict publication audit.

## How to run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/rebuild_public_datasets.py
python scripts/train_public_models.py
python scripts/run_public_notebooks.py
python scripts/audit_public_repo.py --strict
streamlit run app.py
```

Recommended order:

1. `python scripts/rebuild_public_datasets.py`
2. `python scripts/train_public_models.py`
3. `python scripts/run_public_notebooks.py`
4. `python scripts/audit_public_repo.py --strict`

## What to review first

If you want the shortest path through the repo:

1. `README.md`
2. `artifacts/public/public_audit_summary.json`
3. notebooks `19`, `20`, and `21`
4. `scripts/train_public_models.py`
5. `src/ml/`

## License

This public repository is released under the [MIT License](LICENSE).
