from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_MARTS_DIR = DATA_DIR / "public"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "public"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"
DIST_DIR = PROJECT_ROOT / "dist"

BASELINE_MODEL_PATH = MODELS_DIR / "public" / "baseline" / "model.pkl"
BASELINE_METADATA_PATH = MODELS_DIR / "public" / "baseline" / "metadata.json"
CHAMPION_MODEL_PATH = MODELS_DIR / "public" / "champion_pure" / "model.pkl"
CHAMPION_METADATA_PATH = MODELS_DIR / "public" / "champion_pure" / "metadata.json"

RULES_CSV_PATH = CONFIG_DIR / "business_blocklist_rules.csv"
INPUT_CONTRACT_PATH = CONFIG_DIR / "inference_input_contract.yaml"
OFERTAS_LAYOUT_RULES_PATH = CONFIG_DIR / "ofertas_layout_rules.json"
OFERTAS_CALCULOS_LAYOUT_RULES_PATH = CONFIG_DIR / "ofertas_calculos_layout_rules.json"
PRODUCTOS_MAPPING_PATH = CONFIG_DIR / "productos_mapping_v1.csv"
PROVEEDORES_MAPPING_PATH = CONFIG_DIR / "proveedores_mapping_v1.csv"
TERMINALES_MAPPING_PATH = CONFIG_DIR / "terminales_mapping_v1.csv"
REGISTRY_CSV_PATH = REPORTS_DIR / "metrics" / "final_baseline_vs_candidates.csv"
INPUT_VALIDATION_REPORTS_DIR = REPORTS_DIR / "validations" / "input_daily"

SAMPLE_INPUT_PATH = DATA_MARTS_DIR / "inference_inputs" / "example_real_day_2024-05-28.csv"
LEGACY_OUTPUTS_DIR = DATA_MARTS_DIR / "inference_outputs"
DAY01_METRICS_PATH = REPORTS_DIR / "ml_day01_baseline_metrics.md"
DAY02_QUALITY_PATH = REPORTS_DIR / "data_quality_v2_candidates.json"
DAY04_RESULTS_PATH = REPORTS_DIR / "ml_day04_tuning_results.csv"
V1_PATH = DATA_MARTS_DIR / "dataset_modelo_proveedor_v1.csv"
V2_PATH = DATA_MARTS_DIR / "dataset_modelo_proveedor_v2_candidates.csv"
DAY06_DEMO_WORKBOOK_PATH = ASSETS_DIR / "day06_demo_comparativa.xlsx"
DAY06_DEMO_WORKBOOK_SOURCE_NAME = "Comparativa de precios 21-10-2015.xlsx"
DAY06_DEMO_WORKBOOK_PATH_2 = ASSETS_DIR / "day06_demo_comparativa_2.xlsx"
DAY06_DEMO_WORKBOOK_SOURCE_NAME_2 = "Comparativa de precios 13-11-2015.xlsx"
DAY06_SLIDES_DIR = ASSETS_DIR / "slides" / "day06"
DAY06_CLOUD_BUNDLE_DIR = DIST_DIR / "day06_cloud_demo"
DAY06_RENDER_PRIVATE_REPO_DIR = DIST_DIR / "day06_render_private_repo"
