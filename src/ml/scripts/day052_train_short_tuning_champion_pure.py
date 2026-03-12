#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.ml.metrics import registry
    from src.ml.scripts.day05_train_tabular_candidates import (
        append_registry_candidate,
        build_base_model,
        build_pure_metrics_payload,
        evaluate_cv_trial,
        extract_reference_metrics,
        fit_and_evaluate_model,
        load_registry_reference_rows,
        resolve_cutoff_date,
        save_candidate_artifacts,
        smoke_imports,
        write_dataframe_outputs,
        write_metrics_payload,
    )
    from src.ml.shared.day05_tabular import get_day05_dataset_catalog, prepare_day05_model_frame, split_day05_by_cutoff
    from src.ml.shared.helpers import build_run_id, utc_now_iso, write_json
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.ml.metrics import registry
    from src.ml.scripts.day05_train_tabular_candidates import (
        append_registry_candidate,
        build_base_model,
        build_pure_metrics_payload,
        evaluate_cv_trial,
        extract_reference_metrics,
        fit_and_evaluate_model,
        load_registry_reference_rows,
        resolve_cutoff_date,
        save_candidate_artifacts,
        smoke_imports,
        write_dataframe_outputs,
        write_metrics_payload,
    )
    from src.ml.shared.day05_tabular import get_day05_dataset_catalog, prepare_day05_model_frame, split_day05_by_cutoff
    from src.ml.shared.helpers import build_run_id, utc_now_iso, write_json

DAY_ID = "Day 05.2"
TARGET_DATASET_ALIAS = "V2_TRANSPORT_ONLY"
TARGET_MODEL_FAMILY = "LIGHTGBM"
TARGET_BASE_VARIANT = "V2_TRANSPORT_ONLY_LIGHTGBM_CLASS_WEIGHT_BALANCED_v1"
SECONDARY_REFERENCE_VARIANTS = (
    "V2_LIGHTGBM_CLASS_WEIGHT_BALANCED_v1",
    "V2_TRANSPORT_CARRY30D_ONLY_LIGHTGBM_CLASS_WEIGHT_BALANCED_v1",
)
DEFAULT_CANDIDATE_DIR = Path("models/candidates/day05_2_short_tuning")
DEFAULT_CHAMPION_DIR = Path("models/day05_2_champion_pure")


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 05.2 short tuning runner."""
    parser = argparse.ArgumentParser(
        description="Orquesta Day 05.2: tuning corto del champion puro Day 05.1 sobre LIGHTGBM balanceado."
    )
    parser.add_argument(
        "--phase",
        choices=["smoke-imports", "all"],
        required=True,
        help="Fase a ejecutar: smoke-imports o all.",
    )
    parser.add_argument(
        "--baseline-metadata-path",
        type=Path,
        default=Path("models/public/baseline/metadata.json"),
        help="Metadata del baseline oficial para resolver cutoff y contrato.",
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("artifacts/public/metrics/final_baseline_vs_candidates.csv"),
        help="Registry oficial baseline vs candidatos.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports"),
        help="Directorio raíz de reportes.",
    )
    parser.add_argument(
        "--candidate-model-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_DIR,
        help="Directorio raíz de artefactos candidatos Day 05.2.",
    )
    parser.add_argument(
        "--champion-dir",
        type=Path,
        default=DEFAULT_CHAMPION_DIR,
        help="Directorio del champion puro promovido desde Day 05.2 si aplica.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Número de folds temporales expansivos a nivel evento para la CV interna.",
    )
    parser.add_argument(
        "--max-finalists",
        type=int,
        default=2,
        help="Número máximo de variantes tuned finalistas a evaluar en holdout y registrar.",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="No añadir finalistas al registry oficial. Útil para smoke checks.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id opcional para reproducibilidad externa.",
    )
    return parser.parse_args()


# SECTION: Path helpers
def resolve_project_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve all repo-root-relative paths used by the Day 05.2 runner."""
    project_root = Path(__file__).resolve().parents[3]
    return {
        "project_root": project_root,
        "baseline_metadata_path": (project_root / args.baseline_metadata_path).resolve(),
        "registry_csv": (project_root / args.registry_csv).resolve(),
        "reports_root": (project_root / args.reports_root).resolve(),
        "candidate_model_dir": (project_root / args.candidate_model_dir).resolve(),
        "champion_dir": (project_root / args.champion_dir).resolve(),
    }


# SECTION: Path helpers
def build_day052_output_paths(reports_root: Path, run_id: str) -> dict[str, Path]:
    """Build the Day 05.2 summary and search-log paths for one run id."""
    metrics_root = reports_root / "metrics"
    day052_root = metrics_root / "day05_2"
    run_date = run_id[:8]
    candidates_root = metrics_root / "candidates" / run_date
    return {
        "day052_root": day052_root,
        "candidates_root": candidates_root,
        "canonical_csv": day052_root / f"{run_id}_canonical_candidates.csv",
        "canonical_json": day052_root / f"{run_id}_canonical_candidates.json",
        "phase2_trials_csv": day052_root / f"{run_id}_phase2_trials.csv",
        "phase2_trials_json": day052_root / f"{run_id}_phase2_trials.json",
        "run_summary_json": day052_root / f"{run_id}_run_summary.json",
    }


# SECTION: Search grid
def get_day052_trial_params() -> list[dict[str, Any]]:
    """Return the fixed eight-trial LightGBM grid for Day 05.2, always with native class balancing."""
    return [
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "class_weight": "balanced",
            "random_state": 0,
        },
        {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "class_weight": "balanced",
            "random_state": 0,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "class_weight": "balanced",
            "random_state": 0,
        },
        {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "class_weight": "balanced",
            "random_state": 0,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": 0,
        },
        {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": 0,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.10,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 50,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "class_weight": "balanced",
            "random_state": 0,
        },
        {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 10,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": 0,
        },
    ]


# SECTION: Reference helpers
def load_reference_rows(registry_csv: Path) -> dict[str, dict[str, Any]]:
    """Load the baseline, Day 05.1 champion and secondary reference rows needed by Day 05.2."""
    baseline_row, lr_reference_rows = load_registry_reference_rows(registry_csv)
    rows = registry.read_registry_rows(registry_csv)
    required_variants = {TARGET_BASE_VARIANT, *SECONDARY_REFERENCE_VARIANTS}
    day051_rows = {
        row["model_variant"]: row
        for row in rows
        if row.get("model_variant") in required_variants and row.get("day_id") == "Day 05.1"
    }
    missing = sorted(required_variants - set(day051_rows))
    if missing:
        raise ValueError(
            "Faltan referencias Day 05.1 para abrir Day 05.2: "
            f"{missing}"
        )
    return {
        "baseline_row": baseline_row,
        "lr_reference_rows": lr_reference_rows,
        "day051_rows": day051_rows,
    }


# SECTION: Trial ranking
def sort_trial_rows(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Sort Day 05.2 trials by the fixed lexicographic CV ranking."""
    return trials_df.sort_values(
        ["cv_top2_hit_mean", "cv_bal_acc_mean", "cv_f1_pos_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


# SECTION: Metrics helpers
def compute_champion_deltas(
    metrics: dict[str, float | int],
    champion_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute holdout deltas against the current Day 05.1 pure champion."""
    return {
        "delta_top2_vs_day051_champion": float(metrics["top2_hit"]) - champion_metrics["top2_hit"],
        "delta_bal_acc_vs_day051_champion": float(metrics["balanced_accuracy"]) - champion_metrics["balanced_accuracy"],
        "delta_coverage_vs_day051_champion": float(metrics["coverage"]) - champion_metrics["coverage"],
    }


# SECTION: Metrics helpers
def compute_tuned_promotion_flags(
    metrics: dict[str, float | int],
    baseline_metrics: dict[str, float],
    champion_metrics: dict[str, float],
) -> dict[str, Any]:
    """Compute the Day 05.2 decision flags against both baseline and the Day 05.1 champion."""
    delta_top2_vs_baseline = float(metrics["top2_hit"]) - baseline_metrics["top2_hit"]
    delta_bal_acc_vs_baseline = float(metrics["balanced_accuracy"]) - baseline_metrics["balanced_accuracy"]
    delta_coverage_vs_baseline = float(metrics["coverage"]) - baseline_metrics["coverage"]
    gate_top2_ok = delta_top2_vs_baseline >= registry.GATE_MIN_DELTA_TOP2
    gate_bal_acc_ok = delta_bal_acc_vs_baseline >= registry.GATE_MIN_DELTA_BAL_ACC
    gate_coverage_ok = delta_coverage_vs_baseline >= registry.GATE_MIN_DELTA_COVERAGE
    gate_pass = gate_top2_ok and gate_bal_acc_ok and gate_coverage_ok

    champion_deltas = compute_champion_deltas(metrics, champion_metrics)
    better_on_one = (
        champion_deltas["delta_top2_vs_day051_champion"] >= 0.002
        or champion_deltas["delta_bal_acc_vs_day051_champion"] >= 0.002
    )
    other_metric_ok = (
        champion_deltas["delta_top2_vs_day051_champion"] >= -0.003
        and champion_deltas["delta_bal_acc_vs_day051_champion"] >= -0.003
    )
    coverage_ok_vs_champion = float(metrics["coverage"]) >= champion_metrics["coverage"] - 0.005
    promotable_vs_day051_champion = better_on_one and other_metric_ok and coverage_ok_vs_champion

    return {
        "delta_top2_vs_baseline": delta_top2_vs_baseline,
        "delta_bal_acc_vs_baseline": delta_bal_acc_vs_baseline,
        "delta_coverage_vs_baseline": delta_coverage_vs_baseline,
        "gate_top2_ok": gate_top2_ok,
        "gate_bal_acc_ok": gate_bal_acc_ok,
        "gate_coverage_ok": gate_coverage_ok,
        "gate_pass": gate_pass,
        **champion_deltas,
        "promotable_vs_day051_champion": promotable_vs_day051_champion,
    }


# SECTION: Artifact helpers
def serialize_params(params: dict[str, Any]) -> str:
    """Serialize one params dict to stable JSON for trial logs and metadata."""
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


# SECTION: Variant naming
def build_day052_variant(suffix: str) -> str:
    """Build the canonical Day 05.2 tuned variant name."""
    return f"V2_TRANSPORT_ONLY_LIGHTGBM_CLASS_WEIGHT_BALANCED_{suffix}"


# SECTION: Canonical row builders
def build_day052_canonical_row(
    *,
    run_id: str,
    cutoff_date: str,
    dataset_path: Path,
    metrics: dict[str, float | int],
    baseline_metrics: dict[str, float],
    lr_reference_metrics: dict[str, float],
    champion_metrics: dict[str, float],
    model_variant: str,
    trial_rank_cv: int,
    search_log_path: Path,
) -> dict[str, Any]:
    """Build one canonical Day 05.2 finalist row with baseline and champion deltas."""
    flags = compute_tuned_promotion_flags(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        champion_metrics=champion_metrics,
    )
    return {
        "run_id": run_id,
        "cutoff_date": cutoff_date,
        "dataset_alias": TARGET_DATASET_ALIAS,
        "dataset_path": str(dataset_path),
        "base_champion_variant": TARGET_BASE_VARIANT,
        "lr_equivalent_variant": "V2_TRANSPORT_ONLY_LR_smote_0.5_v1",
        "model_family": TARGET_MODEL_FAMILY,
        "model_variant": model_variant,
        "variant_stage": "phase2_short_tuning",
        "balance_tag": "CLASS_WEIGHT_BALANCED",
        "trial_rank_cv": int(trial_rank_cv),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "f1_pos": float(metrics["f1_pos"]),
        "top1_hit": float(metrics["top1_hit"]),
        "top2_hit": float(metrics["top2_hit"]),
        "coverage": float(metrics["coverage"]),
        "test_events": int(metrics["test_events"]),
        "search_log_path": str(search_log_path),
        "delta_top2_vs_lr_equivalente": float(metrics["top2_hit"]) - lr_reference_metrics["top2_hit"],
        "delta_bal_acc_vs_lr_equivalente": float(metrics["balanced_accuracy"]) - lr_reference_metrics["balanced_accuracy"],
        "delta_coverage_vs_lr_equivalente": float(metrics["coverage"]) - lr_reference_metrics["coverage"],
        **flags,
        "promotion_decision": "promote_day052_tuned_champion" if flags["gate_pass"] and flags["promotable_vs_day051_champion"] else "keep_day051_champion",
    }


# SECTION: Promotion helpers
def promote_day052_champion(
    *,
    champion_dir: Path,
    source_model_path: Path,
    source_metadata_path: Path,
    source_metrics_path: Path,
    promoted_variant: str,
    run_id: str,
) -> None:
    """Promote one Day 05.2 tuned finalist into a versioned pure-champion directory."""
    champion_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model_path, champion_dir / "model.pkl")
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "champion_role": "model_pure",
            "promotion_decision": "promote_day052_tuned_champion",
            "promoted_from_run_id": run_id,
            "promoted_from_variant": promoted_variant,
            "source_candidate_model_path": str(source_model_path),
            "source_candidate_metadata_path": str(source_metadata_path),
            "source_metrics_json_path": str(source_metrics_path),
            "deployment_status": "ready_for_rollout_decision",
            "deployment_decision_note": "Day 05.2 closed. Tuned pure champion selected and ready for an explicit serving rollout decision.",
        }
    )
    (champion_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


# SECTION: Registry helpers
def sync_day052_registry_decisions(
    *,
    registry_csv: Path,
    run_id: str,
    finalists_df: pd.DataFrame,
) -> None:
    """Overwrite Day 05.2 registry decisions with the actual keep/promote outcome vs the Day 05.1 champion."""
    registry_df = pd.read_csv(registry_csv)
    for finalist in finalists_df.itertuples(index=False):
        mask = (
            (registry_df["run_id"] == run_id)
            & (registry_df["day_id"] == DAY_ID)
            & (registry_df["model_variant"] == str(finalist.model_variant))
        )
        registry_df.loc[mask, "promotion_decision"] = str(finalist.promotion_decision)
    registry_df.to_csv(registry_csv, index=False)


# SECTION: Summary helpers
def build_run_summary(
    *,
    run_id: str,
    cutoff_date: str,
    trials_df: pd.DataFrame,
    finalists_df: pd.DataFrame,
    base_champion_metrics: dict[str, float],
    decision: str,
    current_model_champion_variant: str,
    serving_default_decision: str,
    can_open_brent: bool,
) -> dict[str, Any]:
    """Build the final Day 05.2 run summary after evaluating the short tuning finalists."""
    finalists = finalists_df.sort_values(
        ["promotable_vs_day051_champion", "gate_pass", "top2_hit", "balanced_accuracy", "f1_pos"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    best_variant = str(finalists.iloc[0]["model_variant"]) if not finalists.empty else TARGET_BASE_VARIANT
    best_payload = finalists.iloc[0].to_dict() if not finalists.empty else {}
    return {
        "run_id": run_id,
        "day_id": DAY_ID,
        "cutoff_date": cutoff_date,
        "base_champion_variant": TARGET_BASE_VARIANT,
        "base_champion_metrics": base_champion_metrics,
        "secondary_reference_variants": list(SECONDARY_REFERENCE_VARIANTS),
        "phase2_trials_total": int(len(trials_df)),
        "phase2_finalists_total": int(len(finalists_df)),
        "finalist_variants": finalists_df["model_variant"].astype(str).tolist(),
        "best_tuned_variant": best_variant,
        "best_tuned_metrics": best_payload,
        "close_decision": decision,
        "current_model_champion_variant": current_model_champion_variant,
        "serving_default_decision": serving_default_decision,
        "can_open_day05_5": bool(can_open_brent),
        "next_block": "Day 05.5" if can_open_brent else "Day 05.2 follow-up",
    }


# SECTION: Main orchestration
def run_all(args: argparse.Namespace) -> int:
    """Execute Day 05.2 end-to-end: bounded CV tuning plus 1-2 holdout finalists."""
    paths = resolve_project_paths(args)
    run_id = args.run_id.strip() or build_run_id()
    ts_utc = utc_now_iso()
    cutoff_date = resolve_cutoff_date(paths["baseline_metadata_path"])
    output_paths = build_day052_output_paths(paths["reports_root"], run_id)
    references = load_reference_rows(paths["registry_csv"])
    baseline_metrics = extract_reference_metrics(references["baseline_row"])
    day051_rows = references["day051_rows"]
    base_champion_row = day051_rows[TARGET_BASE_VARIANT]
    base_champion_metrics = extract_reference_metrics(base_champion_row)
    lr_reference_metrics = extract_reference_metrics(references["lr_reference_rows"]["V2_TRANSPORT_ONLY_LR_smote_0.5_v1"])

    project_root = paths["project_root"]
    dataset_catalog = get_day05_dataset_catalog(project_root=project_root)
    dataset_spec = dataset_catalog[TARGET_DATASET_ALIAS]
    dataset_df = pd.read_csv(dataset_spec["dataset_path"], keep_default_na=False)
    model_df = prepare_day05_model_frame(
        dataset_df=dataset_df,
        feature_cols_num=dataset_spec["feature_cols_num"],
        feature_cols_cat=dataset_spec["feature_cols_cat"],
        target_col=dataset_spec["target_col"],
    )
    train_df, test_df = split_day05_by_cutoff(model_df, cutoff_date=cutoff_date)

    trial_rows: list[dict[str, Any]] = []
    for trial_idx, params in enumerate(get_day052_trial_params(), start=1):
        cv_payload = evaluate_cv_trial(
            model_family=TARGET_MODEL_FAMILY,
            params=params,
            train_df=train_df,
            feature_cols_num=dataset_spec["feature_cols_num"],
            feature_cols_cat=dataset_spec["feature_cols_cat"],
            target_col=dataset_spec["target_col"],
            n_splits=args.n_splits,
        )
        trial_rows.append(
            {
                "run_id": run_id,
                "base_champion_variant": TARGET_BASE_VARIANT,
                "trial_idx": trial_idx,
                "params_json": serialize_params(params),
                "cv_top2_hit_mean": cv_payload["cv_top2_hit_mean"],
                "cv_bal_acc_mean": cv_payload["cv_bal_acc_mean"],
                "cv_f1_pos_mean": cv_payload["cv_f1_pos_mean"],
                "cv_coverage_mean": cv_payload["cv_coverage_mean"],
                "folds_json": json.dumps(cv_payload["folds"], ensure_ascii=False),
            }
        )

    trials_df = sort_trial_rows(pd.DataFrame(trial_rows))
    output_paths["phase2_trials_csv"].parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(output_paths["phase2_trials_csv"], index=False)
    output_paths["phase2_trials_json"].write_text(
        trials_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    finalist_rows: list[dict[str, Any]] = []
    finalists_to_evaluate = trials_df.head(max(1, min(args.max_finalists, 2))).copy()

    for finalist_rank, finalist in enumerate(finalists_to_evaluate.itertuples(index=False), start=1):
        params = json.loads(str(finalist.params_json))
        variant_suffix = "TUNED_v1" if finalist_rank == 1 else "TUNED_ALT_v1"
        model_variant = build_day052_variant(variant_suffix)
        estimator = build_base_model(
            model_family=TARGET_MODEL_FAMILY,
            params=params,
            cat_feature_names=None,
        )
        holdout_result = fit_and_evaluate_model(
            model_family=TARGET_MODEL_FAMILY,
            estimator=estimator,
            train_df=train_df,
            test_df=test_df,
            feature_cols_num=dataset_spec["feature_cols_num"],
            feature_cols_cat=dataset_spec["feature_cols_cat"],
            target_col=dataset_spec["target_col"],
        )
        metrics = {
            "accuracy": holdout_result["row_metrics"]["accuracy"],
            "balanced_accuracy": holdout_result["row_metrics"]["balanced_accuracy"],
            "f1_pos": holdout_result["row_metrics"]["f1_pos"],
            "top1_hit": holdout_result["event_metrics"]["top1_hit"],
            "top2_hit": holdout_result["event_metrics"]["top2_hit"],
            "coverage": holdout_result["event_metrics"]["coverage"],
            "test_events": holdout_result["event_metrics"]["test_events"],
        }
        selection_rule = (
            "day05_2_short_tuning(LIGHTGBM on V2_TRANSPORT_ONLY, class_weight=balanced) -> "
            "cv_rank(top2_hit, balanced_accuracy, f1_pos) -> "
            "compare_vs_day05_1_champion(delta>=0.002 on one key metric, other>=-0.003, coverage>=champion-0.005)"
        )
        model_path, metadata_path = save_candidate_artifacts(
            candidate_model_dir=paths["candidate_model_dir"],
            run_id=run_id,
            model_variant=model_variant,
            model=holdout_result["model"],
            metrics=metrics,
            cutoff_date=cutoff_date,
            dataset_name=dataset_spec["dataset_path"].name,
            feature_columns=holdout_result["feature_columns"],
            selection_rule=selection_rule,
            extra_metadata={
                "day_id": DAY_ID,
                "phase": "phase2_short_tuning",
                "dataset_alias": TARGET_DATASET_ALIAS,
                "model_family": TARGET_MODEL_FAMILY,
                "balance_tag": "CLASS_WEIGHT_BALANCED",
                "balance_strategy": "class_weight=balanced",
                "base_champion_variant": TARGET_BASE_VARIANT,
                "trial_rank_cv": int(finalist_rank),
                "trial_params": params,
                "search_log_path": str(output_paths["phase2_trials_csv"]),
            },
        )
        metrics_payload = build_pure_metrics_payload(
            run_id=run_id,
            ts_utc=ts_utc,
            day_id=DAY_ID,
            cutoff_date=cutoff_date,
            model_variant=model_variant,
            dataset_alias=TARGET_DATASET_ALIAS,
            dataset_path=dataset_spec["dataset_path"],
            model_family=TARGET_MODEL_FAMILY,
            variant_stage="phase2_short_tuning",
            metrics=metrics,
            model_path=model_path,
            metadata_path=metadata_path,
            search_log_path=output_paths["phase2_trials_csv"],
        )
        metrics_json_path = write_metrics_payload(
            payload=metrics_payload,
            candidates_root=output_paths["candidates_root"],
            model_variant=model_variant,
        )
        row = build_day052_canonical_row(
            run_id=run_id,
            cutoff_date=cutoff_date,
            dataset_path=dataset_spec["dataset_path"],
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            lr_reference_metrics=lr_reference_metrics,
            champion_metrics=base_champion_metrics,
            model_variant=model_variant,
            trial_rank_cv=int(finalist_rank),
            search_log_path=output_paths["phase2_trials_csv"],
        )
        row["model_path"] = str(model_path)
        row["metadata_path"] = str(metadata_path)
        row["metrics_json_path"] = str(metrics_json_path)
        row["params_json"] = serialize_params(params)
        finalist_rows.append(row)

    finalists_df = pd.DataFrame(finalist_rows).sort_values(
        ["promotable_vs_day051_champion", "gate_pass", "top2_hit", "balanced_accuracy", "f1_pos"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    write_dataframe_outputs(output_paths["canonical_csv"], output_paths["canonical_json"], finalists_df)

    decision = "keep_day051_champion"
    current_model_champion_variant = TARGET_BASE_VARIANT
    serving_default_decision = "keep_frozen_pending_explicit_rollout_decision"

    promotable_finalists = finalists_df[
        finalists_df["promotion_decision"] == "promote_day052_tuned_champion"
    ].copy()
    if not promotable_finalists.empty:
        best_promotable = promotable_finalists.iloc[0]
        decision = "promote_day052_tuned_champion"
        current_model_champion_variant = str(best_promotable["model_variant"])
        promote_day052_champion(
            champion_dir=paths["champion_dir"],
            source_model_path=Path(str(best_promotable["model_path"])),
            source_metadata_path=Path(str(best_promotable["metadata_path"])),
            source_metrics_path=Path(str(best_promotable["metrics_json_path"])),
            promoted_variant=current_model_champion_variant,
            run_id=run_id,
        )
        serving_default_decision = "ready_for_model_default_rollout_decision"

    if not args.skip_registry:
        for finalist in finalists_df.itertuples(index=False):
            append_registry_candidate(
                registry_csv=paths["registry_csv"],
                run_id=run_id,
                day_id=DAY_ID,
                model_variant=str(finalist.model_variant),
                metadata_path=Path(str(finalist.metadata_path)),
                metrics_json_path=Path(str(finalist.metrics_json_path)),
                dataset_csv=dataset_spec["dataset_path"],
                model_path=Path(str(finalist.model_path)),
            )
        sync_day052_registry_decisions(
            registry_csv=paths["registry_csv"],
            run_id=run_id,
            finalists_df=finalists_df,
        )

    run_summary = build_run_summary(
        run_id=run_id,
        cutoff_date=cutoff_date,
        trials_df=trials_df,
        finalists_df=finalists_df,
        base_champion_metrics=base_champion_metrics,
        decision=decision,
        current_model_champion_variant=current_model_champion_variant,
        serving_default_decision=serving_default_decision,
        can_open_brent=True,
    )
    write_json(output_paths["run_summary_json"], run_summary)
    return 0


# SECTION: Main
def main() -> None:
    """Dispatch the requested Day 05.2 runner mode and exit with explicit status code."""
    args = parse_args()
    if args.phase == "smoke-imports":
        raise SystemExit(smoke_imports())
    if args.phase == "all":
        raise SystemExit(run_all(args))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
