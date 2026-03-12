#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.ml.metrics import registry
    from src.ml.scripts.day05_train_tabular_candidates import (
        append_registry_candidate,
        build_base_model,
        build_canonical_row,
        build_pure_metrics_payload,
        extract_reference_metrics,
        fit_and_evaluate_model,
        load_registry_reference_rows,
        resolve_cutoff_date,
        run_single_policy_check,
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
        build_canonical_row,
        build_pure_metrics_payload,
        extract_reference_metrics,
        fit_and_evaluate_model,
        load_registry_reference_rows,
        resolve_cutoff_date,
        run_single_policy_check,
        save_candidate_artifacts,
        smoke_imports,
        write_dataframe_outputs,
        write_metrics_payload,
    )
    from src.ml.shared.day05_tabular import get_day05_dataset_catalog, prepare_day05_model_frame, split_day05_by_cutoff
    from src.ml.shared.helpers import build_run_id, utc_now_iso, write_json

DATASET_ALIASES = (
    "V2",
    "V2_TRANSPORT_ONLY",
    "V2_TRANSPORT_CARRY30D_ONLY",
)
MODEL_FAMILIES = (
    "LIGHTGBM",
    "XGBOOST",
)
PREVIOUS_MODEL_CHAMPION_VARIANT = "LR_smote_0.5"
PREVIOUS_OPERATIONAL_POLICY_VARIANT = "V2_TRANSPORT_ONLY_LR_smote_0.5_WITH_DETERMINISTIC_LAYER_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009_v1"


# SECTION: CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Day 05.1 balanced tree baseline runner."""
    parser = argparse.ArgumentParser(
        description="Orquesta Day 05.1: balanced tree baselines sobre bases post-fix y contrato fijo Day 05."
    )
    parser.add_argument(
        "--phase",
        choices=["smoke-imports", "all"],
        required=True,
        help="Fase a ejecutar: smoke-imports o all.",
    )
    parser.add_argument(
        "--dataset-aliases",
        nargs="*",
        default=[],
        help="Aliases Day 05.1 a ejecutar. Si vacío, se usan V2/V2_TRANSPORT_ONLY/V2_TRANSPORT_CARRY30D_ONLY.",
    )
    parser.add_argument(
        "--model-families",
        nargs="*",
        default=[],
        help="Familias de modelo a ejecutar. Si vacío, se usan LIGHTGBM/XGBOOST.",
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
        default=Path("models/candidates/day05_1_balanced_trees"),
        help="Directorio raíz de artefactos Day 05.1.",
    )
    parser.add_argument(
        "--rules-csv",
        type=Path,
        default=Path("config/business_blocklist_rules.csv"),
        help="CSV de reglas de negocio para una eventual comparativa secundaria con policy.",
    )
    parser.add_argument(
        "--albaran-policy",
        type=str,
        default="PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009",
        choices=["PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009"],
        help="Policy Day 03 para la comparativa secundaria.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k operativo para scoring Day 05.1.",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="No añadir filas al registry oficial. Útil para smoke checks.",
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
    """Resolve all repo-root-relative paths used by the Day 05.1 runner."""
    project_root = Path(__file__).resolve().parents[3]
    return {
        "project_root": project_root,
        "baseline_metadata_path": (project_root / args.baseline_metadata_path).resolve(),
        "registry_csv": (project_root / args.registry_csv).resolve(),
        "reports_root": (project_root / args.reports_root).resolve(),
        "candidate_model_dir": (project_root / args.candidate_model_dir).resolve(),
        "rules_csv": (project_root / args.rules_csv).resolve(),
    }


# SECTION: Path helpers
def build_day051_output_paths(reports_root: Path, run_id: str) -> dict[str, Path]:
    """Build Day 05.1 output paths for canonical summaries and optional policy outputs."""
    run_date = run_id[:8]
    metrics_root = reports_root / "metrics"
    day051_root = metrics_root / "day05_1"
    candidates_root = metrics_root / "candidates" / run_date
    return {
        "day05_root": day051_root,
        "candidates_root": candidates_root,
        "canonical_csv": day051_root / f"{run_id}_canonical_candidates.csv",
        "canonical_json": day051_root / f"{run_id}_canonical_candidates.json",
        "selection_json": day051_root / f"{run_id}_selection_decisions.json",
        "run_summary_json": day051_root / f"{run_id}_run_summary.json",
        "policy_summary_json": day051_root / f"{run_id}_policy_summary.json",
    }


# SECTION: Reference helpers
def build_unbalanced_variant(dataset_alias: str, model_family: str) -> str:
    """Build the Day 05 unbalanced reference variant for one dataset and tree family."""
    return f"{dataset_alias}_{model_family}_v1"


# SECTION: Reference helpers
def build_balanced_variant(dataset_alias: str, model_family: str) -> str:
    """Build the canonical Day 05.1 balanced-native variant name."""
    if model_family == "LIGHTGBM":
        return f"{dataset_alias}_LIGHTGBM_CLASS_WEIGHT_BALANCED_v1"
    if model_family == "XGBOOST":
        return f"{dataset_alias}_XGBOOST_SCALE_POS_WEIGHT_v1"
    raise ValueError(f"Familia no soportada en Day 05.1: {model_family}")


# SECTION: Reference helpers
def load_day05_unbalanced_reference_rows(registry_csv: Path) -> dict[str, dict[str, Any]]:
    """Load the Day 05 unbalanced tree reference rows for the six comparable pairs."""
    rows = registry.read_registry_rows(registry_csv)
    expected_variants = {
        build_unbalanced_variant(dataset_alias, model_family)
        for dataset_alias in DATASET_ALIASES
        for model_family in MODEL_FAMILIES
    }
    payload = {
        row["model_variant"]: row
        for row in rows
        if row.get("model_variant") in expected_variants and row.get("model_role") == "candidate"
    }
    missing = sorted(expected_variants - set(payload))
    if missing:
        raise ValueError(
            "Faltan referencias Day 05 no balanceadas en el registry oficial para abrir Day 05.1: "
            f"{missing}"
        )
    return payload


# SECTION: Params helpers
def build_day051_base_params(model_family: str, class_ratio: float) -> dict[str, Any]:
    """Build the fixed Day 05.1 base params with native balance enabled from the start."""
    family = model_family.upper()
    if family == "LIGHTGBM":
        return {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 0,
            "class_weight": "balanced",
        }
    if family == "XGBOOST":
        return {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "random_state": 0,
            "scale_pos_weight": class_ratio,
        }
    raise ValueError(f"Familia no soportada en Day 05.1: {model_family}")


# SECTION: Metrics helpers
def compute_unbalanced_deltas(
    metrics: dict[str, float | int],
    unbalanced_reference_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute deltas against the Day 05 unbalanced reference of the same dataset-family pair."""
    return {
        "delta_top2_vs_day05_unbalanced": float(metrics["top2_hit"]) - unbalanced_reference_metrics["top2_hit"],
        "delta_bal_acc_vs_day05_unbalanced": float(metrics["balanced_accuracy"])
        - unbalanced_reference_metrics["balanced_accuracy"],
        "delta_coverage_vs_day05_unbalanced": float(metrics["coverage"]) - unbalanced_reference_metrics["coverage"],
    }


# SECTION: Metrics helpers
def is_almost_gate(
    *,
    defendible: bool,
    canonical_row: dict[str, Any],
) -> bool:
    """Return whether one Day 05.1 candidate is close enough to justify follow-up without passing the gate."""
    if defendible:
        return False
    return bool(
        canonical_row["delta_top2_vs_baseline"] >= -0.002
        and canonical_row["delta_bal_acc_vs_baseline"] >= -0.005
        and canonical_row["coverage"] >= 0.995
        and canonical_row["delta_bal_acc_vs_day05_unbalanced"] >= 0.030
        and canonical_row["delta_top2_vs_day05_unbalanced"] >= -0.003
    )


# SECTION: Metrics helpers
def build_day051_canonical_row(
    *,
    run_id: str,
    cutoff_date: str,
    dataset_alias: str,
    dataset_path: Path,
    lr_equivalent_variant: str,
    day05_unbalanced_variant: str,
    model_family: str,
    model_variant: str,
    metrics: dict[str, float | int],
    baseline_metrics: dict[str, float],
    lr_reference_metrics: dict[str, float],
    day05_unbalanced_metrics: dict[str, float],
) -> dict[str, Any]:
    """Build one Day 05.1 canonical row enriched with Day 05 unbalanced deltas and close-gate status."""
    row = build_canonical_row(
        run_id=run_id,
        cutoff_date=cutoff_date,
        dataset_alias=dataset_alias,
        dataset_path=dataset_path,
        lr_equivalent_variant=lr_equivalent_variant,
        model_family=model_family,
        model_variant=model_variant,
        variant_stage="phase1_balanced_native",
        balance_tag="NATIVE_BALANCED",
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        lr_reference_metrics=lr_reference_metrics,
        search_log_path=None,
    )
    row["day05_unbalanced_variant"] = day05_unbalanced_variant
    row.update(compute_unbalanced_deltas(metrics, day05_unbalanced_metrics))
    row["defendible"] = bool(row["gate_pass"])
    row["almost_gate"] = bool(is_almost_gate(defendible=row["defendible"], canonical_row=row))
    row["selected_for_phase2"] = False
    row["selected_for_phase3"] = False
    row["eligible_policy_check"] = bool(row["defendible"])
    if row["defendible"]:
        row["followup_decision"] = "policy_then_decide_tuning"
    elif row["almost_gate"]:
        row["followup_decision"] = "consider_future_tuning_or_resampling"
    else:
        row["followup_decision"] = "close_tabular"
    return row


# SECTION: Selection helpers
def choose_best_balanced_variant(canonical_df: pd.DataFrame) -> str:
    """Choose the best Day 05.1 balanced-native candidate for summary purposes."""
    ordered = canonical_df.sort_values(
        [
            "defendible",
            "almost_gate",
            "delta_bal_acc_vs_baseline",
            "delta_top2_vs_baseline",
            "balanced_accuracy",
            "top2_hit",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    return str(ordered.iloc[0]["model_variant"])


# SECTION: Summary helpers
def build_selection_payload(canonical_df: pd.DataFrame) -> dict[str, Any]:
    """Build the Day 05.1 selection summary used by notebook 17 and docs."""
    defendible_variants = canonical_df.loc[canonical_df["defendible"], "model_variant"].astype(str).tolist()
    almost_gate_variants = canonical_df.loc[canonical_df["almost_gate"], "model_variant"].astype(str).tolist()
    if defendible_variants:
        pre_policy_close_decision = "open_single_policy_check_then_decide_tuning"
    elif almost_gate_variants:
        pre_policy_close_decision = "decide_followup_iteration_without_policy"
    else:
        pre_policy_close_decision = "close_tabular_and_move_to_day05_5"
    return {
        "defendible_variants": defendible_variants,
        "almost_gate_variants": almost_gate_variants,
        "best_balanced_variant": choose_best_balanced_variant(canonical_df),
        "pre_policy_close_decision": pre_policy_close_decision,
    }


# SECTION: Summary helpers
def resolve_policy_promotion_decision(
    *,
    registry_csv: Path,
    run_id: str,
    policy_variant: str,
) -> str | None:
    """Read back the official registry decision for the one optional Day 05.1 policy variant."""
    for row in registry.read_registry_rows(registry_csv):
        if row.get("run_id") == run_id and row.get("model_variant") == policy_variant:
            decision = row.get("promotion_decision")
            return str(decision) if decision is not None else None
    return None


# SECTION: Summary helpers
def resolve_final_close_decision(
    *,
    selection_payload: dict[str, Any],
    policy_summary: dict[str, Any],
) -> str:
    """Resolve the explicit next step for Day 05.1 after the optional policy check finishes."""
    pre_policy_decision = str(selection_payload["pre_policy_close_decision"])
    if selection_payload["defendible_variants"]:
        if policy_summary.get("executed") and policy_summary.get("promotion_decision") == "promote":
            return "promote_pure_model_and_policy_review_serving"
        return "promote_pure_model_open_short_tuning"
    if pre_policy_decision == "decide_followup_iteration_without_policy":
        return "keep_previous_model_champion_consider_followup_iteration"
    return "close_tabular_and_move_to_day05_5"


# SECTION: Summary helpers
def resolve_pure_model_decision(selection_payload: dict[str, Any]) -> tuple[str, str]:
    """Resolve the pure-model governance decision independently from any policy outcome."""
    if selection_payload["defendible_variants"]:
        return "promote_model_champion", str(selection_payload["best_balanced_variant"])
    return "keep_previous_model_champion", PREVIOUS_MODEL_CHAMPION_VARIANT


# SECTION: Summary helpers
def resolve_policy_decision(policy_summary: dict[str, Any]) -> tuple[str, str]:
    """Resolve the post-inference policy decision while keeping it separate from model promotion."""
    if not policy_summary.get("executed"):
        return "policy_not_run", PREVIOUS_OPERATIONAL_POLICY_VARIANT
    if policy_summary.get("promotion_decision") == "promote":
        return "promote_new_policy", str(policy_summary["policy_variant"])
    return "reject_new_policy_keep_previous_operational_policy", PREVIOUS_OPERATIONAL_POLICY_VARIANT


# SECTION: Summary helpers
def build_governance_payload(
    *,
    selection_payload: dict[str, Any],
    policy_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build the final Day 05.1 governance payload after separating model and policy decisions."""
    pure_model_decision, current_model_champion_variant = resolve_pure_model_decision(selection_payload)
    policy_decision, current_operational_policy_variant = resolve_policy_decision(policy_summary)
    rejected_policy_variant = None
    if policy_decision == "reject_new_policy_keep_previous_operational_policy":
        rejected_policy_variant = str(policy_summary.get("policy_variant", ""))
    return {
        **selection_payload,
        "close_decision": resolve_final_close_decision(
            selection_payload=selection_payload,
            policy_summary=policy_summary,
        ),
        "final_close_decision": resolve_final_close_decision(
            selection_payload=selection_payload,
            policy_summary=policy_summary,
        ),
        "pure_model_decision": pure_model_decision,
        "current_model_champion_variant": current_model_champion_variant,
        "policy_decision": policy_decision,
        "current_operational_policy_variant": current_operational_policy_variant,
        "rejected_policy_variant": rejected_policy_variant,
    }


# SECTION: Summary helpers
def build_run_summary(
    *,
    run_id: str,
    cutoff_date: str,
    canonical_df: pd.DataFrame,
    governance_payload: dict[str, Any],
    policy_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build the final Day 05.1 run summary."""
    return {
        "run_id": run_id,
        "day_id": "Day 05.1",
        "cutoff_date": cutoff_date,
        "canonical_variants_total": int(len(canonical_df)),
        "datasets": canonical_df["dataset_alias"].drop_duplicates().astype(str).tolist(),
        "model_families": canonical_df["model_family"].drop_duplicates().astype(str).tolist(),
        "defendible_variants": governance_payload["defendible_variants"],
        "almost_gate_variants": governance_payload["almost_gate_variants"],
        "best_balanced_variant": governance_payload["best_balanced_variant"],
        "pre_policy_close_decision": governance_payload["pre_policy_close_decision"],
        "close_decision": governance_payload["close_decision"],
        "final_close_decision": governance_payload["final_close_decision"],
        "pure_model_decision": governance_payload["pure_model_decision"],
        "current_model_champion_variant": governance_payload["current_model_champion_variant"],
        "policy_decision": governance_payload["policy_decision"],
        "current_operational_policy_variant": governance_payload["current_operational_policy_variant"],
        "rejected_policy_variant": governance_payload["rejected_policy_variant"],
        "policy_summary": policy_summary,
    }


# SECTION: Execution helpers
def run_balanced_baselines(
    *,
    run_id: str,
    ts_utc: str,
    cutoff_date: str,
    args: argparse.Namespace,
    paths: dict[str, Path],
    output_paths: dict[str, Path],
    baseline_metrics: dict[str, float],
    lr_reference_rows: dict[str, dict[str, Any]],
    day05_unbalanced_rows: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Execute the six Day 05.1 balanced-native canonical runs and persist artifacts."""
    dataset_catalog = get_day05_dataset_catalog(project_root=paths["project_root"])
    dataset_aliases = [alias.upper() for alias in (args.dataset_aliases or DATASET_ALIASES)]
    model_families = [family.upper() for family in (args.model_families or MODEL_FAMILIES)]
    canonical_rows: list[dict[str, Any]] = []

    for dataset_alias in dataset_aliases:
        if dataset_alias not in dataset_catalog:
            raise ValueError(f"Dataset alias no soportado en Day 05.1: {dataset_alias}")
        dataset_spec = dataset_catalog[dataset_alias]
        dataset_df = pd.read_csv(dataset_spec["dataset_path"], keep_default_na=False)
        model_df = prepare_day05_model_frame(
            dataset_df=dataset_df,
            feature_cols_num=dataset_spec["feature_cols_num"],
            feature_cols_cat=dataset_spec["feature_cols_cat"],
            target_col=dataset_spec["target_col"],
        )
        train_df, test_df = split_day05_by_cutoff(model_df, cutoff_date=cutoff_date)
        y_train = train_df[dataset_spec["target_col"]].astype(int)
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        class_ratio = float(neg / pos) if pos else 1.0
        lr_reference_metrics = extract_reference_metrics(lr_reference_rows[dataset_spec["lr_equivalent_variant"]])

        for model_family in model_families:
            day05_unbalanced_variant = build_unbalanced_variant(dataset_alias, model_family)
            day05_unbalanced_metrics = extract_reference_metrics(day05_unbalanced_rows[day05_unbalanced_variant])
            model_variant = build_balanced_variant(dataset_alias, model_family)
            params = build_day051_base_params(model_family=model_family, class_ratio=class_ratio)
            estimator = build_base_model(
                model_family=model_family,
                params=params,
                cat_feature_names=dataset_spec["feature_cols_cat"] if model_family == "CATBOOST" else None,
            )
            result = fit_and_evaluate_model(
                model_family=model_family,
                estimator=estimator,
                train_df=train_df,
                test_df=test_df,
                feature_cols_num=dataset_spec["feature_cols_num"],
                feature_cols_cat=dataset_spec["feature_cols_cat"],
                target_col=dataset_spec["target_col"],
            )
            metrics = {
                "accuracy": result["row_metrics"]["accuracy"],
                "balanced_accuracy": result["row_metrics"]["balanced_accuracy"],
                "f1_pos": result["row_metrics"]["f1_pos"],
                "top1_hit": result["event_metrics"]["top1_hit"],
                "top2_hit": result["event_metrics"]["top2_hit"],
                "coverage": result["event_metrics"]["coverage"],
                "test_events": result["event_metrics"]["test_events"],
            }
            balance_strategy = "class_weight=balanced" if model_family == "LIGHTGBM" else f"scale_pos_weight={class_ratio:.6f}"
            selection_rule = (
                f"day05_1_balanced_native_base({model_family} on {dataset_alias}, {balance_strategy}) -> "
                "registry_gate(top2>=baseline+0.01 & bal_acc>=baseline+0.01 & coverage>=baseline-0.005)"
            )
            model_path, metadata_path = save_candidate_artifacts(
                candidate_model_dir=paths["candidate_model_dir"],
                run_id=run_id,
                model_variant=model_variant,
                model=result["model"],
                metrics=metrics,
                cutoff_date=cutoff_date,
                dataset_name=dataset_spec["dataset_path"].name,
                feature_columns=result["feature_columns"],
                selection_rule=selection_rule,
                extra_metadata={
                    "day_id": "Day 05.1",
                    "phase": "phase1_balanced_native",
                    "dataset_alias": dataset_alias,
                    "model_family": model_family,
                    "day05_unbalanced_variant": day05_unbalanced_variant,
                    "balance_tag": "NATIVE_BALANCED",
                    "balance_strategy": balance_strategy,
                },
            )
            metrics_payload = build_pure_metrics_payload(
                run_id=run_id,
                ts_utc=ts_utc,
                day_id="Day 05.1",
                cutoff_date=cutoff_date,
                model_variant=model_variant,
                dataset_alias=dataset_alias,
                dataset_path=dataset_spec["dataset_path"],
                model_family=model_family,
                variant_stage="phase1_balanced_native",
                metrics=metrics,
                model_path=model_path,
                metadata_path=metadata_path,
            )
            metrics_json_path = write_metrics_payload(
                payload=metrics_payload,
                candidates_root=output_paths["candidates_root"],
                model_variant=model_variant,
            )
            if not args.skip_registry:
                append_registry_candidate(
                    registry_csv=paths["registry_csv"],
                    run_id=run_id,
                    day_id="Day 05.1",
                    model_variant=model_variant,
                    metadata_path=metadata_path,
                    metrics_json_path=metrics_json_path,
                    dataset_csv=dataset_spec["dataset_path"],
                    model_path=model_path,
                )
            canonical_rows.append(
                build_day051_canonical_row(
                    run_id=run_id,
                    cutoff_date=cutoff_date,
                    dataset_alias=dataset_alias,
                    dataset_path=dataset_spec["dataset_path"],
                    lr_equivalent_variant=dataset_spec["lr_equivalent_variant"],
                    day05_unbalanced_variant=day05_unbalanced_variant,
                    model_family=model_family,
                    model_variant=model_variant,
                    metrics=metrics,
                    baseline_metrics=baseline_metrics,
                    lr_reference_metrics=lr_reference_metrics,
                    day05_unbalanced_metrics=day05_unbalanced_metrics,
                )
            )

    canonical_df = pd.DataFrame(canonical_rows).sort_values(
        [
            "defendible",
            "almost_gate",
            "delta_bal_acc_vs_baseline",
            "delta_top2_vs_baseline",
            "balanced_accuracy",
            "top2_hit",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    return canonical_df


# SECTION: Main orchestration
def run_all(args: argparse.Namespace) -> int:
    """Execute Day 05.1 end-to-end: six balanced-native runs plus optional policy if defendible."""
    paths = resolve_project_paths(args)
    run_id = args.run_id.strip() or build_run_id()
    ts_utc = utc_now_iso()
    cutoff_date = resolve_cutoff_date(paths["baseline_metadata_path"])
    output_paths = build_day051_output_paths(paths["reports_root"], run_id)
    baseline_row, lr_reference_rows = load_registry_reference_rows(paths["registry_csv"])
    baseline_metrics = extract_reference_metrics(baseline_row)
    day05_unbalanced_rows = load_day05_unbalanced_reference_rows(paths["registry_csv"])

    canonical_df = run_balanced_baselines(
        run_id=run_id,
        ts_utc=ts_utc,
        cutoff_date=cutoff_date,
        args=args,
        paths=paths,
        output_paths=output_paths,
        baseline_metrics=baseline_metrics,
        lr_reference_rows=lr_reference_rows,
        day05_unbalanced_rows=day05_unbalanced_rows,
    )
    selection_payload = build_selection_payload(canonical_df)
    write_dataframe_outputs(output_paths["canonical_csv"], output_paths["canonical_json"], canonical_df)

    policy_summary: dict[str, Any]
    if selection_payload["defendible_variants"]:
        policy_summary = run_single_policy_check(
            run_id=run_id,
            ts_utc=ts_utc,
            day_id="Day 05.1",
            cutoff_date=cutoff_date,
            canonical_df=canonical_df,
            args=args,
            paths=paths,
            output_paths=output_paths,
        )
        policy_variant = policy_summary.get("policy_variant")
        if isinstance(policy_variant, str) and policy_variant:
            promotion_decision = resolve_policy_promotion_decision(
                registry_csv=paths["registry_csv"],
                run_id=run_id,
                policy_variant=policy_variant,
            )
            if promotion_decision is not None:
                policy_summary["promotion_decision"] = promotion_decision
                write_json(output_paths["policy_summary_json"], policy_summary)
    else:
        policy_summary = {
            "run_id": run_id,
            "executed": False,
            "reason": "no_defendible_variant",
            "best_final_pure_variant": selection_payload["best_balanced_variant"],
        }
        write_json(output_paths["policy_summary_json"], policy_summary)

    governance_payload = build_governance_payload(
        selection_payload=selection_payload,
        policy_summary=policy_summary,
    )
    write_json(output_paths["selection_json"], {"run_id": run_id, **governance_payload})
    write_json(
        output_paths["run_summary_json"],
        build_run_summary(
            run_id=run_id,
            cutoff_date=cutoff_date,
            canonical_df=canonical_df,
            governance_payload=governance_payload,
            policy_summary=policy_summary,
        ),
    )
    return 0


# SECTION: Main
def main() -> None:
    """Dispatch the requested Day 05.1 runner mode and exit with explicit status code."""
    args = parse_args()
    if args.phase == "smoke-imports":
        raise SystemExit(smoke_imports())
    if args.phase == "all":
        raise SystemExit(run_all(args))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
