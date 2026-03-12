from __future__ import annotations

from typing import Any

import pandas as pd


DAY054_MATERIAL_IMPROVEMENT = 0.005


# SECTION: Policy metrics
def compute_policy_trial_metrics(
    policy_event_df: pd.DataFrame,
    policy_resumen_albaran_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute aggregate help/harm metrics for one Day 05.4 policy trial."""
    total_events = int(len(policy_event_df))
    coverage = float(policy_event_df["decision_final"].astype(str).str.strip().ne("").mean()) if total_events else 0.0
    selected_source_counts = (
        policy_event_df["decision_source"].astype(str).value_counts().to_dict()
        if total_events
        else {}
    )
    overrides_count = int(
        (
            policy_event_df["decision_final"].astype(str).str.strip()
            != policy_event_df["decision_pre_policy"].astype(str).str.strip()
        ).sum()
    )
    champion_before = policy_event_df["pure_champion_top1_hit"].astype(int)
    selected_after = policy_event_df["selected_top1_hit"].astype(int)
    overrides_improved = int(((champion_before == 0) & (selected_after == 1)).sum())
    overrides_harmed = int(((champion_before == 1) & (selected_after == 0)).sum())
    overrides_neutral = int((champion_before == selected_after).sum())

    coherence_before = pd.to_numeric(policy_resumen_albaran_df.get("coherence_before"), errors="coerce")
    coherence_after = pd.to_numeric(policy_resumen_albaran_df.get("coherence_after"), errors="coerce")
    valid_coherence = coherence_before.notna() & coherence_after.notna()
    coherence_before_value = float(coherence_before[valid_coherence].mean()) if valid_coherence.any() else 0.0
    coherence_after_value = float(coherence_after[valid_coherence].mean()) if valid_coherence.any() else 0.0

    return {
        "events": total_events,
        "top1_hit": float(policy_event_df["selected_top1_hit"].mean()) if total_events else 0.0,
        "top2_hit": float(policy_event_df["selected_top2_hit"].mean()) if total_events else 0.0,
        "coverage": coverage,
        "coherence_before": coherence_before_value,
        "coherence_after": coherence_after_value,
        "coherence_delta": float(coherence_after_value - coherence_before_value),
        "overrides_count": overrides_count,
        "overrides_improved": overrides_improved,
        "overrides_harmed": overrides_harmed,
        "overrides_neutral": overrides_neutral,
        "selected_source_counts": selected_source_counts,
        "selected_share": float(policy_event_df["policy_applied_event"].astype(int).mean()) if total_events else 0.0,
        "selected_events": int(policy_event_df["policy_applied_event"].astype(int).sum()),
        "delta_top1_vs_champion": float(
            policy_event_df["selected_top1_hit"].mean() - policy_event_df["pure_champion_top1_hit"].mean()
        ) if total_events else 0.0,
        "delta_top2_vs_champion": float(
            policy_event_df["selected_top2_hit"].mean() - policy_event_df["pure_champion_top2_hit"].mean()
        ) if total_events else 0.0,
        "delta_top1_vs_operational": float(
            policy_event_df["selected_top1_hit"].mean()
            - policy_event_df["operational_policy_reference_top1_hit"].mean()
        ) if total_events else 0.0,
        "delta_top2_vs_operational": float(
            policy_event_df["selected_top2_hit"].mean()
            - policy_event_df["operational_policy_reference_top2_hit"].mean()
        ) if total_events else 0.0,
        "delta_top1_vs_baseline": float(
            policy_event_df["selected_top1_hit"].mean() - policy_event_df["baseline_top1_hit"].mean()
        ) if total_events else 0.0,
        "delta_top2_vs_baseline": float(
            policy_event_df["selected_top2_hit"].mean() - policy_event_df["baseline_top2_hit"].mean()
        ) if total_events else 0.0,
    }


# SECTION: Policy metrics
def compute_help_harm_breakdown(policy_event_df: pd.DataFrame, comparator_prefix: str) -> dict[str, int]:
    """Compute event-level help/harm counts versus one named comparator source."""
    comparator_top1 = policy_event_df[f"{comparator_prefix}_top1_hit"].astype(int)
    comparator_top2 = policy_event_df[f"{comparator_prefix}_top2_hit"].astype(int)
    selected_top1 = policy_event_df["selected_top1_hit"].astype(int)
    selected_top2 = policy_event_df["selected_top2_hit"].astype(int)
    return {
        f"improves_top1_vs_{comparator_prefix}": int(((comparator_top1 == 0) & (selected_top1 == 1)).sum()),
        f"harms_top1_vs_{comparator_prefix}": int(((comparator_top1 == 1) & (selected_top1 == 0)).sum()),
        f"same_top1_vs_{comparator_prefix}": int((comparator_top1 == selected_top1).sum()),
        f"improves_top2_vs_{comparator_prefix}": int(((comparator_top2 == 0) & (selected_top2 == 1)).sum()),
        f"harms_top2_vs_{comparator_prefix}": int(((comparator_top2 == 1) & (selected_top2 == 0)).sum()),
        f"same_top2_vs_{comparator_prefix}": int((comparator_top2 == selected_top2).sum()),
    }


# SECTION: Gate helpers
def evaluate_policy_gate(
    *,
    policy_metadata: dict[str, Any],
    trial_metrics: dict[str, Any],
    operational_reference_metrics: dict[str, Any],
    champion_reference_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the Day 05.4 hybrid gate against the operational and champion references."""
    promotion_eligible = bool(policy_metadata["promotion_eligible"])
    family = str(policy_metadata["family"])
    target_metric = str(policy_metadata["target_metric"])

    if not promotion_eligible:
        return {
            "gate_pass": False,
            "decision_label": "research_only" if family == "research" else "diagnostic_only",
            "target_metric_awarded": target_metric,
            "material_improvement_ok": False,
            "non_target_ok_vs_operational": False,
            "coverage_ok_vs_operational": False,
            "coverage_ok_vs_champion": False,
            "top2_ok_vs_champion": False,
            "coherence_ok": trial_metrics["coherence_after"] >= trial_metrics["coherence_before"],
            "overrides_harmed_ok": trial_metrics["overrides_harmed"] == 0,
            "selected_events_ok": int(trial_metrics["selected_events"]) > 0,
            "effective_override_ok": int(trial_metrics["overrides_count"]) > 0,
            "failure_reason": "research_only" if family == "research" else "diagnostic_only",
        }

    delta_top1_vs_operational = float(trial_metrics["delta_top1_vs_operational"])
    delta_top2_vs_operational = float(trial_metrics["delta_top2_vs_operational"])
    coherence_ok = float(trial_metrics["coherence_after"]) >= float(trial_metrics["coherence_before"])
    overrides_harmed_ok = int(trial_metrics["overrides_harmed"]) == 0
    selected_events_ok = int(trial_metrics["selected_events"]) > 0
    effective_override_ok = int(trial_metrics["overrides_count"]) > 0
    coverage_ok_vs_operational = float(trial_metrics["coverage"]) >= float(operational_reference_metrics["coverage"]) - 0.005
    coverage_ok_vs_champion = float(trial_metrics["coverage"]) >= float(champion_reference_metrics["coverage"]) - 0.005
    top2_ok_vs_champion = float(trial_metrics["top2_hit"]) >= float(champion_reference_metrics["top2_hit"])

    if target_metric == "top1_hit":
        material_improvement_ok = delta_top1_vs_operational >= DAY054_MATERIAL_IMPROVEMENT
        non_target_ok_vs_operational = delta_top2_vs_operational >= 0.0
        target_metric_awarded = "top1_hit"
    elif target_metric == "top2_hit":
        material_improvement_ok = delta_top2_vs_operational >= DAY054_MATERIAL_IMPROVEMENT
        non_target_ok_vs_operational = delta_top1_vs_operational >= 0.0
        target_metric_awarded = "top2_hit"
    else:
        top1_candidate = delta_top1_vs_operational >= DAY054_MATERIAL_IMPROVEMENT and delta_top2_vs_operational >= 0.0
        top2_candidate = delta_top2_vs_operational >= DAY054_MATERIAL_IMPROVEMENT and delta_top1_vs_operational >= 0.0
        material_improvement_ok = top1_candidate or top2_candidate
        non_target_ok_vs_operational = top1_candidate or top2_candidate
        target_metric_awarded = "top2_hit" if delta_top2_vs_operational >= delta_top1_vs_operational else "top1_hit"

    gate_pass = all(
        [
            selected_events_ok,
            effective_override_ok,
            material_improvement_ok,
            non_target_ok_vs_operational,
            coverage_ok_vs_operational,
            coverage_ok_vs_champion,
            top2_ok_vs_champion,
            coherence_ok,
            overrides_harmed_ok,
        ]
    )
    failure_reason = ""
    if not selected_events_ok:
        failure_reason = "no_selected_events"
    elif not effective_override_ok:
        failure_reason = "no_effective_override"
    elif not material_improvement_ok or not non_target_ok_vs_operational:
        failure_reason = "operational_gate_fail"
    elif not coverage_ok_vs_operational or not coverage_ok_vs_champion or not top2_ok_vs_champion:
        failure_reason = "champion_guardrail_fail"
    elif not coherence_ok or not overrides_harmed_ok:
        failure_reason = "champion_guardrail_fail"
    return {
        "gate_pass": gate_pass,
        "decision_label": "promotable" if gate_pass else "reject",
        "target_metric_awarded": target_metric_awarded,
        "material_improvement_ok": material_improvement_ok,
        "non_target_ok_vs_operational": non_target_ok_vs_operational,
        "coverage_ok_vs_operational": coverage_ok_vs_operational,
        "coverage_ok_vs_champion": coverage_ok_vs_champion,
        "top2_ok_vs_champion": top2_ok_vs_champion,
        "coherence_ok": coherence_ok,
        "overrides_harmed_ok": overrides_harmed_ok,
        "selected_events_ok": selected_events_ok,
        "effective_override_ok": effective_override_ok,
        "failure_reason": failure_reason,
    }


# SECTION: Trial flattening
def build_policy_trial_record(
    *,
    run_id: str,
    policy_metadata: dict[str, Any],
    trial_metrics: dict[str, Any],
    gate_result: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one policy trial into a CSV/JSON friendly record."""
    record = {
        "run_id": run_id,
        "policy_variant": policy_metadata["policy_variant"],
        "family": policy_metadata["family"],
        "target_metric": policy_metadata["target_metric"],
        "target_metric_awarded": gate_result["target_metric_awarded"],
        "promotion_eligible": bool(policy_metadata["promotion_eligible"]),
        "fallback_source": policy_metadata["fallback_source"],
        "events": trial_metrics["events"],
        "selected_events": trial_metrics["selected_events"],
        "selected_share": trial_metrics["selected_share"],
        "top1_hit": trial_metrics["top1_hit"],
        "top2_hit": trial_metrics["top2_hit"],
        "coverage": trial_metrics["coverage"],
        "coherence_before": trial_metrics["coherence_before"],
        "coherence_after": trial_metrics["coherence_after"],
        "coherence_delta": trial_metrics["coherence_delta"],
        "overrides_count": trial_metrics["overrides_count"],
        "overrides_improved": trial_metrics["overrides_improved"],
        "overrides_harmed": trial_metrics["overrides_harmed"],
        "overrides_neutral": trial_metrics["overrides_neutral"],
        "delta_top1_vs_champion": trial_metrics["delta_top1_vs_champion"],
        "delta_top2_vs_champion": trial_metrics["delta_top2_vs_champion"],
        "delta_top1_vs_operational": trial_metrics["delta_top1_vs_operational"],
        "delta_top2_vs_operational": trial_metrics["delta_top2_vs_operational"],
        "delta_top1_vs_baseline": trial_metrics["delta_top1_vs_baseline"],
        "delta_top2_vs_baseline": trial_metrics["delta_top2_vs_baseline"],
        "gate_pass": gate_result["gate_pass"],
        "decision_label": gate_result["decision_label"],
        "material_improvement_ok": gate_result["material_improvement_ok"],
        "non_target_ok_vs_operational": gate_result["non_target_ok_vs_operational"],
        "coverage_ok_vs_operational": gate_result["coverage_ok_vs_operational"],
        "coverage_ok_vs_champion": gate_result["coverage_ok_vs_champion"],
        "top2_ok_vs_champion": gate_result["top2_ok_vs_champion"],
        "coherence_ok": gate_result["coherence_ok"],
        "overrides_harmed_ok": gate_result["overrides_harmed_ok"],
        "selected_events_ok": gate_result["selected_events_ok"],
        "effective_override_ok": gate_result["effective_override_ok"],
        "failure_reason": gate_result["failure_reason"],
        "selected_source_counts": trial_metrics["selected_source_counts"],
    }
    for metric_name, metric_value in trial_metrics.items():
        if metric_name.startswith("improves_") or metric_name.startswith("harms_") or metric_name.startswith("same_"):
            record[metric_name] = metric_value
    return record
