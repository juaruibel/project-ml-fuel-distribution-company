from __future__ import annotations


# SECTION: Shared refs
def build_day054_shared_code_refs() -> list[str]:
    """Return the shared code refs used by Day 05.4 and Day 05.4b policy catalogs."""
    return [
        "src/ml/shared/day054_policy_helpers.py",
        "src/ml/rules/day054_local_policies.py",
        "src/ml/rules/day054_policy_strategies.py",
        "src/ml/metrics/day054_policy_evaluation.py",
        "src/ml/rules/scripts/day054_evaluate_local_policies.py",
    ]


# SECTION: Catalog builders
def build_day054_policy_catalog(prompt_ref: str, notebook_ref: str) -> list[dict[str, object]]:
    """Return the canonical Day 05.4 policy catalog for run01."""
    shared_code_refs = build_day054_shared_code_refs()
    return [
        {
            "policy_variant": "day054_go_c_fallback_to_baseline_v1",
            "family": "go_c",
            "strategy_name": "fallback",
            "flag_column": "go_c_fallback_flag",
            "target_metric": "top1_hit",
            "promotion_eligible": True,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054_go_b_dominant_flag_only_v1",
            "family": "go_b_dominant",
            "strategy_name": "flag_only",
            "flag_column": "go_b_dominant_low_conf_flag",
            "target_metric": "top2_hit",
            "promotion_eligible": False,
            "fallback_source": "pure_champion",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054_go_b_dominant_fallback_to_baseline_v1",
            "family": "go_b_dominant",
            "strategy_name": "fallback",
            "flag_column": "go_b_dominant_low_conf_flag",
            "target_metric": "top2_hit",
            "promotion_eligible": True,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054_go_b_dominant_fallback_to_baseline_with_policy_v1",
            "family": "go_b_dominant",
            "strategy_name": "fallback",
            "flag_column": "go_b_dominant_low_conf_flag",
            "target_metric": "top2_hit",
            "promotion_eligible": True,
            "fallback_source": "baseline_with_policy",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054_composite_go_c_plus_go_b_to_baseline_v1",
            "family": "composite",
            "strategy_name": "composite_fallback",
            "flag_columns": ["go_c_fallback_flag", "go_b_dominant_low_conf_flag"],
            "target_metric": "composite_topk",
            "promotion_eligible": True,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054_composite_go_c_plus_go_b_to_baseline_with_policy_v1",
            "family": "composite",
            "strategy_name": "composite_fallback",
            "flag_columns": ["go_c_fallback_flag", "go_b_dominant_low_conf_flag"],
            "target_metric": "composite_topk",
            "promotion_eligible": True,
            "fallback_source": "baseline_with_policy",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054_go_b_residual_contextual_selector_research_v1",
            "family": "research",
            "strategy_name": "research_contextual_selector",
            "flag_column": "go_b_residual_contextual_selector_flag",
            "target_metric": "top1_hit",
            "promotion_eligible": False,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
    ]


# SECTION: Catalog builders
def build_day054b_policy_catalog(prompt_ref: str, notebook_ref: str) -> list[dict[str, object]]:
    """Return the canonical Day 05.4b policy catalog for the residual precision microiteration."""
    shared_code_refs = build_day054_shared_code_refs()
    return [
        {
            "policy_variant": "day054b_go_b_residual_SUPPLIER_050_clean_v1",
            "family": "day054b_residual_probe",
            "strategy_name": "fallback",
            "flag_column": "day054b_go_b_residual_SUPPLIER_050_clean_flag",
            "target_metric": "top1_hit",
            "promotion_eligible": False,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054b_go_b_residual_SUPPLIER_019_low_conf_v1",
            "family": "day054b_residual_probe",
            "strategy_name": "fallback",
            "flag_column": "day054b_go_b_residual_SUPPLIER_019_low_conf_flag",
            "target_metric": "top1_hit",
            "promotion_eligible": False,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054b_go_b_residual_outer_terminal_extension_v1",
            "family": "day054b_residual_probe",
            "strategy_name": "fallback",
            "flag_column": "day054b_go_b_residual_outer_terminal_extension_flag",
            "target_metric": "top1_hit",
            "promotion_eligible": False,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054b_composite_go_c_plus_go_b_dominant_plus_residual_precision_v1",
            "family": "day054b_composite",
            "strategy_name": "composite_fallback",
            "flag_columns": [
                "go_c_fallback_flag",
                "go_b_dominant_low_conf_flag",
                "day054b_go_b_residual_precision_flag",
            ],
            "target_metric": "composite_topk",
            "promotion_eligible": True,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
    ]


# SECTION: Catalog builders
def build_day054c_policy_catalog(prompt_ref: str, notebook_ref: str) -> list[dict[str, object]]:
    """Return the canonical Day 05.4c policy catalog for the final residual hardening iteration."""
    shared_code_refs = build_day054_shared_code_refs()
    return [
        {
            "policy_variant": "day054c_go_b_residual_SUPPLIER_050_rank2_clean_v1",
            "family": "day054c_residual_probe",
            "strategy_name": "fallback",
            "flag_column": "day054c_go_b_residual_SUPPLIER_050_rank2_clean_flag",
            "target_metric": "top1_hit",
            "promotion_eligible": False,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054c_go_b_residual_SUPPLIER_019_low_conf_v1",
            "family": "day054c_residual_probe",
            "strategy_name": "fallback",
            "flag_column": "day054c_go_b_residual_SUPPLIER_019_low_conf_flag",
            "target_metric": "top1_hit",
            "promotion_eligible": False,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054c_go_b_residual_outer_terminal_extension_v1",
            "family": "day054c_residual_probe",
            "strategy_name": "fallback",
            "flag_column": "day054c_go_b_residual_outer_terminal_extension_flag",
            "target_metric": "top1_hit",
            "promotion_eligible": False,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
        {
            "policy_variant": "day054c_composite_go_c_plus_go_b_dominant_plus_residual_rank2_precision_v1",
            "family": "day054c_composite",
            "strategy_name": "composite_fallback",
            "flag_columns": [
                "go_c_fallback_flag",
                "go_b_dominant_low_conf_flag",
                "day054c_go_b_residual_rank2_precision_flag",
            ],
            "target_metric": "composite_topk",
            "promotion_eligible": True,
            "fallback_source": "baseline",
            "prompt_ref": prompt_ref,
            "notebook_ref": notebook_ref,
            "code_refs": shared_code_refs,
        },
    ]


# SECTION: Catalog builders
def build_all_day054_policy_catalog(prompt_ref: str, notebook_ref: str) -> list[dict[str, object]]:
    """Return the full catalog Day 05.4 plus Day 05.4b/05.4c for metadata lookup and app fallback."""
    return build_day054_policy_catalog(
        prompt_ref=prompt_ref,
        notebook_ref=notebook_ref,
    ) + build_day054b_policy_catalog(
        prompt_ref=prompt_ref,
        notebook_ref=notebook_ref,
    ) + build_day054c_policy_catalog(
        prompt_ref=prompt_ref,
        notebook_ref=notebook_ref,
    )
