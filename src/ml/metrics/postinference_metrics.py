from __future__ import annotations

from typing import Any

import pandas as pd


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _prepare_detail(detail_df: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "proveedor_candidato", "rank_event_score"}
    missing = required - set(detail_df.columns)
    if missing:
        raise ValueError(f"No se puede calcular métricas Day03. Faltan columnas en detalle: {sorted(missing)}")

    detail = detail_df.copy()
    detail["event_id"] = detail["event_id"].astype(str).str.strip()
    detail["proveedor_candidato"] = detail["proveedor_candidato"].astype(str).str.strip()
    detail["rank_event_score"] = pd.to_numeric(detail["rank_event_score"], errors="coerce")
    if "score_model" in detail.columns:
        detail["score_model"] = pd.to_numeric(detail["score_model"], errors="coerce")
    else:
        detail["score_model"] = 0.0

    if detail["rank_event_score"].isna().any():
        detail["rank_event_score"] = detail.groupby("event_id")["score_model"].rank(method="first", ascending=False)

    detail["rank_event_score"] = detail["rank_event_score"].astype(int)
    return detail


def _prepare_resumen(resumen_df: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "decision_pre_policy", "decision_final", "producto_canonico", "fecha_evento", "albaran_id"}
    missing = required - set(resumen_df.columns)
    if missing:
        raise ValueError(f"No se puede calcular métricas Day03. Faltan columnas en resumen: {sorted(missing)}")

    resumen = resumen_df.copy()
    for column in ["event_id", "decision_pre_policy", "decision_final", "producto_canonico", "fecha_evento", "albaran_id"]:
        resumen[column] = resumen[column].astype(str).str.strip()

    if "policy_applied_event" not in resumen.columns:
        resumen["policy_applied_event"] = 0
    resumen["policy_applied_event"] = pd.to_numeric(resumen["policy_applied_event"], errors="coerce").fillna(0).astype(int)

    return resumen


def _build_true_provider_map(detail_df: pd.DataFrame) -> dict[str, str]:
    if "target_elegido" not in detail_df.columns:
        return {}

    positive = detail_df[pd.to_numeric(detail_df["target_elegido"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if positive.empty:
        return {}

    positive = positive.sort_values(["event_id", "rank_event_score"], kind="mergesort")
    first = positive.groupby("event_id", as_index=False).first()[["event_id", "proveedor_candidato"]]
    return dict(zip(first["event_id"].astype(str), first["proveedor_candidato"].astype(str)))


def _build_ranked_candidates(detail_df: pd.DataFrame) -> dict[str, list[str]]:
    ranked = detail_df.sort_values(["event_id", "rank_event_score", "score_model"], ascending=[True, True, False], kind="mergesort")
    mapping: dict[str, list[str]] = {}
    for event_id, group in ranked.groupby("event_id", sort=False):
        providers = [p for p in group["proveedor_candidato"].astype(str).str.strip().tolist() if p]
        ordered_unique = list(dict.fromkeys(providers))
        mapping[str(event_id)] = ordered_unique
    return mapping


def _decision_topk_hit(
    resumen_df: pd.DataFrame,
    ranked_candidates: dict[str, list[str]],
    true_provider_by_event: dict[str, str],
    decision_col: str,
    k: int,
) -> tuple[float, float]:
    rows = resumen_df[["event_id", decision_col]].copy()
    rows[decision_col] = rows[decision_col].astype(str).str.strip()

    total_events = 0
    with_decision = 0
    hits = 0

    for _, row in rows.iterrows():
        event_id = _clean_text(row.get("event_id"))
        true_provider = _clean_text(true_provider_by_event.get(event_id, ""))
        if not true_provider:
            continue

        total_events += 1
        decision = _clean_text(row.get(decision_col))
        if decision:
            with_decision += 1

        ranked = ranked_candidates.get(event_id, [])
        ordered = []
        if decision:
            ordered.append(decision)
        ordered.extend([provider for provider in ranked if provider != decision])
        ordered = [provider for provider in ordered if provider]
        if true_provider in ordered[:k]:
            hits += 1

    if total_events == 0:
        return 0.0, 0.0

    return float(hits / total_events), float(with_decision / total_events)


def _coherence_rates_from_resumen_albaran(resumen_albaran_df: pd.DataFrame) -> tuple[int, float, float]:
    if resumen_albaran_df.empty:
        return 0, 0.0, 0.0

    working = resumen_albaran_df.copy()
    working["coherence_before"] = pd.to_numeric(working.get("coherence_before"), errors="coerce")
    working["coherence_after"] = pd.to_numeric(working.get("coherence_after"), errors="coerce")

    valid = working[working["coherence_before"].notna() & working["coherence_after"].notna()].copy()
    if valid.empty:
        return 0, 0.0, 0.0

    groups = int(len(valid))
    before = float(valid["coherence_before"].mean())
    after = float(valid["coherence_after"].mean())
    return groups, before, after


def compute_postinference_metrics(
    detail_df: pd.DataFrame,
    resumen_df: pd.DataFrame,
    resumen_albaran_df: pd.DataFrame,
) -> dict[str, float | int]:
    """Calcula métricas Day03 before/after para política por albarán."""
    detail = _prepare_detail(detail_df)
    resumen = _prepare_resumen(resumen_df)

    true_provider_by_event = _build_true_provider_map(detail)
    ranked_candidates = _build_ranked_candidates(detail)

    top1_before, coverage_before = _decision_topk_hit(
        resumen_df=resumen,
        ranked_candidates=ranked_candidates,
        true_provider_by_event=true_provider_by_event,
        decision_col="decision_pre_policy",
        k=1,
    )
    top2_before, _ = _decision_topk_hit(
        resumen_df=resumen,
        ranked_candidates=ranked_candidates,
        true_provider_by_event=true_provider_by_event,
        decision_col="decision_pre_policy",
        k=2,
    )
    top1_after, coverage_after = _decision_topk_hit(
        resumen_df=resumen,
        ranked_candidates=ranked_candidates,
        true_provider_by_event=true_provider_by_event,
        decision_col="decision_final",
        k=1,
    )
    top2_after, _ = _decision_topk_hit(
        resumen_df=resumen,
        ranked_candidates=ranked_candidates,
        true_provider_by_event=true_provider_by_event,
        decision_col="decision_final",
        k=2,
    )

    pair_groups, coherence_before, coherence_after = _coherence_rates_from_resumen_albaran(resumen_albaran_df)
    coherence_delta = float(coherence_after - coherence_before)

    comparison = resumen[["event_id", "decision_pre_policy", "decision_final", "policy_applied_event"]].copy()
    comparison["decision_pre_policy"] = comparison["decision_pre_policy"].astype(str).str.strip()
    comparison["decision_final"] = comparison["decision_final"].astype(str).str.strip()
    comparison["true_provider"] = comparison["event_id"].map(true_provider_by_event).fillna("").astype(str).str.strip()

    comparison = comparison[comparison["true_provider"].ne("")].copy()
    overrides_count = int((comparison["decision_final"] != comparison["decision_pre_policy"]).sum())

    before_hit = comparison["decision_pre_policy"] == comparison["true_provider"]
    after_hit = comparison["decision_final"] == comparison["true_provider"]
    overrides_improved = int((~before_hit & after_hit).sum())
    overrides_harmed = int((before_hit & ~after_hit).sum())
    overrides_neutral = int((before_hit == after_hit).sum())

    test_events = int(comparison["event_id"].nunique()) if not comparison.empty else int(resumen["event_id"].nunique())

    return {
        "top1_hit_before": float(top1_before),
        "top2_hit_before": float(top2_before),
        "top1_hit_after": float(top1_after),
        "top2_hit_after": float(top2_after),
        "pair_groups_PRODUCT_002_PRODUCT_003": int(pair_groups),
        "coherence_before": float(coherence_before),
        "coherence_after": float(coherence_after),
        "coherence_delta": float(coherence_delta),
        "overrides_count": int(overrides_count),
        "overrides_improved": int(overrides_improved),
        "overrides_harmed": int(overrides_harmed),
        "overrides_neutral": int(overrides_neutral),
        "coverage_before": float(coverage_before),
        "coverage_after": float(coverage_after),
        "test_events": int(test_events),
    }
