from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_POLICY_RULE_ID = "P_DAY03_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009"
DEFAULT_POLICY_REASON = "PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009"


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _normalize_summary_frame(resumen_df: pd.DataFrame) -> pd.DataFrame:
    working = resumen_df.copy()
    for column in ["event_id", "fecha_evento", "albaran_id", "producto_canonico", "decision_final", "decision_pre_policy"]:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].astype(str).str.strip()

    if "policy_applied_event" not in working.columns:
        working["policy_applied_event"] = 0
    working["policy_applied_event"] = pd.to_numeric(working["policy_applied_event"], errors="coerce").fillna(0).astype(int)

    for column in ["policy_rule_id", "policy_reason_event"]:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].astype(str).str.strip()

    if "decision_source" not in working.columns:
        working["decision_source"] = ""
    working["decision_source"] = working["decision_source"].astype(str).str.strip()

    if "override_reason" not in working.columns:
        working["override_reason"] = ""
    working["override_reason"] = working["override_reason"].astype(str).str.strip()

    return working


def _build_SUPPLIER_009_unblocked_map(detail_df: pd.DataFrame) -> dict[str, bool]:
    required = {"event_id", "proveedor_candidato", "blocked_by_rule"}
    missing = required - set(detail_df.columns)
    if missing:
        raise ValueError(f"No se puede calcular disponibilidad de SUPPLIER_009 por evento. Faltan columnas: {sorted(missing)}")

    detail = detail_df.copy()
    detail["event_id"] = detail["event_id"].astype(str).str.strip()
    detail["proveedor_candidato"] = detail["proveedor_candidato"].astype(str).str.strip()
    detail["blocked_by_rule"] = pd.to_numeric(detail["blocked_by_rule"], errors="coerce").fillna(0).astype(int)

    mask = detail["proveedor_candidato"].eq("SUPPLIER_009") & detail["blocked_by_rule"].eq(0)
    available = detail.loc[mask, "event_id"].astype(str).str.strip().unique().tolist()
    return {event_id: True for event_id in available}


def apply_PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009_policy(
    resumen_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    policy_rule_id: str = DEFAULT_POLICY_RULE_ID,
    policy_reason: str = DEFAULT_POLICY_REASON,
) -> pd.DataFrame:
    """
    Política Day03:
    - Grano: fecha_evento + albaran_id
    - Trigger: PRODUCT_003 con decision_final=SUPPLIER_009
    - Acción: forzar PRODUCT_002 a SUPPLIER_009 si SUPPLIER_009 está disponible y no bloqueado en PRODUCT_002.
    """
    summary = _normalize_summary_frame(resumen_df)
    SUPPLIER_009_unblocked_by_event = _build_SUPPLIER_009_unblocked_map(detail_df)

    required_cols = {"fecha_evento", "albaran_id", "producto_canonico", "event_id", "decision_final"}
    missing = required_cols - set(summary.columns)
    if missing:
        raise ValueError(f"No se puede aplicar política de albarán. Faltan columnas: {sorted(missing)}")

    for (_, _), idx in summary.groupby(["fecha_evento", "albaran_id"], sort=False).groups.items():
        group = summary.loc[idx]
        has_trigger = bool(
            (
                group["producto_canonico"].astype(str).str.strip().eq("PRODUCT_003")
                & group["decision_final"].astype(str).str.strip().eq("SUPPLIER_009")
            ).any()
        )
        if not has_trigger:
            continue

        PRODUCT_002_idx = group[group["producto_canonico"].astype(str).str.strip().eq("PRODUCT_002")].index
        if len(PRODUCT_002_idx) == 0:
            continue

        for row_idx in PRODUCT_002_idx:
            event_id = _clean_text(summary.at[row_idx, "event_id"])
            current_decision = _clean_text(summary.at[row_idx, "decision_final"])
            SUPPLIER_009_available = SUPPLIER_009_unblocked_by_event.get(event_id, False)

            if not SUPPLIER_009_available:
                continue
            if current_decision == "SUPPLIER_009":
                continue

            previous_reason = _clean_text(summary.at[row_idx, "override_reason"])
            summary.at[row_idx, "decision_final"] = "SUPPLIER_009"
            summary.at[row_idx, "policy_applied_event"] = 1
            summary.at[row_idx, "policy_rule_id"] = policy_rule_id
            summary.at[row_idx, "policy_reason_event"] = policy_reason
            summary.at[row_idx, "decision_source"] = "policy_albaran_coherence"
            if previous_reason:
                summary.at[row_idx, "override_reason"] = f"{previous_reason}|{policy_reason}"
            else:
                summary.at[row_idx, "override_reason"] = policy_reason

    return summary


def apply_albaran_policy(
    resumen_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    albaran_policy: str,
) -> pd.DataFrame:
    """Despacha la política de albarán configurada."""
    policy = _clean_text(albaran_policy) or "none"
    if policy == "none":
        return _normalize_summary_frame(resumen_df)
    if policy == "PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009":
        return apply_PRODUCT_002_follow_PRODUCT_003_SUPPLIER_009_policy(resumen_df=resumen_df, detail_df=detail_df)
    raise ValueError(f"Política de albarán no soportada: {policy}")


def _coherence_flag(group: pd.DataFrame, decision_col: str) -> float | None:
    ab = group[group["producto_canonico"].astype(str).str.strip().isin(["PRODUCT_002", "PRODUCT_003"])].copy()
    if ab.empty:
        return None

    products = set(ab["producto_canonico"].astype(str).str.strip().tolist())
    if not {"PRODUCT_002", "PRODUCT_003"}.issubset(products):
        return None

    providers = ab[decision_col].astype(str).str.strip()
    providers = providers[providers.ne("")]
    if providers.empty:
        return None

    return 1.0 if providers.nunique() == 1 else 0.0


def build_resumen_albaran(resumen_df: pd.DataFrame, run_id: str, ts_utc: str) -> pd.DataFrame:
    """Construye resumen a grano fecha_evento + albaran_id."""
    summary = _normalize_summary_frame(resumen_df)

    records: list[dict[str, Any]] = []
    for (fecha_evento, albaran_id), group in summary.groupby(["fecha_evento", "albaran_id"], sort=False):
        coherence_before = _coherence_flag(group, "decision_pre_policy")
        coherence_after = _coherence_flag(group, "decision_final")

        overrides = int((group["decision_final"].astype(str).str.strip() != group["decision_pre_policy"].astype(str).str.strip()).sum())
        policy_applied = int(pd.to_numeric(group["policy_applied_event"], errors="coerce").fillna(0).astype(int).sum() > 0)
        policy_reasons = (
            group.loc[group["policy_applied_event"].astype(int) == 1, "policy_reason_event"]
            .astype(str)
            .str.strip()
        )
        policy_reasons = [reason for reason in policy_reasons.tolist() if reason]
        policy_reason = "|".join(sorted(set(policy_reasons)))

        records.append(
            {
                "fecha_evento": _clean_text(fecha_evento),
                "albaran_id": _clean_text(albaran_id),
                "eventos_total": int(group["event_id"].astype(str).nunique()),
                "eventos_override": overrides,
                "coherence_before": coherence_before,
                "coherence_after": coherence_after,
                "policy_applied": policy_applied,
                "policy_reason": policy_reason,
                "run_id": run_id,
                "ts_utc": ts_utc,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "fecha_evento",
                "albaran_id",
                "eventos_total",
                "eventos_override",
                "coherence_before",
                "coherence_after",
                "policy_applied",
                "policy_reason",
                "run_id",
                "ts_utc",
            ]
        )

    output = pd.DataFrame(records)
    output = output.sort_values(["fecha_evento", "albaran_id"], kind="mergesort").reset_index(drop=True)
    return output
