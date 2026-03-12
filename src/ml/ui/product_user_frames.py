from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from src.ml.product.day06_excel_raw import (
    apply_linea_sequence_to_blank_rows,
    normalize_excel_enrichment_frame,
)


PROPOSAL_HOMOGENEITY_COLUMNS = [
    "recomendacion",
    "alternativa",
    "confianza",
    "motivo_revision",
    "decision_final",
    "feedback_action",
]


def _stringify(value: Any) -> str:
    """Convert one scalar into a trimmed string while treating nulls as blank."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _unique_non_blank(values: pd.Series) -> list[str]:
    """Return ordered unique non-blank strings from one series."""
    ordered: list[str] = []
    for raw_value in values.tolist():
        value = _stringify(raw_value)
        if value == "" or value in ordered:
            continue
        ordered.append(value)
    return ordered


def _shared_string_value(values: pd.Series) -> str:
    """Return the shared string value for one group, or blank when not unique."""
    unique_values = _unique_non_blank(values)
    if len(unique_values) == 1:
        return unique_values[0]
    return ""


def _sum_nullable_float(values: pd.Series) -> float | pd.NA:
    """Sum one numeric series while preserving the all-null case."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.notna().sum() == 0:
        return pd.NA
    return float(numeric_values.fillna(0.0).sum())


def _build_identifier(prefix: str, parts: list[str]) -> str:
    """Build one deterministic identifier from stable text parts."""
    key = "|".join(parts)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def _coerce_identifier_tuple(value: Any) -> tuple[str, ...]:
    """Normalize one stored identifier sequence into a tuple of strings."""
    if isinstance(value, tuple):
        return tuple(_stringify(item) for item in value if _stringify(item) != "")
    if isinstance(value, list):
        return tuple(_stringify(item) for item in value if _stringify(item) != "")
    if isinstance(value, str):
        if "|" in value:
            return tuple(part for part in (segment.strip() for segment in value.split("|")) if part != "")
        cleaned = value.strip()
        return (cleaned,) if cleaned else ()
    cleaned = _stringify(value)
    return (cleaned,) if cleaned else ()


def infer_shared_albaran_id(enrichment_df: pd.DataFrame) -> str:
    """Infer one shared `albaran_id` from the current enrichment state when possible."""
    working = normalize_excel_enrichment_frame(enrichment_df)
    return _shared_string_value(working["albaran_id"])


def build_input_plan_frame(enrichment_df: pd.DataFrame) -> pd.DataFrame:
    """Build the user-facing input plan at product when safe, otherwise product-terminal."""
    working = normalize_excel_enrichment_frame(enrichment_df)
    if working.empty:
        return pd.DataFrame()

    terminal_counts = (
        working.groupby(["fecha_evento", "producto_canonico"], as_index=False)["terminal_compra"]
        .nunique()
        .rename(columns={"terminal_compra": "terminal_count"})
    )
    working = working.merge(
        terminal_counts,
        on=["fecha_evento", "producto_canonico"],
        how="left",
        validate="many_to_one",
    )
    working["visible_terminal"] = working["terminal_compra"].where(working["terminal_count"].gt(1), "")

    plan_rows: list[dict[str, Any]] = []
    grouped = working.groupby(
        ["fecha_evento", "producto_canonico", "visible_terminal"],
        sort=False,
        dropna=False,
    )
    for (fecha_evento, producto_canonico, visible_terminal), group in grouped:
        source_event_seed_ids = tuple(group["event_seed_id"].astype("string").fillna("").tolist())
        plan_rows.append(
            {
                "input_plan_id": _build_identifier(
                    "plan",
                    [_stringify(fecha_evento), _stringify(producto_canonico), _stringify(visible_terminal)],
                ),
                "fecha_evento": _stringify(fecha_evento),
                "producto_canonico": _stringify(producto_canonico),
                "terminal_compra": _stringify(visible_terminal),
                "litros_evento": _sum_nullable_float(group["litros_evento"]),
                "input_grain": "product_terminal" if _stringify(visible_terminal) else "product",
                "source_event_seed_ids": source_event_seed_ids,
                "source_event_count": int(group["event_seed_id"].nunique()),
                "source_terminal_count": int(group["terminal_compra"].astype("string").nunique()),
            }
        )

    plan_df = pd.DataFrame(plan_rows)
    if plan_df.empty:
        return plan_df
    plan_df["sort_terminal"] = plan_df["terminal_compra"].astype("string").fillna("")
    plan_df = plan_df.sort_values(
        ["fecha_evento", "producto_canonico", "sort_terminal"],
        kind="mergesort",
    ).drop(columns=["sort_terminal"])
    return plan_df.reset_index(drop=True)


def expand_input_plan_to_enrichment_frame(
    *,
    base_enrichment_df: pd.DataFrame,
    input_plan_df: pd.DataFrame,
    albaran_id: str,
) -> pd.DataFrame:
    """Expand one visible input plan back to the event-level enrichment contract."""
    working = normalize_excel_enrichment_frame(base_enrichment_df)
    if "source_event_seed_ids" not in input_plan_df.columns:
        raise ValueError("El input plan no contiene `source_event_seed_ids` para reconstruir el enrichment.")

    litros_map: dict[str, Any] = {}
    for row in input_plan_df.to_dict(orient="records"):
        litros_value = pd.to_numeric(row.get("litros_evento"), errors="coerce")
        normalized_litros: float | pd.NA
        if pd.isna(litros_value):
            normalized_litros = pd.NA
        else:
            normalized_litros = float(litros_value)
        for event_seed_id in _coerce_identifier_tuple(row.get("source_event_seed_ids")):
            litros_map[event_seed_id] = normalized_litros

    working["albaran_id"] = _stringify(albaran_id)
    working["litros_evento"] = working["event_seed_id"].astype("string").map(litros_map)
    working["linea_id"] = ""
    working, _, _ = apply_linea_sequence_to_blank_rows(
        enrichment_df=working,
        start_value=1,
    )
    return normalize_excel_enrichment_frame(working)


def build_event_review_frame(
    *,
    detail_df: pd.DataFrame,
    resumen_df: pd.DataFrame,
    feedback_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one business-facing frame at one row per event from the runtime outputs."""
    if detail_df.empty or resumen_df.empty:
        return pd.DataFrame()

    working_detail = detail_df.copy()
    working_detail["rank_event_score"] = (
        pd.to_numeric(working_detail.get("rank_event_score"), errors="coerce").fillna(999).astype(int)
    )
    working_detail = working_detail.sort_values(["event_id", "rank_event_score"], kind="mergesort")

    top1_rows = (
        working_detail[working_detail["rank_event_score"] == 1]
        .drop_duplicates(subset=["event_id"], keep="first")
        .copy()
    )
    if top1_rows.empty:
        top1_rows = working_detail.groupby("event_id", as_index=False).first()
    top2_rows = (
        working_detail[working_detail["rank_event_score"] == 2][["event_id", "proveedor_candidato"]]
        .drop_duplicates(subset=["event_id"], keep="first")
        .rename(columns={"proveedor_candidato": "alternativa"})
    )

    base_df = top1_rows[
        [
            column
            for column in [
                "event_id",
                "fecha_evento",
                "albaran_id",
                "producto_canonico",
                "terminal_compra",
                "litros_evento",
                "proveedor_candidato",
            ]
            if column in top1_rows.columns
        ]
    ].rename(columns={"proveedor_candidato": "recomendacion"})
    base_df = base_df.merge(top2_rows, on="event_id", how="left", validate="one_to_one")

    summary_cols = [
        column
        for column in [
            "event_id",
            "decision_final",
            "recommended_supplier",
            "review_status",
            "warning_reasons",
            "low_confidence_flag",
        ]
        if column in resumen_df.columns
    ]
    base_df = base_df.merge(
        resumen_df[summary_cols].drop_duplicates(subset=["event_id"], keep="first"),
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    if isinstance(feedback_df, pd.DataFrame) and not feedback_df.empty:
        feedback_cols = [
            column
            for column in [
                "event_id",
                "decision_final",
                "feedback_action",
                "override_reason",
                "feedback_notes",
            ]
            if column in feedback_df.columns
        ]
        feedback_snapshot = feedback_df[feedback_cols].drop_duplicates(subset=["event_id"], keep="last").rename(
            columns={
                "decision_final": "feedback_decision_final",
                "override_reason": "feedback_override_reason",
            }
        )
        base_df = base_df.merge(
            feedback_snapshot,
            on="event_id",
            how="left",
            validate="one_to_one",
        )
    else:
        base_df["feedback_decision_final"] = pd.NA
        base_df["feedback_action"] = "pending_review"
        base_df["feedback_override_reason"] = ""
        base_df["feedback_notes"] = ""

    for column, default_value in [
        ("feedback_action", "pending_review"),
        ("feedback_decision_final", pd.NA),
        ("feedback_override_reason", ""),
        ("feedback_notes", ""),
    ]:
        if column not in base_df.columns:
            base_df[column] = default_value

    base_df["alternativa"] = base_df["alternativa"].fillna("").astype(str)
    base_df["decision_final"] = (
        base_df["feedback_decision_final"]
        .fillna(base_df.get("decision_final"))
        .fillna(base_df["recomendacion"])
        .astype(str)
    )
    base_df["confianza"] = pd.to_numeric(base_df.get("low_confidence_flag"), errors="coerce").fillna(0).astype(int).map(
        {0: "Normal", 1: "Revisar"}
    )
    base_df["motivo_revision"] = base_df.get("warning_reasons", "").fillna("").astype(str)
    base_df["decision_final"] = base_df["decision_final"].replace("nan", "")
    base_df["alternativa"] = base_df["alternativa"].replace("nan", "")
    base_df["motivo_revision"] = base_df["motivo_revision"].replace("nan", "")
    base_df["feedback_override_reason"] = base_df["feedback_override_reason"].fillna("").astype(str)
    base_df["feedback_notes"] = base_df["feedback_notes"].fillna("").astype(str)
    base_df["decision_final"] = base_df["decision_final"].where(
        base_df["decision_final"].astype(str).str.strip().ne(""),
        base_df["recomendacion"].astype(str),
    )
    base_df["sort_confianza"] = base_df["confianza"].map({"Revisar": 0, "Normal": 1}).fillna(1)
    base_df = base_df.sort_values(
        ["sort_confianza", "fecha_evento", "albaran_id", "producto_canonico", "terminal_compra"],
        kind="mergesort",
    ).reset_index(drop=True)
    return base_df


def _is_collapsible(group_df: pd.DataFrame) -> bool:
    """Return whether one event group can collapse into one purchase proposal row."""
    if group_df.empty:
        return False
    for column in PROPOSAL_HOMOGENEITY_COLUMNS:
        unique_values = {_stringify(value) for value in group_df[column].tolist()}
        if len(unique_values) > 1:
            return False
    return True


def _join_notes(*notes: str) -> str:
    """Join short note fragments while avoiding duplicates and blanks."""
    ordered: list[str] = []
    for raw_note in notes:
        note = _stringify(raw_note)
        if note == "" or note in ordered:
            continue
        ordered.append(note)
    return " ".join(ordered)


def _build_proposal_row(
    group_df: pd.DataFrame,
    *,
    proposal_grain: str,
    split_reason: str,
    split_note: str,
) -> dict[str, Any]:
    """Collapse one homogeneous group into one user-facing purchase proposal row."""
    ordered_group = group_df.sort_values(["fecha_evento", "albaran_id", "producto_canonico", "terminal_compra", "event_id"])
    first_row = ordered_group.iloc[0]
    source_event_ids = tuple(ordered_group["event_id"].astype("string").fillna("").tolist())
    source_terminals = tuple(_unique_non_blank(ordered_group["terminal_compra"]))
    producto_base = _stringify(first_row.get("producto_canonico"))
    terminal_suffix = _stringify(first_row.get("terminal_compra"))

    producto_visible = producto_base
    if proposal_grain in {"product_terminal", "event"} and terminal_suffix:
        producto_visible = f"{producto_base} · {terminal_suffix}"

    note = _join_notes(split_note, _stringify(first_row.get("motivo_revision")))
    return {
        "proposal_id": _build_identifier("proposal", list(source_event_ids)),
        "fecha": _stringify(first_row.get("fecha_evento")),
        "albaran": _stringify(first_row.get("albaran_id")),
        "producto": producto_visible,
        "producto_base": producto_base,
        "terminal_visible": terminal_suffix if proposal_grain in {"product_terminal", "event"} else "",
        "litros": _sum_nullable_float(ordered_group["litros_evento"]),
        "proveedor_recomendado": _stringify(first_row.get("recomendacion")),
        "alternativa": _stringify(first_row.get("alternativa")),
        "confianza": _stringify(first_row.get("confianza")),
        "nota": note,
        "decision_final": _stringify(first_row.get("decision_final")),
        "feedback_action": _stringify(first_row.get("feedback_action")) or "pending_review",
        "override_reason": _stringify(first_row.get("feedback_override_reason")),
        "feedback_notes": _stringify(first_row.get("feedback_notes")),
        "source_event_ids": source_event_ids,
        "source_terminals": source_terminals,
        "collapsed_from_events_count": int(len(source_event_ids)),
        "proposal_grain": proposal_grain,
        "split_reason": split_reason,
    }


def build_purchase_proposal_frame(event_review_df: pd.DataFrame) -> pd.DataFrame:
    """Build the final user-facing purchase proposal, collapsing events when the case is clean."""
    if event_review_df.empty:
        return pd.DataFrame()

    proposal_rows: list[dict[str, Any]] = []
    product_groups = event_review_df.groupby(
        ["fecha_evento", "albaran_id", "producto_canonico"],
        sort=False,
        dropna=False,
    )
    for _, product_group in product_groups:
        if _is_collapsible(product_group):
            proposal_rows.append(
                _build_proposal_row(
                    product_group,
                    proposal_grain="product",
                    split_reason="",
                    split_note="",
                )
            )
            continue

        terminal_groups = [
            subgroup
            for _, subgroup in product_group.groupby("terminal_compra", sort=False, dropna=False)
        ]
        if terminal_groups and all(_is_collapsible(subgroup) for subgroup in terminal_groups):
            for subgroup in terminal_groups:
                proposal_rows.append(
                    _build_proposal_row(
                        subgroup,
                        proposal_grain="product_terminal",
                        split_reason="split_by_terminal",
                        split_note="Se muestra por terminal porque la propuesta no es unica para todo el producto.",
                    )
                )
            continue

        event_groups = [
            subgroup
            for _, subgroup in product_group.groupby("event_id", sort=False, dropna=False)
        ]
        for subgroup in event_groups:
            proposal_rows.append(
                _build_proposal_row(
                    subgroup,
                    proposal_grain="event",
                    split_reason="split_by_event",
                    split_note="Se muestra separada porque el producto no tiene una propuesta unica.",
                )
            )

    proposal_df = pd.DataFrame(proposal_rows)
    if proposal_df.empty:
        return proposal_df

    proposal_df["sort_confianza"] = proposal_df["confianza"].map({"Revisar": 0, "Normal": 1}).fillna(1)
    proposal_df["sort_terminal"] = proposal_df["terminal_visible"].astype("string").fillna("")
    proposal_df = proposal_df.sort_values(
        ["sort_confianza", "fecha", "albaran", "producto_base", "sort_terminal"],
        kind="mergesort",
    ).drop(columns=["sort_confianza", "sort_terminal"])
    return proposal_df.reset_index(drop=True)


def fan_out_proposal_feedback(
    *,
    feedback_df: pd.DataFrame,
    proposal_row: pd.Series,
    decision_final: str,
    feedback_action: str,
    override_reason: str,
    feedback_notes: str,
) -> pd.DataFrame:
    """Apply one proposal-level decision back to all underlying event feedback rows."""
    writable = feedback_df.copy()
    source_event_ids = list(_coerce_identifier_tuple(proposal_row.get("source_event_ids")))
    if not source_event_ids:
        raise ValueError("La propuesta no contiene `source_event_ids` para guardar feedback.")

    event_mask = writable["event_id"].astype("string").isin(source_event_ids)
    writable.loc[event_mask, "recommended_supplier"] = _stringify(proposal_row.get("proveedor_recomendado"))
    writable.loc[event_mask, "decision_final"] = _stringify(decision_final)
    writable.loc[event_mask, "feedback_action"] = _stringify(feedback_action)
    writable.loc[event_mask, "override_reason"] = _stringify(override_reason)
    writable.loc[event_mask, "feedback_notes"] = _stringify(feedback_notes)
    writable.loc[event_mask, "reviewed_at_utc"] = ""
    return writable
