from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ml.rules.blocklist import apply_blocklist_candidates, load_blocklist_rules


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _load_blocklist(blocklist_path: Path) -> pd.DataFrame:
    return load_blocklist_rules(blocklist_path)


def _apply_blocklist_candidates(dataset: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    return apply_blocklist_candidates(dataset, rules)


def _build_event_id(row: pd.Series) -> str:
    key_parts = [
        _clean_text(row.get("fecha_evento", "")),
        _clean_text(row.get("albaran_id", "")),
        _clean_text(row.get("linea_id", "")),
        _clean_text(row.get("producto_canonico", "")),
        _clean_text(row.get("proveedor_elegido_real", "")),
        _clean_text(row.get("terminal_compra", "")),
    ]
    key = "|".join(key_parts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _build_column_dictionary() -> dict[str, dict[str, str]]:
    return {
        "event_id": {"rol": "identity", "justificacion": "Id único determinista por evento para trazabilidad."},
        "fecha_evento": {"rol": "context", "justificacion": "Fecha operativa del evento de compra."},
        "albaran_id": {"rol": "audit", "justificacion": "Identificador documental del albarán."},
        "linea_id": {"rol": "audit", "justificacion": "Identificador de línea dentro del albarán."},
        "producto_canonico": {"rol": "context", "justificacion": "Producto normalizado para unir compras y ofertas."},
        "terminal_compra": {"rol": "context", "justificacion": "Terminal operativa registrada en compras."},
        "proveedor_elegido_real": {"rol": "label_source", "justificacion": "Proveedor realmente comprado (ground truth)."},
        "proveedor_candidato": {"rol": "candidate_key", "justificacion": "Proveedor del universo candidato para el evento."},
        "coste_min_dia_proveedor": {"rol": "feature_base", "justificacion": "Coste del candidato en ese día-producto."},
        "rank_coste_dia_producto": {"rol": "feature_base", "justificacion": "Posición relativa de coste del candidato."},
        "terminales_cubiertos": {"rol": "feature_base", "justificacion": "Cobertura de terminales de la oferta."},
        "observaciones_oferta": {"rol": "feature_base", "justificacion": "Número de observaciones agregadas de oferta."},
        "candidatos_evento_count": {"rol": "feature_competition", "justificacion": "Número de proveedores candidatos del evento."},
        "coste_min_evento": {"rol": "feature_competition", "justificacion": "Coste mínimo entre candidatos del evento."},
        "coste_max_evento": {"rol": "feature_competition", "justificacion": "Coste máximo entre candidatos del evento."},
        "spread_coste_evento": {"rol": "feature_competition", "justificacion": "Dispersión de costes en el evento."},
        "delta_vs_min_evento": {"rol": "feature_competition", "justificacion": "Diferencia absoluta del candidato frente al mínimo."},
        "ratio_vs_min_evento": {"rol": "feature_competition", "justificacion": "Diferencia relativa del candidato frente al mínimo."},
        "litros_evento": {"rol": "context", "justificacion": "Volumen comprado en la línea."},
        "precio_unitario_evento": {"rol": "audit", "justificacion": "Precio real pagado en la compra (auditoría)." },
        "importe_total_evento": {"rol": "audit", "justificacion": "Importe total real de la línea (auditoría)."},
        "dia_semana": {"rol": "feature_calendar", "justificacion": "Estacionalidad semanal del evento."},
        "mes": {"rol": "feature_calendar", "justificacion": "Estacionalidad mensual del evento."},
        "fin_mes": {"rol": "feature_calendar", "justificacion": "Flag de cierre de mes."},
        "blocked_by_rule_candidate": {"rol": "feature_business_rule", "justificacion": "Flag de bloqueo por regla de negocio."},
        "block_reason_candidate": {"rol": "feature_business_rule", "justificacion": "Motivo de bloqueo aplicado al candidato."},
        "target_elegido": {"rol": "target", "justificacion": "Etiqueta 1/0: candidato elegido vs no elegido."},
        "v2_run_id": {"rol": "lineage", "justificacion": "Identificador de ejecución ETL V2."},
        "v2_ts_utc": {"rol": "lineage", "justificacion": "Timestamp UTC de generación del dataset."},
    }


def _write_data_dictionary(output_path: Path, dataset: pd.DataFrame) -> None:
    metadata = _build_column_dictionary()
    lines = [
        "# Day02 · Data Dictionary V2 (ETL-only)",
        "",
        "## Alcance",
        "- Documento generado automáticamente desde el schema final de `dataset_modelo_proveedor_v2_candidates.csv`.",
        "- Define variable, tipo, rol y justificación de negocio/técnica.",
        "",
        "| columna | dtype | rol | justificación |",
        "|---|---|---|---|",
    ]
    for column in dataset.columns:
        info = metadata.get(column, {"rol": "unknown", "justificacion": "No documentado."})
        dtype = str(dataset[column].dtype)
        lines.append(
            f"| {column} | {dtype} | {info['rol']} | {info['justificacion']} |"
        )
    lines += [
        "",
        "## Nota de diseño",
        "- `terminal_oferta` inferida no forma parte de V2; el puente `TERM -> CLH` se deja como diagnóstico en reporte separado.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    fact_compras_path: Path,
    fact_ofertas_path: Path,
    blocklist_path: Path,
    output_dataset_path: Path,
    output_excluded_path: Path,
    quality_report_path: Path,
    data_dictionary_path: Path,
    run_id: str | None = None,
) -> dict:
    if not fact_compras_path.exists():
        raise FileNotFoundError(f"No existe fact compras: {fact_compras_path}")
    if not fact_ofertas_path.exists():
        raise FileNotFoundError(f"No existe fact ofertas: {fact_ofertas_path}")

    execution_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_ts_utc = datetime.now(timezone.utc).isoformat()

    compras = pd.read_csv(fact_compras_path, dtype=str, keep_default_na=False).copy()
    ofertas = pd.read_csv(fact_ofertas_path, dtype=str, keep_default_na=False).copy()

    compras = compras.rename(
        columns={
            "fecha_compra": "fecha_evento",
            "terminal_canonico": "terminal_compra",
            "proveedor_canonico": "proveedor_elegido_real",
            "litros": "litros_evento",
            "precio_unitario": "precio_unitario_evento",
            "importe_total": "importe_total_evento",
        }
    )

    compras["fecha_evento"] = pd.to_datetime(compras.get("fecha_evento", ""), errors="coerce").dt.strftime("%Y-%m-%d")
    compras["event_id"] = compras.apply(_build_event_id, axis=1)
    compras["litros_evento"] = pd.to_numeric(compras.get("litros_evento", ""), errors="coerce")
    compras["precio_unitario_evento"] = pd.to_numeric(compras.get("precio_unitario_evento", ""), errors="coerce")
    compras["importe_total_evento"] = pd.to_numeric(compras.get("importe_total_evento", ""), errors="coerce")

    fechas = pd.to_datetime(compras["fecha_evento"], errors="coerce")
    compras["dia_semana"] = fechas.dt.dayofweek
    compras["mes"] = fechas.dt.month
    compras["fin_mes"] = fechas.dt.is_month_end.astype(int)

    ofertas = ofertas.rename(columns={"fecha_oferta": "fecha_evento", "proveedor_canonico": "proveedor_candidato"})
    ofertas["coste_min_dia_proveedor"] = pd.to_numeric(ofertas.get("coste_min_dia_proveedor", ""), errors="coerce")
    ofertas["rank_coste_dia_producto"] = pd.to_numeric(ofertas.get("rank_coste_dia_producto", ""), errors="coerce")
    ofertas["terminales_cubiertos"] = pd.to_numeric(ofertas.get("terminales_cubiertos", ""), errors="coerce")
    ofertas["observaciones_oferta"] = pd.to_numeric(ofertas.get("observaciones_oferta", ""), errors="coerce")

    ofertas_for_join = ofertas[
        [
            "fecha_evento",
            "producto_canonico",
            "proveedor_candidato",
            "coste_min_dia_proveedor",
            "rank_coste_dia_producto",
            "terminales_cubiertos",
            "observaciones_oferta",
        ]
    ].copy()

    event_columns = [
        "event_id",
        "fecha_evento",
        "albaran_id",
        "linea_id",
        "producto_canonico",
        "terminal_compra",
        "proveedor_elegido_real",
        "litros_evento",
        "precio_unitario_evento",
        "importe_total_evento",
        "dia_semana",
        "mes",
        "fin_mes",
    ]
    events = compras[event_columns].copy()

    candidates = events.merge(
        ofertas_for_join,
        on=["fecha_evento", "producto_canonico"],
        how="left",
    )
    candidates["has_candidate"] = candidates["proveedor_candidato"].astype(str).str.strip().ne("")
    candidates["target_elegido"] = (
        candidates["has_candidate"]
        & candidates["proveedor_candidato"].astype(str).str.strip().eq(
            candidates["proveedor_elegido_real"].astype(str).str.strip()
        )
    ).astype(int)

    competition_base = candidates[candidates["has_candidate"]].copy()
    event_stats = (
        competition_base.groupby("event_id", as_index=False)
        .agg(
            candidatos_evento_count=("proveedor_candidato", "nunique"),
            coste_min_evento=("coste_min_dia_proveedor", "min"),
            coste_max_evento=("coste_min_dia_proveedor", "max"),
            positive_labels=("target_elegido", "sum"),
        )
    )
    event_stats["spread_coste_evento"] = event_stats["coste_max_evento"] - event_stats["coste_min_evento"]

    event_presence = (
        candidates.groupby("event_id", as_index=False)
        .agg(has_candidates=("has_candidate", "any"))
        .merge(event_stats[["event_id", "positive_labels"]], on="event_id", how="left")
    )
    event_presence["positive_labels"] = event_presence["positive_labels"].fillna(0).astype(int)

    event_presence["exclusion_reason"] = ""
    event_presence.loc[~event_presence["has_candidates"], "exclusion_reason"] = "no_candidates_for_event"
    event_presence.loc[
        event_presence["has_candidates"] & event_presence["positive_labels"].eq(0),
        "exclusion_reason",
    ] = "chosen_provider_not_in_candidates"
    event_presence.loc[
        event_presence["has_candidates"] & event_presence["positive_labels"].gt(1),
        "exclusion_reason",
    ] = "multiple_positive_labels"
    event_presence["is_included"] = event_presence["exclusion_reason"].eq("")

    candidates = candidates.merge(
        event_stats[
            [
                "event_id",
                "candidatos_evento_count",
                "coste_min_evento",
                "coste_max_evento",
                "spread_coste_evento",
            ]
        ],
        on="event_id",
        how="left",
    ).merge(
        event_presence[["event_id", "is_included", "exclusion_reason"]],
        on="event_id",
        how="left",
    )

    candidates["delta_vs_min_evento"] = candidates["coste_min_dia_proveedor"] - candidates["coste_min_evento"]
    candidates["ratio_vs_min_evento"] = candidates["coste_min_dia_proveedor"] / candidates["coste_min_evento"]

    included_event_ids = set(event_presence.loc[event_presence["is_included"], "event_id"])
    dataset_v2 = candidates[
        candidates["event_id"].isin(included_event_ids) & candidates["has_candidate"]
    ].copy()

    blocklist_rules = _load_blocklist(blocklist_path)
    dataset_v2 = _apply_blocklist_candidates(dataset_v2, blocklist_rules)

    dataset_v2["target_elegido"] = dataset_v2["target_elegido"].astype(int)
    dataset_v2["v2_run_id"] = execution_run_id
    dataset_v2["v2_ts_utc"] = generated_ts_utc

    output_columns = [
        "event_id",
        "fecha_evento",
        "albaran_id",
        "linea_id",
        "producto_canonico",
        "terminal_compra",
        "proveedor_elegido_real",
        "proveedor_candidato",
        "coste_min_dia_proveedor",
        "rank_coste_dia_producto",
        "terminales_cubiertos",
        "observaciones_oferta",
        "candidatos_evento_count",
        "coste_min_evento",
        "coste_max_evento",
        "spread_coste_evento",
        "delta_vs_min_evento",
        "ratio_vs_min_evento",
        "litros_evento",
        "precio_unitario_evento",
        "importe_total_evento",
        "dia_semana",
        "mes",
        "fin_mes",
        "blocked_by_rule_candidate",
        "block_reason_candidate",
        "target_elegido",
        "v2_run_id",
        "v2_ts_utc",
    ]
    dataset_v2 = dataset_v2[output_columns].copy()

    excluded = events.merge(
        event_presence[["event_id", "exclusion_reason", "has_candidates", "positive_labels"]],
        on="event_id",
        how="left",
    )
    excluded = excluded[excluded["exclusion_reason"].astype(str).str.strip().ne("")].copy()
    excluded = excluded.rename(columns={"has_candidates": "event_has_candidates"})

    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    output_excluded_path.parent.mkdir(parents=True, exist_ok=True)
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_v2.to_csv(output_dataset_path, index=False, encoding="utf-8")
    excluded.to_csv(output_excluded_path, index=False, encoding="utf-8")
    _write_data_dictionary(data_dictionary_path, dataset_v2)

    events_total = int(events["event_id"].nunique())
    events_with_candidates = int(event_presence["has_candidates"].sum())
    events_included = int(event_presence["is_included"].sum())
    events_excluded_no_candidates = int((event_presence["exclusion_reason"] == "no_candidates_for_event").sum())
    events_excluded_no_positive = int(
        (event_presence["exclusion_reason"] == "chosen_provider_not_in_candidates").sum()
    )
    events_excluded_multiple_positive = int(
        (event_presence["exclusion_reason"] == "multiple_positive_labels").sum()
    )

    candidates_per_event = (
        dataset_v2.groupby("event_id")["proveedor_candidato"].nunique()
        if not dataset_v2.empty
        else pd.Series(dtype=float)
    )

    critical_columns = ["event_id", "fecha_evento", "producto_canonico", "proveedor_candidato", "target_elegido"]
    critical_null_counts = {
        column: int(dataset_v2[column].isna().sum()) for column in critical_columns if column in dataset_v2
    }
    positive_per_event = dataset_v2.groupby("event_id")["target_elegido"].sum() if not dataset_v2.empty else pd.Series(dtype=float)
    invalid_positive_events = int((positive_per_event != 1).sum()) if len(positive_per_event) else 0

    quality_report = {
        "status": "ok",
        "run_id": execution_run_id,
        "generated_ts_utc": generated_ts_utc,
        "input_fact_compras": str(fact_compras_path),
        "input_fact_ofertas": str(fact_ofertas_path),
        "input_blocklist": str(blocklist_path),
        "output_dataset": str(output_dataset_path),
        "output_excluded_events": str(output_excluded_path),
        "output_data_dictionary": str(data_dictionary_path),
        "events_total": events_total,
        "events_with_candidates": events_with_candidates,
        "events_included": events_included,
        "events_excluded_no_candidates": events_excluded_no_candidates,
        "events_excluded_no_positive": events_excluded_no_positive,
        "events_excluded_multiple_positive": events_excluded_multiple_positive,
        "avg_candidates_per_event": float(candidates_per_event.mean()) if len(candidates_per_event) else 0.0,
        "p50_candidates_per_event": float(candidates_per_event.quantile(0.50)) if len(candidates_per_event) else 0.0,
        "p90_candidates_per_event": float(candidates_per_event.quantile(0.90)) if len(candidates_per_event) else 0.0,
        "rows_output": int(len(dataset_v2)),
        "critical_null_counts": critical_null_counts,
        "events_with_invalid_positive_count": invalid_positive_events,
        "blocked_rows": int(dataset_v2["blocked_by_rule_candidate"].sum())
        if "blocked_by_rule_candidate" in dataset_v2
        else 0,
    }
    quality_report_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye dataset_modelo_proveedor_v2_candidates.csv (ETL-only).")
    parser.add_argument(
        "--fact-compras",
        type=Path,
        default=Path("data/public/support/fact_compras.csv"),
        help="Input fact compras.",
    )
    parser.add_argument(
        "--fact-ofertas",
        type=Path,
        default=Path("data/public/support/fact_ofertas_diarias.csv"),
        help="Input fact ofertas diarias.",
    )
    parser.add_argument(
        "--blocklist",
        type=Path,
        default=Path("config/business_blocklist_rules.csv"),
        help="Reglas de bloqueo de negocio.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v2_candidates.csv"),
        help="Output dataset V2 de candidatos.",
    )
    parser.add_argument(
        "--excluded-output",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v2_excluded_events.csv"),
        help="Output de eventos excluidos.",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("artifacts/public/data_quality_v2_candidates.json"),
        help="Reporte JSON de calidad/cobertura V2.",
    )
    parser.add_argument(
        "--data-dictionary",
        type=Path,
        default=Path("artifacts/public/data_dictionary_v2_candidates.md"),
        help="Reporte markdown diccionario de variables V2.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador opcional de ejecución.",
    )
    args = parser.parse_args()

    summary = run(
        fact_compras_path=args.fact_compras,
        fact_ofertas_path=args.fact_ofertas,
        blocklist_path=args.blocklist,
        output_dataset_path=args.output,
        output_excluded_path=args.excluded_output,
        quality_report_path=args.quality_report,
        data_dictionary_path=args.data_dictionary,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
