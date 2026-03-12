from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _load_blocklist(blocklist_path: Path) -> pd.DataFrame:
    if not blocklist_path.exists():
        return pd.DataFrame(
            columns=[
                "rule_id",
                "active",
                "fecha_inicio",
                "fecha_fin",
                "producto_canonico",
                "terminal_canonico",
                "proveedor_canonico",
                "motivo",
            ]
        )
    rules = pd.read_csv(blocklist_path, dtype=str, keep_default_na=False)
    if rules.empty:
        return rules
    rules["active"] = rules.get("active", "0").astype(str).str.strip()
    rules = rules[rules["active"].isin(["1", "true", "TRUE", "yes", "YES"])].copy()
    if rules.empty:
        return rules
    rules["fecha_inicio"] = pd.to_datetime(rules.get("fecha_inicio", ""), errors="coerce")
    rules["fecha_fin"] = pd.to_datetime(rules.get("fecha_fin", ""), errors="coerce")
    for column in ["producto_canonico", "terminal_canonico", "proveedor_canonico", "motivo", "rule_id"]:
        if column not in rules.columns:
            rules[column] = ""
        rules[column] = rules[column].astype(str).str.strip()
    return rules.reset_index(drop=True)


def _matches_with_wildcard(value: str, rule_value: str) -> bool:
    rule_token = _clean_text(rule_value)
    if rule_token in {"", "*"}:
        return True
    return _clean_text(value) == rule_token


def _apply_blocklist(dataset: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    output = dataset.copy()
    output["blocked_by_rule"] = 0
    output["block_reason"] = ""
    if rules.empty or output.empty:
        return output

    fecha_series = pd.to_datetime(output["fecha_compra"], errors="coerce")
    for _, rule in rules.iterrows():
        rule_provider = _clean_text(rule.get("proveedor_canonico", ""))
        if not rule_provider:
            continue
        provider_mask = output["proveedor_canonico"].astype(str).str.strip().eq(rule_provider)
        product_mask = output["producto_canonico"].astype(str).map(
            lambda value: _matches_with_wildcard(value, rule.get("producto_canonico", "*"))
        )
        terminal_mask = output["terminal_canonico"].astype(str).map(
            lambda value: _matches_with_wildcard(value, rule.get("terminal_canonico", "*"))
        )

        start = rule.get("fecha_inicio")
        end = rule.get("fecha_fin")
        date_mask = pd.Series([True] * len(output), index=output.index)
        if pd.notna(start):
            date_mask = date_mask & (fecha_series >= start)
        if pd.notna(end):
            date_mask = date_mask & (fecha_series <= end)

        final_mask = provider_mask & product_mask & terminal_mask & date_mask
        if not final_mask.any():
            continue

        reason = _clean_text(rule.get("motivo", "")) or "blocked_by_business_rule"
        rule_id = _clean_text(rule.get("rule_id", "RULE"))
        rule_message = f"{rule_id}:{reason}"
        output.loc[final_mask, "blocked_by_rule"] = 1
        output.loc[final_mask, "block_reason"] = rule_message
    return output


def run(
    fact_compras_path: Path,
    fact_ofertas_path: Path,
    blocklist_path: Path,
    output_path: Path,
    run_id: str | None = None,
) -> dict:
    if not fact_compras_path.exists():
        raise FileNotFoundError(f"No existe fact compras: {fact_compras_path}")
    if not fact_ofertas_path.exists():
        raise FileNotFoundError(f"No existe fact ofertas: {fact_ofertas_path}")

    execution_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fact_compras = pd.read_csv(fact_compras_path, dtype=str, keep_default_na=False)
    fact_ofertas = pd.read_csv(fact_ofertas_path, dtype=str, keep_default_na=False)

    fact_compras["fecha_compra"] = pd.to_datetime(
        fact_compras.get("fecha_compra", ""), errors="coerce"
    )
    fact_compras["litros"] = pd.to_numeric(fact_compras.get("litros", ""), errors="coerce")
    fact_compras["precio_unitario"] = pd.to_numeric(
        fact_compras.get("precio_unitario", ""), errors="coerce"
    )
    fact_compras["importe_total"] = pd.to_numeric(fact_compras.get("importe_total", ""), errors="coerce")

    fact_ofertas["coste_min_dia_proveedor"] = pd.to_numeric(
        fact_ofertas.get("coste_min_dia_proveedor", ""), errors="coerce"
    )
    fact_ofertas["fecha_oferta"] = pd.to_datetime(
        fact_ofertas.get("fecha_oferta", ""), errors="coerce"
    )
    fact_ofertas["terminales_cubiertos"] = pd.to_numeric(
        fact_ofertas.get("terminales_cubiertos", ""), errors="coerce"
    )
    fact_ofertas["rank_coste_dia_producto"] = pd.to_numeric(
        fact_ofertas.get("rank_coste_dia_producto", ""), errors="coerce"
    )

    ofertas_join = fact_ofertas.rename(columns={"fecha_oferta": "fecha_compra"})[
        [
            "fecha_compra",
            "producto_canonico",
            "proveedor_canonico",
            "coste_min_dia_proveedor",
            "rank_coste_dia_producto",
            "terminales_cubiertos",
        ]
    ].copy()

    dataset = fact_compras.merge(
        ofertas_join,
        on=["fecha_compra", "producto_canonico", "proveedor_canonico"],
        how="left",
        validate="m:1",
    )

    dataset["feature_oferta_disponible"] = dataset["coste_min_dia_proveedor"].notna().astype(int)
    dataset["dia_semana"] = dataset["fecha_compra"].dt.dayofweek
    dataset["mes"] = dataset["fecha_compra"].dt.month
    dataset["fin_mes"] = dataset["fecha_compra"].dt.is_month_end.astype(int)
    dataset["proveedor_elegido"] = dataset["proveedor_canonico"]
    blocklist_rules = _load_blocklist(blocklist_path)
    dataset = _apply_blocklist(dataset, blocklist_rules)

    dataset["fecha_compra"] = dataset["fecha_compra"].dt.strftime("%Y-%m-%d")
    dataset["marts_run_id"] = execution_run_id
    dataset["marts_ts_utc"] = datetime.now(timezone.utc).isoformat()

    output_columns = [
        "fecha_compra",
        "albaran_id",
        "linea_id",
        "producto_canonico",
        "terminal_canonico",
        "proveedor_canonico",
        "litros",
        "precio_unitario",
        "importe_total",
        "coste_min_dia_proveedor",
        "rank_coste_dia_producto",
        "terminales_cubiertos",
        "feature_oferta_disponible",
        "dia_semana",
        "mes",
        "fin_mes",
        "blocked_by_rule",
        "block_reason",
        "proveedor_elegido",
        "source_file",
        "source_transform_run_id",
        "marts_run_id",
        "marts_ts_utc",
    ]
    output_columns = [column for column in output_columns if column in dataset.columns]
    dataset = dataset[output_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8")

    summary = {
        "status": "ok",
        "marts_run_id": execution_run_id,
        "input_fact_compras": str(fact_compras_path),
        "input_fact_ofertas": str(fact_ofertas_path),
        "input_blocklist": str(blocklist_path),
        "output_dataset": str(output_path),
        "rows_output": int(len(dataset)),
        "distinct_fechas": int(dataset["fecha_compra"].nunique()) if "fecha_compra" in dataset else 0,
        "distinct_productos": int(dataset["producto_canonico"].nunique()) if "producto_canonico" in dataset else 0,
        "distinct_proveedores_target": int(dataset["proveedor_elegido"].nunique())
        if "proveedor_elegido" in dataset
        else 0,
        "blocked_rows": int(dataset["blocked_by_rule"].sum()) if "blocked_by_rule" in dataset else 0,
        "blocked_ratio": float(dataset["blocked_by_rule"].mean()) if len(dataset) else 0.0,
        "oferta_feature_coverage_ratio": float(dataset["feature_oferta_disponible"].mean())
        if len(dataset)
        else 0.0,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye dataset_modelo_proveedor_v1.csv para clasificación de proveedor."
    )
    parser.add_argument(
        "--fact-compras",
        type=Path,
        default=Path("data/public/support/fact_compras.csv"),
        help="Input curated de compras.",
    )
    parser.add_argument(
        "--fact-ofertas",
        type=Path,
        default=Path("data/public/support/fact_ofertas_diarias.csv"),
        help="Input curated de ofertas.",
    )
    parser.add_argument(
        "--blocklist",
        type=Path,
        default=Path("config/business_blocklist_rules.csv"),
        help="Reglas básicas de bloqueo de negocio.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public/dataset_modelo_proveedor_v1.csv"),
        help="Output mart para modelado v1.",
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
        output_path=args.output,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
