# HELPERS DEL ETL

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_RULE_COLUMNS = (
    "rule_id",
    "active",
    "fecha_inicio",
    "fecha_fin",
    "producto_canonico",
    "terminal_canonico",
    "proveedor_canonico",
    "motivo",
)


def clean_text(value: object) -> str:
    """
    Normaliza un valor a texto limpio para comparaciones.

    - Convierte `None` y `NaN` en cadena vacía.
    - Convierte el resto a `str` y elimina espacios laterales.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def matches_with_wildcard(value: str, rule_value: str) -> bool:
    """
    Evalúa si `value` cumple el patrón de regla.

    La regla acepta comodín cuando `rule_value` es `*` o vacío.
    En caso contrario exige igualdad exacta tras limpieza de texto.
    """
    token = clean_text(rule_value)
    if token in {"", "*"}:
        return True
    return clean_text(value) == token


def load_blocklist_rules(blocklist_path: Path) -> pd.DataFrame:
    """
    Carga y prepara el CSV de reglas activas de blocklist.

    Proceso:
    - Si el archivo no existe, devuelve dataframe vacío con esquema esperado.
    - Lee CSV como texto y filtra solo reglas activas.
    - Parsea fechas (`fecha_inicio`, `fecha_fin`) y normaliza columnas clave.

    Retorna un dataframe listo para aplicar en scoring post-inferencia.
    """
    if not blocklist_path.exists():
        return pd.DataFrame(columns=list(REQUIRED_RULE_COLUMNS))

    rules = pd.read_csv(blocklist_path, dtype=str, keep_default_na=False)
    if rules.empty:
        return rules

    rules["active"] = rules.get("active", "0").astype(str).str.strip()
    rules = rules[rules["active"].isin(["1", "true", "TRUE", "yes", "YES"])].copy()
    if rules.empty:
        return rules

    rules["fecha_inicio"] = pd.to_datetime(rules.get("fecha_inicio", ""), errors="coerce")
    rules["fecha_fin"] = pd.to_datetime(rules.get("fecha_fin", ""), errors="coerce")

    for column in REQUIRED_RULE_COLUMNS:
        if column not in rules.columns:
            rules[column] = ""
    for column in ["rule_id", "producto_canonico", "terminal_canonico", "proveedor_canonico", "motivo"]:
        rules[column] = rules[column].astype(str).str.strip()

    return rules.reset_index(drop=True)


def apply_blocklist_candidates(dataset: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas deterministas sobre dataset a grano candidato.

    Requiere en `dataset` al menos:
    - `fecha_evento`, `producto_canonico`, `terminal_compra`, `proveedor_candidato`.

    Añade/reescribe:
    - `blocked_by_rule_candidate` (0/1)
    - `block_reason_candidate` (`rule_id:motivo`)

    No altera scores ni ranking del modelo; solo anota bloqueo para modo shadow/assist.
    """
    output = dataset.copy()
    output["blocked_by_rule_candidate"] = 0
    output["block_reason_candidate"] = ""
    if rules.empty or output.empty:
        return output

    fechas = pd.to_datetime(output["fecha_evento"], errors="coerce")
    for _, rule in rules.iterrows():
        provider_token = clean_text(rule.get("proveedor_canonico", ""))
        if not provider_token:
            continue

        provider_mask = output["proveedor_candidato"].astype(str).str.strip().eq(provider_token)
        product_mask = output["producto_canonico"].astype(str).map(
            lambda value: matches_with_wildcard(value, rule.get("producto_canonico", "*"))
        )
        terminal_mask = output["terminal_compra"].astype(str).map(
            lambda value: matches_with_wildcard(value, rule.get("terminal_canonico", "*"))
        )

        date_mask = pd.Series([True] * len(output), index=output.index)
        start = rule.get("fecha_inicio")
        end = rule.get("fecha_fin")
        if pd.notna(start):
            date_mask = date_mask & (fechas >= start)
        if pd.notna(end):
            date_mask = date_mask & (fechas <= end)

        final_mask = provider_mask & product_mask & terminal_mask & date_mask
        if not final_mask.any():
            continue

        reason = clean_text(rule.get("motivo", "")) or "blocked_by_business_rule"
        rule_id = clean_text(rule.get("rule_id", "RULE"))
        output.loc[final_mask, "blocked_by_rule_candidate"] = 1
        output.loc[final_mask, "block_reason_candidate"] = f"{rule_id}:{reason}"
    return output
