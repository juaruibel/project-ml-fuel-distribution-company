from __future__ import annotations

import pandas as pd


def parse_numeric_series_locale(series: pd.Series) -> pd.Series:
    """
    Parsea valores numéricos soportando punto decimal y fallback europeo.

    Estrategia:
    1) intentar parseo directo (`pd.to_numeric`) sin modificar valores válidos;
    2) solo para no parseados, aplicar fallback de coma decimal (`1,25`)
       y formato europeo con miles (`1.234,56`);
    3) conservar `NaN` cuando el valor no sea convertible.
    """
    raw = series.astype("string").str.strip()
    blank_mask = raw.isna() | raw.eq("") | raw.str.lower().isin(["nan", "none", "null"])
    normalized = raw.mask(blank_mask, pd.NA)

    parsed = pd.to_numeric(normalized, errors="coerce")
    unresolved_mask = parsed.isna() & normalized.notna()
    if not unresolved_mask.any():
        return parsed

    fallback = normalized.loc[unresolved_mask].str.replace(r"\s+", "", regex=True)
    has_comma = fallback.str.contains(",", na=False)
    has_dot = fallback.str.contains(r"\.", regex=True, na=False)

    both_mask = has_comma & has_dot
    comma_only_mask = has_comma & ~has_dot

    fallback.loc[both_mask] = (
        fallback.loc[both_mask]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    fallback.loc[comma_only_mask] = fallback.loc[comma_only_mask].str.replace(",", ".", regex=False)

    parsed_fallback = pd.to_numeric(fallback, errors="coerce")
    parsed.loc[unresolved_mask] = parsed_fallback
    return parsed
