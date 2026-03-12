from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_run_id(ts: datetime | None = None) -> str:
    """Genera identificador UTC estable para trazabilidad."""
    now = ts or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso(ts: datetime | None = None) -> str:
    """Devuelve timestamp UTC en ISO 8601."""
    now = ts or datetime.now(timezone.utc)
    return now.isoformat()


def ensure_parent(path: Path) -> Path:
    """Crea directorio padre si no existe y devuelve la ruta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Escribe JSON UTF-8 con indentación estable."""
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_postinference_audit_paths(
    report_root: Path,
    raw_output_path: Path,
    run_id: str,
    mode: str,
    albaran_policy: str = "none",
) -> dict[str, Path]:
    """Construye rutas canónicas de auditoría post-inferencia."""
    run_date_utc = datetime.now(timezone.utc).strftime("%Y%m%d")
    report_day_dir = report_root / "postinferencia" / "audits" / run_date_utc
    report_day_dir.mkdir(parents=True, exist_ok=True)

    stem = raw_output_path.stem
    mode_token = _normalize_token(mode)
    policy_token = _normalize_token(albaran_policy or "none")
    suffix = mode_token if policy_token == "none" else f"{mode_token}_{policy_token}"

    return {
        "detail": report_day_dir / f"{stem}_detalle_{suffix}_{run_id}.csv",
        "resumen_evento": report_day_dir / f"{stem}_resumen_evento_{suffix}_{run_id}.csv",
        "resumen_albaran": report_day_dir / f"{stem}_resumen_albaran_{suffix}_{run_id}.csv",
        "summary": report_day_dir / f"{stem}_summary_{suffix}_{run_id}.json",
    }
