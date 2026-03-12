from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_excluded(file_path: Path, excluded_prefixes: list[str]) -> bool:
    return any(file_path.name.startswith(prefix) for prefix in excluded_prefixes)


def _discover_ofertas(raw_root: Path, ofertas_cfg: dict) -> list[dict]:
    if not ofertas_cfg.get("enabled", False):
        return []
    globs = ofertas_cfg.get("globs", [])
    excluded_prefixes = ofertas_cfg.get("exclude_name_prefixes", [])
    discovered: list[dict] = []
    for glob_pattern in globs:
        for file_path in raw_root.glob(glob_pattern):
            if not file_path.is_file():
                continue
            if _is_excluded(file_path, excluded_prefixes):
                continue
            discovered.append(
                {
                    "kind": "oferta_file",
                    "path": str(file_path),
                    "suffix": file_path.suffix.lower(),
                    "size_bytes": file_path.stat().st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
    discovered.sort(key=lambda item: item["path"])
    return discovered


def _discover_compras(repo_root: Path, compras_cfg: dict) -> list[dict]:
    discovered: list[dict] = []
    primary = compras_cfg.get("primary", {})
    primary_path = repo_root / primary.get("path", "")
    discovered.append(
        {
            "kind": "compras_primary",
            "enabled": bool(primary.get("enabled", False)),
            "path": str(primary_path),
            "exists": primary_path.exists(),
            "source_format": primary.get("kind", "unknown"),
            "target_staging": primary.get("target_staging"),
        }
    )
    for item in compras_cfg.get("historical", []):
        historical_path = repo_root / item.get("path", "")
        discovered.append(
            {
                "kind": "compras_historical",
                "enabled": bool(item.get("enabled", False)),
                "path": str(historical_path),
                "exists": historical_path.exists(),
                "source_format": item.get("kind", "unknown"),
                "mode": item.get("mode", "unknown"),
            }
        )
    return discovered


def run(config_path: Path, output_path: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    cfg = _load_config(config_path)
    raw_root = repo_root / cfg["raw_root"]
    ofertas_cfg = cfg["sources"]["ofertas"]
    compras_cfg = cfg["sources"]["compras"]
    ofertas = _discover_ofertas(raw_root, ofertas_cfg)
    compras = _discover_compras(repo_root, compras_cfg)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "raw_root": str(raw_root),
        "summary": {
            "ofertas_files": len(ofertas),
            "compras_sources": len(compras),
            "compras_primary_ready": any(
                item["kind"] == "compras_primary"
                and item["enabled"]
                and item["exists"]
                for item in compras
            ),
        },
        "ofertas": ofertas,
        "compras": compras,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight de fuentes ETL para comparativas y compras."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/etl_sources.json"),
        help="Ruta del archivo de configuración de fuentes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public/support/source_manifest_preflight.json"),
        help="Ruta de salida del manifiesto de descubrimiento.",
    )
    args = parser.parse_args()
    manifest = run(config_path=args.config, output_path=args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "ofertas_files": manifest["summary"]["ofertas_files"],
                "compras_primary_ready": manifest["summary"]["compras_primary_ready"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
