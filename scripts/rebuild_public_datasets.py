#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.marts import build_dataset_modelo_day041_ablation as day041_builder
from src.etl.marts import build_dataset_modelo_day042_transport as day042_builder
from src.etl.marts import build_dataset_modelo_day043_transport as day043_builder
from src.etl.marts import build_dataset_modelo_v3_context as v3_builder


REBUILD_GROUPS = {
    "v3_context": {
        "outputs": [
            "data/public/dataset_modelo_proveedor_v3_context.csv",
        ],
    },
    "day041": {
        "outputs": [
            "data/public/day041/dataset_modelo_v2_source_quality.csv",
            "data/public/day041/dataset_modelo_v2_dispersion.csv",
            "data/public/day041/dataset_modelo_v2_competition.csv",
            "data/public/day041/dataset_modelo_v2_transport_only.csv",
        ],
    },
    "day042": {
        "outputs": [
            "data/public/day042/dataset_modelo_v2_transport_rebuilt_only.csv",
            "data/public/day042/dataset_modelo_v2_dispersion_plus_transport_rebuilt.csv",
        ],
    },
    "day043": {
        "outputs": [
            "data/public/day043/dataset_modelo_v2_transport_carry30d_only.csv",
            "data/public/day043/dataset_modelo_v2_dispersion_plus_transport_carry30d.csv",
        ],
    },
}

GROUP_ORDER = ["v3_context", "day041", "day042", "day043"]
PATH_TO_GROUP = {
    output_path: group_name
    for group_name, spec in REBUILD_GROUPS.items()
    for output_path in spec["outputs"]
}
OFERTAS_TYPED_ARCHIVE = PROJECT_ROOT / "data/public/support/ofertas_typed_full.csv.gz"
OFERTAS_TYPED_LOCAL = PROJECT_ROOT / "data/public/support/ofertas_typed.csv"
REPORT_PATHS_TO_SANITIZE = [
    PROJECT_ROOT / "artifacts/public/data_quality_v3_context.json",
    PROJECT_ROOT / "artifacts/public/data_quality_day041_ablation_matrix.json",
    PROJECT_ROOT / "artifacts/public/data_quality_day042_transport_matrix.json",
    PROJECT_ROOT / "artifacts/public/data_quality_day043_transport_matrix.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild public derived datasets that are intentionally omitted from Git."
    )
    parser.add_argument(
        "--path",
        dest="paths",
        action="append",
        default=[],
        help="Repo-relative output path to guarantee. If omitted, all derived datasets are ensured.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the target datasets already exist locally.",
    )
    return parser.parse_args()


def _resolve_requested_groups(paths: list[str]) -> set[str]:
    if not paths:
        return set(GROUP_ORDER)

    groups: set[str] = set()
    unknown: list[str] = []
    for path in paths:
        normalized = path.strip().lstrip("./")
        group_name = PATH_TO_GROUP.get(normalized)
        if group_name is None and normalized in REBUILD_GROUPS:
            group_name = normalized
        if group_name is None and (PROJECT_ROOT / normalized).exists():
            continue
        if group_name is None:
            unknown.append(path)
            continue
        groups.add(group_name)

    if unknown:
        known = sorted(list(PATH_TO_GROUP) + list(REBUILD_GROUPS))
        raise SystemExit(
            "Unknown rebuild target(s): "
            + ", ".join(sorted(unknown))
            + ". Known values: "
            + ", ".join(known)
        )
    return groups


def _group_needs_rebuild(group_name: str, force: bool) -> bool:
    if force:
        return True
    return any(not (PROJECT_ROOT / output_path).exists() for output_path in REBUILD_GROUPS[group_name]["outputs"])


def _needs_full_ofertas_typed(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        header = path.open("r", encoding="utf-8").readline().strip()
    except OSError:
        return True
    return "coste_min" not in header


def ensure_support_inputs(force: bool = False) -> None:
    if not OFERTAS_TYPED_ARCHIVE.exists():
        raise FileNotFoundError(f"Missing compressed public support file: {OFERTAS_TYPED_ARCHIVE}")
    if not force and not _needs_full_ofertas_typed(OFERTAS_TYPED_LOCAL):
        return
    OFERTAS_TYPED_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OFERTAS_TYPED_ARCHIVE, "rb") as source_handle, OFERTAS_TYPED_LOCAL.open("wb") as dest_handle:
        shutil.copyfileobj(source_handle, dest_handle)


def sanitize_generated_reports() -> None:
    for path in REPORT_PATHS_TO_SANITIZE:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        sanitized = content.replace(str(PROJECT_ROOT), ".")
        path.write_text(sanitized, encoding="utf-8")


def _run_v3_context() -> dict:
    return v3_builder.run(
        v2_input_path=PROJECT_ROOT / "data/public/dataset_modelo_proveedor_v2_candidates.csv",
        ofertas_typed_input_path=PROJECT_ROOT / "data/public/support/ofertas_typed.csv",
        raw_matrix_input_path=PROJECT_ROOT / "data/public/support/ofertas_raw_matrix_cells.csv",
        output_dataset_path=PROJECT_ROOT / "data/public/dataset_modelo_proveedor_v3_context.csv",
        quality_report_path=PROJECT_ROOT / "artifacts/public/data_quality_v3_context.json",
        data_dictionary_path=PROJECT_ROOT / "artifacts/public/data_dictionary_v3_context.md",
        cutoff_date="2028-02-21",
        min_coverage=0.80,
        run_id="public_rebuild_v3",
    )


def _run_day041() -> dict:
    return day041_builder.run(
        v2_input_path=PROJECT_ROOT / "data/public/dataset_modelo_proveedor_v2_candidates.csv",
        ofertas_typed_input_path=PROJECT_ROOT / "data/public/support/ofertas_typed.csv",
        transport_input_path=PROJECT_ROOT / "data/public/support/ofertas_transport_signals.csv",
        transport_report_path=PROJECT_ROOT / "artifacts/public/transport_parser_day041.json",
        output_dir=PROJECT_ROOT / "data/public/day041",
        quality_report_path=PROJECT_ROOT / "artifacts/public/data_quality_day041_ablation_matrix.json",
        cutoff_date="2028-02-21",
        min_coverage=0.80,
        run_id="public_rebuild_day041",
    )


def _run_day042() -> dict:
    return day042_builder.run(
        v2_input_path=PROJECT_ROOT / "data/public/dataset_modelo_proveedor_v2_candidates.csv",
        ofertas_typed_input_path=PROJECT_ROOT / "data/public/support/ofertas_typed.csv",
        transport_rebuilt_input_path=PROJECT_ROOT / "data/public/support/ofertas_transport_signals_day042.csv",
        missingness_report_path=PROJECT_ROOT / "artifacts/public/transport_missingness_day042.json",
        output_dir=PROJECT_ROOT / "data/public/day042",
        quality_report_path=PROJECT_ROOT / "artifacts/public/data_quality_day042_transport_matrix.json",
        cutoff_date="2028-02-21",
        min_coverage=0.80,
        run_id="public_rebuild_day042",
    )


def _run_day043() -> dict:
    return day043_builder.run(
        v2_input_path=PROJECT_ROOT / "data/public/dataset_modelo_proveedor_v2_candidates.csv",
        ofertas_typed_input_path=PROJECT_ROOT / "data/public/support/ofertas_typed.csv",
        transport_rebuilt_input_path=PROJECT_ROOT / "data/public/support/ofertas_transport_signals_day043.csv",
        imputation_report_path=PROJECT_ROOT / "artifacts/public/transport_imputation_day043.json",
        output_dir=PROJECT_ROOT / "data/public/day043",
        quality_report_path=PROJECT_ROOT / "artifacts/public/data_quality_day043_transport_matrix.json",
        cutoff_date="2028-02-21",
        min_coverage=0.80,
        run_id="public_rebuild_day043",
    )


RUNNERS = {
    "v3_context": _run_v3_context,
    "day041": _run_day041,
    "day042": _run_day042,
    "day043": _run_day043,
}


def ensure_public_derived_datasets(paths: list[str] | None = None, force: bool = False) -> dict:
    requested_groups = _resolve_requested_groups(paths or [])
    records: list[dict] = []
    ensure_support_inputs(force=force)

    for group_name in GROUP_ORDER:
        if group_name not in requested_groups:
            continue
        if not _group_needs_rebuild(group_name=group_name, force=force):
            records.append(
                {
                    "group": group_name,
                    "status": "SKIP",
                    "outputs": REBUILD_GROUPS[group_name]["outputs"],
                }
            )
            continue
        summary = RUNNERS[group_name]()
        records.append(
            {
                "group": group_name,
                "status": "BUILT",
                "outputs": REBUILD_GROUPS[group_name]["outputs"],
                "summary_keys": sorted(summary.keys()),
            }
        )

    sanitize_generated_reports()
    return {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    payload = ensure_public_derived_datasets(paths=args.paths, force=args.force)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
