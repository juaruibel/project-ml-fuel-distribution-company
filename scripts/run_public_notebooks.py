#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "public" / "notebooks" / "notebook_execution_summary.json"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PRIVATE_ENV_PATH_RE = re.compile(r"/Users/[^\\s\"']+?/.venv[^/]+/")
SITE_PACKAGES_PATH_RE = re.compile(r"/Users/[^\\s\"']+?/site-packages/")


def notebook_already_executed(notebook: nbformat.NotebookNode) -> bool:
    code_cells = [cell for cell in notebook.cells if cell.get("cell_type") == "code"]
    return bool(code_cells) and all(cell.get("execution_count") is not None for cell in code_cells)


def sanitize_output_text(value: str) -> str:
    sanitized = value.replace(str(PROJECT_ROOT), ".")
    sanitized = PRIVATE_ENV_PATH_RE.sub("[python-env]/", sanitized)
    sanitized = SITE_PACKAGES_PATH_RE.sub("[python-env]/site-packages/", sanitized)
    sanitized = ANSI_ESCAPE_RE.sub("", sanitized)
    return sanitized


def sanitize_output_value(value):
    if isinstance(value, str):
        return sanitize_output_text(value)
    if isinstance(value, list):
        return [sanitize_output_value(item) for item in value]
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key.startswith("image/"):
                sanitized[key] = item
            else:
                sanitized[key] = sanitize_output_value(item)
        return sanitized
    return value


def sanitize_notebook_outputs(notebook: nbformat.NotebookNode) -> bool:
    changed = False
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            for field in ["text", "traceback", "data"]:
                if field not in output:
                    continue
                original = output[field]
                sanitized = sanitize_output_value(original)
                if sanitized != original:
                    output[field] = sanitized
                    changed = True
    return changed


def main() -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []

    for notebook_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        started = time.perf_counter()
        record = {
            "notebook": str(notebook_path.relative_to(PROJECT_ROOT)),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": None,
            "status": "PASS",
            "error": "",
        }
        try:
            notebook = nbformat.read(notebook_path, as_version=4)
            if notebook_already_executed(notebook):
                sanitize_notebook_outputs(notebook)
                nbformat.write(notebook, notebook_path)
                record["status"] = "SKIP"
                records.append(record)
                print(f"SKIP {record['notebook']} (already executed)")
                continue
            processor = ExecutePreprocessor(timeout=1800, kernel_name="python3")
            processor.preprocess(notebook, {"metadata": {"path": str(notebook_path.parent)}})
            sanitize_notebook_outputs(notebook)
            nbformat.write(notebook, notebook_path)
        except Exception as exc:  # noqa: BLE001
            record["status"] = "FAIL"
            record["error"] = str(exc)
            failures.append(record["notebook"])
        record["duration_seconds"] = round(time.perf_counter() - started, 3)
        records.append(record)
        print(f"{record['status']} {record['notebook']} ({record['duration_seconds']}s)")

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "notebook_count": len(records),
        "failures": failures,
        "records": records,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if failures:
        raise SystemExit("Notebook execution failed. Review artifacts/public/notebooks/notebook_execution_summary.json")


if __name__ == "__main__":
    main()
