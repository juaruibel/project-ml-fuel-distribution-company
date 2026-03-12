#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "public"
SUMMARY_PATH = ARTIFACTS_DIR / "public_audit_summary.json"
MANIFEST_PATH = ARTIFACTS_DIR / "public_publish_manifest.json"
NOTEBOOK_SUMMARY_PATH = ARTIFACTS_DIR / "notebooks" / "notebook_execution_summary.json"
TRAINING_SUMMARY_PATH = ARTIFACTS_DIR / "models" / "public_training_summary.json"
DATASET_PARITY_PATH = ARTIFACTS_DIR / "metrics" / "public_dataset_metric_parity.json"
TOKEN_SCAN_CONFIG_PATH = ARTIFACTS_DIR / "token_scan_config.json"
CONTRACT_PATH = ARTIFACTS_DIR / "build_contract.json"

DISALLOWED_PATHS = [
    "docs",
    "reports",
    "validations",
    "dist",
    "data/raw",
    "data/staging",
    "data/curated",
    "data/marts",
]
MANAGED_ROOTS = [
    "README.md",
    "app.py",
    ".gitignore",
    ".python-version",
    "LICENSE",
    "requirements.txt",
    "requirements-dev.txt",
    "artifacts/public",
    "config",
    "data/public",
    "models/public",
    "notebooks",
    "scripts",
    "src",
]
TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".csv", ".ipynb", ".sql"}
INLINE_CONTENT_SCAN_LIMIT_BYTES = 5_000_000
CONTENT_SCAN_EXCLUDES = {
    "artifacts/public/token_scan_config.json",
    "artifacts/public/public_audit_summary.json",
}
CONTENT_SCAN_SKIP_LARGE_BYTES = 5_000_000
FULL_SHA256_LIMIT_BYTES = 25_000_000
FINGERPRINT_CHUNK_BYTES = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the public portfolio-ready repository.")
    parser.add_argument("--strict", action="store_true", help="Fail if any audit check is not PASS.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path) -> tuple[str, str]:
    size = path.stat().st_size
    if size <= FULL_SHA256_LIMIT_BYTES:
        return sha256_file(path), "full"

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        head = handle.read(FINGERPRINT_CHUNK_BYTES)
        digest.update(head)
        if size > FINGERPRINT_CHUNK_BYTES:
            handle.seek(max(0, size - FINGERPRINT_CHUNK_BYTES))
            digest.update(handle.read(FINGERPRINT_CHUNK_BYTES))
    digest.update(str(size).encode("utf-8"))
    return digest.hexdigest(), "head_tail_1mb_plus_size"


def git_ls_files(paths: list[str]) -> list[Path]:
    command = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *paths]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        collected: list[Path] = []
        for root in paths:
            path = PROJECT_ROOT / root
            if not path.exists():
                continue
            iterator = [path] if path.is_file() else [candidate for candidate in path.rglob("*") if candidate.is_file()]
            collected.extend(iterator)
        return sorted(collected)

    files = []
    for line in completed.stdout.splitlines():
        relative = line.strip()
        if not relative:
            continue
        candidate = PROJECT_ROOT / relative
        if candidate.is_file():
            files.append(candidate)
    return sorted(files)


def iter_managed_repo_files() -> list[Path]:
    return git_ls_files(MANAGED_ROOTS)


def path_exists_in_repo(path: str) -> bool:
    return bool(git_ls_files([path]))


def build_regex(token: str) -> re.Pattern:
    escaped = re.escape(token)
    if re.fullmatch(r"[A-Za-z0-9_]+", token):
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags=re.IGNORECASE)
    return re.compile(escaped, flags=re.IGNORECASE)


def build_combined_regex(patterns: list[re.Pattern]) -> re.Pattern | None:
    if not patterns:
        return None
    return re.compile("|".join(f"(?:{pattern.pattern})" for pattern in patterns), flags=re.IGNORECASE)


def find_matching_patterns(text: str, patterns: list[re.Pattern], combined: re.Pattern | None) -> list[str]:
    if not text or combined is None or combined.search(text) is None:
        return []
    return sorted({pattern.pattern for pattern in patterns if pattern.search(text)})


def scan_file_content(file_path: Path, patterns: list[re.Pattern], combined: re.Pattern | None) -> list[str]:
    if file_path.stat().st_size > CONTENT_SCAN_SKIP_LARGE_BYTES:
        return []
    if file_path.stat().st_size <= INLINE_CONTENT_SCAN_LIMIT_BYTES:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return find_matching_patterns(content, patterns, combined)

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            matched = find_matching_patterns(line, patterns, combined)
            if matched:
                return matched
    return []


def scan_sensitive_tokens() -> list[dict]:
    payload = json.loads(TOKEN_SCAN_CONFIG_PATH.read_text(encoding="utf-8"))
    patterns = [build_regex(token) for token in payload["tokens"]]
    combined = build_combined_regex(patterns)
    hits = []
    for file_path in iter_managed_repo_files():
        relative_path = str(file_path.relative_to(PROJECT_ROOT))
        matched_path = find_matching_patterns(relative_path, patterns, combined)
        if matched_path:
            hits.append(
                {
                    "file": relative_path,
                    "scope": "path",
                    "patterns": matched_path,
                }
            )
        if file_path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if relative_path in CONTENT_SCAN_EXCLUDES:
            continue
        matched = scan_file_content(file_path, patterns, combined)
        if matched:
            hits.append(
                {
                    "file": relative_path,
                    "scope": "content",
                    "patterns": matched,
                }
            )
    return hits


def build_manifest() -> list[dict]:
    records = []
    for file_path in iter_managed_repo_files():
        fingerprint, hash_mode = fingerprint_file(file_path)
        records.append(
            {
                "path": str(file_path.relative_to(PROJECT_ROOT)),
                "size_bytes": file_path.stat().st_size,
                "sha256": fingerprint,
                "hash_mode": hash_mode,
            }
        )
    MANIFEST_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    disallowed_existing = [path for path in DISALLOWED_PATHS if path_exists_in_repo(path)]
    forbidden_extensions = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in iter_managed_repo_files()
        if path.suffix.lower() in {".xlsx", ".xls", ".pdf"}
    )
    notebooks = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    notebook_summary = json.loads(NOTEBOOK_SUMMARY_PATH.read_text(encoding="utf-8"))
    training_summary = json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))
    dataset_parity = json.loads(DATASET_PARITY_PATH.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    sensitive_hits = scan_sensitive_tokens()
    manifest = build_manifest()
    generated_models = [
        {
            "role": payload["role"],
            "model_variant": payload["model_variant"],
            "model_path": payload["model_path"],
            "metadata_path": payload["metadata_path"],
            "parity_status": payload["parity"]["status"],
        }
        for payload in training_summary["models"]
    ]

    summary = {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "public_root_preserves_git": (PROJECT_ROOT / ".git").exists(),
            "notebook_count": len(notebooks),
            "notebook_expected_count": contract["notebook_count"],
            "notebook_execution_status": notebook_summary["status"],
            "training_status": training_summary["status"],
            "dataset_metric_parity_status": dataset_parity["status"],
            "disallowed_existing": disallowed_existing,
            "forbidden_extensions": forbidden_extensions,
            "sensitive_hits": sensitive_hits,
        },
        "inventory": {
            "datasets": contract["datasets"],
            "published_notebooks": [str(path.relative_to(PROJECT_ROOT)) for path in notebooks],
            "generated_models": generated_models,
            "notebook_execution": notebook_summary,
            "dataset_metric_parity": dataset_parity,
        },
        "manifest_path": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "managed_file_count": len(manifest),
    }

    if len(notebooks) != contract["notebook_count"]:
        summary["status"] = "FAIL"
    if notebook_summary["status"] != "PASS":
        summary["status"] = "FAIL"
    if training_summary["status"] != "PASS":
        summary["status"] = "FAIL"
    if dataset_parity["status"] != "PASS":
        summary["status"] = "FAIL"
    if disallowed_existing or forbidden_extensions or sensitive_hits:
        summary["status"] = "FAIL"

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.strict and summary["status"] != "PASS":
        raise SystemExit("Public audit failed. Review artifacts/public/public_audit_summary.json")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
