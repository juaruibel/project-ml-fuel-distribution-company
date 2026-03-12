#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nbformat
import numpy as np
import pandas as pd
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


DEFAULT_MODE = "public_pseudonymous"
DEFAULT_PUBLIC_NAME = "Fuel Distribution Company, S.L."
DEFAULT_SEED = 42
DEFAULT_DATE_SHIFT_DAYS = 1461

TEXT_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sql",
    ".txt",
    ".yaml",
    ".yml",
}
SOURCE_TEXT_SUFFIXES = TEXT_SUFFIXES | {".toml"}
EXCLUDED_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints"}
METRIC_PARITY_KEYS = [
    "rows_v1",
    "rows_v2",
    "events_v2",
    "v1_dummy_accuracy",
    "v1_knn_accuracy",
    "v2_lr_accuracy",
    "v2_lr_balanced_accuracy",
    "v2_lr_f1_pos",
    "v2_lr_top1_hit",
    "v2_lr_top2_hit",
    "v2_cheapest_top1_hit",
    "v2_cheapest_top2_hit",
]
PROVIDER_COLUMNS = {
    "proveedor_canonico",
    "proveedor_elegido",
    "proveedor_candidato",
    "proveedor_elegido_real",
    "proveedor_real",
    "winner_provider",
    "top1_provider_baseline",
    "top1_provider_pure_champion",
    "provider_selected",
    "provider_recommended",
}
PRODUCT_COLUMNS = {"producto_canonico"}
TERMINAL_COLUMNS = {"terminal_canonico", "terminal_compra"}
RULE_COLUMNS = {"block_reason", "block_reason_candidate"}
ID_PREFIX_MAP = {
    "event_id": "EVENT",
    "albaran_id": "ALBARAN",
    "linea_id": "LINE",
    "source_run_id": "RUN",
    "transform_run_id": "RUN",
    "source_transform_run_id": "RUN",
    "marts_run_id": "RUN",
    "v2_run_id": "RUN",
    "run_id": "RUN",
    "rebuild_run_id": "RUN",
    "invalidated_run_id": "RUN",
    "record_hash": "HASH",
    "reconciliation_key": "KEY",
}
DATE_COLUMNS = {
    "fecha",
    "fecha_compra",
    "fecha_despacho",
    "fecha_evento",
    "fecha_oferta",
    "fecha_raw",
    "source_date_for_transport",
}
TIMESTAMP_COLUMNS = {
    "ingestion_ts_utc",
    "marts_ts_utc",
    "source_ingestion_ts_utc",
    "transform_ts_utc",
    "ts_utc",
    "timestamp_utc",
    "v2_ts_utc",
}


@dataclass(frozen=True)
class ReplacementRule:
    pattern: str
    replacement: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and audit the public pseudonymous bootcamp repo in ../proyecto-ml-public.",
    )
    parser.add_argument(
        "--private-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to the private source repository.",
    )
    parser.add_argument(
        "--public-root",
        default=None,
        help="Path to the canonical public repository. Defaults to ../proyecto-ml-public.",
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=[DEFAULT_MODE],
        help="Publication mode. Only public_pseudonymous is supported in the portfolio-ready flow.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed used for stable hashes.",
    )
    parser.add_argument(
        "--public-name",
        default=DEFAULT_PUBLIC_NAME,
        help="Generic company label used in public artifacts.",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Fail if the final public audit is not PASS.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Do not fail on public audit failure.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def load_manifest(private_root: Path) -> dict:
    manifest_path = private_root / "config" / "publish_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_public_root(private_root: Path, manifest: dict, arg_value: str | None) -> Path:
    raw_value = arg_value or manifest["default_public_root"]
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (private_root / raw_value).resolve()


def validate_public_root(private_root: Path, public_root: Path, manifest: dict) -> None:
    private_root = private_root.resolve()
    if public_root == private_root:
        raise ValueError("El public_root no puede ser el mismo repo privado.")
    if str(public_root).startswith(str(private_root)):
        relative = public_root.relative_to(private_root)
        forbidden_roots = set(manifest.get("forbidden_output_roots", []))
        if relative.parts and relative.parts[0] in forbidden_roots:
            raise ValueError(
                "El flujo público ya no permite publicar dentro del repo privado "
                f"({relative.parts[0]} está prohibido)."
            )
    if not public_root.exists():
        raise FileNotFoundError(f"No existe public_root: {public_root}")
    if not (public_root / ".git").exists():
        raise FileNotFoundError(
            "El repo público canónico debe existir ya y conservar su .git. "
            f"No se encontró {public_root / '.git'}"
        )


def clean_public_root(public_root: Path, manifest: dict) -> None:
    for relative in manifest.get("managed_paths", []):
        remove_path(public_root / relative)
    for relative in manifest.get("cleanup_only_paths", []):
        remove_path(public_root / relative)


def stable_public_hash(value: object, prefix: str, seed: int) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip()
    if not text:
        return value
    digest = hashlib.sha256(f"{seed}|{prefix}|{text}".encode("utf-8")).hexdigest()[:12].upper()
    return f"{prefix}_{digest}"


def build_alias_map(values: Iterable[object], prefix: str) -> dict[str, str]:
    normalized = sorted(
        {
            str(value).strip()
            for value in values
            if pd.notna(value) and str(value).strip()
        }
    )
    return {value: f"{prefix}_{index + 1:03d}" for index, value in enumerate(normalized)}


def apply_alias_map(series: pd.Series, alias_map: dict[str, str]) -> pd.Series:
    return series.map(lambda value: alias_map.get(str(value).strip(), value) if pd.notna(value) else value)


def shift_date_column(frame: pd.DataFrame, column: str, days: int) -> None:
    if column not in frame.columns:
        return
    parsed = pd.to_datetime(frame[column], errors="coerce")
    mask = parsed.notna()
    shifted = parsed + pd.Timedelta(days=days)
    frame[column] = frame[column].astype(object)
    frame.loc[mask, column] = shifted.loc[mask].dt.strftime("%Y-%m-%d")


def shift_timestamp_column(frame: pd.DataFrame, column: str, days: int) -> None:
    if column not in frame.columns:
        return
    parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
    mask = parsed.notna()
    shifted = parsed + pd.Timedelta(days=days)
    frame[column] = frame[column].astype(object)
    frame.loc[mask, column] = shifted.loc[mask].dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def shift_iso_dates_in_text(content: str, days: int) -> str:
    def replace(match: re.Match[str]) -> str:
        text = match.group(1)
        try:
            shifted = pd.Timestamp(text) + pd.Timedelta(days=days)
        except Exception:
            return text
        return shifted.strftime("%Y-%m-%d")

    return re.sub(r"\b(20\d{2}-\d{2}-\d{2})\b", replace, content)


def load_core_private_frames(private_root: Path) -> dict[str, pd.DataFrame]:
    dataset_map = {
        "v1": "data/public/dataset_modelo_proveedor_v1.csv",
        "v2": "data/public/dataset_modelo_proveedor_v2_candidates.csv",
        "v2_excluded": "data/public/dataset_modelo_proveedor_v2_excluded_events.csv",
        "v3_context": "data/public/dataset_modelo_proveedor_v3_context.csv",
        "day041_source_quality": "data/public/day041/dataset_modelo_v2_source_quality.csv",
        "day041_dispersion": "data/public/day041/dataset_modelo_v2_dispersion.csv",
        "day041_competition": "data/public/day041/dataset_modelo_v2_competition.csv",
        "day041_transport_only": "data/public/day041/dataset_modelo_v2_transport_only.csv",
        "day042_dispersion_plus_transport_rebuilt": "data/public/day042/dataset_modelo_v2_dispersion_plus_transport_rebuilt.csv",
        "day042_transport_rebuilt_only": "data/public/day042/dataset_modelo_v2_transport_rebuilt_only.csv",
        "day043_dispersion_plus_transport_carry30d": "data/public/day043/dataset_modelo_v2_dispersion_plus_transport_carry30d.csv",
        "day043_transport_carry30d_only": "data/public/day043/dataset_modelo_v2_transport_carry30d_only.csv",
        "inference_example": "data/public/inference_inputs/example_real_day_2024-05-28.csv",
    }
    return {
        alias: pd.read_csv(private_root / relative, keep_default_na=False)
        for alias, relative in dataset_map.items()
    }


def build_public_alias_maps(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    providers: list[object] = []
    products: list[object] = []
    terminals: list[object] = []
    rules: list[object] = []

    for frame in frames.values():
        for column in frame.columns:
            if column in PROVIDER_COLUMNS:
                providers.extend(frame[column].tolist())
            elif column in PRODUCT_COLUMNS:
                products.extend(frame[column].tolist())
            elif column in TERMINAL_COLUMNS:
                terminals.extend(frame[column].tolist())
            elif column in RULE_COLUMNS:
                rules.extend(frame[column].tolist())

    return {
        "provider": build_alias_map(providers, "SUPPLIER"),
        "product": build_alias_map(products, "PRODUCT"),
        "terminal": build_alias_map(terminals, "TERMINAL"),
        "rule": build_alias_map(rules, "RULE"),
    }


def build_provider_alias_map_from_config(private_root: Path, provider_alias_map: dict[str, str]) -> dict[str, str]:
    mapping_path = private_root / "config" / "proveedores_mapping_v1.csv"
    if not mapping_path.exists():
        return {}

    raw_alias: dict[str, str] = {}
    for index, line in enumerate(mapping_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if index == 0:
            continue
        line = line.strip()
        if not line or "," not in line:
            continue
        raw_value, canon_value = line.rsplit(",", 1)
        raw_value = raw_value.strip()
        canon_value = canon_value.strip()
        if not raw_value or not canon_value:
            continue
        if canon_value in provider_alias_map:
            raw_alias[raw_value] = provider_alias_map[canon_value]
    return raw_alias


def build_replacement_rules(
    private_root: Path,
    public_name: str,
    alias_maps: dict[str, dict[str, str]],
    date_shift_days: int,
) -> list[ReplacementRule]:
    rules: list[ReplacementRule] = [
        ReplacementRule(re.escape(str(private_root.resolve())), "."),
        ReplacementRule(r"SUPPLIER_012, S\.L\.?", public_name),
        ReplacementRule(r"SUPPLIER_012, S\.L\.?", public_name.upper()),
        ReplacementRule(r"Project Author", "Project Author"),
        ReplacementRule(r"project_author", "project_author"),
        ReplacementRule(r"proyecto-ml-source", "proyecto-ml-source"),
        ReplacementRule(r"proyecto-ml-public", "proyecto-ml-public"),
        ReplacementRule(r"SUPPLIER_DAILY_COMPARISON", "SUPPLIER_DAILY_COMPARISON"),
        ReplacementRule(r"artifacts/public/", "artifacts/public/"),
        ReplacementRule(r"data/public/", "data/public/"),
        ReplacementRule(r"data/public/support/", "data/public/support/"),
        ReplacementRule(r"data/public/support/", "data/public/support/"),
        ReplacementRule(
            re.escape('REPORTS_ROOT = PROJECT_ROOT / "artifacts" / "public"'),
            'REPORTS_ROOT = PROJECT_ROOT / "artifacts" / "public"',
        ),
        ReplacementRule(
            re.escape("REPORTS_ROOT = PROJECT_ROOT / 'artifacts' / 'public'"),
            "REPORTS_ROOT = PROJECT_ROOT / 'artifacts' / 'public'",
        ),
        ReplacementRule(
            re.escape('PROJECT_ROOT / "artifacts" / "public" / "metrics"'),
            'PROJECT_ROOT / "artifacts" / "public" / "metrics"',
        ),
        ReplacementRule(
            re.escape("PROJECT_ROOT / 'artifacts' / 'public' / 'metrics'"),
            "PROJECT_ROOT / 'artifacts' / 'public' / 'metrics'",
        ),
        ReplacementRule(
            re.escape('if (CWD / "artifacts" / "public").exists():'),
            'if (CWD / "artifacts" / "public").exists():',
        ),
        ReplacementRule(
            re.escape('elif (CWD.parent / "artifacts" / "public").exists():'),
            'elif (CWD.parent / "artifacts" / "public").exists():',
        ),
        ReplacementRule(r"models/public/baseline/", "models/public/baseline/"),
        ReplacementRule(r"models/public/champion_pure/", "models/public/champion_pure/"),
        ReplacementRule(r"db_views_serving\.sql", "src/sql/public_src/sql/public_db_views_serving.sql"),
        ReplacementRule(r"\bAE 480\b", "ALBARAN_FOCUS"),
    ]

    raw_provider_alias_map = build_provider_alias_map_from_config(private_root, alias_maps["provider"])
    for group_name in ["provider", "product", "terminal", "rule"]:
        items = list(alias_maps[group_name].items())
        if group_name == "provider":
            items += list(raw_provider_alias_map.items())

        for original, alias in sorted(items, key=lambda item: len(item[0]), reverse=True):
            escaped = re.escape(original)
            if re.fullmatch(r"[A-Za-z0-9_]+", original):
                pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
            else:
                pattern = escaped
            rules.append(ReplacementRule(pattern=pattern, replacement=alias))

            if group_name == "product":
                compact = re.sub(r"[^A-Za-z0-9]+", "", original.upper())
                if compact and compact != original:
                    rules.append(
                        ReplacementRule(
                            pattern=rf"(?<![A-Za-z0-9]){re.escape(compact)}(?![A-Za-z0-9])",
                            replacement=alias.replace("-", "_"),
                        )
                    )

    sentinel = f"__DATE_SHIFT_{date_shift_days}__"
    rules.append(ReplacementRule(r"\{\{DATE_SHIFT_DAYS\}\}", str(date_shift_days)))
    rules.append(ReplacementRule(r"\{\{DATE_SHIFT_SENTINEL\}\}", sentinel))
    return rules


def sanitize_text(content: str, rules: list[ReplacementRule], date_shift_days: int) -> str:
    sanitized = content
    for rule in rules:
        sanitized = re.sub(rule.pattern, rule.replacement, sanitized, flags=re.IGNORECASE)
    sanitized = shift_iso_dates_in_text(sanitized, date_shift_days)
    return sanitized.replace(str(Path.cwd().resolve()), ".")


def sanitize_relative_path(relative: Path, rules: list[ReplacementRule], date_shift_days: int) -> Path:
    sanitized = sanitize_text(relative.as_posix(), rules=rules, date_shift_days=date_shift_days)
    return Path(sanitized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def topk_hit_by_event(df_eval: pd.DataFrame, score_col: str, k: int = 1) -> float:
    ranked = df_eval.sort_values(["event_id", score_col], ascending=[True, False], kind="mergesort")
    topk = ranked.groupby("event_id", sort=False).head(k)
    if topk.empty:
        return float("nan")
    return float(topk.groupby("event_id")["target_elegido"].max().mean())


def compute_snapshot_from_paths(v1_path: Path, v2_path: Path, seed: int) -> dict:
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    v1 = pd.read_csv(v1_path)
    v2 = pd.read_csv(v2_path)

    v1["fecha_compra"] = pd.to_datetime(v1["fecha_compra"], errors="coerce")
    v1 = v1.dropna(subset=["fecha_compra", "proveedor_elegido"]).sort_values("fecha_compra").reset_index(drop=True)
    split_idx = int(len(v1) * 0.8)
    train_v1 = v1.iloc[:split_idx].copy()
    test_v1 = v1.iloc[split_idx:].copy()

    feature_cols_v1 = [
        "coste_min_dia_proveedor",
        "rank_coste_dia_producto",
        "terminales_cubiertos",
        "dia_semana",
        "mes",
        "fin_mes",
        "blocked_by_rule",
    ]
    train_v1 = train_v1.dropna(subset=feature_cols_v1 + ["proveedor_elegido"])
    test_v1 = test_v1.dropna(subset=feature_cols_v1 + ["proveedor_elegido"])
    X_train_v1 = train_v1[feature_cols_v1]
    X_test_v1 = test_v1[feature_cols_v1]
    y_train_v1 = train_v1["proveedor_elegido"]
    y_test_v1 = test_v1["proveedor_elegido"]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_v1, y_train_v1)
    knn = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])
    knn.fit(X_train_v1, y_train_v1)

    feature_cols_num = [
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
        "dia_semana",
        "mes",
        "fin_mes",
    ]
    feature_cols_cat = ["proveedor_candidato", "producto_canonico", "terminal_compra"]
    target_col = "target_elegido"

    v2["fecha_evento"] = pd.to_datetime(v2["fecha_evento"], errors="coerce")
    v2 = v2.dropna(subset=["fecha_evento"])
    dates = sorted(v2["fecha_evento"].unique())
    cut_idx = max(1, int(len(dates) * 0.8))
    cutoff_date = dates[cut_idx - 1]
    train_v2 = v2[v2["fecha_evento"] <= cutoff_date].copy()
    test_v2 = v2[v2["fecha_evento"] > cutoff_date].copy()

    X_train_v2 = pd.get_dummies(train_v2[feature_cols_num + feature_cols_cat], drop_first=False)
    X_test_v2 = pd.get_dummies(test_v2[feature_cols_num + feature_cols_cat], drop_first=False)
    X_test_v2 = X_test_v2.reindex(columns=X_train_v2.columns, fill_value=0)
    y_train_v2 = train_v2[target_col].astype(int)
    y_test_v2 = test_v2[target_col].astype(int)

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=1.0, max_iter=4000, random_state=seed, class_weight="balanced")),
        ]
    )
    lr.fit(X_train_v2, y_train_v2)
    y_pred_v2 = lr.predict(X_test_v2)
    y_score_v2 = lr.predict_proba(X_test_v2)[:, 1]

    eval_df = test_v2[["event_id", "coste_min_dia_proveedor", "target_elegido"]].copy()
    eval_df["score_model"] = y_score_v2
    eval_df["target_elegido"] = eval_df["target_elegido"].astype(int)

    cheapest = eval_df.sort_values(["event_id", "coste_min_dia_proveedor"], ascending=[True, True])
    cheapest["score_cheapest"] = -cheapest["coste_min_dia_proveedor"].astype(float)

    return {
        "seed": seed,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows_v1": int(len(v1)),
        "rows_v2": int(len(v2)),
        "events_v2": int(v2["event_id"].nunique()),
        "v1_dummy_accuracy": float(dummy.score(X_test_v1, y_test_v1)),
        "v1_knn_accuracy": float(knn.score(X_test_v1, y_test_v1)),
        "v2_lr_accuracy": float(accuracy_score(y_test_v2, y_pred_v2)),
        "v2_lr_balanced_accuracy": float(balanced_accuracy_score(y_test_v2, y_pred_v2)),
        "v2_lr_f1_pos": float(f1_score(y_test_v2, y_pred_v2, pos_label=1)),
        "v2_lr_top1_hit": topk_hit_by_event(eval_df, "score_model", k=1),
        "v2_lr_top2_hit": topk_hit_by_event(eval_df, "score_model", k=2),
        "v2_cheapest_top1_hit": topk_hit_by_event(cheapest, "score_cheapest", k=1),
        "v2_cheapest_top2_hit": topk_hit_by_event(cheapest, "score_cheapest", k=2),
    }


def compare_metric_snapshots(reference: dict, candidate: dict) -> dict:
    checks: list[dict[str, object]] = []
    status = "PASS"
    for key in METRIC_PARITY_KEYS:
        ref_value = pd.to_numeric(reference.get(key), errors="coerce")
        cand_value = pd.to_numeric(candidate.get(key), errors="coerce")
        matches = bool(np.isclose(ref_value, cand_value, atol=1e-12, rtol=0.0))
        if not matches:
            status = "FAIL"
        checks.append(
            {
                "metric": key,
                "reference": None if pd.isna(ref_value) else float(ref_value),
                "candidate": None if pd.isna(cand_value) else float(cand_value),
                "matches": matches,
            }
        )
    return {"status": status, "metric_checks": checks}


def build_requirements_txt() -> str:
    return textwrap.dedent(
        """
        pandas>=2.2,<2.3
        numpy>=1.26,<2.0
        openpyxl>=3.1,<3.2
        xlrd>=2.0,<3.0
        pyarrow>=16,<19
        scikit-learn>=1.5,<1.8
        jupyter>=1.0,<2.0
        matplotlib>=3.10,<3.11
        seaborn>=0.13,<0.14
        streamlit>=1.39,<2.0
        imbalanced-learn==0.14.1
        catboost>=1.2,<1.3
        lightgbm>=4.5,<4.7
        xgboost>=2.1,<2.2
        shap>=0.46,<0.47
        PyYAML>=6.0,<7.0
        nbformat>=5.10,<6.0
        nbconvert>=7.16,<8.0
        """
    ).strip() + "\n"


def build_requirements_dev_txt() -> str:
    return textwrap.dedent(
        """
        -r requirements.txt
        ipykernel>=6.29,<7
        pytest>=8.2,<9
        """
    ).strip() + "\n"


def build_public_gitignore() -> str:
    return textwrap.dedent(
        """
        __pycache__/
        *.py[cod]
        .pytest_cache/
        .mypy_cache/
        .ipynb_checkpoints/
        .venv/
        .DS_Store
        .idea/
        .vscode/

        artifacts/public/sql/*.db
        data/public/inference_outputs/**
        !data/public/inference_outputs/.gitkeep
        """
    ).lstrip()


def iter_files(source_root: Path) -> Iterable[Path]:
    if source_root.is_file():
        yield source_root
        return
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        yield path


def copy_text_tree(
    source_root: Path,
    dest_root: Path,
    rules: list[ReplacementRule],
    date_shift_days: int,
    extra_replacements: dict[str, str] | None = None,
) -> list[Path]:
    written: list[Path] = []
    for source_path in iter_files(source_root):
        relative = source_path.relative_to(source_root)
        dest_path = dest_root / relative
        if source_path.suffix.lower() in SOURCE_TEXT_SUFFIXES:
            content = source_path.read_text(encoding="utf-8", errors="ignore")
            content = sanitize_text(content, rules=rules, date_shift_days=date_shift_days)
            for old, new in (extra_replacements or {}).items():
                content = content.replace(old, new)
            write_text(dest_path, content)
        else:
            ensure_parent(dest_path)
            shutil.copy2(source_path, dest_path)
        written.append(dest_path)
    return written


def sanitize_frame(
    frame: pd.DataFrame,
    alias_maps: dict[str, dict[str, str]],
    seed: int,
    date_shift_days: int,
) -> pd.DataFrame:
    working = frame.copy()

    for column in working.columns:
        if column in PROVIDER_COLUMNS:
            working[column] = apply_alias_map(working[column], alias_maps["provider"])
        elif column in PRODUCT_COLUMNS:
            working[column] = apply_alias_map(working[column], alias_maps["product"])
        elif column in TERMINAL_COLUMNS:
            working[column] = apply_alias_map(working[column], alias_maps["terminal"])
        elif column in RULE_COLUMNS:
            working[column] = apply_alias_map(working[column], alias_maps["rule"])
        elif column in ID_PREFIX_MAP:
            prefix = ID_PREFIX_MAP[column]
            working[column] = working[column].map(lambda value, p=prefix: stable_public_hash(value, p, seed))
        elif column == "source_file":
            working[column] = working[column].map(lambda value: stable_public_hash(value, "SOURCE", seed))
        elif column in DATE_COLUMNS:
            shift_date_column(working, column, date_shift_days)
        elif column in TIMESTAMP_COLUMNS:
            shift_timestamp_column(working, column, date_shift_days)

    return working


def build_support_frames(private_root: Path, alias_maps: dict[str, dict[str, str]], seed: int) -> dict[str, pd.DataFrame]:
    typed_dates = pd.read_csv(
        private_root / "data/public/support/ofertas_typed.csv",
        usecols=["source_file", "fecha_oferta"],
        keep_default_na=False,
    )
    raw_mincost_dates = pd.read_csv(
        private_root / "data/public/support/ofertas_raw_mincost.csv",
        usecols=["source_file", "fecha_raw"],
        keep_default_na=False,
    )
    transport_day042 = pd.read_csv(
        private_root / "data/public/support/ofertas_transport_signals_day042.csv",
        keep_default_na=False,
    )

    typed_dates["source_file"] = typed_dates["source_file"].map(lambda value: stable_public_hash(value, "SOURCE", seed))
    raw_mincost_dates["source_file"] = raw_mincost_dates["source_file"].map(
        lambda value: stable_public_hash(value, "SOURCE", seed)
    )
    shift_date_column(typed_dates, "fecha_oferta", DEFAULT_DATE_SHIFT_DAYS)
    shift_date_column(raw_mincost_dates, "fecha_raw", DEFAULT_DATE_SHIFT_DAYS)

    transport_day042 = sanitize_frame(
        transport_day042,
        alias_maps=alias_maps,
        seed=seed,
        date_shift_days=DEFAULT_DATE_SHIFT_DAYS,
    )

    return {
        "ofertas_typed.csv": typed_dates,
        "ofertas_raw_mincost.csv": raw_mincost_dates,
        "ofertas_transport_signals_day042.csv": transport_day042,
    }


def write_public_datasets(
    private_root: Path,
    public_root: Path,
    alias_maps: dict[str, dict[str, str]],
    seed: int,
) -> dict[str, Path]:
    source_to_dest = {
        "data/public/dataset_modelo_proveedor_v1.csv": "data/public/dataset_modelo_proveedor_v1.csv",
        "data/public/dataset_modelo_proveedor_v2_candidates.csv": "data/public/dataset_modelo_proveedor_v2_candidates.csv",
        "data/public/dataset_modelo_proveedor_v2_excluded_events.csv": "data/public/dataset_modelo_proveedor_v2_excluded_events.csv",
        "data/public/dataset_modelo_proveedor_v3_context.csv": "data/public/dataset_modelo_proveedor_v3_context.csv",
        "data/public/day041/dataset_modelo_v2_source_quality.csv": "data/public/day041/dataset_modelo_v2_source_quality.csv",
        "data/public/day041/dataset_modelo_v2_dispersion.csv": "data/public/day041/dataset_modelo_v2_dispersion.csv",
        "data/public/day041/dataset_modelo_v2_competition.csv": "data/public/day041/dataset_modelo_v2_competition.csv",
        "data/public/day041/dataset_modelo_v2_transport_only.csv": "data/public/day041/dataset_modelo_v2_transport_only.csv",
        "data/public/day042/dataset_modelo_v2_dispersion_plus_transport_rebuilt.csv": "data/public/day042/dataset_modelo_v2_dispersion_plus_transport_rebuilt.csv",
        "data/public/day042/dataset_modelo_v2_transport_rebuilt_only.csv": "data/public/day042/dataset_modelo_v2_transport_rebuilt_only.csv",
        "data/public/day043/dataset_modelo_v2_dispersion_plus_transport_carry30d.csv": "data/public/day043/dataset_modelo_v2_dispersion_plus_transport_carry30d.csv",
        "data/public/day043/dataset_modelo_v2_transport_carry30d_only.csv": "data/public/day043/dataset_modelo_v2_transport_carry30d_only.csv",
        "data/public/inference_inputs/example_real_day_2024-05-28.csv": "data/public/inference_inputs/example_real_day_2024-05-28.csv",
    }
    written: dict[str, Path] = {}
    for source_rel, dest_rel in source_to_dest.items():
        frame = pd.read_csv(private_root / source_rel, keep_default_na=False)
        public_frame = sanitize_frame(
            frame,
            alias_maps=alias_maps,
            seed=seed,
            date_shift_days=DEFAULT_DATE_SHIFT_DAYS,
        )
        dest_path = public_root / dest_rel
        ensure_parent(dest_path)
        public_frame.to_csv(dest_path, index=False, encoding="utf-8")
        written[source_rel] = dest_path

    support_dir = public_root / "data/public/support"
    support_dir.mkdir(parents=True, exist_ok=True)
    support_frames = build_support_frames(private_root=private_root, alias_maps=alias_maps, seed=seed)
    for filename, frame in support_frames.items():
        frame.to_csv(support_dir / filename, index=False, encoding="utf-8")

    output_dir = public_root / "data/public/inference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / ".gitkeep", "")
    return written


def copy_public_artifacts(private_root: Path, public_root: Path, rules: list[ReplacementRule]) -> None:
    artifacts_root = public_root / "artifacts/public"
    metrics_dest = artifacts_root / "metrics"
    post_dest = artifacts_root / "postinferencia"
    metrics_dest.mkdir(parents=True, exist_ok=True)
    post_dest.mkdir(parents=True, exist_ok=True)

    metrics_root = private_root / "artifacts/public/metrics"
    for source_path in sorted(metrics_root.rglob("*")):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(metrics_root)
        rel_posix = relative.as_posix()
        include = (
            source_path.suffix.lower() == ".json"
            or rel_posix == "final_baseline_vs_candidates.csv"
            or rel_posix == "archive/20260306_pre_date_fix_final_baseline_vs_candidates.csv"
            or re.match(r"^day05(?:_1|_2|_4)?/.+_(canonical_candidates|phase2_trials|policy_trials)\.csv$", rel_posix)
        )
        if not include:
            continue
        dest_relative = sanitize_relative_path(relative, rules=rules, date_shift_days=DEFAULT_DATE_SHIFT_DAYS)
        dest_path = metrics_dest / dest_relative
        content = sanitize_text(
            source_path.read_text(encoding="utf-8", errors="ignore"),
            rules=rules,
            date_shift_days=DEFAULT_DATE_SHIFT_DAYS,
        )
        write_text(dest_path, content)

    notebook10_run_summary = json.loads(
        (
            private_root
            / "artifacts/public/metrics/candidates/20260305/20260305T160207Z_BASELINE_WITH_DETERMINISTIC_LAYER_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009_v1_run_summary.json"
        ).read_text(encoding="utf-8")
    )
    for raw_path in notebook10_run_summary.get("audit_paths", {}).values():
        source_path = Path(raw_path)
        if not source_path.exists():
            continue
        relative = source_path.relative_to(private_root / "artifacts/public/postinferencia")
        dest_relative = sanitize_relative_path(relative, rules=rules, date_shift_days=DEFAULT_DATE_SHIFT_DAYS)
        dest_path = post_dest / dest_relative
        content = sanitize_text(
            source_path.read_text(encoding="utf-8", errors="ignore"),
            rules=rules,
            date_shift_days=DEFAULT_DATE_SHIFT_DAYS,
        )
        write_text(dest_path, content)

    selected_root_reports = [
        "artifacts/public/data_quality_day041_ablation_matrix.json",
        "artifacts/public/data_quality_day042_transport_matrix.json",
        "artifacts/public/data_quality_day043_transport_matrix.json",
        "artifacts/public/data_quality_v2_candidates.json",
        "artifacts/public/data_quality_v3_context.json",
        "artifacts/public/etl_dq_ofertas_tabla_extract.json",
        "artifacts/public/transport_imputation_day043.csv",
        "artifacts/public/transport_imputation_day043.json",
        "artifacts/public/transport_missingness_day042.csv",
        "artifacts/public/transport_missingness_day042.json",
        "artifacts/public/transport_parser_day041.json",
    ]
    for relative in selected_root_reports:
        source_path = private_root / relative
        if not source_path.exists():
            continue
        dest_name = sanitize_text(source_path.name, rules=rules, date_shift_days=DEFAULT_DATE_SHIFT_DAYS)
        dest_path = artifacts_root / dest_name
        content = sanitize_text(
            source_path.read_text(encoding="utf-8", errors="ignore"),
            rules=rules,
            date_shift_days=DEFAULT_DATE_SHIFT_DAYS,
        )
        write_text(dest_path, content)


def shifted_date_text(value: str) -> str:
    return (pd.Timestamp(value) + pd.Timedelta(days=DEFAULT_DATE_SHIFT_DAYS)).strftime("%Y-%m-%d")


def notebook_preamble() -> str:
    return textwrap.dedent(
        """
        from pathlib import Path
        import sys

        PROJECT_ROOT = Path.cwd().resolve()
        if not (PROJECT_ROOT / "src").exists() and (PROJECT_ROOT.parent / "src").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        ARTIFACTS_PUBLIC = PROJECT_ROOT / "artifacts" / "public"
        DATA_PUBLIC = PROJECT_ROOT / "data" / "public"
        MODELS_PUBLIC = PROJECT_ROOT / "models" / "public"
        """
    ).strip()


def notebook19_extra_preamble() -> str:
    return textwrap.dedent(
        """
        import json
        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import shap
        from IPython.display import Markdown, display

        registry_df = pd.read_csv(ARTIFACTS_PUBLIC / "metrics" / "final_baseline_vs_candidates.csv")
        serving_default_metadata = json.loads((MODELS_PUBLIC / "baseline" / "metadata.json").read_text(encoding="utf-8"))
        pure_champion_metadata = json.loads((MODELS_PUBLIC / "champion_pure" / "metadata.json").read_text(encoding="utf-8"))
        """
    ).strip()


def patch_notebook_05(code_cells: list[dict]) -> None:
    replace_code_cell_source(
        code_cells,
        marker='DATASET_PATH = ROOT /',
        new_source=textwrap.dedent(
        """
        # Carga de dataset y checks

        import pandas as pd

        ROOT = PROJECT_ROOT
        DATASET_PATH = ROOT / "data" / "public" / "dataset_modelo_proveedor_v2_candidates.csv"

        df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)
        df["target_elegido"] = pd.to_numeric(df["target_elegido"], errors="coerce").fillna(0).astype(int)
        df["fecha_evento"] = pd.to_datetime(df["fecha_evento"], errors="coerce")

        print("rows:", len(df))
        print("events:", df["event_id"].nunique())
        print("dup(event_id, proveedor_candidato):", df.duplicated(["event_id", "proveedor_candidato"]).sum())

        event_pos = df.groupby("event_id")["target_elegido"].sum()
        print("events con 0 positivos:", int((event_pos == 0).sum()))
        print("events con >1 positivos:", int((event_pos > 1).sum()))
        """
        ).strip(),
    )


def patch_notebook_09(code_cells: list[dict]) -> None:
    replace_code_cell_source(
        code_cells,
        marker="V2_PATH = Path(",
        new_source=textwrap.dedent(
        """
        import pandas as pd
        from pathlib import Path
        import sys

        PROJECT_ROOT = Path.cwd().resolve()
        if not (PROJECT_ROOT / "src").exists() and (PROJECT_ROOT.parent / "src").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))

        from src.ml.product import recommend_supplier as rs

        V2_PATH = PROJECT_ROOT / "data" / "public" / "dataset_modelo_proveedor_v2_candidates.csv"
        MODEL_PATH = PROJECT_ROOT / "models" / "public" / "baseline" / "model.pkl"
        META_PATH = PROJECT_ROOT / "models" / "public" / "baseline" / "metadata.json"
        MODEL_LABEL = "public_baseline"

        df = pd.read_csv(V2_PATH)

        required = {"fecha_evento", "albaran_id", "event_id", "producto_canonico", "proveedor_candidato", "coste_min_dia_proveedor", "target_elegido"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en V2: {sorted(missing)}")

        for c in ["albaran_id", "event_id", "producto_canonico", "proveedor_candidato"]:
            df[c] = df[c].astype(str).str.strip()
        df["fecha_evento"] = pd.to_datetime(df["fecha_evento"], errors="coerce").dt.date
        df["target_elegido"] = pd.to_numeric(df["target_elegido"], errors="coerce").fillna(0).astype(int)
        df["coste_min_dia_proveedor"] = pd.to_numeric(df["coste_min_dia_proveedor"], errors="coerce")
        df["rank_coste_dia_producto"] = pd.to_numeric(df.get("rank_coste_dia_producto"), errors="coerce")
        """
        ).strip(),
    )


def patch_notebook_16(code_cells: list[dict]) -> None:
    replace_code_cell_source(
        code_cells,
        marker="DAY05_ROOT = PROJECT_ROOT /",
        new_source=textwrap.dedent(
        """
        from pathlib import Path
        import json
        import pandas as pd

        PROJECT_ROOT = Path.cwd().resolve()
        if not (PROJECT_ROOT / "artifacts").exists() and (PROJECT_ROOT.parent / "artifacts").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent
        DAY05_ROOT = PROJECT_ROOT / "artifacts" / "public" / "metrics" / "day05"
        RUN_SUMMARIES = sorted(DAY05_ROOT.glob("*_run_summary.json"), key=lambda path: path.stat().st_mtime)
        if not RUN_SUMMARIES:
            raise FileNotFoundError("No hay run summaries Day 05 todavía. Ejecuta el runner público antes de usar el notebook.")

        LATEST_SUMMARY_PATH = RUN_SUMMARIES[-1]
        LATEST_SUMMARY = json.loads(LATEST_SUMMARY_PATH.read_text(encoding="utf-8"))
        RUN_ID = LATEST_SUMMARY["run_id"]
        CANONICAL_CSV = DAY05_ROOT / f"{RUN_ID}_canonical_candidates.csv"
        SELECTION_JSON = DAY05_ROOT / f"{RUN_ID}_selection_decisions.json"
        POLICY_JSON = DAY05_ROOT / f"{RUN_ID}_policy_summary.json"
        canonical_df = pd.read_csv(CANONICAL_CSV)
        selection_payload = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
        policy_payload = json.loads(POLICY_JSON.read_text(encoding="utf-8")) if POLICY_JSON.exists() else {}
        RUN_ID, LATEST_SUMMARY_PATH.name, canonical_df.shape
        """
        ).strip(),
    )


def patch_notebook_17(code_cells: list[dict]) -> None:
    replace_code_cell_source(
        code_cells,
        marker="DAY051_ROOT = PROJECT_ROOT /",
        new_source=textwrap.dedent(
        """
        from pathlib import Path
        import json
        import pandas as pd

        PROJECT_ROOT = Path.cwd().resolve()
        if not (PROJECT_ROOT / "artifacts").exists() and (PROJECT_ROOT.parent / "artifacts").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent
        DAY05_ROOT = PROJECT_ROOT / "artifacts" / "public" / "metrics" / "day05"
        DAY051_ROOT = PROJECT_ROOT / "artifacts" / "public" / "metrics" / "day05_1"
        DAY05_SUMMARIES = sorted(DAY05_ROOT.glob("*_run_summary.json"), key=lambda path: path.stat().st_mtime)
        DAY051_SUMMARIES = sorted(DAY051_ROOT.glob("*_run_summary.json"), key=lambda path: path.stat().st_mtime)
        if not DAY05_SUMMARIES:
            raise FileNotFoundError("No hay run summaries Day 05 todavía.")
        if not DAY051_SUMMARIES:
            raise FileNotFoundError("No hay run summaries Day 05.1 todavía. Ejecuta el runner público antes de usar el notebook.")

        DAY05_RUN_ID = json.loads(DAY05_SUMMARIES[-1].read_text(encoding="utf-8"))["run_id"]
        DAY051_RUN_ID = json.loads(DAY051_SUMMARIES[-1].read_text(encoding="utf-8"))["run_id"]
        day05_df = pd.read_csv(DAY05_ROOT / f"{DAY05_RUN_ID}_canonical_candidates.csv")
        day051_df = pd.read_csv(DAY051_ROOT / f"{DAY051_RUN_ID}_canonical_candidates.csv")
        selection_payload = json.loads((DAY051_ROOT / f"{DAY051_RUN_ID}_selection_decisions.json").read_text(encoding="utf-8"))
        policy_payload = json.loads((DAY051_ROOT / f"{DAY051_RUN_ID}_policy_summary.json").read_text(encoding="utf-8"))
        DAY05_RUN_ID, DAY051_RUN_ID, day05_df.shape, day051_df.shape
        """
        ).strip(),
    )


def patch_notebook_02(code_cells: list[dict]) -> None:
    replace_code_cell_source(
        code_cells,
        marker="from sklearn.dummy import DummyClassifier",
        new_source=textwrap.dedent(
            """
            from sklearn.dummy import DummyClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

            # 1) Baseline Dummy (referencia)
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_train, y_train)
            y_pred_dummy = dummy.predict(X_test)

            print("=== DUMMY ===")
            print("accuracy:", round(accuracy_score(y_test, y_pred_dummy), 4))
            print("balanced_accuracy:", round(balanced_accuracy_score(y_test, y_pred_dummy), 4))
            print("f1_clase_1:", round(f1_score(y_test, y_pred_dummy, pos_label=1, zero_division=0), 4))
            print(classification_report(y_test, y_pred_dummy, digits=4, zero_division=0))

            # 2) KNN con k fijo definido arriba
            knn = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=chosen_k, n_jobs=-1))
            ])
            knn.fit(X_train, y_train)
            y_pred_knn = knn.predict(X_test)

            print(f"\\n=== KNN (k={chosen_k}) ===")
            print("accuracy:", round(accuracy_score(y_test, y_pred_knn), 4))
            print("balanced_accuracy:", round(balanced_accuracy_score(y_test, y_pred_knn), 4))
            print("f1_clase_1:", round(f1_score(y_test, y_pred_knn, pos_label=1, zero_division=0), 4))
            print(classification_report(y_test, y_pred_knn, digits=4, zero_division=0))
            """
        ).strip(),
    )


def optimize_public_runtime(code_cells: list[dict], notebook_name: str) -> None:
    replacements: dict[str, str] = {}
    if notebook_name == "01_ml_knn.ipynb":
        replacements['KNeighborsClassifier(n_neighbors=5)'] = 'KNeighborsClassifier(n_neighbors=5, n_jobs=-1)'
    elif notebook_name == "02_ml_feature_eng.ipynb":
        replacements['KNeighborsClassifier(n_neighbors=chosen_k)'] = 'KNeighborsClassifier(n_neighbors=chosen_k, n_jobs=-1)'
    elif notebook_name == "03_ml_ensemble.ipynb":
        replacements[
            textwrap.dedent(
                """
                bagging = BaggingClassifier(
                    estimator=DecisionTreeClassifier(),
                    n_estimators=100,
                    random_state=0
                )
                """
            ).strip()
        ] = textwrap.dedent(
            """
            bagging = BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=100,
                random_state=0,
                n_jobs=-1,
            )
            """
        ).strip()
        replacements[
            textwrap.dedent(
                """
                rf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=0
                )
                """
            ).strip()
        ] = textwrap.dedent(
            """
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=0,
                n_jobs=-1,
            )
            """
        ).strip()

    if not replacements:
        return

    for cell in code_cells:
        source = cell.get("source", "")
        source_text = "".join(source) if isinstance(source, list) else source
        updated = source_text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        cell["source"] = updated


def patch_notebook_10(code_cells: list[dict]) -> None:
    replace_code_cell_source(
        code_cells,
        marker='candidate_dates = sorted([p for p in CANDIDATES_ROOT.iterdir() if p.is_dir()])',
        new_source=textwrap.dedent(
            """
            from pathlib import Path
            import sys

            # Resolver raíz del repo tanto si ejecutas desde /repo como desde /repo/notebooks
            CWD = Path.cwd().resolve()
            if (CWD / "artifacts" / "public").exists():
                PROJECT_ROOT = CWD
            elif (CWD.parent / "artifacts" / "public").exists():
                PROJECT_ROOT = CWD.parent
            else:
                raise FileNotFoundError(f"No encuentro /artifacts/public ni en {CWD} ni en {CWD.parent}")

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            REPORTS_ROOT = PROJECT_ROOT / "artifacts" / "public"
            CANDIDATES_ROOT = REPORTS_ROOT / "metrics" / "candidates"
            BASELINE_ROOT = REPORTS_ROOT / "metrics" / "baseline"

            assert CANDIDATES_ROOT.exists(), f"No existe carpeta candidates: {CANDIDATES_ROOT}"
            assert BASELINE_ROOT.exists(), f"No existe carpeta baseline: {BASELINE_ROOT}"

            # Este notebook documenta el backtest Day03 y debe apuntar al run summary específico de 20260305.
            CANDIDATES_DIR = CANDIDATES_ROOT / "20260305"
            assert CANDIDATES_DIR.exists(), f"No existe carpeta Day03 esperada: {CANDIDATES_DIR}"
            RUN_DATE = CANDIDATES_DIR.name

            baseline_files = sorted(BASELINE_ROOT.glob("*_postinferencia_albaran_baseline_metrics.json"))
            assert baseline_files, "No hay baseline Day03 postinferencia."
            BASELINE_PATH = baseline_files[-1]

            patterns = [
                "*_BASELINE_WITH_DETERMINISTIC_LAYER_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009_v1_run_summary.json",
                "*_POLICY_PRODUCT_002_FOLLOW_PRODUCT_003_SUPPLIER_009_v1_run_summary.json",
                "*_run_summary.json",
            ]

            run_summary_files = []
            for pat in patterns:
                run_summary_files = sorted(CANDIDATES_DIR.glob(pat))
                if run_summary_files:
                    break

            assert run_summary_files, f"No hay run_summary en {CANDIDATES_DIR}"
            RUN_SUMMARY_PATH = run_summary_files[-1]

            print("PROJECT_ROOT:", PROJECT_ROOT)
            print("RUN_DATE:", RUN_DATE)
            print("BASELINE_PATH:", BASELINE_PATH)
            print("RUN_SUMMARY_PATH:", RUN_SUMMARY_PATH)
            RUN_SUMMARY_PATH
            """
        ).strip(),
    )

    replace_code_cell_source(
        code_cells,
        marker="baseline_payload = json.loads(",
        new_source=textwrap.dedent(
            """
            import json
            from pathlib import Path

            import pandas as pd
            from IPython.display import display

            baseline_payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            run_summary_payload = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))

            candidate_metrics_path = PROJECT_ROOT / Path(run_summary_payload["candidate_metrics_json"])
            candidate_payload = json.loads(candidate_metrics_path.read_text(encoding="utf-8"))

            audit_paths = {key: PROJECT_ROOT / Path(value) for key, value in run_summary_payload["audit_paths"].items()}
            df_detail = pd.read_csv(audit_paths["detail"])
            df_resumen_evento = pd.read_csv(audit_paths["resumen_evento"])
            df_resumen_albaran = pd.read_csv(audit_paths["resumen_albaran"])

            display(pd.DataFrame([baseline_payload["metrics"]]))
            display(pd.DataFrame([candidate_payload["metrics"]]))
            display(df_resumen_albaran.head(20))
            """
        ).strip(),
    )


def replace_code_cell_source(code_cells: list[dict], marker: str, new_source: str) -> None:
    for cell in code_cells:
        source = cell.get("source", "")
        source_text = "".join(source) if isinstance(source, list) else source
        if marker in source_text:
            cell["source"] = new_source
            return
    raise KeyError(f"No se encontró la celda objetivo para el marcador: {marker}")


def build_notebook_07() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(
            "# Notebook 07 · SQL serving público\n\n"
            "Versión portfolio-ready que sustituye MySQL local por SQLite y usa únicamente datasets públicos."
        ),
        new_code_cell(notebook_preamble()),
        new_code_cell(
            textwrap.dedent(
                """
                import json
                import sqlite3
                import re
                from pathlib import Path

                import pandas as pd
                import yaml

                SQL_DIR = ARTIFACTS_PUBLIC / "sql"
                SQL_DIR.mkdir(parents=True, exist_ok=True)
                DB_PATH = SQL_DIR / "public_serving.db"
                CONTRACT_PATH = PROJECT_ROOT / "config" / "inference_input_contract.yaml"
                VIEWS_SQL_PATH = PROJECT_ROOT / "src" / "sql" / "public_src/sql/public_db_views_serving.sql"

                contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
                print({"db_path": str(DB_PATH.relative_to(PROJECT_ROOT)), "contract_path": str(CONTRACT_PATH.relative_to(PROJECT_ROOT))})
                contract
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                public_csv_paths = sorted((DATA_PUBLIC).rglob("*.csv"))
                inventory_df = pd.DataFrame(
                    [
                        {
                            "relative_path": str(path.relative_to(PROJECT_ROOT)),
                            "table_name": re.sub(r"[^a-z0-9_]+", "_", "public_" + path.stem.lower()),
                            "rows_csv": max(sum(1 for _ in path.open("rb")) - 1, 0),
                        }
                        for path in public_csv_paths
                    ]
                )
                inventory_df
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                with sqlite3.connect(DB_PATH) as conn:
                    load_rows = []
                    for row in inventory_df.to_dict(orient="records"):
                        source_path = PROJECT_ROOT / row["relative_path"]
                        table_name = row["table_name"]
                        df = pd.read_csv(source_path)
                        df.to_sql(table_name, conn, if_exists="replace", index=False)
                        rows_loaded = pd.read_sql_query(f"SELECT COUNT(*) AS rows_loaded FROM {table_name}", conn).iloc[0]["rows_loaded"]
                        load_rows.append(
                            {
                                "table_name": table_name,
                                "rows_csv": int(row["rows_csv"]),
                                "rows_loaded": int(rows_loaded),
                                "load_ok": int(row["rows_csv"]) == int(rows_loaded),
                            }
                        )
                    load_audit_df = pd.DataFrame(load_rows)
                load_audit_df
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                with sqlite3.connect(DB_PATH) as conn:
                    conn.executescript(VIEWS_SQL_PATH.read_text(encoding="utf-8"))
                    checks = {
                        "vw_inference_input_daily_rows": int(pd.read_sql_query("SELECT COUNT(*) AS rows FROM vw_inference_input_daily", conn).iloc[0]["rows"]),
                        "vw_event_summary_daily_rows": int(pd.read_sql_query("SELECT COUNT(*) AS rows FROM vw_event_summary_daily", conn).iloc[0]["rows"]),
                    }
                    sample_daily = pd.read_sql_query("SELECT * FROM vw_inference_input_daily LIMIT 10", conn)
                    sample_events = pd.read_sql_query("SELECT * FROM vw_event_summary_daily LIMIT 10", conn)

                print(checks)
                display(sample_daily)
                display(sample_events)
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                required_cols = set()
                for column_group in contract["columns"].values():
                    required_cols.update(column_group)
                input_df = pd.read_csv(DATA_PUBLIC / "inference_inputs" / "example_real_day_2024-05-28.csv")
                missing = required_cols - set(input_df.columns)
                assert not missing, f"Faltan columnas de contrato en el input público: {sorted(missing)}"
                print({"required_columns": len(required_cols), "input_shape": input_df.shape})
                """
            ).strip()
        ),
    ]
    return new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def build_notebook_08() -> nbformat.NotebookNode:
    shifted_dates = [
        shifted_date_text(date_text)
        for date_text in ["2028-05-02", "2028-05-03", "2028-05-06", "2028-05-07", "2028-05-08"]
    ]
    cells = [
        new_markdown_cell(
            "# Notebook 08 · Patrones operativos públicos\n\n"
            "Versión pública basada en el dataset candidato pseudonimizado. Mantiene el objetivo del notebook: "
            "leer patrones de selección histórica frente al mínimo de coste y preparar hipótesis de política."
        ),
        new_code_cell(notebook_preamble()),
        new_code_cell(
            textwrap.dedent(
                f"""
                import numpy as np
                import pandas as pd

                V2_PATH = DATA_PUBLIC / "dataset_modelo_proveedor_v2_candidates.csv"
                TARGET_DATES = [pd.Timestamp(date_text).date() for date_text in {shifted_dates!r}]

                df = pd.read_csv(V2_PATH, low_memory=False)
                df["fecha_evento"] = pd.to_datetime(df["fecha_evento"], errors="coerce").dt.date
                df["coste_min_dia_proveedor"] = pd.to_numeric(df["coste_min_dia_proveedor"], errors="coerce")
                df["rank_coste_dia_producto"] = pd.to_numeric(df["rank_coste_dia_producto"], errors="coerce")
                df["target_elegido"] = pd.to_numeric(df["target_elegido"], errors="coerce").fillna(0).astype(int)
                df_window = df[df["fecha_evento"].isin(TARGET_DATES)].copy()

                print({{"rows_total": len(df), "rows_window": len(df_window), "events_window": df_window["event_id"].nunique()}})
                display(df_window.head(10))
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                event_min = (
                    df_window.groupby("event_id", as_index=False)["coste_min_dia_proveedor"]
                    .min()
                    .rename(columns={"coste_min_dia_proveedor": "event_min_cost"})
                )

                selected_rows = (
                    df_window[df_window["target_elegido"] == 1]
                    .drop_duplicates(subset=["event_id"])
                    .merge(event_min, on="event_id", how="left")
                )
                selected_rows["selected_in_min_cost"] = np.isclose(
                    selected_rows["coste_min_dia_proveedor"],
                    selected_rows["event_min_cost"],
                    equal_nan=False,
                )
                selected_rows["delta_chosen_vs_min_eur_m3"] = (
                    selected_rows["coste_min_dia_proveedor"] - selected_rows["event_min_cost"]
                )

                display(selected_rows.head(20))
                print(
                    {
                        "selected_events": int(selected_rows["event_id"].nunique()),
                        "selected_in_min_cost_rate": round(float(selected_rows["selected_in_min_cost"].mean()), 4),
                    }
                )
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                pattern_by_day = (
                    selected_rows.groupby(["fecha_evento", "producto_canonico"], as_index=False)
                    .agg(
                        events=("event_id", "nunique"),
                        proveedores=("proveedor_elegido_real", lambda s: sorted(set(s.dropna()))),
                        hit_rate_vs_min=("selected_in_min_cost", "mean"),
                        median_delta_vs_min=("delta_chosen_vs_min_eur_m3", "median"),
                    )
                    .sort_values(["fecha_evento", "producto_canonico"])
                )
                display(pattern_by_day)

                pattern_by_terminal = (
                    selected_rows.groupby(["producto_canonico", "terminal_compra"], as_index=False)
                    .agg(
                        events=("event_id", "nunique"),
                        unique_selected=("proveedor_elegido_real", "nunique"),
                        median_delta_vs_min=("delta_chosen_vs_min_eur_m3", "median"),
                    )
                    .sort_values(["events", "median_delta_vs_min"], ascending=[False, False])
                )
                display(pattern_by_terminal.head(20))
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                focus_candidates = (
                    selected_rows.groupby("albaran_id", as_index=False)
                    .agg(events=("event_id", "nunique"), litros=("litros_evento", "sum"))
                    .sort_values(["events", "litros"], ascending=[False, False])
                )
                ALBARAN_FOCUS = focus_candidates.iloc[0]["albaran_id"] if not focus_candidates.empty else None
                focus_df = selected_rows[selected_rows["albaran_id"].eq(ALBARAN_FOCUS)].copy() if ALBARAN_FOCUS else pd.DataFrame()

                print({"albaran_focus": ALBARAN_FOCUS, "focus_rows": len(focus_df)})
                display(focus_df.head(20))
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                rule_rows = []
                for idx, (_, row) in enumerate(pattern_by_terminal.head(5).iterrows(), start=1):
                    rule_rows.append(
                        {
                            "rule_candidate_id": f"RULE_{idx:03d}",
                            "if_condition_data": f"producto_canonico == '{row['producto_canonico']}' and terminal_compra == '{row['terminal_compra']}'",
                            "support_events": int(row["events"]),
                            "median_delta_eur_m3": float(row["median_delta_vs_min"]) if pd.notna(row["median_delta_vs_min"]) else None,
                            "reading": "Priorizar revisión manual si el patrón concentra eventos y sobrecoste frente al mínimo.",
                        }
                    )

                df_rule_candidates = pd.DataFrame(rule_rows)
                display(df_rule_candidates)

                qa_summary = {
                    "window_rows": int(len(df_window)),
                    "window_events": int(df_window["event_id"].nunique()),
                    "selected_rows": int(len(selected_rows)),
                    "rule_candidates": int(len(df_rule_candidates)),
                }
                print("QA PASSED")
                print(qa_summary)
                """
            ).strip()
        ),
    ]
    return new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def build_notebook_18() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(
            "# Notebook 18 · Day 05.2 short tuning público\n\n"
            "Versión pública que carga el resumen persistido del bloque Day 05.2 y deja la evidencia lista para portfolio."
        ),
        new_code_cell(notebook_preamble()),
        new_code_cell(
            textwrap.dedent(
                """
                import json
                import pandas as pd

                DAY052_ROOT = ARTIFACTS_PUBLIC / "metrics" / "day05_2"
                RUN_SUMMARY_PATHS = sorted(DAY052_ROOT.glob("*_run_summary.json"))
                if not RUN_SUMMARY_PATHS:
                    raise FileNotFoundError("No hay run_summary público Day 05.2.")

                RUN_SUMMARY_PATH = RUN_SUMMARY_PATHS[-1]
                run_summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
                RUN_ID = run_summary["run_id"]
                canonical_candidates = pd.read_csv(DAY052_ROOT / f"{RUN_ID}_canonical_candidates.csv")
                phase2_trials = pd.read_csv(DAY052_ROOT / f"{RUN_ID}_phase2_trials.csv")

                print({"run_id": RUN_ID, "canonical_candidates": canonical_candidates.shape, "phase2_trials": phase2_trials.shape})
                pd.DataFrame([run_summary])
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                trials_view = phase2_trials.sort_values(
                    ["cv_top2_hit_mean", "cv_bal_acc_mean", "cv_f1_pos_mean"],
                    ascending=[False, False, False],
                ).reset_index(drop=True)
                trials_view
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                finalists_view = canonical_candidates.loc[
                    canonical_candidates["model_variant"].isin(run_summary["finalist_variants"])
                ].sort_values(
                    ["top2_hit", "balanced_accuracy", "f1_pos"],
                    ascending=[False, False, False],
                ).reset_index(drop=True)
                finalists_view
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                secondary_refs = pd.DataFrame(
                    {
                        "secondary_reference_variant": run_summary["secondary_reference_variants"],
                        "current_model_champion_variant": run_summary["current_model_champion_variant"],
                        "close_decision": run_summary["close_decision"],
                    }
                )
                secondary_refs
                """
            ).strip()
        ),
    ]
    return new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def sanitize_notebook(
    source_path: Path,
    dest_path: Path,
    rules: list[ReplacementRule],
    date_shift_days: int,
) -> None:
    notebook_name = source_path.name
    if notebook_name == "07_sql_serving.ipynb":
        nbformat.write(build_notebook_07(), dest_path)
        return
    if notebook_name == "08_patrones.ipynb":
        nbformat.write(build_notebook_08(), dest_path)
        return
    if notebook_name == "18_day05_2_short_tuning_new_champion_pure.ipynb":
        nbformat.write(build_notebook_18(), dest_path)
        return

    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source_text = "".join(source)
            source_text = sanitize_text(source_text, rules=rules, date_shift_days=date_shift_days)
            cell["source"] = source_text.splitlines(keepends=True)
        elif isinstance(source, str):
            cell["source"] = sanitize_text(source, rules=rules, date_shift_days=date_shift_days)

        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    notebook["cells"].insert(0, new_code_cell(notebook_preamble()))
    if notebook_name == "19_day05_3_error_analysis_and_shap_explainer.ipynb":
        notebook["cells"].insert(1, new_code_cell(notebook19_extra_preamble()))

    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if notebook_name == "02_ml_feature_eng.ipynb":
        patch_notebook_02(code_cells)
    elif notebook_name == "05_ml_postinferencia.ipynb":
        patch_notebook_05(code_cells)
    elif notebook_name == "10_day03_postinferencia_albaran_backtest.ipynb":
        patch_notebook_10(code_cells)
    elif notebook_name == "09_validacion_posinferencia_albaran.ipynb":
        patch_notebook_09(code_cells)
    elif notebook_name == "16_day05_tabular_models_only.ipynb":
        patch_notebook_16(code_cells)
    elif notebook_name == "17_day05_1_balanced_tree_baselines.ipynb":
        patch_notebook_17(code_cells)

    optimize_public_runtime(code_cells, notebook_name=notebook_name)

    ensure_parent(dest_path)
    dest_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def write_notebooks(private_root: Path, public_root: Path, rules: list[ReplacementRule]) -> None:
    notebooks_dest = public_root / "notebooks"
    notebooks_dest.mkdir(parents=True, exist_ok=True)
    for source_path in sorted((private_root / "notebooks").glob("*.ipynb")):
        sanitize_notebook(
            source_path=source_path,
            dest_path=notebooks_dest / source_path.name,
            rules=rules,
            date_shift_days=DEFAULT_DATE_SHIFT_DAYS,
        )


def build_training_contract(
    private_root: Path,
    public_root: Path,
    public_name: str,
    seed: int,
) -> dict:
    baseline_metadata = json.loads((private_root / "models/public/baseline/metadata.json").read_text(encoding="utf-8"))
    champion_metadata = json.loads((private_root / "models/public/champion_pure/metadata.json").read_text(encoding="utf-8"))
    private_v2 = pd.read_csv(private_root / "data/public/dataset_modelo_proveedor_v2_candidates.csv")
    private_v2["fecha_evento"] = pd.to_datetime(private_v2["fecha_evento"], errors="coerce")
    baseline_cutoff = pd.Timestamp(baseline_metadata["cutoff_date"])
    shifted_cutoff = (baseline_cutoff + pd.Timedelta(days=DEFAULT_DATE_SHIFT_DAYS)).strftime("%Y-%m-%d")
    baseline_test_events = int(private_v2.loc[private_v2["fecha_evento"] > baseline_cutoff, "event_id"].nunique())

    contract = {
        "publication_mode": DEFAULT_MODE,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": seed,
        "date_shift_days": DEFAULT_DATE_SHIFT_DAYS,
        "public_name": public_name,
        "public_root": str(public_root),
        "notebook_count": len(list((private_root / "notebooks").glob("*.ipynb"))),
        "datasets": {
            "core": [
                "data/public/dataset_modelo_proveedor_v1.csv",
                "data/public/dataset_modelo_proveedor_v2_candidates.csv",
                "data/public/dataset_modelo_proveedor_v2_excluded_events.csv",
                "data/public/dataset_modelo_proveedor_v3_context.csv",
                "data/public/day041/dataset_modelo_v2_source_quality.csv",
                "data/public/day041/dataset_modelo_v2_dispersion.csv",
                "data/public/day041/dataset_modelo_v2_competition.csv",
                "data/public/day041/dataset_modelo_v2_transport_only.csv",
                "data/public/day042/dataset_modelo_v2_dispersion_plus_transport_rebuilt.csv",
                "data/public/day042/dataset_modelo_v2_transport_rebuilt_only.csv",
                "data/public/day043/dataset_modelo_v2_dispersion_plus_transport_carry30d.csv",
                "data/public/day043/dataset_modelo_v2_transport_carry30d_only.csv",
                "data/public/inference_inputs/example_real_day_2024-05-28.csv",
            ],
            "support": [
                "data/public/support/ofertas_typed.csv",
                "data/public/support/ofertas_raw_mincost.csv",
                "data/public/support/ofertas_transport_signals_day042.csv",
            ],
        },
        "analysis_windows": {
            "pattern_window_dates": [
                shifted_date_text(date)
                for date in ["2028-05-02", "2028-05-03", "2028-05-06", "2028-05-07", "2028-05-08"]
            ]
        },
        "training": {
            "baseline": {
                "dataset_path": "data/public/dataset_modelo_proveedor_v2_candidates.csv",
                "cutoff_date": shifted_cutoff,
                "reference_variant": "LR_smote_0.5",
                "public_model_name": "LR_smote_0.5_public",
                "selection_rule": baseline_metadata["selection_rule"],
                "expected_metrics": {
                    "accuracy": float(baseline_metadata["metrics"]["test_acc"]),
                    "balanced_accuracy": float(baseline_metadata["metrics"]["test_bal_acc"]),
                    "f1_pos": float(baseline_metadata["metrics"]["test_f1_pos"]),
                    "top1_hit": float(baseline_metadata["metrics"]["top1_hit"]),
                    "top2_hit": float(baseline_metadata["metrics"]["top2_hit"]),
                    "coverage": 1.0,
                    "test_events": baseline_test_events,
                },
            },
            "champion_pure": {
                "dataset_path": "data/public/day041/dataset_modelo_v2_transport_only.csv",
                "cutoff_date": shifted_cutoff,
                "reference_variant": champion_metadata["model_name"],
                "public_model_name": f"{champion_metadata['model_name']}_public",
                "selection_rule": champion_metadata["selection_rule"],
                "params": champion_metadata["params"],
                "expected_metrics": {
                    "accuracy": float(champion_metadata["metrics"]["accuracy"]),
                    "balanced_accuracy": float(champion_metadata["metrics"]["balanced_accuracy"]),
                    "f1_pos": float(champion_metadata["metrics"]["f1_pos"]),
                    "top1_hit": float(champion_metadata["metrics"]["top1_hit"]),
                    "top2_hit": float(champion_metadata["metrics"]["top2_hit"]),
                    "coverage": float(champion_metadata["metrics"]["coverage"]),
                    "test_events": int(champion_metadata["metrics"]["test_events"]),
                },
            },
        },
    }
    return contract


def write_public_root_files(private_root: Path, public_root: Path, rules: list[ReplacementRule]) -> None:
    template_root = private_root / "scripts" / "public_templates"
    write_text(public_root / "README.md", (template_root / "README.md").read_text(encoding="utf-8"))
    write_text(public_root / "app.py", (template_root / "app.py").read_text(encoding="utf-8"))
    write_text(public_root / "requirements.txt", build_requirements_txt())
    write_text(public_root / "requirements-dev.txt", build_requirements_dev_txt())
    write_text(public_root / ".gitignore", build_public_gitignore())
    write_text(public_root / ".python-version", "3.11\n")
    shutil.copy2(private_root / "LICENSE", public_root / "LICENSE")

    scripts_dest = public_root / "scripts"
    scripts_dest.mkdir(parents=True, exist_ok=True)
    for source_name, dest_name in [
        ("train_public_models.py", "train_public_models.py"),
        ("run_public_notebooks.py", "run_public_notebooks.py"),
        ("audit_public_repo.py", "audit_public_repo.py"),
    ]:
        content = (template_root / source_name).read_text(encoding="utf-8")
        write_text(scripts_dest / dest_name, content)

    sanitized_publish_script = sanitize_text(
        (private_root / "scripts/publish_to_bootcamp.py").read_text(encoding="utf-8"),
        rules=rules,
        date_shift_days=DEFAULT_DATE_SHIFT_DAYS,
    )
    write_text(scripts_dest / "publish_to_bootcamp.py", sanitized_publish_script)


def write_public_config(private_root: Path, public_root: Path, rules: list[ReplacementRule]) -> None:
    config_dest = public_root / "config"
    config_dest.mkdir(parents=True, exist_ok=True)
    for source_path in iter_files(private_root / "config"):
        if source_path.name == ".gitkeep":
            continue
        dest_path = config_dest / source_path.name
        content = sanitize_text(
            source_path.read_text(encoding="utf-8", errors="ignore"),
            rules=rules,
            date_shift_days=DEFAULT_DATE_SHIFT_DAYS,
        )
        write_text(dest_path, content)


def write_public_src(private_root: Path, public_root: Path, rules: list[ReplacementRule]) -> None:
    for relative in ["src/etl", "src/ml", "src/sql", "src/prompts"]:
        source_root = private_root / relative
        dest_root = public_root / relative
        copy_text_tree(
            source_root=source_root,
            dest_root=dest_root,
            rules=rules,
            date_shift_days=DEFAULT_DATE_SHIFT_DAYS,
        )

    functions_path = public_root / "src/ml/shared/functions.py"
    if functions_path.exists():
        content = functions_path.read_text(encoding="utf-8")
        content = content.replace(
            "BASE_DIR = Path(__file__).resolve().parent.parent.parent",
            "BASE_DIR = Path(__file__).resolve().parents[3]",
        )
        content = content.replace(
            'DATA_CLEAN_DIR = BASE_DIR / "data" / "marts"',
            'DATA_CLEAN_DIR = BASE_DIR / "data" / "public"',
        )
        write_text(functions_path, content)

    project_paths_path = public_root / "src/ml/shared/project_paths.py"
    if project_paths_path.exists():
        content = project_paths_path.read_text(encoding="utf-8")
        content = content.replace('DATA_MARTS_DIR = DATA_DIR / "marts"', 'DATA_MARTS_DIR = DATA_DIR / "public"')
        content = content.replace('REPORTS_DIR = PROJECT_ROOT / "reports"', 'REPORTS_DIR = PROJECT_ROOT / "artifacts" / "public"')
        content = content.replace(
            'BASELINE_MODEL_PATH = MODELS_DIR / "day04_champion_tuned" / "model.pkl"',
            'BASELINE_MODEL_PATH = MODELS_DIR / "public" / "baseline" / "model.pkl"',
        )
        content = content.replace(
            'BASELINE_METADATA_PATH = MODELS_DIR / "day04_champion_tuned" / "metadata.json"',
            'BASELINE_METADATA_PATH = MODELS_DIR / "public" / "baseline" / "metadata.json"',
        )
        content = content.replace(
            'CHAMPION_MODEL_PATH = MODELS_DIR / "day05_1_champion_pure" / "model.pkl"',
            'CHAMPION_MODEL_PATH = MODELS_DIR / "public" / "champion_pure" / "model.pkl"',
        )
        content = content.replace(
            'CHAMPION_METADATA_PATH = MODELS_DIR / "day05_1_champion_pure" / "metadata.json"',
            'CHAMPION_METADATA_PATH = MODELS_DIR / "public" / "champion_pure" / "metadata.json"',
        )
        write_text(project_paths_path, content)


def write_build_metadata(
    private_root: Path,
    public_root: Path,
    rules: list[ReplacementRule],
    alias_maps: dict[str, dict[str, str]],
    public_name: str,
    seed: int,
) -> None:
    artifacts_root = public_root / "artifacts/public"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    contract = build_training_contract(
        private_root=private_root,
        public_root=public_root,
        public_name=public_name,
        seed=seed,
    )
    write_json(artifacts_root / "build_contract.json", contract)

    token_payload = json.loads((private_root / "config/publish_manifest.json").read_text(encoding="utf-8"))
    dynamic_tokens = sorted(
        {
            *alias_maps["provider"].keys(),
            *alias_maps["product"].keys(),
            *alias_maps["terminal"].keys(),
            *alias_maps["rule"].keys(),
        }
    )
    token_config = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tokens": sorted(set(token_payload.get("sensitive_tokens", []) + dynamic_tokens)),
        "seed": seed,
    }
    write_json(artifacts_root / "token_scan_config.json", token_config)


def write_public_metric_parity(private_root: Path, public_root: Path, seed: int) -> dict:
    reference = compute_snapshot_from_paths(
        v1_path=private_root / "data/public/dataset_modelo_proveedor_v1.csv",
        v2_path=private_root / "data/public/dataset_modelo_proveedor_v2_candidates.csv",
        seed=seed,
    )
    candidate = compute_snapshot_from_paths(
        v1_path=public_root / "data/public/dataset_modelo_proveedor_v1.csv",
        v2_path=public_root / "data/public/dataset_modelo_proveedor_v2_candidates.csv",
        seed=seed,
    )
    parity = compare_metric_snapshots(reference, candidate)
    payload = {
        "status": parity["status"],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference_snapshot": reference,
        "candidate_snapshot": candidate,
        "metric_checks": parity["metric_checks"],
    }
    write_json(public_root / "artifacts/public/metrics/public_dataset_metric_parity.json", payload)
    return payload


def run_public_pipeline(public_root: Path, strict: bool) -> None:
    commands = [
        [sys.executable, "scripts/train_public_models.py"],
        [sys.executable, "scripts/run_public_notebooks.py"],
        [sys.executable, "scripts/audit_public_repo.py", "--strict" if strict else ""],
    ]
    for command in commands:
        command = [part for part in command if part]
        print(f"[public] running: {' '.join(command)}", flush=True)
        try:
            subprocess.run(
                command,
                cwd=str(public_root),
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Command failed in public repo: {' '.join(command)}") from exc


def build_public_repo(
    private_root: Path,
    public_root: Path,
    public_name: str,
    seed: int,
    strict: bool,
) -> dict:
    manifest = load_manifest(private_root)
    validate_public_root(private_root=private_root, public_root=public_root, manifest=manifest)
    clean_public_root(public_root=public_root, manifest=manifest)

    core_frames = load_core_private_frames(private_root)
    alias_maps = build_public_alias_maps(core_frames)
    rules = build_replacement_rules(
        private_root=private_root,
        public_name=public_name,
        alias_maps=alias_maps,
        date_shift_days=DEFAULT_DATE_SHIFT_DAYS,
    )

    write_public_root_files(private_root=private_root, public_root=public_root, rules=rules)
    write_public_config(private_root=private_root, public_root=public_root, rules=rules)
    write_public_src(private_root=private_root, public_root=public_root, rules=rules)
    write_public_datasets(private_root=private_root, public_root=public_root, alias_maps=alias_maps, seed=seed)
    copy_public_artifacts(private_root=private_root, public_root=public_root, rules=rules)
    write_notebooks(private_root=private_root, public_root=public_root, rules=rules)
    write_build_metadata(
        private_root=private_root,
        public_root=public_root,
        rules=rules,
        alias_maps=alias_maps,
        public_name=public_name,
        seed=seed,
    )
    dataset_parity = write_public_metric_parity(private_root=private_root, public_root=public_root, seed=seed)
    if dataset_parity["status"] != "PASS":
        raise RuntimeError("La paridad métrica del dataset público falló antes del training.")

    run_public_pipeline(public_root=public_root, strict=strict)
    audit_summary = json.loads((public_root / "artifacts/public/public_audit_summary.json").read_text(encoding="utf-8"))
    if strict and audit_summary["status"] != "PASS":
        raise RuntimeError("La auditoría pública no quedó en PASS.")
    return audit_summary


def main() -> None:
    args = parse_args()
    private_root = Path(args.private_root).resolve()
    manifest = load_manifest(private_root)
    public_root = resolve_public_root(private_root, manifest, args.public_root)

    summary = build_public_repo(
        private_root=private_root,
        public_root=public_root,
        public_name=args.public_name,
        seed=args.seed,
        strict=args.strict,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
