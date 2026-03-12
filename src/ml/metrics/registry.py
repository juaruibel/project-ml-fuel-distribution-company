#!/usr/bin/env python3

# LIBRERIAS

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# CONSTANTES

CSV_COLUMNS = [
    "run_id",
    "day_id",
    "model_variant",
    "model_role",
    "dataset_name",
    "dataset_snapshot_hash",
    "cutoff_date",
    "test_events",
    "accuracy",
    "balanced_accuracy",
    "f1_pos",
    "top1_hit",
    "top2_hit",
    "coverage",
    "delta_top2_vs_baseline",
    "delta_bal_acc_vs_baseline",
    "delta_coverage_vs_baseline",
    "gate_top2_ok",
    "gate_bal_acc_ok",
    "gate_coverage_ok",
    "gate_pass",
    "promotion_decision",
    "selection_rule",
    "model_path",
    "metadata_path",
    "metrics_source",
    "created_at_utc",
]

GATE_MIN_DELTA_TOP2 = 0.01
GATE_MIN_DELTA_BAL_ACC = 0.01
GATE_MIN_DELTA_COVERAGE = -0.005
POLICY_GATE_MIN_DELTA_TOP2 = 0.0
POLICY_GATE_MIN_DELTA_BAL_ACC = 0.0
POLICY_GATE_MIN_DELTA_COVERAGE = -0.005


# FUNCIÓN DE PARSEO DE ARGUMENTOS
def parse_args() -> argparse.Namespace:
    """
    Define la CLI con dos subcomandos:
    - `init-baseline`: crea el acta oficial con la fila baseline.
    - `append-candidate`: agrega una fila de candidato y calcula deltas.
    """
    parser = argparse.ArgumentParser(
        description="Gestiona artifacts/public/metrics/final_baseline_vs_candidates.csv con trazabilidad reproducible."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init-baseline",
        help="Inicializa el CSV oficial con la fila baseline.",
    )
    init_parser.add_argument(
        "--output",
        required=True,
        help="Ruta CSV de salida (recomendado: artifacts/public/metrics/final_baseline_vs_candidates.csv).",
    )
    init_parser.add_argument("--run-id", required=True, help="Identificador único de corrida.")
    init_parser.add_argument("--day-id", required=True, help="Día del plan (ej. Day 02).")
    init_parser.add_argument("--model-variant", default="LR_smote_0.5", help="Nombre del baseline.")
    init_parser.add_argument("--metadata", required=True, help="Ruta metadata.json del baseline.")
    init_parser.add_argument(
        "--metrics-json",
        default="",
        help="Ruta JSON opcional con métricas baseline (si se informa, prevalece sobre metadata).",
    )
    init_parser.add_argument("--dataset", required=True, help="Ruta dataset usado para evaluación.")
    init_parser.add_argument("--coverage", type=float, default=None, help="Cobertura operativa opcional.")
    init_parser.add_argument("--test-events", type=int, default=None, help="Eventos de test opcionales.")
    init_parser.add_argument("--model-path", default="", help="Ruta model.pkl (si se omite, se infiere).")
    init_parser.add_argument("--overwrite", action="store_true", help="Sobrescribe el CSV si ya existe.")

    append_parser = subparsers.add_parser(
        "append-candidate",
        help="Agrega un candidato calculando deltas contra baseline.",
    )
    append_parser.add_argument(
        "--output",
        required=True,
        help="Ruta CSV oficial existente (recomendado: artifacts/public/metrics/final_baseline_vs_candidates.csv).",
    )
    append_parser.add_argument("--run-id", required=True, help="Identificador único de corrida.")
    append_parser.add_argument("--day-id", required=True, help="Día del plan (ej. Day 03).")
    append_parser.add_argument("--model-variant", required=True, help="Nombre del candidato.")
    append_parser.add_argument("--metadata", required=True, help="Ruta metadata.json del candidato.")
    append_parser.add_argument("--metrics-json", required=True, help="Ruta JSON con métricas del candidato.")
    append_parser.add_argument("--dataset", required=True, help="Ruta dataset usado para evaluación.")
    append_parser.add_argument("--coverage", type=float, default=None, help="Cobertura operativa opcional.")
    append_parser.add_argument("--test-events", type=int, default=None, help="Eventos de test opcionales.")
    append_parser.add_argument("--model-path", default="", help="Ruta model.pkl (si se omite, se infiere).")
    append_parser.add_argument(
        "--gate-pass",
        choices=["auto", "true", "false"],
        default="auto",
        help="`auto` calcula gate según tipo de candidato: modelo (Top-2/bal_acc/coverage) o política (Top-1+coherencia+no daño).",
    )
    append_parser.add_argument(
        "--promotion-decision",
        choices=["auto", "promote", "keep_baseline"],
        default="auto",
        help="`auto`: promote solo si gate_pass=true.",
    )

    return parser.parse_args()


# FUNCIÓN DE CARGA JSON
def load_json(path: Path) -> dict[str, Any]:
    """
    Carga un JSON desde disco y devuelve dict.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# FUNCIÓN DE HASH SHA256
def compute_sha256(path: Path) -> str:
    """
    Calcula SHA256 de un archivo para trazabilidad de snapshot.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo para hash: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# FUNCIÓN DE CAST SEGURO A FLOAT
def to_float(value: Any) -> float | None:
    """
    Convierte un valor a float o devuelve `None` si no es convertible.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


# FUNCIÓN DE CAST SEGURO A INT
def to_int(value: Any) -> int | None:
    """
    Convierte un valor a int o devuelve `None` si no es convertible.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


# FUNCIÓN DE SERIALIZACIÓN DE FLOAT
def fmt_float(value: float | None) -> str:
    """
    Serializa float con 6 decimales para el CSV oficial.
    """
    if value is None:
        return ""
    return f"{value:.6f}"


# FUNCIÓN DE SERIALIZACIÓN DE BOOL
def fmt_bool(value: bool | None) -> str:
    """
    Serializa bool a `true`/`false`; vacío si `None`.
    """
    if value is None:
        return ""
    return "true" if value else "false"


# FUNCIÓN EXTRACTORA DE MÉTRICAS
def extract_metrics(payload: dict[str, Any]) -> dict[str, float | int | None]:
    """
    Extrae métricas en formato estable desde payload plano o `payload["metrics"]`.
    """
    scope = payload.get("metrics", payload)

    accuracy = to_float(scope.get("accuracy"))
    if accuracy is None:
        accuracy = to_float(scope.get("test_acc"))

    balanced_accuracy = to_float(scope.get("balanced_accuracy"))
    if balanced_accuracy is None:
        balanced_accuracy = to_float(scope.get("test_bal_acc"))
    if balanced_accuracy is None:
        balanced_accuracy = to_float(scope.get("bal_acc"))

    f1_pos = to_float(scope.get("f1_pos"))
    if f1_pos is None:
        f1_pos = to_float(scope.get("test_f1_pos"))

    top1_hit = to_float(scope.get("top1_hit"))
    top2_hit = to_float(scope.get("top2_hit"))
    coverage = to_float(scope.get("coverage"))
    test_events = to_int(scope.get("test_events"))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_pos": f1_pos,
        "top1_hit": top1_hit,
        "top2_hit": top2_hit,
        "coverage": coverage,
        "test_events": test_events,
    }


def get_metrics_scope(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Devuelve el bloque `metrics` si existe; si no, devuelve el payload original.
    """
    scope = payload.get("metrics", payload)
    if isinstance(scope, dict):
        return scope
    return payload


def is_policy_candidate(model_variant: str, metrics_payload: dict[str, Any]) -> bool:
    """
    Detecta candidatos de postinferencia (modelo base + capa determinista).
    """
    scope_tag = str(metrics_payload.get("scope", "")).strip().lower()
    policy_tag = str(metrics_payload.get("policy", "")).strip()
    variant = str(model_variant).strip().upper()
    if scope_tag == "after_policy":
        return True
    if policy_tag != "":
        return True
    return variant.startswith("POLICY_") or variant.startswith("BASELINE_WITH_DETERMINISTIC_LAYER_")


# FUNCIÓN DE LECTURA DEL CSV OFICIAL
def read_registry_rows(path: Path) -> list[dict[str, str]]:
    """
    Lee el CSV oficial a lista de diccionarios; devuelve lista vacía si no existe.
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return [dict(row) for row in reader]


# FUNCIÓN DE ESCRITURA DEL CSV OFICIAL
def write_registry_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """
    Persiste filas del registro con esquema y orden de columnas estable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe_row = {column: row.get(column, "") for column in CSV_COLUMNS}
            writer.writerow(safe_row)


# FUNCIÓN DE BÚSQUEDA DE BASELINE
def get_baseline_row(rows: list[dict[str, str]]) -> dict[str, str]:
    """
    Devuelve la fila baseline del registro oficial.
    """
    for row in rows:
        if str(row.get("model_role", "")).strip() == "baseline":
            return row
    raise ValueError("No existe fila baseline en el CSV oficial. Ejecuta primero `init-baseline`.")


# FUNCIÓN DE VALIDACIÓN DE DUPLICADOS
def assert_unique_key(rows: list[dict[str, str]], run_id: str, model_variant: str, cutoff_date: str) -> None:
    """
    Verifica unicidad por clave compuesta: `run_id + model_variant + cutoff_date`.
    """
    for row in rows:
        if (
            str(row.get("run_id", "")) == run_id
            and str(row.get("model_variant", "")) == model_variant
            and str(row.get("cutoff_date", "")) == cutoff_date
        ):
            raise ValueError(
                "Clave duplicada en registro oficial: "
                f"run_id={run_id}, model_variant={model_variant}, cutoff_date={cutoff_date}"
            )


# FUNCIÓN CONSTRUCTORA DE FILA BASE
def build_base_row(
    *,
    run_id: str,
    day_id: str,
    model_variant: str,
    model_role: str,
    dataset_name: str,
    dataset_snapshot_hash: str,
    cutoff_date: str,
    metrics: dict[str, float | int | None],
    selection_rule: str,
    model_path: str,
    metadata_path: str,
    metrics_source: str,
) -> dict[str, str]:
    """
    Construye fila normalizada para el CSV oficial.
    """
    return {
        "run_id": run_id,
        "day_id": day_id,
        "model_variant": model_variant,
        "model_role": model_role,
        "dataset_name": dataset_name,
        "dataset_snapshot_hash": dataset_snapshot_hash,
        "cutoff_date": cutoff_date,
        "test_events": str(metrics.get("test_events") or ""),
        "accuracy": fmt_float(to_float(metrics.get("accuracy"))),
        "balanced_accuracy": fmt_float(to_float(metrics.get("balanced_accuracy"))),
        "f1_pos": fmt_float(to_float(metrics.get("f1_pos"))),
        "top1_hit": fmt_float(to_float(metrics.get("top1_hit"))),
        "top2_hit": fmt_float(to_float(metrics.get("top2_hit"))),
        "coverage": fmt_float(to_float(metrics.get("coverage"))),
        "delta_top2_vs_baseline": "",
        "delta_bal_acc_vs_baseline": "",
        "delta_coverage_vs_baseline": "",
        "gate_top2_ok": "",
        "gate_bal_acc_ok": "",
        "gate_coverage_ok": "",
        "gate_pass": "",
        "promotion_decision": "keep_baseline",
        "selection_rule": selection_rule,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metrics_source": metrics_source,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# FUNCIÓN DE COMPOSICIÓN DE DATOS DE ENTRADA
def resolve_input_bundle(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, float | int | None], Path]:
    """
    Resuelve metadata, métricas y dataset path desde argumentos CLI.
    """
    metadata_path = Path(args.metadata).resolve()
    metadata = load_json(metadata_path)
    metadata_metrics = extract_metrics(metadata)

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe dataset: {dataset_path}")

    metrics = dict(metadata_metrics)
    if getattr(args, "coverage", None) is not None:
        metrics["coverage"] = args.coverage
    if getattr(args, "test_events", None) is not None:
        metrics["test_events"] = args.test_events

    return metadata, metrics, dataset_path


# FUNCIÓN `INIT-BASELINE`
def init_baseline(args: argparse.Namespace) -> None:
    """
    Crea el CSV oficial y registra la fila baseline de referencia.
    """
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"El CSV oficial ya existe: {output_path}. Usa --overwrite si quieres recrearlo."
        )

    metadata, metrics, dataset_path = resolve_input_bundle(args)
    metrics_source = str(Path(args.metadata).resolve())
    if args.metrics_json:
        metrics_payload = load_json(Path(args.metrics_json).resolve())
        payload_metrics = extract_metrics(metrics_payload)
        for key, value in payload_metrics.items():
            if value is not None:
                metrics[key] = value
        metrics_source = str(Path(args.metrics_json).resolve())

    cutoff_date = str(metadata.get("cutoff_date", "")).strip()
    if cutoff_date == "":
        raise ValueError("metadata.json no contiene `cutoff_date`.")

    selection_rule = str(metadata.get("selection_rule", "")).strip()
    model_path = args.model_path.strip() or str((Path(args.metadata).resolve().parent / "model.pkl"))

    row = build_base_row(
        run_id=args.run_id,
        day_id=args.day_id,
        model_variant=args.model_variant,
        model_role="baseline",
        dataset_name=dataset_path.name,
        dataset_snapshot_hash=compute_sha256(dataset_path),
        cutoff_date=cutoff_date,
        metrics=metrics,
        selection_rule=selection_rule,
        model_path=model_path,
        metadata_path=str(Path(args.metadata).resolve()),
        metrics_source=metrics_source,
    )

    row["delta_top2_vs_baseline"] = "0.000000"
    row["delta_bal_acc_vs_baseline"] = "0.000000"
    row["delta_coverage_vs_baseline"] = "0.000000"
    row["gate_pass"] = "NA"
    row["promotion_decision"] = "keep_baseline"

    write_registry_rows(output_path, [row])
    print(f"OK · baseline inicializado en {output_path}")


# FUNCIÓN `APPEND-CANDIDATE`
def append_candidate(args: argparse.Namespace) -> None:
    """
    Agrega un candidato al CSV oficial y calcula deltas contra baseline.
    """
    output_path = Path(args.output).resolve()
    rows = read_registry_rows(output_path)
    if not rows:
        raise FileNotFoundError(f"No existe registro oficial o está vacío: {output_path}")

    baseline_row = get_baseline_row(rows)
    metadata, metrics, dataset_path = resolve_input_bundle(args)
    metrics_payload = load_json(Path(args.metrics_json).resolve())
    metrics_from_payload = extract_metrics(metrics_payload)

    for key, value in metrics_from_payload.items():
        if value is not None:
            metrics[key] = value

    cutoff_date = str(metadata.get("cutoff_date", "")).strip()
    if cutoff_date == "":
        raise ValueError("metadata.json del candidato no contiene `cutoff_date`.")

    assert_unique_key(rows, args.run_id, args.model_variant, cutoff_date)

    baseline_top1 = to_float(baseline_row.get("top1_hit"))
    baseline_top2 = to_float(baseline_row.get("top2_hit"))
    baseline_bal = to_float(baseline_row.get("balanced_accuracy"))
    baseline_cov = to_float(baseline_row.get("coverage"))

    candidate_top1 = to_float(metrics.get("top1_hit"))
    candidate_top2 = to_float(metrics.get("top2_hit"))
    candidate_bal = to_float(metrics.get("balanced_accuracy"))
    candidate_cov = to_float(metrics.get("coverage"))

    delta_top1 = None if baseline_top1 is None or candidate_top1 is None else candidate_top1 - baseline_top1
    delta_top2 = None if baseline_top2 is None or candidate_top2 is None else candidate_top2 - baseline_top2
    delta_bal = None if baseline_bal is None or candidate_bal is None else candidate_bal - baseline_bal
    delta_cov = None if baseline_cov is None or candidate_cov is None else candidate_cov - baseline_cov

    payload_scope = get_metrics_scope(metrics_payload)
    policy_candidate = is_policy_candidate(args.model_variant, metrics_payload)

    if policy_candidate:
        gate_top2_ok = None if delta_top2 is None else (delta_top2 >= POLICY_GATE_MIN_DELTA_TOP2)
        gate_bal_ok = None if delta_bal is None else (delta_bal >= POLICY_GATE_MIN_DELTA_BAL_ACC)
        gate_cov_ok = None if delta_cov is None else (delta_cov >= POLICY_GATE_MIN_DELTA_COVERAGE)

        coherence_before = to_float(payload_scope.get("coherence_before"))
        coherence_after = to_float(payload_scope.get("coherence_after"))
        gate_coherence_ok = (
            coherence_after >= coherence_before
            if coherence_before is not None and coherence_after is not None
            else False
        )
        overrides_harmed = to_int(payload_scope.get("overrides_harmed"))
        gate_harmed_ok = overrides_harmed == 0 if overrides_harmed is not None else False
        gate_top1_ok = None if delta_top1 is None else (delta_top1 > 0.0)
        gate_checks = [gate_top1_ok, gate_top2_ok, gate_bal_ok, gate_cov_ok, gate_coherence_ok, gate_harmed_ok]
    else:
        gate_top2_ok = None if delta_top2 is None else (delta_top2 >= GATE_MIN_DELTA_TOP2)
        gate_bal_ok = None if delta_bal is None else (delta_bal >= GATE_MIN_DELTA_BAL_ACC)
        gate_cov_ok = None if delta_cov is None else (delta_cov >= GATE_MIN_DELTA_COVERAGE)
        gate_checks = [gate_top2_ok, gate_bal_ok, gate_cov_ok]

    if args.gate_pass == "true":
        gate_pass = True
    elif args.gate_pass == "false":
        gate_pass = False
    else:
        gate_pass = all(check is True for check in gate_checks)

    if args.promotion_decision == "promote":
        promotion_decision = "promote"
    elif args.promotion_decision == "keep_baseline":
        promotion_decision = "keep_baseline"
    else:
        promotion_decision = "promote" if gate_pass else "keep_baseline"

    if policy_candidate:
        selection_rule = (
            "policy_gate(top1>baseline & top2>=baseline & bal_acc>=baseline & "
            "coverage>=baseline-0.005 & coherence_after>=before & overrides_harmed==0)"
        )
    else:
        selection_rule = str(metadata.get("selection_rule", "")).strip()
    model_path = args.model_path.strip() or str((Path(args.metadata).resolve().parent / "model.pkl"))

    row = build_base_row(
        run_id=args.run_id,
        day_id=args.day_id,
        model_variant=args.model_variant,
        model_role="candidate",
        dataset_name=dataset_path.name,
        dataset_snapshot_hash=compute_sha256(dataset_path),
        cutoff_date=cutoff_date,
        metrics=metrics,
        selection_rule=selection_rule,
        model_path=model_path,
        metadata_path=str(Path(args.metadata).resolve()),
        metrics_source=str(Path(args.metrics_json).resolve()),
    )

    row["delta_top2_vs_baseline"] = fmt_float(delta_top2)
    row["delta_bal_acc_vs_baseline"] = fmt_float(delta_bal)
    row["delta_coverage_vs_baseline"] = fmt_float(delta_cov)
    row["gate_top2_ok"] = fmt_bool(gate_top2_ok)
    row["gate_bal_acc_ok"] = fmt_bool(gate_bal_ok)
    row["gate_coverage_ok"] = fmt_bool(gate_cov_ok)
    row["gate_pass"] = fmt_bool(gate_pass)
    row["promotion_decision"] = promotion_decision

    rows.append(row)
    write_registry_rows(output_path, rows)
    print(f"OK · candidato agregado en {output_path}")


# FUNCIÓN MAIN
def main() -> None:
    """
    Orquesta la ejecución de CLI para el registro final baseline vs candidatos.
    """
    args = parse_args()
    if args.command == "init-baseline":
        init_baseline(args)
        return
    if args.command == "append-candidate":
        append_candidate(args)
        return
    raise ValueError(f"Comando no soportado: {args.command}")


# MAIN GUARD
if __name__ == "__main__":
    main()
