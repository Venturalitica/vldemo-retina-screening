"""Núcleo del eval de retina-screening — AGNÓSTICO al framework MLOps (no importa DVC/MLflow).
Carga los metadatos REALES de ODIR-5K (`data/full_df.csv`, versionado; `data/odir.croissant.json`
documenta la procedencia RAI), entrena el cribador DR-referible vía `build_model`, corre la SDK
venturalitica (`vl.monitor` abre la sesión + los probes —incl. BOMProbe—; `vl.enforce` evalúa el
OSCAL compilado en fase Art.15), PROMUEVE el bom.json del run a `.venturalitica/bom.json` (ruta que
lee el motor) y vuelca `metrics.json` plano `{control_id: {value, power}}`. NO juzga: el veredicto
autoritativo lo pone el motor venth (Rust) contra el MISMO OSCAL.

CLUSTER BOOTSTRAP POR PACIENTE: ODIR-5K tiene 2 filas/paciente (ojo izquierdo + derecho). La
política OSCAL declara `cluster: patient_id` en cada control de validación → el SDK resamplea
pacientes completos (no ojos sueltos) y reporta `n_clusters` en cada `power` (power-stats §8.1).
"""

import os

os.environ.setdefault("VENTURALITICA_NO_ANALYTICS", "1")  # sin telemetría en CI

import contextlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import venturalitica as vl

LABELS_CSV = "data/full_df.csv"
OSCAL = "shared_data/policies/assessment_plan.oscal.yaml"
PARAMS = "params.yaml"
METRICS = "metrics.json"
BOM_ROOT = ".venturalitica/bom.json"
RUNS_DIR = Path(".venturalitica/runs")

TARGET = "dr_referable"
PREDICTION = "prediction"
SEX_COL = "patient_sex"
AGE_COL = "patient_age"
AGE_GROUP_COL = "age_group"
CLUSTER = "patient_id"
AGE_CUTOFF = 50  # joven < 50 a. / mayor ≥ 50 a. (distribución ODIR-5K)


def load_odir(csv_path: str = LABELS_CSV) -> pd.DataFrame:
    """Carga los metadatos ODIR-5K REALES desde `data/full_df.csv` (versionado en el repo).
    Normaliza al convenio interno y añade `patient_id` / `age_group`. Las IMÁGENES no están
    aquí (las lee `train.py` de `$SEI_ODIR_CACHE/preprocessed_images/`); este CSV aporta las
    etiquetas + atributos protegidos para la auditoría de equidad."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path} (etiquetas ODIR-5K versionadas).")

    df = pd.read_csv(path)
    rename_map = {"Patient Age": AGE_COL, "Patient Sex": SEX_COL, "D": TARGET, "ID": "patient_id"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["patient_id"] = (df["patient_id"].astype(str) if "patient_id" in df.columns
                        else df.index.astype(str))
    df[AGE_COL] = df[AGE_COL].astype(int)
    df[AGE_GROUP_COL] = np.where(df[AGE_COL] >= AGE_CUTOFF, "mayor", "joven")
    df[TARGET] = df[TARGET].astype(int)

    print(
        f"[compliance_eval.py] ODIR-5K cargado: {len(df)} filas, "
        f"DR+ = {df[TARGET].sum()} ({df[TARGET].mean()*100:.1f}%), "
        f"Sexo: {df[SEX_COL].value_counts().to_dict()}",
        file=sys.stderr,
    )
    return df


def params() -> dict:
    return yaml.safe_load(open(PARAMS)) or {}


def _control_order(oscal_path: str) -> dict:
    doc = yaml.safe_load(open(oscal_path))
    reqs = doc["component-definition"]["components"][0]["control-implementations"][0][
        "implemented-requirements"
    ]
    return {r["control-id"]: i for i, r in enumerate(reqs)}


def _metric_entry(result) -> float | dict:
    """Una entrada de `metrics.json`: objeto `{value, power}` si el SDK expone el bloque de poder
    (bootstrap, ≥0.6.11), escalar `value` si no. El núcleo Rust acepta ambas formas (untagged)."""
    value = float(result.actual_value)
    power = getattr(result, "power", None)
    return {"value": value, "power": power} if power else value


def _promote_bom() -> None:
    """Promueve el bom.json que BOMProbe dejó en `.venturalitica/runs/<run>/` a la raíz
    `.venturalitica/bom.json` (lo que lee el motor). Elige el run con mtime máximo (misma
    heurística que el CLI push). FAIL-LOUD si no hay ningún bom.json que promover."""
    candidates = sorted(
        (p for p in RUNS_DIR.glob("*/bom.json") if p.parent.name != "latest"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit("compliance_eval: no se generó ningún bom.json en .venturalitica/runs/")
    Path(".venturalitica").mkdir(exist_ok=True)
    shutil.copyfile(candidates[-1], BOM_ROOT)
    print(f"bom → {BOM_ROOT} (desde {candidates[-1]})", file=sys.stderr)


def write_metrics(metrics: dict, path: str = METRICS) -> None:
    json.dump(metrics, open(path, "w"), indent=2)


def run(build_model, df: pd.DataFrame | None = None, oscal_path: str = OSCAL):
    """Orquesta la evaluación sobre datos REALES (ODIR-5K). `build_model(df, seed, mitigate)`
    devuelve `(cohort_test, model, X)` (contrato idéntico a loan-scoring). Mide la fase Art.15
    (validación) contra el OSCAL compilado, con cluster-bootstrap por paciente."""
    if df is None:
        df = load_odir()
    p = params()
    seed = int(p.get("seed", 42))
    mitigate = bool(p.get("mitigate", False))

    with contextlib.redirect_stdout(sys.stderr):
        with vl.monitor(name="retina-screening", label="venth eval"):
            cohort, model, _ = build_model(df, seed, mitigate)  # ENTRENAMIENTO (Art.15)
            # Fase modelo (Art.15): sensibilidad DR (GATE) + paridad demográfica por sexo.
            # Cluster bootstrap por paciente (la cohorte de test lleva `patient_id`).
            model_results = vl.enforce(
                data=cohort,
                policy=oscal_path,
                target=TARGET,
                prediction=PREDICTION,
                phase="validation",
                strict=False,
            )
            order = _control_order(oscal_path)
            results = sorted(model_results, key=lambda r: order.get(r.control_id, 10**6))
            metrics = {r.control_id: _metric_entry(r) for r in results}
        _promote_bom()  # tras cerrar la sesión, el bom.json del run ya existe

    write_metrics(metrics)
    return cohort, model
