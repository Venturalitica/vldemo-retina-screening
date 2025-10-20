"""Tratamiento del cribador DR-referible — modelo y su VARIANTE (V1/V2). El arco honesto
rojo→verde lo gobierna el parámetro `mitigate` (igual contrato que loan-scoring):

  mitigate=False (V1): resnet18 congelado (features 512-d) → LogReg plano. Recall DR bajo
                       (≈0.41 sobre ODIR-5K): el cribador pierde casos → GATE clínico ROJO.
  mitigate=True  (V2): el MISMO pipeline con `class_weight='balanced'` + umbral de decisión
                       rebajado a 0.30 → recupera la sensibilidad por encima del bar clínico
                       0.80 (recall ≈0.88) a costa de exactitud (trade-off de cribado).

Arquitectura (rápida en GPU, honesta):
  · resnet18 de torchvision preentrenado en ImageNet, CONGELADO (eval mode); se extraen las
    features de la penúltima capa (512-d) procesando las imágenes en lotes sobre la GPU.
  · Una LogisticRegression de scikit-learn actúa como cabeza clasificadora.
  · Las imágenes provienen de `$SEI_ODIR_CACHE/preprocessed_images/` (NO versionadas, ~2 GB).
  · Las features se cachean en `/var/tmp/odir_features_v1.npz` para evitar reextracción.

build_model(df, seed, mitigate) -> (cohort_test_con_prediction, modelo, X) — mismo contrato
que loan-scoring (cohort = subconjunto de TEST no visto; X = features completas para robustez).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "dr_referable"
SEX_COL = "patient_sex"
AGE_COL = "patient_age"
AGE_GROUP_COL = "age_group"
AGE_CUTOFF = 50

_FEATURE_CACHE = "/var/tmp/odir_features_v1.npz"
_BATCH_SIZE = 64
_IMG_SIZE = 224  # resnet18 espera 224×224
_THRESHOLD_MITIGATED = 0.30  # umbral V2 calibrado sobre ODIR-5K real (recall ≈0.88)


def _load_resnet18():
    """Carga resnet18 preentrenado (ImageNet), elimina la cabeza clasificadora, congela todos
    los pesos y lo pone en eval mode. Usa la GPU si está disponible."""
    import torch
    import torchvision.models as models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()  # quitar la FC final → salida 512-d (avgpool)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone, device


def _cache_key(image_dir: str, n: int) -> str:
    """Clave de caché derivada del basename del directorio y el número de imágenes (portátil
    entre runs)."""
    return f"{Path(image_dir).name}:{n}"


def extract_features_gpu(image_dir: str, filenames: pd.Series) -> np.ndarray:
    """Extrae features de 512-d con resnet18 congelado sobre la GPU. Lee las imágenes desde
    `image_dir/<filename>`, las normaliza con los estadísticos ImageNet y las procesa en lotes.
    Carga desde `_FEATURE_CACHE` si coincide la clave; en otro caso extrae y guarda."""
    import torch
    from PIL import Image as PILImage
    from torchvision import transforms

    cache_key = _cache_key(image_dir, len(filenames))
    cache_path = Path(_FEATURE_CACHE)
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True)
            if str(cached.get("key", "")) == cache_key:
                print(f"[train.py] Cargando features desde caché: {cache_path}", flush=True)
                return cached["X"]
        except Exception:
            pass

    backbone, device = _load_resnet18()
    preprocess = transforms.Compose([
        transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    fnames = filenames.tolist()
    n = len(fnames)
    all_feats = []
    with torch.no_grad():
        for start in range(0, n, _BATCH_SIZE):
            batch_files = fnames[start:start + _BATCH_SIZE]
            tensors = []
            for fname in batch_files:
                img_path = Path(image_dir) / fname
                try:
                    img = PILImage.open(img_path).convert("RGB")
                    tensors.append(preprocess(img))
                except Exception as e:
                    print(f"[train.py] AVISO: no se pudo leer {img_path}: {e}", flush=True)
                    tensors.append(torch.zeros(3, _IMG_SIZE, _IMG_SIZE))
            batch = torch.stack(tensors).to(device)
            feats = backbone(batch).cpu().numpy()  # (B, 512)
            all_feats.append(feats)
            if (start // _BATCH_SIZE) % 10 == 0:
                print(f"[train.py] Lote {start // _BATCH_SIZE + 1} / {(n + _BATCH_SIZE - 1) // _BATCH_SIZE}", flush=True)

    X = np.vstack(all_feats).astype(np.float32)
    np.savez_compressed(cache_path, X=X, key=cache_key)
    print(f"[train.py] Features guardadas en caché: {cache_path}  shape={X.shape}", flush=True)
    return X


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra al convenio interno + materializa patient_id / age_group si faltan."""
    col_map = {"Patient Sex": SEX_COL, "Patient Age": AGE_COL, "D": TARGET, "ID": "patient_id"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "patient_id" not in df.columns:
        df = df.copy()
        df["patient_id"] = df.index.astype(str)
    if AGE_GROUP_COL not in df.columns:
        df = df.copy()
        df[AGE_GROUP_COL] = np.where(df[AGE_COL].astype(int) >= AGE_CUTOFF, "mayor", "joven")
    return df


def build_model(df: pd.DataFrame, seed: int, mitigate: bool = False):
    """Entrena el cribador DR-referible sobre features resnet18.

    mitigate=False (V1): LogReg plano, umbral 0.50.
    mitigate=True  (V2): LogReg `class_weight='balanced'`, umbral 0.30 (recall DR ≥ 0.80).

    Devuelve `(cohort_test, model, X)` — cohort_test = partición de TEST (20%, stratify,
    split determinista por `seed`) con la columna `prediction`; X = features completas.
    """
    cache_dir = os.environ.get("SEI_ODIR_CACHE", "/var/tmp/odir-cache")
    image_dir = str(Path(cache_dir) / "preprocessed_images")

    df = _normalize_columns(df)
    if "filename" not in df.columns:
        raise KeyError(f"Columna 'filename' no encontrada; columnas: {list(df.columns)}")

    X = extract_features_gpu(image_dir, df["filename"])
    y = df[TARGET].astype(int).to_numpy()

    idx = np.arange(len(df))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=seed,
            C=0.1,
            class_weight="balanced" if mitigate else None,
        ),
    )
    model.fit(X[idx_tr], y_tr)

    df_reset = df.reset_index(drop=True)
    test_cohort = df_reset.iloc[idx_te].copy().reset_index(drop=True)
    if mitigate:
        probas = model.predict_proba(X[idx_te])[:, 1]
        test_cohort["prediction"] = (probas >= _THRESHOLD_MITIGATED).astype(int)
    else:
        test_cohort["prediction"] = model.predict(X[idx_te]).astype(int)

    rec = recall_score(y_te, test_cohort["prediction"].to_numpy(), zero_division=0)
    prec = precision_score(y_te, test_cohort["prediction"].to_numpy(), zero_division=0)
    acc = accuracy_score(y_te, test_cohort["prediction"].to_numpy())
    variant = "V2 mitigado" if mitigate else "V1 sin mitigar"
    print(
        f"[train.py {variant}] train={len(idx_tr)} test={len(idx_te)}  "
        f"recall={rec:.3f} precision={prec:.3f} accuracy={acc:.3f}  "
        f"DR+ test={int(y_te.sum())} pred+ test={int(test_cohort['prediction'].sum())}",
        flush=True,
    )
    return test_cohort, model, X
