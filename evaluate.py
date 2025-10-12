"""Stage `evaluate` (DVC): corre el eval agnóstico sobre los metadatos ODIR-5K (las imágenes
las lee `train.py` de `$SEI_ODIR_CACHE`) y persiste el modelo como out cacheado (Art.15). La
medición (SDK) vive en compliance_eval; el tratamiento (variante V1/V2) en train.py."""

import joblib

import compliance_eval
import train

_, model = compliance_eval.run(train.build_model)
joblib.dump(model, "model.pkl")
