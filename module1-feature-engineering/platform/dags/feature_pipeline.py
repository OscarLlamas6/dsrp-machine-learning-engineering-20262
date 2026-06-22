"""Reusable feature-engineering logic for the Module 1 Airflow DAG.

Keeping the *what* (feature logic) separate from the *when/how* (the DAG wiring)
makes the pipeline testable on its own:

    python -c "from feature_pipeline import build_features; build_features(...)"

The functions here mirror the transforms taught in notebook 01 and produce a
parquet file whose schema matches ``feature_repo/features.py`` so that
``feast apply`` / ``feast materialize`` work unchanged.

Dataset: Kaggle Ames "House Prices". Task: regression on ``SalePrice``.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --- Paths shared with the containers (see docker-compose volumes) -----------
# Raw landing zone (named volume), and the writable Feast repo the DAG manages.
DATA_DIR = Path(os.environ.get("PIPELINE_DATA_DIR", "/opt/airflow/data"))
FEAST_REPO_DIR = Path(os.environ.get("FEAST_REPO_DIR", "/opt/airflow/feast_repo"))
FEAST_REPO_SRC = Path(os.environ.get("FEAST_REPO_SRC", "/opt/airflow/feature_repo_src"))

# Where the Ames CSV is mounted inside the containers (see docker-compose).
HOUSING_CSV = Path(os.environ.get("HOUSING_CSV", "/opt/airflow/housing_data/housing_train.csv"))

RAW_PATH = DATA_DIR / "housing_raw.parquet"
FEATURES_PATH = FEAST_REPO_DIR / "data" / "housing_features.parquet"

# Ordinal quality scale: Po < Fa < TA < Gd < Ex; NaN/None means "absent" -> 0.
QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}


# ---------------------------------------------------------------------------
# 1. Extract
# ---------------------------------------------------------------------------
def extract() -> str:
    """Load the raw Ames Housing CSV, with an offline synthetic fallback.

    Reads ``HOUSING_CSV`` (mounted from the repo-root ``data/`` folder). If the
    file is missing, builds deterministic synthetic data with the key columns so
    the DAG still runs. Returns the path to the raw parquet for downstream tasks.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if HOUSING_CSV.exists():
        df = pd.read_csv(HOUSING_CSV)
        source = f"csv:{HOUSING_CSV}"
    else:  # offline / missing file -> deterministic synthetic data
        rng = np.random.default_rng(42)
        n = 1460
        neighborhoods = [
            "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert",
            "NridgHt", "Sawyer", "NWAmes", "BrkSide", "Crawfor", "Mitchel",
        ]
        quality_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
        overall_qual = rng.integers(3, 10, n)
        gr_liv = (overall_qual * 180 + rng.normal(400, 250, n)).clip(400, 5500)
        df = pd.DataFrame(
            {
                "Id": np.arange(1, n + 1),
                "OverallQual": overall_qual,
                "GrLivArea": gr_liv.round().astype(int),
                "TotalBsmtSF": (gr_liv * 0.6 + rng.normal(0, 150, n)).clip(0, 3000).round().astype(int),
                "1stFlrSF": (gr_liv * 0.65 + rng.normal(0, 120, n)).clip(300, 3000).round().astype(int),
                "GarageCars": rng.integers(0, 4, n),
                "GarageArea": rng.integers(0, 900, n),
                "YearBuilt": rng.integers(1900, 2010, n),
                "LotArea": rng.gamma(2.0, 5000, n).clip(1300, 60000).round().astype(int),
                "FullBath": rng.integers(0, 4, n),
                "Neighborhood": rng.choice(neighborhoods, n),
                "ExterQual": rng.choice(quality_levels, n, p=[0.0, 0.0, 0.6, 0.35, 0.05]),
            }
        )
        df["SalePrice"] = (
            overall_qual * 22000 + df["GrLivArea"] * 55 + rng.normal(0, 25000, n)
        ).clip(40000, 600000).round().astype(int)
        source = "synthetic"

    df.to_parquet(RAW_PATH, index=False)
    print(f"[extract] source={source} rows={len(df)} -> {RAW_PATH}")
    return str(RAW_PATH)


# ---------------------------------------------------------------------------
# 2. Prepare the Feast repo (config-per-environment)
# ---------------------------------------------------------------------------
def prepare_feast_repo() -> str:
    """Materialize a writable, container-targeted copy of the Feast repo.

    The notebook uses ``feature_repo`` with ``localhost:6379``. Inside Docker,
    the online store is reachable as the ``redis`` service instead, so we render a
    fresh ``feature_store.yaml`` here pointing at the right host. This is the same
    "one definition, per-environment config" idea a feature store gives you in
    production.
    """
    (FEAST_REPO_DIR / "data").mkdir(parents=True, exist_ok=True)

    # Copy the feature definitions (read-only source -> writable repo).
    if FEAST_REPO_SRC.exists():
        shutil.copy(FEAST_REPO_SRC / "features.py", FEAST_REPO_DIR / "features.py")

    redis_host = os.environ.get("FEAST_REDIS_HOST", "redis")
    redis_port = os.environ.get("FEAST_REDIS_PORT", "6379")

    yaml_text = f"""# Auto-generated by the Airflow DAG for in-container execution.
project: module1_features
provider: local
registry: data/registry.db
online_store:
  type: redis
  connection_string: "{redis_host}:{redis_port}"
offline_store:
  type: file
entity_key_serialization_version: 3
"""
    (FEAST_REPO_DIR / "feature_store.yaml").write_text(yaml_text)
    print(f"[prepare] feast repo ready at {FEAST_REPO_DIR} (online={redis_host}:{redis_port})")
    return str(FEAST_REPO_DIR)


# ---------------------------------------------------------------------------
# 3. Transform (the actual feature engineering)
# ---------------------------------------------------------------------------
def build_features() -> str:
    """Engineer model-ready housing features and write them where Feast expects them.

    The output schema matches ``features.py``: house_id (entity), event_timestamp
    + created (Feast bookkeeping), the feature columns (snake_case, with
    digit-leading columns renamed) and ``sale_price`` (the regression target,
    carried in the parquet so train_model can build the training set).
    """
    df = pd.read_parquet(RAW_PATH)

    out = pd.DataFrame()
    # Entity key (from the Ames `Id`, or a 1..n range if missing).
    if "Id" in df.columns:
        out["house_id"] = df["Id"].astype("int64")
    else:
        out["house_id"] = np.arange(1, len(df) + 1, dtype="int64")

    # Regression target (no es una feature servida; viaja en el parquet para que
    # la tarea train_model pueda construir el set de entrenamiento).
    if "SalePrice" in df.columns:
        out["sale_price"] = df["SalePrice"].astype("float32")
    else:
        out["sale_price"] = np.float32(0.0)

    # Numeric features (impute "absent" feature columns with 0; see notebook 01).
    out["overall_qual"] = df["OverallQual"].astype("int64")
    out["gr_liv_area"] = df["GrLivArea"].astype("float32")
    out["total_bsmt_sf"] = df["TotalBsmtSF"].fillna(0).astype("float32")
    out["first_flr_sf"] = df["1stFlrSF"].astype("float32")          # renamed snake_case
    out["garage_cars"] = df["GarageCars"].fillna(0).astype("int64")
    out["garage_area"] = df["GarageArea"].fillna(0).astype("float32")
    out["year_built"] = df["YearBuilt"].astype("int64")
    out["lot_area"] = df["LotArea"].astype("float32")
    out["full_bath"] = df["FullBath"].astype("int64")

    # Nominal categorical kept as string (Feast serves it as-is; encode at train time).
    out["neighborhood"] = df["Neighborhood"].fillna("None").astype(str)

    # Ordinal quality grade -> meaningful integer (Po<Fa<TA<Gd<Ex; NaN/None=0).
    out["exter_qual_ord"] = (
        df["ExterQual"].fillna("None").map(QUAL_MAP).fillna(0).astype("int64")
    )

    # Feast bookkeeping columns.
    now = datetime.now(timezone.utc)
    out["event_timestamp"] = pd.Timestamp(now)
    out["created"] = pd.Timestamp(now)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATURES_PATH, index=False)
    print(f"[transform] wrote {len(out)} rows, {out.shape[1]} cols -> {FEATURES_PATH}")
    return str(FEATURES_PATH)


# ---------------------------------------------------------------------------
# 4. Validate (a tiny data-quality gate)
# ---------------------------------------------------------------------------
def validate_features() -> dict:
    """Fail the DAG early if the feature table is malformed."""
    df = pd.read_parquet(FEATURES_PATH)

    expected = {
        "house_id", "sale_price", "overall_qual", "gr_liv_area", "total_bsmt_sf",
        "first_flr_sf", "garage_cars", "garage_area", "year_built", "lot_area",
        "full_bath", "neighborhood", "exter_qual_ord",
        "event_timestamp", "created",
    }
    missing = expected - set(df.columns)
    assert not missing, f"missing columns: {missing}"
    assert len(df) > 0, "feature table is empty"
    assert df["house_id"].is_unique, "house_id must be unique"
    key_nulls = df[["house_id", "event_timestamp"]].isnull().any().any()
    assert not key_nulls, "null entity key or timestamp"

    report = {"rows": int(len(df)), "cols": int(df.shape[1])}
    print(f"[validate] OK {report}")
    return report


# ---------------------------------------------------------------------------
# 5. Train (cierra el ciclo: feature store -> training set -> modelo evaluado)
# ---------------------------------------------------------------------------
def train_model() -> dict:
    """Entrena y evalua un regresor con las features materializadas.

    Aqui se cierra el ciclo del modulo: el mismo parquet que alimenta el offline
    store de Feast se usa para construir el training set, se entrena un modelo de
    regresion que predice ``sale_price`` (sobre ``log1p`` del precio, como en
    Kaggle), se evalua (RMSE / MAE / R2 y el RMSE sobre log(price)) y el modelo
    entrenado se guarda en disco (joblib) en el volumen de datos.

    El seguimiento de experimentos (experiment tracking) y el registro de modelos
    con MLflow se introducen en el Modulo 2; aqui nos quedamos en entrenar +
    evaluar + persistir, registrando las metricas en los logs de la tarea.
    """
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    df = pd.read_parquet(FEATURES_PATH)

    target = "sale_price"
    numeric = [
        "overall_qual", "gr_liv_area", "total_bsmt_sf", "first_flr_sf",
        "garage_cars", "garage_area", "year_built", "lot_area",
        "full_bath", "exter_qual_ord",
    ]
    categorical = ["neighborhood"]

    X = df[numeric + categorical].copy()
    y = df[target].astype(float)
    y_log = np.log1p(y)                       # entrenamos en escala logaritmica

    X_train, X_test, y_train, y_test, ylog_train, ylog_test = train_test_split(
        X, y, y_log, test_size=0.25, random_state=42
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("pre", pre),
            ("reg", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    model.fit(X_train, ylog_train)            # objetivo = log1p(sale_price)
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    rmse_log = float(np.sqrt(mean_squared_error(ylog_test, pred_log)))  # metrica Kaggle

    # Metricas a los logs de la tarea (sin servidor de tracking).
    print(f"[train] RMSE={rmse:,.0f}  MAE={mae:,.0f}  R2={r2:.4f}  RMSE_log={rmse_log:.4f}")

    # Persistimos el modelo entrenado en el volumen de datos.
    model_path = DATA_DIR / "housing_model.joblib"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    report = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmse_log": rmse_log,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_path": str(model_path),
    }
    print(f"[train] {report}")
    return report
