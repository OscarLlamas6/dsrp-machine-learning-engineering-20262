"""Feast feature definitions for Module 1.

This file is discovered by `feast apply`. It declares:

  * an Entity            -> the join key features are attached to (house_id)
  * a FileSource         -> where the raw/engineered rows live (a parquet file)
  * a FeatureView        -> a named, reusable group of features with a TTL

The parquet file is produced by notebook 03 (or by running this module's
data-prep step / Airflow DAG). Each row must contain the entity column, an event
timestamp column, and one column per feature.

Dataset: Kaggle Ames "House Prices". Task: regression on `SalePrice`. Note the
snake_case feature names (Feast cannot use digit-leading columns like
`1stFlrSF`, so it becomes `first_flr_sf`).

Compatible with Feast >= 0.40 (current API: Entity / FeatureView / FileSource /
Field, types imported from feast.types).
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, FileSource, Field, ValueType
from feast.types import Float32, Int64, String

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve the parquet path relative to this file so `feast apply` works no
# matter the current working directory.
DATA_DIR = Path(__file__).resolve().parent / "data"
PARQUET_PATH = str(DATA_DIR / "housing_features.parquet")

# ---------------------------------------------------------------------------
# Entity: the "thing" we attach features to.
# ---------------------------------------------------------------------------
house = Entity(
    name="house",
    join_keys=["house_id"],
    value_type=ValueType.INT64,
    description="An Ames house, identified by house_id (from the Kaggle `Id`).",
)

# ---------------------------------------------------------------------------
# Source: the offline data backing the features (a parquet file).
# ---------------------------------------------------------------------------
housing_source = FileSource(
    name="housing_source",
    path=PARQUET_PATH,
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
    description="Engineered Ames Housing features written by the Module 1 pipeline.",
)

# ---------------------------------------------------------------------------
# FeatureView: a reusable, named group of features tied to an entity + source.
# TTL controls how far back materialization/serving will look for a value.
# ---------------------------------------------------------------------------
house_features = FeatureView(
    name="house_features",
    entities=[house],
    ttl=timedelta(days=365),
    schema=[
        Field(name="overall_qual", dtype=Int64),
        Field(name="gr_liv_area", dtype=Float32),
        Field(name="total_bsmt_sf", dtype=Float32),
        Field(name="first_flr_sf", dtype=Float32),
        Field(name="garage_cars", dtype=Int64),
        Field(name="garage_area", dtype=Float32),
        Field(name="year_built", dtype=Int64),
        Field(name="lot_area", dtype=Float32),
        Field(name="full_bath", dtype=Int64),
        Field(name="neighborhood", dtype=String),
        Field(name="exter_qual_ord", dtype=Int64),
        Field(name="sale_price", dtype=Float32),
    ],
    online=True,
    source=housing_source,
    tags={"team": "ml-platform", "module": "1"},
)
