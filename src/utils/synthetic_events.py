"""
Synthetic OTT event generator (OPTIONAL module).
Generates simulated viewing sessions clearly labelled as SYNTHETIC.
This is NOT required for the core pipeline.
"""
import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

import random
from datetime import datetime, timedelta
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, TimestampType
)

from src.utils.spark_session import get_spark
from src.utils.logger import get_logger
from config.settings import PARQUET, DATA_SYNTHETIC

logger = get_logger(__name__)

# SYNTHETIC event schema
SYNTHETIC_EVENT_SCHEMA = StructType([
    StructField("event_id",          StringType(),   False),  # SYNTHETIC
    StructField("userId",            IntegerType(),  True),   # from real data
    StructField("movieId",           IntegerType(),  True),   # from real data
    StructField("event_type",        StringType(),   True),   # SYNTHETIC
    StructField("watch_start",       TimestampType(),True),   # SYNTHETIC
    StructField("watch_duration_min",FloatType(),    True),   # SYNTHETIC
    StructField("completion_pct",    FloatType(),    True),   # SYNTHETIC
    StructField("device",            StringType(),   True),   # SYNTHETIC
    StructField("session_id",        StringType(),   True),   # SYNTHETIC
])

DEVICES = ["mobile", "smart_tv", "laptop", "tablet", "desktop"]
EVENT_TYPES = ["play", "pause", "resume", "stop", "complete"]


def generate_synthetic_events(
    spark: SparkSession,
    n_events: int = 10_000,
    seed: int = 42,
) -> DataFrame:
    """
    Generate n_events synthetic OTT viewing events.

    ALL fields except userId and movieId are SYNTHETIC.
    This data is created for demonstration of streaming analytics only.
    It does NOT represent real viewing behaviour.

    Parameters
    ----------
    spark : SparkSession
    n_events : int
        Number of synthetic events to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DataFrame with SYNTHETIC_EVENT_SCHEMA
    """
    logger.warning("Generating SYNTHETIC events (n=%d). These are NOT real data.", n_events)

    # Sample real userId / movieId pairs from ratings
    ratings = spark.read.parquet(PARQUET["ratings"])
    sample_pairs = (
        ratings.select("userId", "movieId")
        .sample(fraction=min(1.0, n_events / 10_000_000), seed=seed)
        .limit(n_events)
        .collect()
    )

    random.seed(seed)
    base_date = datetime(2020, 1, 1)
    events = []
    for i, row in enumerate(sample_pairs):
        start = base_date + timedelta(
            days=random.randint(0, 365*4),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        duration = round(random.uniform(5.0, 180.0), 1)
        completion = round(random.uniform(0.0, 100.0), 1)
        events.append((
            f"evt_{i:08d}",            # SYNTHETIC event_id
            int(row["userId"]),
            int(row["movieId"]),
            random.choice(EVENT_TYPES), # SYNTHETIC
            start,                      # SYNTHETIC
            duration,                   # SYNTHETIC
            completion,                 # SYNTHETIC
            random.choice(DEVICES),     # SYNTHETIC
            f"sess_{i // 3:06d}",      # SYNTHETIC session_id
        ))

    df = spark.createDataFrame(events, schema=SYNTHETIC_EVENT_SCHEMA)
    logger.info("Generated %d SYNTHETIC events.", df.count())
    return df


def save_synthetic_events(spark: SparkSession, n_events: int = 10_000) -> str:
    """Generate and save synthetic events as Parquet."""
    df = generate_synthetic_events(spark, n_events)
    out_path = str(DATA_SYNTHETIC / "ott_events.parquet")
    df.write.mode("overwrite").parquet(out_path)
    logger.info("SYNTHETIC events saved to: %s", out_path)
    return out_path


if __name__ == "__main__":
    spark = get_spark("OTT_Synthetic")
    save_synthetic_events(spark, n_events=50_000)
    spark.stop()
