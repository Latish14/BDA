"""
Phase 1 – Ingestion
Reads all raw CSV files, performs schema validation, and writes Parquet.
"""
import sys
import os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, StringType, TimestampType
)

from src.utils.spark_session import get_spark
from src.utils.logger import get_logger, timer
from config.settings import RAW_FILES, PARQUET

logger = get_logger(__name__)


# ── Explicit schemas (no schema inference for large files) ────────────────────
RATINGS_SCHEMA = StructType([
    StructField("userId",    IntegerType(), True),
    StructField("movieId",   IntegerType(), True),
    StructField("rating",    FloatType(),   True),
    StructField("timestamp", StringType(),  True),   # parsed later
])

MOVIES_SCHEMA = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title",   StringType(),  True),
    StructField("genres",  StringType(),  True),
])

TAGS_SCHEMA = StructType([
    StructField("userId",    IntegerType(), True),
    StructField("movieId",   IntegerType(), True),
    StructField("tag",       StringType(),  True),
    StructField("timestamp", StringType(),  True),
])

LINKS_SCHEMA = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("imdbId",  StringType(),  True),
    StructField("tmdbId",  StringType(),  True),
])

GENOME_TAGS_SCHEMA = StructType([
    StructField("tagId", IntegerType(), True),
    StructField("tag",   StringType(),  True),
])

GENOME_SCORES_SCHEMA = StructType([
    StructField("movieId",   IntegerType(), True),
    StructField("tagId",     IntegerType(), True),
    StructField("relevance", FloatType(),   True),
])


def _read_csv(spark: SparkSession, path: str, schema: StructType) -> DataFrame:
    """Read CSV with explicit schema and header."""
    logger.info("Reading: %s", path)
    return (
        spark.read
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .csv(str(path))
    )


def _parse_timestamp(df: DataFrame, col: str = "timestamp") -> DataFrame:
    """
    Parse timestamp column – MovieLens 20M uses 'YYYY-MM-DD HH:MM:SS' strings.
    Falls back to unix epoch integer if string parsing fails.
    """
    return df.withColumn(
        col,
        F.coalesce(
            F.to_timestamp(F.col(col), "yyyy-MM-dd HH:mm:ss"),
            F.to_timestamp(F.col(col).cast("long")),
        )
    )


def ingest_all(spark: SparkSession) -> dict[str, DataFrame]:
    """
    Read all raw CSV files with explicit schemas.

    Returns
    -------
    dict mapping dataset name → Spark DataFrame (NOT yet persisted).
    """
    dfs: dict[str, DataFrame] = {}

    with timer("Ingest ratings", logger):
        ratings = _read_csv(spark, RAW_FILES["ratings"], RATINGS_SCHEMA)
        ratings = _parse_timestamp(ratings)
        dfs["ratings"] = ratings

    with timer("Ingest movies", logger):
        dfs["movies"] = _read_csv(spark, RAW_FILES["movies"], MOVIES_SCHEMA)

    with timer("Ingest tags", logger):
        tags = _read_csv(spark, RAW_FILES["tags"], TAGS_SCHEMA)
        tags = _parse_timestamp(tags)
        dfs["tags"] = tags

    with timer("Ingest links", logger):
        dfs["links"] = _read_csv(spark, RAW_FILES["links"], LINKS_SCHEMA)

    with timer("Ingest genome_tags", logger):
        dfs["genome_tags"] = _read_csv(spark, RAW_FILES["genome_tags"], GENOME_TAGS_SCHEMA)

    with timer("Ingest genome_scores", logger):
        dfs["genome_scores"] = _read_csv(spark, RAW_FILES["genome_scores"], GENOME_SCORES_SCHEMA)

    return dfs


def write_parquet(dfs: dict[str, DataFrame]) -> None:
    """Persist each DataFrame as Parquet (overwrite mode)."""
    mapping = {
        "ratings"      : PARQUET["ratings"],
        "movies"       : PARQUET["movies"],
        "tags"         : PARQUET["tags"],
        "links"        : PARQUET["links"],
        "genome_scores": PARQUET["genome_scores"],
    }
    for name, path in mapping.items():
        if name in dfs:
            with timer(f"Write Parquet: {name}", logger):
                dfs[name].write.mode("overwrite").parquet(path)
            logger.info("Parquet written: %s", path)


def run_ingestion(spark: SparkSession = None) -> dict[str, DataFrame]:
    """Entry point: ingest CSVs → write Parquet → return DataFrames."""
    own_spark = spark is None
    if own_spark:
        spark = get_spark("OTT_Ingestion")

    logger.info("=== INGESTION PHASE ===")
    dfs = ingest_all(spark)
    write_parquet(dfs)
    logger.info("=== INGESTION COMPLETE ===")

    if own_spark:
        # Don't stop – caller may reuse spark
        pass

    return dfs


if __name__ == "__main__":
    spark = get_spark("OTT_Ingestion")
    run_ingestion(spark)
    spark.stop()
