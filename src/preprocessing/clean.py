"""
Phase 2 – Preprocessing & Cleaning
Applies data-quality checks, removes invalids, and reports before/after counts.
"""
import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.utils.spark_session import get_spark
from src.utils.logger import get_logger, timer
from config.settings import PARQUET, MIN_RATING, MAX_RATING

logger = get_logger(__name__)


def _report_quality(df: DataFrame, name: str, before: int = None) -> int:
    """Count rows and optionally log delta."""
    count = df.count()
    if before is not None:
        removed = before - count
        logger.info("  %s: %d rows  (removed %d, %.2f%%)",
                    name, count, removed, 100 * removed / max(before, 1))
    else:
        logger.info("  %s: %d rows (raw)", name, count)
    return count


def clean_ratings(spark: SparkSession) -> DataFrame:
    """
    Load ratings Parquet and apply cleaning:
    1. Drop nulls in userId, movieId, rating, timestamp
    2. Filter ratings outside [0.5, 5.0]
    3. Filter invalid IDs (userId/movieId <= 0)
    4. Drop duplicates (userId, movieId)
    5. Add temporal features (year, month, weekday, is_weekend)
    """
    df = spark.read.parquet(PARQUET["ratings"])
    
    # Combined cleaning transformations (single execution pipeline)
    df = (
        df.dropna(subset=["userId", "movieId", "rating", "timestamp"])
          .filter((F.col("rating") >= MIN_RATING) & (F.col("rating") <= MAX_RATING))
          .filter((F.col("userId") > 0) & (F.col("movieId") > 0))
          .dropDuplicates(["userId", "movieId"])
    )

    # Add temporal features (DERIVED fields)
    df = (
        df.withColumn("year",    F.year("timestamp"))
          .withColumn("month",   F.month("timestamp"))
          .withColumn("weekday", F.dayofweek("timestamp"))   # 1=Sun … 7=Sat
          .withColumn("is_weekend",
                      F.when(F.col("weekday").isin(1, 7), 1).otherwise(0))
    )
    return df



def clean_movies(spark: SparkSession) -> DataFrame:
    """
    Load movies Parquet and apply cleaning:
    1. Drop rows with null movieId or title
    2. Drop duplicates on movieId
    3. Handle '(no genres listed)' as a valid value (keep, document)
    4. Extract release year from title as DERIVED field
    """
    df = spark.read.parquet(PARQUET["movies"])
    before = _report_quality(df, "movies_raw")

    df = df.dropna(subset=["movieId", "title"])
    _report_quality(df, "after_dropna", before)

    df = df.dropDuplicates(["movieId"])

    # DERIVED: release year from title pattern "(YYYY)"
    df = df.withColumn(
        "release_year",
        F.regexp_extract(F.col("title"), r"\((\d{4})\)$", 1).cast("int")
    )

    # DERIVED: genre array
    df = df.withColumn(
        "genre_array",
        F.split(F.col("genres"), r"\|")
    )

    return df


def clean_tags(spark: SparkSession) -> DataFrame:
    """
    Load tags Parquet:
    1. Drop nulls in userId, movieId, tag
    2. Trim and lowercase tags
    3. Drop empty tags after trim
    """
    df = spark.read.parquet(PARQUET["tags"])
    before = _report_quality(df, "tags_raw")

    df = df.dropna(subset=["userId", "movieId", "tag"])
    df = df.withColumn("tag", F.trim(F.lower(F.col("tag"))))
    df = df.filter(F.length(F.col("tag")) > 0)
    _report_quality(df, "after_clean", before)

    return df


def run_cleaning(spark: SparkSession = None) -> dict[str, DataFrame]:
    """
    Run all cleaning steps and write cleaned Parquet back.
    Returns dict of cleaned DataFrames.
    """
    own_spark = spark is None
    if own_spark:
        spark = get_spark("OTT_Cleaning")

    logger.info("=== CLEANING PHASE ===")
    cleaned: dict[str, DataFrame] = {}

    def _safe_write_parquet(df: DataFrame, target_path: str):
        import shutil
        tmp_path = target_path + "_tmp"
        df.write.mode("overwrite").parquet(tmp_path)
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.move(tmp_path, target_path)

    with timer("Clean ratings", logger):
        cleaned["ratings"] = clean_ratings(spark)
        _safe_write_parquet(cleaned["ratings"], PARQUET["ratings"])
        logger.info("Clean ratings written.")

    with timer("Clean movies", logger):
        cleaned["movies"] = clean_movies(spark)
        _safe_write_parquet(cleaned["movies"], PARQUET["movies"])
        logger.info("Clean movies written.")

    with timer("Clean tags", logger):
        cleaned["tags"] = clean_tags(spark)
        _safe_write_parquet(cleaned["tags"], PARQUET["tags"])
        logger.info("Clean tags written.")

    logger.info("=== CLEANING COMPLETE ===")
    return cleaned



if __name__ == "__main__":
    spark = get_spark("OTT_Cleaning")
    run_cleaning(spark)
    spark.stop()
