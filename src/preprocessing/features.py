"""
Phase 3 – Feature Engineering
Builds user-level and movie-level feature tables using PySpark window functions.
All new columns are marked as DERIVED.
"""
import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils.spark_session import get_spark
from src.utils.logger import get_logger, timer
from config.settings import PARQUET, ALL_GENRES

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# User Features
# ─────────────────────────────────────────────────────────────────────────────

def build_user_features(spark: SparkSession) -> DataFrame:
    """
    Aggregate per-user statistics from cleaned ratings.

    DERIVED columns
    ───────────────
    interaction_count   – total ratings given
    unique_movies       – distinct movies rated
    avg_rating          – mean rating
    rating_variance     – variance of ratings
    rating_std          – std dev of ratings
    min_rating          – minimum rating given
    max_rating          – maximum rating given
    first_interaction   – earliest timestamp
    last_interaction    – latest timestamp
    days_active         – span in days
    recency_days        – days since last interaction (relative to dataset max)
    weekday_count       – ratings on weekdays
    weekend_count       – ratings on weekends
    genre_diversity     – number of distinct genres rated (see below)
    top_genre           – single most-rated genre
    """
    ratings  = spark.read.parquet(PARQUET["ratings"])
    movies   = spark.read.parquet(PARQUET["movies"])

    # Explode genres per rating so we can count per user per genre
    ratings_genre = (
        ratings.join(movies.select("movieId", "genre_array"), "movieId", "left")
               .withColumn("genre", F.explode_outer(F.col("genre_array")))
               .filter(F.col("genre").isNotNull() & (F.col("genre") != "(no genres listed)"))
    )

    # Genre counts per user
    genre_per_user = (
        ratings_genre
        .groupBy("userId", "genre")
        .agg(F.count("*").alias("genre_count"))
    )

    # Top genre per user
    w_genre = Window.partitionBy("userId").orderBy(F.col("genre_count").desc())
    top_genre_df = (
        genre_per_user
        .withColumn("_rn", F.row_number().over(w_genre))
        .filter(F.col("_rn") == 1)
        .select("userId", F.col("genre").alias("top_genre"))
    )

    # Genre diversity = number of distinct genres per user
    genre_diversity_df = (
        genre_per_user
        .groupBy("userId")
        .agg(F.countDistinct("genre").alias("genre_diversity"))
    )

    # Max timestamp in dataset for recency calculation
    max_ts = ratings.agg(F.max("timestamp")).collect()[0][0]

    # Core user aggregates
    user_agg = (
        ratings.groupBy("userId")
        .agg(
            F.count("*").alias("interaction_count"),
            F.countDistinct("movieId").alias("unique_movies"),
            F.avg("rating").alias("avg_rating"),
            F.variance("rating").alias("rating_variance"),
            F.stddev("rating").alias("rating_std"),
            F.min("rating").alias("min_rating"),
            F.max("rating").alias("max_rating"),
            F.min("timestamp").alias("first_interaction"),
            F.max("timestamp").alias("last_interaction"),
            F.sum(F.when(F.col("is_weekend") == 0, 1).otherwise(0)).alias("weekday_count"),
            F.sum(F.when(F.col("is_weekend") == 1, 1).otherwise(0)).alias("weekend_count"),
        )
        .withColumn(
            "days_active",
            F.datediff(F.col("last_interaction"), F.col("first_interaction"))
        )
        .withColumn(
            "recency_days",
            F.datediff(F.lit(max_ts), F.col("last_interaction"))
        )
    )

    # Join derived columns
    user_features = (
        user_agg
        .join(top_genre_df,       "userId", "left")
        .join(genre_diversity_df, "userId", "left")
    )

    # Fill nulls for variance (single-rating users have null variance)
    user_features = user_features.fillna({"rating_variance": 0.0, "rating_std": 0.0,
                                          "days_active": 0, "recency_days": 0})
    return user_features


# ─────────────────────────────────────────────────────────────────────────────
# Movie Features
# ─────────────────────────────────────────────────────────────────────────────

def build_movie_features(spark: SparkSession) -> DataFrame:
    """
    Aggregate per-movie statistics.

    DERIVED columns
    ───────────────
    interaction_count   – total ratings received
    unique_users        – distinct raters
    avg_rating          – mean rating
    rating_variance     – variance of ratings
    popularity_score    – log1p(interaction_count) * avg_rating  (DERIVED)
    """
    ratings = spark.read.parquet(PARQUET["ratings"])
    movies  = spark.read.parquet(PARQUET["movies"])

    movie_agg = (
        ratings.groupBy("movieId")
        .agg(
            F.count("*").alias("interaction_count"),
            F.countDistinct("userId").alias("unique_users"),
            F.avg("rating").alias("avg_rating"),
            F.variance("rating").alias("rating_variance"),
        )
        .fillna({"rating_variance": 0.0})
        .withColumn(
            "popularity_score",
            F.log1p(F.col("interaction_count")) * F.col("avg_rating")
        )
    )

    movie_features = (
        movies.join(movie_agg, "movieId", "left")
              .fillna({"interaction_count": 0, "unique_users": 0,
                       "avg_rating": 0.0, "popularity_score": 0.0})
    )
    return movie_features


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(spark: SparkSession = None) -> dict[str, DataFrame]:
    own_spark = spark is None
    if own_spark:
        spark = get_spark("OTT_Features")

    logger.info("=== FEATURE ENGINEERING PHASE ===")
    result: dict[str, DataFrame] = {}

    with timer("Build user features", logger):
        uf = build_user_features(spark)
        uf.write.mode("overwrite").parquet(PARQUET["user_features"])
        result["user_features"] = uf
        logger.info("User features: %d users", uf.count())

    with timer("Build movie features", logger):
        mf = build_movie_features(spark)
        mf.write.mode("overwrite").parquet(PARQUET["movie_features"])
        result["movie_features"] = mf
        logger.info("Movie features: %d movies", mf.count())

    logger.info("=== FEATURE ENGINEERING COMPLETE ===")
    return result


if __name__ == "__main__":
    spark = get_spark("OTT_Features")
    run_feature_engineering(spark)
    spark.stop()
