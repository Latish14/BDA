"""
Phase 5 – User Segmentation
K-Means clustering using Spark MLlib with silhouette evaluation.
"""
import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline

from src.utils.spark_session import get_spark
from src.utils.logger import get_logger, timer
from config.settings import (
    PARQUET, KMEANS_K_RANGE, KMEANS_SEED, KMEANS_MAX_ITER
)

logger = get_logger(__name__)


# Features used for clustering (all DERIVED)
CLUSTER_FEATURES = [
    "interaction_count",
    "avg_rating",
    "unique_movies",
    "genre_diversity",
    "recency_days",
    "rating_variance",
]


def _prepare_features(user_features: DataFrame) -> DataFrame:
    """Fill nulls and assemble feature vector."""
    df = user_features.fillna(0.0, subset=CLUSTER_FEATURES)
    assembler = VectorAssembler(
        inputCols=CLUSTER_FEATURES,
        outputCol="raw_features",
        handleInvalid="keep",
    )
    df = assembler.transform(df)
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    return df, scaler_model


def find_optimal_k(df: DataFrame, k_range=KMEANS_K_RANGE) -> tuple[int, dict]:
    """
    Train K-Means for each k in k_range, evaluate silhouette score,
    and return the best k plus the full score dict.
    """
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="segment",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    scores: dict[int, float] = {}
    for k in k_range:
        kmeans = KMeans(
            k=k, seed=KMEANS_SEED, maxIter=KMEANS_MAX_ITER,
            featuresCol="features", predictionCol="segment"
        )
        model = kmeans.fit(df)
        preds = model.transform(df)
        score = evaluator.evaluate(preds)
        scores[k] = round(score, 4)
        logger.info("  k=%d  silhouette=%.4f", k, score)

    best_k = max(scores, key=scores.get)
    logger.info("Best k=%d (silhouette=%.4f)", best_k, scores[best_k])
    return best_k, scores


def train_segmentation(spark: SparkSession, k: int = None) -> DataFrame:
    """
    Train K-Means on user features, assign segments, persist results.

    Returns
    -------
    DataFrame with userId + CLUSTER_FEATURES + segment
    """
    user_feat = spark.read.parquet(PARQUET["user_features"])
    logger.info("Users for clustering: %d", user_feat.count())

    with timer("Prepare features", logger):
        df, scaler_model = _prepare_features(user_feat)

    # Cache for repeated K evaluations
    df.cache()

    if k is None:
        with timer("Find optimal k", logger):
            k, sil_scores = find_optimal_k(df)
    else:
        sil_scores = {}

    logger.info("Training final K-Means with k=%d …", k)
    with timer(f"Train KMeans k={k}", logger):
        kmeans = KMeans(
            k=k, seed=KMEANS_SEED, maxIter=KMEANS_MAX_ITER,
            featuresCol="features", predictionCol="segment"
        )
        km_model = kmeans.fit(df)

    segmented = km_model.transform(df)

    # Describe each cluster with actual statistics
    cluster_stats = (
        segmented
        .groupBy("segment")
        .agg(
            F.count("*").alias("cluster_size"),
            F.avg("interaction_count").alias("avg_interactions"),
            F.avg("avg_rating").alias("avg_rating"),
            F.avg("unique_movies").alias("avg_unique_movies"),
            F.avg("genre_diversity").alias("avg_genre_diversity"),
            F.avg("recency_days").alias("avg_recency_days"),
            F.avg("rating_variance").alias("avg_rating_variance"),
        )
        .orderBy("segment")
    )
    logger.info("=== CLUSTER STATISTICS ===")
    cluster_stats.show(truncate=False)

    # Save segment assignments
    result = segmented.select(
        "userId", "interaction_count", "avg_rating",
        "unique_movies", "genre_diversity", "recency_days",
        "rating_variance", "top_genre", "segment"
    )
    result.write.mode("overwrite").parquet(PARQUET["segments"])
    logger.info("Segments written to: %s", PARQUET["segments"])

    # Save model
    km_model.write().overwrite().save(PARQUET["kmeans_model"])
    scaler_model.write().overwrite().save(PARQUET["scaler_model"])

    df.unpersist()
    return result, cluster_stats, sil_scores


def run_segmentation(spark: SparkSession = None, k: int = None):
    own_spark = spark is None
    if own_spark:
        spark = get_spark("OTT_Segmentation")

    logger.info("=== SEGMENTATION PHASE ===")
    result, cluster_stats, sil_scores = train_segmentation(spark, k=k)
    logger.info("=== SEGMENTATION COMPLETE ===")
    return result, cluster_stats, sil_scores


if __name__ == "__main__":
    spark = get_spark("OTT_Segmentation")
    run_segmentation(spark)
    spark.stop()
