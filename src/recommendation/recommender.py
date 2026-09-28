"""
Phase 6 – Recommendation System
ALS Collaborative Filtering via Spark MLlib + content-based fallback.
"""
import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

from src.utils.spark_session import get_spark
from src.utils.logger import get_logger, timer
from config.settings import (
    PARQUET,
    ALS_RANK, ALS_MAX_ITER, ALS_REG_PARAM,
    ALS_COLD_START, ALS_SEED, ALS_TRAIN_RATIO, ALS_TOP_N,
    MIN_USER_INTERACTIONS, MIN_MOVIE_INTERACTIONS,
)

logger = get_logger(__name__)


class RecommendationEngine:
    """ALS-based collaborative filtering recommendation engine."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self._als_model   = None
        self._movies_df   = None
        self._ratings_df  = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, force: bool = False) -> None:
        """
        Load or train ALS model.
        Only users/movies meeting minimum interaction thresholds are used.
        """
        model_path = PARQUET["als_model"]

        if not force:
            try:
                from pyspark.ml.recommendation import ALSModel
                self._als_model = ALSModel.load(model_path)
                logger.info("ALS model loaded from: %s", model_path)
                self._load_support_tables()
                return
            except Exception:
                logger.info("No saved model found – training fresh.")

        ratings = spark_or_load(self.spark, "ratings")

        # Filter to active users & movies
        user_counts  = ratings.groupBy("userId").count().filter(
            F.col("count") >= MIN_USER_INTERACTIONS
        )
        movie_counts = ratings.groupBy("movieId").count().filter(
            F.col("count") >= MIN_MOVIE_INTERACTIONS
        )
        filtered = (
            ratings
            .join(user_counts.select("userId"),  "userId",  "inner")
            .join(movie_counts.select("movieId"), "movieId", "inner")
            .select("userId", "movieId", "rating")
        )
        logger.info("Ratings for ALS: %d", filtered.count())

        train, test = filtered.randomSplit([ALS_TRAIN_RATIO, 1 - ALS_TRAIN_RATIO], seed=ALS_SEED)

        with timer("Train ALS", logger):
            als = ALS(
                rank=ALS_RANK,
                maxIter=ALS_MAX_ITER,
                regParam=ALS_REG_PARAM,
                userCol="userId",
                itemCol="movieId",
                ratingCol="rating",
                coldStartStrategy=ALS_COLD_START,
                seed=ALS_SEED,
            )
            self._als_model = als.fit(train)

        # Evaluate on test set
        evaluator = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        )
        preds = self._als_model.transform(test)
        rmse  = evaluator.evaluate(preds)
        logger.info("ALS RMSE on test split: %.4f", rmse)

        # Save model
        self._als_model.write().overwrite().save(model_path)
        logger.info("ALS model saved to: %s", model_path)
        self._load_support_tables()

        return rmse

    def _load_support_tables(self):
        self._movies_df  = self.spark.read.parquet(PARQUET["movies"])
        self._ratings_df = self.spark.read.parquet(PARQUET["ratings"])

    # ── Inference ─────────────────────────────────────────────────────────────

    def recommend_movies(self, user_id: int, n: int = ALS_TOP_N) -> list[dict]:
        """
        Return top-N movie recommendations for user_id.
        Filters out movies the user has already rated.

        Returns
        -------
        list of dicts: {movieId, title, genres, predicted_score}
        """
        if self._als_model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Movies the user has already rated
        seen = (
            self._ratings_df
            .filter(F.col("userId") == user_id)
            .select("movieId")
        )
        seen_ids = set(r["movieId"] for r in seen.collect())

        # All movies
        all_movies = self._movies_df.select("movieId")
        unseen_movies = all_movies.filter(~F.col("movieId").isin(seen_ids))

        # Build user-movie pairs
        user_df = self.spark.createDataFrame([(user_id,)], ["userId"])
        pairs   = user_df.crossJoin(unseen_movies)

        preds = (
            self._als_model.transform(pairs)
            .filter(F.col("prediction").isNotNull() & (~F.isnan(F.col("prediction"))))
            .orderBy(F.col("prediction").desc())
            .limit(n)
            .join(self._movies_df.select("movieId", "title", "genres"), "movieId")
        )

        rows = preds.collect()
        return [
            {
                "movieId"        : r["movieId"],
                "title"          : r["title"],
                "genres"         : r["genres"],
                "predicted_score": round(float(r["prediction"]), 3),
            }
            for r in rows
        ]

    def recommend_batch(self, n: int = ALS_TOP_N) -> DataFrame:
        """
        Generate top-N recommendations for ALL users as a Spark DataFrame.
        Useful for pre-computing the dashboard.
        """
        if self._als_model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        recs = self._als_model.recommendForAllUsers(n)
        return recs

    # ── Content-based fallback ────────────────────────────────────────────────

    def genre_based_recommendations(self, user_id: int, n: int = 10) -> DataFrame:
        """
        Fallback: recommend top-rated unseen movies whose genres overlap
        with the user's most-rated genre.
        DERIVED from user's genre preference and movie avg_rating.
        """
        user_feat = self.spark.read.parquet(PARQUET["user_features"])
        user_row  = user_feat.filter(F.col("userId") == user_id).limit(1).collect()
        if not user_row:
            return self.spark.createDataFrame([], schema=self._movies_df.schema)

        top_genre = user_row[0]["top_genre"] or ""
        seen_ids  = set(
            r["movieId"] for r in
            self._ratings_df.filter(F.col("userId") == user_id).select("movieId").collect()
        )
        movie_feat = self.spark.read.parquet(PARQUET["movie_features"])
        return (
            movie_feat
            .filter(F.col("genres").contains(top_genre))
            .filter(~F.col("movieId").isin(seen_ids))
            .filter(F.col("interaction_count") >= MIN_MOVIE_INTERACTIONS)
            .orderBy(F.col("avg_rating").desc())
            .select("movieId", "title", "genres", "avg_rating", "interaction_count")
            .limit(n)
        )


def spark_or_load(spark, key: str) -> DataFrame:
    return spark.read.parquet(PARQUET[key])


def run_recommendation(spark: SparkSession = None, force_train: bool = False):
    own_spark = spark is None
    if own_spark:
        spark = get_spark("OTT_Recommendation")

    logger.info("=== RECOMMENDATION PHASE ===")
    engine = RecommendationEngine(spark)
    rmse = engine.train(force=force_train)
    logger.info("=== RECOMMENDATION COMPLETE ===")
    return engine, rmse


if __name__ == "__main__":
    spark = get_spark("OTT_Recommendation")
    engine, rmse = run_recommendation(spark, force_train=True)

    # Demo
    demo_user = 1
    logger.info("Recommendations for user %d:", demo_user)
    recs = engine.recommend_movies(demo_user, n=5)
    for r in recs:
        logger.info("  %s (%s) – score %.3f", r["title"], r["genres"], r["predicted_score"])

    spark.stop()
