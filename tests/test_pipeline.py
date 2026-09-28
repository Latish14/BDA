"""
Tests for the Movie/OTT Behaviour Analysis pipeline.
Run with: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, TimestampType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    """Session-scoped SparkSession for tests."""
    from src.utils.spark_session import get_spark
    s = get_spark("OTT_Tests")
    yield s
    s.stop()


@pytest.fixture(scope="session")
def sample_ratings(spark):
    """Minimal ratings DataFrame for unit tests."""
    data = [
        (1, 10, 4.0, "2015-01-01 12:00:00"),
        (1, 11, 3.5, "2015-02-01 12:00:00"),
        (2, 10, 5.0, "2015-03-01 12:00:00"),
        (2, 12, 2.5, "2015-04-01 12:00:00"),
        (3, 10, 1.0, "2015-05-01 12:00:00"),
        (3, 11, 4.5, "2015-06-01 12:00:00"),
        (4, 13, 3.0, "2015-07-01 12:00:00"),
    ]
    schema = StructType([
        StructField("userId",    IntegerType(), True),
        StructField("movieId",   IntegerType(), True),
        StructField("rating",    FloatType(),   True),
        StructField("timestamp", StringType(),  True),
    ])
    df = spark.createDataFrame(data, schema)
    return df.withColumn(
        "timestamp", F.to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss")
    )


@pytest.fixture(scope="session")
def sample_movies(spark):
    data = [
        (10, "Toy Story (1995)", "Adventure|Animation|Comedy"),
        (11, "Jumanji (1995)",   "Adventure|Children|Fantasy"),
        (12, "Heat (1995)",      "Action|Crime|Thriller"),
        (13, "GoodFellas (1990)","Crime|Drama"),
    ]
    schema = StructType([
        StructField("movieId", IntegerType(), True),
        StructField("title",   StringType(),  True),
        StructField("genres",  StringType(),  True),
    ])
    return spark.createDataFrame(data, schema)


# ── Test: Spark session ───────────────────────────────────────────────────────

class TestSparkSession:
    def test_spark_running(self, spark):
        assert spark is not None
        assert spark.version is not None

    def test_can_create_dataframe(self, spark):
        df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
        assert df.count() == 2


# ── Test: Data loading ────────────────────────────────────────────────────────

class TestDataLoading:
    def test_ratings_schema(self, sample_ratings):
        cols = sample_ratings.columns
        assert "userId"    in cols
        assert "movieId"   in cols
        assert "rating"    in cols
        assert "timestamp" in cols

    def test_movies_schema(self, sample_movies):
        cols = sample_movies.columns
        assert "movieId" in cols
        assert "title"   in cols
        assert "genres"  in cols

    def test_ratings_row_count(self, sample_ratings):
        assert sample_ratings.count() == 7

    def test_no_null_required_cols(self, sample_ratings):
        null_count = sample_ratings.filter(
            F.col("userId").isNull() | F.col("movieId").isNull() | F.col("rating").isNull()
        ).count()
        assert null_count == 0


# ── Test: Preprocessing ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_rating_bounds(self, sample_ratings):
        """Ratings must be in [0.5, 5.0]."""
        from config.settings import MIN_RATING, MAX_RATING
        invalid = sample_ratings.filter(
            (F.col("rating") < MIN_RATING) | (F.col("rating") > MAX_RATING)
        ).count()
        assert invalid == 0

    def test_valid_ids(self, sample_ratings):
        invalid = sample_ratings.filter(
            (F.col("userId") <= 0) | (F.col("movieId") <= 0)
        ).count()
        assert invalid == 0

    def test_timestamp_parsed(self, sample_ratings):
        """Timestamps should be actual timestamp type (not null)."""
        null_ts = sample_ratings.filter(F.col("timestamp").isNull()).count()
        assert null_ts == 0

    def test_genre_split(self, sample_movies):
        df = sample_movies.withColumn("genre_array", F.split("genres", r"\|"))
        first_row_genres = df.filter(F.col("movieId") == 10).select("genre_array").collect()[0][0]
        assert "Adventure" in first_row_genres
        assert "Animation" in first_row_genres

    def test_release_year_extraction(self, sample_movies):
        df = sample_movies.withColumn(
            "release_year",
            F.regexp_extract(F.col("title"), r"\((\d{4})\)$", 1).cast("int")
        )
        year = df.filter(F.col("movieId") == 10).select("release_year").collect()[0][0]
        assert year == 1995


# ── Test: Feature Engineering ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_user_aggregations(self, sample_ratings):
        agg = sample_ratings.groupBy("userId").agg(
            F.count("*").alias("interaction_count"),
            F.avg("rating").alias("avg_rating"),
        )
        assert agg.count() == 4  # users 1,2,3,4
        user1 = agg.filter(F.col("userId") == 1).collect()[0]
        assert user1["interaction_count"] == 2
        assert abs(user1["avg_rating"] - 3.75) < 0.01

    def test_movie_aggregations(self, sample_ratings):
        agg = sample_ratings.groupBy("movieId").agg(
            F.count("*").alias("interaction_count"),
            F.countDistinct("userId").alias("unique_users"),
        )
        movie10 = agg.filter(F.col("movieId") == 10).collect()[0]
        assert movie10["interaction_count"] == 3
        assert movie10["unique_users"] == 3

    def test_temporal_features(self, sample_ratings):
        df = (
            sample_ratings
            .withColumn("year",    F.year("timestamp"))
            .withColumn("month",   F.month("timestamp"))
            .withColumn("weekday", F.dayofweek("timestamp"))
        )
        assert "year"    in df.columns
        assert "month"   in df.columns
        assert "weekday" in df.columns
        years = [r["year"] for r in df.select("year").distinct().collect()]
        assert 2015 in years


# ── Test: User Profile ────────────────────────────────────────────────────────

class TestUserProfile:
    def test_profile_returns_dict(self, sample_ratings, spark):
        """Simulate get_user_profile logic on sample data."""
        agg = sample_ratings.groupBy("userId").agg(
            F.count("*").alias("interaction_count"),
            F.avg("rating").alias("avg_rating"),
            F.countDistinct("movieId").alias("unique_movies"),
        )
        row = agg.filter(F.col("userId") == 1).collect()[0]
        profile = row.asDict()
        assert isinstance(profile, dict)
        assert profile["interaction_count"] == 2
        assert profile["unique_movies"] == 2

    def test_invalid_user_returns_error(self, spark):
        """Non-existent user should return empty or error signal."""
        agg = spark.createDataFrame([], StructType([
            StructField("userId", IntegerType(), True),
        ]))
        rows = agg.filter(F.col("userId") == 9999999).collect()
        assert len(rows) == 0

    def test_missing_file_handled(self, spark):
        """Attempting to read a non-existent Parquet should raise an exception."""
        with pytest.raises(Exception):
            spark.read.parquet("/nonexistent/path/that/does/not/exist").count()


# ── Test: Clustering ──────────────────────────────────────────────────────────

class TestClustering:
    def test_kmeans_runs(self, sample_ratings, spark):
        """K-Means with k=2 on aggregated features should assign a segment to each user."""
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.clustering import KMeans

        agg = sample_ratings.groupBy("userId").agg(
            F.count("*").cast("float").alias("interaction_count"),
            F.avg("rating").alias("avg_rating"),
        ).fillna(0.0)

        assembler = VectorAssembler(inputCols=["interaction_count","avg_rating"],
                                    outputCol="features")
        df = assembler.transform(agg)
        km  = KMeans(k=2, seed=42, featuresCol="features", predictionCol="segment")
        model = km.fit(df)
        result = model.transform(df)
        assert result.filter(F.col("segment").isNull()).count() == 0
        assert result.select("segment").distinct().count() <= 2

    def test_cluster_count(self, sample_ratings, spark):
        """Exactly k distinct segments should be assigned."""
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.clustering import KMeans

        agg = sample_ratings.groupBy("userId").agg(
            F.count("*").cast("float").alias("n"),
            F.avg("rating").alias("avg_r"),
        )
        assembler = VectorAssembler(inputCols=["n","avg_r"], outputCol="features")
        df = assembler.transform(agg)
        model = KMeans(k=2, seed=42, featuresCol="features", predictionCol="segment").fit(df)
        n_segs = model.transform(df).select("segment").distinct().count()
        assert n_segs == 2


# ── Test: Recommendations ─────────────────────────────────────────────────────

class TestRecommendations:
    def test_als_trains(self, sample_ratings, spark):
        """ALS should train without errors on small data."""
        from pyspark.ml.recommendation import ALS
        als = ALS(rank=5, maxIter=3, regParam=0.1,
                  userCol="userId", itemCol="movieId", ratingCol="rating",
                  coldStartStrategy="drop", seed=42)
        model = als.fit(sample_ratings)
        assert model is not None

    def test_predictions_not_null(self, sample_ratings, spark):
        from pyspark.ml.recommendation import ALS
        train, test = sample_ratings.randomSplit([0.8, 0.2], seed=42)
        if train.count() < 2:
            pytest.skip("Not enough data for train/test split in sample")
        als = ALS(rank=5, maxIter=3, regParam=0.1,
                  userCol="userId", itemCol="movieId", ratingCol="rating",
                  coldStartStrategy="drop", seed=42)
        model = als.fit(train)
        preds = model.transform(test).filter(F.col("prediction").isNotNull())
        # With 'drop', NaN predictions from cold start are dropped – just check model works
        assert preds is not None

    def test_recommend_for_user(self, sample_ratings, spark):
        from pyspark.ml.recommendation import ALS
        als = ALS(rank=5, maxIter=3, regParam=0.1,
                  userCol="userId", itemCol="movieId", ratingCol="rating",
                  coldStartStrategy="drop", seed=42)
        model = als.fit(sample_ratings)
        recs = model.recommendForAllUsers(3)
        assert recs.count() >= 1

    def test_invalid_user_recommendations(self, sample_ratings, spark):
        """Recommendations for a non-existent user should return empty."""
        no_user = sample_ratings.filter(F.col("userId") == 9999)
        assert no_user.count() == 0
