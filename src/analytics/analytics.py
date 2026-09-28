"""
Phase 4 – Analytics
Spark SQL + DataFrame-based behavioural and content analytics.
"""
import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils.spark_session import get_spark
from src.utils.logger import get_logger, timer
from config.settings import PARQUET, MIN_MOVIE_INTERACTIONS, MIN_USER_INTERACTIONS, OUTPUTS_DIR

logger = get_logger(__name__)


class AnalyticsEngine:
    """Encapsulates all analytics queries using Spark SQL and DataFrames."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self._ratings    = None
        self._movies     = None
        self._user_feat  = None
        self._movie_feat = None
        self._load()

    def _load(self):
        logger.info("Loading Parquet for analytics …")
        self._ratings    = self.spark.read.parquet(PARQUET["ratings"])
        self._movies     = self.spark.read.parquet(PARQUET["movies"])
        self._user_feat  = self.spark.read.parquet(PARQUET["user_features"])
        self._movie_feat = self.spark.read.parquet(PARQUET["movie_features"])

        # Register temp views for Spark SQL
        self._ratings.createOrReplaceTempView("ratings")
        self._movies.createOrReplaceTempView("movies")
        self._user_feat.createOrReplaceTempView("user_features")
        self._movie_feat.createOrReplaceTempView("movie_features")
        logger.info("Temp views registered.")

    # ── Overview stats ────────────────────────────────────────────────────────

    def overview_stats(self) -> dict:
        """Return high-level dataset statistics."""
        total_ratings = self._ratings.count()
        total_users   = self._ratings.select("userId").distinct().count()
        total_movies  = self._movies.count()
        avg_rating    = self._ratings.agg(F.avg("rating")).collect()[0][0]

        top_genre_row = (
            self._movies
            .withColumn("genre", F.explode(F.split("genres", r"\|")))
            .filter(F.col("genre") != "(no genres listed)")
            .groupBy("genre").agg(F.count("*").alias("cnt"))
            .orderBy(F.col("cnt").desc())
            .limit(1).collect()
        )
        top_genre = top_genre_row[0]["genre"] if top_genre_row else "N/A"

        return {
            "total_ratings" : total_ratings,
            "total_users"   : total_users,
            "total_movies"  : total_movies,
            "avg_rating"    : round(float(avg_rating), 3),
            "top_genre"     : top_genre,
        }

    # ── User analytics ────────────────────────────────────────────────────────

    def most_active_users(self, n: int = 20) -> DataFrame:
        """Top N users by interaction count."""
        return (
            self._user_feat
            .select("userId", "interaction_count", "avg_rating",
                    "unique_movies", "top_genre")
            .orderBy(F.col("interaction_count").desc())
            .limit(n)
        )

    def user_rating_distribution(self) -> DataFrame:
        """Count of ratings at each 0.5-step value."""
        return self.spark.sql("""
            SELECT rating, COUNT(*) AS count
            FROM ratings
            GROUP BY rating
            ORDER BY rating
        """)

    def user_activity_over_time(self) -> DataFrame:
        """Monthly rating counts (year-month aggregation)."""
        return self.spark.sql("""
            SELECT year, month,
                   COUNT(*) AS rating_count,
                   AVG(rating) AS avg_rating
            FROM ratings
            WHERE year IS NOT NULL AND month IS NOT NULL
            GROUP BY year, month
            ORDER BY year, month
        """)

    def weekday_activity(self) -> DataFrame:
        """Rating counts by weekday (1=Sun … 7=Sat)."""
        return self.spark.sql("""
            SELECT weekday,
                   COUNT(*) AS rating_count,
                   AVG(rating) AS avg_rating
            FROM ratings
            GROUP BY weekday
            ORDER BY weekday
        """)

    # ── Content analytics ─────────────────────────────────────────────────────

    def most_rated_movies(self, n: int = 20) -> DataFrame:
        """Top N movies by interaction count (minimum threshold applied)."""
        return (
            self._movie_feat
            .filter(F.col("interaction_count") >= MIN_MOVIE_INTERACTIONS)
            .select("movieId", "title", "genres", "interaction_count",
                    "unique_users", "avg_rating", "popularity_score")
            .orderBy(F.col("interaction_count").desc())
            .limit(n)
        )

    def highest_rated_movies(self, n: int = 20) -> DataFrame:
        """Top N movies by average rating (min interactions threshold)."""
        return (
            self._movie_feat
            .filter(F.col("interaction_count") >= MIN_MOVIE_INTERACTIONS)
            .select("movieId", "title", "genres", "avg_rating",
                    "interaction_count", "popularity_score")
            .orderBy(F.col("avg_rating").desc(), F.col("interaction_count").desc())
            .limit(n)
        )

    def genre_popularity(self) -> DataFrame:
        """
        Genre statistics: total ratings, unique movies, avg rating.
        Uses Spark SQL with LATERAL VIEW explode.
        """
        return self.spark.sql("""
            SELECT g.genre,
                   COUNT(r.rating)           AS total_ratings,
                   COUNT(DISTINCT r.movieId) AS unique_movies,
                   AVG(r.rating)             AS avg_rating
            FROM ratings r
            JOIN movies m ON r.movieId = m.movieId
            LATERAL VIEW explode(split(m.genres, '\\\\|')) g AS genre
            WHERE g.genre != '(no genres listed)'
            GROUP BY g.genre
            ORDER BY total_ratings DESC
        """)

    def genre_avg_rating(self) -> DataFrame:
        """Average rating per genre (minimum interaction filtered)."""
        return self.spark.sql("""
            SELECT g.genre,
                   AVG(r.rating)  AS avg_rating,
                   COUNT(*)       AS rating_count
            FROM ratings r
            JOIN movies m ON r.movieId = m.movieId
            LATERAL VIEW explode(split(m.genres, '\\\\|')) g AS genre
            WHERE g.genre != '(no genres listed)'
            GROUP BY g.genre
            HAVING rating_count >= 100
            ORDER BY avg_rating DESC
        """)

    # ── User Profile ──────────────────────────────────────────────────────────

    def get_user_profile(self, user_id: int) -> dict:
        """
        Return a complete user profile dictionary.
        Fields: userId, interaction_count, avg_rating, unique_movies,
                top_genre, genre_diversity, first_interaction, last_interaction,
                recency_days, days_active, weekend_count, weekday_count,
                rating_variance, segment (if available).
        """
        row = (
            self._user_feat
            .filter(F.col("userId") == user_id)
            .limit(1)
            .collect()
        )
        if not row:
            return {"error": f"User {user_id} not found"}

        r = row[0].asDict()

        # Attach segment if available
        try:
            seg_df = self.spark.read.parquet(PARQUET["segments"])
            seg_row = seg_df.filter(F.col("userId") == user_id).limit(1).collect()
            r["segment"] = seg_row[0]["segment"] if seg_row else None
        except Exception:
            r["segment"] = None

        return r

    # ── Temporal Trends ───────────────────────────────────────────────────────

    def yearly_trends(self) -> DataFrame:
        return self.spark.sql("""
            SELECT year,
                   COUNT(*)       AS rating_count,
                   COUNT(DISTINCT userId)  AS active_users,
                   COUNT(DISTINCT movieId) AS unique_movies,
                   AVG(rating)    AS avg_rating
            FROM ratings
            WHERE year IS NOT NULL AND year BETWEEN 1995 AND 2016
            GROUP BY year
            ORDER BY year
        """)

    def top_movies_by_year(self, n_per_year: int = 3) -> DataFrame:
        """Top N most-rated movies per year."""
        w = Window.partitionBy("year").orderBy(F.col("cnt").desc())
        return (
            self._ratings
            .filter(F.col("year").between(1995, 2016))
            .groupBy("year", "movieId").agg(F.count("*").alias("cnt"))
            .withColumn("rank", F.rank().over(w))
            .filter(F.col("rank") <= n_per_year)
            .join(self._movies.select("movieId", "title", "genres"), "movieId")
            .orderBy("year", "rank")
        )


def run_analytics(spark: SparkSession = None) -> AnalyticsEngine:
    """Initialise analytics engine."""
    if spark is None:
        spark = get_spark("OTT_Analytics")
    return AnalyticsEngine(spark)


if __name__ == "__main__":
    spark = get_spark("OTT_Analytics")
    engine = run_analytics(spark)

    logger.info("=== OVERVIEW ===")
    stats = engine.overview_stats()
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    logger.info("=== MOST ACTIVE USERS ===")
    engine.most_active_users(10).show(10, truncate=False)

    logger.info("=== GENRE POPULARITY ===")
    engine.genre_popularity().show(20, truncate=False)

    logger.info("=== YEARLY TRENDS ===")
    engine.yearly_trends().show(25, truncate=False)

    spark.stop()
