"""
Global configuration settings for the Movie/OTT Behaviour Analysis project.
All paths, Spark settings, and model hyperparameters are centralised here.
"""
import os
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW      = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED= PROJECT_ROOT / "data" / "processed"
DATA_SYNTHETIC= PROJECT_ROOT / "data" / "synthetic"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"

# ── Raw file names ────────────────────────────────────────────────────────────
RAW_FILES = {
    "ratings"       : PROJECT_ROOT / "rating.csv",
    "movies"        : PROJECT_ROOT / "movie.csv",
    "tags"          : PROJECT_ROOT / "tag.csv",
    "links"         : PROJECT_ROOT / "link.csv",
    "genome_tags"   : PROJECT_ROOT / "genome_tags.csv",
    "genome_scores" : PROJECT_ROOT / "genome_scores.csv",
}

# ── Processed Parquet paths ───────────────────────────────────────────────────
PARQUET = {
    "ratings"        : str(DATA_PROCESSED / "ratings.parquet"),
    "movies"         : str(DATA_PROCESSED / "movies.parquet"),
    "tags"           : str(DATA_PROCESSED / "tags.parquet"),
    "links"          : str(DATA_PROCESSED / "links.parquet"),
    "genome_scores"  : str(DATA_PROCESSED / "genome_scores.parquet"),
    "user_features"  : str(DATA_PROCESSED / "user_features.parquet"),
    "movie_features" : str(DATA_PROCESSED / "movie_features.parquet"),
    "segments"       : str(DATA_PROCESSED / "user_segments.parquet"),
    "als_model"      : str(DATA_PROCESSED / "als_model"),
    "kmeans_model"   : str(DATA_PROCESSED / "kmeans_model"),
    "scaler_model"   : str(DATA_PROCESSED / "scaler_model"),
}

# ── Spark configuration ───────────────────────────────────────────────────────
SPARK_CONFIG = {
    "app_name"              : "OTT_Behaviour_Analysis",
    "master"                : "local[*]",
    "driver_memory"         : "4g",
    "executor_memory"       : "4g",
    "sql_shuffle_partitions": "50",
    "adaptive_enabled"      : "true",
}

# ── Data quality thresholds ───────────────────────────────────────────────────
MIN_RATING   = 0.5
MAX_RATING   = 5.0
MIN_USER_INTERACTIONS = 5    # for analytics / recommendations
MIN_MOVIE_INTERACTIONS= 10   # minimum ratings before a movie appears in rankings

# ── K-Means segmentation ─────────────────────────────────────────────────────
KMEANS_K_RANGE = range(2, 6)
KMEANS_SEED    = 42
KMEANS_MAX_ITER= 20


# ── ALS recommendation ────────────────────────────────────────────────────────
ALS_RANK           = 20
ALS_MAX_ITER       = 10
ALS_REG_PARAM      = 0.1
ALS_COLD_START     = "drop"
ALS_SEED           = 42
ALS_TRAIN_RATIO    = 0.8
ALS_TOP_N          = 10

# ── Genre list (from MovieLens) ────────────────────────────────────────────────
ALL_GENRES = [
    "Action","Adventure","Animation","Children","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","IMAX",
    "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western",
    "(no genres listed)",
]
