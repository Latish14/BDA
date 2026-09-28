# Architecture – Movie/OTT Behaviour Analysis

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        Raw Data Layer                            │
│  rating.csv (20M)  movie.csv  tag.csv  link.csv  genome_*.csv   │
└─────────────────────────────┬────────────────────────────────────┘
                              │ PySpark explicit-schema read
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Phase 1: Ingestion                            │
│  src/ingestion/ingest.py                                         │
│  • Explicit StructType schemas (no inference on large files)     │
│  • Timestamp parsing (YYYY-MM-DD HH:MM:SS)                       │
│  • Write Parquet (Snappy) → data/processed/                      │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Phase 2: Cleaning                             │
│  src/preprocessing/clean.py                                      │
│  • Drop nulls (userId, movieId, rating, timestamp)               │
│  • Rating bounds [0.5, 5.0]                                      │
│  • Valid IDs (> 0)                                               │
│  • Deduplicate (userId, movieId) → keep latest                   │
│  • Before/after row counts reported                              │
│  • Temporal feature extraction (year, month, weekday, weekend)   │
│  • Overwrite cleaned Parquet                                     │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Phase 3: Feature Engineering                    │
│  src/preprocessing/features.py                                   │
│  USER FEATURES (DERIVED):                                        │
│    interaction_count, unique_movies, avg_rating, rating_variance │
│    genre_diversity, top_genre, recency_days, days_active,        │
│    weekday_count, weekend_count, first/last_interaction          │
│  MOVIE FEATURES (DERIVED):                                       │
│    interaction_count, unique_users, avg_rating, rating_variance  │
│    popularity_score = log1p(interactions) × avg_rating           │
│  → Parquet: user_features.parquet, movie_features.parquet        │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                 ┌────────────┼─────────────┐
                 ▼            ▼             ▼
┌──────────────┐ ┌──────────────────┐ ┌──────────────────────────┐
│  Phase 4:    │ │   Phase 5:       │ │   Phase 6:               │
│  Analytics   │ │   Segmentation   │ │   Recommendation         │
│              │ │                  │ │                          │
│  Spark SQL   │ │  K-Means MLlib   │ │  ALS MLlib               │
│  + DataFrames│ │  VectorAssembler │ │  rank=20, iter=10        │
│              │ │  StandardScaler  │ │  regParam=0.1            │
│  • Overview  │ │  Silhouette eval │ │  Train/Test split 80/20  │
│  • Genre pop │ │  → k clusters    │ │  RMSE evaluation         │
│  • Activity  │ │  Describe from   │ │  Model persisted         │
│  • Top movies│ │  actual stats    │ │  Filter seen movies      │
│  • Trends    │ │  → Parquet       │ │  Genre fallback          │
└──────┬───────┘ └────────┬─────────┘ └────────────┬─────────────┘
       │                  │                         │
       └──────────────────┴─────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                           │
│  dashboard/app.py                                                │
│  • @st.cache_resource for SparkSession                           │
│  • @st.cache_data for all Parquet reads                          │
│  • 6 pages: Overview, User, Content, Temporal, Segment, Recs    │
│  • Plotly charts                                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow (File I/O)

```
rating.csv ──ingest──► ratings.parquet ──clean──► ratings.parquet (overwrite)
movie.csv  ──ingest──► movies.parquet  ──clean──► movies.parquet  (overwrite)
tag.csv    ──ingest──► tags.parquet    ──clean──► tags.parquet    (overwrite)

ratings.parquet + movies.parquet
    ──features──► user_features.parquet
    ──features──► movie_features.parquet

user_features.parquet
    ──segment──► user_segments.parquet
    ──segment──► kmeans_model/  (MLlib saved model)
    ──segment──► scaler_model/

ratings.parquet
    ──als──► als_model/  (MLlib saved model)
```

---

## Spark Optimisations Applied

| Optimisation | Where |
|---|---|
| Explicit schemas on CSV reads | `ingest.py` |
| Parquet with Snappy compression | All write operations |
| Adaptive Query Execution (AQE) | `spark_session.py` |
| `spark.sql.shuffle.partitions=50` | Prevents 200-partition overhead |
| Window functions over GROUP BY loops | `features.py` |
| `dropna` before `distinct` | `clean.py` |
| `cache()` only during K-Means k-search | `cluster.py` |
| `createOrReplaceTempView` for SQL reuse | `analytics.py` |
| `@st.cache_data` to avoid re-queries | `dashboard/app.py` |
| `coldStartStrategy=drop` in ALS | `recommender.py` |

---

## Java Compatibility

PySpark 3.5 runs on Java 25 (present) with explicit `--add-opens` JVM flags added in `spark_session.py` to satisfy Java module system requirements introduced in Java 9+.

---

## Module Dependency Graph

```
run_pipeline.py
    └─ src.utils.spark_session
    └─ src.ingestion.ingest
    └─ src.preprocessing.clean
    └─ src.preprocessing.features
    └─ src.analytics.analytics
    └─ src.segmentation.cluster
    └─ src.recommendation.recommender

dashboard/app.py
    └─ src.utils.spark_session
    └─ src.analytics.analytics
    └─ src.recommendation.recommender
    └─ config.settings
```
