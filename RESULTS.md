# Results – Movie/OTT Behaviour Analysis

> **B.Tech CSE (Data Science) – 7th Semester Big Data Analytics Project**  
> Engine: Apache Spark 3.5.3 · Python 3.11 · OpenJDK 17 LTS  
> All metrics reported below were computed directly by Apache Spark on the full **MovieLens 20M** dataset.

---

## 1. Pipeline Execution Times

| Phase | Description | Measured Runtime |
|---|---|---|
| **Ingestion** | Ingest 6 CSVs (explicit schemas) → Snappy Parquet | **15.2 s** |
| **Cleaning** | Deduplication, null checks, bounds filtering, temporal parsing | **28.4 s** |
| **Feature Engineering** | User window aggregates (138k users) + Movie aggregates (27k movies) | **44.3 s** |
| **Analytics Queries** | Spark SQL overview, temporal, and content aggregation queries | **6.9 s** |
| **Segmentation** | K-Means feature vector assembler, standard scaling, Silhouette eval ($k=2..5$) | **19.2 s** |
| **Recommendation** | ALS matrix factorization on 19.96M ratings + test split RMSE eval | **57.7 s** |
| **Total Pipeline** | End-to-end execution | **~ 2.8 minutes** |

---

## 2. Dataset Statistics (Post-Cleaning)

| Metric | Raw Value | Cleaned Value | Notes |
|---|---|---|---|
| **Total Ratings** | 20,000,263 | **20,000,263** | 0 nulls detected; ratings within [0.5, 5.0] |
| **Unique Users** | 138,493 | **138,493** | Anonymous IDs 1 – 138,493 |
| **Unique Movies** | 27,278 | **27,278** | Catalog spans 1891 to 2015 |
| **Average Rating** | — | **3.526 / 5.0** | Distribution leans positive |
| **Most Popular Genre** | — | **Drama** | 8,857,853 total ratings |

---

## 3. Machine Learning Models & Evaluation

### Spark MLlib ALS Collaborative Filtering

| Parameter / Metric | Value |
|---|---|
| **Ratings Used for Training** | **19,964,833** (filtered to active users $\ge 5$ ratings, movies $\ge 10$ ratings) |
| **Train / Test Split** | 80% Train / 20% Test (`seed=42`) |
| **Latent Factor Rank** | 20 |
| **Max Iterations** | 10 |
| **Regularization Parameter ($\lambda$)** | 0.1 |
| **Cold Start Strategy** | `drop` |
| **Test Set RMSE** | **0.8109** |

*Analysis:* An RMSE of **0.8109** on a 5-point scale demonstrates that the ALS latent factor representations generalize effectively to unseen ratings across the 138,493 user base.

---

### Spark MLlib K-Means User Segmentation

Evaluated using `ClusteringEvaluator` with `squaredEuclidean` distance across normalized feature vectors (`interaction_count`, `avg_rating`, `unique_movies`, `genre_diversity`, `recency_days`, `rating_variance`):

| $k$ (Clusters) | Silhouette Score | Status |
|---|---|---|
| **2** | **0.5458** | **Optimal Configuration (Selected)** |
| 3 | 0.3037 | Sub-optimal separation |
| 4 | 0.3521 | Secondary local maximum |
| 5 | 0.3144 | Over-fragmented |

#### Cluster Profiles (Computed from Actual Spark MLlib Assignments)

| Segment | Label | User Count | % of Users | Avg Interactions | Avg Rating | Unique Movies | Genre Diversity | Avg Recency (Days) | Rating Variance |
|---|---|---|---|---|---|---|---|---|---|
| **0** | **Power Viewers (Cinephiles)** | 22,519 | 16.3% | **506.4** | 3.49 | 506.4 | **18.8 genres** | 2,572.1 days | 0.884 |
| **1** | **Mainstream / Casual Viewers** | 115,974 | 83.7% | **74.1** | 3.65 | 74.1 | 15.9 genres | 4,333.8 days | 0.980 |

*Insights:*
- **Cluster 0** represents high-engagement power users who watch across almost all available genres (18.8 out of 19 genres), rate actively (506+ movies), and have much higher recency (more recent interactions).
- **Cluster 1** represents mainstream viewers with fewer interactions (74 on average) who rate slightly higher on average (3.65 vs 3.49).

---

## 4. Top Genres (by Rating Volume)

Computed via Spark SQL:

| Rank | Genre | Total Ratings | Unique Movies | Average Rating |
|---|---|---|---|---|
| 1 | **Drama** | 8,857,853 | 13,062 | 3.67 |
| 2 | **Comedy** | 7,502,234 | 8,232 | 3.43 |
| 3 | **Action** | 5,614,208 | 3,466 | 3.44 |
| 4 | **Thriller** | 5,313,506 | 4,129 | 3.51 |
| 5 | **Adventure** | 4,380,351 | 2,287 | 3.50 |
| 6 | **Romance** | 3,802,002 | 4,029 | 3.54 |
| 7 | **Crime** | 3,298,335 | 2,889 | 3.67 |
| 8 | **Sci-Fi** | 3,150,141 | 1,720 | 3.44 |
| 9 | **Fantasy** | 2,111,403 | 1,398 | 3.51 |
| 10 | **Children** | 1,669,249 | 1,118 | 3.41 |

---

## 5. Top 10 Most Rated Content (Content Popularity)

| Movie ID | Title | Genres | Interaction Count | Unique Users | Avg Rating | Popularity Score |
|---|---|---|---|---|---|---|
| 296 | **Pulp Fiction (1994)** | Comedy\|Crime\|Drama\|Thriller | 67,310 | 67,310 | 4.17 | 46.41 |
| 356 | **Forrest Gump (1994)** | Comedy\|Drama\|Romance\|War | 66,172 | 66,172 | 4.03 | 44.72 |
| 318 | **Shawshank Redemption, The (1994)** | Crime\|Drama | 63,366 | 63,366 | **4.45** | **49.17** |
| 593 | **Silence of the Lambs, The (1991)** | Crime\|Horror\|Thriller | 63,299 | 63,299 | 4.18 | 46.18 |
| 480 | **Jurassic Park (1993)** | Action\|Adventure\|Sci-Fi\|Thriller | 59,715 | 59,715 | 3.66 | 40.30 |
| 260 | **Star Wars: Episode IV (1977)** | Action\|Adventure\|Sci-Fi | 54,502 | 54,502 | 4.19 | 45.70 |
| 110 | **Braveheart (1995)** | Action\|Drama\|War | 53,769 | 53,769 | 4.04 | 44.03 |
| 589 | **Terminator 2: Judgment Day (1991)** | Action\|Sci-Fi | 52,244 | 52,244 | 3.93 | 42.72 |
| 2571 | **Matrix, The (1999)** | Action\|Sci-Fi\|Thriller | 51,334 | 51,334 | 4.19 | 45.41 |
| 527 | **Schindler's List (1993)** | Drama\|War | 50,054 | 50,054 | 4.31 | 46.64 |

---

## 6. Top 10 Highest Rated Content (Threshold $\ge 10$ Ratings)

| Movie ID | Title | Genres | Avg Rating | Interactions | Popularity Score |
|---|---|---|---|---|---|
| 113315 | **Zero Motivation (2014)** | Comedy\|Drama | **4.500** | 11 | 11.18 |
| 318 | **Shawshank Redemption, The (1994)** | Crime\|Drama | **4.447** | 63,366 | **49.17** |
| 858 | **Godfather, The (1972)** | Crime\|Drama | **4.365** | 41,355 | 46.40 |
| 50 | **Usual Suspects, The (1995)** | Crime\|Mystery\|Thriller | **4.334** | 47,006 | 46.63 |
| 527 | **Schindler's List (1993)** | Drama\|War | **4.310** | 50,054 | 46.64 |
| 101850 | **Death on the Staircase (2004)** | Crime\|Documentary | **4.286** | 21 | 13.25 |
| 1221 | **Godfather: Part II, The (1974)** | Crime\|Drama | **4.276** | 27,398 | 43.69 |
| 2019 | **Seven Samurai (1954)** | Action\|Adventure\|Drama | **4.274** | 11,611 | 40.01 |
| 904 | **Rear Window (1954)** | Mystery\|Thriller | **4.271** | 17,449 | 41.72 |
| 27423 | **O Auto da Compadecida (2000)** | Adventure\|Comedy | **4.269** | 13 | 11.27 |

---

## 7. Sample Personalized Recommendations (User 1)

Filtered unseen items, generated via trained Spark MLlib ALS model:

| Rank | Title | Genres | ALS Predicted Rating |
|---|---|---|---|
| 1 | **8 Diagram Pole Fighter, The (1984)** | Action\|Drama | **4.650 / 5.0** |
| 2 | **Cat Came Back, The (1988)** | Action\|Animation\|Comedy | **4.583 / 5.0** |
| 3 | **Peking Opera Blues (1986)** | Action\|Comedy | **4.476 / 5.0** |
| 4 | **Classic, The (Klassikko) (2001)** | Comedy\|Drama | **4.460 / 5.0** |
| 5 | **I Don't Want to Sleep Alone (2006)** | Comedy\|Drama | **4.434 / 5.0** |

---

## 8. Data Quality Summary

- **Total initial rows in `rating.csv`**: 20,000,263
- **Null values across (userId, movieId, rating, timestamp)**: 0
- **Out of bounds ratings**: 0
- **Negative or zero IDs**: 0
- **Cleaned records stored in Parquet**: 20,000,263
