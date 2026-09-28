# Dataset Analysis – MovieLens 20M

> Created by inspecting all 6 raw CSV files before any processing.  
> **No columns were assumed or invented.**

---

## Files Inspected

| File | Rows | Columns | Size |
|---|---|---|---|
| `rating.csv` | 20,000,263 | 4 | ~658 MB |
| `movie.csv` | 27,278 | 3 | ~1.4 MB |
| `tag.csv` | 465,564 | 4 | ~20.7 MB |
| `link.csv` | 27,278 | 3 | ~527 KB |
| `genome_tags.csv` | 1,128 | 2 | ~20 KB |
| `genome_scores.csv` | 11,709,768 | 3 | ~204 MB |

---

## Column Classification

### rating.csv

| Column | Type | Classification | Notes |
|---|---|---|---|
| `userId` | Integer | **RAW** | Anonymous user ID, range 1–138493 |
| `movieId` | Integer | **RAW** | Movie ID, joins to movie.csv |
| `rating` | Float | **RAW** | 0.5–5.0 in 0.5 steps |
| `timestamp` | String → Timestamp | **RAW** | `YYYY-MM-DD HH:MM:SS` format |
| `year` | Integer | **DERIVED** | Extracted from timestamp |
| `month` | Integer | **DERIVED** | Extracted from timestamp |
| `weekday` | Integer | **DERIVED** | 1=Sun … 7=Sat |
| `is_weekend` | Integer | **DERIVED** | 1 if Sat/Sun, else 0 |

### movie.csv

| Column | Type | Classification | Notes |
|---|---|---|---|
| `movieId` | Integer | **RAW** | Primary key |
| `title` | String | **RAW** | Includes release year e.g. `Toy Story (1995)` |
| `genres` | String | **RAW** | Pipe-delimited genre list |
| `release_year` | Integer | **DERIVED** | Regex-extracted from title |
| `genre_array` | Array[String] | **DERIVED** | Split from `genres` on `\|` |

### tag.csv

| Column | Type | Classification | Notes |
|---|---|---|---|
| `userId` | Integer | **RAW** | |
| `movieId` | Integer | **RAW** | |
| `tag` | String | **RAW** | Free-text; normalised (trim/lower) in cleaning |
| `timestamp` | String → Timestamp | **RAW** | Same format as ratings |

### link.csv

| Column | Type | Classification | Notes |
|---|---|---|---|
| `movieId` | Integer | **RAW** | Joins to movie.csv |
| `imdbId` | String | **RAW** | IMDb numeric ID (leading zeros possible) |
| `tmdbId` | String | **RAW** | TMDb numeric ID |

### genome_tags.csv

| Column | Type | Classification | Notes |
|---|---|---|---|
| `tagId` | Integer | **RAW** | |
| `tag` | String | **RAW** | Controlled vocabulary, 1128 tags |

### genome_scores.csv

| Column | Type | Classification | Notes |
|---|---|---|---|
| `movieId` | Integer | **RAW** | |
| `tagId` | Integer | **RAW** | |
| `relevance` | Float | **RAW** | 0.0–1.0, model-computed relevance score |

---

## User Feature Columns (all DERIVED from raw rating.csv)

| Column | Description |
|---|---|
| `interaction_count` | Total ratings given by user |
| `unique_movies` | Distinct movies rated |
| `avg_rating` | Mean rating |
| `rating_variance` | Variance of all ratings |
| `rating_std` | Standard deviation of ratings |
| `min_rating` / `max_rating` | Extremes |
| `first_interaction` | Earliest rating timestamp |
| `last_interaction` | Latest rating timestamp |
| `days_active` | Span of activity in days |
| `recency_days` | Days since last rating (from dataset max) |
| `weekday_count` / `weekend_count` | Ratings by day type |
| `genre_diversity` | Number of distinct genres rated |
| `top_genre` | Single most-rated genre |

## Movie Feature Columns (all DERIVED)

| Column | Description |
|---|---|
| `interaction_count` | Total ratings received |
| `unique_users` | Distinct raters |
| `avg_rating` | Mean rating |
| `rating_variance` | Variance of ratings |
| `popularity_score` | `log1p(interaction_count) × avg_rating` |

---

## Dataset Relationships

```
rating.csv ──[movieId]──► movie.csv
rating.csv ──[movieId]──► link.csv
tag.csv    ──[movieId]──► movie.csv
genome_scores.csv ──[movieId]──► movie.csv
genome_scores.csv ──[tagId]──► genome_tags.csv
```

---

## Missing / Unavailable Fields

| Field | Status |
|---|---|
| Watch duration | **NOT AVAILABLE** – ratings only |
| Completion rate | **NOT AVAILABLE** |
| Device type | **NOT AVAILABLE** |
| Subscription tier | **NOT AVAILABLE** |
| Hour of day | **NOT AVAILABLE** – timestamp has date but hour is present (HH in `HH:MM:SS`) ✓ |

> **Note on hour:** The timestamp format `YYYY-MM-DD HH:MM:SS` does include hour.  
> Hour-of-day analysis is therefore available if enabled.

---

## Data Quality Summary (raw)

| File | Known Issues |
|---|---|
| `rating.csv` | Some duplicate (userId, movieId) pairs (user re-rated a movie) |
| `movie.csv` | Some movies listed as `(no genres listed)` |
| `tag.csv` | Free-text tags – requires normalisation |
| `link.csv` | `tmdbId` may have blanks for older movies |
| `genome_scores.csv` | Not all movies have genome scores |

---

## Synthetic Data

No synthetic data is used in the core pipeline.  
The `data/synthetic/` directory is reserved for optional Kafka streaming simulation (Phase: Optional Advanced Module).  
Any synthetic data generated will be clearly labelled `SYNTHETIC` in logs and documentation.
