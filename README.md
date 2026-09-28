# Movie / OTT User Behaviour Analysis Using Big Data Analytics

> **B.Tech CSE (Data Science) – 7th Semester Project**  
> Dataset: MovieLens 20M · Engine: Apache Spark (PySpark) · Dashboard: Streamlit + Plotly

---

## Quick Start

```bash
# 1. Create virtual environment (uv)
uv venv .venv --python 3.14
# Activate:  .venv\Scripts\activate   (Windows)

# 2. Install dependencies
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 3. Run full pipeline (first time – ~15–30 min on 20M rows)
.venv\Scripts\python.exe run_pipeline.py

# 4. Run with existing Parquet (skip slow ingestion/cleaning)
.venv\Scripts\python.exe run_pipeline.py --skip-ingest --skip-clean --skip-features --skip-segment

# 5. Launch dashboard
.venv\Scripts\streamlit.exe run dashboard/app.py

# 6. Run tests
.venv\Scripts\pytest.exe tests/ -v
```

---

## Project Structure

```
BDA Proj/
├── data/
│   ├── raw/              # Original CSV files (untouched)
│   ├── processed/        # Cleaned Parquet files
│   └── synthetic/        # (optional) synthetic streaming events
├── src/
│   ├── ingestion/        # Phase 1 – CSV → Parquet ingestion
│   ├── preprocessing/    # Phase 2 – Cleaning · Phase 3 – Features
│   ├── analytics/        # Phase 4 – Spark SQL analytics engine
│   ├── segmentation/     # Phase 5 – K-Means clustering
│   ├── recommendation/   # Phase 6 – ALS collaborative filtering
│   └── utils/            # Spark session factory, logger, timer
├── dashboard/
│   └── app.py            # Streamlit multi-page dashboard
├── config/
│   └── settings.py       # Centralised config (paths, hyperparams)
├── tests/
│   └── test_pipeline.py  # pytest test suite
├── outputs/              # JSON timings, saved results
├── run_pipeline.py       # Master orchestrator
├── requirements.txt
├── README.md
├── DATASET_ANALYSIS.md
├── ARCHITECTURE.md
├── RESULTS.md
└── LIMITATIONS.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Processing engine | Apache Spark 3.5 (PySpark) |
| Query layer | Spark SQL (temp views) |
| ML / Clustering | Spark MLlib – K-Means, ALS |
| Storage format | Apache Parquet (Snappy compression) |
| Dashboard | Streamlit 1.28+ |
| Visualisation | Plotly |
| Python | 3.14 |
| Java | OpenJDK 25 |

---

## Dataset

| File | Rows | Description |
|---|---|---|
| `rating.csv` | 20,000,263 | User–movie ratings (userId, movieId, rating, timestamp) |
| `movie.csv` | 27,278 | Movie catalogue (movieId, title, genres) |
| `tag.csv` | 465,564 | User-applied tags |
| `link.csv` | 27,278 | IMDb / TMDb cross-references |
| `genome_tags.csv` | 1,128 | Tag vocabulary |
| `genome_scores.csv` | 11,709,768 | Movie–tag relevance scores |

---

## Pipeline Phases

1. **Ingestion** – reads raw CSV with explicit schemas → Parquet
2. **Cleaning** – drops nulls / duplicates / out-of-range ratings
3. **Feature Engineering** – user & movie aggregated features
4. **Analytics** – Spark SQL behavioural & content queries
5. **Segmentation** – K-Means (MLlib) with silhouette selection
6. **Recommendation** – ALS collaborative filtering (MLlib)
7. **Dashboard** – Streamlit + Plotly interactive UI

---

## Dashboard Pages

| Page | Contents |
|---|---|
| 📊 Overview | KPIs, rating distribution, genre popularity, activity timeline |
| 👤 User Behaviour | User profile lookup, most active users |
| 🎥 Content Analytics | Most/highest rated movies, genre statistics |
| 📅 Temporal Analysis | Yearly trends, avg rating, day-of-week patterns |
| 🔵 Segmentation | Cluster sizes, characteristics, scatter plot |
| 🎯 Recommendations | ALS top-N with predicted scores |

---

## Pipeline CLI Options

```
--skip-ingest     Skip CSV ingestion (Parquet already exists)
--skip-clean      Skip cleaning phase
--skip-features   Skip feature engineering
--skip-segment    Skip K-Means segmentation
--skip-train      Load existing ALS model instead of retraining
--k K             Force number of K-Means clusters
--demo-user UID   User ID for demo recommendation output
```

---

## Notes

- Watch duration / completion data is **not available** in MovieLens 20M — see `LIMITATIONS.md`
- Cold-start users (< 5 ratings) may not receive ALS recommendations
- All derived/synthetic fields are clearly labelled — see `DATASET_ANALYSIS.md`
