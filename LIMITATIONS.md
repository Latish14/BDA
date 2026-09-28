# Limitations – Movie/OTT Behaviour Analysis

## Dataset Limitations

### 1. No Watch Duration or Completion Rate
**Limitation:** MovieLens 20M contains only explicit ratings (0.5–5.0 stars).  
There is no watch duration, completion percentage, re-watch count, or engagement depth metric.

**Impact:**
- Cannot compute "engagement rate" in the OTT sense
- Cannot distinguish "rated immediately" vs "rated after re-watch"
- Popularity is approximated by rating count, not actual views

**Documented decision:** `popularity_score = log1p(interaction_count) × avg_rating` is used as a proxy and clearly labelled **DERIVED**.

---

### 2. No Hour-of-Day Granularity (Partial)
**Status:** The timestamp format `YYYY-MM-DD HH:MM:SS` **does** include hours.  
However, MovieLens timestamps represent when a user *submitted* a rating, not when they *watched* the movie.  
**Impact:** Hour-of-day analysis reflects rating submission habits, not actual viewing times.

---

### 3. No Subscription / Demographic Data
Age, gender, location, subscription tier, and device type are not present.  
User segmentation is based entirely on interaction behaviour (ratings, genres, frequency).

---

### 4. No Streaming-Session Data
MovieLens is a rating dataset, not a streaming-session log.  
There are no session start/end events, buffer events, or quality metrics.

---

### 5. Cold-Start Problem in ALS
Users with fewer than 5 ratings and movies with fewer than 10 ratings are excluded from ALS training.  
These users receive a genre-based content fallback recommendation instead.

---

### 6. Temporal Bias in Ratings
Older movies (1990s) have had more time to accumulate ratings.  
Recency-normalised popularity is not implemented in this version (future work).

---

### 7. Genome Scores Coverage
`genome_scores.csv` covers 11.7M (movieId, tagId) pairs but not all 27,278 movies have genome scores.  
Genome scores are available in the data but not currently incorporated into the recommendation model (content-based features are genre-only).

---

## Implementation Limitations

### 8. Single-Node Spark
The pipeline runs `local[*]` Spark (all cores on a single machine).  
This is appropriate for a 7th-semester project but would require a proper cluster (YARN/Kubernetes) for true distributed processing at scale.

### 9. ALS Hyperparameter Tuning
ALS is trained with fixed hyperparameters (rank=20, regParam=0.1).  
Cross-validated grid search is not implemented to keep runtime manageable.

### 10. K-Means Convergence
K-Means is run with `maxIter=30` and Euclidean distance.  
Results may vary slightly between runs due to random initialisation (seed is fixed to 42 for reproducibility).

---

## What This Project Does NOT Claim

- Watch hours or streaming sessions are **not fabricated**
- Completion rates are **not invented**  
- Recommendation scores are ALS **predicted ratings**, not actual user satisfaction scores
- Cluster labels (segments) are derived from actual cluster statistics, not pre-defined personas

---

## Future Work

- Incorporate genome tag relevance into a hybrid recommendation model
- Add hour-of-day analysis for viewing patterns
- Implement cross-validated ALS tuning
- Deploy on a real Spark cluster (AWS EMR / Databricks)
- Integrate optional Kafka streaming for real-time OTT simulation
