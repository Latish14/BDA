"""
run_pipeline.py – Master pipeline orchestrator.
Runs all phases in sequence: Ingest → Clean → Features → Analytics → Segment → Recommend.
Usage:
    python run_pipeline.py [--skip-ingest] [--skip-train] [--k K]
"""
import sys, os, argparse, time, json
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.spark_session import get_spark, stop_spark
from src.utils.logger import get_logger, timer

logger = get_logger("pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="OTT Behaviour Analysis Pipeline")
    p.add_argument("--skip-ingest",  action="store_true", help="Skip CSV ingestion (Parquet already exists)")
    p.add_argument("--skip-clean",   action="store_true", help="Skip cleaning phase")
    p.add_argument("--skip-features",action="store_true", help="Skip feature engineering")
    p.add_argument("--skip-segment", action="store_true", help="Skip K-Means segmentation")
    p.add_argument("--skip-train",   action="store_true", help="Skip ALS training (load existing model)")
    p.add_argument("--k", type=int,  default=None,        help="Force K-Means k (default: auto)")
    p.add_argument("--demo-user",    type=int, default=1, help="User ID for demo recommendations")
    return p.parse_args()


def main():
    args = parse_args()
    timings: dict[str, float] = {}

    logger.info("+==================================================+")
    logger.info("|   Movie/OTT User Behaviour Analysis Pipeline     |")
    logger.info("+==================================================+")


    spark = get_spark("OTT_Pipeline")

    try:
        # Phase 1 – Ingestion
        if not args.skip_ingest:
            from src.ingestion.ingest import run_ingestion
            t0 = time.time()
            run_ingestion(spark)
            timings["ingestion"] = round(time.time() - t0, 2)
        else:
            logger.info("[SKIP] Ingestion")

        # Phase 2 – Cleaning
        if not args.skip_clean:
            from src.preprocessing.clean import run_cleaning
            t0 = time.time()
            run_cleaning(spark)
            timings["cleaning"] = round(time.time() - t0, 2)
        else:
            logger.info("[SKIP] Cleaning")

        # Phase 3 – Feature Engineering
        if not args.skip_features:
            from src.preprocessing.features import run_feature_engineering
            t0 = time.time()
            run_feature_engineering(spark)
            timings["features"] = round(time.time() - t0, 2)
        else:
            logger.info("[SKIP] Feature Engineering")

        # Phase 4 – Analytics Summary
        from src.analytics.analytics import run_analytics
        t0 = time.time()
        engine = run_analytics(spark)
        stats  = engine.overview_stats()
        timings["analytics"] = round(time.time() - t0, 2)
        logger.info("OVERVIEW: %s", stats)

        # Phase 5 – Segmentation
        if not args.skip_segment:
            from src.segmentation.cluster import run_segmentation
            t0 = time.time()
            _, cluster_stats, sil_scores = run_segmentation(spark, k=args.k)
            timings["segmentation"] = round(time.time() - t0, 2)
        else:
            logger.info("[SKIP] Segmentation")

        # Phase 6 – Recommendations
        from src.recommendation.recommender import run_recommendation
        t0 = time.time()
        rec_engine, rmse = run_recommendation(spark, force_train=not args.skip_train)
        timings["recommendation"] = round(time.time() - t0, 2)
        if rmse:
            logger.info("ALS RMSE: %.4f", rmse)

        # Demo recommendations
        recs = rec_engine.recommend_movies(args.demo_user, n=5)
        logger.info("Top-5 recommendations for user %d:", args.demo_user)
        for r in recs:
            logger.info("  %-50s [%s]  score=%.3f",
                        r["title"], r["genres"], r["predicted_score"])

        # Save timings
        out_path = os.path.join(os.path.dirname(__file__), "outputs", "pipeline_timings.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(timings, f, indent=2)
        logger.info("Pipeline timings saved: %s", out_path)

    finally:
        stop_spark(spark)

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  PIPELINE COMPLETE  Total phases: %d        ║", len(timings))
    logger.info("╚══════════════════════════════════════════════╝")
    for phase, secs in timings.items():
        logger.info("  %-20s  %6.1f s", phase, secs)


if __name__ == "__main__":
    main()
