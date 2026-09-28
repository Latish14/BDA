"""
Streamlit Dashboard – Movie/OTT User Behaviour Analysis
Run with: streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OTT Behaviour Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Spark session (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Starting Spark session…")
def get_engine():
    from src.utils.spark_session import get_spark
    from src.analytics.analytics import AnalyticsEngine
    spark = get_spark("OTT_Dashboard")
    return AnalyticsEngine(spark), spark


@st.cache_data(show_spinner="Loading data…", ttl=3600)
def cached_overview():
    engine, _ = get_engine()
    return engine.overview_stats()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_genre_popularity():
    engine, _ = get_engine()
    return engine.genre_popularity().toPandas()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_yearly():
    engine, _ = get_engine()
    return engine.yearly_trends().toPandas()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_weekday():
    engine, _ = get_engine()
    return engine.weekday_activity().toPandas()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_rating_dist():
    engine, _ = get_engine()
    return engine.user_rating_distribution().toPandas()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_most_rated(n=25):
    engine, _ = get_engine()
    return engine.most_rated_movies(n).toPandas()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_highest_rated(n=25):
    engine, _ = get_engine()
    return engine.highest_rated_movies(n).toPandas()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_genre_avg():
    engine, _ = get_engine()
    return engine.genre_avg_rating().toPandas()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_most_active(n=20):
    engine, _ = get_engine()
    return engine.most_active_users(n).toPandas()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_segments():
    from config.settings import PARQUET
    _, spark = get_engine()
    try:
        return spark.read.parquet(PARQUET["segments"]).toPandas()
    except Exception:
        return pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 OTT Analytics")
st.sidebar.markdown("**MovieLens 20M Dataset**")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "👤 User Behaviour",
    "🎥 Content Analytics",
    "📅 Temporal Analysis",
    "🔵 Segmentation",
    "🎯 Recommendations",
])

# ── PAGE: Overview ────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Dataset Overview")
    st.caption("MovieLens 20M · Source: GroupLens Research · Processed with Apache Spark")

    with st.spinner("Computing overview statistics…"):
        stats = cached_overview()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Ratings",    f"{stats['total_ratings']:,}")
    c2.metric("Unique Users",     f"{stats['total_users']:,}")
    c3.metric("Unique Movies",    f"{stats['total_movies']:,}")
    c4.metric("Avg Rating",       f"{stats['avg_rating']:.2f} / 5.0")
    c5.metric("Top Genre",        stats['top_genre'])

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rating Distribution")
        rd = cached_rating_dist()
        fig = px.bar(rd, x="rating", y="count", color="count",
                     color_continuous_scale="Blues", labels={"count": "# Ratings"})
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Genre Popularity (by # Ratings)")
        gp = cached_genre_popularity().head(15)
        fig = px.bar(gp, x="total_ratings", y="genre", orientation="h",
                     color="avg_rating", color_continuous_scale="RdYlGn",
                     labels={"total_ratings": "Total Ratings", "genre": "Genre"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Activity Over the Years")
    yr = cached_yearly()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yr["year"], y=yr["rating_count"],
                             mode="lines+markers", name="Ratings", fill="tozeroy"))
    fig.add_trace(go.Scatter(x=yr["year"], y=yr["active_users"],
                             mode="lines+markers", name="Active Users",
                             yaxis="y2", line=dict(dash="dash")))
    fig.update_layout(
        yaxis=dict(title="Rating Count"),
        yaxis2=dict(title="Active Users", overlaying="y", side="right"),
        height=380, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE: User Behaviour ──────────────────────────────────────────────────────
elif page == "👤 User Behaviour":
    st.title("👤 User Behaviour Analysis")

    tab1, tab2 = st.tabs(["User Profile", "Most Active Users"])

    with tab1:
        user_id = st.number_input("Enter User ID", min_value=1, max_value=138493, value=1, step=1)
        if st.button("Get Profile"):
            engine, _ = get_engine()
            with st.spinner("Loading user profile…"):
                profile = engine.get_user_profile(int(user_id))
            if "error" in profile:
                st.error(profile["error"])
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Interactions", f"{profile.get('interaction_count', 0):,}")
                col2.metric("Unique Movies", f"{profile.get('unique_movies', 0):,}")
                col3.metric("Avg Rating", f"{(profile.get('avg_rating') or 0):.2f}")

                col4, col5, col6 = st.columns(3)
                col4.metric("Top Genre", str(profile.get("top_genre") or "N/A"))
                col5.metric("Genre Diversity", str(profile.get("genre_diversity") or "N/A"))
                col6.metric("Segment", str(profile.get("segment", "N/A")))

                st.markdown("**Activity Window**")
                st.write(f"- First interaction: `{profile.get('first_interaction')}`")
                st.write(f"- Last interaction:  `{profile.get('last_interaction')}`")
                st.write(f"- Days active span:  `{profile.get('days_active', 0)}`")
                st.write(f"- Recency (days ago):`{profile.get('recency_days', 0)}`")

                st.markdown("**Viewing Habits**")
                weekend = profile.get("weekend_count") or 0
                weekday = profile.get("weekday_count") or 0
                total   = weekend + weekday or 1
                habit_df = pd.DataFrame({
                    "Day Type": ["Weekday", "Weekend"],
                    "Ratings":  [weekday, weekend],
                    "Share":    [f"{100*weekday/total:.1f}%", f"{100*weekend/total:.1f}%"],
                })
                st.dataframe(habit_df, hide_index=True)

    with tab2:
        st.subheader("Top 20 Most Active Users")
        ma = cached_most_active(20)
        fig = px.bar(ma, x="userId", y="interaction_count", color="avg_rating",
                     color_continuous_scale="Viridis",
                     hover_data=["unique_movies", "top_genre"],
                     labels={"userId": "User ID", "interaction_count": "# Ratings"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ma, hide_index=True)

# ── PAGE: Content Analytics ───────────────────────────────────────────────────
elif page == "🎥 Content Analytics":
    st.title("🎥 Content & Genre Analytics")

    tab1, tab2, tab3 = st.tabs(["Most Rated Movies", "Highest Rated Movies", "Genre Statistics"])

    with tab1:
        st.subheader("Top 25 Most Rated Movies")
        mr = cached_most_rated(25)
        fig = px.bar(mr, x="interaction_count", y="title", orientation="h",
                     color="avg_rating", color_continuous_scale="RdYlGn",
                     hover_data=["genres", "unique_users"],
                     labels={"interaction_count": "# Ratings", "title": ""})
        fig.update_layout(height=700, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(mr[["title","genres","interaction_count","avg_rating"]],
                     hide_index=True)

    with tab2:
        st.subheader("Top 25 Highest Rated Movies (min 10 ratings)")
        hr = cached_highest_rated(25)
        fig = px.scatter(hr, x="interaction_count", y="avg_rating",
                         size="interaction_count", color="avg_rating",
                         hover_name="title", hover_data=["genres"],
                         color_continuous_scale="RdYlGn", size_max=40,
                         labels={"interaction_count": "# Ratings", "avg_rating": "Avg Rating"})
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hr[["title","genres","avg_rating","interaction_count"]],
                     hide_index=True)

    with tab3:
        st.subheader("Average Rating per Genre")
        ga = cached_genre_avg()
        fig = px.bar(ga, x="genre", y="avg_rating", color="rating_count",
                     color_continuous_scale="Blues",
                     labels={"avg_rating": "Avg Rating", "rating_count": "# Ratings"})
        fig.update_layout(height=400, xaxis_tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Genre Coverage (# unique movies)")
        gp = cached_genre_popularity()
        fig2 = px.pie(gp, values="unique_movies", names="genre",
                      hole=0.4, title="Movies per Genre")
        fig2.update_layout(height=480)
        st.plotly_chart(fig2, use_container_width=True)

# ── PAGE: Temporal Analysis ───────────────────────────────────────────────────
elif page == "📅 Temporal Analysis":
    st.title("📅 Temporal Analysis")

    tab1, tab2 = st.tabs(["Yearly / Monthly Trends", "Day-of-Week Patterns"])

    with tab1:
        st.subheader("Yearly Rating Trends (1995–2016)")
        yr = cached_yearly()
        fig = px.area(yr, x="year", y="rating_count",
                      labels={"rating_count": "# Ratings", "year": "Year"},
                      color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(yr, x="year", y="avg_rating",
                       markers=True, labels={"avg_rating": "Average Rating"},
                       title="Average Rating Per Year")
        fig2.update_traces(line_color="#EF553B")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Active Users Over Time")
        fig3 = px.bar(yr, x="year", y="active_users",
                      labels={"active_users": "Unique Users"})
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader("Rating Activity by Day of Week")
        wd = cached_weekday()
        day_map = {1:"Sun",2:"Mon",3:"Tue",4:"Wed",5:"Thu",6:"Fri",7:"Sat"}
        wd["day_name"] = wd["weekday"].map(day_map)
        fig = px.bar(wd, x="day_name", y="rating_count", color="avg_rating",
                     color_continuous_scale="Plasma",
                     labels={"rating_count": "# Ratings", "day_name": "Day"})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Peak Day", wd.loc[wd["rating_count"].idxmax(), "day_name"])
        col2.metric("Quietest Day", wd.loc[wd["rating_count"].idxmin(), "day_name"])

# ── PAGE: Segmentation ────────────────────────────────────────────────────────
elif page == "🔵 Segmentation":
    st.title("🔵 User Segmentation (K-Means)")
    st.info("Clusters derived via Spark MLlib K-Means with silhouette evaluation. "
            "Labels describe actual cluster statistics, not pre-defined archetypes.")

    seg_df = cached_segments()
    if seg_df.empty:
        st.warning("Run the pipeline first: `python run_pipeline.py --skip-ingest --skip-clean`")
    else:
        n_clusters = seg_df["segment"].nunique()
        st.metric("Number of Clusters", n_clusters)

        # Cluster sizes
        sizes = seg_df.groupby("segment").size().reset_index(name="size")
        fig = px.pie(sizes, values="size", names="segment",
                     title="Cluster Sizes", hole=0.35,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster characteristics
        st.subheader("Cluster Characteristics")
        stats_cols = ["interaction_count","avg_rating","unique_movies",
                      "genre_diversity","recency_days","rating_variance"]
        cluster_stats = seg_df.groupby("segment")[stats_cols].mean().round(2).reset_index()
        st.dataframe(cluster_stats, hide_index=True)

        # Scatter: interactions vs avg_rating
        fig2 = px.scatter(
            seg_df.sample(min(5000, len(seg_df))),
            x="interaction_count", y="avg_rating",
            color=seg_df["segment"].astype(str).reindex(
                seg_df.sample(min(5000, len(seg_df))).index
            ),
            opacity=0.6, size_max=8,
            labels={"interaction_count": "# Interactions", "avg_rating": "Avg Rating"},
            title="User Scatter by Segment",
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)

        # Top genre per cluster
        top_genres = (
            seg_df.dropna(subset=["top_genre"])
            .groupby(["segment","top_genre"]).size()
            .reset_index(name="count")
            .sort_values(["segment","count"], ascending=[True,False])
            .groupby("segment").first().reset_index()
        )
        st.subheader("Most Common Top Genre per Cluster")
        st.dataframe(top_genres[["segment","top_genre","count"]], hide_index=True)

# ── PAGE: Recommendations ─────────────────────────────────────────────────────
elif page == "🎯 Recommendations":
    st.title("🎯 Personalised Recommendations")
    st.info("Powered by ALS Collaborative Filtering (Spark MLlib). "
            "Predicted scores are the model's estimated rating for that user-movie pair.")

    user_id = st.number_input("Select User ID", min_value=1, max_value=138493, value=1, step=1)
    n_recs  = st.slider("Number of Recommendations", 5, 20, 10)

    if st.button("Get Recommendations", type="primary"):
        from src.recommendation.recommender import RecommendationEngine
        _, spark = get_engine()
        with st.spinner("Generating recommendations…"):
            try:
                rec_engine = RecommendationEngine(spark)
                rec_engine.train(force=False)
                recs = rec_engine.recommend_movies(int(user_id), n=n_recs)

                if not recs:
                    st.warning("No recommendations generated (user may be new or filtered out).")
                    st.subheader("Content-Based Fallback (Genre Match)")
                    fb = rec_engine.genre_based_recommendations(int(user_id), n=n_recs)
                    st.dataframe(fb.toPandas(), hide_index=True)
                else:
                    recs_df = pd.DataFrame(recs)
                    recs_df.index += 1
                    st.subheader(f"Top {len(recs)} Recommendations for User {user_id}")
                    fig = px.bar(recs_df, x="predicted_score", y="title",
                                 orientation="h", color="predicted_score",
                                 color_continuous_scale="RdYlGn",
                                 hover_data=["genres"],
                                 labels={"predicted_score": "Predicted Rating",
                                         "title": "Movie"})
                    fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

                    for _, row in recs_df.iterrows():
                        with st.expander(f"#{int(row.name)}  {row['title']}"):
                            st.write(f"**Genres:** {row['genres']}")
                            st.write(f"**Predicted Score:** {row['predicted_score']:.3f} / 5.0")
                            st.caption(
                                "Score is ALS model's predicted rating based on collaborative "
                                "patterns of similar users."
                            )
            except Exception as e:
                st.error(f"Recommendation error: {e}")
                st.info("Ensure the pipeline has been run: `python run_pipeline.py`")

    st.markdown("---")
    st.caption(
        "Note: Watch duration / completion metrics are **not available** in MovieLens 20M. "
        "Recommendations are based solely on explicit ratings. "
        "Cold-start users (< 5 ratings) may not receive ALS-based recommendations."
    )
