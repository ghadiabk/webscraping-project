# app.py
# Lebanon Short-Stay Market Explorer + rich insights + NLP search

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Config
# -------------------------------------------------
DATA_PATH = "cleaned_scraping_project.csv"

st.set_page_config(
    page_title="Lebanon Short-Stay Market Explorer",
    layout="wide",
)

# -------------------------------------------------
# Data loading
# -------------------------------------------------


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    # robust CSV read (commas inside quotes)
    df = pd.read_csv(
        path,
        engine="python",
        quotechar='"',
        escapechar="\\",
    )

    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # expected columns
    expected = [
        "platform",
        "listing_id",
        "title",
        "city",
        "bedrooms",
        "bathrooms",
        "beds",
        "price",
        "rating",
        "review_count",
        "amenities_count",
        "minimum_nights",
        "last_scraped",
    ]

    # ensure required columns exist
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # fallback listing_id if missing / all NaN
    if df["listing_id"].isna().all():
        df["listing_id"] = np.arange(1, len(df) + 1)

    # numeric conversion
    num_cols = [
        "bedrooms",
        "bathrooms",
        "beds",
        "price",
        "rating",
        "review_count",
        "amenities_count",
        "minimum_nights",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # basic string cleaning
    df["city"] = df["city"].astype(str).str.strip()
    df["city"] = df["city"].replace({"nan": np.nan, "": np.nan})

    df["title"] = df["title"].astype(str).str.strip()

    # parse date (optional)
    df["last_scraped"] = pd.to_datetime(df["last_scraped"], errors="coerce")

    # drop perfect duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    return df


df = load_data()


# -------------------------------------------------
# FIX CITY COLUMN (remove false city entries)
# -------------------------------------------------

df["city"] = df["city"].astype(str).str.strip()

bad_city_patterns = (
    df["city"].str.contains(r"[0-9]", regex=True) |   # contains numbers
    df["city"].str.contains(",", regex=False) |       # contains commas → full addresses
    df["city"].str.contains("Apartment", case=False) |
    df["city"].str.contains("Studio", case=False) |
    df["city"].str.contains("Sq", case=False) |
    df["city"].str.contains("BR", case=False) |
    df["city"].str.contains("Guest", case=False)
)

# also remove extremely long fake cities (> 25 chars)
too_long = df["city"].str.len() > 25

df.loc[bad_city_patterns | too_long, "city"] = np.nan

# drop empty city values from sidebar filter only (not from data)
clean_city_values = (
    df["city"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

if df.empty:
    st.error(
        "Dataframe is empty. Ensure `cleaned_scraping_project.csv` "
        "is in the same directory as this app."
    )
    st.stop()

# -------------------------------------------------
# Derived features (on full data)
# -------------------------------------------------

# price-based efficiency
df["price_per_bed"] = df["price"] / df["beds"].replace(0, np.nan)
df["price_per_bedroom"] = df["price"] / df["bedrooms"].replace(0, np.nan)

# reviews per dollar
df["value_index"] = df["review_count"] / df["price"]
df.loc[~np.isfinite(df["value_index"]), "value_index"] = np.nan

# composite score (0–1): cheap + many reviews + many amenities
def _minmax(col: pd.Series) -> pd.Series:
    col = col.copy()
    if col.max() > col.min():
        return (col - col.min()) / (col.max() - col.min())
    return pd.Series(0, index=col.index)


score_price_inv = 1 - _minmax(df["price"])
score_reviews = _minmax(df["review_count"])
score_amenities = _minmax(df["amenities_count"])

df["composite_score"] = (
    0.45 * score_price_inv
    + 0.30 * score_reviews
    + 0.25 * score_amenities
)

# price segments
try:
    df["price_segment"] = pd.qcut(
        df["price"], q=4, labels=["Budget", "Mid-range", "Premium", "Luxury"]
    )
except ValueError:
    df["price_segment"] = "Unknown"

# trust score = rating * log(1 + reviews)
df["trust_score"] = df["rating"] * np.log1p(df["review_count"])

# -------------------------------------------------
# Sidebar filters (working behaviour preserved)
# -------------------------------------------------

st.sidebar.header("Filters")

platforms = sorted(df["platform"].dropna().unique())
selected_platforms = st.sidebar.multiselect(
    "Platform",
    options=platforms,
    default=platforms,
)

city_values = (
    df.loc[df["city"].notna(), "city"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)
selected_cities = st.sidebar.multiselect(
    "City",
    options=clean_city_values,
    default=clean_city_values,
)


price_min = float(df["price"].dropna().min()) if df["price"].notna().any() else 0.0
price_max = float(df["price"].dropna().max()) if df["price"].notna().any() else 1000.0
price_range = st.sidebar.slider(
    "Nightly price range (USD)",
    min_value=0.0,
    max_value=max(price_max, 100.0),
    value=(price_min, price_max),
    step=10.0,
)

rating_min = float(df["rating"].dropna().min()) if df["rating"].notna().any() else 0.0
rating_max = float(df["rating"].dropna().max()) if df["rating"].notna().any() else 5.0
rating_threshold = st.sidebar.slider(
    "Minimum rating",
    min_value=0.0,
    max_value=max(5.0, rating_max),
    value=min(4.0, rating_max),
    step=0.1,
)

min_reviews = st.sidebar.slider(
    "Minimum reviews",
    min_value=0,
    max_value=int(df["review_count"].fillna(0).max()),
    value=0,
    step=5,
)

# apply filters
data = df.copy()
if selected_platforms:
    data = data[data["platform"].isin(selected_platforms)]
if selected_cities:
    data = data[data["city"].isin(selected_cities)]

data = data[
    (data["price"].fillna(0) >= price_range[0])
    & (data["price"].fillna(0) <= price_range[1])
    & (data["rating"].fillna(0) >= rating_threshold)
    & (data["review_count"].fillna(0) >= min_reviews)
].copy()

st.sidebar.markdown(f"**Filtered listings:** {len(data):,}")

if data.empty:
    st.warning("No listings match the current filters. Adjust filters to see results.")
    st.stop()

# -------------------------------------------------
# KPIs
# -------------------------------------------------

 
df_reviews_only = data[data["review_count"] > 0].copy()

median_reviews = (
    df_reviews_only["review_count"].median()
    if not df_reviews_only.empty else 0
)

st.title("Lebanon Short-Stay Market Explorer")
st.caption(
    "Combined view of Airbnb, Stayinn and LebanonRental listings in Lebanon. "
    "Explore city-level patterns, value-for-money, amenities and search "
    "for best-fit stays using NLP."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total listings", f"{len(data)}")
with col2:
    med_price = data["price"].median()
    st.metric(
        "Median nightly price (USD)",
        f"{med_price:,.0f}" if not np.isnan(med_price) else "N/A",
    )
with col3:
    med_rating = data["rating"].median()
    st.metric(
        "Median rating",
        f"{med_rating:.2f}" if not np.isnan(med_rating) else "N/A",
    )
with col4:
    st.metric("Median reviews", f"{median_reviews:.0f}")


# -------------------------------------------------
# Tabs
# -------------------------------------------------

tab_overview, tab_cities, tab_value, tab_amenities, tab_segments, tab_best, tab_search = st.tabs(
    [
        "📊 Overview",
        "🏙 City Insights",
        "💰 Value for Money",
        "🧩 Amenities & Quality",
        "📐 Segments & Outliers",
        "🏅 Best Listings",
        "🔎 NLP Search",
    ]
)

# =================================================
# 1) Overview
# =================================================
with tab_overview:
    st.subheader("Platform mix and core distributions (filtered)")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Listings by platform**")
        plat_counts = (
            data["platform"]
            .value_counts()
            .rename_axis("platform")
            .reset_index(name="count")
        )
        fig = px.bar(
            plat_counts,
            x="platform",
            y="count",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Median price per platform (USD)**")
        med_price_platform = (
            data.groupby("platform")["price"].median().reset_index()
        )
        fig = px.bar(
            med_price_platform,
            x="platform",
            y="price",
            labels={"price": "Median price (USD)"},
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**Nightly price distribution (USD)**")
        fig = px.histogram(
            data,
            x="price",
            nbins=40,
            color="platform",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown("**Rating distribution**")
        fig = px.histogram(
            data,
            x="rating",
            nbins=25,
            color="platform",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top cities by number of listings (filtered)**")
    top_city_listings = (
        data.groupby("city")["listing_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(20)
        .reset_index(name="listings")
    )
    fig = px.bar(
        top_city_listings,
        x="city",
        y="listings",
        template="plotly_dark",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Listings")
    st.plotly_chart(fig, use_container_width=True)

# =================================================
# 2) City Insights
# =================================================
with tab_cities:
    st.subheader("City-level supply, pricing and demand (filtered)")

    city_stats = (
        data.groupby("city")
        .agg(
            listings=("city", "size"),
            median_price=("price", "median"),
            median_rating=("rating", "median"),
            median_reviews=("review_count", "median"),
        )
        .sort_values("listings", ascending=False)
        .reset_index()
    )

    st.markdown("**City metrics table**")
    st.dataframe(
        city_stats.style.format(
            {
                "median_price": "{:,.0f}",
                "median_rating": "{:.2f}",
                "median_reviews": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top cities by median nightly price (USD)**")
        fig = px.bar(
            city_stats.sort_values("median_price", ascending=False).head(10),
            x="city",
            y="median_price",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Median price vs median reviews per city**")
        fig = px.scatter(
            city_stats,
            x="median_price",
            y="median_reviews",
            size="listings",
            hover_name="city",
            template="plotly_dark",
        )
        fig.update_layout(
            xaxis_title="Median price (USD)",
            yaxis_title="Median reviews",
        )
        st.plotly_chart(fig, use_container_width=True)

# =================================================
# 3) Value for Money
# =================================================
with tab_value:
    st.subheader("Value-for-money: price vs demand")

    df_val = data.copy()
    df_val = df_val.dropna(subset=["price", "rating", "review_count"])

    if df_val.empty:
        st.info("Not enough data to compute value scores.")
    else:
        df_val["norm_rating"] = df_val["rating"] / df_val["rating"].max()
        df_val["norm_reviews"] = df_val["review_count"] / df_val["review_count"].max()
        df_val["norm_amen"] = df_val["amenities_count"] / df_val["amenities_count"].max()
        df_val["norm_price"] = df_val["price"] / df_val["price"].max()

        df_val["value_score"] = (
            0.45 * df_val["norm_rating"]
            + 0.25 * df_val["norm_reviews"]
            + 0.15 * df_val["norm_amen"].fillna(0)
            - 0.35 * df_val["norm_price"]
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Price vs Reviews**")
            fig = px.scatter(
                df_val,
                x="price",
                y="review_count",
                color="platform",
                hover_data=["title", "city"],
                template="plotly_dark",
            )
            fig.update_layout(
                xaxis_title="Price (USD / night)",
                yaxis_title="Review count",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Top value listings (high rating & reviews, low price)**")
            top_val = (
                df_val.sort_values("value_score", ascending=False)
                .head(20)[
                    [
                        "platform",
                        "city",
                        "title",
                        "price",
                        "rating",
                        "review_count",
                        "amenities_count",
                        "minimum_nights",
                        "value_score",
                    ]
                ]
            )
            st.dataframe(
                top_val.style.format(
                    {
                        "price": "{:,.0f}",
                        "rating": "{:.2f}",
                        "review_count": "{:.0f}",
                        "amenities_count": "{:.0f}",
                        "minimum_nights": "{:.0f}",
                        "value_score": "{:.3f}",
                    }
                ),
                use_container_width=True,
                height=430,
            )

# =================================================
# 4) Amenities & Quality
# =================================================
with tab_amenities:
    st.subheader("Amenities and perceived quality")

    df_am = data[data["amenities_count"].notna()].copy()

    if df_am.empty:
        st.info("No amenities_count data available in the filtered subset.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Distribution of amenities per listing**")
            fig = px.histogram(
                df_am,
                x="amenities_count",
                nbins=25,
                color="platform",
                template="plotly_dark",
            )
            fig.update_layout(
                xaxis_title="Amenities count",
                yaxis_title="Listings",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Price vs Amenities**")
            fig = px.scatter(
                df_am,
                x="amenities_count",
                y="price",
                color="platform",
                hover_data=["title", "city"],
                template="plotly_dark",
            )
            fig.update_layout(
                xaxis_title="Amenities count",
                yaxis_title="Price (USD / night)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**City-level amenities density (filtered cities)**")
        am_city = (
            df_am.groupby("city")
            .agg(
                listings=("city", "size"),
                avg_amenities=("amenities_count", "mean"),
                median_price=("price", "median"),
            )
            .reset_index()
        )
        am_city["avg_amenities"] = am_city["avg_amenities"].round(1)

        fig = px.scatter(
            am_city,
            x="avg_amenities",
            y="median_price",
            size="listings",
            hover_name="city",
            template="plotly_dark",
        )
        fig.update_layout(
            xaxis_title="Average amenities per listing",
            yaxis_title="Median price (USD)",
        )
        st.plotly_chart(fig, use_container_width=True)

# =================================================
# 5) Segments & Outliers
# =================================================
with tab_segments:
    st.subheader("Segments and pricing outliers")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Price segments (Budget → Luxury)**")
        seg_counts = (
            data["price_segment"].value_counts(dropna=False)
            .rename_axis("segment")
            .reset_index(name="listings")
        )
        fig = px.bar(
            seg_counts,
            x="segment",
            y="listings",
            template="plotly_dark",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Listings")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Price per bedroom by segment**")
        fig = px.box(
            data,
            x="price_segment",
            y="price_per_bedroom",
            color="platform",
            template="plotly_dark",
        )
        fig.update_layout(
            xaxis_title="Price segment",
            yaxis_title="Price per bedroom (USD)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Potential pricing outliers vs city medians**")
    city_median_price = data.groupby("city")["price"].median().to_dict()
    df_o = data.copy()
    df_o["city_median_price"] = df_o["city"].map(city_median_price)
    df_o["price_rel_city"] = df_o["price"] / df_o["city_median_price"]

    high_outliers = (
        df_o[df_o["price_rel_city"] > 2]
        .sort_values("price_rel_city", ascending=False)
        .head(15)
    )
    low_outliers = (
        df_o[df_o["price_rel_city"] < 0.5]
        .sort_values("price_rel_city", ascending=True)
        .head(15)
    )

    col_h, col_l = st.columns(2)
    with col_h:
        st.markdown("**Very expensive vs city median (possible luxury / overpricing)**")
        st.dataframe(
            high_outliers[
                [
                    "platform",
                    "city",
                    "title",
                    "price",
                    "city_median_price",
                    "price_rel_city",
                ]
            ],
            use_container_width=True,
        )
    with col_l:
        st.markdown("**Very cheap vs city median (hidden gems or errors)**")
        st.dataframe(
            low_outliers[
                [
                    "platform",
                    "city",
                    "title",
                    "price",
                    "city_median_price",
                    "price_rel_city",
                ]
            ],
            use_container_width=True,
        )

# =================================================
# 6) Best Listings (composite score)
# =================================================
with tab_best:
    st.subheader("Best overall listings (composite score)")

    df_best = data.copy().sort_values("composite_score", ascending=False)

    top_n = st.slider(
        "How many listings to show?", min_value=10, max_value=100, value=30, step=5
    )

    cols_show = [
        "platform",
        "city",
        "title",
        "price",
        "rating",
        "review_count",
        "amenities_count",
        "bedrooms",
        "bathrooms",
        "minimum_nights",
        "composite_score",
    ]
    cols_show = [c for c in cols_show if c in df_best.columns]

    st.dataframe(df_best[cols_show].head(top_n), use_container_width=True)

# =================================================
# 7) NLP search
# =================================================
with tab_search:
    st.subheader("NLP-based smart search")

    st.markdown(
        "Type what you are looking for (e.g. "
        "*\"sea view cabin with jacuzzi in Batroun\"* or "
        "*\"cheap studio in Beirut for one night\"*). "
        "Results are ranked using TF-IDF similarity on titles."
    )

    @st.cache_resource
    def build_vectorizer(texts: pd.Series):
        texts = texts.fillna("").astype(str).tolist()
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(texts)
        return vec, mat

    # Build on *filtered* subset so preferences apply
    vec, mat = build_vectorizer(data["title"])

    query = st.text_input("Describe your ideal stay:", "")
    max_results = st.slider("How many results to show", 5, 50, 15, step=5)

    if query.strip():
        q_vec = vec.transform([query])
        sims = cosine_similarity(q_vec, mat).ravel()
        search_df = data.copy()
        search_df["similarity"] = sims
        search_df = search_df.sort_values("similarity", ascending=False).head(
            max_results
        )

        cols_search = [
            "similarity",
            "platform",
            "city",
            "title",
            "price",
            "rating",
            "review_count",
            "amenities_count",
            "minimum_nights",
        ]
        cols_search = [c for c in cols_search if c in search_df.columns]

        st.markdown("**Top matches:**")
        st.dataframe(
            search_df[cols_search].style.format(
                {
                    "similarity": "{:.3f}",
                    "price": "{:,.0f}",
                    "rating": "{:.2f}",
                    "review_count": "{:.0f}",
                    "amenities_count": "{:.0f}",
                    "minimum_nights": "{:.0f}",
                }
            ),
            use_container_width=True,
            height=500,
        )
    else:
        st.info("Enter a query above to get ranked recommendations.")
