# ==========================================================
# BANKNOTE AUTHENTICATION DASHBOARD V2
# Premium Streamlit Dashboard
# Part 1 / 8
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Banknote Authentication Dashboard",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    paths = [
        "BankNote_Authentication.csv",
        "banknote_authentication.csv",
        "data.csv",
        "dataset.csv"
    ]

    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)

    return None


df = load_data()

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    models = {}

    model_files = {
        "Logistic Regression":"logistic_regression.pkl",
        "Decision Tree":"decision_tree.pkl",
        "Random Forest":"random_forest.pkl",
        "SVM":"svm.pkl",
        "KNN":"knn.pkl"
    }

    for name,file in model_files.items():

        if os.path.exists(file):

            try:
                models[name]=joblib.load(file)
            except:
                pass

    return models


models = load_models()

# ==========================================================
# PREMIUM CSS
# ==========================================================

st.markdown("""

<style>

html,
body,
[data-testid="stAppViewContainer"]{

background:#070b16;
color:white;
font-family:Inter,sans-serif;

}

/* Sidebar */

[data-testid="stSidebar"]{

background:linear-gradient(
180deg,
#0d1326,
#111827,
#0b1220);

border-right:1px solid rgba(255,255,255,0.08);

}

/* Hide Streamlit */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Hero */

.hero{

padding:45px;

border-radius:30px;

background:
linear-gradient(
135deg,
rgba(34,211,238,.18),
rgba(99,102,241,.15),
rgba(168,85,247,.18)
);

backdrop-filter:blur(18px);

border:1px solid rgba(255,255,255,.08);

margin-bottom:25px;

box-shadow:
0px 10px 45px rgba(0,0,0,.45);

}

/* Title */

.hero-title{

font-size:48px;

font-weight:800;

background:
linear-gradient(
90deg,
#67e8f9,
#60a5fa,
#c084fc);

-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

}

/* Subtitle */

.hero-sub{

font-size:18px;

color:#cbd5e1;

margin-top:15px;

line-height:1.7;

}

/* Cards */

.glass{

background:
rgba(255,255,255,.05);

border-radius:22px;

padding:22px;

backdrop-filter:blur(18px);

border:1px solid rgba(255,255,255,.08);

transition:.35s;

box-shadow:
0px 8px 28px rgba(0,0,0,.25);

}

.glass:hover{

transform:translateY(-6px);

border:1px solid #60a5fa;

}

/* KPI */

.kpi{

text-align:center;

padding:22px;

border-radius:24px;

background:

linear-gradient(
145deg,
rgba(255,255,255,.06),
rgba(255,255,255,.03));

border:1px solid rgba(255,255,255,.07);

transition:.3s;

}

.kpi:hover{

transform:scale(1.04);

box-shadow:
0px 0px 30px rgba(96,165,250,.35);

}

.kpi-number{

font-size:40px;

font-weight:800;

color:#67e8f9;

}

.kpi-text{

font-size:15px;

color:#cbd5e1;

}

.section{

font-size:28px;

font-weight:700;

margin-top:15px;

margin-bottom:15px;

}

/* Buttons */

.stButton>button{

background:linear-gradient(
90deg,
#2563eb,
#7c3aed);

color:white;

border:none;

border-radius:12px;

padding:0.6rem 1.3rem;

font-weight:700;

}

.stButton>button:hover{

background:linear-gradient(
90deg,
#1d4ed8,
#9333ea);

}

/* Metric */

[data-testid="metric-container"]{

background:rgba(255,255,255,.05);

border-radius:20px;

padding:20px;

border:1px solid rgba(255,255,255,.08);

}

/* Scrollbar */

::-webkit-scrollbar{

width:10px;

}

::-webkit-scrollbar-thumb{

background:#4f46e5;

border-radius:20px;

}

</style>

""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("# 💵 Banknote Dashboard")

st.sidebar.markdown("---")

page = st.sidebar.radio(

"Navigation",

[
    "Dashboard",
    "Prediction",
    "Model Comparison",
    "About"
]

)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Premium AI Dashboard

Version 2.0
"""
)

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""

<div class="hero">

<div class="hero-title">
Banknote Authentication Dashboard
</div>

<div class="hero-sub">

Interactive Machine Learning Dashboard
for detecting genuine and forged banknotes.

Explore the dataset, visualize statistics,
compare machine learning models,
and perform real-time predictions.

</div>

</div>

""", unsafe_allow_html=True)

# ==========================================================
# DATA CHECK
# ==========================================================

if df is None:

    st.error("Dataset not found.")

    st.stop()

# ==========================================================
# BASIC DATA
# ==========================================================

rows = df.shape[0]
cols = df.shape[1]

features = list(df.columns[:-1])

target = df.columns[-1]

fake_count = int((df[target] == 1).sum())
real_count = int((df[target] == 0).sum())

# ==========================================================
# DASHBOARD PAGE START
# ==========================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section">📊 Dashboard Overview</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-number">{rows}</div>
                <div class="kpi-text">Total Samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-number">{cols}</div>
                <div class="kpi-text">Columns</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-number">{real_count}</div>
                <div class="kpi-text">Genuine Notes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-number">{fake_count}</div>
                <div class="kpi-text">Forged Notes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
            # ==========================================================
    # DATASET PREVIEW
    # ==========================================================

    st.markdown(
        '<div class="section">📁 Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([2, 1])

    with left:

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        show_rows = st.slider(
            "Rows to Display",
            min_value=5,
            max_value=min(100, len(df)),
            value=10
        )

        st.dataframe(
            df.head(show_rows),
            use_container_width=True,
            height=380
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.subheader("Dataset Information")

        st.write(f"**Rows:** {rows}")
        st.write(f"**Columns:** {cols}")
        st.write(f"**Features:** {len(features)}")
        st.write(f"**Target:** {target}")

        memory = df.memory_usage(deep=True).sum() / 1024

        st.write(f"**Memory Usage:** {memory:.2f} KB")

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # SUMMARY STATISTICS
    # ==========================================================

    st.markdown(
        '<div class="section">📈 Statistical Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    summary = df.describe().T

    summary["Range"] = summary["max"] - summary["min"]

    st.dataframe(
        summary.style.format("{:.3f}"),
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # MISSING VALUES
    # ==========================================================

    st.markdown(
        '<div class="section">🩺 Data Quality Check</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        missing = df.isnull().sum()

        fig = px.bar(
            x=missing.index,
            y=missing.values,
            labels={
                "x": "Column",
                "y": "Missing Values"
            },
            title="Missing Values"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        duplicate = df.duplicated().sum()

        quality = pd.DataFrame(
            {
                "Metric": [
                    "Missing Values",
                    "Duplicate Rows"
                ],
                "Count": [
                    int(missing.sum()),
                    int(duplicate)
                ]
            }
        )

        fig = px.bar(
            quality,
            x="Metric",
            y="Count",
            color="Metric",
            text="Count"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==========================================================
    # CLASS DISTRIBUTION
    # ==========================================================

    st.markdown(
        '<div class="section">💵 Genuine vs Forged Notes</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        class_df = pd.DataFrame(
            {
                "Type": ["Genuine", "Forged"],
                "Count": [real_count, fake_count]
            }
        )

        pie = px.pie(
            class_df,
            names="Type",
            values="Count",
            hole=0.6
        )

        pie.update_layout(
            template="plotly_dark",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    with col2:

        bar = px.bar(
            class_df,
            x="Type",
            y="Count",
            color="Type",
            text="Count"
        )

        bar.update_layout(
            template="plotly_dark",
            height=450,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            bar,
            use_container_width=True
        )

    # ==========================================================
    # FEATURE EXPLORER
    # ==========================================================

    st.markdown(
        '<div class="section">🔍 Feature Explorer</div>',
        unsafe_allow_html=True
    )

    selected_feature = st.selectbox(
        "Select Feature",
        features
    )

    c1, c2 = st.columns(2)

    with c1:

        histogram = px.histogram(
            df,
            x=selected_feature,
            nbins=40,
            color=target,
            marginal="rug"
        )

        histogram.update_layout(
            template="plotly_dark",
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            histogram,
            use_container_width=True
        )

    with c2:

        box = px.box(
            df,
            y=selected_feature,
            color=df[target].astype(str),
            points="all"
        )

        box.update_layout(
            template="plotly_dark",
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            box,
            use_container_width=True
        )

    # ==========================================================
    # END OF PART 2
    # ==========================================================
        # ==========================================================
    # CORRELATION ANALYSIS
    # ==========================================================

    st.markdown(
        '<div class="section">🔥 Correlation Analysis</div>',
        unsafe_allow_html=True
    )

    corr = df.corr(numeric_only=True)

    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Viridis",
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hoverongaps=False
        )
    )

    heatmap.update_layout(
        template="plotly_dark",
        height=650,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=40, b=30)
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    # ==========================================================
    # CORRELATION TABLE
    # ==========================================================

    st.markdown(
        '<div class="section">📊 Correlation Matrix</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.dataframe(
        corr.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # FEATURE RELATIONSHIP
    # ==========================================================

    st.markdown(
        '<div class="section">📈 Feature Relationship</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        x_feature = st.selectbox(
            "X Axis",
            features,
            index=0
        )

    with col2:

        y_feature = st.selectbox(
            "Y Axis",
            features,
            index=1
        )

    scatter = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color=target,
        size_max=12,
        opacity=0.75,
        hover_data=df.columns
    )

    scatter.update_layout(
        template="plotly_dark",
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        scatter,
        use_container_width=True
    )

    # ==========================================================
    # FEATURE DISTRIBUTION COMPARISON
    # ==========================================================

    st.markdown(
        '<div class="section">📦 Distribution Comparison</div>',
        unsafe_allow_html=True
    )

    feature_compare = st.selectbox(
        "Choose Feature",
        features,
        key="distribution_compare"
    )

    violin = px.violin(
        df,
        y=feature_compare,
        color=target,
        box=True,
        points="all"
    )

    violin.update_layout(
        template="plotly_dark",
        height=550,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        violin,
        use_container_width=True
    )

    # ==========================================================
    # KDE STYLE DISTRIBUTION
    # ==========================================================

    st.markdown(
        '<div class="section">📉 Density Comparison</div>',
        unsafe_allow_html=True
    )

    genuine = df[df[target] == 0][feature_compare]
    forged = df[df[target] == 1][feature_compare]

    density = ff.create_distplot(
        [genuine, forged],
        ["Genuine", "Forged"],
        show_hist=False,
        show_rug=False
    )

    density.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        density,
        use_container_width=True
    )

    # ==========================================================
    # PAIRWISE FEATURE SCATTER
    # ==========================================================

    st.markdown(
        '<div class="section">🧩 Pairwise Feature Analysis</div>',
        unsafe_allow_html=True
    )

    pair_features = st.multiselect(
        "Choose up to 4 Features",
        features,
        default=features[:4]
    )

    if len(pair_features) >= 2:

        pair = px.scatter_matrix(
            df,
            dimensions=pair_features,
            color=target,
            opacity=0.7
        )

        pair.update_layout(
            template="plotly_dark",
            height=800,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            pair,
            use_container_width=True
        )

    else:

        st.info("Select at least two features.")

    # ==========================================================
    # FEATURE IMPORTANCE PREVIEW
    # ==========================================================

    st.markdown(
        '<div class="section">⭐ Feature Statistics</div>',
        unsafe_allow_html=True
    )

    stats = pd.DataFrame({
        "Feature": features,
        "Mean": [df[f].mean() for f in features],
        "Median": [df[f].median() for f in features],
        "Std Dev": [df[f].std() for f in features],
        "Variance": [df[f].var() for f in features],
        "Minimum": [df[f].min() for f in features],
        "Maximum": [df[f].max() for f in features]
    })

    st.dataframe(
        stats.style.format("{:.3f}"),
        use_container_width=True
    )

    # ==========================================================
    # FEATURE RANGE VISUALIZATION
    # ==========================================================

    st.markdown(
        '<div class="section">📏 Feature Range</div>',
        unsafe_allow_html=True
    )

    range_df = pd.DataFrame({
        "Feature": features,
        "Range": [
            df[f].max() - df[f].min()
            for f in features
        ]
    })

    range_chart = px.bar(
        range_df,
        x="Feature",
        y="Range",
        color="Range",
        text_auto=".2f"
    )

    range_chart.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        range_chart,
        use_container_width=True
    )

    # ==========================================================
    # END OF PART 3
    # ==========================================================
        # ==========================================================
    # ADVANCED FEATURE ANALYTICS
    # ==========================================================

    st.markdown(
        '<div class="section">📌 Advanced Feature Analytics</div>',
        unsafe_allow_html=True
    )

    feature = st.selectbox(
        "Select Feature for Detailed Analysis",
        features,
        key="advanced_feature"
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = go.Figure()

        fig.add_trace(
            go.Box(
                y=df[feature],
                name=feature,
                boxmean=True,
                marker_color="#60A5FA"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            title=f"{feature} Outlier Analysis",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = df[
            (df[feature] < lower) |
            (df[feature] > upper)
        ]

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.metric("Mean", f"{df[feature].mean():.3f}")
        st.metric("Median", f"{df[feature].median():.3f}")
        st.metric("Std Deviation", f"{df[feature].std():.3f}")
        st.metric("Minimum", f"{df[feature].min():.3f}")
        st.metric("Maximum", f"{df[feature].max():.3f}")
        st.metric("Outliers", len(outliers))

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # FEATURE COMPARISON
    # ==========================================================

    st.markdown(
        '<div class="section">⚡ Compare Two Features</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        feature1 = st.selectbox(
            "Feature 1",
            features,
            key="feature_compare_1"
        )

    with c2:
        feature2 = st.selectbox(
            "Feature 2",
            features,
            index=1,
            key="feature_compare_2"
        )

    comparison = go.Figure()

    comparison.add_trace(
        go.Scatter(
            x=df[feature1],
            y=df[feature2],
            mode="markers",
            marker=dict(
                size=8,
                color=df[target],
                colorscale="Turbo",
                opacity=0.75
            )
        )
    )

    comparison.update_layout(
        title=f"{feature1} vs {feature2}",
        template="plotly_dark",
        height=550,
        xaxis_title=feature1,
        yaxis_title=feature2,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        comparison,
        use_container_width=True
    )

    # ==========================================================
    # DATASET INSIGHTS
    # ==========================================================

    st.markdown(
        '<div class="section">🧠 Dataset Insights</div>',
        unsafe_allow_html=True
    )

    insight1, insight2 = st.columns(2)

    with insight1:

        strongest = corr[target].drop(target).abs().sort_values(
            ascending=False
        )

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.subheader("Top Correlated Features")

        for feat, value in strongest.items():

            st.progress(float(value))

            st.write(
                f"**{feat}** : {value:.3f}"
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with insight2:

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.subheader("Quick Statistics")

        st.write(f"Dataset Shape : **{df.shape}**")

        st.write(f"Duplicate Rows : **{df.duplicated().sum()}**")

        st.write(f"Missing Values : **{df.isnull().sum().sum()}**")

        st.write(
            f"Average Feature Mean : **{df[features].mean().mean():.3f}**"
        )

        st.write(
            f"Average Feature Std : **{df[features].std().mean():.3f}**"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # DOWNLOAD DATASET
    # ==========================================================

    st.markdown(
        '<div class="section">⬇ Download Dataset</div>',
        unsafe_allow_html=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        data=csv,
        file_name="banknote_dataset.csv",
        mime="text/csv"
    )

    # ==========================================================
    # DASHBOARD SUMMARY
    # ==========================================================

    st.markdown(
        '<div class="section">📋 Dashboard Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="glass">

### ✅ What this dashboard provides

- Premium interactive analytics
- Dataset exploration
- Feature distributions
- Correlation analysis
- Scatter matrix visualization
- Outlier detection
- Statistical summaries
- Interactive Plotly charts
- Downloadable dataset

The following pages extend the dashboard further with:

- 🤖 Machine Learning Model Comparison
- 🎯 Live Prediction Interface
- ℹ About the Project

</div>
""",
        unsafe_allow_html=True
    )

# ==========================================================
# END OF DASHBOARD PAGE
# ==========================================================

elif page == "Model Comparison":

    st.markdown(
        """
<div class="hero">

<div class="hero-title">
🤖 Model Comparison
</div>

<div class="hero-sub">

Compare the performance of every trained
Machine Learning model used for
banknote authentication.

</div>

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section">🏆 Model Performance</div>',
        unsafe_allow_html=True
    )

    if len(models) == 0:

        st.warning(
            "No trained model (.pkl) files were found."
        )

    else:

        st.info(
            "Performance comparison begins below..."
        )

        # Part 5 starts from here.
            # ==========================================================
    # PREPARE TEST DATA
    # ==========================================================

    from sklearn.model_selection import train_test_split

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ==========================================================
    # CHECK MODELS
    # ==========================================================

    if len(models) == 0:

        st.warning("No trained model (.pkl) files found.")

    else:

        st.markdown(
            '<div class="section">🏆 Model Leaderboard</div>',
            unsafe_allow_html=True
        )

        metrics = []

        predictions = {}

        probabilities = {}

        # ==========================================================
        # EVALUATE MODELS
        # ==========================================================

        for name, model in models.items():

            pred = model.predict(X_test)

            predictions[name] = pred

            if hasattr(model, "predict_proba"):

                try:
                    probabilities[name] = model.predict_proba(X_test)[:, 1]
                except:
                    probabilities[name] = None

            else:

                probabilities[name] = None

            metrics.append({

                "Model": name,

                "Accuracy":
                accuracy_score(
                    y_test,
                    pred
                ),

                "Precision":
                precision_score(
                    y_test,
                    pred,
                    zero_division=0
                ),

                "Recall":
                recall_score(
                    y_test,
                    pred,
                    zero_division=0
                ),

                "F1 Score":
                f1_score(
                    y_test,
                    pred,
                    zero_division=0
                )

            })

        metrics_df = pd.DataFrame(metrics)

        metrics_df = metrics_df.sort_values(
            "Accuracy",
            ascending=False
        )

        # ==========================================================
        # BEST MODEL CARD
        # ==========================================================

        winner = metrics_df.iloc[0]

        st.success(
            f"🥇 Best Model: {winner['Model']} "
            f"({winner['Accuracy']:.2%} Accuracy)"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Best Accuracy",
                f"{winner['Accuracy']:.2%}"
            )

        with c2:

            st.metric(
                "Precision",
                f"{winner['Precision']:.2%}"
            )

        with c3:

            st.metric(
                "Recall",
                f"{winner['Recall']:.2%}"
            )

        with c4:

            st.metric(
                "F1 Score",
                f"{winner['F1 Score']:.2%}"
            )

        # ==========================================================
        # LEADERBOARD TABLE
        # ==========================================================

        st.markdown(
            '<div class="section">📊 Performance Table</div>',
            unsafe_allow_html=True
        )

        styled = metrics_df.style.format({

            "Accuracy": "{:.3f}",

            "Precision": "{:.3f}",

            "Recall": "{:.3f}",

            "F1 Score": "{:.3f}"

        })

        st.dataframe(
            styled,
            use_container_width=True
        )

        # ==========================================================
        # ACCURACY BAR CHART
        # ==========================================================

        fig = px.bar(

            metrics_df,

            x="Model",

            y="Accuracy",

            color="Accuracy",

            text="Accuracy"

        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig.update_layout(

            template="plotly_dark",

            height=550,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==========================================================
        # GROUPED METRIC CHART
        # ==========================================================

        melt = metrics_df.melt(

            id_vars="Model",

            value_vars=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],

            var_name="Metric",

            value_name="Score"

        )

        fig = px.bar(

            melt,

            x="Model",

            y="Score",

            color="Metric",

            barmode="group"

        )

        fig.update_layout(

            template="plotly_dark",

            height=600,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==========================================================
        # RADAR CHART
        # ==========================================================

        st.markdown(
            '<div class="section">🕸 Radar Comparison</div>',
            unsafe_allow_html=True
        )

        radar = go.Figure()

        categories = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]

        for _, row in metrics_df.iterrows():

            radar.add_trace(

                go.Scatterpolar(

                    r=[
                        row["Accuracy"],
                        row["Precision"],
                        row["Recall"],
                        row["F1 Score"]
                    ],

                    theta=categories,

                    fill="toself",

                    name=row["Model"]

                )

            )

        radar.update_layout(

            template="plotly_dark",

            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),

            height=650,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)"

        )

        st.plotly_chart(
            radar,
            use_container_width=True
        )

        # ==========================================================
        # PART 6 STARTS HERE
        # Confusion Matrix + ROC + Prediction Page
        # ==========================================================
                # ==========================================================
        # CONFUSION MATRIX
        # ==========================================================

        st.markdown(
            '<div class="section">🔥 Confusion Matrix</div>',
            unsafe_allow_html=True
        )

        selected_model = st.selectbox(
            "Select Model",
            list(models.keys()),
            key="cm_model"
        )

        cm = confusion_matrix(
            y_test,
            predictions[selected_model]
        )

        cm_fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(
                x="Predicted",
                y="Actual",
                color="Count"
            )
        )

        cm_fig.update_layout(
            template="plotly_dark",
            height=500,
            title=f"{selected_model} Confusion Matrix",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            cm_fig,
            use_container_width=True
        )

        # ==========================================================
        # ROC CURVE
        # ==========================================================

        from sklearn.metrics import roc_curve, auc

        st.markdown(
            '<div class="section">📈 ROC Curve Comparison</div>',
            unsafe_allow_html=True
        )

        roc_fig = go.Figure()

        for model_name in models.keys():

            prob = probabilities.get(model_name)

            if prob is None:
                continue

            fpr, tpr, _ = roc_curve(
                y_test,
                prob
            )

            roc_auc = auc(fpr, tpr)

            roc_fig.add_trace(

                go.Scatter(

                    x=fpr,

                    y=tpr,

                    mode="lines",

                    name=f"{model_name} (AUC={roc_auc:.3f})"

                )

            )

        roc_fig.add_trace(

            go.Scatter(

                x=[0,1],

                y=[0,1],

                mode="lines",

                line=dict(dash="dash"),

                showlegend=False

            )

        )

        roc_fig.update_layout(

            template="plotly_dark",

            height=600,

            title="ROC Curve",

            xaxis_title="False Positive Rate",

            yaxis_title="True Positive Rate",

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)"

        )

        st.plotly_chart(
            roc_fig,
            use_container_width=True
        )

# ==========================================================
# END MODEL COMPARISON
# ==========================================================

elif page == "Prediction":

    st.markdown("""

<div class="hero">

<div class="hero-title">
🎯 Live Prediction
</div>

<div class="hero-sub">

Enter the banknote features below and
predict whether the note is Genuine
or Forged using any trained model.

</div>

</div>

""", unsafe_allow_html=True)

    if len(models) == 0:

        st.warning(
            "No trained models available."
        )

    else:

        st.markdown(
            '<div class="section">⚙ Prediction Settings</div>',
            unsafe_allow_html=True
        )

        model_name = st.selectbox(
            "Choose Model",
            list(models.keys())
        )

        model = models[model_name]

        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:

            variance = st.number_input(
                "Variance",
                value=0.0,
                format="%.4f"
            )

            skewness = st.number_input(
                "Skewness",
                value=0.0,
                format="%.4f"
            )

        with c2:

            curtosis = st.number_input(
                "Curtosis",
                value=0.0,
                format="%.4f"
            )

            entropy = st.number_input(
                "Entropy",
                value=0.0,
                format="%.4f"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        input_df = pd.DataFrame([[
            variance,
            skewness,
            curtosis,
            entropy
        ]], columns=features)

        if st.button("🚀 Predict"):

            prediction = model.predict(input_df)[0]

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(
                    input_df
                )[0]

            else:

                probability = None

            st.markdown(
                '<div class="section">📋 Prediction Result</div>',
                unsafe_allow_html=True
            )

            if prediction == 0:

                st.success(
                    "✅ Genuine Banknote"
                )

            else:

                st.error(
                    "❌ Forged Banknote"
                )

            if probability is not None:

                prob_df = pd.DataFrame({

                    "Class":[
                        "Genuine",
                        "Forged"
                    ],

                    "Probability":probability

                })

                fig = px.bar(

                    prob_df,

                    x="Class",

                    y="Probability",

                    color="Probability",

                    text="Probability"

                )

                fig.update_traces(
                    texttemplate="%{text:.2f}"
                )

                fig.update_layout(

                    template="plotly_dark",

                    height=450,

                    paper_bgcolor="rgba(0,0,0,0)",

                    plot_bgcolor="rgba(0,0,0,0)",

                    showlegend=False

                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

# ==========================================================
# PART 7 STARTS HERE
# About Page
# ==========================================================
# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "About":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            ℹ️ About This Dashboard
        </div>

        <div class="hero-sub">
            A premium interactive Machine Learning dashboard for
            Banknote Authentication using multiple classification
            algorithms and rich data visualizations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================
    # PROJECT OVERVIEW
    # ==========================================================

    st.markdown(
        '<div class="section">📖 Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="glass">

    ### 💵 Banknote Authentication

    Counterfeit currency detection is an important application of
    Machine Learning.

    This dashboard enables users to:

    ✅ Explore the complete dataset

    ✅ Visualize feature distributions

    ✅ Understand feature correlations

    ✅ Compare multiple ML models

    ✅ Predict whether a banknote is Genuine or Forged

    All visualizations are interactive and built using Plotly.

    </div>
    """, unsafe_allow_html=True)

    # ==========================================================
    # DATASET INFORMATION
    # ==========================================================

    st.markdown(
        '<div class="section">📂 Dataset Information</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.subheader("Dataset Summary")

        st.write(f"Rows : **{rows}**")
        st.write(f"Columns : **{cols}**")
        st.write(f"Features : **{len(features)}**")
        st.write(f"Target Column : **{target}**")

        st.write(f"Genuine Notes : **{real_count}**")
        st.write(f"Forged Notes : **{fake_count}**")

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.subheader("Input Features")

        for f in features:

            st.write("✔", f)

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # MACHINE LEARNING PIPELINE
    # ==========================================================

    st.markdown(
        '<div class="section">⚙️ Machine Learning Workflow</div>',
        unsafe_allow_html=True
    )

    pipeline = pd.DataFrame({

        "Step":[

            "Load Dataset",

            "Data Analysis",

            "Visualization",

            "Train ML Models",

            "Evaluate Models",

            "Prediction"

        ],

        "Description":[

            "Read banknote dataset",

            "Explore statistics",

            "Generate Plotly charts",

            "Fit ML algorithms",

            "Compare performance",

            "Predict note authenticity"

        ]

    })

    st.dataframe(
        pipeline,
        use_container_width=True
    )

    # ==========================================================
    # TECHNOLOGY STACK
    # ==========================================================

    st.markdown(
        '<div class="section">🛠 Technology Stack</div>',
        unsafe_allow_html=True
    )

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.markdown("""
<div class="glass">

### 🎨 Frontend

- Streamlit
- HTML
- CSS
- Glassmorphism
- Responsive Layout

</div>
""", unsafe_allow_html=True)

    with tech2:

        st.markdown("""
<div class="glass">

### 📊 Visualization

- Plotly Express
- Plotly Graph Objects
- Interactive Charts
- Heatmaps
- ROC Curves

</div>
""", unsafe_allow_html=True)

    with tech3:

        st.markdown("""
<div class="glass">

### 🤖 Machine Learning

- Scikit-learn
- Logistic Regression
- Random Forest
- SVM
- Decision Tree
- KNN

</div>
""", unsafe_allow_html=True)

    # ==========================================================
    # FEATURES
    # ==========================================================

    st.markdown(
        '<div class="section">⭐ Dashboard Features</div>',
        unsafe_allow_html=True
    )

    feature_list = [

        "Premium Dark Theme",

        "Glassmorphism UI",

        "Gradient Hero Section",

        "Interactive KPI Cards",

        "Dataset Explorer",

        "Correlation Heatmap",

        "Histogram Selector",

        "Scatter Matrix",

        "Model Comparison",

        "ROC Curve",

        "Confusion Matrix",

        "Live Prediction",

        "Probability Visualization",

        "Responsive Layout"

    ]

    feature_df = pd.DataFrame({

        "Available Features":feature_list

    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )

    # ==========================================================
    # PROJECT STATISTICS
    # ==========================================================

    st.markdown(
        '<div class="section">📈 Project Statistics</div>',
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)

    with a:
        st.metric("ML Models", len(models))

    with b:
        st.metric("Dataset Rows", rows)

    with c:
        st.metric("Features", len(features))

    with d:
        st.metric("Visualizations", "12+")

    # ==========================================================
    # DEVELOPER
    # ==========================================================

    st.markdown(
        '<div class="section">👨‍💻 Developer</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="glass">

## Banknote Authentication Dashboard V2

Designed as a premium Machine Learning dashboard
demonstrating interactive analytics,
classification models,
and real-time predictions.

Built using

- Streamlit
- Plotly
- Scikit-learn
- Pandas
- NumPy

Designed with

- Premium Dark Theme
- Glassmorphism
- Responsive Layout
- Interactive Components

</div>
""", unsafe_allow_html=True)

# ==========================================================
# PART 8 STARTS HERE
# Footer + Final Polish
# ==========================================================
# ==========================================================
# GLOBAL FOOTER
# ==========================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<style>

.footer-card{

margin-top:60px;

padding:28px;

border-radius:24px;

background:linear-gradient(
135deg,
rgba(59,130,246,0.15),
rgba(139,92,246,0.15),
rgba(34,211,238,0.12)
);

border:1px solid rgba(255,255,255,0.08);

backdrop-filter:blur(18px);

text-align:center;

box-shadow:0 8px 30px rgba(0,0,0,.35);

}

.footer-title{

font-size:26px;

font-weight:700;

background:linear-gradient(
90deg,
#67e8f9,
#60a5fa,
#a78bfa
);

-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

}

.footer-text{

color:#CBD5E1;

font-size:15px;

line-height:1.8;

margin-top:12px;

}

.badge{

display:inline-block;

padding:8px 16px;

margin:6px;

border-radius:999px;

background:rgba(255,255,255,.07);

border:1px solid rgba(255,255,255,.08);

font-size:13px;

}

hr{

border:none;

height:1px;

background:rgba(255,255,255,.08);

margin-top:35px;

margin-bottom:35px;

}

</style>
""", unsafe_allow_html=True)

st.markdown("""

<div class="footer-card">

<div class="footer-title">

💵 Banknote Authentication Dashboard V2

</div>

<div class="footer-text">

Premium interactive dashboard for counterfeit
banknote detection using Machine Learning.

Designed with modern UI principles,
glassmorphism,
interactive Plotly visualizations,
and real-time prediction support.

</div>

<br>

<span class="badge">Streamlit</span>

<span class="badge">Plotly</span>

<span class="badge">Scikit-Learn</span>

<span class="badge">Pandas</span>

<span class="badge">NumPy</span>

<span class="badge">Machine Learning</span>

<span class="badge">Interactive Dashboard</span>

<hr>

<div style="color:#94A3B8;font-size:14px;">

Made with ❤️ using Python & Streamlit

</div>

</div>

""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR FOOTNOTE
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.caption(
"""
Banknote Dashboard V2

Premium Edition

© 2026
"""
)

# ==========================================================
# SUCCESS MESSAGE
# ==========================================================

st.sidebar.success(
"Dashboard Loaded Successfully"
)

# ==========================================================
# END OF APPLICATION
# ==========================================================