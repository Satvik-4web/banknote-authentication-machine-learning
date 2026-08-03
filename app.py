import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Banknote Authentication",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = pd.read_csv("dataset/dataset.txt", header=None)

data.columns = [
    "Variance",
    "Skewness",
    "Curtosis",
    "Entropy",
    "Class"
]

model = joblib.load("models/currency_model.pkl")

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

html, body, [class*="css"]{
    font-family:'Segoe UI';
}

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

.main{
    background:#0f172a;
}

.metric-card{
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:20px;
    padding:20px;
    box-shadow:0px 0px 20px rgba(0,0,0,0.3);
}

.big-title{
    font-size:52px;
    font-weight:800;
}

.subtitle{
    font-size:20px;
    color:#94a3b8;
}

.green{
    color:#22c55e;
}

hr{
    border:none;
    height:1px;
    background:#334155;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("💵 Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠 Dashboard",
        "📊 Dataset Analysis",
        "🤖 Model Comparison",
        "💰 Predict Banknote",
        "ℹ️ About"
    ]
)
# ==========================================================
# DASHBOARD
# ==========================================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="big-title">
        💵 Banknote Authentication Dashboard
    </div>
    <div class="subtitle">
        Machine Learning powered counterfeit banknote detection using multiple classification algorithms.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    # -------------------------
    # TOP METRICS
    # -------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📄 Samples", len(data))
    c2.metric("📊 Features", 4)
    c3.metric("🤖 Models", 5)
    c4.metric("🏆 Best Accuracy", "100%")

    st.write("")
    st.divider()

    # -------------------------
    # SECOND ROW
    # -------------------------

    left, right = st.columns([2,1])

    with left:

        st.subheader("📖 Project Overview")

        st.write("""
This project uses Machine Learning algorithms to classify banknotes as **Genuine** or **Counterfeit**.

The workflow includes:

- Exploratory Data Analysis
- Data Visualization
- Feature Engineering
- Training Multiple Models
- Model Comparison
- Saving the Best Model
        """)

        st.write("")

        st.subheader("⚙ Workflow")

        st.code("""
Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Train-Test Split
      │
      ▼
Train Multiple Models
      │
      ▼
Compare Accuracy
      │
      ▼
Save Best Model
        """)

    with right:

        st.subheader("🎯 Model Accuracy")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=100,
            number={"suffix":"%"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":"limegreen"},
                "steps":[
                    {"range":[0,60],"color":"#3f3f46"},
                    {"range":[60,80],"color":"#52525b"},
                    {"range":[80,100],"color":"#16a34a"}
                ]
            }
        ))

        fig.update_layout(
            height=320,
            margin=dict(l=20,r=20,t=20,b=20),
            paper_bgcolor="#0f172a",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------------
    # QUICK DATASET PREVIEW
    # -------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True,
        height=350
    )
    # ==========================================================
# DATASET ANALYSIS
# ==========================================================

elif page == "📊 Dataset Analysis":

    st.title("📊 Dataset Analysis")
    st.write("Explore the dataset used for training the machine learning models.")

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", len(data))
    c2.metric("Columns", len(data.columns))
    c3.metric("Missing Values", int(data.isnull().sum().sum()))

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Class Distribution")

        class_count = data["Class"].value_counts().reset_index()
        class_count.columns = ["Class","Count"]

        class_count["Class"] = class_count["Class"].replace({
            0:"Genuine",
            1:"Counterfeit"
        })

        fig = px.bar(
            class_count,
            x="Class",
            y="Count",
            color="Class",
            text="Count",
            height=420,
            template="plotly_dark"
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(fig,use_container_width=True)

    with right:

        st.subheader("Correlation Heatmap")

        corr = data.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )

        fig.update_layout(
            height=420,
            template="plotly_dark"
        )

        st.plotly_chart(fig,use_container_width=True)

    st.divider()

    st.subheader("Feature Distributions")

    feature = st.selectbox(
        "Select Feature",
        [
            "Variance",
            "Skewness",
            "Curtosis",
            "Entropy"
        ]
    )

    fig = px.histogram(
        data,
        x=feature,
        nbins=40,
        color="Class",
        marginal="box",
        template="plotly_dark"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        data,
        use_container_width=True,
        height=350
    )

    st.divider()

    st.subheader("Statistical Summary")

    st.dataframe(
        data.describe(),
        use_container_width=True
    )
    # ==========================================================
# MODEL COMPARISON
# ==========================================================

elif page == "🤖 Model Comparison":

    st.title("🤖 Machine Learning Model Comparison")
    st.write("Comparison of all classification algorithms used in this project.")

    st.divider()

    comparison = pd.DataFrame({
        "Model":[
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "K-Nearest Neighbors",
            "Support Vector Machine"
        ],
        "Accuracy":[
            98.55,
            98.18,
            98.91,
            100.00,
            100.00
        ]
    })

    left, right = st.columns([2,1])

    with left:

        st.subheader("🏆 Accuracy Comparison")

        fig = px.bar(
            comparison,
            x="Model",
            y="Accuracy",
            color="Accuracy",
            text="Accuracy",
            template="plotly_dark",
            height=450
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_range=[95,101],
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("🥇 Best Model")

        st.success("""
### K-Nearest Neighbors

Accuracy

**100.00%**
""")

        st.success("""
### Support Vector Machine

Accuracy

**100.00%**
""")

        st.info("""
Compared Models

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
""")

    st.divider()

    st.subheader("📋 Model Performance Table")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📈 Model Ranking")

    ranking = comparison.sort_values(
        by="Accuracy",
        ascending=True
    )

    fig = px.bar(
        ranking,
        x="Accuracy",
        y="Model",
        orientation="h",
        color="Accuracy",
        text="Accuracy",
        template="plotly_dark",
        height=400
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    # ==========================================================
# PREDICT BANKNOTE
# ==========================================================

elif page == "💰 Predict Banknote":

    st.title("💰 Banknote Prediction")

    st.write(
        "Adjust the feature values below and let the trained model predict whether the banknote is Genuine or Counterfeit."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        variance = st.slider(
            "Variance",
            float(data["Variance"].min()),
            float(data["Variance"].max()),
            0.0
        )

        skewness = st.slider(
            "Skewness",
            float(data["Skewness"].min()),
            float(data["Skewness"].max()),
            0.0
        )

    with col2:

        curtosis = st.slider(
            "Curtosis",
            float(data["Curtosis"].min()),
            float(data["Curtosis"].max()),
            0.0
        )

        entropy = st.slider(
            "Entropy",
            float(data["Entropy"].min()),
            float(data["Entropy"].max()),
            0.0
        )

    st.write("")

    if st.button(
        "🔍 Predict Banknote",
        use_container_width=True
    ):

        sample = np.array([[
            variance,
            skewness,
            curtosis,
            entropy
        ]])

        prediction = model.predict(sample)[0]

        st.divider()

        if prediction == 0:

            st.success("## ✅ Genuine Banknote")

        else:

            st.error("## ❌ Counterfeit Banknote")

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(sample)[0]

            confidence = np.max(probability) * 100

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

        st.subheader("Input Values")

        preview = pd.DataFrame({

            "Feature":[
                "Variance",
                "Skewness",
                "Curtosis",
                "Entropy"
            ],

            "Value":[
                variance,
                skewness,
                curtosis,
                entropy
            ]

        })

        st.table(preview)

    st.divider()

    st.info(
        "These values represent statistical image features extracted from banknotes, not manually measurable properties."
    )
    # ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.divider()

    left, right = st.columns([2,1])

    with left:

        st.subheader("💵 Banknote Authentication")

        st.write("""
This project classifies banknotes as **Genuine** or **Counterfeit**
using Machine Learning.

Instead of training only one model,
multiple classification algorithms were trained,
evaluated and compared.

The best performing model was then saved
and used for making future predictions.
        """)

        st.subheader("📊 Dataset")

        st.write("""
- **Dataset:** Banknote Authentication Dataset
- **Samples:** 1372
- **Features:** 4
- **Classes:** Genuine / Counterfeit
        """)

        st.subheader("🤖 Machine Learning Models")

        st.write("""
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
        """)

    with right:

        st.metric(
            "Dataset Size",
            "1372"
        )

        st.metric(
            "Features",
            "4"
        )

        st.metric(
            "Models",
            "5"
        )

        st.metric(
            "Best Accuracy",
            "100%"
        )

    st.divider()

    st.subheader("🛠️ Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.success("""
Python

Pandas

NumPy
""")

    with tech2:

        st.success("""
Scikit-Learn

Plotly

Joblib
""")

    with tech3:

        st.success("""
Streamlit

Matplotlib

Seaborn
""")

    st.divider()

    st.subheader("📚 What I Learned")

    st.write("""
✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Data Visualization

✔ Machine Learning Classification

✔ Model Comparison

✔ Cross Validation

✔ Model Serialization

✔ Dashboard Development using Streamlit
    """)

    st.divider()

    st.subheader("🚀 Future Improvements")

    st.write("""
- Image-based Currency Detection

- Deep Learning (CNN)

- Support Multiple Currencies

- Real-time Camera Detection

- Cloud Deployment
    """)

    st.divider()

    st.markdown(
        """
### 👨‍💻 Developer

**Satvik**

Third Year Computer Engineering Student

Built using ❤️ with Python & Streamlit.
"""
    )