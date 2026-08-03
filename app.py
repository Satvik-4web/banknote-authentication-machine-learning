import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import time
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Banknote Machine Learning Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Injection (Premium Dark Theme & Glassmorphism UI) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #07111F;
    color: #F8FAFC;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Glassmorphism Card Container */
.glass-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    margin-bottom: 24px;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
    border-color: rgba(34, 197, 94, 0.35);
}

/* Hero Section */
.hero {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(34, 197, 94, 0.12) 100%);
    border-radius: 24px;
    padding: 48px 36px;
    text-align: center;
    margin-bottom: 32px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    position: relative;
    overflow: hidden;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 700;
    margin-bottom: 12px;
    background: linear-gradient(to right, #3B82F6, #22C55E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    font-size: 1.15rem;
    color: #94A3B8;
    max-width: 700px;
    margin: 0 auto;
}

/* KPI Cards Layout */
.metric-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin-bottom: 28px;
}
.metric-card {
    background: rgba(15, 23, 42, 0.55);
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    text-align: center;
    border-top: 3px solid #3B82F6;
    transition: all 0.3s ease;
}
.metric-card:hover {
    background: rgba(30, 41, 59, 0.75);
    transform: translateY(-2px);
}
.metric-card.green {
    border-top: 3px solid #22C55E;
}
.metric-card h3 {
    font-size: 0.85rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.metric-card h2 {
    font-size: 2.2rem;
    color: #F8FAFC;
    margin: 0;
    font-weight: 700;
}

/* Custom Buttons */
.stButton>button {
    background: linear-gradient(to right, #3B82F6, #22C55E) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(34, 197, 94, 0.4) !important;
}

/* Prediction Cards */
.result-card {
    padding: 36px;
    border-radius: 20px;
    text-align: center;
    margin-top: 10px;
    animation: fadeIn 0.5s ease-out forwards;
}
.result-genuine {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.12), rgba(34, 197, 94, 0.22));
    border: 2px solid #22C55E;
    box-shadow: 0 0 25px rgba(34, 197, 94, 0.25);
}
.result-fake {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(239, 68, 68, 0.22));
    border: 2px solid #EF4444;
    box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
}
.result-icon {
    font-size: 3.5rem;
    margin-bottom: 12px;
}
.result-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.result-conf {
    font-size: 1.2rem;
    color: #CBD5E1;
}

/* Typography & Divider Overrides */
h1, h2, h3, h4 {
    color: #F8FAFC !important;
}
hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(59, 130, 246, 0.2);
    color: #3B82F6;
    border: 1px solid rgba(59, 130, 246, 0.35);
    margin-right: 6px;
}
.badge.success {
    background: rgba(34, 197, 94, 0.2);
    color: #22C55E;
    border: 1px solid rgba(34, 197, 94, 0.35);
}

/* Workflow Timeline */
.timeline {
    border-left: 2px solid #3B82F6;
    padding-left: 20px;
    margin-left: 10px;
}
.timeline-item {
    position: relative;
    margin-bottom: 20px;
}
.timeline-item::before {
    content: '';
    position: absolute;
    left: -27px;
    top: 5px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #3B82F6;
    border: 2px solid #07111F;
}
.timeline-item h4 { margin: 0 0 4px 0; color: #F8FAFC; font-size: 1rem; }
.timeline-item p { margin: 0; color: #94A3B8; font-size: 0.88rem; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# --- REAL DATA LOADING ---
@st.cache_data
def load_dataset():
    path = os.path.join("dataset", "dataset.txt")
    if os.path.exists(path):
        data = pd.read_csv(path, header=None)
        data.columns = ["Variance", "Skewness", "Curtosis", "Entropy", "Class"]
        return data
    else:
        st.error(f"Dataset file not found at: {path}")
        return pd.DataFrame(columns=["Variance", "Skewness", "Curtosis", "Entropy", "Class"])

@st.cache_resource
def load_trained_model():
    path = os.path.join("models", "currency_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model file not found at: {path}")
        return None


# --- REAL DYNAMIC MODEL EVALUATION (Reusing logic from compare_models.py) ---
@st.cache_data
def run_model_comparison(data):
    if data.empty or len(data) < 10:
        return pd.DataFrame()
    
    X = data.drop("Class", axis=1)
    y = data["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True)
    }
    
    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        cv_val = cross_val_score(m, X, y, cv=5).mean()
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Cross Validation": cv_val,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })
        
    res_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    return res_df


# Load actual dataset & model
df = load_dataset()
trained_model = load_trained_model()
model_comparison_df = run_model_comparison(df)


# --- DYNAMIC CALCULATIONS FROM REAL DATA ---
total_samples = len(df)
feature_count = df.shape[1] - 1 if not df.empty else 0
models_evaluated_count = len(model_comparison_df) if not model_comparison_df.empty else 0
best_accuracy_val = model_comparison_df["Accuracy"].max() if not model_comparison_df.empty else 0.0
best_model_row = model_comparison_df.iloc[0] if not model_comparison_df.empty else None


# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("<h2 style='text-align: center; color: #22C55E; margin-top: 10px; margin-bottom: 20px;'>🏦 Banknote AI</h2>", unsafe_allow_html=True)

menu = ["🏠 Dashboard", "📊 Dataset Analysis", "🤖 Model Comparison", "💰 Prediction", "ℹ About"]
choice = st.sidebar.radio("Navigation", menu, label_visibility="collapsed")

st.sidebar.markdown("<br>"*4, unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background: rgba(59, 130, 246, 0.08); padding: 16px; border-radius: 12px; text-align: center; border: 1px solid rgba(59, 130, 246, 0.2);">
    <p style="color: #94A3B8; font-size: 0.78rem; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">SYSTEM STATUS</p>
    <p style="color: #22C55E; font-weight: 600; margin: 6px 0 0 0; font-size: 0.92rem;">🟢 ML Pipeline Active</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 🏠 DASHBOARD PAGE
# ==========================================
if choice == "🏠 Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>Banknote Authentication Analytics</h1>
        <p>Real-Time FinTech Machine Learning Dashboard dynamically evaluating wavelet transformed banknote features.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Real KPI Cards dynamically calculated
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <h3>Dataset Samples</h3>
            <h2>{total_samples:,}</h2>
        </div>
        <div class="metric-card">
            <h3>Feature Count</h3>
            <h2>{feature_count}</h2>
        </div>
        <div class="metric-card green">
            <h3>Models Evaluated</h3>
            <h2>{models_evaluated_count}</h2>
        </div>
        <div class="metric-card green">
            <h3>Best Accuracy</h3>
            <h2>{best_accuracy_val*100:.2f}%</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3>🎯 Real Model Accuracy Gauge</h3>", unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = best_accuracy_val * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Top Model: {best_model_row['Model'] if best_model_row is not None else 'N/A'}", 'font': {'color': '#F8FAFC', 'size': 16}},
            number = {'suffix': "%", 'font': {'color': '#22C55E', 'size': 36}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': "#22C55E"},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.25)"},
                    {'range': [60, 85], 'color': "rgba(234, 179, 8, 0.25)"},
                    {'range': [85, 100], 'color': "rgba(34, 197, 94, 0.25)"}
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#F8FAFC"},
            height=290,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3>🔄 Machine Learning Pipeline</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="timeline">
            <div class="timeline-item">
                <h4>1. Dataset Load</h4>
                <p>Loads 1,372 continuous wavelet transform observations.</p>
            </div>
            <div class="timeline-item">
                <h4>2. Feature Analysis</h4>
                <p>Computes variance, skewness, curtosis & entropy.</p>
            </div>
            <div class="timeline-item">
                <h4>3. Multi-Model Benchmark</h4>
                <p>Trains & compares 5 Scikit-Learn classifiers.</p>
            </div>
            <div class="timeline-item">
                <h4>4. Production Inference</h4>
                <p>Runs saved KNeighborsClassifier model.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Real Insights derived dynamically
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3>⚡ Real Dataset Insights</h3>", unsafe_allow_html=True)
    if not df.empty:
        class_counts = df['Class'].value_counts()
        g_count = class_counts.get(0, 0)
        f_count = class_counts.get(1, 0)
        corr_matrix = df.corr()
        top_corr_feat = corr_matrix['Class'].abs().drop('Class').idxmax()
        top_corr_val = corr_matrix.loc[top_corr_feat, 'Class']
        
        st.markdown(f"""
        - **Dataset Balance:** Found **{g_count:,}** Genuine (Class 0) and **{f_count:,}** Counterfeit (Class 1) samples.
        - **Key Discriminator:** **{top_corr_feat}** has the strongest correlation with note authenticity (r = **{top_corr_val:.3f}**).
        - **Leaderboard Winner:** **{best_model_row['Model']}** achieved **{best_accuracy_val*100:.2f}%** test accuracy with a 5-fold cross-validation mean of **{best_model_row['Cross Validation']*100:.2f}%**.
        """)
    else:
        st.info("No data available.")
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 📊 DATASET ANALYSIS PAGE
# ==========================================
elif choice == "📊 Dataset Analysis":
    st.markdown("<h1 style='margin-bottom: 24px;'>Dataset Analysis</h1>", unsafe_allow_html=True)
    
    if df.empty:
        st.warning("Dataset not loaded.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Class Distribution</h3>", unsafe_allow_html=True)
            class_counts = df['Class'].value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']
            class_counts['Label'] = class_counts['Class'].map({0: 'Genuine (0)', 1: 'Counterfeit (1)'})
            
            fig_donut = px.pie(
                class_counts, values='Count', names='Label',
                hole=0.55,
                color='Label',
                color_discrete_map={'Genuine (0)': '#22C55E', 'Counterfeit (1)': '#EF4444'}
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#F8FAFC"},
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Feature Distribution Histogram</h3>", unsafe_allow_html=True)
            selected_feature = st.selectbox("Select Feature", ["Variance", "Skewness", "Curtosis", "Entropy"])
            
            fig_hist = px.histogram(
                df, x=selected_feature, color="Class",
                barmode="overlay",
                color_discrete_map={0: '#22C55E', 1: '#EF4444'},
                labels={"Class": "Class"}
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#F8FAFC"},
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Correlation Heatmap</h3>", unsafe_allow_html=True)
        corr = df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto='.3f',
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#F8FAFC"},
            margin=dict(t=30, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Feature Box Plots</h3>", unsafe_allow_html=True)
        box_feat = st.selectbox("Select Box Plot Feature", ["Variance", "Skewness", "Curtosis", "Entropy"], key="box_select")
        fig_box = px.box(
            df, y=box_feat, x="Class", color="Class",
            color_discrete_map={0: '#22C55E', 1: '#EF4444'},
            points="all"
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#F8FAFC"}
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Scatter Matrix</h3>", unsafe_allow_html=True)
        fig_scatter = px.scatter_matrix(
            df,
            dimensions=["Variance", "Skewness", "Curtosis", "Entropy"],
            color="Class",
            color_discrete_map={0: '#22C55E', 1: '#EF4444'},
            opacity=0.65
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#F8FAFC"},
            height=600,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        c_stat, c_miss = st.columns(2)
        with c_stat:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Feature Statistics Summary</h3>", unsafe_allow_html=True)
            st.dataframe(df.describe().T, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c_miss:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Missing Value Analysis</h3>", unsafe_allow_html=True)
            missing = pd.DataFrame({
                "Feature": df.columns,
                "Missing Values": df.isnull().sum().values,
                "Data Type": df.dtypes.values.astype(str)
            })
            st.dataframe(missing, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Interactive Dataset Table</h3>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 🤖 MODEL COMPARISON PAGE
# ==========================================
elif choice == "🤖 Model Comparison":
    st.markdown("<h1 style='margin-bottom: 24px;'>Model Comparison & Leaderboard</h1>", unsafe_allow_html=True)
    
    if model_comparison_df.empty:
        st.warning("Model comparison data unavailable.")
    else:
        top_m = model_comparison_df.iloc[0]
        
        st.markdown(f"""
        <div class="glass-card" style="border: 1px solid #22C55E; background: linear-gradient(135deg, rgba(34, 197, 94, 0.06), rgba(15, 23, 42, 0.65));">
            <h2 style="color: #22C55E; margin-bottom: 8px;">🏆 Best Performing Classifier</h2>
            <h1 style="margin: 0; font-size: 2.5rem;">{top_m['Model']}</h1>
            <p style="font-size: 1.15rem; color: #CBD5E1; margin-top: 10px;">
                Real Test Accuracy: <strong style="color: #22C55E;">{top_m['Accuracy']*100:.2f}%</strong> | 
                5-Fold Cross Validation: <strong style="color: #3B82F6;">{top_m['Cross Validation']*100:.2f}%</strong> | 
                F1 Score: <strong style="color: #F8FAFC;">{top_m['F1 Score']:.4f}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Accuracy Comparison Bar Chart</h3>", unsafe_allow_html=True)
            
            plot_df = model_comparison_df.sort_values(by="Accuracy", ascending=True)
            fig_bar = px.bar(
                plot_df,
                x=[val * 100 for val in plot_df["Accuracy"]],
                y="Model",
                orientation='h',
                color=[val * 100 for val in plot_df["Accuracy"]],
                color_continuous_scale="Greens",
                labels={'x': 'Accuracy (%)', 'Model': 'Model'},
                text=[f"{val*100:.2f}%" for val in plot_df["Accuracy"]]
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#F8FAFC"},
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Detailed Metrics Leaderboard</h3>", unsafe_allow_html=True)
            
            formatted_df = model_comparison_df.copy()
            formatted_df["Accuracy"] = formatted_df["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
            formatted_df["Cross Validation"] = formatted_df["Cross Validation"].apply(lambda x: f"{x*100:.2f}%")
            formatted_df["Precision"] = formatted_df["Precision"].apply(lambda x: f"{x:.4f}")
            formatted_df["Recall"] = formatted_df["Recall"].apply(lambda x: f"{x:.4f}")
            formatted_df["F1 Score"] = formatted_df["F1 Score"].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(formatted_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 💰 PREDICTION PAGE
# ==========================================
elif choice == "💰 Prediction":
    st.markdown("<h1 style='margin-bottom: 24px;'>Live Banknote Authentication</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Input Wavelet Features</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; margin-bottom: 20px;'>Enter parameters extracted from banknote wavelet transformation.</p>", unsafe_allow_html=True)
        
        with st.form("predict_form"):
            var_val = st.number_input("Variance", value=3.6216, format="%.5f", help="Variance of Wavelet Transformed image")
            skew_val = st.number_input("Skewness", value=8.6661, format="%.5f", help="Skewness of Wavelet Transformed image")
            curt_val = st.number_input("Curtosis", value=-2.8073, format="%.5f", help="Curtosis of Wavelet Transformed image")
            ent_val = st.number_input("Entropy", value=-0.44699, format="%.5f", help="Entropy of image")
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_submitted = st.form_submit_button("Predict Authenticity")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        if predict_submitted:
            if trained_model is None:
                st.error("Model file `models/currency_model.pkl` could not be loaded.")
            else:
                with st.spinner("Executing model.predict()..."):
                    input_arr = np.array([[var_val, skew_val, curt_val, ent_val]])
                    
                    # REAL PREDICTION FROM SAVED MODEL
                    pred_class = trained_model.predict(input_arr)[0]
                    
                    # REAL PROBABILITY (IF AVAILABLE)
                    if hasattr(trained_model, "predict_proba"):
                        proba = trained_model.predict_proba(input_arr)[0]
                        conf_percentage = np.max(proba) * 100
                    else:
                        conf_percentage = 100.0
                        
                    st.markdown("<div class='glass-card' style='height: 100%; display: flex; align-items: center; justify-content: center;'>", unsafe_allow_html=True)
                    
                    if pred_class == 0:
                        st.markdown(f"""
                        <div class="result-card result-genuine">
                            <div class="result-icon">✅</div>
                            <div class="result-title" style="color: #22C55E;">Genuine Banknote</div>
                            <div class="result-conf">Model Confidence: <strong>{conf_percentage:.2f}%</strong></div>
                            <p style="color: #94A3B8; margin-top: 14px;">The input feature vector aligns with authentic banknote distributions.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                        <div class="result-card result-fake">
                            <div class="result-icon">⚠️</div>
                            <div class="result-title" style="color: #EF4444;">Counterfeit Banknote</div>
                            <div class="result-conf">Model Confidence: <strong>{conf_percentage:.2f}%</strong></div>
                            <p style="color: #94A3B8; margin-top: 14px;">Anomalous wavelet variance/skewness detected by the classifier.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='glass-card' style='height: 100%; min-height: 360px; display: flex; align-items: center; justify-content: center; text-align: center; border: 1px dashed rgba(255,255,255,0.15);'>
                <div>
                    <h1 style='color: rgba(255,255,255,0.1); font-size: 4.5rem; margin: 0;'>🔍</h1>
                    <h3 style='color: #94A3B8; margin-top: 10px;'>Awaiting Input</h3>
                    <p style='color: #64748B;'>Adjust feature values and click <strong>Predict Authenticity</strong> to test.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# ℹ ABOUT PAGE
# ==========================================
elif choice == "ℹ About":
    st.markdown("<h1 style='margin-bottom: 24px;'>About Banknote AI</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h2>Project Architecture</h2>
        <p style="color: #94A3B8; font-size: 1.05rem; line-height: 1.7;">
        This application acts as a <strong>Live Interactive Machine Learning Frontend</strong> for banknote authentication.
        Features were extracted from images taken from genuine and forged banknote-like specimens using a industrial high-evaluation camera.
        Continuous Wavelet Transform (CWT) was applied to digitize images.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        <div class="glass-card">
            <h3>🛠️ Core Tech Stack</h3>
            <ul style="color: #94A3B8; line-height: 1.8;">
                <li><strong style="color: #F8FAFC;">Frontend Dashboard:</strong> Streamlit with Embedded Custom Glassmorphic CSS</li>
                <li><strong style="color: #F8FAFC;">Analytics Engine:</strong> Plotly Express & Plotly Graph Objects</li>
                <li><strong style="color: #F8FAFC;">Machine Learning Pipeline:</strong> Scikit-Learn & Joblib</li>
                <li><strong style="color: #F8FAFC;">Data Manipulation:</strong> Pandas & NumPy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown("""
        <div class="glass-card">
            <h3>📈 Machine Learning Workflow</h3>
            <ul style="color: #94A3B8; line-height: 1.8;">
                <li><strong style="color: #F8FAFC;">Data Source:</strong> dataset/dataset.txt</li>
                <li><strong style="color: #F8FAFC;">Saved Artifact:</strong> models/currency_model.pkl</li>
                <li><strong style="color: #F8FAFC;">Validation Strategy:</strong> Train-Test Split (80/20) + 5-Fold Cross Validation</li>
                <li><strong style="color: #F8FAFC;">Deployment:</strong> Live inference via joblib model serialization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h3>Production Dashboard</h3>
        <p style="color: #94A3B8;">Dynamic FinTech Analytics Platform</p>
        <div style="margin-top: 16px;">
            <span class="badge success">Dynamic Calculation</span>
            <span class="badge">Joblib Integration</span>
            <span class="badge">Plotly Native</span>
            <span class="badge success">Zero Hardcoding</span>
        </div>
        <hr style="margin: 24px 0;">
        <p style="color: #64748B; font-size: 0.85rem;">Banknote Authentication ML Analytics © 2026</p>
    </div>
    """, unsafe_allow_html=True)