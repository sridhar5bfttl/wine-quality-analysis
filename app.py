import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configurations and Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="VintEdge: Interactive Wine Quality Workspace",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS injection for glassmorphic dark theme
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    /* Global Typography & Font override */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0a15 0%, #06050a 100%);
        color: #e2e8f0;
    }
    
    /* Custom Card Style */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
        transform: translateY(-2px);
    }
    
    /* Wine specific alerts/sections */
    .sommelier-panel {
        border-left: 6px solid #800020;
        background: rgba(128, 0, 32, 0.05);
    }
    .metric-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper to find project paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# ---------------------------------------------------------
# Data & Model Loaders
# ---------------------------------------------------------
@st.cache_data
def load_data(wine_type):
    file_path = os.path.join(DATA_DIR, f"winequality-{wine_type}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, sep=';')
    return None

@st.cache_resource
def load_model(wine_type):
    model_path = os.path.join(DATA_DIR, f"{wine_type}_wine_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

# Load both datasets and models
red_df = load_data("red")
white_df = load_data("white")
red_model_data = load_model("red")
white_model_data = load_model("white")

# Define the exact properties and ranges for input sliders (min, max, mean, step)
FEATURE_META = {
    "red": {
        'fixed acidity': (4.6, 15.9, 8.32, 0.1),
        'volatile acidity': (0.12, 1.58, 0.53, 0.01),
        'citric acid': (0.0, 1.0, 0.27, 0.01),
        'residual sugar': (0.9, 15.5, 2.54, 0.1),
        'chlorides': (0.01, 0.61, 0.09, 0.001),
        'free sulfur dioxide': (1.0, 72.0, 15.87, 1.0),
        'total sulfur dioxide': (6.0, 289.0, 46.47, 1.0),
        'density': (0.990, 1.004, 0.9967, 0.0001),
        'pH': (2.74, 4.01, 3.31, 0.01),
        'sulphates': (0.33, 2.0, 0.66, 0.01),
        'alcohol': (8.4, 14.9, 10.42, 0.1)
    },
    "white": {
        'fixed acidity': (3.8, 14.2, 6.85, 0.1),
        'volatile acidity': (0.08, 1.10, 0.28, 0.01),
        'citric acid': (0.0, 1.66, 0.32, 0.01),
        'residual sugar': (0.6, 65.8, 6.39, 0.1),
        'chlorides': (0.009, 0.346, 0.046, 0.001),
        'free sulfur dioxide': (2.0, 289.0, 35.31, 1.0),
        'total sulfur dioxide': (9.0, 440.0, 138.36, 1.0),
        'density': (0.987, 1.039, 0.9940, 0.0001),
        'pH': (2.72, 3.82, 3.19, 0.01),
        'sulphates': (0.22, 1.08, 0.49, 0.01),
        'alcohol': (8.0, 14.2, 10.51, 0.1)
    }
}

# ---------------------------------------------------------
# Sidebar Panel Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #bf4060;'>🍷 VintEdge Control</h2>", unsafe_allow_html=True)
    st.markdown("Configure the parameters of the wine study workspace.")
    
    # Toggle between red and white wine variants
    wine_selection = st.radio(
        "Select Wine Type",
        options=["Red Wine", "White Wine"],
        index=0
    )
    
    wine_type = "red" if wine_selection == "Red Wine" else "white"
    active_df = red_df if wine_type == "red" else white_df
    active_model_data = red_model_data if wine_type == "red" else white_model_data
    
    st.markdown("---")
    st.markdown("### Dataset Summary")
    if active_df is not None:
        st.markdown(f"**Total Samples:** `{len(active_df)}` rows")
        st.markdown(f"**Features:** `11` chemical metrics")
        st.markdown(f"**Quality Range:** `{active_df['quality'].min()} - {active_df['quality'].max()}`")
    else:
        st.error("No dataset loaded. Run `scripts/download_data.py` first.")
        
    st.markdown("---")
    st.markdown("<p style='font-size: 11px; opacity: 0.5;'>Powered by RAIL Framework methodologies & Random Forest classifiers.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main App Headers
# ---------------------------------------------------------
st.markdown(f"<h1 style='text-align: center; color: #f8fafc; margin-bottom: 0px;'>VintEdge Wine Workspace</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-size: 18px; color: #94a3b8; margin-top: 5px; margin-bottom: 30px;'>Analyzing the famous physicochemical wine dataset via ML & interactive sommelier modeling.</p>", unsafe_allow_html=True)

# Setup Workspace Tabs
tab1, tab2, tab3 = st.tabs([
    "🍷 The Virtual Sommelier", 
    "📊 Chemical Analytics (EDA)", 
    "🤖 Model Diagnostics"
])

# ---------------------------------------------------------
# TAB 1: Virtual Sommelier (Real-time Prediction)
# ---------------------------------------------------------
with tab1:
    st.markdown("### Predict Wine Quality in Real-Time")
    st.write("Adjust the physicochemical attributes below to simulate a custom vintage and predict whether the ML model classifies it as high quality.")
    
    if active_model_data is None:
        st.warning("Please make sure the machine learning models are trained by running `python3 scripts/train_model.py`.")
    else:
        # Load elements from pickled model
        model = active_model_data['model']
        features = active_model_data['features']
        
        # Grid layout for inputs
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4>🧪 Chemical Adjustments</h4>", unsafe_allow_html=True)
            
            # Draw sliders dynamically based on wine metadata
            inputs = {}
            # Divide into columns for compactness
            scol1, scol2 = st.columns(2)
            
            meta = FEATURE_META[wine_type]
            
            # Left half of features
            feature_list = list(meta.keys())
            midpoint = len(feature_list) // 2 + 1
            
            with scol1:
                for feat in feature_list[:midpoint]:
                    min_v, max_v, mean_v, step = meta[feat]
                    inputs[feat] = st.slider(
                        feat.title(),
                        min_value=float(min_v),
                        max_value=float(max_v),
                        value=float(mean_v),
                        step=float(step),
                        help=f"Mean: {mean_v}"
                    )
            
            # Right half of features
            with scol2:
                for feat in feature_list[midpoint:]:
                    min_v, max_v, mean_v, step = meta[feat]
                    inputs[feat] = st.slider(
                        feat.title(),
                        min_value=float(min_v),
                        max_value=float(max_v),
                        value=float(mean_v),
                        step=float(step),
                        help=f"Mean: {mean_v}"
                    )
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card sommelier-panel'>", unsafe_allow_html=True)
            st.markdown("<h4>🧐 Sommelier's Assessment</h4>", unsafe_allow_html=True)
            
            # Prepare feature array for prediction
            input_df = pd.DataFrame([inputs])
            # Reorder columns matching training columns
            input_df = input_df[features]
            
            # Run prediction
            pred_class = model.predict(input_df)[0]
            pred_probs = model.predict_proba(input_df)[0]
            
            prob_good = pred_probs[1]
            prob_not_good = pred_probs[0]
            
            # Visual display of the prediction
            if pred_class == 1:
                st.markdown("<h2 style='color: #2ecc71; text-align: center; margin-bottom: 5px;'>🏆 Premium Class</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: #86efac; font-size: 16px;'>Predicted high quality (score &ge; 6)<br><b>Confidence: {prob_good*100:.1f}%</b></p>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color: #e74c3c; text-align: center; margin-bottom: 5px;'>📉 Standard Class</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: #fca5a5; font-size: 16px;'>Predicted average quality (score &lt; 6)<br><b>Confidence: {prob_not_good*100:.1f}%</b></p>", unsafe_allow_html=True)
            
            # Display gauge / probability bar
            st.markdown("---")
            st.write("**Probability of High Quality:**")
            st.progress(float(prob_good))
            
            # Dynamic descriptive notes matching the specific input properties
            st.markdown("---")
            st.write("**Virtual Tasting Notes:**")
            
            notes = []
            if inputs['alcohol'] > meta['alcohol'][2] + 0.5:
                notes.append("High alcohol content adds full-bodied warmth and legs to the glass.")
            elif inputs['alcohol'] < meta['alcohol'][2] - 0.5:
                notes.append("Lower alcohol level provides a lighter, more refreshing palate presence.")
                
            if inputs['volatile acidity'] > meta['volatile acidity'][2] + 0.1:
                notes.append("Elevated volatile acidity presents a sharp, vinegar-like complexity that may overpower standard balances.")
            elif inputs['volatile acidity'] < meta['volatile acidity'][2] - 0.05:
                notes.append("Low volatile acidity keeps the aromatics clean, floral, and deeply polished.")
                
            if inputs['sulphates'] > meta['sulphates'][2] + 0.15:
                notes.append("High sulphates work efficiently to preserve freshness and stability.")
            
            if len(notes) == 0:
                notes.append("An exceptionally balanced, textbook vintage representative. Fits typical profiles perfectly.")
                
            st.info(" \n".join([f"• {note}" for note in notes]))
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: Chemical Analytics (EDA)
# ---------------------------------------------------------
with tab2:
    st.markdown("### Exploratory Data Visualizations")
    st.write("Understand correlations, distribution splits, and chemical profiles directly from the datasets.")
    
    if active_df is None:
        st.warning("No data found. Ensure the dataset CSVs are in the `data/` folder.")
    else:
        # Correlation Heatmap
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>🔗 Feature Correlation Heatmap</h4>", unsafe_allow_html=True)
        
        corr_matrix = active_df.corr()
        
        # Color palette depending on wine type
        color_scale = px.colors.sequential.Burg if wine_type == "red" else px.colors.sequential.YlOrBr
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale=color_scale,
            aspect="auto",
            title=f"Correlation Matrix ({wine_selection})"
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Grid of Distribution plots
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4>🍷 Quality Distribution</h4>", unsafe_allow_html=True)
            quality_counts = active_df['quality'].value_counts().sort_index().reset_index()
            quality_counts.columns = ['Quality Rating', 'Count']
            
            fig_bar = px.bar(
                quality_counts,
                x='Quality Rating',
                y='Count',
                color='Count',
                color_continuous_scale=color_scale,
                title="Number of Wines per Quality Score"
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_dist2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4>🧬 Chemical Distributions by Quality Class</h4>", unsafe_allow_html=True)
            
            # Select feature for analysis
            selected_feat = st.selectbox(
                "Select feature to plot against quality class:",
                options=list(FEATURE_META[wine_type].keys()),
                index=10 # Default to Alcohol
            )
            
            # Bin quality class
            temp_df = active_df.copy()
            temp_df['Quality Class'] = temp_df['quality'].apply(lambda x: 'High (>=6)' if x >= 6 else 'Standard (<6)')
            
            fig_box = px.box(
                temp_df,
                x='Quality Class',
                y=selected_feat,
                color='Quality Class',
                color_discrete_map={'High (>=6)': '#2ecc71', 'Standard (<6)': '#e74c3c'},
                points="all",
                title=f"{selected_feat.title()} Distribution by Quality Category"
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: Model Diagnostics
# ---------------------------------------------------------
with tab3:
    st.markdown("### Machine Learning Model Diagnostics")
    st.write("Understand how the Random Forest classifier makes decisions on wine properties.")
    
    if active_model_data is None:
        st.warning("Model data not loaded.")
    else:
        importance_list = active_model_data['feature_importances']
        importance_df = pd.DataFrame(importance_list)
        
        col_diag1, col_diag2 = st.columns([1, 1])
        
        with col_diag1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4>🔑 Feature Importance Chart</h4>", unsafe_allow_html=True)
            
            wine_color = "#800020" if wine_type == "red" else "#d4af37"
            
            fig_imp = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Random Forest Feature Importance",
                color_discrete_sequence=[wine_color]
            )
            fig_imp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_diag2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4>📈 Classifier Validation Metrics</h4>", unsafe_allow_html=True)
            
            # Print metrics depending on selected wine type
            if wine_type == "red":
                acc = 0.7375
                precision_g = 0.76
                recall_g = 0.81
                f1_g = 0.79
            else:
                acc = 0.7969
                precision_g = 0.82
                recall_g = 0.92
                f1_g = 0.86
                
            # Display stats in columns
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"""
                <div class='metric-container'>
                    <span style='font-size: 14px; opacity: 0.7;'>ACCURACY</span>
                    <span style='font-size: 32px; font-weight: bold; color: #60a5fa;'>{acc*100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='metric-container'>
                    <span style='font-size: 14px; opacity: 0.7;'>PRECISION (High Quality)</span>
                    <span style='font-size: 32px; font-weight: bold; color: #34d399;'>{precision_g*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div class='metric-container'>
                    <span style='font-size: 14px; opacity: 0.7;'>RECALL (High Quality)</span>
                    <span style='font-size: 32px; font-weight: bold; color: #fbbf24;'>{recall_g*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='metric-container'>
                    <span style='font-size: 14px; opacity: 0.7;'>F1-SCORE (High Quality)</span>
                    <span style='font-size: 32px; font-weight: bold; color: #a78bfa;'>{f1_g*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("The classifier is built using a Random Forest model with 100 decision trees, evaluated on a stratified 20% test partition to ensure validation stability.")
            st.markdown("</div>", unsafe_allow_html=True)
