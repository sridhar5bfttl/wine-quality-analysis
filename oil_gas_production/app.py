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
    page_title="PetroPulse: Upstream Virtual Sommelier & Flow Meter",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS injection for glassmorphic petroleum theme
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
        background: linear-gradient(135deg, #091512 0%, #030807 100%);
        color: #e2e8f0;
    }
    
    /* Custom Card Style */
    .metric-card {
        background: rgba(16, 28, 25, 0.45);
        border: 1px solid rgba(0, 242, 180, 0.15);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 20px;
        transition: all 0.3s ease-in-out;
    }
    .metric-card:hover {
        border-color: rgba(0, 242, 180, 0.4);
        box-shadow: 0 8px 32px 0 rgba(0, 242, 180, 0.1);
        transform: translateY(-2px);
    }
    
    /* Custom headers and colors */
    .highlight-green {
        color: #00f2b4;
        font-weight: 700;
    }
    
    /* Streamlit overrides */
    .stButton>button {
        background: linear-gradient(90deg, #00f2b4 0%, #00b4d8 100%);
        color: #030807 !important;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 12px 24px;
        transition: opacity 0.2s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: #030807 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load dataset
@st.cache_data
def load_data():
    csv_path = "oil_gas_production/data/well_production_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None

# Helper function to load model
@st.cache_resource
def load_model():
    model_path = "oil_gas_production/data/virtual_flow_meter_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

# Load resources
df = load_data()
model_payload = load_model()

# ---------------------------------------------------------
# Sidebar Navigation and Control Panel
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🛢️ PetroPulse</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0aec0; font-size: 0.9em;'>Upstream Virtual Flow Meter & Decline Analytics</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation
    app_mode = st.radio(
        "Navigation",
        ["📈 Production Dashboard", "⚙️ Virtual Flow Meter", "🔍 Diagnostic Center"]
    )
    st.markdown("---")
    
    # Well status card
    if df is not None:
        last_row = df.iloc[-1]
        st.markdown(f"""
        <div class="metric-card" style="padding: 16px; margin-bottom: 10px;">
            <p style="margin:0; font-size:0.8em; color:#a0aec0;">Well Status</p>
            <h3 style="margin:5px 0 0 0; color:#00f2b4;">ACTIVE (PRODUCING)</h3>
            <p style="margin:5px 0 0 0; font-size:0.85em;">📅 Last Log: {last_row['date'].strftime('%Y-%m-%d')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Dataset not found. Please run generate_data.py.")

# ---------------------------------------------------------
# MAIN APP ROUTING
# ---------------------------------------------------------
if df is None:
    st.title("🛢️ Welcome to PetroPulse Upstream Workspace")
    st.error("No production data found! Please execute the generation script to populate the database.")
    if st.button("Generate Dataset & Compile Models"):
        with st.spinner("Simulating reservoir and training ML pipelines..."):
            os.system("python3 oil_gas_production/scripts/generate_data.py")
            os.system("python3 oil_gas_production/scripts/train_forecaster.py")
            st.rerun()

else:
    # ---------------------------------------------------------
    # TAB 1: PRODUCTION DASHBOARD
    # ---------------------------------------------------------
    if app_mode == "📈 Production Dashboard":
        st.markdown("<h1 style='margin-bottom: 0px;'>📈 Operational Decline Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #a0aec0;'>Analyze production rates, pressures, and water encroachment trends over the well life-cycle</p>", unsafe_allow_html=True)
        
        # High-level Metrics Row
        last_row = df.iloc[-1]
        cum_oil = (df["oil_volume_bbl"].sum()) / 1000.0
        cum_gas = (df["gas_volume_mscf"].sum()) / 1000.0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p style="margin:0; font-size:0.9em; color:#a0aec0;">Current Oil Rate</p>
                <h2 style="margin:5px 0 0 0; font-size:2.2em; color:#e2e8f0;">{last_row['oil_volume_bbl']:.1f} <span style="font-size:0.5em; color:#00f2b4;">BPD</span></h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p style="margin:0; font-size:0.9em; color:#a0aec0;">Water Cut</p>
                <h2 style="margin:5px 0 0 0; font-size:2.2em; color:#e2e8f0;">{last_row['water_cut_percentage']:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p style="margin:0; font-size:0.9em; color:#a0aec0;">Cumulative Oil</p>
                <h2 style="margin:5px 0 0 0; font-size:2.2em; color:#e2e8f0;">{cum_oil:.1f}k <span style="font-size:0.5em; color:#00b4d8;">BBL</span></h2>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p style="margin:0; font-size:0.9em; color:#a0aec0;">Cumulative Gas</p>
                <h2 style="margin:5px 0 0 0; font-size:2.2em; color:#e2e8f0;">{cum_gas:.1f}k <span style="font-size:0.5em; color:#e2e8f0;">MCF</span></h2>
            </div>
            """, unsafe_allow_html=True)
            
        # Time-series plotting tabs
        plot_tab1, plot_tab2, plot_tab3 = st.tabs(["🛢️ Flow Volumes", "📉 Pressures & Temperatures", "💧 Water Encroachment"])
        
        with plot_tab1:
            st.subheader("Multiphase Production Decline Curves")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=df["date"], y=df["oil_volume_bbl"], name="Oil Volume (bbl)", line=dict(color="#00f2b4", width=2)))
            fig_vol.add_trace(go.Scatter(x=df["date"], y=df["gas_volume_mscf"], name="Gas Volume (mscf)", line=dict(color="#00b4d8", width=1.5)))
            fig_vol.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Daily Rate"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_vol, use_container_width=True)
            
        with plot_tab2:
            st.subheader("Pressure & Temperature Decline Profile")
            fig_pres = go.Figure()
            fig_pres.add_trace(go.Scatter(x=df["date"], y=df["bottomhole_pressure_psi"], name="Bottomhole Pressure (PSI)", line=dict(color="#ff5e7e", width=2)))
            fig_pres.add_trace(go.Scatter(x=df["date"], y=df["wellhead_pressure_psi"], name="Wellhead Pressure (PSI)", line=dict(color="#ffd200", width=1.5)))
            fig_pres.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Pressure (PSI)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pres, use_container_width=True)
            
        with plot_tab3:
            st.subheader("Water Cut Encroachment Curve (Aquifer Coning)")
            fig_wc = px.line(df, x="date", y="water_cut_percentage", labels={"water_cut_percentage": "Water Cut (%)"})
            fig_wc.update_traces(line=dict(color="#a29bfe", width=2.5))
            fig_wc.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Water Cut (%)")
            )
            st.plotly_chart(fig_wc, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: VIRTUAL FLOW METER
    # ---------------------------------------------------------
    elif app_mode == "⚙️ Virtual Flow Meter":
        st.markdown("<h1>⚙️ Virtual Flow Meter Sandbox</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #a0aec0;'>Enter sensor values and operating parameters to estimate daily oil production via the Random Forest model.</p>", unsafe_allow_html=True)
        
        if model_payload is None:
            st.error("Model files not found! Go to the Diagnostic Center or run train_forecaster.py to generate models.")
        else:
            features = model_payload["features"]
            model = model_payload["model"]
            metrics = model_payload["metrics"]
            
            st.markdown(f"""
            <div style="background: rgba(0, 242, 180, 0.05); border-left: 4px solid #00f2b4; padding: 12px; border-radius: 4px; margin-bottom: 25px;">
                💡 <b>Virtual Flow Metering:</b> Using a Random Forest Regressor trained on physical well data. Model accuracy (R²): <b>{metrics['r2']*100:.2f}%</b> (MAE: {metrics['mae']:.1f} bbl/day).
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Sliders for Input
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                st.markdown("### 🔌 Operational Parameters")
                on_stream_hours = st.slider("Well Uptime (hours/day)", 0.0, 24.0, 24.0, step=1.0)
                choke_size = st.slider("Choke Valve Size (%)", 0.0, 100.0, 75.0, step=1.0)
                bottomhole_pressure = st.slider("Bottomhole Pressure (PSI)", 500.0, 4500.0, 2800.0, step=50.0)
                
            with col_in2:
                st.markdown("### 🌡️ Temperature & Pressure Sensors")
                wellhead_pressure = st.slider("Wellhead Pressure (PSI)", 100.0, 3000.0, 1100.0, step=50.0)
                wellhead_temperature = st.slider("Wellhead Temperature (°F)", 50.0, 150.0, 105.0, step=1.0)
                bottomhole_temperature = st.slider("Reservoir Temp (BHT - °F)", 150.0, 200.0, 180.0, step=1.0)
                
            # Perform Inference
            input_data = pd.DataFrame([{
                "on_stream_hours": on_stream_hours,
                "choke_size_percentage": choke_size,
                "bottomhole_pressure_psi": bottomhole_pressure,
                "bottomhole_temperature_f": bottomhole_temperature,
                "wellhead_pressure_psi": wellhead_pressure,
                "wellhead_temperature_f": wellhead_temperature
            }])
            
            # If well is shut down, flow is physically zero
            if on_stream_hours == 0:
                predicted_oil = 0.0
            else:
                predicted_oil = model.predict(input_data)[0]
                
            # Result Display Card
            st.markdown("---")
            st.subheader("🔮 Estimated Production Result")
            
            col_res1, col_res2 = st.columns([2, 3])
            with col_res1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; border-color: #00f2b4;">
                    <p style="margin:0; font-size:1.1em; color:#00f2b4; font-weight:600;">Estimated Oil Flow Rate</p>
                    <h1 style="font-size:3.5em; margin:10px 0; color:#e2e8f0;">{predicted_oil:.2f}</h1>
                    <p style="margin:0; color:#a0aec0; font-size:1.1em;">Barrels Per Day (BPD)</p>
                </div>
                """, unsafe_allow_html=True)
            with col_res2:
                # Gauge plot
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_oil,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Flow Meter Capacity Indicator"},
                    gauge={
                        'axis': {'range': [0, 5000]},
                        'bar': {'color': "#00f2b4"},
                        'steps': [
                            {'range': [0, 1500], 'color': "rgba(255, 94, 126, 0.15)"},
                            {'range': [1500, 3500], 'color': "rgba(255, 210, 0, 0.15)"},
                            {'range': [3500, 5000], 'color': "rgba(0, 242, 180, 0.15)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 4500
                        }
                    }
                ))
                fig_gauge.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(t=30, b=0, l=10, r=10)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: DIAGNOSTIC CENTER
    # ---------------------------------------------------------
    elif app_mode == "🔍 Diagnostic Center":
        st.markdown("<h1>🔍 Subsurface & Model Diagnostics</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #a0aec0;'>Explore correlation metrics and feature sensitivities calculated from the active dataset.</p>", unsafe_allow_html=True)
        
        diag_tab1, diag_tab2 = st.tabs(["🔥 Correlation Matrix", "📊 Model Feature Importance"])
        
        with diag_tab1:
            st.subheader("Physical Feature Correlations")
            numeric_cols = [
                "on_stream_hours", "choke_size_percentage", "bottomhole_pressure_psi",
                "bottomhole_temperature_f", "wellhead_pressure_psi", "wellhead_temperature_f",
                "oil_volume_bbl", "gas_volume_mscf", "water_volume_bbl"
            ]
            corr_mat = df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_mat,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                labels=dict(x="Feature", y="Feature", color="Correlation")
            )
            fig_corr.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with diag_tab2:
            st.subheader("Random Forest Feature Importance Profile")
            if model_payload is None:
                st.warning("Please compile the model to check feature importances.")
            else:
                features = model_payload["features"]
                importances = model_payload["model"].feature_importances_
                
                imp_df = pd.DataFrame({
                    "Feature": [f.replace("_", " ").title() for f in features],
                    "Importance (%)": importances * 100
                }).sort_values(by="Importance (%)", ascending=True)
                
                fig_imp = px.bar(
                    imp_df,
                    x="Importance (%)",
                    y="Feature",
                    orientation="h",
                    color="Importance (%)",
                    color_continuous_scale="viridis"
                )
                fig_imp.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
