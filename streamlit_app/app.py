import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
import base64
import time
import numpy as np

# Page configuration
st.set_page_config(page_title="DisasterSync: Risk Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a futuristic, sleek design
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #00D4FF;
        text-align: center;r
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 20px;
        color: #BDC3C7;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00D4FF, #7B00FF);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 12px 30px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.5);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #7B00FF, #00D4FF);
        box-shadow: 0 6px 20px rgba(123, 0, 255, 0.7);
    }
    .risk-output {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        margin-top: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .tab-content {
        padding: 25px;
        background: rgba(44, 62, 80, 0.9);
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        color: #ECF0F1;
    }
    .stSlider>.st-bq {
        background-color: #3498DB;
        border-radius: 10px;
    }
    .stProgress>.st-bo {
        background-color: #00D4FF;
    }
    </style>
""", unsafe_allow_html=True)

# Prediction function
def predict_disaster(input_data):
    try:
        model = joblib.load("random_forest_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        return label_encoder.inverse_transform(prediction)[0]
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# Generate downloadable report
def generate_report(input_data, predicted_risk):
    report = f"""
    # DisasterSync Prediction Report
    **Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Input Parameters
    {chr(10).join([f"- {k}: {v}" for k, v in input_data.items()])}
    
    ## Predicted Risk
    **Level:** {predicted_risk}
    """
    return report

# Theme toggle in sidebar
with st.sidebar:
    st.header("DisasterSync")
    theme = st.selectbox("Theme", ["Dark", "Light"])
    if theme == "Light":
        st.markdown("<style>body {background-color: #FFFFFF; color: #000000;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>body {background-color: #1C2526; color: #ECF0F1;}</style>", unsafe_allow_html=True)
    
    st.write("Advanced disaster risk analysis at your fingertips.")
    st.markdown("### Navigation")
    st.link_button("Documentation", "https://example.com", type="primary")

# Header
st.markdown('<p class="main-title">DisasterSync: Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Real-Time Environmental Risk Assessment</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Visualization", "Insights"])

# Tab 1: Prediction
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    with st.expander("Input Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Weather Metrics")
            temp = st.slider("Temperature (°C)", -50.0, 60.0, 30.5, 0.1, key="temp")
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0, 0.1, key="humidity")
            rainfall = st.slider("Rainfall (mm)", 0.0, 1000.0, 120.0, 1.0, key="rainfall")

        with col2:
            st.markdown("#### Atmospheric & Seismic")
            wind_speed = st.slider("Wind Speed (km/h)", 0.0, 300.0, 80.0, 0.1, key="wind")
            air_pressure = st.slider("Air Pressure (hPa)", 900.0, 1100.0, 1015.0, 0.1, key="pressure")
            seismic_activity = st.slider("Seismic Activity (Richter)", 0.0, 10.0, 3.2, 0.1, key="seismic")

        with col3:
            st.markdown("#### Environmental")
            soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 45.0, 0.1, key="soil")
            river_water_level = st.slider("River Water Level (m)", 0.0, 50.0, 7.0, 0.1, key="river")
            co2_levels = st.slider("CO2 Levels (ppm)", 200.0, 1000.0, 400.0, 1.0, key="co2")
            lightning_strikes = st.slider("Lightning Strikes (count/hr)", 0, 100, 5, 1, key="lightning")

    # Real-time feedback
    input_data = {
        "Temperature (°C)": temp,
        "Humidity (%)": humidity,
        "Rainfall (mm)": rainfall,
        "Wind Speed (km/h)": wind_speed,
        "Seismic Activity (Richter scale)": seismic_activity,
        "Air Pressure (hPa)": air_pressure,
        "Soil Moisture (%)": soil_moisture,
        "River Water Level (m)": river_water_level,
        "CO2 Levels (ppm)": co2_levels,
        "Lightning Strikes (count per hour)": lightning_strikes
    }
    st.write("**Current Input Summary:**", input_data["Temperature (°C)"], "°C, ", 
             input_data["Humidity (%)"], "%, ", input_data["Wind Speed (km/h)"], "km/h")

    if st.button("Analyze Risk"):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        
        predicted_risk = predict_disaster(input_data)
        if predicted_risk:
            risk_styles = {
                "Low": "background: linear-gradient(90deg, #27AE60, #2ECC71); color: white;",
                "Medium": "background: linear-gradient(90deg, #F1C40F, #F39C12); color: black;",
                "High": "background: linear-gradient(90deg, #E74C3C, #C0392B); color: white;"
            }
            style = risk_styles.get(predicted_risk, "background-color: #555555; color: white;")
            st.markdown(f'<div class="risk-output" style="{style}">Risk Level: {predicted_risk}</div>', unsafe_allow_html=True)
            
            # Download report
            report = generate_report(input_data, predicted_risk)
            st.download_button("Download Report", report, "disaster_risk_report.txt", "text/plain")
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Visualization
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("Interactive Visualizations")
    if 'input_data' in locals():
        df = pd.DataFrame([input_data])

        # Interactive Bar Chart with Plotly
        fig_bar = px.bar(df.melt(), x="variable", y="value", title="Parameter Values",
                         color="variable", height=400, template="plotly_dark" if theme == "Dark" else "plotly_white")
        fig_bar.update_layout(xaxis_title="Parameters", yaxis_title="Values", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        # 3D Scatter Plot
        st.write("### 3D Risk Factor Analysis")
        fig_3d = px.scatter_3d(df, x="Temperature (°C)", y="Wind Speed (km/h)", z="Rainfall (mm)",
                               size="Seismic Activity (Richter scale)", color="CO2 Levels (ppm)",
                               title="3D Environmental Factors", height=500,
                               template="plotly_dark" if theme == "Dark" else "plotly_white")
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("Input data in the Prediction tab to see visualizations.")
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Insights
with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("Risk Insights")
    if 'predicted_risk' in locals() and predicted_risk:
        st.write(f"**Risk Level:** {predicted_risk}")
        if predicted_risk == "High":
            st.warning("Immediate action recommended: High risk detected.")
        elif predicted_risk == "Medium":
            st.info("Monitor closely: Moderate risk identified.")
        else:
            st.success("Stable conditions: Low risk.")
        
        # Placeholder for AI-driven insights
        st.write("### AI-Driven Recommendations")
        st.write("- Adjust monitoring frequency based on seismic and rainfall data.")
        st.write("- Review emergency protocols if wind speed exceeds 100 km/h.")
    else:
        st.write("Run a prediction to see insights.")
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.markdown("### System Info")
    st.write("- **Model:** Random Forest")
    st.write("- **Features:** 10")
    st.write("- **Last Updated:** March 21, 2025")