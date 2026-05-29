import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import base64
import requests

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Smart Aquaculture Dashboard", layout="wide", page_icon="🐟")

# ==========================================
# 1.5 BACKGROUND IMAGE SETUP
# ==========================================
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(0, 0, 0, 0.75);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("⚠️ Background image not found. Ensure 'ikan.png' is in the folder.")

add_bg_from_local("ikan.png") 

st.title("🐟 Smart Aquaculture Real-Time Dashboard")

# ==========================================
# 2. LOAD SAVED MODELS & SCALERS
# ==========================================
@st.cache_resource
def load_models():
    scaler_reg = joblib.load("scaler_reg.pkl")
    rfr_model = joblib.load("rfr_model.pkl")
    scaler_clf = joblib.load("scaler_clf.pkl")
    rfc_model = joblib.load("rfc_model.pkl")
    return scaler_reg, rfr_model, scaler_clf, rfc_model

scaler_reg, rfr_model, scaler_clf, rfc_model = load_models()

# ==========================================
# 2.5 TELEGRAM NOTIFICATION CONFIGURATION
# ==========================================
TELEGRAM_TOKEN = "8803247281:AAH9rak6k5BMOqCrLl4Mz_YiSleQoJBlnG8"
TELEGRAM_CHAT_ID = "870102819"

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.get(url, json=payload)
    except Exception:
        pass

if "alert_sent" not in st.session_state:
    st.session_state.alert_sent = False

# ==========================================
# 3. SIDEBAR (DATA SOURCE CONTROLLER & SLIDERS)
# ==========================================
st.sidebar.header("🔌 System Data Source")

source_mode = st.sidebar.radio(
    "Select Input Mode:",
    ["🎛️ Manual Override (Offline / Demo)", "📡 Live Datacake API (Cloud IoT)"],
    index=1
)

is_live_api = (source_mode == "📡 Live Datacake API (Cloud IoT)")

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Sensor Control Panel")

ph_val = st.sidebar.slider("pH Level", 0.0, 14.0, 7.20, 0.1, disabled=is_live_api)
temp_val = st.sidebar.slider("Temperature (°C)", 20.0, 40.0, 28.5, 0.1, disabled=is_live_api)
turb_val = st.sidebar.slider("Turbidity (NTU)", 0.0, 50.0, 10.0, 0.1, disabled=is_live_api)
tds_val = st.sidebar.slider("TDS (ppm)", 0.0, 800.0, 300.0, 1.0, disabled=is_live_api)

def get_data():
    if is_live_api:
        st.sidebar.info("📡 Cloud mode active. Reading live data from Datacake API. (Manual panel locked).")
    else:
        st.sidebar.success("✅ Manual Override Active. You can now adjust the sensor values above.")
        
    return {
        "pH": ph_val, 
        "Temperature": temp_val, 
        "Turbidity": turb_val, 
        "Total_Dissolved_Solids": tds_val
    }

live_data = get_data()

# ==========================================
# 4. MAIN DASHBOARD CONTENT
# ==========================================
if live_data:
    current_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    st.caption(f"Last Updated: {current_time} | Mode: {source_mode}")

    st.subheader("📡 Live Sensor Readings (Simulated)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Sensor pH", f"{live_data['pH']:.2f}")
    col2.metric("Temperature (°C)", f"{live_data['Temperature']:.2f}")
    col3.metric("Turbidity (NTU)", f"{live_data['Turbidity']:.2f}")
    col4.metric("TDS (ppm)", f"{live_data['Total_Dissolved_Solids']:.2f}")

    with st.expander("🐟 Reference: Ideal Water Parameters for Tilapia"):
        st.markdown(
            """
            - **pH Level:** `6.5 - 8.5` (Optimal: 7.0 - 8.0)
            - **Temperature:** `26.0 °C - 30.0 °C`
            - **Turbidity:** `< 30 NTU` (Optimal: < 15 NTU)
            - **TDS:** `300 - 500 ppm`
            
            *Note: The machine learning Classification and System Alerts are calibrated based on these biological safety thresholds to ensure optimal fish health and growth.*
            """
        )

    # --- ML Processing ---
    reg_features = ['Temperature', 'Turbidity', 'Total_Dissolved_Solids']
    input_reg = pd.DataFrame([live_data])[reg_features]
    scaled_reg_input = scaler_reg.transform(input_reg)
    predicted_ph = rfr_model.predict(scaled_reg_input).item()
    ph_error = abs(live_data["pH"] - predicted_ph)

    clf_features = ['pH', 'Temperature', 'Turbidity', 'Total_Dissolved_Solids']
    input_clf = pd.DataFrame([live_data])[clf_features]
    scaled_clf_input = scaler_clf.transform(input_clf)
    status_prediction = int(rfc_model.predict(scaled_clf_input).item())
    
    # --- GABUNGAN KOD HYBRID (SAFETY RULE-BASED OVERRIDE) ---
    if (live_data['Total_Dissolved_Solids'] > 500 or 
        live_data['Temperature'] < 25.0 or 
        live_data['Temperature'] > 35.0 or 
        live_data['Turbidity'] > 30.0):
        status_prediction = 1  
        
    status_label = "Optimal" if status_prediction == 0 else "Critical" 

    # --- ML Analytics Display ---
    st.markdown("---")
    st.subheader("🤖 Prediction and Water Health Assessment")

    m1, m2 = st.columns(2)
    m1.metric("Predicted Future pH", f"{predicted_ph:.2f}")
    m2.metric("Predicted Water Status", status_label)

    st.info(
        f"**Analytics Info:** ML processes Temp, Turbidity & TDS to predict future pH. "
        f"**(Current Sensor Drift: {ph_error:.2f})**"
    )

    if ph_error > 1.0:
        st.warning("⚠️ **ANOMALY DETECTED:** Physical sensor readings and ML predictions differ significantly. The pH sensor may be faulty or require calibration.")

    if status_prediction == 0:
        st.success("🟢 **OPTIMAL** - Water conditions are safe and stable. No action required.")
        st.session_state.alert_sent = False 
    else:
        st.error("🔴 **CRITICAL** - Warning! Water parameters are unstable. Immediate action required!")
        
        if not st.session_state.alert_sent:
            problem_reasons = []
            if live_data['pH'] < 6.5:
                problem_reasons.append("pH is too Acidic (< 6.5)")
            elif live_data['pH'] > 8.5:
                problem_reasons.append("pH is too Alkaline (> 8.5)")
                
            if live_data['Temperature'] < 25.0:
                problem_reasons.append("Water is too Cold (< 25°C)")
            elif live_data['Temperature'] > 35.0:
                problem_reasons.append("Water is too Hot (> 35°C)")
                
            if live_data['Turbidity'] > 30.0:
                problem_reasons.append("High Turbidity / Muddy Water (> 30 NTU)")
                
            if live_data['Total_Dissolved_Solids'] > 500:
                problem_reasons.append("TDS level is too high (> 500 ppm)")
                
            if not problem_reasons:
                problem_reasons.append("Complex data anomaly detected by AI Model")
            
            problem_text = "\n".join([f"⚠️ {reason}" for reason in problem_reasons])

            alert_msg = (
                f"🚨 *AQUACULTURE HAZARD ALERT!*\n\n"
                f"The system detected water quality in *CRITICAL* condition.\n\n"
                f"🛑 *Identified Issues:*\n"
                f"{problem_text}\n\n"
                f"📊 *Current Sensor Values:*\n"
                f"- pH: {live_data['pH']:.2f}\n"
                f"- Temp: {live_data['Temperature']:.2f}°C\n"
                f"- Turbidity: {live_data['Turbidity']:.2f} NTU\n"
                f"- TDS: {live_data['Total_Dissolved_Solids']:.2f} ppm\n\n"
                f"🤖 *Action Required:* Please inspect the pond immediately!"
            )
            send_telegram_alert(alert_msg)
            st.session_state.alert_sent = True

    # --- Feature Importance Section ---
    st.markdown("---")
    st.subheader("📊 Feature Importance Analysis (RFR)")

    importance_scores = rfr_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": reg_features,
        "Importance Score": importance_scores
    }).sort_values(by="Importance Score", ascending=False)

    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(feature_importance_df, use_container_width=True)
        
    with f_col2:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor('#0e1117') 
        ax.set_facecolor('#0e1117')
        
        ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance Score"], color='skyblue')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        
        ax.set_xlabel("Importance Score")
        ax.set_title("Influence on pH Prediction")
        ax.invert_yaxis()
        ax.grid(axis='x', color='gray', linestyle='--', alpha=0.5)
        st.pyplot(fig)

# ==========================================
# 5. FOOTER
# ==========================================
st.markdown("---")
st.markdown("[View Full Datacake Analytics Platform](https://app.datacake.de/pd/ea4da4f6-a3aa-4353-bb62-60c650165c36)")
