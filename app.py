import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Smart Aquaculture Dashboard", layout="wide")
st.title("🐟 Smart Aquaculture Real-Time Dashboard")

# 2. Load 4 fail berbeza (Skaler & Model)
@st.cache_resource
def load_models():
    scaler_reg = joblib.load("scaler_reg.pkl")
    rfr_model = joblib.load("rfr_model.pkl")
    scaler_clf = joblib.load("scaler_clf.pkl")
    rfc_model = joblib.load("rfc_model.pkl")
    return scaler_reg, rfr_model, scaler_clf, rfc_model

scaler_reg, rfr_model, scaler_clf, rfc_model = load_models()

# 3. Fetch Data (Simulasi)
def get_datacake_data():
    try:
        # Dummy data untuk simulasi
        return {"pH": 10.0, "Temperature": 28.5, "Turbidity": 10.0, "Total_Dissolved_Solids": 30.0}
    except Exception as e:
        st.error("Failed to connect to Datacake API.")
        return None

live_data = get_datacake_data()

if live_data:
    st.subheader("📡 Live Sensor Readings (Datacake)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("pH Level (Sensor)", live_data["pH"])
    col2.metric("Temperature (°C)", live_data["Temperature"])
    col3.metric("Turbidity (NTU)", live_data["Turbidity"])
    col4.metric("TDS (ppm)", live_data["Total_Dissolved_Solids"])
    
    # ---------------------------------------------------------
    # RAMALAN pH (REGRESSION) - Guna 3 Parameter
    # ---------------------------------------------------------
    reg_features = ['Temperature', 'Turbidity', 'Total_Dissolved_Solids']
    input_reg = pd.DataFrame([live_data])[reg_features]
    scaled_reg_input = scaler_reg.transform(input_reg)
    
    # PENYELESAIAN RALAT: Guna .item() untuk cabut nombor dari array
    predicted_ph = rfr_model.predict(scaled_reg_input).item()

    # ---------------------------------------------------------
    # STATUS KESIHATAN AIR (CLASSIFICATION) - Guna 4 Parameter
    # ---------------------------------------------------------
    clf_features = ['pH', 'Temperature', 'Turbidity', 'Total_Dissolved_Solids']
    input_clf = pd.DataFrame([live_data])[clf_features]
    scaled_clf_input = scaler_clf.transform(input_clf)
    
    # PENYELESAIAN RALAT: Guna .item() untuk cabut nombor dari array
    status_prediction = int(rfc_model.predict(scaled_clf_input).item())
    
    # ---------------------------------------------------------
    # PAPARAN ANALITIK
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 Machine Learning Analytics")
    
    # Memaparkan nilai ramalan pH (Kini tidak akan ralat lagi)
    st.info(f"**Predicted pH Value:** `{predicted_ph:.2f}` *(Based on current Temp, Turbidity & TDS)*")
    
    if status_prediction == 0:
        st.success("🟢 **OPTIMAL (Class 0)** - Water conditions are safe and stable. No action required.")
    else:
        st.error("🔴 **CRITICAL (Class 1)** - Warning! Water parameters are unstable. Immediate action required!")

st.markdown("---")
st.markdown("[View Full Datacake Analytics Platform](https://app.datacake.de/pd/ea4da4f6-a3aa-4353-bb62-60c650165c36)")