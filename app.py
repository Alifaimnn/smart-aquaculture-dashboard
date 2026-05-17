import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import base64
import requests  # Diperlukan untuk mengirim data ke API Telegram

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Smart Aquaculture Dashboard", layout="wide", page_icon="🐟")

# ==========================================
# 1.5 BACKGROUND IMAGE SETUP
# ==========================================
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
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
            background-color: rgba(0, 0, 0, 0.7); /* Overlay gelap */
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("⚠️ Gambar background tidak ditemukan. Pastikan nama fail betul dan berada di dalam folder yang sama.")

# MASUKKAN NAMA FILE GAMBAR BACKGROUND ANDA DI SINI
add_bg_from_local("ikan.jpg") 

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
# MASUKKAN TOKEN BOT DAN CHAT ID ANDA DI SINI
TELEGRAM_TOKEN = "8803247281:AAH9rak6k5BMOqCrLl4Mz_YiSleQoJBlnG8"
TELEGRAM_CHAT_ID = "870102819"

def send_telegram_alert(message):
    if TELEGRAM_TOKEN != "MASUKKAN_TOKEN_BOT_TELEGRAM_DI_SINI":
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.get(url, json=payload)
        except Exception as e:
            pass

# Inisialisasi variabel anti-spam di Streamlit Memory
if "alert_sent" not in st.session_state:
    st.session_state.alert_sent = False

# ==========================================
# 3. SIDEBAR (SIMULASI DATA INTERAKTIF)
# ==========================================
st.sidebar.header("🎛️ Sensor Control Panel")
st.sidebar.markdown("Ubah nilai di bawah untuk menguji respon Machine Learning secara *real-time*.")

def get_simulated_data():
    ph_val = st.sidebar.slider("pH Level", 0.0, 14.0, 7.20, 0.1)
    temp_val = st.sidebar.slider("Temperature (°C)", 20.0, 40.0, 28.5, 0.1)
    turb_val = st.sidebar.slider("Turbidity (NTU)", 0.0, 50.0, 10.0, 0.1)
    tds_val = st.sidebar.slider("TDS (ppm)", 0.0, 800.0, 300.0, 1.0)
    
    return {
        "pH": ph_val, 
        "Temperature": temp_val, 
        "Turbidity": turb_val, 
        "Total_Dissolved_Solids": tds_val
    }

live_data = get_simulated_data()

# ==========================================
# 4. LIVE SENSOR READINGS METRICS
# ==========================================
if live_data:
    current_time = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
    st.caption(f"Last Updated: {current_time}")

    st.subheader("📡 Live Sensor Readings (Simulated)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Sensor pH", f"{live_data['pH']:.2f}")
    col2.metric("Temperature (°C)", f"{live_data['Temperature']:.2f}")
    col3.metric("Turbidity (NTU)", f"{live_data['Turbidity']:.2f}")
    col4.metric("TDS (ppm)", f"{live_data['Total_Dissolved_Solids']:.2f}")

    # ==========================================
    # 5. MACHINE LEARNING PROCESSING
    # ==========================================
    # --- Regression (Predicting pH) ---
    reg_features = ['Temperature', 'Turbidity', 'Total_Dissolved_Solids']
    input_reg = pd.DataFrame([live_data])[reg_features]
    scaled_reg_input = scaler_reg.transform(input_reg)
    
    predicted_ph = rfr_model.predict(scaled_reg_input).item()
    ph_error = abs(live_data["pH"] - predicted_ph)

    # --- Classification (Optimal vs Critical) ---
    clf_features = ['pH', 'Temperature', 'Turbidity', 'Total_Dissolved_Solids']
    input_clf = pd.DataFrame([live_data])[clf_features]
    scaled_clf_input = scaler_clf.transform(input_clf)
    
    status_prediction = int(rfc_model.predict(scaled_clf_input).item())
    status_label = "Optimal" if status_prediction == 0 else "Critical"

    if hasattr(rfc_model, "predict_proba"):
        confidence = float(np.max(rfc_model.predict_proba(scaled_clf_input)) * 100)
    else:
        confidence = None

    # ==========================================
    # 6. ML PREDICTION & HEALTH ASSESSMENT
    # ==========================================
    st.markdown("---")
    st.subheader("🤖 Prediction and Water Health Assessment")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Future pH", f"{predicted_ph:.2f}")
    m2.metric("Predicted Water Status", status_label)
    if confidence is not None:
        m3.metric("Model Confidence", f"{confidence:.2f}%")
    else:
        m3.metric("Model Confidence", "N/A")

    st.info(
        f"**Analytics Info:** ML memproses Temp, Turbidity & TDS untuk meramal pH pada masa hadapan. "
        f"**(Current Drift: {ph_error:.2f})**"
    )

    if ph_error > 1.0:
        st.warning("⚠️ **ANOMALY DETECTED:** Bacaan sensor fizikal dan ramalan ML sangat berbeza. Sensor pH mungkin rosak.")

    # Logik Peringatan Keselamatan & Notifikasi Telegram
    if status_prediction == 0:
        st.success("🟢 **OPTIMAL** - Water conditions are safe and stable. No action required.")
        st.session_state.alert_sent = False # Reset status peringatan jika kondisi air kembali normal
    else:
        st.error("🔴 **CRITICAL** - Warning! Water parameters are unstable. Immediate action required!")
        
        # Kirim Telegram jika belum dikirim sebelumnya
        if not st.session_state.alert_sent:
            alert_msg = (
                f"🚨 *PERINGATAN BAHAYA AKUAKULTUR!*\n\n"
                f"Sistem mendeteksi kualitas air dalam kondisi *CRITICAL*.\n"
                f"📊 *Detail Sensor:*\n"
                f"- pH: {live_data['pH']:.2f}\n"
                f"- Suhu: {live_data['Temperature']:.2f}°C\n"
                f"- Kekeruhan: {live_data['Turbidity']:.2f} NTU\n"
                f"- TDS: {live_data['Total_Dissolved_Solids']:.2f} ppm\n\n"
                f"🤖 *Prediksi ML:* Status Air Tidak Stabil!"
            )
            send_telegram_alert(alert_msg)
            st.session_state.alert_sent = True # Tandai bahwa pesan sudah dikirim

    # ==========================================
    # 7. FEATURE IMPORTANCE ANALYSIS
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Feature Importance Analysis")

    importance_scores = rfr_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": reg_features,
        "Importance Score": importance_scores
    }).sort_values(by="Importance Score", ascending=False)

    # Menggunakan pembagian kolum sederhana (angka 2) untuk menghindari ralat spec
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.dataframe(feature_importance_df, use_container_width=True)
        
    with f_col2:
        fig, ax = plt.subplots(figsize=(8, 4))
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

st.markdown("---")
st.markdown("[View Full Datacake Analytics Platform](https://app.datacake.de/pd/ea4da4f6-a3aa-4353-bb62-60c650165c36)")
