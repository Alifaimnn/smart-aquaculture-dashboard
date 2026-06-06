import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import base64
import requests

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Aquaculture Dashboard",
    layout="wide",
    page_icon="🐟"
)

# ============================================================
# 2. BACKGROUND IMAGE SETUP
# ============================================================

def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{encoded_string});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.75);
                z-index: -1;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    except FileNotFoundError:
        st.warning("⚠️ Background image not found. Ensure 'ikan.png' is in the same folder.")

add_bg_from_local("ikan.png")

st.title("🐟 Smart Aquaculture Real-Time Dashboard")

# ============================================================
# 3. LOAD SAVED MODELS & METADATA
# ============================================================

@st.cache_resource
def load_models():
    scaler_reg = joblib.load("scaler_reg.pkl")
    rfr_model = joblib.load("rfr_model.pkl")

    scaler_clf = joblib.load("scaler_clf.pkl")
    rfc_model = joblib.load("rfc_model.pkl")

    try:
        model_metadata = joblib.load("model_metadata.pkl")
    except Exception:
        model_metadata = {}

    return scaler_reg, rfr_model, scaler_clf, rfc_model, model_metadata

scaler_reg, rfr_model, scaler_clf, rfc_model, model_metadata = load_models()

# ============================================================
# 4. THRESHOLDS
# ============================================================

classification_thresholds = model_metadata.get(
    "classification_thresholds",
    {
        "ph_min": 6.5,
        "ph_max": 8.5,
        "temp_min": 25.0,
        "temp_max": 32.0,
        "turbidity_max": 30.0,
        "tds_max": 500.0
    }
)

emergency_thresholds = model_metadata.get(
    "emergency_thresholds",
    {
        "ph_min": 6.5,
        "ph_max": 8.5,
        "temp_min": 25.0,
        "temp_max": 35.0,
        "turbidity_max": 30.0,
        "tds_max": 500.0
    }
)

PH_MIN = classification_thresholds["ph_min"]
PH_MAX = classification_thresholds["ph_max"]
TEMP_MIN = classification_thresholds["temp_min"]
TEMP_MAX = classification_thresholds["temp_max"]
TURBIDITY_MAX = classification_thresholds["turbidity_max"]
TDS_MAX = classification_thresholds["tds_max"]

EMERGENCY_TEMP_MIN = emergency_thresholds["temp_min"]
EMERGENCY_TEMP_MAX = emergency_thresholds["temp_max"]
EMERGENCY_TURBIDITY_MAX = emergency_thresholds["turbidity_max"]
EMERGENCY_TDS_MAX = emergency_thresholds["tds_max"]

# ============================================================
# 5. TELEGRAM NOTIFICATION CONFIGURATION
# ============================================================
# IMPORTANT:
# Do not hardcode Telegram token in this file.
# Put it inside .streamlit/secrets.toml:
#
# TELEGRAM_TOKEN = "your_new_bot_token"
# TELEGRAM_CHAT_ID = "your_chat_id"

def get_telegram_credentials():
    try:
        token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        return token, chat_id
    except Exception:
        return None, None

TELEGRAM_TOKEN, TELEGRAM_CHAT_ID = get_telegram_credentials()

def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        st.warning("⚠️ Telegram alert not sent. TELEGRAM_TOKEN or TELEGRAM_CHAT_ID is missing in Streamlit secrets.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, data=payload, timeout=10)

        if response.status_code == 200:
            return True
        else:
            st.warning(f"⚠️ Telegram alert failed. Status code: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ Telegram alert failed: {e}")
        return False

if "alert_sent" not in st.session_state:
    st.session_state.alert_sent = False

# ============================================================
# 6. SIDEBAR: DATA SOURCE CONTROLLER & SENSOR INPUT
# ============================================================

st.sidebar.header("🔌 System Data Source")

source_mode = st.sidebar.radio(
    "Select Input Mode:",
    [
        "🎛️ Manual Override (Offline / Demo)",
        "📡 Simulated Live Mode"
    ],
    index=0
)

is_simulated_live = source_mode == "📡 Simulated Live Mode"

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Sensor Control Panel")

ph_val = st.sidebar.slider(
    "pH Level",
    min_value=0.0,
    max_value=14.0,
    value=7.20,
    step=0.1,
    disabled=is_simulated_live
)

temp_val = st.sidebar.slider(
    "Temperature (°C)",
    min_value=20.0,
    max_value=40.0,
    value=28.5,
    step=0.1,
    disabled=is_simulated_live
)

turb_val = st.sidebar.slider(
    "Turbidity (NTU)",
    min_value=0.0,
    max_value=50.0,
    value=10.0,
    step=0.1,
    disabled=is_simulated_live
)

tds_val = st.sidebar.slider(
    "TDS (ppm)",
    min_value=0.0,
    max_value=800.0,
    value=200.0,
    step=1.0,
    disabled=is_simulated_live
)

def get_data():
    if is_simulated_live:
        st.sidebar.info("📡 Simulated Live Mode active. Using fixed sample sensor values.")

        return {
            "pH": 7.20,
            "Temperature": 28.50,
            "Turbidity": 10.00,
            "Total_Dissolved_Solids": 200.00
        }

    else:
        st.sidebar.success("✅ Manual Override Active. You can adjust the sensor values above.")

        return {
            "pH": ph_val,
            "Temperature": temp_val,
            "Turbidity": turb_val,
            "Total_Dissolved_Solids": tds_val
        }

live_data = get_data()

# ============================================================
# 7. MAIN DASHBOARD CONTENT
# ============================================================

if live_data:
    current_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

    st.caption(f"Last Updated: {current_time} | Mode: {source_mode}")

    st.subheader("📡 Current Sensor Readings")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Actual Sensor pH", f"{live_data['pH']:.2f}")
    col2.metric("Temperature (°C)", f"{live_data['Temperature']:.2f}")
    col3.metric("Turbidity (NTU)", f"{live_data['Turbidity']:.2f}")
    col4.metric("TDS (ppm)", f"{live_data['Total_Dissolved_Solids']:.2f}")

    with st.expander("🐟 Reference: Ideal Water Parameters for Tilapia"):
        st.markdown(
            f"""
            - **pH Level:** `{PH_MIN} - {PH_MAX}`
            - **Temperature:** `{TEMP_MIN} °C - {TEMP_MAX} °C`
            - **Turbidity:** `< {TURBIDITY_MAX} NTU`
            - **TDS:** `≤ {TDS_MAX} ppm`

            *Note: The classifier uses these biological safety thresholds to classify water condition as Optimal or Critical.*
            """
        )

    # ========================================================
    # 8. AI ESTIMATED pH USING RFR
    # ========================================================

    reg_features = [
        "Temperature",
        "Turbidity",
        "Total_Dissolved_Solids"
    ]

    input_reg = pd.DataFrame([live_data])[reg_features]
    scaled_reg_input = scaler_reg.transform(input_reg)

    estimated_ph = float(rfr_model.predict(scaled_reg_input)[0])

    actual_ph = float(live_data["pH"])
    ph_error = abs(actual_ph - estimated_ph)

    # ========================================================
    # 9. CHOOSE pH FOR CLASSIFICATION
    # ========================================================
    # Default: use actual pH sensor
    # Backup: use AI estimated pH if actual pH is invalid or suspicious

    if (actual_ph <= 0) or (actual_ph > 14) or (ph_error > 1.0):
        ph_for_classification = estimated_ph
        ph_source = "AI Estimated pH"
    else:
        ph_for_classification = actual_ph
        ph_source = "Actual Sensor pH"

    # ========================================================
    # 10. RFC CLASSIFICATION
    # ========================================================

    clf_features = [
        "pH",
        "Temperature",
        "Turbidity",
        "Total_Dissolved_Solids"
    ]

    input_clf = pd.DataFrame([{
        "pH": ph_for_classification,
        "Temperature": live_data["Temperature"],
        "Turbidity": live_data["Turbidity"],
        "Total_Dissolved_Solids": live_data["Total_Dissolved_Solids"]
    }])[clf_features]

    scaled_clf_input = scaler_clf.transform(input_clf)

    status_prediction = int(rfc_model.predict(scaled_clf_input)[0])

    # ========================================================
    # 11. HYBRID RULE-BASED OVERRIDE
    # ========================================================
    # This fail-safe forces Critical status during extreme biological risk.

    if (
        ph_for_classification < PH_MIN or
        ph_for_classification > PH_MAX or
        live_data["Temperature"] < EMERGENCY_TEMP_MIN or
        live_data["Temperature"] > EMERGENCY_TEMP_MAX or
        live_data["Turbidity"] > EMERGENCY_TURBIDITY_MAX or
        live_data["Total_Dissolved_Solids"] > EMERGENCY_TDS_MAX
    ):
        status_prediction = 1

    status_label = "Optimal" if status_prediction == 0 else "Critical"

    # ========================================================
    # 12. AI ANALYTICS DISPLAY
    # ========================================================

    st.markdown("---")
    st.subheader("🤖 AI Water Quality Assessment")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Actual Sensor pH", f"{actual_ph:.2f}")
    m2.metric("AI Estimated pH", f"{estimated_ph:.2f}")
    m3.metric("pH Used for RFC", f"{ph_for_classification:.2f}")
    m4.metric("Water Status", status_label)

    st.caption(f"Classification pH source: **{ph_source}**")

    if ph_error > 1.0:
        st.warning(
            "⚠️ Possible pH sensor anomaly detected. "
            "The actual pH sensor reading and AI estimated pH differ significantly."
        )

    # ========================================================
    # 13. STATUS MESSAGE + TELEGRAM ALERT
    # ========================================================

    if status_prediction == 0:
        st.success("🟢 **OPTIMAL** - Water conditions are safe and stable. No action required.")
        st.session_state.alert_sent = False

    else:
        st.error("🔴 **CRITICAL** - Warning! Water parameters are unstable. Immediate action required.")

        if not st.session_state.alert_sent:
            problem_reasons = []

            # pH issue based on pH used by classifier
            if ph_for_classification < PH_MIN:
                problem_reasons.append(f"pH is too acidic ({ph_for_classification:.2f} < {PH_MIN})")
            elif ph_for_classification > PH_MAX:
                problem_reasons.append(f"pH is too alkaline ({ph_for_classification:.2f} > {PH_MAX})")

            # Sensor anomaly issue
            if ph_error > 1.0:
                problem_reasons.append(
                    f"Possible pH sensor anomaly detected "
                    f"(Actual: {actual_ph:.2f}, Estimated: {estimated_ph:.2f})"
                )

            # Temperature issue
            if live_data["Temperature"] < EMERGENCY_TEMP_MIN:
                problem_reasons.append(f"Water temperature is too cold (< {EMERGENCY_TEMP_MIN}°C)")
            elif live_data["Temperature"] > EMERGENCY_TEMP_MAX:
                problem_reasons.append(f"Water temperature is too hot (> {EMERGENCY_TEMP_MAX}°C)")

            # Turbidity issue
            if live_data["Turbidity"] > EMERGENCY_TURBIDITY_MAX:
                problem_reasons.append(f"High turbidity detected (> {EMERGENCY_TURBIDITY_MAX} NTU)")

            # TDS issue
            if live_data["Total_Dissolved_Solids"] > EMERGENCY_TDS_MAX:
                problem_reasons.append(f"Dangerous TDS level detected (> {EMERGENCY_TDS_MAX} ppm)")

            if not problem_reasons:
                problem_reasons.append("Critical condition detected by RFC model")

            problem_text = "\n".join([f"⚠️ {reason}" for reason in problem_reasons])

            alert_msg = (
                f"🚨 *AQUACULTURE HAZARD ALERT!*\n\n"
                f"The system detected water quality in *CRITICAL* condition.\n\n"
                f"🛑 *Identified Issues:*\n"
                f"{problem_text}\n\n"
                f"📊 *Current Sensor Values:*\n"
                f"- Actual pH: {actual_ph:.2f}\n"
                f"- AI Estimated pH: {estimated_ph:.2f}\n"
                f"- pH Used for Classification: {ph_for_classification:.2f} ({ph_source})\n"
                f"- Temperature: {live_data['Temperature']:.2f}°C\n"
                f"- Turbidity: {live_data['Turbidity']:.2f} NTU\n"
                f"- TDS: {live_data['Total_Dissolved_Solids']:.2f} ppm\n\n"
                f"🤖 *Action Required:* Please inspect the pond immediately and perform corrective action if required."
            )

            sent = send_telegram_alert(alert_msg)

            if sent:
                st.session_state.alert_sent = True

    # ========================================================
    # 14. FEATURE IMPORTANCE SECTION
    # ========================================================

    st.markdown("---")
    st.subheader("📊 Feature Importance Analysis (RFR)")

    try:
        importance_scores = rfr_model.feature_importances_

        feature_importance_df = pd.DataFrame({
            "Feature": reg_features,
            "Importance Score": importance_scores
        }).sort_values(by="Importance Score", ascending=False)

        f_col1, f_col2 = st.columns(2)

        with f_col1:
            st.dataframe(feature_importance_df, use_container_width=True)

        with f_col2:
            fig, ax = plt.subplots(figsize=(8, 3.5))

            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")

            ax.barh(
                feature_importance_df["Feature"],
                feature_importance_df["Importance Score"],
                color="skyblue"
            )

            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

            ax.set_xlabel("Importance Score")
            ax.set_title("Influence on pH Prediction")
            ax.invert_yaxis()
            ax.grid(axis="x", color="gray", linestyle="--", alpha=0.5)

            st.pyplot(fig)

    except Exception as e:
        st.warning(f"Feature importance could not be displayed: {e}")

# ============================================================
# 15. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    **Model Flow:**  
    Temperature + Turbidity + TDS → **AI Estimated pH**  
    pH + Temperature + Turbidity + TDS → **Optimal / Critical Classification**
    """
)

st.markdown("[View Full Datacake Analytics Platform](https://app.datacake.de/pd/ea4da4f6-a3aa-4353-bb62-60c650165c36)")
