# =========================
# prediksi_app.py
# =========================

import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import shap

# =========================
# Load model
# =========================
try:
    model_weekday = joblib.load(model_weekday.pkl")
    model_weekend = joblib.load(model_weekend.pkl")
except FileNotFoundError:
    st.error("File model_weekday.pkl atau model_weekend.pkl tidak ditemukan. Pastikan sudah disimpan di folder yang sama.")
    st.stop()

st.title("Prediksi Volume Kendaraan dengan Feature Importance Lokal")

# =========================
# Pilih jenis hari
# =========================
day_type = st.radio("Pilih jenis hari:", ("Weekday", "Weekend"))

# =========================
# Input Manual
# =========================
if day_type == "Weekday":
    data_input = pd.DataFrame([{
        'MC_Vehicle_kend/15menit_Weekday': st.number_input("MC Vehicle/15menit (Weekday)", value=0.0),
        'LV_Vehicle_kend/15menit_Weekday': st.number_input("LV Vehicle/15menit (Weekday)", value=0.0),
        'HV_Vehicle_kend/15menit_Weekday': st.number_input("HV Vehicle/15menit (Weekday)", value=0.0),
        'Spot_Speed_Weekday': st.number_input("Spot Speed Weekday", value=0.0),
        'Space_Mean_Speed_Weekday': st.number_input("Space Mean Speed Weekday", value=0.0),
        'Peak_Flag_Weekday': st.number_input("Peak Flag Weekday (0/1)", min_value=0, max_value=1, value=0),
        'RoadSideActivity_Pedestrian_Weekday': st.number_input("Pedestrian Activity Weekday", value=0.0),
        'RoadSideActivity_ParkStop_Weekday': st.number_input("Park/Stop Activity Weekday", value=0.0),
        'RoadSideActivity_ExitEntry_Weekday': st.number_input("Exit/Entry Activity Weekday", value=0.0),
        'RoadSideActivity_NonMotor_Weekday': st.number_input("NonMotor Activity Weekday", value=0.0)
    }])
    model = model_weekday
    X_train = pd.DataFrame(model.feature_importances_).T  # placeholder untuk SHAP explainer

else:  # Weekend
    data_input = pd.DataFrame([{
        'MC_Vehicle_kend/15menit_Weekend': st.number_input("MC Vehicle/15menit (Weekend)", value=0.0),
        'LV_Vehicle_kend/15menit_Weekend': st.number_input("LV Vehicle/15menit (Weekend)", value=0.0),
        'HV_Vehicle_kend/15menit_Weekend': st.number_input("HV Vehicle/15menit (Weekend)", value=0.0),
        'Spot_Speed_Weekend': st.number_input("Spot Speed Weekend", value=0.0),
        'Space_Mean_Speed_Weekend': st.number_input("Space Mean Speed Weekend", value=0.0),
        'Peak_Flag_Weekend': st.number_input("Peak Flag Weekend (0/1)", min_value=0, max_value=1, value=0),
        'RoadSideActivity_Pedestrian_Weekend': st.number_input("Pedestrian Activity Weekend", value=0.0),
        'RoadSideActivity_ParkStop_Weekend': st.number_input("Park/Stop Activity Weekend", value=0.0),
        'RoadSideActivity_ExitEntry_Weekend': st.number_input("Exit/Entry Activity Weekend", value=0.0),
        'RoadSideActivity_NonMotor_Weekend': st.number_input("NonMotor Activity Weekend", value=0.0)
    }])
    model = model_weekend
    X_train = pd.DataFrame(model.feature_importances_).T  # placeholder untuk SHAP explainer

# =========================
# Prediksi
# =========================
if st.button("Prediksi"):
    pred = model.predict(data_input)[0]
    st.success(f"Prediksi Total Vehicle Count ({day_type}): {pred:.0f}")

    # =========================
    # Feature Importance Lokal (SHAP)
    # =========================
    explainer = shap.Explainer(model, X_train)  # X_train sebaiknya data training asli
    shap_values = explainer(data_input)

    st.subheader("Feature Importance Lokal (SHAP)")
    shap_df = pd.DataFrame({
        "Feature": data_input.columns,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False)
    st.dataframe(shap_df)

    st.subheader("Visualisasi SHAP (Waterfall)")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight')  # untuk menampilkan chart SHAP

