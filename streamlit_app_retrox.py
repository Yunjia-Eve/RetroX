# =====================================================
# 🌿 RetroX Toolkit – Streamlit Dashboard (v4.3)
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------
# 1️⃣ Building & Baseline
# -----------------------------------------------------
GFA = 939.62
RoofA, WallA, WinA = 939.62, 397.7, 214.15
total_wall_roof = RoofA + WallA
BASELINE = {
    "Lighting_kWh": 44637.14,
    "Cooling_kWh": 105351.5,
    "Room_kWh": 31598.3,
    "Cooling_Load_kWh": 632108.96,
}
BASELINE["Total_kWh"] = BASELINE["Lighting_kWh"] + BASELINE["Cooling_kWh"] + BASELINE["Room_kWh"]
BASELINE["EUI"] = 193.26

# -----------------------------------------------------
# 2️⃣ Load surrogate models
# -----------------------------------------------------
st.sidebar.header("🧠 Model Selection")
model_options = ["Linear Regression (LR)", "Random Forest (RF)", "XGBoost (XGB)"]
prefix_map = {"Linear Regression (LR)": "LR", "Random Forest (RF)": "RF", "XGBoost (XGB)": "XGB"}
default = {"Lighting_kWh": "XGBoost (XGB)", "Cooling_kWh": "Linear Regression (LR)", "Cooling_Load_kWh": "Linear Regression (LR)"}

lighting_choice = st.sidebar
