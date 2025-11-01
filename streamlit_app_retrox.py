# =====================================================
# 🌿 RetroX Toolkit – Streamlit Dashboard (v4.1 fixed)
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

lighting_choice = st.sidebar.selectbox("Lighting model", model_options, index=model_options.index(default["Lighting_kWh"]))
cooling_choice  = st.sidebar.selectbox("Cooling model", model_options,  index=model_options.index(default["Cooling_kWh"]))
coolload_choice = st.sidebar.selectbox("Cooling Load model", model_options, index=model_options.index(default["Cooling_Load_kWh"]))

models = {
    "Lighting_kWh": joblib.load(f"{prefix_map[lighting_choice]}_Lighting_kWh_model.pkl"),
    "Cooling_kWh":  joblib.load(f"{prefix_map[cooling_choice]}_Cooling_kWh_model.pkl"),
    "Cooling_Load_kWh": joblib.load(f"{prefix_map[coolload_choice]}_Cooling_Load_kWh_model.pkl"),
}

# -----------------------------------------------------
# 3️⃣ User inputs
# -----------------------------------------------------
st.sidebar.header("🏗️ Building Inputs")
glazing  = st.sidebar.selectbox("Glazing Type", ["Single", "Double", "Low-E"])
insul    = st.sidebar.selectbox("Insulation", ["Low", "Med", "High"])
LPD      = st.sidebar.slider("Lighting Power Density (W/m²)", 8.0, 14.0, 10.0)
hvac     = st.sidebar.slider("HVAC Setpoint (°C)", 24.0, 27.0, 25.0)
shading  = st.sidebar.slider("Shading Depth (m)", 0.0, 1.0, 0.5)
schedule = st.sidebar.radio("Schedule Adjustment", ["Base", "Adjusted"])
ctrl     = st.sidebar.radio("Linear Control", ["No", "Yes"])
albedo   = st.sidebar.radio("High-Albedo Wall/Roof", ["Base", "Cool"])

# -----------------------------------------------------
# 4️⃣ Cost settings
# -----------------------------------------------------
st.sidebar.header("💰 Tariff & Cost Settings")
def flexible_input(label, refs, default=1):
    mode = st.sidebar.radio(f"{label} Input", ["Reference", "Custom"], key=label)
    if mode == "Reference":
        return st.sidebar.selectbox(f"{label} Reference (SGD/unit)", refs, index=default)
    else:
        return st.sidebar.number_input(f"Custom {label} (SGD/unit)", value=float(refs[default]), step=5.0)

tariff         = flexible_input("Electricity Tariff (kWh)", [0.30,0.35,0.40])
carbon_factor  = flexible_input("Carbon Factor (kgCO₂/kWh)", [0.40,0.43,0.45])
glazing_cost_double = flexible_input("Glazing Double", [150,200,250])
glazing_cost_lowe   = flexible_input("Glazing Low-E", [250,300,350])
insul_cost_med      = flexible_input("Insulation Med", [35,45,55])
insul_cost_high     = flexible_input("Insulation High", [45,55,65])
shading_cost        = flexible_input("Shading", [100,120,150])
led_cost            = flexible_input("LED", [20,25,30])
hvac_cost           = flexible_input("HVAC", [1500,2000,2500])
albedo_cost         = flexible_input("High Albedo", [20,25,30])
schedule_cost       = flexible_input("Schedule Adjustment", [1500,2000,2500])
linearctrl_cost     = flexible_input("Linear Control", [25,30,35])

# -----------------------------------------------------
# 5️⃣ CAPEX calculation (fixed)
# -----------------------------------------------------
CAPEX = 0
if glazing == "Double": CAPEX += glazing_cost_double * WinA
elif glazing == "Low-E": CAPEX += glazing_cost_lowe * WinA
if insul == "Med": CAPEX += insul_cost_med * total_wall_roof
elif insul == "High": CAPEX += insul_cost_high * total_wall_roof
if shading > 0: CAPEX += shading_cost * WinA
if LPD < 10: CAPEX += led_cost * GFA
if hvac > 24: CAPEX += hvac_cost
if albedo == "Cool": CAPEX += albedo_cost * (RoofA + WallA)
# ✅ Added missing ones
if schedule == "Adjusted": CAPEX += schedule_cost
if ctrl == "Yes": CAPEX += linearctrl_cost * GFA

# -----------------------------------------------------
# 6️⃣ Model input (consistent with training)
# -----------------------------------------------------
feature_names = [
    "LPD_Wm2","HVAC_Setpoint_C","ShadingDepth_m",
    "Glazing_Low-E","Glazing_Single",
    "Insulation_Low","Insulation_Medium",
    "ScheduleAdj_Base","LinearControl_Yes","HighAlbedoWall_Cool"
]
X_input = pd.DataFrame(np.zeros((1,len(feature_names))), columns=feature_names)
X_input["LPD_Wm2"] = LPD
X_input["HVAC_Setpoint_C"] = hvac
X_input["ShadingDepth_m"] = shading
X_input["Glazing_Low-E"] = 1 if glazing=="Low-E" else 0
X_input["Glazing_Single"] = 1 if glazing=="Single" else 0   # Double → both 0
X_input["Insulation_Low"] = 1 if insul=="Low" else 0
X_input["Insulation_Medium"] = 1 if insul=="Med" else 0
X_input["ScheduleAdj_Base"] = 1 if schedule=="Base" else 0
X_input["LinearControl_Yes"] = 1 if ctrl=="Yes" else 0
X_input["HighAlbedoWall_Cool"] = 1 if albedo=="Cool" else 0

# -----------------------------------------------------
# 7️⃣ Predictions
# -----------------------------------------------------
lighting_pred = models["Lighting_kWh"].predict(X_input)[0]
cooling_pred  = models["Cooling_kWh"].predict(X_input)[0]
cool_load_pred= models["Cooling_Load_kWh"].predict(X_input)[0]
room_elec = BASELINE["Room_kWh"]
total_energy = lighting_pred + cooling_pred + room_elec

energy_saving_pct = (BASELINE["Total_kWh"]-total_energy)/BASELINE["Total_kWh"]*100
cool_saving_pct   = (BASELINE["Cooling_Load_kWh"]-cool_load_pred)/BASELINE["Cooling_Load_kWh"]*100
EUI = total_energy / GFA
carbon_emission = total_energy * carbon_factor
annual_saving = (BASELINE["Total_kWh"]-total_energy)*tariff
payback_years = CAPEX/annual_saving if annual_saving>0 else None

# -----------------------------------------------------
# 8️⃣ Tabs – Energy / Environment / Economics
# -----------------------------------------------------
st.title("🌿 RetroX Surrogate Toolkit v4.1")

tabs = st.tabs(["⚡ Energy","🌍 Environment","💰 Economics"])

# ENERGY TAB
with tabs[0]:
    st.subheader("⚡ Energy Breakdown vs Baseline")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Lighting (kWh)",f"{lighting_pred:,.0f}")
    col2.metric("Cooling (kWh)",f"{cooling_pred:,.0f}")
    col3.metric("Room Elec (kWh)",f"{room_elec:,.0f}")
    col4.metric("Total (kWh)",f"{total_energy:,.0f}")
    st.metric("Energy Saving (%)",f"{energy_saving_pct:.1f}%")
    st.metric("EUI (kWh/m²·yr)",f"{EUI:.2f}")
    st.metric("Cooling Load Saving (%)",f"{cool_saving_pct:.1f}%")

# ENVIRONMENT TAB
with tabs[1]:
    st.subheader("🌍 Environmental KPIs")
    st.metric("Carbon Emission (kg CO₂e)",f"{carbon_emission:,.1f}")
    st.metric("Carbon Factor (kgCO₂/kWh)",f"{carbon_factor:.2f}")

# ECONOMIC TAB
with tabs[2]:
    st.subheader("💰 Economic KPIs")
    Retrofit_Cost_Capex = CAPEX
    Annual_Cost_Saving = annual_saving
    Payback_Period = Payback_Years = payback_years

    col1,col2,col3 = st.columns(3)
    col1.metric("Retrofit Cost (Capex, SGD)",f"{Retrofit_Cost_Capex:,.0f}")
    col2.metric("Annual Cost Saving (SGD)",f"{Annual_Cost_Saving:,.0f}")
    col3.metric("Payback Period (years)",f"{Payback_Period:.1f}" if Payback_Period else "—")

    st.markdown("<p style='font-size:15px;'><b>Economic KPI Formulas</b></p>", unsafe_allow_html=True)
    st.latex(r"\text{Capex}=\sum_i C_i")
    st.latex(r"\text{Annual Cost Saving}=E_{\text{saving}}\times T_{\text{elec}}")
    st.latex(r"\text{Payback}=\frac{\text{Capex}}{\text{Annual Cost Saving}}")

# -----------------------------------------------------
# 9️⃣ Summary message
# -----------------------------------------------------
msg = f"Your building achieves **{energy_saving_pct:.1f}% energy saving** with a payback of **{payback_years:.1f} years**."
if (EUI<120) or (energy_saving_pct>=35): msg += " 🏆 Green Mark Platinum achieved!"
elif (EUI<135) or (energy_saving_pct>=30): msg += " 🥇 Green Mark Gold achieved!"
st.info(msg)
