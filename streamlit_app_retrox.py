# =====================================================
# 🌿 RetroX Toolkit – Streamlit Dashboard (v3, 2025)
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------
# 1️⃣ Fixed Building & Baseline Information
# -----------------------------------------------------
GFA = 939.62
RoofA, WallA, WinA = 939.62, 397.7, 214.15
total_wall_roof = RoofA + WallA

# ✅ Baseline Case 0 values (from your data)
BASELINE = {
    'Lighting_kWh': 44637.14,
    'Cooling_kWh': 105351.5,
    'Room_kWh': 31598.3,
    'Cooling_Load_kWh': 632108.96
}
BASELINE['Total_kWh'] = BASELINE['Lighting_kWh'] + BASELINE['Cooling_kWh'] + BASELINE['Room_kWh']
BASELINE['EUI'] = 193.26

# -----------------------------------------------------
# 2️⃣ Load Models (RF + XGB)
# -----------------------------------------------------
models = {
    'Lighting_kWh': joblib.load('XGB_Lighting_kWh_model.pkl'),
    'Cooling_kWh': joblib.load('RF_Cooling_kWh_model.pkl'),
    'Cooling_Load_kWh': joblib.load('RF_Cooling_Load_kWh_model.pkl')
}

# -----------------------------------------------------
# 3️⃣ Sidebar – Building Inputs
# -----------------------------------------------------
st.sidebar.header("🏗️ Building Inputs")

glazing = st.sidebar.selectbox("Glazing Type", ['Single', 'Double', 'Low-E'])
insul = st.sidebar.selectbox("Insulation", ['Low', 'Med', 'High'])
LPD = st.sidebar.slider("Lighting Power Density (W/m²)", 8.0, 14.0, 10.0)
hvac = st.sidebar.slider("HVAC Setpoint (°C)", 24.0, 27.0, 25.0)
shading = st.sidebar.slider("Shading Depth (m)", 0.0, 1.0, 0.5)
schedule = st.sidebar.radio("Schedule Adjustment", ['Base', 'Adjusted'])
ctrl = st.sidebar.radio("Linear Control", ['No', 'Yes'])
albedo = st.sidebar.radio("High Albedo Wall/Roof", ['Base', 'Cool'])

# -----------------------------------------------------
# 4️⃣ Sidebar – Cost & Tariff Inputs
# -----------------------------------------------------
st.sidebar.header("💰 Tariff & Cost Settings")

tariff_mode = st.sidebar.radio("Electricity Tariff Mode", ["Reference", "Custom"])
if tariff_mode == "Reference":
    tariff = st.sidebar.selectbox("Electricity Tariff (SGD/kWh)", [0.30, 0.35, 0.40])
else:
    tariff = st.sidebar.number_input("Custom Tariff (SGD/kWh)", value=0.30, step=0.01)

carbon_mode = st.sidebar.radio("Carbon Factor Mode", ["Reference", "Custom"])
if carbon_mode == "Reference":
    carbon_factor = st.sidebar.selectbox("Carbon Factor (kgCO₂/kWh)", [0.40, 0.43, 0.45])
else:
    carbon_factor = st.sidebar.number_input("Custom Carbon Factor", value=0.43, step=0.01)

# Measure cost rates
st.sidebar.markdown("### 🏷️ Measure Unit Costs")

def cost_input(label, ref_values, default_idx=1):
    mode = st.sidebar.radio(f"{label} cost mode", [f"{label} reference", "Custom"], key=label)
    if mode == f"{label} reference":
        return st.sidebar.selectbox(f"{label} (SGD/unit)", ref_values, index=default_idx, key=label+"_ref")
    else:
        return st.sidebar.number_input(f"Custom {label} cost", value=float(ref_values[default_idx]), step=5.0, key=label+"_custom")

glazing_cost_double = cost_input("Glazing (Double)", [150, 200, 250])
glazing_cost_lowe = cost_input("Glazing (Low-E)", [250, 300, 350])
insul_cost_med = cost_input("Insulation (Med)", [35, 45, 55])
insul_cost_high = cost_input("Insulation (High)", [45, 55, 65])
shading_cost = cost_input("Shading", [100, 120, 150])
led_cost = cost_input("Lighting Retrofit (LED)", [20, 25, 30])
hvac_cost = cost_input("HVAC Adjustment", [1500, 2000, 2500])
albedo_cost = cost_input("High Albedo Coating", [20, 25, 30])

# -----------------------------------------------------
# 5️⃣ Cost Calculation (Based on selections)
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

# -----------------------------------------------------
# 6️⃣ Input Preparation for Models
# -----------------------------------------------------
feature_names = ['LPD_Wm2', 'HVAC_Setpoint_C', 'ShadingDepth_m',
                 'Glazing_Low-E', 'Glazing_Single',
                 'Insulation_Low', 'Insulation_Medium',
                 'ScheduleAdj_Base', 'LinearControl_Yes',
                 'HighAlbedoWall_Cool']

X_input = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
X_input['LPD_Wm2'] = LPD
X_input['HVAC_Setpoint_C'] = hvac
X_input['ShadingDepth_m'] = shading
X_input['Glazing_Low-E'] = 1 if glazing == 'Low-E' else 0
X_input['Glazing_Single'] = 1 if glazing == 'Single' else 0
X_input['Insulation_Low'] = 1 if insul == 'Low' else 0
X_input['Insulation_Medium'] = 1 if insul == 'Med' else 0
X_input['ScheduleAdj_Base'] = 1 if schedule == 'Base' else 0
X_input['LinearControl_Yes'] = 1 if ctrl == 'Yes' else 0
X_input['HighAlbedoWall_Cool'] = 1 if albedo == 'Cool' else 0

# -----------------------------------------------------
# 7️⃣ Predictions
# -----------------------------------------------------
lighting_pred = models['Lighting_kWh'].predict(X_input)[0]
cooling_pred = models['Cooling_kWh'].predict(X_input)[0]
cool_load_pred = models['Cooling_Load_kWh'].predict(X_input)[0]
room_elec = BASELINE['Room_kWh']
total_energy = lighting_pred + cooling_pred + room_elec

# ✅ Compare with Baseline
energy_saving_pct = (BASELINE['Total_kWh'] - total_energy) / BASELINE['Total_kWh'] * 100
cool_saving_pct = (BASELINE['Cooling_Load_kWh'] - cool_load_pred) / BASELINE['Cooling_Load_kWh'] * 100
EUI = total_energy / GFA
carbon_emission = total_energy * carbon_factor
annual_cost = total_energy * tariff
annual_saving = (BASELINE['Total_kWh'] - total_energy) * tariff
payback_years = CAPEX / annual_saving if annual_saving > 0 else None

# -----------------------------------------------------
# 8️⃣ KPI Dashboard
# -----------------------------------------------------
st.title("🌿 RetroX Surrogate Toolkit – v3")

tabs = st.tabs(["⚡ Energy", "🌍 Environment", "💰 Economics"])

with tabs[0]:
    st.subheader("⚡ Energy Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lighting (kWh)", f"{lighting_pred:,.0f}")
    col2.metric("Cooling (kWh)", f"{cooling_pred:,.0f}")
    col3.metric("Room Elec (kWh)", f"{room_elec:,.0f}")
    col4.metric("Total Energy (kWh)", f"{total_energy:,.0f}")
    st.metric("Energy Saving (%)", f"{energy_saving_pct:.1f}%")
    st.metric("EUI (kWh/m²·yr)", f"{EUI:.2f}")
    st.metric("Cooling Load Saving (%)", f"{cool_saving_pct:.1f}%")

    # --- Energy Distribution Chart
    energy_df = pd.DataFrame({
        'Category': ['Lighting', 'Cooling', 'Room'],
        'Baseline (kWh)': [BASELINE['Lighting_kWh'], BASELINE['Cooling_kWh'], BASELINE['Room_kWh']],
        'Retrofit (kWh)': [lighting_pred, cooling_pred, room_elec]
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=energy_df['Category'], y=energy_df['Baseline (kWh)'], name='Baseline'))
    fig.add_trace(go.Bar(x=energy_df['Category'], y=energy_df['Retrofit (kWh)'], name='Retrofit'))
    fig.update_layout(barmode='group', title="Energy Breakdown vs Baseline")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("🌍 Environmental KPIs")
    st.metric("Carbon Emission (kg CO₂e)", f"{carbon_emission:,.1f}")
    st.metric("Carbon Factor (kgCO₂/kWh)", f"{carbon_factor:.2f}")

with tabs[2]:
    st.subheader("💰 Economic KPIs")
    st.metric("CAPEX (SGD)", f"{CAPEX:,.0f}")
    st.metric("Annual Cost (SGD)", f"{annual_cost:,.0f}")
    st.metric("Annual Saving (SGD)", f"{annual_saving:,.0f}")
    st.metric("Payback (years)", f"{payback_years:,.1f}" if payback_years else "—")

# -----------------------------------------------------
# 9️⃣ Green Mark Achievement
# -----------------------------------------------------
msg = f"Your building achieves **~{energy_saving_pct:.1f}% energy saving** with a payback of **{payback_years:.1f} years**."
if (EUI < 120) or (energy_saving_pct >= 35):
    msg += " 🏆 **Green Mark Platinum achieved!**"
elif (EUI < 135) or (energy_saving_pct >= 30):
    msg += " 🥇 **Green Mark Gold achieved!**"
else:
    msg += " 🌱 Meets base efficiency level."
st.info(msg)

# -----------------------------------------------------
# 🔽 Download Summary
# -----------------------------------------------------
summary = pd.DataFrame({
    'Lighting_kWh': [lighting_pred],
    'Cooling_kWh': [cooling_pred],
    'Room_kWh': [room_elec],
    'Total_Energy_kWh': [total_energy],
    'Energy_Saving_%': [energy_saving_pct],
    'EUI_kWh_m2': [EUI],
    'Cooling_Load_Saving_%': [cool_saving_pct],
    'Carbon_Emission_kg': [carbon_emission],
    'CAPEX_SGD': [CAPEX],
    'Annual_Saving_SGD': [annual_saving],
    'Payback_Years': [payback_years]
})
st.download_button("📥 Download KPI Summary (CSV)",
                   summary.to_csv(index=False).encode('utf-8'),
                   "RetroX_summary.csv", "text/csv")

st.success("✅ RetroX v3 analysis completed successfully.")
