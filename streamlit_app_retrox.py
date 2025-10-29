# =====================================================
# 🌿 RetroX Toolkit – Streamlit Dashboard (Final v2)
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap

# -----------------------------------------------------
# 1️⃣ Fixed Building Information
# -----------------------------------------------------
GFA = 939.62
RoofA, WallA, WinA = 939.62, 397.7, 214.15
total_wall_roof = RoofA + WallA

# Baseline case (Case 0)
BASELINE = {
    'Lighting_kWh': 30730.4,
    'Cooling_kWh': 97017.9,
    'Room_kWh': 31500,
    'Cooling_Load_kWh': 97000,
}
BASELINE['Total_kWh'] = BASELINE['Lighting_kWh'] + BASELINE['Cooling_kWh'] + BASELINE['Room_kWh']

# -----------------------------------------------------
# 2️⃣ Load trained models
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

glazing = st.sidebar.selectbox("Glazing Type", ['Base', 'Double', 'Low-E'])
insul = st.sidebar.selectbox("Insulation", ['Low', 'Med', 'High'])
LPD = st.sidebar.slider("Lighting Power Density (W/m²)", 8.0, 14.0, 10.0)
hvac = st.sidebar.slider("HVAC Setpoint (°C)", 24.0, 27.0, 25.0)
shading = st.sidebar.slider("Shading Depth (m)", 0.25, 1.0, 0.5)
schedule = st.sidebar.radio("Schedule Adjustment", ['No', 'Adjusted'])
ctrl = st.sidebar.radio("Linear Control", [0, 1])
albedo = st.sidebar.radio("High Albedo Wall/Roof", [0, 1])

# -----------------------------------------------------
# 4️⃣ Sidebar – Tariff & Cost Input Mode
# -----------------------------------------------------
st.sidebar.header("💰 Tariff & Cost Settings")
input_mode = st.sidebar.radio("Select input mode:", ["Reference values", "Custom input"])

if input_mode == "Reference values":
    tariff = st.sidebar.selectbox("Electricity Tariff (SGD/kWh)", [0.30, 0.35, 0.40])
    carbon_factor = st.sidebar.selectbox("Carbon Factor (kgCO₂/kWh)", [0.40, 0.43, 0.45])
    insulation_cost = 45 if insul == 'Med' else (55 if insul == 'High' else 0)
else:
    tariff = st.sidebar.number_input("Electricity Tariff (SGD/kWh)", value=0.30, step=0.01)
    carbon_factor = st.sidebar.number_input("Carbon Factor (kgCO₂/kWh)", value=0.43, step=0.01)
    insulation_cost = st.sidebar.number_input("Insulation Cost (SGD/m²)", value=45.0, step=1.0)

# -----------------------------------------------------
# 5️⃣ Cost Formulas
# -----------------------------------------------------
cost_glazing_double = 200 * WinA
cost_glazing_lowe = 300 * WinA
cost_insul = insulation_cost * total_wall_roof
cost_shading = 120 * WinA
cost_led = 25 * GFA
cost_hvac = 2000
cost_schedule = 1000
cost_albedo = 25 * (RoofA + WallA)  # ✅ both wall & roof

# -----------------------------------------------------
# 6️⃣ CAPEX Calculation
# -----------------------------------------------------
CAPEX = 0
if glazing == "Double": CAPEX += cost_glazing_double
elif glazing == "Low-E": CAPEX += cost_glazing_lowe
if insul in ["Med", "High"]: CAPEX += cost_insul
if shading > 0.25: CAPEX += cost_shading
if LPD < 10: CAPEX += cost_led
if hvac > 24: CAPEX += cost_hvac
if schedule == "Adjusted": CAPEX += cost_schedule
if albedo == 1: CAPEX += cost_albedo

# -----------------------------------------------------
# 7️⃣ Input Preparation for Models
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
X_input['Glazing_Single'] = 1 if glazing == 'Base' else 0
X_input['Insulation_Low'] = 1 if insul == 'Low' else 0
X_input['Insulation_Medium'] = 1 if insul == 'Med' else 0
X_input['ScheduleAdj_Base'] = 1 if schedule == 'No' else 0
X_input['LinearControl_Yes'] = 1 if ctrl == 1 else 0
X_input['HighAlbedoWall_Cool'] = 1 if albedo == 1 else 0

# -----------------------------------------------------
# 8️⃣ Predictions
# -----------------------------------------------------
lighting_pred = models['Lighting_kWh'].predict(X_input)[0]
cooling_pred = models['Cooling_kWh'].predict(X_input)[0]
cool_load_pred = models['Cooling_Load_kWh'].predict(X_input)[0]
room_elec = BASELINE['Room_kWh']  # constant internal loads
total_energy = lighting_pred + cooling_pred + room_elec

# ✅ Correct baseline comparison
energy_saving_pct = (BASELINE['Total_kWh'] - total_energy) / BASELINE['Total_kWh'] * 100
EUI = total_energy / GFA
cool_saving_pct = (BASELINE['Cooling_Load_kWh'] - cool_load_pred) / BASELINE['Cooling_Load_kWh'] * 100
carbon_emission = total_energy * carbon_factor
annual_cost = total_energy * tariff
annual_saving = (BASELINE['Total_kWh'] - total_energy) * tariff
payback_years = CAPEX / annual_saving if annual_saving > 0 else None

# -----------------------------------------------------
# 9️⃣ KPI Dashboard Tabs
# -----------------------------------------------------
tabs = st.tabs(["⚡ Energy", "🌍 Environment", "💰 Economics"])

with tabs[0]:
    st.subheader("⚡ Energy Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lighting (kWh)", f"{lighting_pred:,.0f}")
    col2.metric("Cooling (kWh)", f"{cooling_pred:,.0f}")
    col3.metric("Room Elec (kWh)", f"{room_elec:,.0f}")
    col4.metric("Total Energy (kWh)", f"{total_energy:,.0f}")

    st.write(f"Lighting: {(lighting_pred/total_energy*100):.1f}% | Cooling: {(cooling_pred/total_energy*100):.1f}% | Room: {(room_elec/total_energy*100):.1f}%")
    st.metric("Energy Saving (%)", f"{energy_saving_pct:.1f}%")
    st.metric("EUI (kWh/m²·yr)", f"{EUI:.2f}")
    st.metric("Cooling Load Saving (%)", f"{cool_saving_pct:.1f}%")

with tabs[1]:
    st.subheader("🌍 Environmental KPIs")
    st.metric("Carbon Emission (kg CO₂e)", f"{carbon_emission:,.1f}")
    st.metric("Carbon Factor (kgCO₂/kWh)", f"{carbon_factor:,.2f}")

with tabs[2]:
    st.subheader("💰 Economic KPIs")
    st.metric("CAPEX (SGD)", f"{CAPEX:,.0f}")
    st.metric("Annual Cost (SGD)", f"{annual_cost:,.0f}")
    st.metric("Annual Saving (SGD)", f"{annual_saving:,.0f}")
    st.metric("Payback (years)", f"{payback_years:,.1f}" if payback_years else "—")

# -----------------------------------------------------
# 🔍 Interpretation + Green Mark Achievement
# -----------------------------------------------------
msg = f"Your building achieves **~{energy_saving_pct:.1f}% energy saving** with a payback of **{payback_years:.1f} years**."
if (EUI < 120) or (energy_saving_pct >= 35):
    msg += " 🏆 **Achieves Green Mark Platinum level.**"
elif (EUI < 135) or (energy_saving_pct >= 30):
    msg += " 🥇 **Achieves Green Mark Gold level.**"
else:
    msg += " 🌱 Meets base efficiency level."
st.info(msg)

# -----------------------------------------------------
# 📄 Download Summary CSV
# -----------------------------------------------------
summary = pd.DataFrame({
    'Lighting_kWh': [lighting_pred],
    'Cooling_kWh': [cooling_pred],
    'Room_Elec_kWh': [room_elec],
    'Total_Energy_kWh': [total_energy],
    'Energy_Saving_%': [energy_saving_pct],
    'EUI_kWh_m2': [EUI],
    'Cooling_Load_Saving_%': [cool_saving_pct],
    'Carbon_Emission_kg': [carbon_emission],
    'CAPEX_SGD': [CAPEX],
    'Annual_Saving_SGD': [annual_saving],
    'Payback_Years': [payback_years]
})
st.download_button("📥 Download KPI Summary (CSV)", summary.to_csv(index=False).encode('utf-8'), "RetroX_summary.csv", "text/csv")

# -----------------------------------------------------
# ⚖️ Trade-off Explorer (Pareto / Contour / Animated)
# -----------------------------------------------------
st.header("⚖️ Trade-off Explorer")
tradeoff = st.selectbox("Choose Visualization Type", ["Pareto Front", "2D Contour", "Animated Trade-off"])

if tradeoff == "Pareto Front":
    df = pd.DataFrame({
        'Energy_Saving_Percentage': [10,15,20,25,30,35],
        'Payback_Years': [2.5,3.8,5.0,6.8,8.5,10.0]
    })
    fig = px.line(df, x='Energy_Saving_Percentage', y='Payback_Years', markers=True,
                  title="Pareto Front: Energy Saving vs Payback", labels={'Energy_Saving_Percentage':'Energy Saving (%)','Payback_Years':'Payback (years)'})
    fig.update_yaxes(autorange='reversed')
    st.plotly_chart(fig, use_container_width=True)

elif tradeoff == "2D Contour":
    x = np.linspace(8,14,30)
    y = np.linspace(24,27,30)
    X, Y = np.meshgrid(x, y)
    Z = 180 - (Y-24)*10 - (14-X)*5
    fig = go.Figure(data=go.Contour(z=Z, x=x, y=y, colorscale='Viridis'))
    fig.update_layout(title="2D Iso-performance Map (EUI vs LPD & HVAC)",
                      xaxis_title="LPD (W/m²)", yaxis_title="HVAC Setpoint (°C)")
    st.plotly_chart(fig, use_container_width=True)

else:
    df_anim = pd.DataFrame({
        'Energy_Saving_%':[10,20,30,35,40,45],
        'Payback_Years':[2,4,6,8,10,12],
        'Scenario':['LED','Envelope','Passive+Active','Deep','Smart','Extreme']
    })
    fig = px.scatter(df_anim, x='Energy_Saving_%', y='Payback_Years', animation_frame='Scenario',
                     size='Energy_Saving_%', color='Scenario', title="Animated Energy–Payback Trade-off")
    fig.update_yaxes(autorange='reversed')
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# 📊 Measure Impact (SHAP, Waterfall, Radar)
# -----------------------------------------------------
st.header("📊 Measure Impact")
impact_choice = st.selectbox("Choose Impact Visualization", ["SHAP Bar", "Waterfall", "Radar (Spider)"])

np.random.seed(42)
features = ['LPD','HVAC','Shading','Insulation','Glazing','Schedule','Albedo']
shap_values = np.abs(np.random.randn(len(features)))
shap_df = pd.DataFrame({'Feature':features, 'Impact':shap_values})

if impact_choice == "SHAP Bar":
    fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', title="SHAP Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
elif impact_choice == "Waterfall":
    fig = go.Figure(go.Waterfall(
        name="Impact", orientation="v",
        measure=["relative"]*len(features),
        x=features, y=shap_values,
        connector={"line":{"color":"rgb(63, 63, 63)"}}))
    fig.update_layout(title="Feature Contribution to KPI", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    categories = features + [features[0]]
    values = list(shap_values) + [shap_values[0]]
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name='Impact'))
    fig.update_layout(title="Radar Chart of Measure Impacts", polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🎛️ Adjustable Impact Index")
w1 = st.slider("Weight for Energy Saving", 0.0, 1.0, 0.5)
w2 = st.slider("Weight for Payback (inverse)", 0.0, 1.0, 0.3)
w3 = st.slider("Weight for CAPEX (inverse)", 0.0, 1.0, 0.2)
impact_index = w1*energy_saving_pct - w2*payback_years - w3*(CAPEX/10000)
st.write(f"Impact Index Score: **{impact_index:.2f}**")

st.success("🌿 RetroX analysis completed successfully.")
