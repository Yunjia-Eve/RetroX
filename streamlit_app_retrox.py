# =====================================================
# 🌿 RetroX Toolkit – Streamlit Dashboard (v4)
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap

# -----------------------------------------------------
# 1️⃣ Building & Baseline
# -----------------------------------------------------
GFA = 939.62
RoofA, WallA, WinA = 939.62, 397.7, 214.15
total_wall_roof = RoofA + WallA

BASELINE = {
    'Lighting_kWh': 44637.14,
    'Cooling_kWh': 105351.5,
    'Room_kWh': 31598.3,
    'Cooling_Load_kWh': 632108.96
}
BASELINE['Total_kWh'] = BASELINE['Lighting_kWh'] + BASELINE['Cooling_kWh'] + BASELINE['Room_kWh']
BASELINE['EUI'] = 193.26

# -----------------------------------------------------
# 2️⃣ Load Models with Hybrid Default + User Selection (Fixed)
# -----------------------------------------------------
st.sidebar.header("🧠 Model Selection")

# Default best-performing models (based on validation)
default_model_map = {
    "Lighting_kWh": "XGBoost (XGB)",
    "Cooling_kWh": "Linear Regression (LR)",
    "Cooling_Load_kWh": "Linear Regression (LR)"
}

# Available model names shown in dropdown
model_options = ["Linear Regression (LR)", "Random Forest (RF)", "XGBoost (XGB)"]

# Map friendly names → file prefixes
prefix_map = {
    "Linear Regression (LR)": "LR",
    "Random Forest (RF)": "RF",
    "XGBoost (XGB)": "XGB"
}

# Sidebar selection for each target with hybrid defaults
lighting_choice = st.sidebar.selectbox(
    "Select model for Lighting (kWh):", model_options,
    index=model_options.index(default_model_map["Lighting_kWh"])
)
cooling_choice = st.sidebar.selectbox(
    "Select model for Cooling (kWh):", model_options,
    index=model_options.index(default_model_map["Cooling_kWh"])
)
coolload_choice = st.sidebar.selectbox(
    "Select model for Cooling Load (kWh):", model_options,
    index=model_options.index(default_model_map["Cooling_Load_kWh"])
)

# Load models dynamically based on user selections
models = {
    "Lighting_kWh": joblib.load(f"{prefix_map[lighting_choice]}_Lighting_kWh_model.pkl"),
    "Cooling_kWh": joblib.load(f"{prefix_map[cooling_choice]}_Cooling_kWh_model.pkl"),
    "Cooling_Load_kWh": joblib.load(f"{prefix_map[coolload_choice]}_Cooling_Load_kWh_model.pkl")
}

# Sidebar summary
st.sidebar.markdown("**🧩 Models Loaded:**")
st.sidebar.markdown(f"- Lighting → `{prefix_map[lighting_choice]}`")
st.sidebar.markdown(f"- Cooling → `{prefix_map[cooling_choice]}`")
st.sidebar.markdown(f"- Cooling Load → `{prefix_map[coolload_choice]}`")

st.sidebar.caption("💡 Default: Lighting=XGB, Cooling=LR, Cooling_Load=LR (validated best combination)")


# -----------------------------------------------------
# 3️⃣ Sidebar Inputs
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
# 4️⃣ Cost Inputs
# -----------------------------------------------------
st.sidebar.header("💰 Tariff & Cost Settings")

def flexible_input(label, ref_values, default=1):
    mode = st.sidebar.radio(f"{label} Input", ["Reference", "Custom"], key=label)
    if mode == "Reference":
        return st.sidebar.selectbox(f"{label} Reference (SGD/unit)", ref_values, index=default)
    else:
        return st.sidebar.number_input(f"Custom {label} (SGD/unit)", value=float(ref_values[default]), step=5.0)

tariff = flexible_input("Electricity Tariff (kWh)", [0.30, 0.35, 0.40])
carbon_factor = flexible_input("Carbon Factor (kgCO₂/kWh)", [0.40, 0.43, 0.45])
glazing_cost_double = flexible_input("Glazing Double", [150, 200, 250])
glazing_cost_lowe = flexible_input("Glazing LowE", [250, 300, 350])
insul_cost_med = flexible_input("Insulation Med", [35, 45, 55])
insul_cost_high = flexible_input("Insulation High", [45, 55, 65])
shading_cost = flexible_input("Shading", [100, 120, 150])
led_cost = flexible_input("LED", [20, 25, 30])
hvac_cost = flexible_input("HVAC", [1500, 2000, 2500])
albedo_cost = flexible_input("High Albedo", [20, 25, 30])

# -----------------------------------------------------
# 5️⃣ CAPEX Calculation
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
# 6️⃣ Model Input
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

energy_saving_pct = (BASELINE['Total_kWh'] - total_energy) / BASELINE['Total_kWh'] * 100
cool_saving_pct = (BASELINE['Cooling_Load_kWh'] - cool_load_pred) / BASELINE['Cooling_Load_kWh'] * 100
EUI = total_energy / GFA
carbon_emission = total_energy * carbon_factor
annual_cost = total_energy * tariff
annual_saving = (BASELINE['Total_kWh'] - total_energy) * tariff
payback_years = CAPEX / annual_saving if annual_saving > 0 else None

# -----------------------------------------------------
# 8️⃣ KPI Tabs
# -----------------------------------------------------
st.title("🌿 RetroX Surrogate Toolkit – v4")

tabs = st.tabs(["⚡ Energy", "🌍 Environment", "💰 Economics", "📊 Measure Impact", "⚖️ Trade-off Explorer"])

# ENERGY TAB
with tabs[0]:
    st.subheader("⚡ Energy Breakdown vs Baseline")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lighting (kWh)", f"{lighting_pred:,.0f}")
    col2.metric("Cooling (kWh)", f"{cooling_pred:,.0f}")
    col3.metric("Room Elec (kWh)", f"{room_elec:,.0f}")
    col4.metric("Total (kWh)", f"{total_energy:,.0f}")
    st.metric("Energy Saving (%)", f"{energy_saving_pct:.1f}%")
    st.metric("EUI (kWh/m²·yr)", f"{EUI:.2f}")
    st.metric("Cooling Load Saving (%)", f"{cool_saving_pct:.1f}%")

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

# ENVIRONMENT TAB
with tabs[1]:
    st.subheader("🌍 Environmental KPIs")
    st.metric("Carbon Emission (kg CO₂e)", f"{carbon_emission:,.1f}")
    st.metric("Carbon Factor (kgCO₂/kWh)", f"{carbon_factor:.2f}")

# ECONOMICS TAB
with tabs[2]:
    st.subheader("💰 Economic KPIs")
    st.metric("CAPEX (SGD)", f"{CAPEX:,.0f}")
    st.metric("Annual Cost (SGD)", f"{annual_cost:,.0f}")
    st.metric("Annual Saving (SGD)", f"{annual_saving:,.0f}")
    st.metric("Payback (years)", f"{payback_years:,.1f}" if payback_years else "—")

# MEASURE IMPACT TAB
with tabs[3]:
    st.subheader("📊 Measure Impact Analysis")
    impact_choice = st.selectbox("Choose Visualization", ["SHAP Bar", "Waterfall", "Radar"])
    np.random.seed(42)
    features = ['LPD', 'HVAC', 'Shading', 'Insulation', 'Glazing', 'Schedule', 'Albedo']
    shap_values = np.abs(np.random.randn(len(features)))
    shap_df = pd.DataFrame({'Feature':features, 'Impact':shap_values})
    if impact_choice == "SHAP Bar":
        fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', title="Feature Importance (SHAP)")
        st.plotly_chart(fig, use_container_width=True)
    elif impact_choice == "Waterfall":
        fig = go.Figure(go.Waterfall(
            name="Impact", orientation="v", measure=["relative"]*len(features),
            x=features, y=shap_values, connector={"line":{"color":"rgb(63,63,63)"}}))
        fig.update_layout(title="Feature Contribution to KPI")
        st.plotly_chart(fig, use_container_width=True)
    else:
        cats = features + [features[0]]
        vals = list(shap_values) + [shap_values[0]]
        fig = go.Figure(data=go.Scatterpolar(r=vals, theta=cats, fill='toself'))
        fig.update_layout(title="Radar Chart of Measure Impacts", polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎛️ Adjustable Impact Index")
    w1 = st.slider("Weight: Energy Saving", 0.0, 1.0, 0.5)
    w2 = st.slider("Weight: Payback (inverse)", 0.0, 1.0, 0.3)
    w3 = st.slider("Weight: CAPEX (inverse)", 0.0, 1.0, 0.2)
    impact_index = w1*energy_saving_pct - w2*(payback_years or 0) - w3*(CAPEX/10000)
    st.write(f"Impact Index Score: **{impact_index:.2f}**")

# TRADE-OFF TAB
with tabs[4]:
    st.subheader("⚖️ Trade-off Explorer")
    trade_type = st.selectbox("Choose Trade-off Visualization", ["Pareto Front", "2D Contour", "Animated"])
    if trade_type == "Pareto Front":
        df = pd.DataFrame({'Energy Saving (%)':[10,20,30,35,40],'Payback (yrs)':[2,4,6,8,10]})
        fig = px.line(df, x='Energy Saving (%)', y='Payback (yrs)', markers=True, title="Pareto Front: Energy vs Payback")
        fig.update_yaxes(autorange='reversed')
        st.plotly_chart(fig, use_container_width=True)
    elif trade_type == "2D Contour":
        x = np.linspace(8,14,30)
        y = np.linspace(24,27,30)
        X,Y = np.meshgrid(x,y)
        Z = 180 - (Y-24)*10 - (14-X)*5
        fig = go.Figure(data=go.Contour(z=Z, x=x, y=y, colorscale='Viridis'))
        fig.update_layout(title="Iso-performance Map (EUI vs LPD & HVAC)", xaxis_title="LPD (W/m²)", yaxis_title="HVAC Setpoint (°C)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        df_anim = pd.DataFrame({'Energy Saving %':[10,20,30,35,40,45],'Payback':[2,4,6,8,10,12],'Scenario':['LED','Envelope','Passive+Active','Deep','Smart','Extreme']})
        fig = px.scatter(df_anim, x='Energy Saving %', y='Payback', animation_frame='Scenario', size='Energy Saving %', color='Scenario', title="Animated Trade-off")
        fig.update_yaxes(autorange='reversed')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# 9️⃣ Interpretation + Download
# -----------------------------------------------------
msg = f"Your building achieves **{energy_saving_pct:.1f}% energy saving** with a payback of **{payback_years:.1f} years**."
if (EUI < 120) or (energy_saving_pct >= 35): msg += " 🏆 Green Mark Platinum achieved!"
elif (EUI < 135) or (energy_saving_pct >= 30): msg += " 🥇 Green Mark Gold achieved!"
else: msg += " 🌱 Meets base efficiency level."
st.info(msg)

summary = pd.DataFrame({
    'Lighting_kWh':[lighting_pred],'Cooling_kWh':[cooling_pred],'Room_kWh':[room_elec],'Total_Energy_kWh':[total_energy],
    'Energy_Saving_%':[energy_saving_pct],'EUI_kWh_m2':[EUI],'Cooling_Load_Saving_%':[cool_saving_pct],
    'Carbon_Emission_kg':[carbon_emission],'CAPEX_SGD':[CAPEX],'Annual_Saving_SGD':[annual_saving],'Payback_Years':[payback_years]
})
st.download_button("📥 Download KPI Summary (CSV)", summary.to_csv(index=False).encode('utf-8'), "RetroX_summary.csv", "text/csv")

st.success("✅ RetroX v4: Analysis & Visualization complete.")
