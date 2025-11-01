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

# --- Existing cost items ---
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

# --- Newly added measures ---
schedule_cost = flexible_input("Schedule Adjustment", [1500, 2000, 2500])        # SGD per project
linearctrl_cost = flexible_input("Linear Control", [25, 30, 35])               # SGD per m² or sensor point


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

# -----------------------------------------------------
# 💰 ECONOMICS TAB – Simplified and Corrected
# -----------------------------------------------------
with tabs[2]:
    st.subheader("💰 Economic KPIs")

    # -------------------------------------------------
    # 1️⃣ Calculate KPIs
    # -------------------------------------------------
    # Total retrofit cost = sum of applied measure costs (already computed as CAPEX)
    Retrofit_Cost_Capex = CAPEX

    # Annual cost saving = Energy saving × electricity tariff
    Annual_Cost_Saving = (BASELINE['Total_kWh'] - total_energy) * tariff

    # Payback period (years)
    if Annual_Cost_Saving > 0:
        Payback_Period = Retrofit_Cost_Capex / Annual_Cost_Saving
    else:
        Payback_Period = None

    # -------------------------------------------------
    # 2️⃣ Display KPIs
    # -------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Retrofit Cost (Capex, SGD)", f"{Retrofit_Cost_Capex:,.0f}")
    col2.metric("Annual Cost Saving (SGD)", f"{Annual_Cost_Saving:,.0f}")
    col3.metric(
        "Payback Period (years)",
        f"{Payback_Period:.1f}" if Payback_Period else "—"
    )

    # -------------------------------------------------
    # 3️⃣ Show calculation formula for clarity
    # -------------------------------------------------
    st.markdown("<p style='font-size:15px;'><b>Economic KPI Formulas:</b></p>", unsafe_allow_html=True)
    st.latex(r"""
    \text{Retrofit Cost (Capex)} = \sum_{i=1}^{n} C_i
    """)
    st.latex(r"""
    \text{Annual Cost Saving} = E_{\text{saving}} \times T_{\text{elec}}
    """)
    st.latex(r"""
    \text{Payback Period (years)} =
    \frac{\text{Capex}}{\text{Annual Cost Saving}}
    """)

    st.caption("These KPIs evaluate the economic feasibility of your selected retrofit combination.")


# =====================================================
# 📊 Measure Impact Tab – True Contribution Analysis
# =====================================================
with tabs[3]:
    st.subheader("📊 Measure Contribution Analysis")
    st.caption("Quantifies how much each retrofit measure contributes to total energy saving and retrofit cost.")

    # -------------------------------------------------
    # 1️⃣ Define baseline configuration (no retrofit)
    # -------------------------------------------------
    X_base = X_input.copy()
    X_base.iloc[0] = 0
    X_base['LPD_Wm2'] = 12
    X_base['HVAC_Setpoint_C'] = 24
    X_base['ShadingDepth_m'] = 0
    X_base['Glazing_Single'] = 1
    X_base['Insulation_Low'] = 1
    X_base['ScheduleAdj_Base'] = 1
    X_base['LinearControl_Yes'] = 0
    X_base['HighAlbedoWall_Cool'] = 0

    # Predict baseline energy
    E_base = (
        models['Lighting_kWh'].predict(X_base)[0]
        + models['Cooling_kWh'].predict(X_base)[0]
        + BASELINE['Room_kWh']
    )

    # -------------------------------------------------
    # 2️⃣ Compute each measure’s ΔEnergy & ΔCost
    # -------------------------------------------------
    measures = [
        'Glazing', 'Insulation', 'LPD', 'HVAC', 'Shading',
        'Schedule', 'LinearControl', 'Albedo'
    ]
    results = []

    for m in measures:
        X_test = X_base.copy()

        # Apply only one measure
        if m == 'Glazing':
            if glazing == 'Double':
                X_test['Glazing_Single'] = 0
                X_test['Glazing_Double'] = 1
            elif glazing == 'Low-E':
                X_test['Glazing_Single'] = 0
                X_test['Glazing_Low-E'] = 1

        elif m == 'Insulation':
            if insul == 'Med':
                X_test['Insulation_Low'] = 0
                X_test['Insulation_Medium'] = 1
            elif insul == 'High':
                X_test['Insulation_Low'] = 0
                X_test['Insulation_Medium'] = 0

        elif m == 'LPD' and LPD < 12:
            X_test['LPD_Wm2'] = LPD

        elif m == 'HVAC' and hvac > 24:
            X_test['HVAC_Setpoint_C'] = hvac

        elif m == 'Shading' and shading > 0:
            X_test['ShadingDepth_m'] = shading

        elif m == 'Schedule' and schedule == 'Adjusted':
            X_test['ScheduleAdj_Base'] = 0

        elif m == 'LinearControl' and ctrl == 'Yes':
            X_test['LinearControl_Yes'] = 1

        elif m == 'Albedo' and albedo == 'Cool':
            X_test['HighAlbedoWall_Cool'] = 1

        # Skip if still baseline
        if X_test.equals(X_base):
            continue

        # Predict new energy
        E_new = (
            models['Lighting_kWh'].predict(X_test)[0]
            + models['Cooling_kWh'].predict(X_test)[0]
            + BASELINE['Room_kWh']
        )

        ΔE = E_base - E_new  # kWh saved
        ΔC = 0               # cost (CAPEX proxy)
        if m == 'Glazing':
            ΔC = glazing_cost_double * WinA if glazing == 'Double' else glazing_cost_lowe * WinA
        elif m == 'Insulation':
            ΔC = insul_cost_med * total_wall_roof if insul == 'Med' else insul_cost_high * total_wall_roof
        elif m == 'LPD' and LPD < 12:
            ΔC = led_cost * GFA
        elif m == 'HVAC' and hvac > 24:
            ΔC = hvac_cost
        elif m == 'Shading' and shading > 0:
            ΔC = shading_cost * WinA
        elif m == 'Schedule' and schedule == 'Adjusted':
            ΔC = schedule_cost
        elif m == 'LinearControl' and ctrl == 'Yes':
            ΔC = linearctrl_cost * GFA
        elif m == 'Albedo' and albedo == 'Cool':
            ΔC = albedo_cost * total_wall_roof

        results.append({
            'Measure': m,
            'Energy_Saving_kWh': ΔE,
            'Cost_SGD': ΔC
        })

    contrib_df = pd.DataFrame(results)
    contrib_df = contrib_df.sort_values('Energy_Saving_kWh', ascending=False)

    # -------------------------------------------------
    # 3️⃣ Visualizations
    # -------------------------------------------------
    palette = ['#5979A0', '#A8B5C5', '#EDE59D', '#7A9544', '#243C2C']
    impact_choice = st.selectbox("Choose Visualization", ["Bar", "Waterfall", "Radar"])

    # ---------- BAR PLOTS ----------
    if impact_choice == "Bar":
        st.markdown("### 🌿 Energy Saving Contribution")
        fig1 = px.bar(
            contrib_df, x='Energy_Saving_kWh', y='Measure',
            orientation='h', color='Measure', color_discrete_sequence=palette,
            labels={'Energy_Saving_kWh': 'Energy Saving (kWh)'},
            hover_data={'Energy_Saving_kWh': ':.2f'}
        )
        fig1.update_traces(hovertemplate='%{y}: %{x:.0f} kWh')
        fig1.update_layout(xaxis_title="Energy Saving (kWh)", yaxis_title="")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### 💰 Retrofit Cost Contribution")
        fig2 = px.bar(
            contrib_df, x='Cost_SGD', y='Measure',
            orientation='h', color='Measure', color_discrete_sequence=palette,
            labels={'Cost_SGD': 'Cost (SGD)'},
            hover_data={'Cost_SGD': ':.2f'}
        )
        fig2.update_traces(hovertemplate='%{y}: %{x:.0f} SGD')
        fig2.update_layout(xaxis_title="Retrofit Cost (SGD)", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- WATERFALL ----------
    elif impact_choice == "Waterfall":
        st.markdown("### 🌿 Energy Saving Contribution")
        fig3 = go.Figure(go.Waterfall(
            name="Energy Impact", orientation="v",
            measure=["relative"] * len(contrib_df),
            x=contrib_df['Measure'], y=contrib_df['Energy_Saving_kWh'],
            connector={"line": {"color": "#243C2C"}},
            increasing={"marker": {"color": "#7A9544"}},
            totals={"marker": {"color": "#EDE59D"}}
        ))
        fig3.update_layout(title="Energy Saving Contribution (kWh)",
                           yaxis_title="kWh", font=dict(color='#243C2C'))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### 💰 Retrofit Cost Contribution (Waterfall)")
        fig4 = go.Figure(go.Waterfall(
            name="Cost Impact", orientation="v",
            measure=["relative"] * len(contrib_df),
            x=contrib_df['Measure'], y=contrib_df['Cost_SGD'],
            connector={"line": {"color": "#243C2C"}},
            increasing={"marker": {"color": "#EDE59D"}},
            totals={"marker": {"color": "#EDE59D"}}
        ))
        fig4.update_layout(title="Retrofit Cost Contribution (SGD)",
                           yaxis_title="SGD", font=dict(color='#243C2C'))
        st.plotly_chart(fig4, use_container_width=True)

    # ---------- RADAR ----------
    else:
        st.markdown("### 🌿 Energy Saving Contribution (Radar)")
        cats1 = list(contrib_df['Measure']) + [contrib_df['Measure'].iloc[0]]
        vals1 = list(contrib_df['Energy_Saving_kWh']) + [contrib_df['Energy_Saving_kWh'].iloc[0]]
        fig5 = go.Figure(data=go.Scatterpolar(
            r=vals1, theta=cats1, fill='toself', marker=dict(color='#7A9544')
        ))
        fig5.update_layout(title="Energy Saving Contribution (kWh)",
                           polar=dict(radialaxis=dict(visible=True)),
                           font=dict(color='#243C2C'))
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("### 💰 Retrofit Cost Contribution (Radar)")
        cats2 = list(contrib_df['Measure']) + [contrib_df['Measure'].iloc[0]]
        vals2 = list(contrib_df['Cost_SGD']) + [contrib_df['Cost_SGD'].iloc[0]]
        fig6 = go.Figure(data=go.Scatterpolar(
            r=vals2, theta=cats2, fill='toself', marker=dict(color='#5979A0')
        ))
        fig6.update_layout(title="Retrofit Cost Contribution (SGD)",
                           polar=dict(radialaxis=dict(visible=True)),
                           font=dict(color='#243C2C'))
        st.plotly_chart(fig6, use_container_width=True)

# -------------------------------------------------
# 4️⃣ Impact Index (for weighted ranking)
# -------------------------------------------------
st.subheader("🎛️ Weighted Impact Index")
st.caption("Adjust the weights to balance between energy saving and retrofit cost efficiency.")

# --- Weight sliders ---
col1, col2 = st.columns(2)
with col1:
    w1 = st.slider("Weight: Energy Saving", 0.0, 1.0, 0.6)
with col2:
    w2 = st.slider("Weight: Retrofit Cost (inverse)", 0.0, 1.0, 0.4)

# --- Compute raw index ---
contrib_df["Impact_Index"] = (
    w1 * contrib_df["Energy_Saving_kWh"] -
    w2 * (contrib_df["Cost_SGD"] / 1000)
)

# -------------------------------------------------
# 🔟 Normalize Impact Index to 1–10 scale
# -------------------------------------------------
min_idx = contrib_df["Impact_Index"].min()
max_idx = contrib_df["Impact_Index"].max()

if max_idx != min_idx:
    contrib_df["Impact_Index_Scaled"] = 1 + 9 * (contrib_df["Impact_Index"] - min_idx) / (max_idx - min_idx)
else:
    contrib_df["Impact_Index_Scaled"] = 5  # fallback if all same

# Sort by scaled index
contrib_df = contrib_df.sort_values("Impact_Index_Scaled", ascending=False).reset_index(drop=True)

# -------------------------------------------------
# 🧮 Display formula + table
# -------------------------------------------------
st.markdown("<p style='font-size:15px;'><b>Impact Index Formula:</b></p>", unsafe_allow_html=True)
st.latex(r"""
\small
\text{Impact Index}_i =
( w_1 \times \text{Energy Saving (kWh)} ) -
( w_2 \times \frac{\text{Cost (SGD)}}{1{,}000} )
""")

st.markdown("**Scaled 1–10 score makes comparisons easier:**")
st.dataframe(
    contrib_df[["Measure", "Energy_Saving_kWh", "Cost_SGD", "Impact_Index_Scaled"]],
    use_container_width=True
)

# -------------------------------------------------
# 🏆 Summary & visualization
# -------------------------------------------------
top_measure = contrib_df.iloc[0]
total_index = contrib_df["Impact_Index_Scaled"].sum()

st.markdown("---")
st.markdown(
    f"🏆 **Top-performing measure:** `{top_measure['Measure']}` "
    f"with score **{top_measure['Impact_Index_Scaled']:.1f} / 10**"
)
st.markdown(
    f"📊 **Overall combined index (sum of all measures):** "
    f"**{total_index:.1f} / {10 * len(contrib_df):.0f}**"
)
st.caption("Higher index = better combined performance after weighting.")

# -------------------------------------------------
# 📊 Horizontal bar chart for visualization
# -------------------------------------------------
palette = ['#5979A0', '#A8B5C5', '#EDE59D', '#7A9544', '#243C2C']
fig7 = px.bar(
    contrib_df,
    x="Impact_Index_Scaled",
    y="Measure",
    orientation="h",
    color="Measure",
    color_discrete_sequence=palette,
    title="Weighted Impact Index (1–10 scale)",
    labels={"Impact_Index_Scaled": "Score (1–10)"},
    hover_data={"Impact_Index_Scaled": ':.2f'}
)
fig7.update_traces(
    text=contrib_df["Impact_Index_Scaled"].round(1),
    textposition="outside",
    hovertemplate="%{y}: %{x:.2f} / 10"
)
fig7.update_layout(
    xaxis=dict(range=[0, 10]),
    yaxis_title="",
    xaxis_title="Impact Score (1–10)",
    font=dict(color="#243C2C")
)
st.plotly_chart(fig7, use_container_width=True)



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
if (EUI < 120) or (energy_saving_pct >= 35):
    msg += " 🏆 Green Mark Platinum achieved!"
elif (EUI < 135) or (energy_saving_pct >= 30):
    msg += " 🥇 Green Mark Gold achieved!"

st.info(msg)


summary = pd.DataFrame({
    'Lighting_kWh':[lighting_pred],'Cooling_kWh':[cooling_pred],'Room_kWh':[room_elec],'Total_Energy_kWh':[total_energy],
    'Energy_Saving_%':[energy_saving_pct],'EUI_kWh_m2':[EUI],'Cooling_Load_Saving_%':[cool_saving_pct],
    'Carbon_Emission_kg':[carbon_emission],'CAPEX_SGD':[CAPEX],'Annual_Saving_SGD':[annual_saving],'Payback_Years':[payback_years]
})
st.download_button("📥 Download KPI Summary (CSV)", summary.to_csv(index=False).encode('utf-8'), "RetroX_summary.csv", "text/csv")

st.success("✅ RetroX v4: Analysis & Visualization complete.")
