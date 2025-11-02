# =====================================================
# RetroX Toolkit – Streamlit Dashboard (v4.3)
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
st.sidebar.header("Model Selection")

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
# 3️⃣ User Inputs
# -----------------------------------------------------
st.sidebar.header("Building Inputs")
glazing  = st.sidebar.selectbox("Glazing Type", ["Single", "Double", "Low-E"])
insul    = st.sidebar.selectbox("Insulation", ["Low", "Med", "High"])
LPD      = st.sidebar.slider("Lighting Power Density (W/m²)", 8.0, 14.0, 10.0)
hvac     = st.sidebar.slider("HVAC Setpoint (°C)", 24.0, 27.0, 25.0)
shading  = st.sidebar.slider("Shading Depth (m)", 0.0, 1.0, 0.5)
schedule = st.sidebar.radio("Schedule Adjustment", ["Base", "Adjusted"])
ctrl     = st.sidebar.radio("Linear Control", ["No", "Yes"])
albedo   = st.sidebar.radio("High-Albedo Wall/Roof", ["Base", "Cool"])

# -----------------------------------------------------
# 4️⃣ Cost Settings
# -----------------------------------------------------
st.sidebar.header("Tariff & Cost Settings")
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
if schedule == "Adjusted": CAPEX += schedule_cost
if ctrl == "Yes": CAPEX += linearctrl_cost * GFA

# -----------------------------------------------------
# 6️⃣ Model Input
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
X_input["Glazing_Single"] = 1 if glazing=="Single" else 0
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

# --- Room electricity fixed by insulation & albedo ---
if insul == "Low" and albedo == "Base":
    room_elec = 31598.3
elif insul == "Low" and albedo == "Cool":
    room_elec = 31556.6
elif insul == "Med" and albedo == "Base":
    room_elec = 31452.5
elif insul == "Med" and albedo == "Cool":
    room_elec = 31410.9
elif insul == "High" and albedo == "Base":
    room_elec = 31203.0
elif insul == "High" and albedo == "Cool":
    room_elec = 31161.4
else:
    room_elec = BASELINE["Room_kWh"]

# --- Total energy & KPIs ---
total_energy = lighting_pred + cooling_pred + room_elec
energy_saving_pct = (BASELINE["Total_kWh"] - total_energy) / BASELINE["Total_kWh"] * 100
cool_saving_pct   = (BASELINE["Cooling_Load_kWh"] - cool_load_pred) / BASELINE["Cooling_Load_kWh"] * 100
EUI = total_energy / GFA
carbon_emission = total_energy * carbon_factor
annual_saving = (BASELINE["Total_kWh"] - total_energy) * tariff
payback_years = CAPEX / annual_saving if annual_saving > 0 else None


# -----------------------------------------------------
# 8️⃣ Tabs
# -----------------------------------------------------
palette = ['#a3b565','#fcdd9d','#c4c3e3','#5979A0','#243C2C']
st.title("RetroX SG Toolkit")

tabs = st.tabs(["Energy","Environment","Economics","Measure Impact","Trade-off Explorer"])

# -----------------------------------------------------
# ⚡ ENERGY TAB
# -----------------------------------------------------
with tabs[0]:
    st.subheader("Energy Breakdown vs Baseline")

    # --- First row: Energy use breakdown ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total (kWh)", f"{total_energy:,.0f}")
    col2.metric("Lighting (kWh)", f"{lighting_pred:,.0f}")
    col3.metric("Cooling (kWh)", f"{cooling_pred:,.0f}")
    col4.metric("Room Elec (kWh)", f"{room_elec:,.0f}")

    # --- Second row: Performance indicators ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("EUI (kWh/m²·yr)", f"{EUI:.2f}")
    col2.metric("Energy Saving (%)", f"{energy_saving_pct:.1f}%")
    col3.metric("Cooling Load Saving (%)", f"{cool_saving_pct:.1f}%")
    col4.markdown("")

    # --- Energy Breakdown Bar Chart ---
    energy_df = pd.DataFrame({
        'Category': ['Lighting', 'Cooling', 'Room'],
        'Baseline (kWh)': [BASELINE['Lighting_kWh'], BASELINE['Cooling_kWh'], BASELINE['Room_kWh']],
        'Retrofit (kWh)': [lighting_pred, cooling_pred, room_elec]
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=energy_df['Category'], y=energy_df['Baseline (kWh)'],
                         name='Baseline', marker_color="#504e76"))
    fig.add_trace(go.Bar(x=energy_df['Category'], y=energy_df['Retrofit (kWh)'],
                         name='Retrofit', marker_color="#a3b565"))
    fig.update_layout(barmode='group', title="Energy Breakdown vs Baseline",
                      font=dict(color="#243C2C"))
    st.plotly_chart(fig, use_container_width=True)

    # --- Summary Message ---
    msg = f"Your building achieves {energy_saving_pct:.1f}% energy saving with a payback of {payback_years:.1f} years."

    # --- EUI Quartile Benchmark ---
    if EUI <= 109:
        quartile_text = "Top Quartile (best-performing buildings)"
        comment = "\n\nExcellent performance: your building is among Singapore’s most energy-efficient offices."
    elif EUI <= 142:
        quartile_text = "2nd Quartile"
        comment = "\n\nGood performance: your building performs better than the national median."
    elif EUI <= 184:
        quartile_text = "3rd Quartile"
        comment = "\n\nModerate performance: your building performs close to the national average."
    else:
        quartile_text = "Bottom Quartile"
        comment = "\n\nBelow average: your building consumes more energy than typical offices."

    msg += f"<br><br>Compared to the BCA 2024 Building Energy Benchmarking Report, your building’s EUI is {EUI:.1f} kWh/m²·yr, falling in the {quartile_text}. {comment}"

    # --- Green Mark Achievement ---
    if (EUI < 120) or (energy_saving_pct >= 35):
        msg += "<br><br><span style='color:#4C9A2A; font-weight:bold;'>Green Mark Platinum achieved!</span>"
    elif (EUI < 135) or (energy_saving_pct >= 30):
        msg += "<br><br><span style='color:#C2A23A; font-weight:bold;'>Green Mark Gold achieved!</span>"

    st.markdown(
        f"<div style='background-color:#eef6fb; padding:15px; border-radius:8px;'>{msg}</div>",
        unsafe_allow_html=True
    )

    st.markdown("<p style='color:grey; font-size:14px; font-weight:bold;'>BCA Benchmark 2024 Reference</p>",
                unsafe_allow_html=True)



# -----------------------------------------------------
# 🌍 ENVIRONMENT TAB
# -----------------------------------------------------
with tabs[1]:
    st.subheader("Environmental KPIs")
    col1, col2 = st.columns(2)
    col1.metric("Carbon Emission (kg CO₂e)", f"{carbon_emission:,.1f}")
    col2.metric("Carbon Factor (kgCO₂/kWh)", f"{carbon_factor:.2f}")

    carbon_intensity = carbon_emission / GFA  # kg CO₂e/m²·yr

    # --- Benchmark classification ---
    if carbon_intensity <= 55:
        carbon_comment = "\n\nExcellent: aligns with Green Mark Platinum benchmark."
        carbon_level = "Platinum"
    elif carbon_intensity <= 70:
        carbon_comment = "\n\nGood: aligns with Green Mark Gold benchmark."
        carbon_level = "Gold"
    elif carbon_intensity <= 85:
        carbon_comment = "\n\nAverage: comparable to typical Singapore offices."
        carbon_level = "Average"
    else:
        carbon_comment = "\n\nHigh: above national average operational carbon intensity."
        carbon_level = "Below Benchmark"

    # --- Larger Gauge Chart for Carbon Intensity ---
    fig_carbon = go.Figure(go.Indicator(
        mode="gauge+number",
        value=carbon_intensity,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Operational Carbon Intensity (kg CO₂e/m²·yr)",
               'font': {'size': 17, 'color': '#243C2C'}},
        number={'font': {'size': 28, 'color': '#243C2C', 'family': 'Arial', 'weight': 'bold'}},
        gauge={
            'shape': 'angular',
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "white"},
            'bar': {'color': "#243C2C", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 55], 'color': '#7A9544'},
                {'range': [55, 70], 'color': '#fcdd9d'},
                {'range': [70, 85], 'color': '#c4c3e3'},
                {'range': [85, 100], 'color': '#504e76'}
            ],
            'threshold': {'line': {'color': "#243C2C", 'width': 4},
                          'thickness': 0.9, 'value': carbon_intensity}
        }
    ))
    fig_carbon.update_layout(
        margin=dict(t=70, b=30, l=10, r=10),
        height=360,
        paper_bgcolor="white",
        font=dict(color="#243C2C", family="Arial"),
    )
    st.plotly_chart(fig_carbon, use_container_width=True)

    # --- Summary Message ---
    msg = f"Your building achieves {energy_saving_pct:.1f}% energy saving with a payback of {payback_years:.1f} years."
    msg += f"<br><br>The operational carbon intensity is {carbon_intensity:.1f} kg CO₂e/m²·yr, classified as {carbon_level}. {carbon_comment}"

    if (EUI < 120) or (energy_saving_pct >= 35):
        msg += "<br><br><span style='color:#4C9A2A; font-weight:bold;'>Green Mark Platinum achieved!</span>"
    elif (EUI < 135) or (energy_saving_pct >= 30):
        msg += "<br><br><span style='color:#C2A23A; font-weight:bold;'>Green Mark Gold achieved!</span>"

    st.markdown(
        f"<div style='background-color:#eef6fb; padding:15px; border-radius:8px;'>{msg}</div>",
        unsafe_allow_html=True
    )

    st.markdown("<p style='color:grey; font-size:14px; font-weight:bold;'>BCA Benchmark 2024 Reference</p>",
                unsafe_allow_html=True)



# -----------------------------------------------------
# 💰 ECONOMICS TAB
# -----------------------------------------------------
with tabs[2]:
    st.subheader("Economic KPIs")

    col1, col2, col3 = st.columns(3)
    col1.metric("Retrofit Cost (SGD)", f"{CAPEX:,.0f}")
    col2.metric("Annual Saving (SGD)", f"{annual_saving:,.0f}")
    col3.metric("Payback (years)", f"{payback_years:.1f}")

    # --- Larger Gauge Chart for Payback Years ---
    fig_payback = go.Figure(go.Indicator(
        mode="gauge+number",
        value=payback_years,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Retrofit Payback Period (Years)",
               'font': {'size': 17, 'color': '#243C2C'}},
        number={'font': {'size': 28, 'color': '#243C2C', 'family': 'Arial', 'weight': 'bold'}},
        gauge={
            'shape': 'angular',
            'axis': {'range': [0, 12], 'tickwidth': 0, 'tickcolor': "white"},
            'bar': {'color': "#243C2C", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 5], 'color': '#7A9544'},
                {'range': [5, 7], 'color': '#fcdd9d'},
                {'range': [7, 9], 'color': '#c4c3e3'},
                {'range': [9, 12], 'color': '#504e76'}
            ],
            'threshold': {'line': {'color': "#243C2C", 'width': 4},
                          'thickness': 0.9, 'value': payback_years}
        }
    ))
    fig_payback.update_layout(
        margin=dict(t=70, b=30, l=10, r=10),
        height=360,
        paper_bgcolor="white",
        font=dict(color="#243C2C", family="Arial"),
    )
    st.plotly_chart(fig_payback, use_container_width=True)

    # --- Summary Message ---
    msg = f"Your building achieves {energy_saving_pct:.1f}% energy saving with a payback of {payback_years:.1f} years."

    if payback_years <= 5:
        payback_comment = "\n\nExcellent: rapid return on investment."
    elif payback_years <= 7:
        payback_comment = "\n\nGood: in line with national benchmark."
    elif payback_years <= 9:
        payback_comment = "\n\nModerate: slightly longer than typical retrofits."
    else:
        payback_comment = "Slow – beyond expected economic range for retrofits."

    msg += f"<br><br>Compared to typical office retrofits in Singapore (6–7 years average, BCA/IEA study), your payback performance is classified as: {payback_comment}"

    if (EUI < 120) or (energy_saving_pct >= 35):
        msg += "<br><br><span style='color:#4C9A2A; font-weight:bold;'>Green Mark Platinum achieved!</span>"
    elif (EUI < 135) or (energy_saving_pct >= 30):
        msg += "<br><br><span style='color:#C2A23A; font-weight:bold;'>Green Mark Gold achieved!</span>"

    st.markdown(
        f"<div style='background-color:#eef6fb; padding:15px; border-radius:8px;'>{msg}</div>",
        unsafe_allow_html=True
    )

    st.markdown("<p style='color:grey; font-size:14px; font-weight:bold;'>BCA & IEA Retrofit Benchmark Reference</p>",
                unsafe_allow_html=True)


# -----------------------------------------------------
# 📊 Measure Impact Tab
# -----------------------------------------------------
with tabs[3]:
    st.subheader(" Measure Contribution Analysis")
    st.caption("Quantifies how much each retrofit measure contributes to total energy saving and retrofit cost.")

    # --- Baseline (no retrofit)
    X_base = X_input.copy()
    X_base.iloc[0] = 0
    X_base["LPD_Wm2"]=12; X_base["HVAC_Setpoint_C"]=24; X_base["ShadingDepth_m"]=0
    X_base["Glazing_Single"]=1; X_base["Insulation_Low"]=1; X_base["ScheduleAdj_Base"]=1
    X_base["LinearControl_Yes"]=0; X_base["HighAlbedoWall_Cool"]=0
    E_base = models["Lighting_kWh"].predict(X_base)[0]+models["Cooling_kWh"].predict(X_base)[0]+BASELINE["Room_kWh"]

    measures = ["Glazing","Insulation","LPD","HVAC","Shading","Schedule","LinearControl","Albedo"]
    results=[]
    for m in measures:
        X_test=X_base.copy()
        if m=="Glazing" and glazing in ["Double","Low-E"]:
            if glazing=="Low-E": X_test["Glazing_Low-E"]=1; X_test["Glazing_Single"]=0
        elif m=="Insulation" and insul in ["Med","High"]:
            X_test["Insulation_Low"]=0; X_test["Insulation_Medium"]=1 if insul=="Med" else 0
        elif m=="LPD" and LPD<12: X_test["LPD_Wm2"]=LPD
        elif m=="HVAC" and hvac>24: X_test["HVAC_Setpoint_C"]=hvac
        elif m=="Shading" and shading>0: X_test["ShadingDepth_m"]=shading
        elif m=="Schedule" and schedule=="Adjusted": X_test["ScheduleAdj_Base"]=0
        elif m=="LinearControl" and ctrl=="Yes": X_test["LinearControl_Yes"]=1
        elif m=="Albedo" and albedo=="Cool": X_test["HighAlbedoWall_Cool"]=1
        if X_test.equals(X_base): continue

        E_new = models["Lighting_kWh"].predict(X_test)[0]+models["Cooling_kWh"].predict(X_test)[0]+BASELINE["Room_kWh"]
        ΔE = E_base - E_new
        ΔC = 0
        if m=="Glazing": ΔC = glazing_cost_double*WinA if glazing=="Double" else glazing_cost_lowe*WinA
        elif m=="Insulation": ΔC = insul_cost_med*total_wall_roof if insul=="Med" else insul_cost_high*total_wall_roof
        elif m=="LPD": ΔC = led_cost*GFA
        elif m=="HVAC": ΔC = hvac_cost
        elif m=="Shading": ΔC = shading_cost*WinA
        elif m=="Schedule": ΔC = schedule_cost
        elif m=="LinearControl": ΔC = linearctrl_cost*GFA
        elif m=="Albedo": ΔC = albedo_cost*total_wall_roof
        results.append({"Measure":m,"Energy_Saving_kWh":ΔE,"Cost_SGD":ΔC})

    contrib_df=pd.DataFrame(results).sort_values("Energy_Saving_kWh",ascending=False)

    impact_choice=st.selectbox("Choose Visualization",["Bar","Waterfall","Radar"])
    if impact_choice=="Bar":
        fig1=px.bar(contrib_df,x="Energy_Saving_kWh",y="Measure",orientation="h",
                    color="Measure",color_discrete_sequence=palette,
                    labels={"Energy_Saving_kWh":"Energy Saving (kWh)"})
        st.plotly_chart(fig1,use_container_width=True)
        fig2=px.bar(contrib_df,x="Cost_SGD",y="Measure",orientation="h",
                    color="Measure",color_discrete_sequence=palette,
                    labels={"Cost_SGD":"Retrofit Cost (SGD)"})
        st.plotly_chart(fig2,use_container_width=True)
    elif impact_choice=="Waterfall":
        fig3=go.Figure(go.Waterfall(x=contrib_df["Measure"],y=contrib_df["Energy_Saving_kWh"],
                    connector={"line":{"color":"#243C2C"}},increasing={"marker":{"color":"#a3b565"}}))
        fig3.update_layout(title="Energy Saving Contribution (kWh)")
        st.plotly_chart(fig3,use_container_width=True)
        fig4=go.Figure(go.Waterfall(x=contrib_df["Measure"],y=contrib_df["Cost_SGD"],
                    connector={"line":{"color":"#243C2C"}},increasing={"marker":{"color":"#c4c3e3"}}))
        fig4.update_layout(title="Retrofit Cost Contribution (SGD)")
        st.plotly_chart(fig4,use_container_width=True)
    else:
        cats=list(contrib_df["Measure"])+[contrib_df["Measure"].iloc[0]]
        valsE=list(contrib_df["Energy_Saving_kWh"])+[contrib_df["Energy_Saving_kWh"].iloc[0]]
        valsC=list(contrib_df["Cost_SGD"])+[contrib_df["Cost_SGD"].iloc[0]]
        fig5=go.Figure(go.Scatterpolar(r=valsE,theta=cats,fill="toself",marker=dict(color="#a3b565")))
        fig5.update_layout(title="Energy Saving Contribution (kWh)")
        st.plotly_chart(fig5,use_container_width=True)
        fig6=go.Figure(go.Scatterpolar(r=valsC,theta=cats,fill="toself",marker=dict(color="#c4c3e3")))
        fig6.update_layout(title="Retrofit Cost Contribution (SGD)")
        st.plotly_chart(fig6,use_container_width=True)

    # --- Weighted Impact Index (Table only) ---
    st.subheader(" Weighted Impact Index")
    st.caption("Adjust weights to balance between energy saving and cost efficiency.")
    w1 = st.slider("Weight: Energy Saving", 0.0, 1.0, 0.6)
    w2 = st.slider("Weight: Retrofit Cost (inverse)", 0.0, 1.0, 0.4)
    contrib_df["Impact_Index"]=w1*contrib_df["Energy_Saving_kWh"]-w2*(contrib_df["Cost_SGD"]/1000)
    min_idx,max_idx=contrib_df["Impact_Index"].min(),contrib_df["Impact_Index"].max()
    if max_idx!=min_idx:
        contrib_df["Impact_Index_Scaled"]=1+9*(contrib_df["Impact_Index"]-min_idx)/(max_idx-min_idx)
    else:
        contrib_df["Impact_Index_Scaled"]=5
    contrib_df=contrib_df.sort_values("Impact_Index_Scaled",ascending=False).reset_index(drop=True)
    st.markdown("<p style='font-size:15px;'><b>Impact Index Formula:</b></p>",unsafe_allow_html=True)
    st.latex(r"""\small \text{Impact Index}_i=(w_1\times E_{\text{saving}})-(w_2\times\frac{Cost}{1{,}000})""")
    st.dataframe(contrib_df[["Measure","Energy_Saving_kWh","Cost_SGD","Impact_Index_Scaled"]],use_container_width=True)
    top_measure=contrib_df.iloc[0]
    total_index=contrib_df["Impact_Index_Scaled"].sum()
    st.markdown("---")
    st.markdown(f" **Top-performing measure:** `{top_measure['Measure']}` with score **{top_measure['Impact_Index_Scaled']:.1f} / 10**")
    st.markdown(f" **Overall combined index (sum of all measures):** **{total_index:.1f} / {10*len(contrib_df):.0f}**")
    st.caption("Higher index = better combined performance after weighting.")

# -----------------------------------------------------
# ⚖️ Trade-off Explorer Tab (Model-Driven Optimization)
# -----------------------------------------------------
with tabs[4]:
    st.subheader("Trade-off Explorer")
    st.caption("Use surrogate models to explore energy–economic trade-offs and find retrofit combinations meeting your targets.")

    import joblib
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px

    # === 1️⃣ Load trained surrogate models ===
    @st.cache_resource
    def load_models():
        models = {
            "cooling": joblib.load("LR_Cooling_kWh_model.pkl"),
            "lighting": joblib.load("XGB_Lighting_kWh_model.pkl")
        }
        return models

    models = load_models()

    # === 2️⃣ Generate random retrofit combinations (once, reproducible) ===
    np.random.seed(42)
    n_samples = 500
    LHS = np.random.rand(n_samples, 8)
    data = pd.DataFrame({
        "Glazing": np.where(LHS[:,0] < 0.33, "Single",
                     np.where(LHS[:,0] < 0.66, "Double", "LowE")),
        "Insulation": np.where(LHS[:,1] < 0.5, "Med", "High"),
        "LPD_Wm2": 8 + LHS[:,2] * (12 - 8),
        "HVAC_Setpoint_C": 24 + LHS[:,3] * (27 - 24),
        "ShadingDepth_m": LHS[:,4] * 1.0,
        "ScheduleAdj": np.where(LHS[:,5] < 0.5, "Base", "Tight"),
        "LinearControl": np.where(LHS[:,6] < 0.5, "No", "Yes"),
        "HighAlbedoWall": np.where(LHS[:,7] < 0.5, "Base", "Cool")
    })

    # === 3️⃣ Encode + predict with surrogate models ===
    data_encoded = pd.get_dummies(data, drop_first=False)
    for model in models.values():
        for col in model.feature_names_in_:
            if col not in data_encoded.columns:
                data_encoded[col] = 0

    data["Cooling_kWh"] = models["cooling"].predict(data_encoded[models["cooling"].feature_names_in_])
    data["Lighting_kWh"] = models["lighting"].predict(data_encoded[models["lighting"].feature_names_in_])

    # Room electricity fixed by insulation & albedo
    room_lookup = {
        ("Low", "Base"): 31598.3,
        ("Low", "Cool"): 31556.6,
        ("Med", "Base"): 31452.5,
        ("Med", "Cool"): 31410.9,
        ("High", "Base"): 31203.0,
        ("High", "Cool"): 31161.4
    }
    data["Room_kWh"] = data.apply(lambda r: room_lookup.get((r["Insulation"], r["HighAlbedoWall"]), 31452.5), axis=1)
    data["Total_kWh"] = data["Cooling_kWh"] + data["Lighting_kWh"] + data["Room_kWh"]

    # === 4️⃣ Economic calculations (dynamic unit rates) ===
    baseline_energy = BASELINE["Total_kWh"]
    data["Retrofit Cost (SGD)"] = (
        (data["Glazing"] == "Double") * glazing_cost_double * WinA +
        (data["Glazing"] == "LowE") * glazing_cost_lowe * WinA +
        (data["Insulation"] == "Med") * insul_cost_med * total_wall_roof +
        (data["Insulation"] == "High") * insul_cost_high * total_wall_roof +
        (12 - data["LPD_Wm2"]) * led_cost * GFA +
        (data["HVAC_Setpoint_C"] > 24) * hvac_cost +
        (data["ShadingDepth_m"]) * shading_cost * WinA +
        (data["LinearControl"] == "Yes") * linearctrl_cost * GFA +
        (data["HighAlbedoWall"] == "Cool") * albedo_cost * total_wall_roof
    )
    data["Energy Saving (%)"] = (1 - data["Total_kWh"] / baseline_energy) * 100
    data["EUI (kWh/m²·yr)"] = data["Total_kWh"] / GFA
    data["Annual Saving (SGD)"] = (baseline_energy - data["Total_kWh"]) * tariff
    data["Payback (yrs)"] = data["Retrofit Cost (SGD)"] / data["Annual Saving (SGD)"]

    # === 5️⃣ Pareto computation ===
    def pareto_front(df, x_col, y_col):
        points = df[[x_col, y_col]].values
        is_dominated = np.zeros(len(points), dtype=bool)
        for i, p in enumerate(points):
            if any((points[:, 0] >= p[0]) & (points[:, 1] <= p[1]) &
                   ((points[:, 0] > p[0]) | (points[:, 1] < p[1]))):
                is_dominated[i] = True
        return df[~is_dominated]

    pareto_df = pareto_front(data, "Energy Saving (%)", "Payback (yrs)").sort_values("Energy Saving (%)")

    # === 6️⃣ Visualization selector ===
    trade_type = st.selectbox("Choose Visualization", ["Pareto Front", "2D Contour"])

    # === 7️⃣ Pareto Front Mode ===
    if trade_type == "Pareto Front":
        st.markdown("### 🎯 Set Your Targets")
        col1, col2 = st.columns(2)
        target_saving = col1.slider("Minimum Energy Saving (%)", 0, 50, 25, step=1)
        max_payback = col2.slider("Maximum Payback (years)", 1, 12, 6, step=1)

        feasible = data[(data["Energy Saving (%)"] >= target_saving) &
                        (data["Payback (yrs)"] <= max_payback)].copy()

        # --- Pareto Chart ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["Energy Saving (%)"], y=data["Payback (yrs)"],
            mode="markers", name="All Predictions",
            marker=dict(color="#c4c3e3", size=7, opacity=0.5)
        ))
        fig.add_trace(go.Scatter(
            x=pareto_df["Energy Saving (%)"], y=pareto_df["Payback (yrs)"],
            mode="lines+markers", name="Pareto Front",
            line=dict(color="#7A9544", width=3)
        ))

        if not feasible.empty:
            feasible = feasible.reset_index(drop=True)
            feasible["Case_ID"] = feasible.index + 1
            fig.add_trace(go.Scatter(
                x=feasible["Energy Saving (%)"], y=feasible["Payback (yrs)"],
                mode="markers", name="Feasible Solutions",
                marker=dict(color="#504e76", size=8)
            ))
            best_case = feasible.sort_values(by="Energy Saving (%)", ascending=False).iloc[0]
            fig.add_trace(go.Scatter(
                x=[best_case["Energy Saving (%)"]], y=[best_case["Payback (yrs)"]],
                mode="markers", name="Best Feasible Option",
                marker=dict(color="#f1642e", size=11, line=dict(color="white", width=1))
            ))

        fig.update_yaxes(autorange="reversed", title="Payback (years)")
        fig.update_xaxes(title="Energy Saving (%)")
        fig.update_layout(
            title="Model-Predicted Trade-off Frontier",
            font=dict(color="#243C2C"),
            legend=dict(orientation="h", y=-0.25),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Feasible Table ---
        if not feasible.empty:
            st.success(f"{len(feasible)} feasible retrofit option(s) found meeting your targets.")
            show_cols = ["Case_ID","Glazing","Insulation","LPD_Wm2","HVAC_Setpoint_C",
                         "ShadingDepth_m","ScheduleAdj","LinearControl","HighAlbedoWall",
                         "Energy Saving (%)","EUI (kWh/m²·yr)","Payback (yrs)"]
            st.dataframe(feasible[show_cols].sort_values(by="Payback (yrs)"), use_container_width=True)

            combo_info = f"""
            <p style='color:#f1642e; font-weight:bold; font-size:16px;'>
                Best Feasible Option – Case {int(best_case["Case_ID"])}<br>
                Glazing: {best_case["Glazing"]} &nbsp;|&nbsp;
                Insulation: {best_case["Insulation"]} &nbsp;|&nbsp;
                LPD: {best_case["LPD_Wm2"]:.1f} W/m² &nbsp;|&nbsp;
                HVAC: {best_case["HVAC_Setpoint_C"]:.1f} °C &nbsp;|&nbsp;
                Shading: {best_case["ShadingDepth_m"]:.2f} m<br>
                Schedule: {best_case["ScheduleAdj"]} &nbsp;|&nbsp;
                Linear Control: {best_case["LinearControl"]} &nbsp;|&nbsp;
                Albedo: {best_case["HighAlbedoWall"]}<br>
                Energy Saving: {best_case["Energy Saving (%)"]:.1f}% &nbsp;|&nbsp;
                EUI: {best_case["EUI (kWh/m²·yr)"]:.1f} &nbsp;|&nbsp;
                Payback: {best_case["Payback (yrs)"]:.1f} years
            </p>
            """
            st.markdown(combo_info, unsafe_allow_html=True)
        else:
            st.warning("No combination meets your targets. Try adjusting thresholds.")

        # --- Cluster Explorer (only for Pareto) ---
        st.markdown("### 🔍 Cluster Explorer")
        st.caption("Explore how each measure distributes across the trade-off space.")
        measure_choice = st.selectbox(
            "Select a measure to visualize:",
            ["HVAC_Setpoint_C", "LPD_Wm2", "ShadingDepth_m", "ScheduleAdj", "Glazing", "Insulation", "LinearControl", "HighAlbedoWall"]
        )
        if measure_choice in ["HVAC_Setpoint_C", "LPD_Wm2", "ShadingDepth_m"]:
            fig_cluster = px.scatter(
                data, x="Energy Saving (%)", y="Payback (yrs)",
                color=measure_choice, color_continuous_scale="Viridis",
                title=f"Cluster Distribution by {measure_choice}",
            )
        else:
            fig_cluster = px.scatter(
                data, x="Energy Saving (%)", y="Payback (yrs)",
                color=measure_choice,
                color_discrete_sequence=["#a3b565", "#fcdd9d", "#c4c3e3", "#504e76"],
                title=f"Cluster Distribution by {measure_choice}",
            )
        fig_cluster.update_yaxes(autorange="reversed", title="Payback (years)")
        fig_cluster.update_xaxes(title="Energy Saving (%)")
        fig_cluster.update_layout(height=500, font=dict(color="#243C2C"), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_cluster, use_container_width=True)

    # === 8️⃣ 2D Contour Mode ===
elif trade_type == "2D Contour":
    st.markdown("### 🌈 2D Contour Analysis")
    kpi_choice = st.selectbox("Select KPI:", ["Energy Saving (%)", "EUI (kWh/m²·yr)", "Payback (yrs)"])
    x_measure = st.selectbox(
        "X-axis measure:",
        ["HVAC_Setpoint_C", "LPD_Wm2", "ShadingDepth_m", "ScheduleAdj", "Glazing", "Insulation", "LinearControl", "HighAlbedoWall"],
        index=0
    )
    y_measure = st.selectbox(
        "Y-axis measure:",
        ["HVAC_Setpoint_C", "LPD_Wm2", "ShadingDepth_m", "ScheduleAdj", "Glazing", "Insulation", "LinearControl", "HighAlbedoWall"],
        index=1
    )

    if x_measure == y_measure:
        st.warning("Please select two different measures.")
    else:
        # check if variables are categorical or continuous
        x_is_cat = data[x_measure].dtype == 'object'
        y_is_cat = data[y_measure].dtype == 'object'

        if x_is_cat and y_is_cat:
            # both categorical → fixed order for known variables
            cat_order = {
                "Glazing": ["Single", "Double", "LowE"],
                "Insulation": ["Low", "Med", "High"],
                "ScheduleAdj": ["Base", "Tight"],
                "LinearControl": ["No", "Yes"],
                "HighAlbedoWall": ["Base", "Cool"]
            }
            x_levels = cat_order.get(x_measure, sorted(np.unique(data[x_measure])))
            y_levels = cat_order.get(y_measure, sorted(np.unique(data[y_measure])))

            combos = pd.MultiIndex.from_product([x_levels, y_levels], names=[x_measure, y_measure])
            contour_df = pd.DataFrame(index=combos).reset_index()

        else:
            # at least one continuous → create meshgrid (2 decimal precision)
            grid_x = np.linspace(data[x_measure].min(), data[x_measure].max(), 30) if not x_is_cat else np.unique(data[x_measure])
            grid_y = np.linspace(data[y_measure].min(), data[y_measure].max(), 30) if not y_is_cat else np.unique(data[y_measure])
            X, Y = np.meshgrid(np.round(grid_x, 2), np.round(grid_y, 2))
            contour_df = pd.DataFrame({
                x_measure: X.flatten(),
                y_measure: Y.flatten()
            })

        # fill other measures with default or mean values
        fixed_values = {
            "Glazing": "LowE",
            "Insulation": "High",
            "ScheduleAdj": "Base",
            "LinearControl": "No",
            "HighAlbedoWall": "Base"
        }
        for k, v in fixed_values.items():
            if k not in contour_df.columns:
                contour_df[k] = v

        for m in ["LPD_Wm2", "HVAC_Setpoint_C", "ShadingDepth_m"]:
            if m not in [x_measure, y_measure]:
                contour_df[m] = np.round(data[m].mean(), 2)

        # === Predict KPIs ===
        contour_encoded = pd.get_dummies(contour_df, drop_first=False)
        for model in models.values():
            for col in model.feature_names_in_:
                if col not in contour_encoded.columns:
                    contour_encoded[col] = 0

        contour_df["Cooling_kWh"] = models["cooling"].predict(contour_encoded[models["cooling"].feature_names_in_])
        contour_df["Lighting_kWh"] = models["lighting"].predict(contour_encoded[models["lighting"].feature_names_in_])
        contour_df["Total_kWh"] = contour_df["Cooling_kWh"] + contour_df["Lighting_kWh"]
        contour_df["EUI (kWh/m²·yr)"] = contour_df["Total_kWh"] / GFA
        contour_df["Energy Saving (%)"] = (1 - contour_df["Total_kWh"] / baseline_energy) * 100
        contour_df["Payback (yrs)"] = 50000 / ((baseline_energy - contour_df["Total_kWh"]) * tariff)

        # === Reshape for contour plotting ===
        if x_is_cat and y_is_cat:
            Z = contour_df.pivot(index=y_measure, columns=x_measure, values=kpi_choice)
            x_vals, y_vals = np.arange(len(Z.columns)), np.arange(len(Z.index))
        else:
            Z = contour_df[kpi_choice].values.reshape(
                len(np.unique(contour_df[y_measure])),
                len(np.unique(contour_df[x_measure]))
            )
            x_vals = np.unique(np.round(contour_df[x_measure], 2))
            y_vals = np.unique(np.round(contour_df[y_measure], 2))

        # === Colors ===
        if not x_is_cat and not y_is_cat:
            colorscale = [[0, "#a3b565"], [0.5, "#fcdd9d"], [1, "#c4c3e3"]]
        else:
            colorscale = [
                [0, "#E0D1E6"], [0.2, "#DBC7E0"], [0.4, "#CFB6D6"],
                [0.6, "#AF93BB"], [0.8, "#9C83A3"], [1, "#897191"]
            ]

        # === Draw contour ===
        fig_contour = go.Figure(data=go.Contour(
            z=Z if isinstance(Z, np.ndarray) else Z.values,
            x=x_vals if isinstance(Z, np.ndarray) else np.arange(len(Z.columns)),
            y=y_vals if isinstance(Z, np.ndarray) else np.arange(len(Z.index)),
            colorscale=colorscale,
            contours=dict(showlabels=True, labelfont=dict(size=10, color="black")),
            colorbar=dict(title=kpi_choice),
        ))

        # === Layout and axis labels ===
        fig_contour.update_layout(
            title=f"Iso-performance Map: {kpi_choice}<br>({x_measure} vs {y_measure})",
            xaxis_title=x_measure,
            yaxis_title=y_measure,
            xaxis=dict(
                tickmode='array',
                tickvals=np.arange(len(Z.columns)) if x_is_cat and y_is_cat else x_vals,
                ticktext=Z.columns if x_is_cat and y_is_cat else [f"{v:.2f}" for v in x_vals]
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=np.arange(len(Z.index)) if x_is_cat and y_is_cat else y_vals,
                ticktext=Z.index if x_is_cat and y_is_cat else [f"{v:.2f}" for v in y_vals]
            ),
            font=dict(color="#243C2C"),
            height=500
        )

        st.plotly_chart(fig_contour, use_container_width=True)
