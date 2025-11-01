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
room_elec = BASELINE["Room_kWh"]
total_energy = lighting_pred + cooling_pred + room_elec

energy_saving_pct = (BASELINE["Total_kWh"]-total_energy)/BASELINE["Total_kWh"]*100
cool_saving_pct   = (BASELINE["Cooling_Load_kWh"]-cool_load_pred)/BASELINE["Cooling_Load_kWh"]*100
EUI = total_energy / GFA
carbon_emission = total_energy * carbon_factor
annual_saving = (BASELINE["Total_kWh"]-total_energy)*tariff
payback_years = CAPEX/annual_saving if annual_saving>0 else None

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
        comment = "Excellent performance – your building is among Singapore’s most energy-efficient offices."
    elif EUI <= 142:
        quartile_text = "2nd Quartile"
        comment = "Good performance – your building performs better than the national median."
    elif EUI <= 184:
        quartile_text = "3rd Quartile"
        comment = "Moderate performance – your building performs close to the national average."
    else:
        quartile_text = "Bottom Quartile"
        comment = "Below average – your building consumes more energy than typical offices."

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
    st.metric("Carbon Emission (kg CO₂e)", f"{carbon_emission:,.1f}")
    st.metric("Carbon Factor (kgCO₂/kWh)", f"{carbon_factor:.2f}")

    carbon_intensity = carbon_emission / GFA  # kg CO₂e/m²·yr

    # --- Benchmark classification ---
    if carbon_intensity <= 55:
        carbon_comment = "Excellent – aligns with Green Mark Platinum benchmark."
        carbon_level = "Platinum"
    elif carbon_intensity <= 70:
        carbon_comment = "Good – aligns with Green Mark Gold benchmark."
        carbon_level = "Gold"
    elif carbon_intensity <= 85:
        carbon_comment = "Average – comparable to typical Singapore offices."
        carbon_level = "Average"
    else:
        carbon_comment = "High – above national average operational carbon intensity."
        carbon_level = "Below Benchmark"

    # --- Refined Gauge Chart for Carbon Intensity ---
    fig_carbon = go.Figure(go.Indicator(
        mode="gauge+number",
        value=carbon_intensity,
        domain={'x': [0, 1], 'y': [0, 0.85]},
        title={'text': "Operational Carbon Intensity (kg CO₂e/m²·yr)",
               'font': {'size': 15, 'color': '#243C2C'},
               'offset': 30},
        number={'font': {'size': 20, 'color': '#243C2C'}},
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
        margin=dict(t=40, b=10, l=10, r=10),
        height=240,
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

    # --- Refined Gauge Chart for Payback Years ---
    fig_payback = go.Figure(go.Indicator(
        mode="gauge+number",
        value=payback_years,
        domain={'x': [0, 1], 'y': [0, 0.85]},
        title={'text': "Retrofit Payback Period (Years)",
               'font': {'size': 15, 'color': '#243C2C'},
               'offset': 30},
        number={'font': {'size': 20, 'color': '#243C2C'}},
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
        margin=dict(t=40, b=10, l=10, r=10),
        height=240,
        paper_bgcolor="white",
        font=dict(color="#243C2C", family="Arial"),
    )
    st.plotly_chart(fig_payback, use_container_width=True)

    # --- Summary Message ---
    msg = f"Your building achieves {energy_saving_pct:.1f}% energy saving with a payback of {payback_years:.1f} years."

    if payback_years <= 5:
        payback_comment = "Excellent – rapid return on investment."
    elif payback_years <= 7:
        payback_comment = "Good – in line with national benchmark."
    elif payback_years <= 9:
        payback_comment = "Moderate – slightly longer than typical retrofits."
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
# ⚖️ Trade-off Explorer Tab
# -----------------------------------------------------
with tabs[4]:
    st.subheader("Trade-off Explorer")
    st.caption("Explore trade-offs between energy savings, cost, and payback performance.")
    trade_type=st.selectbox("Choose Visualization",["Pareto Front","2D Contour","Animated"])
    if trade_type=="Pareto Front":
        df=pd.DataFrame({"Energy Saving (%)":[10,20,30,35,40],"Payback (yrs)":[2,4,6,8,10]})
        fig=px.line(df,x="Energy Saving (%)",y="Payback (yrs)",markers=True,
                    title="Pareto Front: Energy vs Payback",color_discrete_sequence=["#a3b565"])
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig,use_container_width=True)
    elif trade_type=="2D Contour":
        x=np.linspace(8,14,30);y=np.linspace(24,27,30)
        X,Y=np.meshgrid(x,y)
        Z=180-(Y-24)*10-(14-X)*5
        fig=go.Figure(data=go.Contour(z=Z,x=x,y=y,colorscale=[[0,"#c4c3e3"],[0.5,"#fcdd9d"],[1,"#a3b565"]]))
        fig.update_layout(title="Iso-performance Map (EUI vs LPD & HVAC)",xaxis_title="LPD (W/m²)",yaxis_title="HVAC Setpoint (°C)")
        st.plotly_chart(fig,use_container_width=True)
    else:
        df_anim=pd.DataFrame({"Energy Saving %":[10,20,30,35,40,45],"Payback":[2,4,6,8,10,12],
                              "Scenario":["LED","Envelope","Passive+Active","Deep","Smart","Extreme"]})
        fig=px.scatter(df_anim,x="Energy Saving %",y="Payback",animation_frame="Scenario",size="Energy Saving %",
                       color="Scenario",color_discrete_sequence=palette,title="Animated Trade-off Explorer")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig,use_container_width=True)
