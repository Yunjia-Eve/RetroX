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

    # === 1️⃣ Load trained surrogate models ===
    @st.cache_resource
    def load_models():
        models = {
            "cooling": joblib.load("LR_Cooling_kWh_model.pkl"),
            "lighting": joblib.load("XGB_Lighting_kWh_model.pkl")
        }
        return models

    models = load_models()

    # === 2️⃣ Generate random retrofit combinations (cached, fixed seed) ===
    @st.cache_resource
    def generate_dataset(models):
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

        # --- Predict energy using surrogate models ---
        data["Cooling_kWh"] = models["cooling"].predict(data[models["cooling"].feature_names_in_])
        data["Lighting_kWh"] = models["lighting"].predict(data[models["lighting"].feature_names_in_])

        # --- Room electricity fixed by insulation & albedo ---
        room_lookup = {
            ("Low", "Base"): 31598.3,
            ("Low", "Cool"): 31556.6,
            ("Med", "Base"): 31452.5,
            ("Med", "Cool"): 31410.9,
            ("High", "Base"): 31203.0,
            ("High", "Cool"): 31161.4
        }
        data["Room_kWh"] = data.apply(lambda r: room_lookup[(r["Insulation"], r["HighAlbedoWall"])], axis=1)
        data["Total_kWh"] = data["Cooling_kWh"] + data["Lighting_kWh"] + data["Room_kWh"]

        # --- Retrofit cost (same logic as main toolkit) ---
        WinA = 214.15
        RoofA, WallA, GFA = 939.62, 397.7, 939.62
        total_wall_roof = RoofA + WallA

        glazing_cost_double, glazing_cost_lowe = 200, 300
        insul_cost_med, insul_cost_high = 45, 55
        shading_cost, led_cost, hvac_cost = 120, 25, 2000
        albedo_cost, schedule_cost, linearctrl_cost = 25, 2000, 30

        data["Retrofit Cost (SGD)"] = (
            np.where(data["Glazing"] == "Double", glazing_cost_double * WinA,
            np.where(data["Glazing"] == "LowE", glazing_cost_lowe * WinA, 0)) +
            np.where(data["Insulation"] == "Med", insul_cost_med * total_wall_roof,
            np.where(data["Insulation"] == "High", insul_cost_high * total_wall_roof, 0)) +
            np.where(data["ShadingDepth_m"] > 0, shading_cost * WinA, 0) +
            np.where(data["LPD_Wm2"] < 10, led_cost * GFA, 0) +
            np.where(data["HVAC_Setpoint_C"] > 24, hvac_cost, 0) +
            np.where(data["HighAlbedoWall"] == "Cool", albedo_cost * total_wall_roof, 0) +
            np.where(data["ScheduleAdj"] == "Tight", schedule_cost, 0) +
            np.where(data["LinearControl"] == "Yes", linearctrl_cost * GFA, 0)
        )

        return data

    data = generate_dataset(models)

    # === 3️⃣ Calculate energy & payback ===
    baseline_energy = BASELINE["Total_kWh"]
    data["Energy Saving (%)"] = (1 - data["Total_kWh"] / baseline_energy) * 100
    data["Annual Saving (SGD)"] = (baseline_energy - data["Total_kWh"]) * tariff
    data["Payback (yrs)"] = data["Retrofit Cost (SGD)"] / data["Annual Saving (SGD)"]

    # === 4️⃣ User target inputs ===
    st.markdown("### 🎯 Set Your Targets")
    col1, col2 = st.columns(2)
    target_saving = col1.slider("Minimum Energy Saving (%)", 0, 50, 25, step=1)
    max_payback = col2.slider("Maximum Payback (years)", 1, 12, 6, step=1)

    feasible = data[
        (data["Energy Saving (%)"] >= target_saving) &
        (data["Payback (yrs)"] <= max_payback)
    ]

    # === 5️⃣ Compute global Pareto front ===
    def pareto_front(df, x_col, y_col):
        points = df[[x_col, y_col]].values
        is_dominated = np.zeros(len(points), dtype=bool)
        for i, p in enumerate(points):
            if any(
                (points[:, 0] >= p[0]) & (points[:, 1] <= p[1]) &
                ((points[:, 0] > p[0]) | (points[:, 1] < p[1]))
            ):
                is_dominated[i] = True
        return df[~is_dominated]

    pareto_df = pareto_front(data, "Energy Saving (%)", "Payback (yrs)").sort_values("Energy Saving (%)")

    # === 6️⃣ Plot Pareto front ===
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
        fig.add_trace(go.Scatter(
            x=feasible["Energy Saving (%)"], y=feasible["Payback (yrs)"],
            mode="markers", name="Feasible Solutions",
            marker=dict(color="#504e76", size=8)
        ))
        # Highlight best feasible (highest saving)
        best_case = feasible.sort_values(by="Energy Saving (%)", ascending=False).head(1).iloc[0]
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

    # === 7️⃣ Display results ===
    if not feasible.empty:
        feasible = feasible.reset_index(drop=True)
        feasible["Case_ID"] = feasible.index + 1
        st.success(f"{len(feasible)} feasible retrofit option(s) found meeting your targets.")
        show_cols = ["Case_ID","Glazing","Insulation","LPD_Wm2","HVAC_Setpoint_C",
                     "ShadingDepth_m","ScheduleAdj","LinearControl","HighAlbedoWall",
                     "Energy Saving (%)","Payback (yrs)"]
        st.dataframe(feasible[show_cols].sort_values(by="Payback (yrs)"), use_container_width=True)

        # Display best feasible option details
        combo_info = f"""
        <p style='color:#f1642e; font-weight:bold; font-size:16px;'>
            Best Feasible Option – Case {int(best_case['Case_ID'])}<br>
            Glazing: {best_case['Glazing']} &nbsp;|&nbsp;
            Insulation: {best_case['Insulation']} &nbsp;|&nbsp;
            LPD: {best_case['LPD_Wm2']:.1f} W/m² &nbsp;|&nbsp;
            HVAC: {best_case['HVAC_Setpoint_C']:.1f} °C &nbsp;|&nbsp;
            Shading: {best_case['ShadingDepth_m']:.2f} m<br>
            Schedule: {best_case['ScheduleAdj']} &nbsp;|&nbsp;
            Linear Control: {best_case['LinearControl']} &nbsp;|&nbsp;
            Albedo: {best_case['HighAlbedoWall']}<br>
            Energy Saving: {best_case['Energy Saving (%)']:.1f}% &nbsp;|&nbsp;
            Payback: {best_case['Payback (yrs)']:.1f} years
        </p>
        """
        st.markdown(combo_info, unsafe_allow_html=True)
    else:
        st.warning("No combination meets your targets. Try adjusting thresholds.")

    # === 8️⃣ Guidance ===
    st.markdown("""
    <div style='background-color:#eef6fb; padding:15px; border-radius:8px; font-size:15px;'>
        <span style='color:#7A9544; font-weight:bold;'>Pareto front</span> shows globally optimal trade-offs between energy savings and payback.<br>
        <span style='color:#504e76; font-weight:bold;'>Feasible solutions</span> update dynamically with your targets.<br>
        <span style='color:#f1642e; font-weight:bold;'>Best feasible option</span> achieves the highest savings within your limits and is detailed above.
    </div>
    """, unsafe_allow_html=True)
