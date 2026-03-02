import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================================
# LOAD ARTIFACTS
# =====================================================

model = joblib.load("energy_model.pkl")
best_solution = joblib.load("best_solution.pkl")
df_time_active = joblib.load("active_data.pkl")
golden_signature = joblib.load("golden_signature.pkl")

FEATURES = [
    "Temperature_C",
    "Pressure_Bar",
    "Motor_Speed_RPM",
    "Flow_Rate_LPM",
    "Vibration_mm_s"
]

EMISSION_FACTOR = 0.82
ELECTRICITY_COST = 8.0
BATCHES_PER_DAY = 20
WORKING_DAYS = 25
SYSTEM_COST = 200000

# =====================================================
# HEADER
# =====================================================

st.title("🏭 Smart Manufacturing Optimization Platform")

# =====================================================
# CONTROLS
# =====================================================

system_mode = st.selectbox(
    "⚙ System Mode",
    ["Production Mode", "Research Mode"]
)

objective_mode = st.selectbox(
    "🎯 Optimization Priority",
    ["Balanced", "Energy Saving", "Reliability", "Low Carbon"]
)

view_mode = st.selectbox(
    "🧭 Dashboard View",
    ["Executive View", "Engineer View"]
)

st.info(f"Optimization Profile: {objective_mode}")

# =====================================================
# CURRENT STATE
# =====================================================

current = df_time_active.iloc[-1][FEATURES]
current_energy = model.predict(current.values.reshape(1, -1))[0]

golden = golden_signature[FEATURES]
golden_energy = golden_signature["Power_Consumption_kW"]

# ---- Dual Mode AI Logic ----
if system_mode == "Production Mode":
    optimal = golden
    optimized_energy = golden_energy
else:
    optimal = pd.Series(best_solution, index=FEATURES)
    optimized_energy = model.predict(
        np.array(best_solution).reshape(1, -1)
    )[0]

# =====================================================
# KPI CALCULATIONS
# =====================================================

energy_saved = current_energy - optimized_energy
savings_pct = (energy_saved / current_energy) * 100

money_saved_batch = energy_saved * ELECTRICITY_COST
monthly_savings = money_saved_batch * BATCHES_PER_DAY * WORKING_DAYS
annual_savings = monthly_savings * 12
roi_percent = (annual_savings / SYSTEM_COST) * 100
carbon_saved = energy_saved * EMISSION_FACTOR

# =====================================================
# PHASE INTELLIGENCE
# =====================================================

if "Phase" in df_time_active.columns:

    current_phase = df_time_active["Phase"].iloc[-1]

    phase_transition = (
        df_time_active["Phase"].iloc[-1]
        != df_time_active["Phase"].iloc[-2]
    )

else:
    current_phase = "Unknown"
    phase_transition = False

# =====================================================
# EXECUTIVE VIEW
# =====================================================

if view_mode == "Executive View":

    st.header("📊 Executive Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Energy Improvement", f"{round(savings_pct,2)} %")
    c2.metric("Monthly Savings", f"₹ {round(monthly_savings,2)}")
    c3.metric("ROI", f"{round(roi_percent,2)} %")

    c4, c5 = st.columns(2)
    c4.metric("Annual Savings", f"₹ {round(annual_savings,2)}")
    c5.metric("Carbon Reduction", f"{round(carbon_saved,2)} kg CO₂")

    st.write("**System Mode:**", system_mode)
    st.success("AI Optimization Active")

# =====================================================
# ENGINEER VIEW
# =====================================================

else:

    st.header("🧠 Engineer Dashboard")

    st.info(f"Current Phase: {current_phase}")

    if phase_transition:
        st.warning("⚡ Phase Transition Detected")

    st.subheader("Current vs Golden vs Optimized")

    compare_df = pd.DataFrame({
        "Current": current,
        "Golden": golden,
        "Optimized": optimal
    })

    st.dataframe(compare_df)

    # --------------------------
    # AI Recommendations
    # --------------------------

    st.subheader("AI Recommendations")

    for key in FEATURES:

        diff = optimal[key] - current[key]

        if abs(diff) < 0.5:
            continue

        if diff > 0:
            st.success(f"Increase {key} by {round(diff,2)}")
        else:
            st.warning(f"Decrease {key} by {round(abs(diff),2)}")

    # --------------------------
    # Feature Importance
    # --------------------------

    st.subheader("Feature Importance (Energy Drivers)")

    importance = model.feature_importances_

    fig, ax = plt.subplots()
    ax.bar(FEATURES, importance)
    plt.xticks(rotation=30)

    st.pyplot(fig)

    # --------------------------
    # Human in Loop
    # --------------------------

    st.subheader("👨‍🏭 Operator Approval")

    decision = st.radio(
        "Approve AI Optimization?",
        ["Pending Review", "Approve", "Reject"]
    )

    if decision == "Approve":
        st.success("Optimization Approved")
    elif decision == "Reject":
        st.error("Optimization Rejected")
    else:
        st.info("Awaiting Operator Decision")

# =====================================================
# SCENARIO SIMULATION ENGINE
# =====================================================

st.subheader("🧪 Scenario Simulation (What-If Analysis)")

sim_temp = st.slider("Temperature", 20.0, 80.0, 40.0)
sim_pressure = st.slider("Pressure", 0.5, 2.0, 1.0)
sim_motor = st.slider("Motor Speed", 50.0, 200.0, 120.0)
sim_flow = st.slider("Flow Rate", 1.0, 6.0, 3.0)
sim_vibration = st.slider("Vibration", 0.5, 3.0, 1.5)

scenario = pd.DataFrame([[
    sim_temp,
    sim_pressure,
    sim_motor,
    sim_flow,
    sim_vibration
]], columns=FEATURES)

sim_energy = model.predict(scenario)[0]
sim_carbon = sim_energy * EMISSION_FACTOR
sim_cost = sim_energy * ELECTRICITY_COST
improvement = current_energy - sim_energy

c1, c2, c3, c4 = st.columns(4)
c1.metric("Energy", round(sim_energy,2))
c2.metric("Carbon", round(sim_carbon,2))
c3.metric("Cost", f"₹ {round(sim_cost,2)}")
c4.metric("Improvement", round(improvement,2))

# =====================================================
# FOOTER
# =====================================================

st.caption("AI Manufacturing Optimization Platform — Final V2") 
