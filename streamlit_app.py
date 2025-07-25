
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Multiverse Predictor", layout="centered")

# --- Custom Dark Theme Styling ---
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #5E60CE;
        color: white;
        font-weight: bold;
        border-radius: 0.4rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        background-color: #1e1e1e;
        color: white;
    }
    .stSelectbox>div>div>div {
        background-color: #1e1e1e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🔮 Doctor Strange's Multiverse Predictor")
st.markdown("Out of 14,000,605 timelines, only a few lead to victory. Let this ML model help find them.")

# --- Model Choice ---
model_choice = st.selectbox("Choose ML Model", ["Logistic Regression", "Random Forest"])
model_path = "multiverse_logistic.pkl" if model_choice == "Logistic Regression" else "multiverse_random_forest.pkl"
model = joblib.load(model_path)

# --- Input Form ---
with st.form("timeline_form"):
    st.subheader("🧠 Simulation Parameters")

    col1, col2 = st.columns(2)

    with col1:
        team_strength = st.number_input("Team Strength", value=60.0)
        team_coordination = st.number_input("Team Coordination", value=0.75)
        strategic_plan_complexity = st.number_input("Strategic Plan Complexity", value=6.0)
        diversion_success_rate = st.number_input("Diversion Success Rate", value=0.4)
        intel_accuracy = st.number_input("Intel Accuracy", value=0.85)
        previous_failures = st.number_input("Previous Failures", value=1.0)
        universe_variability = st.number_input("Universe Variability", value=0.6)

    with col2:
        enemy_strength = st.number_input("Enemy Strength", value=80.0)
        num_heroes = st.number_input("Number of Heroes", value=5.0)
        num_enemies = st.number_input("Number of Enemies", value=10.0)
        enemy_stone_count = st.number_input("Enemy Infinity Stones", value=4.0)
        has_time_stone = st.selectbox("Has Time Stone?", ["yes", "no"])
        has_surprise_element = st.selectbox("Has Surprise Element?", ["yes", "no"])
        terrain_advantage = st.selectbox("Terrain Advantage?", ["yes", "no"])
        enemy_mind_state = st.selectbox("Enemy Mind State", ["confident", "hesitant", "arrogant", "fearful"])
        has_ironman = st.selectbox("Has Ironman?", ["yes", "no"])
        sacrifice_possible = st.selectbox("Sacrifice Possible?", ["yes", "no"])

    submit = st.form_submit_button("🔍 Predict Timeline Outcome")

# --- Prediction Logic ---
if submit:
    input_dict = {
        "team_strength": team_strength,
        "enemy_strength": enemy_strength,
        "team_coordination": team_coordination,
        "strategic_plan_complexity": strategic_plan_complexity,
        "diversion_success_rate": diversion_success_rate,
        "intel_accuracy": intel_accuracy,
        "universe_variability": universe_variability,
        "previous_failures": previous_failures,
        "num_heroes": num_heroes,
        "num_enemies": num_enemies,
        "enemy_stone_count": enemy_stone_count,
        "has_time_stone": has_time_stone,
        "has_surprise_element": has_surprise_element,
        "terrain_advantage": terrain_advantage,
        "enemy_mind_state": enemy_mind_state,
        "has_ironman": has_ironman,
        "sacrifice_possible": sacrifice_possible
    }

    try:
        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[0][1]
        prediction = int(prob >= 0.3)

        st.subheader("🧾 Timeline Prediction")
        if prediction == 1:
            st.success(f"🎉 Victory Timeline Detected! (Probability: {prob:.3f})")
            st.markdown("> *“There was no other way.”* — Dr. Strange")
        else:
            st.error(f"💀 Defeat Timeline. (Probability: {prob:.3f})")
            st.markdown("> *Not this time. Try another strategy.*")

        # --- Bar Chart ---
        st.markdown("### 📊 Victory vs Defeat Probability")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        labels = ["Victory 🟢", "Defeat 🔴"]
        values = [prob, 1 - prob]
        colors = ["#00c853", "#d50000"]

        bars = ax.barh(labels, values, color=colors)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{width:.2f}", va='center', color='white', fontsize=12, fontweight='bold')

        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", color='white')
        ax.set_title("Predicted Outcome Breakdown", color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor("#121212")
        fig.patch.set_facecolor("#FFFDFD")
        st.pyplot(fig)

        # --- Pie Chart ---
        st.markdown("### 🌀 Timeline Outcome Pie")
        fig2, ax2 = plt.subplots()
        wedges, texts, autotexts = ax2.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops=dict(color="white", fontsize=12)
        )
        ax2.set_title("Probability Distribution", color='white')
        ax2.axis("equal")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
