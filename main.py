# main.py — CLI or Streamlit launcher for NutriElite

import argparse
import json
from api import generate_plan, AthleteProfile

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def cli_mode():
    print("\nNutriElite Nutrition Plan Generator (CLI Mode)")
    profile = AthleteProfile(
        age=int(input("Enter age: ")),
        height=int(input("Enter height (cm): ")),
        weight=int(input("Enter weight (kg): ")),
        body_fat_percent=int(input("Enter body fat %: ")),
        goal=input("Enter goal (cutting, bulking, etc.): "),
        activity_level=input("Enter activity level: "),
        sport=input("Enter sport: "),
        position=input("Enter position: ")
    )
    plan = generate_plan(profile)
    print("\nGenerated Plan:")
    print(json.dumps(plan, indent=2))


def streamlit_mode():
    st.set_page_config(page_title="NutriElite Generator", layout="centered")
    st.title("🏋️ NutriElite — Athlete Meal Plan Generator")
    st.markdown("Real-time personalised nutrition plans for elite performance.")

    with st.form("profile_form"):
        age = st.number_input("Age", 16, 60, 26)
        height = st.number_input("Height (cm)", 140, 230, 198)
        weight = st.number_input("Weight (kg)", 50, 150, 96)
        body_fat = st.number_input("Body Fat %", 5, 40, 12)
        goal = st.selectbox("Goal", ["cutting", "bulking", "maintenance"])
        activity_level = st.selectbox("Activity Level", ["sedentary", "active", "very_active"])
        sport = st.text_input("Sport", "basketball")
        position = st.text_input("Position", "forward")
        submitted = st.form_submit_button("Generate Plan")

    if submitted:
        profile = AthleteProfile(
            age=age,
            height=height,
            weight=weight,
            body_fat_percent=body_fat,
            goal=goal,
            activity_level=activity_level,
            sport=sport,
            position=position
        )
        st.info("Generating plan with fine-tuned Mistral model...")
        plan = generate_plan(profile)
        st.success("Plan generated!")
        st.json(plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--streamlit", action="store_true", help="Launch in Streamlit UI mode")
    args = parser.parse_args()

    if args.streamlit and STREAMLIT_AVAILABLE:
        streamlit_mode()
    else:
        cli_mode()
