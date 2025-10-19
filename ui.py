# ui.py
import streamlit as st
import pandas as pd
import json
import joblib
from pathlib import Path

st.title("🧠 Tech Gadget Mental Wellbeing — Prediction App")

# # Load the model
model = joblib.load("svm_adaboost.pkl")

st.write("""
Welcome to the **🧠 Tech Gadget Mental Wellbeing — Prediction App**!  

Use this tool to explore how different lifestyle habits and device usage patterns may relate to your mental wellbeing.  
Simply fill in your daily screen time, sleep hours, stress level, and other details below — then click **Predict** to see your wellbeing category (Healthy, Moderate Issues, or Chronic Issues).  

📊 **About the Dataset:**  
This app was trained on the *Tech Gadget and Mental Wellbeing* dataset, which includes anonymized survey responses from individuals aged 16–64 across multiple countries (USA, India, UK, Germany, Canada, and Australia).  
The dataset examines the relationship between technology usage (screen time, gaming, social media) and mental health indicators such as stress, sleep quality, and productivity.

⚙️ **How to Use:**  
1. Enter your information in the input boxes.  
2. Press the **Predict** button to get your result.  
3. Review the prediction and reflect on possible changes to your daily habits.

---
""")

# ---------- Helpers ----------
def left_label_right_input(label: str, key: str, placeholder: str = ""):
    col_l, col_r = st.columns([1.2, 2])
    with col_l:
        st.markdown(f"**{label}**")
    with col_r:
        return st.text_input(
            label="",
            key=key,
            placeholder=placeholder,
            label_visibility="collapsed"
        )

def to_float_or_none(x: str):
    try:
        return float(x)
    except Exception:
        return None

# ---------- UI (Form) ----------
with st.form("predict_form", clear_on_submit=False):
    st.subheader("Enter your daily metrics")

    age              = left_label_right_input("Age (16–64)", "age", "e.g., 25")
    screentime       = left_label_right_input("Daily Screen Time (hrs)", "screentime", "e.g., 6.5")
    socialtime       = left_label_right_input("Social Media (hrs)", "socialtime", "e.g., 2")
    worktime         = left_label_right_input("Work/Study Screen (hrs)", "worktime", "e.g., 5")
    gamingtime       = left_label_right_input("Gaming (hrs)", "gamingtime", "e.g., 1")
    sleeptime        = left_label_right_input("Sleep (hrs)", "sleeptime", "e.g., 7.5")
    sleepquality     = left_label_right_input("Sleep Quality (1–10)", "sleepquality", "e.g., 7")
    stresslevel      = left_label_right_input("Stress Level (1–10)", "stresslevel", "e.g., 4")
    physical         = left_label_right_input("Physical Activity (mins)", "physical", "e.g., 45")
    score            = left_label_right_input("Productivity Score (1–10)", "score", "e.g., 6")

    # Centered submit button
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        submitted = st.form_submit_button("Predict 🔮")

# ---------- Handle submission ----------
if submitted:
    # Convert numerics safely
    age_f          = to_float_or_none(age)
    screentime_f   = to_float_or_none(screentime)
    socialtime_f   = to_float_or_none(socialtime)
    worktime_f     = to_float_or_none(worktime)
    gamingtime_f   = to_float_or_none(gamingtime)
    sleeptime_f    = to_float_or_none(sleeptime)
    sleepquality_f = to_float_or_none(sleepquality)
    stresslevel_f  = to_float_or_none(stresslevel)
    physical_f     = to_float_or_none(physical)
    score_f        = to_float_or_none(score)

    # Basic validation
    missing = []
    if age_f is None or not (16 <= age_f <= 64): missing.append("Age (16–64)")
    for label, val in [
        ("Daily Screen Time (hrs)", screentime_f),
        ("Social Media (hrs)", socialtime_f),
        ("Work/Study Screen (hrs)", worktime_f),
        ("Gaming (hrs)", gamingtime_f),
        ("Sleep (hrs)", sleeptime_f),
        ("Sleep Quality (1–10)", sleepquality_f),
        ("Stress Level (1–10)", stresslevel_f),
        ("Physical Activity (mins)", physical_f),
        ("Productivity Score (1–10)", score_f),
    ]:
        if val is None:
            missing.append(label)

    if missing:
        st.warning("Please fix these fields: " + ", ".join(missing))
    else:
        input_df = pd.DataFrame([{
            "Age": age_f,
            "Daily.Screen.Time": screentime_f,
            "Social.Media.Hours": socialtime_f,
            "Work.Screen.Hours": worktime_f,
            "Gaming.Hours": gamingtime_f,
            "Sleep.Hours": sleeptime_f,
            "Sleep.Quality": sleepquality_f,
            "Stress.Level": stresslevel_f,
            "Physical.Activity.Min": physical_f,
            "Productivity.Score": score_f,
        }])

        pred = model.predict(input_df)[0]
        st.success(f"🧩 Predicted Wellbeing Category: **{pred}**")

        # Also show a clean table version (handy for debugging / model input preview)
        df = pd.DataFrame([{
            "Age": int(age_f),
            "Daily.Screen.Time": screentime_f,
            "Social.Media.Hours": socialtime_f,
            "Work.Screen.Hours": worktime_f,
            "Gaming.Hours": gamingtime_f,
            "Sleep.Hours": sleeptime_f,
            "Sleep.Quality": int(sleepquality_f),
            "Stress.Level": int(stresslevel_f),
            "Physical.Activity.Min": int(physical_f),
            "Productivity.Score": int(score_f),
        }])
        st.markdown("#### Data View")
        st.dataframe(df, use_container_width=True)
else:
    st.info("Fill in the fields and click **Predict** to see your summary.")

st.markdown("""
---
<p style="text-align: center; color: gray; font-size: 14px;">
© 2025 Tech Gadget Mental Wellbeing Project — Created by Aaron Tan.  
Dataset Reference: Tech Gadget & Mental Wellbeing (https://www.kaggle.com/datasets/kasunvishvajith/tech-gadget-usage-and-mental-wellbeing)
</p>
""", unsafe_allow_html=True) 