# ui.py
import streamlit as st
import pandas as pd
import json, joblib, time, os
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="Tech Gadget Mental Wellbeing", page_icon="🧠", layout="centered")
st.title("🧠 Tech Gadget Mental Wellbeing — Prediction App")

# ---- Load model (Pipeline: pre + classifier) ----
MODEL_PATH = "svm_adaboost.pkl"
model = joblib.load(MODEL_PATH)


MODEL_PATH = "svm_adaboost.pkl"
META_PATH  = "svm_adaboost_metadata.json"

# Show file info
st.caption(f"Model file: `{os.path.abspath(MODEL_PATH)}`")
st.caption(f"Last modified: {time.ctime(os.path.getmtime(MODEL_PATH))}")

# Load metadata if available and show AUC
meta = None
if Path(META_PATH).exists():
    meta = json.loads(Path(META_PATH).read_text())
    auc_meta = meta.get("test_metrics", {}).get("AUC")
    st.caption(f"Exported model: **{meta.get('model_name')}**, AUC (test): **{auc_meta}**")

# Inspect the pipeline to prove it's AdaBoost(SVC linear)
with st.expander("🔎 Verify loaded pipeline"):
    steps = list(getattr(model, "named_steps", {}).keys())
    st.write("Pipeline steps:", steps)

    clf = model.named_steps.get("clf", None)
    if isinstance(clf, AdaBoostClassifier):
        st.write("`clf` is AdaBoostClassifier ✅")
        base = clf.estimator
        st.write("Base estimator:", type(base).__name__)
        if isinstance(base, SVC):
            st.write("Base SVC params:", base.get_params())
            kernel = base.get_params().get("kernel")
            st.write("Base SVC kernel:", kernel)
            if kernel == "linear":
                st.success("Confirmed: AdaBoost over **linear SVM** ✅")
            else:
                st.error("Base SVC kernel is not linear ❌")
        else:
            st.error("Base estimator is not SVC ❌")
        st.write("AdaBoost params:", {k: clf.get_params()[k] for k in ["n_estimators","learning_rate","algorithm"]})
    else:
        st.error("`clf` is not AdaBoostClassifier ❌")

    # Show class order used by predict_proba
    try:
        st.write("Classes:", list(model.classes_))
    except Exception as e:
        st.warning(f"Could not read model.classes_: {e}")

# If you saved metadata:
META_PATH = MODEL_PATH.replace(".pkl", "_metadata.json")
if os.path.exists(META_PATH):
    meta = json.loads(Path(META_PATH).read_text())
    st.caption(f"Loaded model: {meta.get('model_name')} | AUC: {meta.get('test_metrics',{}).get('AUC')}")

# ---- IMPORTANT: exact raw feature names used in training ----
RAW_COLUMNS = [
    "Age",
    "Daily.Screen.Time",
    "Social.Media.Hours",
    "Work.Screen.Hours",
    "Gaming.Hours",
    "Sleep.Hours",
    "Sleep.Quality",
    "Stress.Level",
    "Physical.Activity.Min",
    "Productivity.Score",
    "Gender",
    "Country",
    "Device.Type",
]

st.write("""
Use this tool to explore how lifestyle/device usage patterns may relate to mental wellbeing.
Enter your info below and click **Predict**.
""")

# ---------- Helpers ----------
def left_label_right_input(label: str, key: str, placeholder: str = ""):
    col_l, col_r = st.columns([1.2, 2])
    with col_l:  st.markdown(f"**{label}**")
    with col_r:
        return st.text_input("", key=key, placeholder=placeholder, label_visibility="collapsed")

def to_float_or_none(x: str):
    try: return float(x)
    except: return None

# ---------- UI (Form) ----------
with st.form("predict_form", clear_on_submit=False):
    st.subheader("Enter your daily metrics")

    # Numerics
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

    # Categoricals (must exist; OHE has handle_unknown='ignore', so free values are safe)
    st.markdown("**Demographics & Device**")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
    with c2:
        country = st.selectbox("Country", ["USA", "India", "UK", "Germany", "Canada", "Australia"], index=5)
    with c3:
        device_type = st.selectbox("Device Type", ["Phone", "Laptop", "Tablet"], index=0)

    # Centered submit button
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
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
        if val is None: missing.append(label)

    if missing:
        st.warning("Please fix these fields: " + ", ".join(missing))
    else:
        # Build row with EXACT training column names
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
            "Gender": gender,
            "Country": country,
            "Device.Type": device_type,
        }])

        # Reindex to the exact schema (prevents silent mis-order bugs)
        input_df = input_df.reindex(columns=RAW_COLUMNS)

        # Predict
        pred = model.predict(input_df)[0]
        proba = getattr(model, "predict_proba")(input_df)[0]
        classes = list(getattr(model, "classes_"))

        st.success(f"🧩 Predicted Wellbeing Category: **{pred}**")

        with st.expander("🔍 Debug view (model input & probabilities)"):
            st.dataframe(input_df, use_container_width=True)
            st.write({cls: float(p) for cls, p in zip(classes, proba)})

else:
    st.info("Fill in the fields and click **Predict** to see your result.")

st.markdown("---")
st.caption("© 2025 Tech Gadget Mental Wellbeing Project — Aaron Tan")
