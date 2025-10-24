# ui_lung.py
# 🫁 Lung Cancer Prediction App — Clean version (no duplicate fields)
# Place this in the same folder as: logreg_model.pkl  (+ optionally survey_lung_cancer.csv)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -------------------- Page --------------------
st.set_page_config(page_title="🫁 Lung Cancer Prediction", page_icon="🫁", layout="centered")
st.title("🫁 Lung Cancer Prediction — Interactive App")

MODEL_PATH = Path("logreg_model.pkl")
CSV_CANDIDATES = [Path("survey_lung_cancer.csv"), Path("data/survey_lung_cancer.csv")]

# -------------------- Helpers --------------------
YESNO_TO_INT = {"YES": 1, "NO": 0, "yes": 1, "no": 0, True: 1, False: 0, 1: 1, 0: 0}
GENDER_MAP = {"M": 1, "F": 0}

def _norm_name(c: str) -> str:
    """Normalize name for matching: strip, collapse spaces, replace underscores; uppercase."""
    c = str(c).strip().replace("_", " ")
    c = " ".join(c.split())
    return c.upper()

# -------------------- Load model and reference CSV --------------------
@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_data
def load_reference_csv():
    for p in CSV_CANDIDATES:
        if p.exists():
            try:
                df = pd.read_csv(p)
                # 🔧 Normalize column names to avoid duplicates (YELLOW_FINGERS → YELLOW FINGERS)
                df.columns = (
                    df.columns.str.strip()
                    .str.replace("_", " ", regex=False)
                    .str.replace("  +", " ", regex=True)
                )
                return df
            except Exception:
                pass
    return None

model = load_model(MODEL_PATH)
ref_df = load_reference_csv()

if model is None:
    st.error("❌ Model file **logreg_model.pkl** not found. Put it next to this script and rerun.")
    st.stop()

# -------------------- Define canonical info --------------------
if hasattr(model, "feature_names_in_"):
    TRAIN_COLS = list(model.feature_names_in_)
else:
    TRAIN_COLS = list(ref_df.columns.drop("LUNG CANCER", errors="ignore")) if ref_df is not None else []

CANON_MAP = {_norm_name(c): c for c in TRAIN_COLS}

KNOWN_YESNO = {
    "SMOKING", "YELLOW FINGERS", "ANXIETY", "PEER PRESSURE", "CHRONIC DISEASE",
    "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING",
    "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
}

FORM_ORDER = ["GENDER", "AGE"] + sorted(KNOWN_YESNO)

# Default values (from CSV if present)
defaults = {}
if ref_df is not None:
    for col in ref_df.columns:
        if col.upper() == "LUNG CANCER":
            continue
        s = ref_df[col].dropna()
        if s.empty:
            continue
        if s.dtype.kind in "biufc":
            defaults[col] = float(s.median())
        else:
            defaults[col] = str(s.mode().iloc[0])

st.caption("Place **logreg_model.pkl** (and optionally **survey_lung_cancer.csv**) beside this file.")

# -------------------- UI Form --------------------
st.subheader("Enter Patient Details")

with st.form("patient_form"):
    inputs = {}
    seen = set()  # prevent duplicate questions

    for col in FORM_ORDER:
        key = _norm_name(col)
        if key in seen:
            continue
        seen.add(key)

        if key == "GENDER":
            inputs["GENDER"] = st.selectbox("Gender", ["M", "F"], index=0)
        elif key == "AGE":
            default_age = int(defaults.get("AGE", 50))
            inputs["AGE"] = st.number_input("Age", min_value=1, max_value=120, value=default_age, step=1)
        else:
            default_val = "YES" if str(defaults.get(col, "NO")).upper() == "YES" else "NO"
            inputs[col] = st.radio(col, ["NO", "YES"], index=1 if default_val == "YES" else 0, horizontal=True)

    submitted = st.form_submit_button("🔍 Predict Cancer Risk")

# -------------------- Normalization --------------------
def normalize_inputs_for_training(raw: dict) -> pd.DataFrame:
    """
    1) Map YES/NO + GENDER to numeric
    2) Rename to exact training names
    3) Add any missing training columns (NaN → imputed)
    4) Reorder columns to match model.feature_names_in_
    """
    cleaned = {}
    for k, v in raw.items():
        key_norm = _norm_name(k)
        val = v
        if key_norm in KNOWN_YESNO:
            val = YESNO_TO_INT.get(v, v)
        elif key_norm == "GENDER":
            val = GENDER_MAP.get(str(v), v)
        cleaned[k] = val

    renamed = {}
    for k, v in cleaned.items():
        key_norm = _norm_name(k)
        if key_norm in CANON_MAP:
            renamed[CANON_MAP[key_norm]] = v
        else:
            renamed[k] = v

    X = pd.DataFrame([renamed])

    for c in TRAIN_COLS:
        if c not in X.columns:
            X[c] = np.nan

    X = X[TRAIN_COLS]

    for c in X.columns:
        if isinstance(X[c].iloc[0], str) and X[c].iloc[0].replace(".", "", 1).isdigit():
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return X

# -------------------- Prediction --------------------
if submitted:
    try:
        X = normalize_inputs_for_training(inputs)

        # Determine positive class index
        classes_ = getattr(model, "classes_", None)
        if classes_ is None:
            last = list(model.named_steps.keys())[-1]
            classes_ = model.named_steps[last].classes_
        classes_ = list(classes_)

        if {"NO", "YES"}.issubset(set(map(str, classes_))):
            pos_idx = list(map(str, classes_)).index("YES")
        elif 1 in classes_ and 0 in classes_:
            pos_idx = classes_.index(1)
        else:
            pos_idx = int(np.argmax(classes_))

        proba = model.predict_proba(X)[:, pos_idx]
        pred = model.predict(X)[0]
        risk_pct = float(proba[0]) * 100.0

        st.subheader("Result")
        st.metric("Estimated cancer risk", f"{risk_pct:.1f}%")
        if risk_pct >= 50:
            st.error(f"Prediction: **{pred}** — Elevated risk indicated. Consider professional medical evaluation.")
        else:
            st.success(f"Prediction: **{pred}** — Lower risk indicated. (This is *not* a diagnosis.)")

        with st.expander("🔍 Debug — features sent to the model"):
            st.write("Expected columns:", TRAIN_COLS)
            st.write("Input columns:", list(X.columns))
            st.write("Dtypes:", X.dtypes.astype(str))
            st.dataframe(X)

    except Exception as e:
        st.exception(e)
        st.error(
            "Prediction failed. Common causes:\n"
            "• Column names (spaces/underscores) differ from training\n"
            "• Input dtypes differ from training\n"
            "• Missing or mismatched columns"
        )

st.divider()
st.caption("⚠️ This provides a machine-learning estimate, not medical advice. Consult a clinician for any concerns.")
