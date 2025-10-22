import json, pickle
import pandas as pd

MODEL_PKL = "svm_adaboost.pkl"
META_JSON = "svm_adaboost_metadata.json"  # you already save this

# 1) Load pipeline (pre + model)
with open(MODEL_PKL, "rb") as f:
    model = pickle.load(f)

# 2) Build ONE row with EXACT training column names
row = {
    "Age": age,  # int
    "Daily.Screen.Time": daily_screen_time,         # float
    "Social.Media.Hours": social_media_hours,       # float
    "Work.Screen.Hours": work_study_hours,          # float
    "Gaming.Hours": gaming_hours,                   # float
    "Sleep.Hours": sleep_hours,                     # float
    "Sleep.Quality": sleep_quality,                 # int 1–10
    "Stress.Level": stress_level,                   # int 1–10
    "Physical.Activity.Min": physical_activity_min, # int (minutes)
    "Productivity.Score": productivity_score,       # int 1–10
    "Gender": gender_str,                           # e.g. "Male" / "Female" / "Other"
    "Country": country_str,                         # e.g. "Australia"
    "Device.Type": device_type_str                  # e.g. "Phone" / "Laptop" / "Tablet"
}

# 3) Create DataFrame **with names** and no manual re-ordering
X_input = pd.DataFrame([row])

# (Optional) If you want to be extra safe, reindex to training columns.
# Save these once during training: json.dump({"raw_feature_names": X.columns.tolist()}, META_JSON)
try:
    with open(META_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "raw_feature_names" in meta:
        X_input = X_input.reindex(columns=meta["raw_feature_names"])
except Exception:
    pass

# 4) Predict
pred = model.predict(X_input)[0]
proba = model.predict_proba(X_input)[0]
classes = model.classes_

# 5) If you want a dict of class->probability (for display)
proba_map = {cls: float(p) for cls, p in zip(classes, proba)}

test_row = {
    "Age": 16,
    "Daily.Screen.Time": 0.0,
    "Social.Media.Hours": 0.0,
    "Work.Screen.Hours": 0.0,
    "Gaming.Hours": 0.0,
    "Sleep.Hours": 10.0,
    "Sleep.Quality": 10,
    "Stress.Level": 1,
    "Physical.Activity.Min": 140,
    "Productivity.Score": 10,
    "Gender": "Male",
    "Country": "Australia",
    "Device.Type": "Phone"
}
X_test_one = pd.DataFrame([test_row])
print(model.predict(X_test_one)[0])
print(dict(zip(model.classes_, model.predict_proba(X_test_one)[0])))
