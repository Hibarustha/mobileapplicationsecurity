# ============================================
# 🔥 PHASE 6: FINAL APK TEST PIPELINE
# ============================================

import os
import pandas as pd
import numpy as np
import joblib
import shap
import re

# ================================
# LOAD MODELS
# ================================
print("🔄 Loading models...")

xgb = joblib.load("xgb_model.pkl")
rf  = joblib.load("rf_model.pkl")
lgb = joblib.load("lgb_model.pkl")
cat = joblib.load("cat_model_chunk.pkl")

meta     = joblib.load("meta_model.pkl")
selector = joblib.load("feature_selector.pkl")
scaler   = joblib.load("scaler.pkl")

final_model = joblib.load("final_model.pkl")

with open("best_threshold.txt") as f:
    threshold = float(f.read().strip())

# IMPORTANT
top_syscalls = joblib.load("top_syscalls.pkl")

print("✅ Models loaded")

# ================================
# IMPORT FUNCTIONS
# ================================
from phase1_chunk_feature_builder import extract_chunk_features
from phase3_app_feature_dataset import extract_features, chunk_sequence

# ================================
# SELECT FILE
# ================================
FILE_PATH = "clean_test/test_apk_malware.txt"
# FILE_PATH = "clean_test/test_apk_benign.txt"

def load_syscalls(path):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]

filename = os.path.basename(FILE_PATH)
print(f"\n🔍 Testing: {filename}")

seq = load_syscalls(FILE_PATH)

if len(seq) < 30:
    print("❌ Not enough syscalls")
    exit()

label = 1 if "malware" in filename.lower() else 0

# ============================================
# STEP 1: PHASE 1 (CHUNK FEATURES)
# ============================================
rows_phase1 = []

for i in range(0, len(seq), 300):
    chunk = seq[i:i+300]

    if len(chunk) < 50:
        continue

    rows_phase1.extend(
        extract_chunk_features(
            chunk,
            label,
            filename + f"_chunk{i}",
            top_syscalls
        )
    )

chunk_df = pd.DataFrame(rows_phase1).fillna(0)
print(f"✅ Phase1 chunks: {len(chunk_df)}")

# ============================================
# STEP 2: CHUNK MODEL
# ============================================
X_chunk = chunk_df.drop(columns=["label", "app_name"], errors="ignore")

expected_features = selector.feature_names_in_

for col in expected_features:
    if col not in X_chunk.columns:
        X_chunk[col] = 0

X_chunk = X_chunk[expected_features]

X_chunk = selector.transform(X_chunk)
X_chunk = scaler.transform(X_chunk)

p_xgb = xgb.predict_proba(X_chunk)[:, 1]
p_rf  = rf.predict_proba(X_chunk)[:, 1]
p_lgb = lgb.predict_proba(X_chunk)[:, 1]
p_cat = cat.predict_proba(X_chunk)[:, 1]

stack_X = np.vstack((p_xgb, p_rf, p_lgb, p_cat)).T
chunk_probs = meta.predict_proba(stack_X)[:, 1]

chunk_df["chunk_prob"] = chunk_probs

# Boosting
def boost_prob(p):
    if p > 0.8:
        return min(1.0, p + 0.1)
    elif p < 0.2:
        return max(0.0, p - 0.1)
    return p

chunk_df["chunk_prob_boosted"] = chunk_df["chunk_prob"].apply(boost_prob)

# ============================================
# STEP 3: PHASE 3 FEATURES (SEPARATE)
# ============================================
rows_phase3 = []

chunks = chunk_sequence(seq, 300)

for i, chunk in enumerate(chunks):
    if len(chunk) < 50:
        continue

    rows_phase3.append(
        extract_features(chunk, label, filename + f"_chunk{i}")
    )

phase3_df = pd.DataFrame(rows_phase3).fillna(0)

phase3_df["app_base"] = phase3_df["app_name"].str.replace(r"_chunk\d+$", "", regex=True)

# 🔥 KEEP ONLY NUMERIC + REMOVE LABEL
numeric_df = phase3_df.select_dtypes(include=[np.number]).copy()
if "label" in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=["label"])

numeric_df["app_base"] = phase3_df["app_base"]

app_features = numeric_df.groupby("app_base").agg(["mean", "std", "max", "min"])
app_features.columns = ["_".join(col) for col in app_features.columns]
app_features = app_features.reset_index()

# ============================================
# STEP 4: PROBABILITY FEATURES
# ============================================
chunk_df["app_base"] = chunk_df["app_name"].str.replace(r"_chunk\d+$", "", regex=True)

chunk_df["is_mal_08"] = (chunk_df["chunk_prob_boosted"] > 0.8).astype(int)
chunk_df["is_mal_09"] = (chunk_df["chunk_prob_boosted"] > 0.9).astype(int)
chunk_df["is_mal_05"] = (chunk_df["chunk_prob_boosted"] > 0.5).astype(int)

prob_df = chunk_df.groupby("app_base").agg({
    "chunk_prob_boosted": ["mean", "max", "std", "min", "median"],
    "is_mal_08": "sum",
    "is_mal_09": "sum",
    "is_mal_05": ["sum", "mean"],
    "label": "first"
}).reset_index()

prob_df.columns = [
    "app_base",
    "prob_mean", "prob_max", "prob_std", "prob_min", "prob_median",
    "mal_chunks_08", "mal_chunks_09", "mal_chunks_05", "mal_ratio_05",
    "label"
]

counts = chunk_df.groupby("app_base").size().reset_index(name="num_chunks")
prob_df = prob_df.merge(counts, on="app_base")

prob_df["mal_ratio_08"] = prob_df["mal_chunks_08"] / prob_df["num_chunks"]
prob_df["mal_ratio_09"] = prob_df["mal_chunks_09"] / prob_df["num_chunks"]

# ============================================
# STEP 5: MERGE
# ============================================
final_df = prob_df.merge(app_features, on="app_base", how="left")

# ============================================
# STEP 6: FINAL INPUT
# ============================================
X_final = final_df.drop(columns=["label", "app_base"], errors="ignore")

X_final = X_final.apply(pd.to_numeric, errors="coerce").fillna(0)
X_final.replace([np.inf, -np.inf], 0, inplace=True)

# 🔥 ALIGN FEATURES WITH MODEL
expected_final_features = final_model.feature_names_

for col in expected_final_features:
    if col not in X_final.columns:
        X_final[col] = 0

X_final = X_final[expected_final_features]

# ============================================
# STEP 7: PREDICTION
# ============================================
prob = final_model.predict_proba(X_final)[0][1]
pred = int(prob >= threshold)

print("\n🔥 FINAL RESULT")
print(f"Probability : {prob:.4f}")
print(f"Prediction  : {'MALWARE' if pred==1 else 'BENIGN'}")

# ============================================
# STEP 8: SHAP
# ============================================
print("\n🔍 SHAP Explanation")

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_final)

importance = pd.Series(shap_values[0], index=X_final.columns)
importance = importance.abs().sort_values(ascending=False)

print("\nTop 10 Features:")
print(importance.head(10))

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_final.iloc[0],
        feature_names=X_final.columns
    )
)