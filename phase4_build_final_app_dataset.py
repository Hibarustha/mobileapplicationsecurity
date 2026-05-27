import pandas as pd
import numpy as np
import re
from scipy.stats import kurtosis

print("🔥 FINAL DATASET (FULL FEATURE FUSION) — FIXED v2")

chunk_df = pd.read_csv("chunk_dataset_with_probs.csv")
app_features = pd.read_csv("app_dataset.csv")

# =====================
# BULLETPROOF MERGE FIX 🔥
# =====================
chunk_df["chunk_idx"] = chunk_df["app_name"].str.extract(r'_chunk(\d+)$').astype(float)
chunk_df = chunk_df.fillna(0)

def extract_hash(name):
    name = str(name).lower().strip()
    match = re.search(r'[a-f0-9]{32,64}', name)
    return match.group(0) if match else name

chunk_df["app_name"] = chunk_df["app_name"].apply(extract_hash)
app_features["app_name"] = app_features["app_name"].apply(extract_hash)

matched_apps = set(chunk_df["app_name"]).intersection(set(app_features["app_name"]))
print(f"🔥 MERGE INTERSECTION: {len(matched_apps)} apps match between datasets.")

# =====================
# PROB FEATURES
# =====================
chunk_df["is_mal_08"] = (chunk_df["chunk_prob_boosted"] > 0.8).astype(int)
chunk_df["is_mal_09"] = (chunk_df["chunk_prob_boosted"] > 0.9).astype(int)
chunk_df["is_mal_05"] = (chunk_df["chunk_prob_boosted"] > 0.5).astype(int)

prob_df = chunk_df.groupby("app_name").agg({
    "chunk_prob_boosted": ["mean", "max", "std", "min", "median"],
    "is_mal_08": "sum",
    "is_mal_09": "sum",
    "is_mal_05": ["sum", "mean"],
    "label": "first"
}).reset_index()

prob_df.columns = [
    "app_name",
    "prob_mean", "prob_max", "prob_std", "prob_min", "prob_median",
    "mal_chunks_08", "mal_chunks_09", "mal_chunks_05", "mal_ratio_05",
    "label"
]

# =====================
# COUNT + NORMALIZE
# =====================
counts = chunk_df.groupby("app_name").size().reset_index(name="num_chunks")
prob_df = prob_df.merge(counts, on="app_name")

prob_df["mal_ratio_08"] = prob_df["mal_chunks_08"] / prob_df["num_chunks"]
prob_df["mal_ratio_09"] = prob_df["mal_chunks_09"] / prob_df["num_chunks"]

# =====================
# ORIGINAL APP FEATURES
# =====================
if "label" in app_features.columns:
    app_features = app_features.drop(columns=["label"])

app_features = app_features.groupby("app_name").agg(["mean", "std", "max", "min"])
app_features.columns = ["_".join(col) for col in app_features.columns]
app_features = app_features.reset_index()

# =====================
# MERGE
# =====================
final_df = prob_df.merge(app_features, on="app_name", how="left")

# =====================
# ADVANCED PROBABILITY FEATURES (FIXED 🔥)
# =====================
print("Adding advanced probability burst & distribution features...")

prob_group = chunk_df.groupby("app_name")["chunk_prob_boosted"]

final_df["prob_skew"]     = prob_group.skew()
final_df["prob_kurtosis"] = prob_group.apply(
    lambda x: kurtosis(x, fisher=False, nan_policy="omit") if len(x) > 1 and x.std() > 1e-8 else 0.0
)
final_df["prob_q25"]      = prob_group.quantile(0.25)
final_df["prob_q75"]      = prob_group.quantile(0.75)
final_df["prob_iqr"]      = final_df["prob_q75"] - final_df["prob_q25"]   # ← FIXED
final_df["prob_range"]    = final_df["prob_max"] - final_df["prob_min"]   # ← ADDED

# Burst / transition patterns (unchanged)
def advanced_streak_stats(group):
    probs = group["chunk_prob_boosted"].values
    is_mal = (probs > 0.8).astype(int)
    max_streak = 0
    current = 0
    transitions = 0
    first_mal_idx = -1
    last_mal_idx = -1
    for i, val in enumerate(is_mal):
        if val == 1:
            current += 1
            if first_mal_idx == -1:
                first_mal_idx = i
            last_mal_idx = i
            max_streak = max(max_streak, current)
        else:
            current = 0
            if i > 0 and is_mal[i-1] == 1:
                transitions += 1
    if first_mal_idx == -1:
        first_mal_idx = len(is_mal)
        last_mal_idx = 0
    return pd.Series({
        "max_malicious_streak": max_streak,
        "mal_transition_count": transitions,
        "first_mal_chunk_ratio": first_mal_idx / len(is_mal),
        "last_mal_chunk_ratio": last_mal_idx / len(is_mal)
    })

streak_df = chunk_df.groupby("app_name").apply(advanced_streak_stats, include_groups=False).reset_index()
final_df = final_df.merge(streak_df, on="app_name", how="left")

# High-signal interaction features (FIXED)
final_df["prob_max_x_streak"]      = final_df["prob_max"] * final_df["max_malicious_streak"]
final_df["prob_mean_x_mal_ratio"]  = final_df["prob_mean"] * final_df["mal_ratio_08"]
final_df["prob_skew_x_kurt"]       = final_df["prob_skew"] * final_df["prob_kurtosis"]
final_df["mal_ratio_09_x_streak"]  = final_df["mal_ratio_09"] * final_df["max_malicious_streak"]
final_df["prob_range_x_streak"]    = final_df["prob_range"] * final_df["max_malicious_streak"]   # ← NOW WORKS

# =====================
# FINAL CLEANUP & SAVE
# =====================
print("Cleaning and saving final dataset...")

final_df = final_df.fillna(0)

numeric_cols = final_df.select_dtypes(include=[np.number]).columns
constant_cols = [col for col in numeric_cols if final_df[col].std() < 1e-4]
final_df = final_df.drop(columns=constant_cols, errors="ignore")

print(f"Dropped {len(constant_cols)} near-constant features.")
print(f"✅ FINAL SHAPE: {final_df.shape} (rows × columns)")

final_df.to_csv("final_app_dataset.csv", index=False)
print("✅ final_app_dataset.csv SAVED SUCCESSFULLY! (with fixed IQR + range features)")