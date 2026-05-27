# ================================
# IMPORTS
# ================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


from collections import Counter
from imblearn.over_sampling import SMOTE

print(" FINAL CHUNK TRAINING (STACKED ENSEMBLE - FIXED)")

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("chunk_dataset.csv")

feature_columns = [col for col in df.columns if col not in ["label", "app_name"]]

X = df[feature_columns]
y = df["label"]

print("Total samples:", len(df))
print("Total apps:", df["app_name"].nunique())

# ================================
# FEATURE CLEANING
# ================================
X = X.loc[:, X.std() > 0.01]

selector = SelectKBest(f_classif, k=min(300, X.shape[1]))
X = selector.fit_transform(X, y)

print("Selected feature shape:", X.shape)

# ================================
# CLASS WEIGHT
# ================================
counter = Counter(y)
scale_pos_weight = counter[0] / counter[1]

# ================================
# OOF STACKING (NO LEAKAGE)
# ================================
# Extract base app name so chunks from the same app stay together!
df["app_base"] = df["app_name"].str.replace(r"_chunk\d+$", "", regex=True)

kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)



oof_xgb = np.zeros(len(X))
oof_rf = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))



for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=df["app_base"])):
    print(f"\n🔥 Fold {fold+1}")

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 🔥 SMOTE INSIDE FOLD (CORRECT WAY)
    smote = SMOTE()
    X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # ================= XGBOOST =================
    xgb = XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb.predict_proba(X_val)[:, 1]
 
   
    # ================= RANDOM FOREST =================
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    oof_rf[val_idx] = rf.predict_proba(X_val)[:, 1]
   

    # ================= LIGHTGBM =================
    lgb = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=64
    )
    lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb.predict_proba(X_val)[:, 1]
   

    # ================= CATBOOST =================
    cat = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        verbose=0
    )
    cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = cat.predict_proba(X_val)[:, 1]
  

# ================================
# STACKING
# ================================
stack_X = np.vstack((oof_xgb, oof_rf, oof_lgb, oof_cat)).T

meta_model = LogisticRegression()
meta_model.fit(stack_X, y)
df["chunk_prob"] = meta_model.predict_proba(stack_X)[:, 1]

# 🔥 CHUNK BOOSTING (VERY POWERFUL)
def boost_prob(p):
    if p > 0.8:
        return min(1.0, p + 0.1)
    elif p < 0.2:
        return max(0.0, p - 0.1)
    else:
        return p

df["chunk_prob_boosted"] = df["chunk_prob"].apply(boost_prob)

print("✅ Stacked chunk probabilities created")

# ================================
# APP-LEVEL DATASET
# ================================
print("\nBuilding app-level dataset...")

app_df = df.groupby("app_name").agg({
    "chunk_prob": ["mean", "max", "std", "min", "median"],
    "label": "first"
})

app_df.columns = ["mean", "max", "std", "min", "median", "label"]
app_df = app_df.reset_index()


# ================================
# SAVE
# ================================
df.to_csv("chunk_dataset_with_probs.csv", index=False)


joblib.dump(meta_model, "meta_model.pkl")
joblib.dump(selector, "feature_selector.pkl")
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(rf, "rf_model.pkl")
joblib.dump(lgb, "lgb_model.pkl")
joblib.dump(cat, "cat_model_chunk.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Saved all models")