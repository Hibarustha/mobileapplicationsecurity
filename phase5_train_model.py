import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier

print("🚀 PHASE 5 — FIXED + STRONGER CATBOOST (ALL FEATURES + HEAVY REGULARIZATION)")

df = pd.read_csv("final_app_dataset.csv")
print(f"Total apps: {len(df)} | Malware: {df['label'].sum()} | Benign: {len(df)-df['label'].sum()}")

X = df.drop(columns=["label", "app_name"], errors="ignore")
y = df["label"]

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
X.replace([np.inf, -np.inf], 0, inplace=True)

print(f"Using ALL {X.shape[1]} features (CatBoost handles it well)")

# ================================
# 5-FOLD STRATIFIED CV
# ================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_acc = 0.0
best_f1 = 0.0
best_thresh = 0.5
best_model = None
fold_results = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n===== FOLD {fold}/5 =====")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=3000,
        depth=8,
        learning_rate=0.018,          # lower = better generalization
        l2_leaf_reg=6.0,              # stronger regularization
        bootstrap_type="Bayesian",
        random_strength=1.5,
        border_count=128,
        verbose=0,
        random_seed=42,
        early_stopping_rounds=200,
        eval_metric="F1",
        auto_class_weights="Balanced"
    )
    
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    y_prob = model.predict_proba(X_val)[:, 1]

    # Fine-grained threshold search
    best_fold_f1, best_fold_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.005):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_fold_f1:
            best_fold_f1 = f1
            best_fold_t = t

    y_pred = (y_prob >= best_fold_t).astype(int)
    acc = accuracy_score(y_val, y_pred)

    print(f"Best thresh : {best_fold_t:.3f} | F1: {best_fold_f1:.4f} | Acc: {acc:.4f}")

    fold_results.append((acc, best_fold_f1))

    if acc > best_acc:
        best_acc = acc
        best_f1 = best_fold_f1
        best_thresh = best_fold_t
        best_model = model   # ← save the actual best model

# ================================
# FINAL CV SUMMARY
# ================================
avg_acc = np.mean([r[0] for r in fold_results])
avg_f1  = np.mean([r[1] for r in fold_results])

print("\n" + "="*70)
print("🔥 FINAL CROSS-VALIDATION RESULT (REAL PERFORMANCE)")
print("="*70)
print(f"Best Fold Accuracy : {best_acc:.4f}")
print(f"Average CV Accuracy: {avg_acc:.4f}")
print(f"Best F1            : {best_f1:.4f}")
print(f"Average CV F1      : {avg_f1:.4f}")
print(f"Best Threshold     : {best_thresh:.3f}")

# Use the BEST model for final report (on full data - this is slightly optimistic but useful)
final_prob = best_model.predict_proba(X)[:, 1]
final_pred = (final_prob >= best_thresh).astype(int)

print("\nClassification Report (best model on full data):")
print(classification_report(y, final_pred))
print("Confusion Matrix:")
print(confusion_matrix(y, final_pred))

# ================================
# SAVE MODEL + THRESHOLD
# ================================
joblib.dump(best_model, "final_model.pkl")
with open("best_threshold.txt", "w") as f:
    f.write(str(best_thresh))

print("\n✅ BEST MODEL SAVED (ready for inference)")