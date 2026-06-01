import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/career_dataset.csv")
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")

SKILLS = [
    "python", "javascript", "sql", "machine_learning", "statistics",
    "react", "nodejs", "docker", "kubernetes", "networking",
    "linux", "swift", "kotlin", "cloud", "data_analysis",
    "deep_learning", "communication", "project_management", "uiux_design", "git"
]

X = df[SKILLS]
y = df["career_track"]

# ── 2. Encode Labels ──────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"🎯 Career tracks: {list(le.classes_)}\n")

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"📊 Train size: {len(X_train)} | Test size: {len(X_test)}\n")

# ── 4. Train Random Forest ────────────────────────────────────────────────────
print("⏳ Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Model trained!\n")

# ── 5. Cross-Validation ───────────────────────────────────────────────────────
print("🔁 Running 5-Fold Cross Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="accuracy")
print(f"   CV Scores : {[round(s, 4) for s in cv_scores]}")
print(f"   Mean Acc  : {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
print(f"   Std Dev   : {cv_scores.std():.4f}\n")

# ── 6. Test Set Evaluation ────────────────────────────────────────────────────
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"🎯 Test Set Accuracy: {test_acc*100:.2f}%\n")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── 7. Save Confusion Matrix Plot ─────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix", fontsize=16)
plt.ylabel("Actual", fontsize=12)
plt.xlabel("Predicted", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png", dpi=150)
plt.close()
print("📊 Confusion matrix saved → data/confusion_matrix.png")

# ── 8. Feature Importance Plot ────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=SKILLS).sort_values(ascending=True)
plt.figure(figsize=(10, 7))
importances.plot(kind="barh", color="steelblue")
plt.title("Feature Importance (Skills)", fontsize=16)
plt.xlabel("Importance Score", fontsize=12)
plt.tight_layout()
plt.savefig("data/feature_importance.png", dpi=150)
plt.close()
print("📊 Feature importance chart saved → data/feature_importance.png\n")

# ── 9. Save Model & Encoder ───────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/career_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("💾 Model saved    → models/career_model.pkl")
print("💾 Encoder saved  → models/label_encoder.pkl")
print("\n🚀 All done! Ready for Phase 4 — Streamlit Dashboard.")