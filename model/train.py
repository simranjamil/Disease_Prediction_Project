import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("../data/disease_data.csv")
df.columns = df.columns.str.strip()

# ── 2. Encode target ───────────────────────────────────────────────────────────
le = LabelEncoder()
df["Disease_encoded"] = le.fit_transform(df["Disease"])

feature_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing",
                "Gender", "Blood Pressure", "Cholesterol Level"]

X = df[feature_cols].values
y = df["Disease_encoded"].values

# ── 3. Train on full dataset (small data, no split needed) ─────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
)
model.fit(X, y)

# ── 4. Evaluate on training data just to verify ────────────────────────────────
y_pred = model.predict(X)
print(f"Training Accuracy: {accuracy_score(y, y_pred) * 100:.2f}%\n")
print(classification_report(y, y_pred, target_names=le.classes_, zero_division=0))

# ── 5. Save artefacts ──────────────────────────────────────────────────────────
os.makedirs(".", exist_ok=True)
joblib.dump(model, "model.pkl")
joblib.dump(le,    "label_encoder.pkl")
print("Model saved  →  model/model.pkl")
print("Encoder saved →  model/label_encoder.pkl")