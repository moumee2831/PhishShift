import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Dataset and model paths
DATA_PATH = "data/spam_ham_india.csv"
MODEL_PATH = "models/phishshift_model.pkl"

print("=" * 50)
print("       PHISHSHIFT AI - MODEL TRAINING")
print("=" * 50)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Keep required columns
df = df[["Msg", "Label"]].copy()

# Clean messages and labels
df["Msg"] = df["Msg"].fillna("").astype(str).str.strip()
df["Label"] = df["Label"].fillna("").astype(str).str.strip().str.lower()

# Remove empty messages and invalid labels
df = df[
    (df["Msg"] != "") &
    (df["Label"].isin(["spam", "ham"]))
]

# Remove exact duplicate messages
df = df.drop_duplicates(subset=["Msg"])

# Map labels to the names expected by the dashboard
df["Label"] = df["Label"].map({
    "spam": "phishing",
    "ham": "legitimate"
})

print("\nDataset loaded successfully!")
print("Total messages:", len(df))
print("\nClass distribution:")
print(df["Label"].value_counts())

# Separate inputs and labels
X = df["Msg"]
y = df["Label"]

# Split data into training and testing portions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# Build the ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])

# Train model
print("\nTraining model... Please wait.")
model.fit(X_train, y_train)

# Evaluate on unseen test messages
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 50)
print("           MODEL EVALUATION")
print("=" * 50)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["legitimate", "phishing"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["legitimate", "phishing"]
    )
)

# Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)
print("\nTraining completed!")
print("=" * 50)