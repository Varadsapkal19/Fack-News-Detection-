"""
Fake News Detection System
===========================
Enhanced ML pipeline using TF-IDF + Passive Aggressive Classifier
with detailed evaluation metrics and prediction utilities.
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, f1_score, precision_score, recall_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Text Preprocessing ───────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Clean and normalize article text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)          # Remove URLs
    text = re.sub(r"<.*?>", "", text)                       # Remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                  # Keep only letters
    text = re.sub(r"\s+", " ", text).strip()               # Normalize whitespace
    return text


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate the dataset."""
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    required_cols = {"Body", "Label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df["Body"] = df["Body"].fillna("").apply(preprocess_text)
    df = df.dropna(subset=["Label"])
    df = df[df["Body"].str.strip() != ""]

    logger.info(f"Loaded {len(df):,} samples | Labels: {df['Label'].value_counts().to_dict()}")
    return df


# ─── Model Training ───────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 7):
    """Train the fake news classifier and return model artifacts + metrics."""
    X = df["Body"]
    y = df["Label"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {len(x_train):,} | Test: {len(x_test):,}")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 2),       # Bigrams improve context capture
        sublinear_tf=True,        # Log normalization
        max_features=50_000,
    )
    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_test = vectorizer.transform(x_test)

    # Passive Aggressive Classifier
    pac = PassiveAggressiveClassifier(
        max_iter=100,
        C=0.5,
        tol=1e-4,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
    )
    pac.fit(tfidf_train, y_train)

    # Predictions & metrics
    y_pred = pac.predict(tfidf_test)
    metrics = compute_metrics(y_test, y_pred)

    logger.info(f"Accuracy: {metrics['accuracy']:.2%} | F1: {metrics['f1']:.4f}")
    return pac, vectorizer, x_test, y_test, y_pred, metrics


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    """Compute comprehensive evaluation metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report":    classification_report(y_true, y_pred),
    }


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, labels, save_path: str = "confusion_matrix.png"):
    """Plot and save the confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor="white", annot_kws={"size": 14})
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


def plot_metrics_bar(metrics: dict, save_path: str = "metrics.png"):
    """Bar chart of key performance metrics."""
    keys   = ["accuracy", "f1", "precision", "recall"]
    values = [metrics[k] for k in keys]
    colors = ["#2563eb", "#16a34a", "#d97706", "#dc2626"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(keys, values, color=colors, height=0.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Metrics chart saved → {save_path}")


# ─── Save / Load Model ────────────────────────────────────────────────────────

def save_model(pac, vectorizer, path: str = "model"):
    """Persist model artifacts."""
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/classifier.pkl", "wb") as f:
        pickle.dump(pac, f)
    with open(f"{path}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Model saved → {path}/")


def load_model(path: str = "model"):
    """Load persisted model artifacts."""
    with open(f"{path}/classifier.pkl", "rb") as f:
        pac = pickle.load(f)
    with open(f"{path}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return pac, vectorizer


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict(text: str, pac, vectorizer) -> dict:
    """Classify a single article body as REAL or FAKE."""
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    label = pac.predict(vec)[0]
    # Decision function score → proxy confidence
    score = abs(pac.decision_function(vec)[0])
    confidence = min(score / 3.0, 1.0)    # Normalize to [0, 1]
    return {"label": label, "confidence": round(float(confidence), 4)}


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "data.csv"

    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found at '{DATA_PATH}'. Please add data.csv.")
    else:
        df = load_data(DATA_PATH)
        pac, vectorizer, x_test, y_test, y_pred, metrics = train_model(df)

        print("\n" + "="*50)
        print(f"  Accuracy  : {metrics['accuracy']:.2%}")
        print(f"  F1 Score  : {metrics['f1']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print("="*50)
        print("\nClassification Report:\n")
        print(metrics["report"])

        labels = sorted(y_test.unique())
        plot_confusion_matrix(metrics["confusion_matrix"], labels)
        plot_metrics_bar(metrics)
        save_model(pac, vectorizer)
