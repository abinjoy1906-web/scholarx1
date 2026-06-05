"""
train.py
--------
Training pipeline for the document classifier.

Dataset : 20 Newsgroups (sklearn built-in, downloaded if network is available)
          Falls back to a locally-generated synthetic dataset when offline.
Model   : Multinomial Naive Bayes (fast, strong baseline for text classification)
Features: TF-IDF with sublinear term-frequency scaling
"""

import os
import sys
import csv
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for headless environments)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_corpus

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
DATA_DIR        = os.path.join(BASE_DIR, "data")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

MODEL_PATH      = os.path.join(MODELS_DIR, "classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
LABELS_PATH     = os.path.join(MODELS_DIR, "labels.pkl")
SYNTHETIC_CSV   = os.path.join(DATA_DIR, "newsgroups_synthetic.csv")

# ── Categories used when fetching 20 Newsgroups ───────────────────────────────
NEWSGROUPS_CATEGORIES = [
    "rec.sport.hockey",
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
    "talk.politics.guns",
    "talk.religion.misc",
    "comp.graphics",
    "soc.religion.christian",
]

SHORT_LABELS = {
    "rec.sport.hockey":       "Hockey",
    "rec.sport.baseball":     "Baseball",
    "sci.med":                "Medicine",
    "sci.space":              "Space",
    "talk.politics.guns":     "Politics",
    "talk.religion.misc":     "Religion",
    "comp.graphics":          "Computing",
    "soc.religion.christian": "Christianity",
    # Synthetic dataset uses these directly
    "Hockey":      "Hockey",
    "Baseball":    "Baseball",
    "Medicine":    "Medicine",
    "Space":       "Space",
    "Politics":    "Politics",
    "Religion":    "Religion",
    "Computing":   "Computing",
    "Christianity":"Christianity",
}


# ── 1. Load Dataset ───────────────────────────────────────────────────────────

def _load_from_newsgroups():
    """Try to fetch the 20 Newsgroups dataset from sklearn."""
    from sklearn.datasets import fetch_20newsgroups
    dataset = fetch_20newsgroups(
        subset="all",
        categories=NEWSGROUPS_CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )
    return dataset.data, dataset.target, dataset.target_names


def _load_from_synthetic_csv():
    """Load the locally generated synthetic CSV dataset."""
    if not os.path.exists(SYNTHETIC_CSV):
        print("   Generating synthetic dataset …")
        import make_dataset as mkd
        rows = mkd.generate_dataset(n_per_class=400)
        mkd.save_dataset(rows, SYNTHETIC_CSV)

    texts, labels_str = [], []
    with open(SYNTHETIC_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels_str.append(row["label"])

    target_names = sorted(set(labels_str))
    label_map    = {name: i for i, name in enumerate(target_names)}
    labels       = [label_map[l] for l in labels_str]
    return texts, labels, target_names


def load_data():
    """Load dataset — 20 Newsgroups if reachable, synthetic CSV otherwise.

    Returns:
        Tuple of (texts, labels, target_names).
    """
    print("📥  Loading dataset …")
    try:
        texts, labels, target_names = _load_from_newsgroups()
        print(f"    ✓ 20 Newsgroups: {len(texts)} documents, {len(target_names)} categories")
        return texts, labels, target_names
    except Exception as e:
        print(f"    ⚠  20 Newsgroups unavailable ({type(e).__name__}); using synthetic dataset.")
        texts, labels, target_names = _load_from_synthetic_csv()
        print(f"    ✓ Synthetic CSV: {len(texts)} documents, {len(target_names)} categories")
        return texts, labels, target_names


# ── 2. Preprocess ─────────────────────────────────────────────────────────────

def preprocess_data(texts):
    """Run the full text-cleaning pipeline over all documents."""
    print("🔧  Preprocessing text …")
    cleaned = preprocess_corpus(texts, use_lemmatization=True)
    print("    ✓ Preprocessing complete")
    return cleaned


# ── 3. Vectorise with TF-IDF ─────────────────────────────────────────────────

def build_vectorizer():
    """Construct a TF-IDF vectorizer with tuned parameters.

    Key parameters:
        max_features – Vocabulary cap; keeps the 15 000 most frequent terms.
        ngram_range  – (1, 2) captures unigrams + bigrams for richer context.
        sublinear_tf – Applies log(1+tf) to dampen extreme term frequencies.
        min_df       – Ignore terms appearing in fewer than 2 documents.
        max_df       – Ignore terms appearing in > 95 % of documents (noise).
    """
    return TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )


# ── 4. Train ──────────────────────────────────────────────────────────────────

def train_model(X_train, y_train):
    """Fit a Multinomial Naive Bayes classifier.

    alpha=0.1 is mild Laplace smoothing; empirically better than the
    default 1.0 when used with TF-IDF features.
    """
    print("🤖  Training Multinomial Naive Bayes …")
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train, y_train)
    print("    ✓ Training complete")
    return clf


# ── 5. Evaluate ───────────────────────────────────────────────────────────────

def evaluate_model(clf, vectorizer, X_test_raw, y_test, target_names):
    """Compute accuracy, print classification report, and save confusion matrix."""
    X_test = vectorizer.transform(X_test_raw)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊  Accuracy : {acc:.4f}  ({acc*100:.2f} %)\n")

    short = [SHORT_LABELS.get(n, n) for n in target_names]
    print("Classification Report:")
    print("─" * 60)
    print(classification_report(y_test, y_pred, target_names=short))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short,
        yticklabels=short,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "Confusion Matrix – Document Classifier",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_path = os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    print(f"💾  Confusion matrix saved → {plot_path}")
    plt.close()

    return acc


# ── 6. Persist ────────────────────────────────────────────────────────────────

def save_artifacts(clf, vectorizer, target_names):
    """Pickle the trained classifier, vectorizer, and label list."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(list(target_names), f)

    print(f"\n💾  Saved: {MODEL_PATH}")
    print(f"💾  Saved: {VECTORIZER_PATH}")
    print(f"💾  Saved: {LABELS_PATH}")


def load_artifacts():
    """Load persisted model, vectorizer, and labels from disk."""
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(LABELS_PATH, "rb") as f:
        target_names = pickle.load(f)
    return clf, vectorizer, target_names


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Document Classifier – Training Pipeline")
    print("=" * 60)

    # 1. Load
    texts, labels, target_names = load_data()

    # 2. Preprocess
    cleaned_texts = preprocess_data(texts)

    # 3. Split  (80 % train / 20 % test, stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        cleaned_texts, labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    print(f"\n📂  Train: {len(X_train_raw)}  |  Test: {len(X_test_raw)}")

    # 4. Vectorise (fit ONLY on training data to prevent leakage)
    vectorizer = build_vectorizer()
    X_train    = vectorizer.fit_transform(X_train_raw)
    print(f"    ✓ Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # 5. Train
    clf = train_model(X_train, y_train)

    # 6. Evaluate
    evaluate_model(clf, vectorizer, X_test_raw, y_test, target_names)

    # 7. Save
    save_artifacts(clf, vectorizer, target_names)

    print("\n✅  Pipeline complete.\n")


if __name__ == "__main__":
    main()
