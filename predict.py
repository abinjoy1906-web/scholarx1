"""
predict.py
----------
Prediction module for the trained document classifier.

Loads persisted model artefacts and exposes a simple API for
classifying new text documents.

Usage (CLI):
    python src/predict.py "NASA scientists discovered a new exoplanet."

Usage (module):
    from src.predict import predict_category
    result = predict_category("Space exploration news text here")
"""

import os
import pickle
import sys

# Allow direct execution from any working directory
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_text

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
MODEL_PATH      = os.path.join(MODELS_DIR, "classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
LABELS_PATH     = os.path.join(MODELS_DIR, "labels.pkl")

# Short human-readable labels
SHORT_LABELS = {
    "rec.sport.hockey":       "Hockey",
    "rec.sport.baseball":     "Baseball",
    "sci.med":                "Medicine",
    "sci.space":              "Space",
    "talk.politics.guns":     "Politics",
    "talk.religion.misc":     "Religion",
    "comp.graphics":          "Computing",
    "soc.religion.christian": "Christianity",
}


def load_artifacts():
    """Load trained model, vectorizer, and label list from disk.

    Returns:
        Tuple (clf, vectorizer, target_names).

    Raises:
        FileNotFoundError: If any artefact file is missing.
    """
    for path in (MODEL_PATH, VECTORIZER_PATH, LABELS_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artefact: {path}\n"
                "Please run `python src/train.py` first."
            )

    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(LABELS_PATH, "rb") as f:
        target_names = pickle.load(f)

    return clf, vectorizer, target_names


def predict_category(text: str, top_n: int = 3) -> dict:
    """Predict the category of a single document.

    Args:
        text:  Raw input text to classify.
        top_n: Number of top predictions to return.

    Returns:
        Dictionary with keys:
            predicted_label  – Human-readable predicted category.
            raw_label        – Full newsgroup label string.
            confidence       – Probability of the top prediction (0–1).
            top_predictions  – List of dicts {label, confidence} for top_n classes.

    Raises:
        ValueError: If input text is empty after preprocessing.
    """
    # Load artefacts (cached at module level on repeated calls via _cache)
    clf, vectorizer, target_names = load_artifacts()

    # Preprocess
    cleaned = preprocess_text(text)
    if not cleaned.strip():
        raise ValueError("Input text is empty or contains no meaningful content after preprocessing.")

    # Vectorise
    X = vectorizer.transform([cleaned])

    # Predict probabilities
    proba = clf.predict_proba(X)[0]           # shape: (n_classes,)
    pred_idx = proba.argmax()

    raw_label       = target_names[pred_idx]
    predicted_label = SHORT_LABELS.get(raw_label, raw_label)
    confidence      = float(proba[pred_idx])

    # Build top-N list
    top_indices = proba.argsort()[::-1][:top_n]
    top_predictions = [
        {
            "label":      SHORT_LABELS.get(target_names[i], target_names[i]),
            "raw_label":  target_names[i],
            "confidence": float(proba[i]),
        }
        for i in top_indices
    ]

    return {
        "predicted_label": predicted_label,
        "raw_label":       raw_label,
        "confidence":      confidence,
        "top_predictions": top_predictions,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    """Interactive CLI for document classification."""
    # If text is passed as a command-line argument, classify it and exit
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        print("\n📄  Document Classifier – Prediction Module")
        print("─" * 48)
        print("Enter text to classify (empty line to quit):\n")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "":
                break
            lines.append(line)
        text = " ".join(lines)

    if not text.strip():
        print("No input provided. Exiting.")
        sys.exit(0)

    try:
        result = predict_category(text)
    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n⚠️  {e}")
        sys.exit(1)

    print(f"\n🏷️  Predicted Category : {result['predicted_label']}")
    print(f"📊  Confidence         : {result['confidence']*100:.1f} %")
    print("\nTop Predictions:")
    print("─" * 36)
    for i, p in enumerate(result["top_predictions"], 1):
        bar_len = int(p["confidence"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {i}. {p['label']:<14} {bar}  {p['confidence']*100:5.1f} %")
    print()


if __name__ == "__main__":
    main()
