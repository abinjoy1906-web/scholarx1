"""
preprocess.py
-------------
Text preprocessing utilities for document classification.
Handles cleaning, tokenization, stopword removal, and stemming/lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources (safe to call multiple times)
def download_nltk_resources():
    """Download all required NLTK data packages."""
    packages = ["stopwords", "punkt", "wordnet", "omw-1.4", "punkt_tab"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


download_nltk_resources()

# Initialize NLP tools
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def to_lowercase(text: str) -> str:
    """Convert all characters in text to lowercase.

    Args:
        text: Raw input string.

    Returns:
        Lowercased string.
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Strip punctuation and special characters from text.

    Args:
        text: Input string.

    Returns:
        String with punctuation removed.
    """
    # Remove punctuation using str.translate
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace artifacts
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords from text.

    Args:
        text: Input string (should already be lowercased).

    Returns:
        String with stopwords removed.
    """
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in _stop_words and word.isalpha()]
    return " ".join(filtered)


def stem_text(text: str) -> str:
    """Apply Porter Stemming to reduce words to their root form.

    Example: 'running' -> 'run', 'happily' -> 'happili'

    Args:
        text: Input string.

    Returns:
        Stemmed string.
    """
    tokens = word_tokenize(text)
    stemmed = [_stemmer.stem(word) for word in tokens]
    return " ".join(stemmed)


def lemmatize_text(text: str) -> str:
    """Apply WordNet Lemmatization to reduce words to dictionary form.

    More linguistically accurate than stemming.
    Example: 'running' -> 'running', 'better' -> 'good'

    Args:
        text: Input string.

    Returns:
        Lemmatized string.
    """
    tokens = word_tokenize(text)
    lemmatized = [_lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)


def preprocess_text(
    text: str,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> str:
    """Full preprocessing pipeline for a single document.

    Steps applied in order:
        1. Lowercase conversion
        2. Punctuation removal
        3. Stopword removal
        4. Optional stemming OR lemmatization

    Args:
        text:               Raw document string.
        use_stemming:       If True, apply Porter Stemming.
        use_lemmatization:  If True (and use_stemming is False), apply lemmatization.

    Returns:
        Cleaned, processed string ready for vectorization.
    """
    if not isinstance(text, str):
        text = str(text)

    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)

    if use_stemming:
        text = stem_text(text)
    elif use_lemmatization:
        text = lemmatize_text(text)

    return text


def preprocess_corpus(
    texts: list,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> list:
    """Apply preprocessing pipeline to a list of documents.

    Args:
        texts:              List of raw document strings.
        use_stemming:       Apply stemming if True.
        use_lemmatization:  Apply lemmatization if True and stemming is False.

    Returns:
        List of cleaned document strings.
    """
    return [
        preprocess_text(doc, use_stemming=use_stemming, use_lemmatization=use_lemmatization)
        for doc in texts
    ]
