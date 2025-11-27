import unicodedata
import re
from difflib import SequenceMatcher
from jiwer import wer

def normalize_text(text: str) -> str:
    """
    Normalize input text for comparison:
    - Unicode normalization (NFKC)
    - Lowercase
    - Remove punctuation
    - Collapse multiple spaces
    """
    if not text:
        return ""
    # Unicode normalization + lowercase
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    """Tokenize normalized text into words."""
    text = normalize_text(text)
    return text.split() if text else []

def compute_text_score(target: str, recognized: str) -> dict:
    """
    Compute detailed scoring for recognized text vs target text.

    Returns a dict:
    {
      'text_score': float (0-100),
      'wer': float (Word Error Rate),
      'word_alignment': [
         {
           'target': 'expected word or ""',
           'recognized': 'recognized word or ""',
           'operation': 'correct' / 'substitution' / 'insertion' / 'deletion'
         },
         ...
      ]
    }
    """
    target_tokens = tokenize(target)
    rec_tokens = tokenize(recognized)

    if not target_tokens:
        return {
            "text_score": 0.0,
            "wer": 1.0,
            "word_alignment": []
        }

    # Compute global WER
    error_rate = wer(" ".join(target_tokens), " ".join(rec_tokens))
    text_score = max(0.0, 100.0 * (1.0 - error_rate))

    # Compute word-level alignment
    sm = SequenceMatcher(None, target_tokens, rec_tokens)
    word_alignment = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for ti, ri in zip(range(i1, i2), range(j1, j2)):
                word_alignment.append({
                    "target": target_tokens[ti],
                    "recognized": rec_tokens[ri],
                    "operation": "correct"
                })
        elif tag == "replace":
            for ti, ri in zip(range(i1, i2), range(j1, j2)):
                word_alignment.append({
                    "target": target_tokens[ti],
                    "recognized": rec_tokens[ri],
                    "operation": "substitution"
                })
        elif tag == "insert":
            for ri in range(j1, j2):
                word_alignment.append({
                    "target": "",
                    "recognized": rec_tokens[ri],
                    "operation": "insertion"
                })
        elif tag == "delete":
            for ti in range(i1, i2):
                word_alignment.append({
                    "target": target_tokens[ti],
                    "recognized": "",
                    "operation": "deletion"
                })

    return {
        "text_score": float(text_score),
        "wer": float(error_rate),
        "word_alignment": word_alignment
    }
