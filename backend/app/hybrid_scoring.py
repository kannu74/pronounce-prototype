import difflib
import re
from .transcribe import transcribe_with_words

def compute_per_word_scores(target_text, lang_code, audio_path):
    """
    Fuzzy word comparison for Indic languages.
    Allows partial matches (e.g., 'मुझे' vs 'मूजे', 'ನೀನು' vs 'ನಿನು').
    """

    # 1. TRANSCRIBE
    transcription_result = transcribe_with_words(audio_path, lang_code)
    recognized_text = transcription_result["text"]

    # 2. SPLIT SENTENCES INTO WORDS (UPDATED TO HANDLE PARAGRAPH WORD TIMESTAMPS)
    target_words = re.findall(r"\w+", target_text, flags=re.UNICODE)
    recog_words  = re.findall(r"\w+", recognized_text, flags=re.UNICODE)

    # 3. ALIGN TARGET ↔ RECOGNIZED USING DIFF LOGIC
    matcher = difflib.SequenceMatcher(None, target_words, recog_words)

    detailed_words = []
    total_score = 0
    matched_count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        # --- EXACT MATCH ---
        if tag == "equal":
            for i in range(i1, i2):
                word = target_words[i]
                detailed_words.append({
                    "word": word,
                    "recognized": word,
                    "status": "correct",
                    "total_score": 100.0
                })
                total_score += 100.0
                matched_count += 1

        # --- SUBSTITUTION (Fuzzy Match) ---
        elif tag == "replace":
            len_t = i2 - i1
            len_r = j2 - j1

            for k in range(max(len_t, len_r)):
                t_word = target_words[i1 + k] if k < len_t else ""
                r_word = recog_words[j1 + k] if k < len_r else ""

                if t_word and r_word:
                    similarity = difflib.SequenceMatcher(None, t_word, r_word).ratio()
                    word_score = round(similarity * 100, 1)
                else:
                    word_score = 0.0

                status = "correct" if word_score >= 80 else "incorrect"

                detailed_words.append({
                    "word": t_word if t_word else "(Extra)",
                    "recognized": r_word if r_word else "(Missed)",
                    "status": status,
                    "total_score": word_score
                })

                if t_word:
                    total_score += word_score
                    matched_count += 1

        # --- DELETION ---
        elif tag == "delete":
            for i in range(i1, i2):
                detailed_words.append({
                    "word": target_words[i],
                    "recognized": "-",
                    "status": "incorrect",
                    "total_score": 0.0
                })
                matched_count += 1

        # --- INSERTION (Extra Words) ---
        elif tag == "insert":
            for j in range(j1, j2):
                detailed_words.append({
                    "word": "(Extra)",
                    "recognized": recog_words[j],
                    "status": "extra",
                    "total_score": 0.0
                })

    # 4. FINAL SCORES
    if matched_count > 0:
        final_pronunciation_score = round(total_score / matched_count, 1)
    else:
        final_pronunciation_score = 0.0

    # Fix: Only evaluate accuracy for real target words
    target_only = [w for w in detailed_words if w["word"] not in ["(Extra)", ""]]

    perfect_matches = len([w for w in target_only if w["total_score"] >= 90])
    reading_accuracy = round((perfect_matches / len(target_only) * 100), 1) if target_only else 0

    return {
        "overall_pronunciation_score": final_pronunciation_score,
        "overall_text_score": reading_accuracy,
        "words": detailed_words
    }
