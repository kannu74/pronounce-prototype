import whisper

model = whisper.load_model("tiny")  # or "medium", "large" if GPU available

def transcribe_with_words(audio_path: str, language: str = "hi"):
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        word_timestamps=True
    )
    # result["segments"] each have "words" with {"word", "start", "end"}
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": float(w["start"]),
                "end": float(w["end"])
            })
    return {
        "text": result["text"],
        "words": words
    }