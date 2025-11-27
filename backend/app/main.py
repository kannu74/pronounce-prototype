from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil

from backend.app.transcribe import transcribe_with_words
from backend.app.scoring import compute_text_score
from backend.app.tts import create_tts

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for saving uploaded audio and TTS files
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Serve uploaded TTS files
app.mount("/static", StaticFiles(directory="uploads"), name="static")


@app.post("/process-audio/")
async def process_audio(
    file: UploadFile = File(...),
    target_text: str = Form(...),
    language: str = Form("hi")
):
    """
    Process uploaded audio:
    1. Save audio
    2. Transcribe audio
    3. Compute word-level scoring
    4. Generate TTS for target text
    """

    # Save uploaded audio
    filepath = UPLOAD_DIR / file.filename
    with filepath.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Transcribe audio
    result = transcribe_with_words(str(filepath), language)
    transcription = result["text"]
    segments = result["words"]  # optional: word-level timestamps

    # Compute scoring (target_text first, then recognized transcription)
    score_full = compute_text_score(target_text, transcription)

    # Convert scoring to frontend-friendly format
    word_results = []
    for w in score_full["word_alignment"]:
        word_results.append({
            "word": w["recognized"] if w["recognized"] else w["target"],
            "correct": w["operation"] == "correct"
        })

    # Generate TTS for the target text
    tts_path = create_tts(target_text, language)
    tts_url = f"/static/{tts_path.name}"

    return {
        "transcription": transcription,
        "word_results": word_results,
        "tts_url": tts_url
    }
