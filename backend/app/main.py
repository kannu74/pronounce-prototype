from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydub import AudioSegment
import os
import time
import shutil
import random

# Internal imports
from .transcribe import transcribe_with_words
from .hybrid_scoring import compute_per_word_scores

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Serve static TTS audio
app.mount("/static", StaticFiles(directory=str(UPLOAD_DIR)), name="static")

# Supported language mappings
LANG_MAP = {
    "hindi": "hi", "hi": "hi",
    "english": "en", "en": "en",
    "spanish": "es", "es": "es",
    "french": "fr", "fr": "fr",
    "german": "de", "de": "de",
    "japanese": "ja", "ja": "ja",
    "kannada": "kn", "kn": "kn",
    "tamil": "ta", "ta": "ta",
    "telugu": "te", "te": "te",
    "gujarati": "gu", "gu": "gu"
}

PASSAGE_BANK = {
    "hi": [
        ("hi_1", "आज सुबह मौसम बहुत अच्छा था। मैं पार्क में टहलने गया और वहाँ कई बच्चे खेल रहे थे। कुछ लोग योग कर रहे थे और पक्षियों की आवाज़ें सुनाई दे रही थीं। मुझे यह शांत वातावरण बहुत पसंद आया।"),
        ("hi_2", "विद्यालय में आज एक रोचक कार्यक्रम हुआ। हमारे शिक्षक ने हमें किताबों का महत्व समझाया और कहा कि रोज़ थोड़ा पढ़ना चाहिए। मैंने तय किया कि मैं हर दिन नई कहानी पढ़ूँगा।"),
    ],
    "en": [
        ("en_1", "This morning the weather was pleasant. I went for a walk in the park and saw children playing happily. Some people were exercising, and the sound of birds made the place feel calm and peaceful."),
        ("en_2", "Today we had an interesting session at school. Our teacher explained why reading is important and encouraged us to read daily. I decided to read a new story every day."),
    ],
    "ta": [
        ("ta_1", "இன்று காலை வானிலை மிகவும் நன்றாக இருந்தது. நான் பூங்காவில் நடக்க சென்றேன். அங்கு பல குழந்தைகள் மகிழ்ச்சியாக விளையாடினர். பறவைகளின் குரல் அமைதியாக இருந்தது."),
    ],
    "te": [
        ("te_1", "ఈ రోజు ఉదయం వాతావరణం చాలా మంచిగా ఉంది. నేను పార్క్‌కు నడకకు వెళ్లాను. అక్కడ పిల్లలు ఆనందంగా ఆడుతున్నారు. పక్షుల కిలకిలలు విని నాకు చాలా సంతోషంగా అనిపించింది."),
    ],
    "kn": [
        ("kn_1", "ಇಂದು ಬೆಳಿಗ್ಗೆ ಹವಾಮಾನ ತುಂಬ ಚೆನ್ನಾಗಿತ್ತು. ನಾನು ಉದ್ಯಾನವನಕ್ಕೆ ನಡೆದುಕೊಂಡು ಹೋದೆ. ಅಲ್ಲಿ ಮಕ್ಕಳು ಸಂತೋಷವಾಗಿ ಆಟವಾಡುತ್ತಿದ್ದರು. ಪಕ್ಷಿಗಳ ಶಬ್ದಗಳು ಮನಸ್ಸಿಗೆ ನೆಮ್ಮದಿ ನೀಡಿದವು."),
    ],
    "gu": [
        ("gu_1", "આજે સવારનું હવામાન ખૂબ સરસ હતું. હું બગીચામાં ફરવા ગયો. ત્યાં બાળકો ખુશીથી રમતા હતા. પક્ષીઓનો અવાજ સાંભળીને મને શાંતિ અનુભવાઈ."),
    ],
}

def detect_and_rename(filepath: Path) -> Path:
    """Detect WebM or WAV via magic bytes and correct extension."""
    with open(filepath, "rb") as f:
        header = f.read(4)

    new_path = filepath
    detected = "unknown"

    if header.startswith(b'\x1a\x45\xdf\xa3'):
        detected = "webm"
        if filepath.suffix != ".webm":
            new_path = filepath.with_suffix(".webm")
            os.rename(filepath, new_path)

    elif header.startswith(b'RIFF'):
        detected = "wav"
        if filepath.suffix != ".wav":
            new_path = filepath.with_suffix(".wav")
            os.rename(filepath, new_path)

    print(f"   🔎 Format Detected: {detected.upper()} (Header: {header.hex()})")
    return new_path


@app.post("/process-audio/")
def process_audio(
    file: UploadFile = File(...),
    target_text: str = Form(...),
    language: str = Form("hi")
):
    start_time = time.time()
    temp_raw_path = None
    
    try:
        print("\n" + "="*40)
        print("--- 🎤 Processing Request ---")

        # 1. MAP LANGUAGE
        iso_lang_code = LANG_MAP.get(language.lower().strip(), "en")

        # 2. SAVE RAW AUDIO
        original_ext = Path(file.filename).suffix
        temp_filename = f"raw_{int(time.time())}{original_ext}"
        temp_raw_path = UPLOAD_DIR / temp_filename

        with open(temp_raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. FIX EXTENSION
        final_raw_path = detect_and_rename(temp_raw_path)
        print(f"   💾 Saved as: {final_raw_path}")

        # 4. LOAD + CHECK AUDIO (No Normalization!)
        try:
            audio = AudioSegment.from_file(str(final_raw_path))

            max_db = audio.max_dBFS
            print(f"   🔊 Volume Level: {max_db:.2f} dB")

            if max_db == -float("inf"):
                raise HTTPException(status_code=400, detail="Input audio is silent.")

            if audio.duration_seconds < 0.4:
                raise HTTPException(status_code=400, detail="Audio too short (<0.4s).")

        except Exception as e:
            print(f"❌ Audio Decode Error: {e}")
            raise HTTPException(status_code=400, detail=f"Audio error: {str(e)}")

        # 5. EXPORT CLEAN WAV (No volume normalization)
        clean_filename = f"clean_{int(time.time())}.wav"
        filepath = UPLOAD_DIR / clean_filename

        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(filepath, format="wav")

        # 6. RUN HYBRID SCORING
        print(f"5. Sending to AI (Lang: {iso_lang_code})...")
        scores = compute_per_word_scores(target_text, iso_lang_code, str(filepath))

        recog_text = " ".join([w.get("recognized", "") for w in scores.get("words", [])])
        print(f"6. ✅ Recognized: '{recog_text}'")

        print("="*40 + "\n")
        return scores

    except Exception as e:
        print(f"❌ ERROR: {e}")
        if temp_raw_path and os.path.exists(temp_raw_path):
            try:
                os.remove(temp_raw_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-passage/")
def get_passage(language: str = "hi"):
    iso_lang_code = LANG_MAP.get(language.lower().strip(), "en")
    if iso_lang_code not in PASSAGE_BANK or not PASSAGE_BANK[iso_lang_code]:
        raise HTTPException(status_code=404, detail="No passages available for this language.")

    passage_id, passage = random.choice(PASSAGE_BANK[iso_lang_code])
    return {"language": iso_lang_code, "passage_id": passage_id, "passage": passage}