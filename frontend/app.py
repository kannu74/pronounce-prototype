import streamlit as st
import requests
import tempfile
import os
import random
import re
from collections import Counter

st.set_page_config(page_title="Pronounce Prototype", layout="centered")

# --- CSS STYLES ---
st.markdown("""
<style>
    .word-container { padding: 15px; border: 1px solid #eee; border-radius: 8px; margin-bottom: 10px; background: white;}
    .word-correct { color: #155724; background: #d4edda; padding: 2px 5px; border-radius: 4px; font-weight: bold; }
    .word-incorrect { color: #721c24; background: #f8d7da; padding: 2px 5px; border-radius: 4px; font-weight: bold; text-decoration: line-through; }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = "http://localhost:8000/process-audio/"
PASSAGE_URL = "http://localhost:8000/get-passage/"

# --- CONFIGURATION ---
LANGUAGES = {
    "Hindi (हिंदी)": "hi",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Gujarati (ગુજરાતી)": "gu",
    "English": "en"
}

def highlight_paragraph(passage: str, word_results: list) -> str:
    """
    Returns HTML where incorrect target-words are highlighted inline.
    We highlight by word occurrence count so repeated words are handled.
    """
    # Count how many times each target word should be marked incorrect
    incorrect_counts = Counter()
    for w in word_results:
        if w.get("status") == "incorrect":
            target_word = (w.get("word") or "").strip()
            if target_word and target_word not in ["(Extra)"]:
                incorrect_counts[target_word] += 1

    # Tokenize passage preserving separators (spaces/punctuation)
    tokens = re.findall(r"\w+|[^\w\s]|\s+", passage, flags=re.UNICODE)

    seen = Counter()
    out = []
    for tok in tokens:
        # Only process word tokens; keep punctuation/space as-is
        if re.match(r"^\w+$", tok, flags=re.UNICODE):
            seen[tok] += 1
            if incorrect_counts.get(tok, 0) >= seen[tok]:
                out.append(f"<span class='word-incorrect'>{tok}</span>")
            else:
                out.append(f"<span class='word-correct'>{tok}</span>")
        else:
            out.append(tok)

    return "".join(out)

# Simple offline sentence bank
SENTENCE_BANK = {
    "hi": ["नमस्ते, आप कैसे हैं?", "भारत एक विशाल देश है।", "मुझे पानी चाहिए।"],
    "ta": ["வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "எனக்கு தண்ணீர் வேண்டும்.", "தமிழ் என் தாய்மொழி."],
    "te": ["నమస్కారం, మీరు ఎలా ఉన్నారు?", "భారతదేశం నా మాతృభూమి.", "నాకు ఆకలిగా ఉంది."],
    "kn": ["ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?", "ಬೆಂಗಳೂರು ಸುಂದರ ನಗರ.", "ನಾನು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇನೆ."],
    "gu": ["નમસ્તે, તમે કેમ છો?", "મારે પાણી જોઈએ છે.", "ગુજરાત એક સુંદર રાજ્ય છે."],
    "en": ["Hello, how are you?", "The quick brown fox jumps over the lazy dog.", "I love coding."]
}

st.title("🗣️ Polyglot Pronounce")
st.write("AI-Powered Speech Therapy for Indian Languages")

# 1. LANGUAGE SELECTOR
selected_lang_name = st.selectbox("Select Language:", list(LANGUAGES.keys()))
lang_code = LANGUAGES[selected_lang_name]

# 2. CONTENT GENERATION
col1, col2 = st.columns([3, 1])
with col1:
    if "current_passage" not in st.session_state:
        # initial fetch
        r = requests.get(PASSAGE_URL, params={"language": lang_code})
        st.session_state.current_passage = r.json()["passage"] if r.status_code == 200 else ""

    target_text = st.text_area("Read this aloud (Paragraph Test):", value=st.session_state.current_passage, height=180)

with col2:
    if st.button("🎲 New Paragraph"):
        r = requests.get(PASSAGE_URL, params={"language": lang_code})
        if r.status_code == 200:
            st.session_state.current_passage = r.json()["passage"]
        st.rerun()

# 3. RECORDING
audio_data = st.audio_input(f"🎤 Record in {selected_lang_name}")

if audio_data is not None:
    # --- CRITICAL FIX 1: PLAYBACK CHECK ---
    # Try playing this in the browser FIRST. 
    # If you can't hear yourself here, the Microphone isn't working.
    st.audio(audio_data) 

    if st.button("Analyze Pronunciation", type="primary"):
        
        # --- CRITICAL FIX 2: RESET FILE POINTER ---
        # The file might have been read by the st.audio player above.
        # We must rewind to the start (byte 0) to read it again.
        audio_data.seek(0)
        
        # --- CRITICAL FIX 3: USE WEBM EXTENSION ---
        # Browsers record in WebM. Naming it .wav confuses some tools.
        # We use .webm here, and the Backend will convert it to WAV.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_data.read())
            tmp_path = tmp.name

        # Prepare payload
        files = {"file": open(tmp_path, "rb")}
        data = {"target_text": target_text, "language": lang_code}

        with st.spinner("Processing speech..."):
            try:
                response = requests.post(BACKEND_URL, data=data, files=files)
                files["file"].close()
                os.unlink(tmp_path)

                if response.status_code == 200:
                    res = response.json()
                    
                    # --- RESULTS UI ---
                    st.divider()
                    m1, m2 = st.columns(2)
                    m1.metric("Pronunciation Score", f"{res.get('overall_pronunciation_score', 0)}%")
                    m2.metric("Reading Accuracy", f"{res.get('overall_text_score', 0)}%")

                    st.subheader("Detailed Breakdown")
                    words = res.get("words", [])
                    
                    if not words:
                        st.warning("⚠️ No words detected. Audio might be too quiet.")

                    # PASSAGE DISPLAY WITH HIGHLIGHTING
                    st.subheader("Your Passage (Highlighted)")
                    highlighted_html = highlight_paragraph(st.session_state.current_passage, words)
                    st.markdown(f"<div class='word-container' style='font-size:20px; line-height:1.8;'>{highlighted_html}</div>", unsafe_allow_html=True)
                    
                    for w in words:
                        target = w.get("word", "")
                        recognized = w.get("recognized", "")
                        status = w.get("status", "unknown")
                        score = w.get("total_score", 0)

                        color_class = "word-correct" if status == "correct" else "word-incorrect"
                        emoji = "✅" if status == "correct" else "❌"
                        
                        st.markdown(f"""
                        <div class="word-container">
                            <div style="display:flex; justify-content:space-between;">
                                <span style="font-size:1.2em;">{emoji} <b>{target}</b></span>
                                <span style="font-weight:bold; color:{'green' if score > 80 else 'red'}">{score}/100</span>
                            </div>
                            <div style="margin-top:5px; color:#666;">
                                Heard: <span class="{color_class}">{recognized}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")