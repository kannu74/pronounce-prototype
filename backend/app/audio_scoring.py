import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path
import soundfile as sf
from typing import Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

_processor = None
_model = None

def _load_model():
    global _processor, _model
    if _processor is None or _model is None:
        _processor = Wav2Vec2Processor.from_pretrained(_MODEL_NAME)
        _model = Wav2Vec2Model.from_pretrained(_MODEL_NAME).to(DEVICE)
        _model.eval()
    return _processor, _model

def load_audio_mono(path: str, sample_rate: int = 16000):
    wav, sr = torchaudio.load(path)
    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0), sample_rate

def get_embedding(audio_path: str) -> torch.Tensor:
    processor, model = _load_model()
    waveform, sr = load_audio_mono(audio_path, sample_rate=16000)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(inputs.input_values.to(DEVICE))
        # outputs.last_hidden_state: (batch, time, hidden_dim)
        hidden_states = outputs.last_hidden_state[0]  # (time, hidden_dim)

    # Mean pooling over time → single vector
    embedding = hidden_states.mean(dim=0)
    return embedding

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    return float(torch.dot(a, b).item())