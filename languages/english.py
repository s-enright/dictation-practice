from pathlib import Path
import torch
import librosa
import uuid
from transformers import pipeline
from .base import Language
from .tts import TtsManager
from .utils import ensure_piper_model

class English(Language):
    def __init__(self, temp_audio_dir: Path, tts_manager: TtsManager):
        super().__init__()
        self.temp_audio_dir = temp_audio_dir
        self.tts_manager = tts_manager
        self.lang_code = 'en'
        self.sentences = [
            'The quick brown fox jumps over the lazy dog.',
            'She sells seashells by the seashore.',
            'Peter Piper picked a peck of pickled peppers.',
            'How much wood would a woodchuck chuck if a woodchuck could chuck wood?',
            'The rain in Spain stays mainly in the plain.'
        ]
        
        print('Loading English models...')
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # ASR model
        self.asr_pipeline = pipeline('automatic-speech-recognition', model='openai/whisper-base.en', device=self.device)
        
        # Ensure model is downloaded if using Piper
        if self.tts_manager.tts_engine == 'piper':
            ensure_piper_model(self.lang_code)

        # TTS model is loaded lazily by the TtsManager
        self.tts_manager.load_voice(self.lang_code)
        print('English models loaded.')

    def transcribe(self, audio_file):
        temp_filename = self.temp_audio_dir / f'temp_upload_{uuid.uuid4().hex}.wav'
        audio_file.save(temp_filename)
        
        try:
            speech_array, _ = librosa.load(temp_filename, sr=16000, mono=True)
            result = self.asr_pipeline(speech_array)
            return result['text']
        finally:
            if temp_filename.is_file():
                temp_filename.unlink()

    def synthesize(self, text):
        return self.tts_manager.synthesize(text, self.lang_code)
