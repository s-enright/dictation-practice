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
        self.sentences = self._load_sentences()
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.asr_pipeline = None  # Will be loaded when language is selected
        self.models_loaded = False
        
        # Ensure model files are downloaded if using Piper (but don't load yet)
        if self.tts_manager.tts_engine == 'piper':
            ensure_piper_model(self.lang_code)
        
        print('English language initialized (models will load on selection).')

    def _load_sentences(self):
        """Load sentences from the English sentences file."""
        sentences_file = Path(__file__).parent / 'sentences_en.txt'
        try:
            with open(sentences_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            return sentences
        except FileNotFoundError:
            print(f"Warning: {sentences_file} not found. Using default sentences.")
            return [
                'The quick brown fox jumps over the lazy dog.',
                'She sells seashells by the seashore.',
                'Peter Piper picked a peck of pickled peppers.',
                'How much wood would a woodchuck chuck if a woodchuck could chuck wood?',
                'The rain in Spain stays mainly in the plain.'
            ]

    def load_models(self):
        """Load ASR and TTS models for this language."""
        if self.models_loaded:
            return
        
        print('Loading English models...')
        
        # Load ASR model
        self.asr_pipeline = pipeline('automatic-speech-recognition', model='openai/whisper-base.en', device=self.device)
        
        # Load TTS model
        self.tts_manager.load_voice(self.lang_code)
        
        self.models_loaded = True
        print('English models loaded.')

    def transcribe(self, audio_file):
        if not self.models_loaded or self.asr_pipeline is None:
            raise RuntimeError("English models not loaded. Call load_models() first.")
        
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
