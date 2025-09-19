from pathlib import Path
from .base import Language
from .tts import TtsManager
from .utils import ensure_piper_model

class Vietnamese(Language):
    def __init__(self, temp_audio_dir: Path, tts_manager: TtsManager):
        super().__init__()
        self.temp_audio_dir = temp_audio_dir
        self.tts_manager = tts_manager
        self.lang_code = 'vi'
        self.sentences = self._load_sentences()
        
        # Placeholder for ASR model (will be implemented later)
        # self.asr_pipeline = None
        self.models_loaded = False
        
        # Ensure model files are downloaded if using Piper (but don't load yet)
        if self.tts_manager.tts_engine == 'piper':
            ensure_piper_model(self.lang_code)

        print('Vietnamese language initialized (models will load on selection).')

    def _load_sentences(self):
        """Load sentences from the Vietnamese sentences file."""
        sentences_file = Path(__file__).parent / 'sentences_vi.txt'
        try:
            with open(sentences_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            return sentences
        except FileNotFoundError:
            print(f"Warning: {sentences_file} not found. Using default sentences.")
            return [
                'Con mèo trèo cây cau.',
                'Bảy quả dưa hấu, ba nghìn một quả.'
            ]

    def load_models(self):
        """Load ASR and TTS models for this language."""
        if self.models_loaded:
            return
        
        print('Loading Vietnamese models...')
        
        # Placeholder for ASR model loading
        # self.asr_pipeline = ...
        
        # Load TTS model
        self.tts_manager.load_voice(self.lang_code)
        
        self.models_loaded = True
        print('Vietnamese models loaded.')

    def transcribe(self, audio_file):
        if not self.models_loaded:
            raise RuntimeError("Vietnamese models not loaded. Call load_models() first.")
        
        # Placeholder for Vietnamese transcription.
        # You would replace this with your actual ASR model.
        print("Warning: Vietnamese ASR is not implemented. Using placeholder.")
        return 'Đây là bản ghi lại tiếng Việt (giả).'

    def synthesize(self, text):
        if not self.models_loaded:
            raise RuntimeError("Vietnamese models not loaded. Call load_models() first.")
        return self.tts_manager.synthesize(text, self.lang_code)
