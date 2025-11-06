from pathlib import Path
from .base import Language
from .tts import TtsManager
from .asr import AsrManager
from .utils import ensure_piper_model


class Vietnamese(Language):
    """Vietnamese language implementation with TTS and ASR support."""
    
    def __init__(self, temp_audio_dir: Path, tts_manager: TtsManager, asr_manager: AsrManager):
        """
        Initialize the Vietnamese language.
        
        Args:
            temp_audio_dir: Directory for temporary audio files
            tts_manager: TTS manager instance
            asr_manager: ASR manager instance
        """
        super().__init__()
        self.temp_audio_dir = temp_audio_dir
        self.tts_manager = tts_manager
        self.asr_manager = asr_manager
        self.lang_code = 'vi'
        self.sentences = self._load_sentences()
        self.models_loaded = False
        
        # Check if ASR is available for Vietnamese
        self._has_asr = self.asr_manager.is_available(self.lang_code)
        
        # Ensure model files are downloaded if using Piper (but don't load yet)
        if self.tts_manager.tts_engine == 'piper':
            ensure_piper_model(self.lang_code)

        print(f'Vietnamese language initialized (ASR available: {self._has_asr}, models will load on selection).')

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
        
        # Load ASR model if available
        if self._has_asr:
            try:
                self.asr_manager.load_model(self.lang_code)
                print('Vietnamese ASR model loaded.')
            except Exception as e:
                print(f'Warning: Failed to load Vietnamese ASR model: {e}')
                self._has_asr = False
        
        # Load TTS model
        self.tts_manager.load_voice(self.lang_code)
        
        self.models_loaded = True
        print('Vietnamese models loaded.')

    def transcribe(self, audio_file):
        """
        Transcribe audio file to text using ASR.
        
        Args:
            audio_file: Audio file to transcribe
        
        Returns:
            Transcribed text
        
        Raises:
            NotImplementedError: If ASR is not available
            RuntimeError: If models are not loaded
        """
        if not self._has_asr:
            raise NotImplementedError(
                "ASR (Automatic Speech Recognition) is not available for Vietnamese. "
                "Please check that the required models are installed."
            )
        
        if not self.models_loaded:
            raise RuntimeError("Vietnamese models not loaded. Call load_models() first.")
        
        return self.asr_manager.transcribe(audio_file, self.lang_code)

    def synthesize(self, text):
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
        
        Returns:
            URL path to the synthesized audio file
        """
        if not self.models_loaded:
            raise RuntimeError("Vietnamese models not loaded. Call load_models() first.")
        
        return self.tts_manager.synthesize(text, self.lang_code)
