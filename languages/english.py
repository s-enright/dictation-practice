from pathlib import Path
from .base import Language
from .tts import TtsManager
from .asr import AsrManager
from .utils import ensure_piper_model


class English(Language):
    """English language implementation with ASR and TTS support."""
    
    def __init__(self, temp_audio_dir: Path, tts_manager: TtsManager, asr_manager: AsrManager):
        """
        Initialize the English language.
        
        Args:
            temp_audio_dir: Directory for temporary audio files
            tts_manager: TTS manager instance
            asr_manager: ASR manager instance
        """
        super().__init__()
        self.temp_audio_dir = temp_audio_dir
        self.tts_manager = tts_manager
        self.asr_manager = asr_manager
        self.lang_code = 'en'
        self.sentences = self._load_sentences()
        self.models_loaded = False
        
        # Check if ASR is available for English
        self._has_asr = self.asr_manager.is_available(self.lang_code)
        
        # Ensure model files are downloaded if using Piper (but don't load yet)
        if self.tts_manager.tts_engine == 'piper':
            ensure_piper_model(self.lang_code)
        
        print(f'English language initialized (ASR available: {self._has_asr}, models will load on selection).')

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
        
        # Load ASR model if available
        if self._has_asr:
            try:
                self.asr_manager.load_model(self.lang_code)
                print('English ASR model loaded.')
            except Exception as e:
                print(f'Warning: Failed to load English ASR model: {e}')
                self._has_asr = False
        
        # Load TTS model
        self.tts_manager.load_voice(self.lang_code)
        
        self.models_loaded = True
        print('English models loaded.')

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
                "ASR (Automatic Speech Recognition) is not available for English. "
                "Please check that the required models are installed."
            )
        
        if not self.models_loaded:
            raise RuntimeError("English models not loaded. Call load_models() first.")
        
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
            raise RuntimeError("English models not loaded. Call load_models() first.")
        
        return self.tts_manager.synthesize(text, self.lang_code)
