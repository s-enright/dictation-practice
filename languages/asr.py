"""
ASR (Automatic Speech Recognition) Manager for handling speech-to-text models.
"""
from pathlib import Path
import uuid
import torch
import librosa
from transformers import pipeline
from .model_config import get_asr_model_name, is_asr_available


class AsrManager:
    """
    Manages ASR models for multiple languages.
    Uses lazy loading to only load models when needed.
    """
    
    def __init__(self, temp_audio_dir: Path):
        """
        Initialize the ASR Manager.
        
        Args:
            temp_audio_dir: Directory for temporary audio file storage
        """
        self.temp_audio_dir = temp_audio_dir
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.models = {}
        print(f"ASR Manager initialized with device: {self.device}")
    
    def is_available(self, lang_code: str) -> bool:
        """
        Check if ASR is available for a given language.
        
        Args:
            lang_code: Language code (e.g., 'en', 'vi')
        
        Returns:
            True if ASR is available, False otherwise
        """
        return is_asr_available(lang_code)
    
    def load_model(self, lang_code: str) -> None:
        """
        Load the ASR model for a given language.
        
        Args:
            lang_code: Language code (e.g., 'en', 'vi')
        
        Raises:
            ValueError: If ASR is not available for the language
        """
        if lang_code in self.models:
            return
        
        if not is_asr_available(lang_code):
            raise ValueError(f"ASR is not available for language: {lang_code}")
        
        print(f"Loading ASR model for {lang_code}...")
        model_name = get_asr_model_name(lang_code)
        self.models[lang_code] = pipeline(
            'automatic-speech-recognition',
            model=model_name,
            device=self.device
        )
        print(f"ASR model loaded for {lang_code}")
    
    def transcribe(self, audio_file, lang_code: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_file: File object containing audio data
            lang_code: Language code (e.g., 'en', 'vi')
        
        Returns:
            Transcribed text
        
        Raises:
            ValueError: If ASR is not available for the language
            RuntimeError: If model is not loaded
        """
        if not is_asr_available(lang_code):
            raise ValueError(f"ASR is not available for language: {lang_code}")
        
        if lang_code not in self.models:
            raise RuntimeError(f"ASR model not loaded for {lang_code}. Call load_model() first.")
        
        # Save uploaded audio to temporary file
        temp_filename = self.temp_audio_dir / f'temp_upload_{uuid.uuid4().hex}.wav'
        audio_file.save(temp_filename)
        
        try:
            # Load audio and convert to 16kHz mono
            speech_array, _ = librosa.load(temp_filename, sr=16000, mono=True)
            
            # Transcribe using the ASR pipeline
            result = self.models[lang_code](speech_array)
            return result['text']
        finally:
            # Clean up temporary file
            if temp_filename.is_file():
                temp_filename.unlink()


# Module-level singleton instance
_asr_manager_instance = None


def get_asr_manager(temp_audio_dir: Path = None) -> AsrManager:
    """
    Get or create the singleton ASR Manager instance.
    
    Args:
        temp_audio_dir: Directory for temporary audio files (required on first call)
    
    Returns:
        AsrManager instance
    
    Raises:
        ValueError: If temp_audio_dir is not provided on first call
    """
    global _asr_manager_instance
    
    if _asr_manager_instance is None:
        if temp_audio_dir is None:
            raise ValueError("temp_audio_dir must be provided on first call")
        _asr_manager_instance = AsrManager(temp_audio_dir)
    
    return _asr_manager_instance

