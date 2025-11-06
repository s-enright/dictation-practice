from abc import ABC, abstractmethod
import random


class Language(ABC):
    """
    Abstract base class for a language, defining the interface for transcription,
    text-to-speech, and providing dictation sentences.
    
    ASR (transcription) is optional and may not be available for all languages.
    TTS (synthesis) is required for all languages.
    """
    
    def __init__(self):
        """Initialize the language with empty sentences and ASR availability flag."""
        self.sentences = []
        self._has_asr = True  # Default to True, subclasses can override
    
    @property
    def has_asr(self) -> bool:
        """
        Indicates whether ASR (Automatic Speech Recognition) is available
        for this language.
        
        Returns:
            True if ASR is available, False otherwise
        """
        return self._has_asr

    def transcribe(self, audio_data):
        """
        Transcribes the given audio data to text.
        
        Note: ASR may not be available for all languages. Check has_asr property
        before calling this method.
        
        Args:
            audio_data: The audio data to transcribe
        
        Returns:
            The transcribed text
        
        Raises:
            NotImplementedError: If ASR is not available for this language
        """
        if not self.has_asr:
            raise NotImplementedError(
                f"ASR (Automatic Speech Recognition) is not available for this language. "
                f"This language only supports text-to-speech synthesis."
            )
        # Subclasses with ASR support should override this method
        raise NotImplementedError("Transcription not implemented for this language")

    @abstractmethod
    def synthesize(self, text):
        """
        Synthesizes speech from the given text.
        
        Args:
            text: The text to synthesize
        
        Returns:
            The URL path to the synthesized audio file
        """
        pass

    def get_sentence(self):
        """
        Returns a random sentence for dictation.
        
        Returns:
            A sentence string
        """
        if not self.sentences:
            return "No sentences available for this language."
        return random.choice(self.sentences)
