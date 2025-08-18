from abc import ABC, abstractmethod

class Language(ABC):
    """
    Abstract base class for a language, defining the interface for transcription,
    text-to-speech, and providing dictation sentences.
    """
    def __init__(self):
        self.sentences = []
        self._current_sentence_index = 0

    @abstractmethod
    def transcribe(self, audio_data):
        """
        Transcribes the given audio data.
        
        :param audio_data: The audio data to transcribe.
        :return: The transcribed text.
        """
        pass

    @abstractmethod
    def synthesize(self, text):
        """
        Synthesizes speech from the given text.
        
        :param text: The text to synthesize.
        :return: The path to the synthesized audio file.
        """
        pass

    def get_sentence(self):
        """
        Returns the next sentence for dictation.
        
        :return: A sentence string.
        """
        sentence = self.sentences[self._current_sentence_index % len(self.sentences)]
        self._current_sentence_index += 1
        return sentence
