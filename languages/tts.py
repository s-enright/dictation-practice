from pathlib import Path
import torch
import soundfile
import wave
import uuid
from transformers import VitsTokenizer, VitsModel
from piper.voice import PiperVoice
from .model_config import get_piper_model_path, get_mms_model_name


class TtsManager:
    """
    Manages TTS models for multiple languages and engines.
    Uses lazy loading to only load models when needed.
    """
    
    def __init__(self, tts_engine: str, temp_audio_dir: Path):
        """
        Initialize the TTS Manager.
        
        Args:
            tts_engine: TTS engine to use ('piper' or 'mms')
            temp_audio_dir: Directory for temporary audio file storage
        """
        self.tts_engine = tts_engine
        self.temp_audio_dir = temp_audio_dir
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.models = {}
        print(f"TTS Manager initialized with engine: {self.tts_engine}, device: {self.device}")

    def _load_piper_voice(self, lang_code: str):
        """Load a Piper TTS voice model."""
        onnx_path, json_path = get_piper_model_path(lang_code)

        if not onnx_path.exists() or not json_path.exists():
            raise FileNotFoundError(
                f"Model files for {lang_code} not found at {onnx_path}. "
                "Ensure they are downloaded."
            )
            
        use_cuda = self.device.startswith('cuda')
        return PiperVoice.load(str(onnx_path), config_path=str(json_path), use_cuda=use_cuda)

    def _load_mms_voice(self, lang_code: str):
        """Load an MMS TTS voice model."""
        model_name = get_mms_model_name(lang_code)
        
        tokenizer = VitsTokenizer.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name).to(self.device)
        return {'tokenizer': tokenizer, 'model': model}

    def load_voice(self, lang_code: str):
        """Loads a voice model for a given language if not already loaded."""
        if lang_code in self.models:
            return
        
        print(f"Loading {self.tts_engine} model for language: {lang_code}...")
        if self.tts_engine == 'piper':
            self.models[lang_code] = self._load_piper_voice(lang_code)
        elif self.tts_engine == 'mms':
            self.models[lang_code] = self._load_mms_voice(lang_code)
        else:
            raise ValueError(f"Unsupported TTS engine: {self.tts_engine}")
        print(f"{lang_code} model loaded.")

    def synthesize(self, text: str, lang_code: str) -> str:
        """
        Synthesize speech and return the URL to the audio file.
        
        Args:
            text: Text to synthesize
            lang_code: Language code (e.g., 'en', 'vi')
        
        Returns:
            URL path to the synthesized audio file
        """
        if lang_code not in self.models:
            self.load_voice(lang_code)

        output_filename = f'tts_output_{uuid.uuid4().hex}.wav'
        output_path = self.temp_audio_dir / output_filename
        
        if self.tts_engine == 'piper':
            voice = self.models[lang_code]
            
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(voice.config.sample_rate)
                for audio_chunk in voice.synthesize(text):
                    wav_file.writeframes(audio_chunk.audio_int16_bytes)

        elif self.tts_engine == 'mms':
            mms_model = self.models[lang_code]
            inputs = mms_model['tokenizer'](text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                speech = mms_model['model'](**inputs).waveform
            
            sampling_rate = mms_model['model'].config.sampling_rate
            soundfile.write(output_path, speech.cpu().numpy().squeeze(), samplerate=sampling_rate)

        # Return URL path using Path for consistency
        return str(Path('static') / 'temp_audio' / output_filename)


# Module-level singleton instance
_tts_manager_instance = None


def get_tts_manager(tts_engine: str = None, temp_audio_dir: Path = None) -> TtsManager:
    """
    Get or create the singleton TTS Manager instance.
    
    Args:
        tts_engine: TTS engine to use (required on first call)
        temp_audio_dir: Directory for temporary audio files (required on first call)
    
    Returns:
        TtsManager instance
    
    Raises:
        ValueError: If required parameters are not provided on first call
    """
    global _tts_manager_instance
    
    if _tts_manager_instance is None:
        if tts_engine is None or temp_audio_dir is None:
            raise ValueError("tts_engine and temp_audio_dir must be provided on first call")
        _tts_manager_instance = TtsManager(tts_engine, temp_audio_dir)
    
    return _tts_manager_instance
