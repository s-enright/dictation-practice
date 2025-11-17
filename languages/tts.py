from pathlib import Path
import torch
import soundfile
import wave
import uuid
import numpy as np
from transformers import VitsTokenizer, VitsModel
from piper.voice import PiperVoice
from huggingface_hub import hf_hub_download
from .model_config import get_piper_model_path, get_mms_model_name, get_vieneu_model_info


class TtsManager:
    """
    Manages TTS models for multiple languages and engines.
    Uses lazy loading to only load models when needed.
    Supports per-language engine configuration.
    """
    
    def __init__(self, tts_engine: str = None, temp_audio_dir: Path = None, tts_engines: dict = None):
        """
        Initialize the TTS Manager.
        
        Args:
            tts_engine: Default TTS engine to use (for backward compatibility)
            temp_audio_dir: Directory for temporary audio file storage
            tts_engines: Dictionary mapping language codes to TTS engines (e.g., {'en': 'piper', 'vi': 'vieneu-tts'})
        """
        self.temp_audio_dir = temp_audio_dir
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.models = {}
        
        # Support per-language engine configuration
        if tts_engines:
            self.tts_engines = tts_engines
            self.tts_engine = tts_engine  # Fallback for unsupported languages
            engines_str = ', '.join([f"{lang}: {eng}" for lang, eng in tts_engines.items()])
            print(f"TTS Manager initialized with per-language engines: {engines_str}, device: {self.device}")
        else:
            # Backward compatibility: use single engine for all languages
            self.tts_engine = tts_engine
            self.tts_engines = {}
            print(f"TTS Manager initialized with engine: {self.tts_engine}, device: {self.device}")
    
    def _get_engine_for_language(self, lang_code: str) -> str:
        """
        Get the TTS engine for a specific language.
        
        Args:
            lang_code: Language code (e.g., 'en', 'vi')
        
        Returns:
            TTS engine name
        """
        if self.tts_engines and lang_code in self.tts_engines:
            return self.tts_engines[lang_code]
        return self.tts_engine

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

    def _load_vieneu_voice(self, lang_code: str):
        """Load a VieNeu-TTS voice model."""
        try:
            # VieNeu-TTS uses a custom implementation that needs to be loaded from the hub
            # The model is based on VITS architecture with Vietnamese phonemization
            from huggingface_hub import snapshot_download
            import sys
            import os
        except ImportError:
            raise ImportError(
                "VieNeu-TTS dependencies are not installed. Please ensure huggingface_hub is installed."
            )
        
        model_info = get_vieneu_model_info(lang_code)
        
        # Download the entire model repository
        model_path = snapshot_download(
            repo_id=model_info['repo_id'],
            allow_patterns=["*.pth", "*.json", "*.txt", "*.py", "config/*", "checkpoints/*"],
        )
        
        print(f"Loading VieNeu-TTS model from {model_path}")
        
        # Add the model path to sys.path to import its modules
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
        
        try:
            # Import VieNeu-TTS modules
            from vieneu_tts import VieNeuTTS
            
            # Initialize the model
            model = VieNeuTTS(model_path, device=self.device)
            
            return {
                'model': model,
                'sample_rate': model_info['sample_rate'],
                'model_path': model_path
            }
        except ImportError as e:
            raise ImportError(
                f"Failed to import VieNeu-TTS modules: {e}. "
                "Please ensure eSpeak NG is installed on your system. "
                "Installation instructions: https://github.com/espeak-ng/espeak-ng"
            )

    def _synthesize_vieneu(self, text: str, lang_code: str, output_path: Path):
        """
        Synthesize speech using VieNeu-TTS model.
        
        Args:
            text: Text to synthesize
            lang_code: Language code
            output_path: Path to save the output audio file
        """
        vieneu_model = self.models[lang_code]
        model = vieneu_model['model']
        sample_rate = vieneu_model['sample_rate']
        
        print(f"Generating audio with VieNeu-TTS (device: {self.device})...")
        
        try:
            # Generate speech using VieNeu-TTS
            audio = model.synthesize(text)
            
            # Save the audio file
            soundfile.write(output_path, audio, sample_rate)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to synthesize speech with VieNeu-TTS: {e}. "
                "Please ensure eSpeak NG is installed on your system."
            )

    def load_voice(self, lang_code: str):
        """Loads a voice model for a given language if not already loaded."""
        if lang_code in self.models:
            return
        
        # Get the engine for this specific language
        engine = self._get_engine_for_language(lang_code)
        
        print(f"Loading {engine} TTS model for language: {lang_code}...")
        if engine == 'piper':
            self.models[lang_code] = self._load_piper_voice(lang_code)
        elif engine == 'mms':
            self.models[lang_code] = self._load_mms_voice(lang_code)
        elif engine == 'vieneu-tts':
            self.models[lang_code] = self._load_vieneu_voice(lang_code)
        else:
            raise ValueError(f"Unsupported TTS engine: {engine}")
        print(f"{lang_code} TTS model loaded.")

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
        
        # Get the engine for this specific language
        engine = self._get_engine_for_language(lang_code)
        
        if engine == 'piper':
            voice = self.models[lang_code]
            
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(voice.config.sample_rate)
                for audio_chunk in voice.synthesize(text):
                    wav_file.writeframes(audio_chunk.audio_int16_bytes)

        elif engine == 'mms':
            mms_model = self.models[lang_code]
            inputs = mms_model['tokenizer'](text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                speech = mms_model['model'](**inputs).waveform
            
            sampling_rate = mms_model['model'].config.sampling_rate
            soundfile.write(output_path, speech.cpu().numpy().squeeze(), samplerate=sampling_rate)

        elif engine == 'vieneu-tts':
            self._synthesize_vieneu(text, lang_code, output_path)

        # Return URL path using Path for consistency
        return str(Path('static') / 'temp_audio' / output_filename)


# Module-level singleton instance
_tts_manager_instance = None


def get_tts_manager(tts_engine: str = None, temp_audio_dir: Path = None, tts_engines: dict = None) -> TtsManager:
    """
    Get or create the singleton TTS Manager instance.
    
    Args:
        tts_engine: Default TTS engine to use (for backward compatibility)
        temp_audio_dir: Directory for temporary audio files (required on first call)
        tts_engines: Dictionary mapping language codes to TTS engines (optional)
    
    Returns:
        TtsManager instance
    
    Raises:
        ValueError: If required parameters are not provided on first call
    """
    global _tts_manager_instance
    
    if _tts_manager_instance is None:
        if temp_audio_dir is None:
            raise ValueError("temp_audio_dir must be provided on first call")
        if tts_engine is None and not tts_engines:
            raise ValueError("Either tts_engine or tts_engines must be provided on first call")
        _tts_manager_instance = TtsManager(tts_engine, temp_audio_dir, tts_engines)
    
    return _tts_manager_instance
