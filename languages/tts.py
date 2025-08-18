from pathlib import Path
import torch
import soundfile
import wave
import uuid
import json
from transformers import VitsTokenizer, VitsModel
from piper.voice import PiperVoice

class TtsManager:
    _instance = None

    def __new__(cls, tts_engine: str, temp_audio_dir: Path):
        if cls._instance is None:
            cls._instance = super(TtsManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, tts_engine: str, temp_audio_dir: Path):
        if self.initialized:
            return
        
        self.tts_engine = tts_engine
        self.temp_audio_dir = temp_audio_dir
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.models = {}

        print(f"Initializing TTS Manager with engine: {self.tts_engine}")
        self.initialized = True

    def _load_piper_voice(self, lang_code: str):
        model_name_map = {
            'en': 'en_US-lessac-medium',
            'vi': 'vi_VN-25hours_single-low'
        }
        model_name = model_name_map.get(lang_code)
        if not model_name:
            raise ValueError(f"Piper TTS model not found for language: {lang_code}")

        onnx_path = Path('models') / f'{model_name}.onnx'
        json_path = Path('models') / f'{model_name}.onnx.json'

        if not onnx_path.exists() or not json_path.exists():
            raise FileNotFoundError(f"Model files for {model_name} not found. Ensure they are downloaded.")
            
        use_cuda = self.device.startswith('cuda')
        return PiperVoice.load(str(onnx_path), config_path=str(json_path), use_cuda=use_cuda)

    def _load_mms_voice(self, lang_code: str):
        model_name_map = {
            'en': 'facebook/mms-tts-eng',
            'vi': 'facebook/mms-tts-vie'
        }
        model_name = model_name_map.get(lang_code)
        if not model_name:
            raise ValueError(f"MMS TTS model not found for language: {lang_code}")
        
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
        """Synthesizes speech and returns the URL to the audio file."""
        if lang_code not in self.models:
            self.load_voice(lang_code)

        output_filename = f'tts_output_{uuid.uuid4().hex}.wav'
        output_path = self.temp_audio_dir / output_filename
        
        if self.tts_engine == 'piper':
            voice = self.models[lang_code]
            with wave.open(str(output_path), 'wb') as wav_file:
                voice.synthesize(text, wav_file)
        
        elif self.tts_engine == 'mms':
            mms_model = self.models[lang_code]
            inputs = mms_model['tokenizer'](text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                speech = mms_model['model'](**inputs).waveform
            
            sampling_rate = mms_model['model'].config.sampling_rate
            soundfile.write(output_path, speech.cpu().numpy().squeeze(), samplerate=sampling_rate)

        return str(Path('static', 'temp_audio', output_filename))
