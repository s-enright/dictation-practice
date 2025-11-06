"""
Centralized configuration for all models used in the application.
This includes TTS (Piper, MMS) and ASR (Whisper) model configurations.
"""
from pathlib import Path

# Base directories
MODELS_DIR = Path('models')

# Piper TTS Model Configuration
PIPER_MODELS = {
    'en': {
        'name': 'en_US-lessac-medium',
        'onnx': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx',
        'json': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json'
    },
    'vi': {
        'name': 'vi_VN-25hours_single-low',
        'onnx': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/25hours_single/low/vi_VN-25hours_single-low.onnx',
        'json': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/25hours_single/low/vi_VN-25hours_single-low.onnx.json'
    }
}

# MMS TTS Model Configuration
MMS_TTS_MODELS = {
    'en': 'facebook/mms-tts-eng',
    'vi': 'facebook/mms-tts-vie'
}

# ASR (Automatic Speech Recognition) Model Configuration
# Note: Vietnamese model can be overridden via config
ASR_MODELS = {
    'en': {
        'model': 'openai/whisper-base.en',
        'available': True
    },
    'vi': {
        'model': 'openai/whisper-small',  # Default: Multilingual Whisper model for Vietnamese
        'available': True
    }
}

def set_vietnamese_asr_model(model_name: str):
    """
    Set the ASR model for Vietnamese language.
    This allows configuration-based model selection.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'openai/whisper-small')
    """
    if 'vi' in ASR_MODELS:
        ASR_MODELS['vi']['model'] = model_name
        print(f"Vietnamese ASR model set to: {model_name}")

def get_piper_model_path(lang_code: str) -> tuple[Path, Path]:
    """
    Get the paths to the ONNX and JSON files for a Piper model.
    
    Args:
        lang_code: Language code (e.g., 'en', 'vi')
    
    Returns:
        Tuple of (onnx_path, json_path)
    
    Raises:
        ValueError: If language is not supported
    """
    if lang_code not in PIPER_MODELS:
        raise ValueError(f"Piper TTS model not found for language: {lang_code}")
    
    model_info = PIPER_MODELS[lang_code]
    model_name = model_info['name']
    
    onnx_path = MODELS_DIR / f'{model_name}.onnx'
    json_path = MODELS_DIR / f'{model_name}.onnx.json'
    
    return onnx_path, json_path

def get_mms_model_name(lang_code: str) -> str:
    """
    Get the HuggingFace model name for MMS TTS.
    
    Args:
        lang_code: Language code (e.g., 'en', 'vi')
    
    Returns:
        Model name string
    
    Raises:
        ValueError: If language is not supported
    """
    if lang_code not in MMS_TTS_MODELS:
        raise ValueError(f"MMS TTS model not found for language: {lang_code}")
    
    return MMS_TTS_MODELS[lang_code]

def get_asr_model_name(lang_code: str) -> str:
    """
    Get the ASR model name for a language.
    
    Args:
        lang_code: Language code (e.g., 'en', 'vi')
    
    Returns:
        Model name string or None if not available
    
    Raises:
        ValueError: If language is not supported or ASR not available
    """
    if lang_code not in ASR_MODELS:
        raise ValueError(f"ASR configuration not found for language: {lang_code}")
    
    config = ASR_MODELS[lang_code]
    if not config['available']:
        raise ValueError(f"ASR is not available for language: {lang_code}")
    
    return config['model']

def is_asr_available(lang_code: str) -> bool:
    """
    Check if ASR is available for a given language.
    
    Args:
        lang_code: Language code (e.g., 'en', 'vi')
    
    Returns:
        True if ASR is available, False otherwise
    """
    if lang_code not in ASR_MODELS:
        return False
    return ASR_MODELS[lang_code]['available']

