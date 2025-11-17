"""
Pytest configuration and fixtures for model validation tests.
"""
import tempfile
import shutil
from pathlib import Path

import toml
import pytest

from languages.asr import AsrManager
from languages.tts import TtsManager
from languages.model_config import set_vietnamese_asr_model


@pytest.fixture(scope='session')
def config():
    """
    Load configuration from config.toml.
    
    Returns:
        Dictionary containing configuration values
    """
    try:
        with open('config.toml', 'r') as f:
            config_data = toml.load(f)
        
        # Support per-language TTS engine configuration
        if 'tts_engine_en' not in config_data and 'tts_engine' in config_data:
            config_data['tts_engine_en'] = config_data['tts_engine']
        if 'tts_engine_vi' not in config_data and 'tts_engine' in config_data:
            config_data['tts_engine_vi'] = config_data['tts_engine']
        
        # Set defaults if not present
        config_data.setdefault('tts_engine_en', 'piper')
        config_data.setdefault('tts_engine_vi', 'piper')
        
        return config_data
    except FileNotFoundError:
        # Return default configuration
        return {
            'tts_engine_en': 'piper',
            'tts_engine_vi': 'piper',
            'vietnamese_asr_model': 'openai/whisper-small'
        }


@pytest.fixture(scope='session')
def temp_audio_dir():
    """
    Create a temporary directory for audio files during testing.
    This directory is cleaned up after all tests complete.
    
    Yields:
        Path to temporary audio directory
    """
    # Create a temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix='test_audio_'))
    
    yield temp_dir
    
    # Cleanup after all tests
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture(scope='session')
def tts_manager(config, temp_audio_dir):
    """
    Create and configure the TTS manager for testing.
    
    Args:
        config: Configuration fixture
        temp_audio_dir: Temporary audio directory fixture
    
    Returns:
        TtsManager instance
    """
    # Build per-language engine configuration
    tts_engines = {
        'en': config.get('tts_engine_en', 'piper'),
        'vi': config.get('tts_engine_vi', 'piper')
    }
    
    # Create TTS manager with per-language configuration
    manager = TtsManager(temp_audio_dir=temp_audio_dir, tts_engines=tts_engines)
    
    return manager


@pytest.fixture(scope='session')
def asr_manager(config, temp_audio_dir):
    """
    Create and configure the ASR manager for testing.
    
    Args:
        config: Configuration fixture
        temp_audio_dir: Temporary audio directory fixture
    
    Returns:
        AsrManager instance
    """
    # Apply Vietnamese ASR model configuration
    vietnamese_asr_model = config.get('vietnamese_asr_model', 'openai/whisper-small')
    set_vietnamese_asr_model(vietnamese_asr_model)
    
    # Create ASR manager
    manager = AsrManager(temp_audio_dir=temp_audio_dir)
    
    return manager


def pytest_addoption(parser):
    """
    Add custom command-line options for pytest.
    """
    parser.addoption(
        "--num-sentences",
        action="store",
        default="5",
        help="Number of sentences to test (default: 5)"
    )
    parser.addoption(
        "--lang",
        action="store",
        default="all",
        help="Language to test: 'en', 'vi', or 'all' (default: all)"
    )


@pytest.fixture
def num_sentences(request):
    """
    Get the number of sentences to test from command-line option.
    
    Returns:
        Number of sentences to test
    """
    return int(request.config.getoption("--num-sentences"))

