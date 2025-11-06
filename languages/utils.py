"""
Utility functions for downloading and managing model files.
"""
from pathlib import Path
import requests
from .model_config import PIPER_MODELS, MODELS_DIR


def download_file(url: str, dest_path: Path):
    """
    Download a file from a URL to a destination path.
    
    Args:
        url: URL to download from
        dest_path: Local path to save the file
    
    Raises:
        requests.exceptions.RequestException: If download fails
    """
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise


def ensure_piper_model(lang_code: str):
    """
    Ensure that the required Piper TTS model for a language is present.
    Downloads the model files if they don't exist locally.
    
    Args:
        lang_code: Language code (e.g., 'en', 'vi')
    """
    if lang_code not in PIPER_MODELS:
        print(f"Warning: No Piper model definition for language '{lang_code}'.")
        return
        
    MODELS_DIR.mkdir(exist_ok=True)
    
    model_info = PIPER_MODELS[lang_code]
    model_name = model_info['name']
    
    onnx_path = MODELS_DIR / f'{model_name}.onnx'
    json_path = MODELS_DIR / f'{model_name}.onnx.json'
    
    if not onnx_path.exists():
        download_file(model_info['onnx'], onnx_path)
    if not json_path.exists():
        download_file(model_info['json'], json_path)
