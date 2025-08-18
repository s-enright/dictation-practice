from pathlib import Path
import requests

MODELS_DIR = Path('models')
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

def download_file(url: str, dest_path: Path):
    """Downloads a file from a URL to a destination path."""
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
    """Ensures that the required Piper TTS model for a language is present."""
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
