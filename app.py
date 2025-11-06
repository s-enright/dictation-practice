import toml
import time
from pathlib import Path

from flask import Flask, render_template, jsonify, request, session

from languages.english import English
from languages.vietnamese import Vietnamese
from languages.tts import get_tts_manager
from languages.asr import get_asr_manager
from languages.model_config import set_vietnamese_asr_model


app = Flask(__name__)
app.secret_key = 'bun_bo_hue'

# Load configuration
try:
    with open('config.toml', 'r') as f:
        config = toml.load(f)
    tts_engine = config.get('tts_engine', 'piper')  # Default to piper if not specified
    vietnamese_asr_model = config.get('vietnamese_asr_model', 'openai/whisper-small')
except FileNotFoundError:
    print("Warning: config.toml not found. Using defaults.")
    tts_engine = 'piper'
    vietnamese_asr_model = 'openai/whisper-small'

# Apply Vietnamese ASR model configuration
set_vietnamese_asr_model(vietnamese_asr_model)

# Temporary directory to hold TTS audio files
TEMP_AUDIO_DIR = Path(app.static_folder) / 'temp_audio'
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# Cleanup old audio files on startup
def cleanup_old_audio_files(max_age_hours=1):
    """
    Delete temporary audio files older than max_age_hours.
    
    Args:
        max_age_hours: Maximum age of files to keep (in hours)
    """
    if not TEMP_AUDIO_DIR.exists():
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0
    
    for audio_file in TEMP_AUDIO_DIR.glob('*.wav'):
        try:
            file_age = current_time - audio_file.stat().st_mtime
            if file_age > max_age_seconds:
                audio_file.unlink()
                deleted_count += 1
        except Exception as e:
            print(f"Warning: Could not delete {audio_file}: {e}")
    
    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} old audio file(s)")

# Run cleanup on startup
cleanup_old_audio_files()

# Initialize managers (Singletons)
tts_manager = get_tts_manager(tts_engine=tts_engine, temp_audio_dir=TEMP_AUDIO_DIR)
asr_manager = get_asr_manager(temp_audio_dir=TEMP_AUDIO_DIR)

# Language manager
languages = {
    'en': English(TEMP_AUDIO_DIR, tts_manager, asr_manager),
    'vi': Vietnamese(TEMP_AUDIO_DIR, tts_manager, asr_manager)
}

# Default language
DEFAULT_LANGUAGE = 'en'

def get_language(lang_code=None):
    """
    Get language instance for the given code, or from session if not provided.
    
    Args:
        lang_code: Language code (e.g., 'en', 'vi'), or None to use session default
    
    Returns:
        Language instance or None if invalid
    """
    if lang_code is None:
        lang_code = session.get('language', DEFAULT_LANGUAGE)
    return languages.get(lang_code)

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html', languages=languages.keys())

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    """Provides a new dictation sentence to the frontend."""
    language = get_language()
    if not language:
        return jsonify({'error': 'Invalid language selected'}), 400
    
    # Ensure models are loaded for the selected language
    if not language.models_loaded:
        try:
            language.load_models()
        except Exception as e:
            print(f'Error loading models for {language.lang_code}: {e}')
            return jsonify({'error': f'Failed to load models: {str(e)}'}), 500
    
    sentence = language.get_sentence()
    return jsonify({'sentence': sentence})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Receives audio data from the frontend, performs ASR, 
    and returns the transcribed text.
    """
    language = get_language()
    if not language:
        return jsonify({'error': 'Invalid language selected'}), 400

    # Check if ASR is available for this language
    if not language.has_asr:
        return jsonify({
            'error': f'Speech recognition is not available for {language.lang_code.upper()}. '
                    'Only text-to-speech is currently supported for this language.'
        }), 400

    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio_data']
    
    try:
        transcription = language.transcribe(audio_file)
        return jsonify({'transcription': transcription})
    except NotImplementedError as e:
        print(f'ASR not implemented: {e}')
        return jsonify({'error': str(e)}), 501
    except Exception as e:
        print(f'Error processing audio: {e}')
        return jsonify({'error': f'Failed to process audio: {str(e)}'}), 500

@app.route('/synthesize_speech', methods=['POST'])
def synthesize_speech():
    """
    Receives text, synthesizes speech, saves it as a temporary WAV file,
    and returns the URL to that file.
    """
    language = get_language()
    if not language:
        return jsonify({'error': 'Invalid language selected'}), 400

    data = request.get_json()
    text_to_speak = data.get('text')

    if not text_to_speak:
        return jsonify({'error': 'No text provided'}), 400

    try:
        audio_url = language.synthesize(text_to_speak)
        return jsonify({'audio_url': audio_url})
    except Exception as e:
        print(f'Error synthesizing speech: {e}')
        return jsonify({'error': f'Failed to synthesize speech: {str(e)}'}), 500

@app.route('/set_language', methods=['POST'])
def set_language():
    """Sets the language for the session, loads the language models, and returns a random sentence."""
    data = request.get_json()
    lang_code = data.get('language')
    if lang_code in languages:
        session['language'] = lang_code
        
        # Load models for the selected language
        language = get_language(lang_code)
        try:
            language.load_models()
            # Get a random sentence for the selected language
            sentence = language.get_sentence()
            
            # Include information about ASR availability
            asr_status = 'available' if language.has_asr else 'not available'
            
            return jsonify({
                'message': f'Language set to {lang_code.upper()} and models loaded.',
                'sentence': sentence,
                'has_asr': language.has_asr,
                'asr_status': asr_status
            })
        except Exception as e:
            print(f'Error loading models for {lang_code}: {e}')
            return jsonify({'error': f'Failed to load models for {lang_code}: {str(e)}'}), 500
    return jsonify({'error': 'Invalid language selected'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)