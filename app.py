import sys
import toml
from pathlib import Path
import os
import requests

from flask import Flask, render_template, jsonify, request, session

from languages.english import English
from languages.vietnamese import Vietnamese
from languages.tts import TtsManager

if sys.version_info[0:2] != (3, 12):
    raise Exception('Requires Python 3.12')

app = Flask(__name__)
app.secret_key = 'bun_bo_hue'

# Load configuration
try:
    with open('config.toml', 'r') as f:
        config = toml.load(f)
    tts_engine = config.get('tts_engine', 'piper')  # Default to piper if not specified
except FileNotFoundError:
    print("Warning: config.toml not found. Using default TTS engine 'piper'.")
    tts_engine = 'piper'

# Temporary directory to hold TTS audio files
TEMP_AUDIO_DIR = Path(app.static_folder) / 'temp_audio'
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# TTS Manager (Singleton)
tts_manager = TtsManager(tts_engine=tts_engine, temp_audio_dir=TEMP_AUDIO_DIR)

# Language manager
languages = {
    'en': English(TEMP_AUDIO_DIR, tts_manager),
    'vi': Vietnamese(TEMP_AUDIO_DIR, tts_manager)
}

def get_language(lang_code):
    return languages.get(lang_code)

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html', languages=languages.keys())

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    """Provides a new dictation sentence to the frontend."""
    lang_code = session.get('language', 'en')
    language = get_language(lang_code)
    if not language:
        return jsonify({'error': 'Invalid language selected'}), 400
    sentence = language.get_sentence()
    return jsonify({'sentence': sentence})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Receives audio data from the frontend, performs ASR, 
    and returns the transcribed text.
    """
    lang_code = session.get('language', 'en')
    language = get_language(lang_code)
    if not language:
        return jsonify({'error': 'Invalid language selected'}), 400

    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio_data']
    
    try:
        transcription = language.transcribe(audio_file)
        return jsonify({'transcription': transcription})
    except Exception as e:
        print(f'Error processing audio: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/synthesize_speech', methods=['POST'])
def synthesize_speech():
    """
    Receives text, synthesizes speech, saves it as a temporary WAV file,
    and returns the URL to that file.
    """
    lang_code = session.get('language', 'en')
    language = get_language(lang_code)
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
        return jsonify({'error': str(e)}), 500

@app.route('/set_language', methods=['POST'])
def set_language():
    """Sets the language for the session."""
    data = request.get_json()
    lang_code = data.get('language')
    if lang_code in languages:
        session['language'] = lang_code
        return jsonify({'message': f'Language set to {lang_code}'})
    return jsonify({'error': 'Invalid language selected'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)