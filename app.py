from pathlib import Path
import sys
import uuid # For unique filenames
from datasets import load_dataset
from flask import Flask, render_template, request, jsonify
import torch # PyTorch, for HuggingFace models
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile # For saving audio
import librosa # For audio processing


if sys.version_info[0:2] != (3, 12):
    raise Exception('Requires Python 3.12')
app = Flask(__name__)

# Temporary directory to hold TTS audio files
TEMP_AUDIO_DIR = Path(app.static_folder, 'temp_audio')
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# ASR model
print('Loading ASR model...')
device='cuda:0' if torch.cuda.is_available() else 'cpu'
asr_pipeline = pipeline('automatic-speech-recognition', 
                        model='openai/whisper-base.en', device=device)
print('ASR model loaded.')

# TTS model
print('Loading TTS model components...')
tts_processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
tts_model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts').to(device)
tts_vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan').to(device)

# xvector speaker embeddings from the SpeechT5 demo page
speaker_id = 7306
embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]['xvector']).unsqueeze(0).to(device)
print('TTS model components loaded.')

# Sample sentences for dictation
sentences = [
    'The quick brown fox jumps over the lazy dog.',
    'She sells seashells by the seashore.',
    'Peter Piper picked a peck of pickled peppers.',
    'How much wood would a woodchuck chuck if a woodchuck could chuck wood?',
    'The rain in Spain stays mainly in the plain.'
]
current_sentence_index = 0

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    """Provides a new dictation sentence to the frontend."""
    global current_sentence_index
    sentence = sentences[current_sentence_index % len(sentences)]
    current_sentence_index += 1
    return jsonify({'sentence': sentence})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Receives audio data from the frontend, performs ASR, 
    and returns the transcribed text.
    """
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio_data']
    
    # Save the received audio temporarily to be read by librosa/transformers
    # TODO: process in memory
    temp_filename = Path(TEMP_AUDIO_DIR, f'temp_upload_{uuid.uuid4().hex}.wav')
    audio_file.save(temp_filename)

    try:
        # Load audio file using librosa to ensure correct format/sampling rate.
        # Whisper models expect 16kHz mono audio.
        speech_array, _ = librosa.load(temp_filename, sr=16000, mono=True)
        
        # Perform ASR and return transcription
        result = asr_pipeline(speech_array)
        transcription = result['text']
        return jsonify({'transcription': transcription})
    except Exception as e:
        print(f'Error processing audio: {e}')
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the temporary audio file
        if temp_filename.is_file():
            temp_filename.unlink()

@app.route('/synthesize_speech', methods=['POST'])
def synthesize_speech():
    """
    Receives text, synthesizes speech, saves it as a temporary WAV file,
    and returns the URL to that file.
    """
    data = request.get_json()
    text_to_speak = data.get('text')

    if not text_to_speak:
        return jsonify({'error': 'No text provided'}), 400

    try:
        inputs = tts_processor(text=text_to_speak, return_tensors='pt').to(device)
        
        # Generate speech waveform
        with torch.no_grad():
            speech = tts_model.generate_speech(
                inputs['input_ids'], speaker_embeddings, vocoder=tts_vocoder)

        # Save the speech to a temporary file in the static/temp_audio
        # directory to make it accessible via a URL.
        output_filename = f'tts_output_{uuid.uuid4().hex}.wav'
        output_path = Path(TEMP_AUDIO_DIR, output_filename)
        
        # SpeechT5 outputs a PyTorch tensor. Convert to NumPy array and save.
        # The sampling rate for SpeechT5 with HiFiGAN vocoder is typically 16kHz.
        # TODO: get speech from device
        soundfile.write(output_path, speech.cpu().numpy(), samplerate=16000) 
        
        # Construct the URL for the frontend to fetch the audio
        audio_url = str(Path('static', 'temp_audio', output_filename))
        return jsonify({'audio_url': audio_url})
        # TODO: clean up files in finally clause
    except Exception as e:
        print(f'Error synthesizing speech: {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)