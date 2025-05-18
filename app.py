import os
import uuid # For unique filenames
from datasets import load_dataset
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch # PyTorch, for HuggingFace models
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf # For saving audio
import librosa # For audio processing

# --- Application Setup ---
app = Flask(__name__)
# Create a temporary directory for audio files if it doesn't exist
# This is where TTS output will be saved before being sent to the user.
# For a production app, you'd want a more robust solution for temp files.
TEMP_AUDIO_DIR = os.path.join(app.static_folder, 'temp_audio')
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)


# --- Global Variables & Model Loading ---
# It's crucial to load models only ONCE when the app starts, not per request.
# This significantly speeds up responses after the initial load.

# ASR Model (e.g., Whisper)
# Using a smaller Whisper model for quicker loading and less resource usage.
# For higher accuracy, consider "openai/whisper-medium" or "openai/whisper-large-v2"
# but they require more VRAM/RAM and are slower.
print("Loading ASR model...")
# device="cuda:0" if torch.cuda.is_available() else "cpu" # Use GPU if available
# For simplicity and broader compatibility on Windows without specific CUDA setup:
device = "cpu" 
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=device)
print("ASR model loaded.")

# TTS Model (e.g., SpeechT5)
print("Loading TTS model components...")
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
# Download xvector speaker embeddings from the SpeechT5 demo page (HuggingFace Hub)
# This provides a standard voice. You can explore other speaker embeddings for different voices.

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device) # Example speaker
print("TTS model components loaded.")

# Sample sentences for dictation
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "The rain in Spain stays mainly in the plain."
]
current_sentence_index = 0 # Simple way to cycle through sentences

# --- Flask Routes (API Endpoints) ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    """Provides a new sentence to the frontend."""
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
    # A more advanced approach might process it in memory.
    temp_filename = os.path.join(TEMP_AUDIO_DIR, f"temp_upload_{uuid.uuid4().hex}.wav")
    audio_file.save(temp_filename)

    try:
        # Load audio file using librosa to ensure correct format/sampling rate if needed
        # Whisper models expect 16kHz mono audio.
        speech_array, sampling_rate = librosa.load(temp_filename, sr=16000, mono=True)
        
        # Perform ASR
        # For whisper, we can pass the numpy array directly.
        # If passing a file path: result = asr_pipeline(temp_filename)
        result = asr_pipeline(speech_array)
        transcription = result["text"]
        
        return jsonify({'transcription': transcription})
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

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
        inputs = tts_processor(text=text_to_speak, return_tensors="pt").to(device)
        
        # Generate speech waveform
        with torch.no_grad(): # Important for inference to save memory and compute
            speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=tts_vocoder)

        # Save the speech to a temporary file in the static/temp_audio directory
        # This makes it accessible via a URL.
        output_filename = f"tts_output_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(TEMP_AUDIO_DIR, output_filename)
        
        # SpeechT5 outputs a PyTorch tensor. Convert to NumPy array and save.
        # The sampling rate for SpeechT5 with HiFiGAN vocoder is typically 16kHz.
        sf.write(output_path, speech.cpu().numpy(), samplerate=16000) 
        
        # Construct the URL for the frontend to fetch the audio
        # url_for('static', filename=...) creates the correct path
        audio_url = os.path.join('static', 'temp_audio', output_filename).replace("\\","/") # Ensure forward slashes for URL
        
        return jsonify({'audio_url': audio_url})
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return jsonify({'error': str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # debug=True is helpful for development (auto-reloads on code changes)
    # but should be False in production.
    # host='0.0.0.0' makes the app accessible from other devices on your network.
    app.run(debug=True, host='0.0.0.0', port=5000)