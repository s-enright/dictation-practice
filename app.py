from pathlib import Path
import sys
import uuid
from datasets import load_dataset
from flask import Flask, render_template, request, redirect
import torch
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile # For saving audio
import librosa # For audio processing


if sys.version_info[0:2] != (3, 12):
    raise Exception('Requires Python 3.12')
app = Flask(__name__)


class AudioServer:
    # Temporary directory to hold TTS audio files
    TEMP_AUDIO_DIR = Path(app.static_folder, 'temp_audio')
    TEMP_AUDIO_DIR.mkdir(exist_ok=True)
    # Sample sentences for dictation
    sentences = [
        'The quick brown fox jumps over the lazy dog.',
        'She sells seashells by the seashore.',
        'Peter Piper picked a peck of pickled peppers.',
        'How much wood would a woodchuck chuck if a woodchuck could chuck wood?',
        'The rain in Spain stays mainly in the plain.'
    ]

    def __init__(self):
        self.current_sentence_index = 0
        self.device = 'cpu'
        self.asr_pipeline = None
        self.tts_processor = None
        self.tts_model = None
        self.tts_vocoder = None
        self.speaker_embeddings = None

    def load_asr_model(self):
        # ASR model
        print('Loading ASR model...')
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
        self.asr_pipeline = pipeline('automatic-speech-recognition',
                                model='openai/whisper-base.en', device=self.device)
        print('ASR model loaded.')

    def load_tts_model(self):
        # TTS model
        print('Loading TTS model components...')
        self.tts_processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts').to(self.device)
        self.tts_vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan').to(self.device)

        # xvector speaker embeddings from the SpeechT5 demo page
        speaker_id = 7306
        embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
        self.speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]['xvector']).unsqueeze(0).to(self.device)
        print('TTS model components loaded.')

    def get_sentence(self):
        sentence = self.sentences[self.current_sentence_index % len(self.sentences)]
        self.current_sentence_index += 1
        return {'sentence': sentence}

    def transcribe_audio(self):
        if 'audio_data' not in request.files:
            return {'error': 'No audio file provided'}, 400

        audio_file = request.files['audio_data']

        # Save the received audio temporarily to be read by librosa/transformers
        # TODO: process in memory
        temp_filename = Path(self.TEMP_AUDIO_DIR, f'temp_upload_{uuid.uuid4().hex}.wav')
        audio_file.save(temp_filename)

        try:
            # Load audio file using librosa to ensure correct format/sampling rate.
            # Whisper models expect 16kHz mono audio.
            speech_array, _ = librosa.load(temp_filename, sr=16000, mono=True)

            # Perform ASR and return transcription
            result = self.asr_pipeline(speech_array)
            transcription = result['text']
            return {'transcription': transcription}
        except Exception as e:
            print(f'Error processing audio: {e}')
            return {'error': str(e)}, 500
        finally:
            # Clean up the temporary audio file
            if temp_filename.is_file():
                temp_filename.unlink()

    def synthesize_speech(self):
        data = request.get_json()
        text_to_speak = data.get('text')

        if not text_to_speak:
            return {'error': 'No text provided'}, 400
        try:
            inputs = self.tts_processor(text=text_to_speak, return_tensors='pt').to(self.device)

            # Generate speech waveform
            with torch.no_grad():
                speech = self.tts_model.generate_speech(
                    inputs['input_ids'], self.speaker_embeddings, vocoder=self.tts_vocoder)

            # Save the speech to a temporary file in the static/temp_audio
            # directory to make it accessible via a URL.
            output_filename = f'tts_output_{uuid.uuid4().hex}.wav'
            output_path = Path(self.TEMP_AUDIO_DIR, output_filename)

            # SpeechT5 outputs a PyTorch tensor. Convert to NumPy array and save.
            # The sampling rate for SpeechT5 with HiFiGAN vocoder is typically 16kHz.
            # TODO: get speech from device
            soundfile.write(output_path, speech.cpu().numpy(), samplerate=16000)

            # Construct the URL for the frontend to fetch the audio
            audio_url = str(Path('static', 'temp_audio', output_filename))
            return {'audio_url': audio_url}
            # TODO: clean up files in finally clause
        except Exception as e:
            print(f'Error synthesizing speech: {e}')
            return {'error': str(e)}, 500

audio_server = AudioServer()

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html', colours=['red', 'blue'])

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    """Provides a new dictation sentence to the frontend."""
    return audio_server.get_sentence()

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Receives audio data from the frontend, performs ASR, 
    and returns the transcribed text.
    """
    return audio_server.transcribe_audio()

@app.route('/synthesize_speech', methods=['POST'])
def synthesize_speech():
    """
    Receives text, synthesizes speech, saves it as a temporary WAV file,
    and returns the URL to that file.
    """
    return audio_server.synthesize_speech()

@app.route('/dropdown', methods = ['POST'])
def lang_dropdown():
    dropdownval = request.form.get('colour')
    print(dropdownval)
    return redirect("/", code=302)

if __name__ == '__main__':
    audio_server.load_asr_model()
    audio_server.load_tts_model()
    app.run(debug=True, host='0.0.0.0', port=5000)