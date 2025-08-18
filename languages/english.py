from pathlib import Path
import torch
import soundfile
import librosa
import uuid
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from .base import Language

class English(Language):
    def __init__(self, temp_audio_dir: Path):
        super().__init__()
        self.temp_audio_dir = temp_audio_dir
        self.sentences = [
            'The quick brown fox jumps over the lazy dog.',
            'She sells seashells by the seashore.',
            'Peter Piper picked a peck of pickled peppers.',
            'How much wood would a woodchuck chuck if a woodchuck could chuck wood?',
            'The rain in Spain stays mainly in the plain.'
        ]
        
        print('Loading English models...')
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # ASR model
        self.asr_pipeline = pipeline('automatic-speech-recognition', model='openai/whisper-base.en', device=self.device)
        
        # TTS model
        self.tts_processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts').to(self.device)
        self.tts_vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan').to(self.device)
        
        # Speaker embeddings
        speaker_id = 7306
        embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
        self.speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]['xvector']).unsqueeze(0).to(self.device)
        print('English models loaded.')

    def transcribe(self, audio_file):
        temp_filename = self.temp_audio_dir / f'temp_upload_{uuid.uuid4().hex}.wav'
        audio_file.save(temp_filename)
        
        try:
            speech_array, _ = librosa.load(temp_filename, sr=16000, mono=True)
            result = self.asr_pipeline(speech_array)
            return result['text']
        finally:
            if temp_filename.is_file():
                temp_filename.unlink()

    def synthesize(self, text):
        inputs = self.tts_processor(text=text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            speech = self.tts_model.generate_speech(
                inputs['input_ids'], self.speaker_embeddings, vocoder=self.tts_vocoder
            )
        
        output_filename = f'tts_output_{uuid.uuid4().hex}.wav'
        output_path = self.temp_audio_dir / output_filename
        
        soundfile.write(output_path, speech.cpu().numpy(), samplerate=16000)
        
        return str(Path('static', 'temp_audio', output_filename))
