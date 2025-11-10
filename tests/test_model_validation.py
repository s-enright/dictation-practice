"""
Test suite for validating TTS and ASR model accuracy.

This module tests the end-to-end pipeline of:
1. Generating speech from text using TTS
2. Transcribing the generated audio using ASR
3. Comparing the transcription with the original text
4. Reporting accuracy statistics
"""
import random
import string
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by converting to lowercase and removing punctuation.
    
    Args:
        text: Text to normalize
    
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def calculate_word_match_percentage(original: str, transcribed: str) -> float:
    """
    Calculate the percentage of words that match between original and transcribed text.
    
    Args:
        original: Original input text
        transcribed: Transcribed text from ASR
    
    Returns:
        Percentage of matching words (0-100)
    """
    # Normalize both texts
    original_normalized = normalize_text(original)
    transcribed_normalized = normalize_text(transcribed)
    
    # Split into words
    original_words = original_normalized.split()
    transcribed_words = transcribed_normalized.split()
    
    # Handle edge case of empty original
    if not original_words:
        return 100.0 if not transcribed_words else 0.0
    
    # Handle edge case of empty transcription
    if not transcribed_words:
        return 0.0
    
    # Count matching words
    # We'll use a simple approach: count how many words from the original
    # appear in the transcribed text in the same order
    matches = 0
    transcribed_idx = 0
    
    for original_word in original_words:
        # Look for the word in the remaining transcribed words
        while transcribed_idx < len(transcribed_words):
            if transcribed_words[transcribed_idx] == original_word:
                matches += 1
                transcribed_idx += 1
                break
            transcribed_idx += 1
    
    # Calculate percentage based on original word count
    percentage = (matches / len(original_words)) * 100.0
    
    return percentage


class TestModelValidation:
    """Test class for TTS and ASR model validation."""
    
    def test_english_models(self, tts_manager, asr_manager, temp_audio_dir, num_sentences):
        """
        Test English TTS and ASR models.
        
        Args:
            tts_manager: TTS manager fixture
            asr_manager: ASR manager fixture
            temp_audio_dir: Temporary audio directory fixture
            num_sentences: Number of sentences to test (from pytest fixture)
        """
        self._test_language_models('en', num_sentences, tts_manager, asr_manager, temp_audio_dir)
    
    def test_vietnamese_models(self, tts_manager, asr_manager, temp_audio_dir, num_sentences):
        """
        Test Vietnamese TTS and ASR models.
        
        Args:
            tts_manager: TTS manager fixture
            asr_manager: ASR manager fixture
            temp_audio_dir: Temporary audio directory fixture
            num_sentences: Number of sentences to test (from pytest fixture)
        """
        self._test_language_models('vi', num_sentences, tts_manager, asr_manager, temp_audio_dir)
    
    def _test_language_models(
        self, 
        lang_code: str, 
        num_sentences: int, 
        tts_manager, 
        asr_manager, 
        temp_audio_dir: Path
    ):
        """
        Test TTS and ASR models for a specific language.
        
        Args:
            lang_code: Language code ('en' or 'vi')
            num_sentences: Number of sentences to test
            tts_manager: TTS manager instance
            asr_manager: ASR manager instance
            temp_audio_dir: Temporary directory for audio files
        """
        print(f"\n{'='*80}")
        print(f"Testing {lang_code.upper()} Models")
        print(f"{'='*80}\n")
        
        # Check if ASR is available for this language
        if not asr_manager.is_available(lang_code):
            pytest.skip(f"ASR not available for {lang_code}")
        
        # Load models
        print(f"Loading {lang_code.upper()} models...")
        tts_manager.load_voice(lang_code)
        asr_manager.load_model(lang_code)
        print("Models loaded successfully.\n")
        
        # Load sentences
        sentences = self._load_sentences(lang_code)
        
        if len(sentences) < num_sentences:
            print(f"Warning: Only {len(sentences)} sentences available, testing all of them.")
            num_sentences = len(sentences)
        
        # Randomly select sentences
        selected_sentences = random.sample(sentences, num_sentences)
        
        # Store results
        results: List[Tuple[str, str, float]] = []
        
        # Test each sentence
        for idx, sentence in enumerate(selected_sentences, 1):
            print(f"Test {idx}/{num_sentences}")
            print(f"{'-'*80}")
            
            audio_file_path = None
            
            try:
                # Step 1: Generate audio using TTS
                print(f"   Original: {sentence}")
                audio_path_url = tts_manager.synthesize(sentence, lang_code)
                
                # Convert URL path to actual file path
                # audio_path_url is like 'static/temp_audio/tts_output_xxx.wav'
                # But the actual file is in temp_audio_dir
                filename = Path(audio_path_url).name
                audio_file_path = temp_audio_dir / filename
                
                # Step 2: Transcribe audio using ASR
                # Create a file-like object that the ASR manager expects
                class AudioFile:
                    """Wrapper class to provide a file-like interface for the ASR manager."""
                    def __init__(self, path):
                        self.path = Path(path)
                    
                    def save(self, destination):
                        """Copy the audio file to the destination."""
                        import shutil
                        shutil.copy(self.path, destination)
                
                audio_file_obj = AudioFile(audio_file_path)
                transcription = asr_manager.transcribe(audio_file_obj, lang_code)
                
                print(f"Transcribed: {transcription.strip()}")
                
                # Step 3: Calculate match percentage
                match_percentage = calculate_word_match_percentage(sentence, transcription)
                print(f"      Match: {match_percentage:.2f}%")
                
                results.append((sentence, transcription, match_percentage))
                
            except Exception as e:
                print(f"Error processing sentence: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            finally:
                # Step 4: Clean up temporary files
                if audio_file_path and audio_file_path.exists():
                    try:
                        audio_file_path.unlink()
                    except Exception as e:
                        print(f"Warning: Could not delete {audio_file_path}: {e}")
            
            print()
        
        # Display summary statistics
        self._display_summary(results, lang_code)
    
    def _load_sentences(self, lang_code: str) -> List[str]:
        """
        Load sentences from the language-specific file.
        
        Args:
            lang_code: Language code ('en' or 'vi')
        
        Returns:
            List of sentences
        """
        sentences_file = Path('languages') / f'sentences_{lang_code}.txt'
        
        try:
            with open(sentences_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            return sentences
        except FileNotFoundError:
            pytest.fail(f"Sentences file not found: {sentences_file}")
    
    def _display_summary(self, results: List[Tuple[str, str, float]], lang_code: str):
        """
        Display summary statistics for the test results.
        
        Args:
            results: List of tuples (original, transcribed, match_percentage)
            lang_code: Language code
        """
        print(f"{'='*80}")
        print(f"Summary for {lang_code.upper()}")
        print(f"{'='*80}\n")
        
        if not results:
            print("No results to summarize.")
            return
        
        # Extract match percentages
        percentages = [result[2] for result in results]
        
        # Calculate statistics
        mean_percentage = np.mean(percentages)
        std_percentage = np.std(percentages)
        min_percentage = np.min(percentages)
        max_percentage = np.max(percentages)
        
        print(f"Total sentences tested:  {len(results)}")
        print(f"Average match:           {mean_percentage:.2f}%")
        print(f"Standard deviation:      {std_percentage:.2f}%")
        print(f"Minimum match:           {min_percentage:.2f}%")
        print(f"Maximum match:           {max_percentage:.2f}%")
        print(f"\n{'='*80}\n")

