# Model Validation Tests - Implementation Summary

## Overview

A comprehensive pytest-based test suite has been created to validate TTS and ASR model accuracy. The tests generate speech from text and measure how accurately the ASR system can transcribe it back.

## Files Created

### 1. `tests/__init__.py`
Empty package initializer for the tests module.

### 2. `tests/conftest.py`
Pytest configuration with fixtures:
- `config`: Loads configuration from `config.toml`
- `temp_audio_dir`: Creates and manages temporary directory for test audio files
- `tts_manager`: Initializes TTS manager with proper configuration
- `asr_manager`: Initializes ASR manager with proper configuration
- `num_sentences`: Command-line option fixture for test customization

Custom pytest options:
- `--num-sentences N`: Specify number of sentences to test (default: 5)
- `--lang LANG`: Specify language ('en', 'vi', or 'all')

### 3. `tests/test_model_validation.py`
Main test implementation with:

**Utility Functions:**
- `normalize_text()`: Normalizes text for comparison (lowercase, no punctuation)
- `calculate_word_match_percentage()`: Calculates percentage of matching words between original and transcribed text

**Test Class: `TestModelValidation`**
- `test_english_models()`: Tests English TTS and ASR
- `test_vietnamese_models()`: Tests Vietnamese TTS and ASR
- `_test_language_models()`: Core testing logic that:
  1. Loads TTS and ASR models
  2. Selects N random sentences
  3. For each sentence:
     - Generates audio with TTS
     - Transcribes audio with ASR
     - Compares transcription with original
     - Calculates word match percentage
     - Cleans up audio files
  4. Displays statistical summary

**Statistical Analysis:**
- Average match percentage
- Standard deviation
- Minimum match percentage
- Maximum match percentage

### 4. `tests/README.md`
Comprehensive documentation covering:
- Test overview and methodology
- Installation instructions
- Usage examples
- Output format
- Configuration details
- Troubleshooting guide

### 5. `requirements.txt` (Updated)
Added `pytest==8.3.4` to dependencies.

## Key Features

### 1. Generalized Testing
- Works with any language that has both TTS and ASR support
- Parameterized by language code and number of sentences
- Automatically loads language-specific sentence files

### 2. Word Matching Algorithm
- Normalizes text (lowercase, removes punctuation)
- Matches words in sequence (maintains word order)
- Returns percentage based on original word count
- Handles edge cases (empty strings, etc.)

### 3. Proper Cleanup
- Uses pytest fixtures with proper teardown
- Cleans up temporary audio files after each test
- Uses try/finally blocks to ensure cleanup even on errors
- Temporary test directory is completely removed after all tests

### 4. Detailed Output
For each sentence:
```
Test 1/5
--------------------------------------------------------------------------------
Original:     The quick brown fox jumps over the lazy dog.
Transcribed:  The quick brown fox jumps over the lazy dog.
Match:        100.00%
```

Summary statistics:
```
================================================================================
Summary for EN
================================================================================

Total sentences tested:  5
Average match:           92.50%
Standard deviation:      5.23%
Minimum match:           82.14%
Maximum match:           100.00%
```

## Usage Examples

### Basic Usage
```bash
# Test all languages with default settings (5 sentences each)
# Note: -s flag shows detailed output, -v shows verbose test names
pytest tests/test_model_validation.py -s -v

# Test with 10 sentences per language
pytest tests/test_model_validation.py -s -v --num-sentences 10

# Test only English
pytest tests/test_model_validation.py::TestModelValidation::test_english_models -s -v

# Test only Vietnamese with 15 sentences
pytest tests/test_model_validation.py::TestModelValidation::test_vietnamese_models -s -v --num-sentences 15
```

**Note:** The `-s` flag (or `--capture=no`) disables output capturing, allowing you to see all the detailed progress output from the tests. These flags are configured by default in `pytest.ini`.

### With Coverage
```bash
pytest tests/test_model_validation.py -s -v --cov=languages --cov-report=html
```

## Technical Details

### TTS → ASR Pipeline
1. **Text Input**: Original sentence from language-specific file
2. **TTS Synthesis**: Generate audio using configured TTS engine (Piper/MMS)
3. **Audio File**: Temporary WAV file saved to test directory
4. **ASR Transcription**: Transcribe audio back to text using ASR model
5. **Comparison**: Calculate word-level match percentage
6. **Cleanup**: Remove temporary audio files

### Word Matching Algorithm
The algorithm performs sequential word matching:
1. Normalize both texts (lowercase, strip punctuation)
2. Split into word lists
3. Iterate through original words
4. For each word, find it in the transcribed text (maintaining order)
5. Count matches and calculate percentage

This approach:
- Respects word order (important for speech recognition)
- Handles insertions and deletions gracefully
- Provides meaningful accuracy metrics

### Error Handling
- Skips tests if ASR is not available for a language
- Provides detailed error messages with stack traces
- Ensures cleanup even on exceptions
- Validates sentence file existence

## Integration with Existing Code

The tests leverage existing infrastructure:
- Uses same TTS and ASR managers as main application
- Respects `config.toml` configuration
- Loads sentences from existing `languages/sentences_*.txt` files
- Compatible with all supported TTS engines and ASR models

## Future Enhancements

Potential improvements:
1. Add more sophisticated matching algorithms (Levenshtein distance, BLEU score)
2. Support for testing specific sentences via command-line
3. Export results to JSON/CSV for analysis
4. Benchmark performance metrics (speed, memory usage)
5. Integration with CI/CD pipelines
6. Comparative analysis between different TTS engines or ASR models

