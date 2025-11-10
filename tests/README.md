# Model Validation Tests

This test suite validates the accuracy of TTS (Text-to-Speech) and ASR (Automatic Speech Recognition) models by generating speech from text and comparing the transcription with the original input.

## Overview

The tests perform the following steps for each sentence:
1. Generate audio from text using the TTS model
2. Transcribe the generated audio using the ASR model
3. Compare the transcription with the original text
4. Calculate word match percentage
5. Clean up temporary audio files

After testing N sentences, the tests display:
- Average match percentage
- Standard deviation
- Minimum and maximum match percentages

## Installation

Ensure you have pytest installed:

```bash
pip install pytest
```

All other dependencies should already be installed from the main `requirements.txt` file.

## Running the Tests

### Basic Usage

Run all tests with default settings (5 sentences per language):

```bash
pytest tests/test_model_validation.py -s -v
```

**Note:** The `-s` flag (or `--capture=no`) disables output capturing so you can see detailed progress. The `-v` flag provides verbose test names. These flags are set by default in `pytest.ini`, so you can also just run `pytest tests/test_model_validation.py`.

### Custom Number of Sentences

Test with a specific number of sentences:

```bash
pytest tests/test_model_validation.py -s -v --num-sentences 10
```

### Test Specific Language

Test only English models:

```bash
pytest tests/test_model_validation.py::TestModelValidation::test_english_models -s -v
```

Test only Vietnamese models:

```bash
pytest tests/test_model_validation.py::TestModelValidation::test_vietnamese_models -s -v
```

### Combined Options

Test English with 15 sentences:

```bash
pytest tests/test_model_validation.py::TestModelValidation::test_english_models -s -v --num-sentences 15
```

## Output Format

The tests will display detailed output for each sentence:

```
Test 1/5
--------------------------------------------------------------------------------
Original:     The quick brown fox jumps over the lazy dog.
Transcribed:  The quick brown fox jumps over the lazy dog.
Match:        100.00%

Test 2/5
--------------------------------------------------------------------------------
Original:     She sells seashells by the seashore.
Transcribed:  She sells sea shells by the seashore.
Match:        85.71%

...

================================================================================
Summary for EN
================================================================================

Total sentences tested:  5
Average match:           92.50%
Standard deviation:      5.23%
Minimum match:           82.14%
Maximum match:           100.00%
```

## Configuration

The tests use the same configuration as the main application (`config.toml`):
- `tts_engine`: TTS engine to use ('piper' or 'mms')
- `vietnamese_asr_model`: ASR model for Vietnamese

## Notes

- Tests require that both TTS and ASR models are available for the language
- If ASR is not available for a language, that test will be skipped
- Temporary audio files are automatically cleaned up after each test
- All test audio files are stored in a temporary directory that is deleted after test completion

## Troubleshooting

### Model Loading Issues

If you encounter model loading errors, ensure:
1. The model files are present in the `models/` directory
2. The required Python packages (transformers, torch, etc.) are installed
3. You have sufficient disk space and memory

### Memory Issues

If you run out of memory:
- Reduce the number of sentences being tested with `--num-sentences`
- Close other applications
- Test languages separately instead of running all tests at once

### ASR Not Available

If you see "ASR not available" messages:
- Check that the ASR model is properly configured in `config.toml`
- Verify that the model can be loaded by the ASR manager
- Some languages may only have TTS support without ASR

