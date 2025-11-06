# Vietnamese ASR Implementation

## Overview

Vietnamese Automatic Speech Recognition (ASR) has been successfully implemented using OpenAI's Whisper-small model. The implementation focuses on supporting the Southern Vietnamese dialect and is optimized to run on systems with 8GB VRAM or less.

## Model Information

**Default Model:** `openai/whisper-small`
- **Size:** ~967MB (~1GB VRAM)
- **Language Support:** Multilingual, including Vietnamese
- **Dialect Support:** Works with all Vietnamese dialects including Southern
- **Performance:** Good accuracy while maintaining resource efficiency

## Configuration

The Vietnamese ASR model can be configured in `config.toml`:

```toml
# ASR model for Vietnamese
vietnamese_asr_model = "openai/whisper-small"
```

### Alternative Models

You can switch to other models if desired:

1. **dragonSwing/wav2vec2-base-vn-270h** - Vietnamese-specific model (~270h training data)
   - Lighter weight option
   - Trained specifically on Vietnamese speech

To use an alternative model, simply update the `vietnamese_asr_model` value in `config.toml`.

## Features

- ✅ Real-time speech-to-text transcription for Vietnamese
- ✅ Support for Southern Vietnamese dialect
- ✅ Runs efficiently on systems with 8GB VRAM or less
- ✅ CPU fallback when GPU is unavailable
- ✅ Graceful error handling when ASR is unavailable
- ✅ UI indicators for ASR availability per language

## Usage

1. **Start the Application:**
   ```bash
   python app.py
   ```

2. **Select Vietnamese Language:**
   - Open the web interface
   - Select "VI" from the language dropdown
   - Click "Set Language"
   - Wait for models to load

3. **Use Speech Recognition:**
   - Click "Start Recording" to begin recording
   - Speak in Vietnamese (Southern dialect supported)
   - Click "Stop Recording" to process
   - View transcription in the output field

## Testing

The implementation includes:
- Model loading verification
- VRAM usage monitoring
- End-to-end workflow testing
- Startup configuration validation

## Architecture

The Vietnamese ASR implementation follows the refactored architecture:

- **AsrManager** (`languages/asr.py`): Manages ASR models across languages
- **Vietnamese Class** (`languages/vietnamese.py`): Delegates to AsrManager
- **Model Config** (`languages/model_config.py`): Centralized model definitions
- **Frontend** (`static/js/main.js`): ASR availability UI updates

## Notes

- **First Use:** The Whisper-small model will be downloaded automatically (~967MB)
- **VRAM Usage:** Model uses approximately 1GB VRAM on GPU
- **CPU Mode:** Works on CPU if no GPU available (slower processing)
- **Dialect Accuracy:** While optimized for general Vietnamese, the model handles Southern dialect well

## Troubleshooting

**If ASR is not available:**
1. Check that models are downloading correctly
2. Verify internet connection for first-time download
3. Check console for error messages
4. Ensure sufficient disk space for model storage

**If transcription quality is poor:**
1. Ensure clear audio input
2. Minimize background noise
3. Speak clearly and at moderate pace
4. Consider trying alternative model in config.toml

## Future Enhancements

Potential improvements:
- Fine-tuning on Southern Vietnamese dialect dataset
- Support for additional Vietnamese-specific models
- Dialect-specific configuration options
- Custom vocabulary support

