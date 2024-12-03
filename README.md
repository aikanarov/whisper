# Whisper Project

## Overview
This project is a Python-based implementation utilizing OpenAI's Whisper model for speech recognition and transcription. It provides an easy-to-use interface for converting audio files into text.

## Features
- Speech-to-text transcription using OpenAI's Whisper model
- Support for multiple audio format
- Easy-to-use command-line interface
- Batch processing capabilities

## Prerequisites
- Python 3.8 or higher
- FFmpeg (required for audio processing)
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/whisper.git
cd whisper
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg (if not already installed):
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
- **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Usage

Basic usage:
```bash
python transcribe.py --input path/to/audio/file.mp3
```

Advanced options:
```bash
python transcribe.py --input path/to/audio/file.mp3 --model base --language en --output output.txt
```

### Parameters
- `--input`: Path to the input audio file
- `--model`: Model size (tiny, base, small, medium, large)
- `--language`: Language code (e.g., en, es, fr)
- `--output`: Output file path (default: transcript.txt)

## Models
Available model sizes:
- `tiny`: Fastest, least accurate
- `base`: Good balance of speed and accuracy
- `small`: Better accuracy, slower
- `medium`: High accuracy
- `large`: Highest accuracy, slowest

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [OpenAI Whisper](https://github.com/openai/whisper) for the base model and implementation
- All contributors to this project

## Support
For support, please open an issue in the GitHub repository.
