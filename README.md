# Kannada_Speaker_Conversation_Transcription_And_Speaker_Reocgnition

A Flask application that allows users to upload or record Kannada conversations and automatically identifies speakers, transcribes audio to text with timestamps, and supports real-time addition of new speakers.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [API Endpoints](#api-endpoints)
- [Adding New Speakers](#adding-new-speakers)
- [Configuration](#configuration)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This application provides an end-to-end solution for Kannada speech processing, including speaker diarization (who spoke when), speech-to-text transcription, and speaker identification. It's designed to work with both pre-recorded audio files and real-time recordings.

## Features

- **Audio Input Options**:
  - Upload audio files (.mp3, .wav, .m4a)
  - Record conversations directly in the browser
  
- **Speech Processing**:
  - Automatic speaker diarization (determining who spoke when)
  - Kannada speech-to-text transcription with timestamps
  - Speaker identification from a database of known voices
  
- **Speaker Management**:
  - Register new speakers in real-time
  - Build and maintain a speaker database
  - Auto-update speaker models as more audio samples are provided

- **User Interface**:
  - Clean, intuitive web interface
  - Real-time processing feedback
  - Downloadable transcripts with speaker annotations

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg
- MongoDB (for speaker database)
- CUDA-compatible GPU (recommended for faster processing)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/LokeshBhaskarNR/Kannada_Speaker_Conversation_Transcription_And_Speaker_Reocgnition.git
   cd Kannada_Speaker_Conversation_Transcription_And_Speaker_Reocgnition
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download pre-trained models:
   ```bash
   python download_models.py
   ```

5. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Usage

### Starting the Application
```bash
flask run
```
Access the application at http://localhost:5000

### Processing a Conversation
1. **Upload/Record**: Select an audio file or record directly in the browser
2. **Process**: Click "Process Audio" to start the analysis
3. **Review**: View the transcription with speaker labels and timestamps
4. **Download**: Save the results as JSON, TXT, or SRT format

### Example
```python
# Using the API programmatically
import requests

files = {'audio': open('conversation.wav', 'rb')}
response = requests.post('http://localhost:5000/api/process', files=files)
transcript = response.json()

print(transcript)
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Audio Input    │───▶│  Preprocessing  │───▶│  Diarization    │
│  (Upload/Record)│    │  (Noise, Format)│    │  (Speaker Turns)│
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌────────▼────────┐
│  Final Output   │◀───│  Speaker        │◀───│  Transcription  │
│  (UI/Download)  │    │  Identification │    │  (ASR)          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components
1. **Voice Activity Detection**: Identifies speech segments in audio
2. **Speaker Diarization**: Determines speaker turn-taking using PyAnnote or similar libraries
3. **Kannada ASR Model**: Converts speech to Kannada text (using Whisper fine-tuned model or AI4Bharat's IndicWav2Vec)
4. **Speaker Embeddings**: Creates voice prints for identification using x-vectors or d-vectors
5. **Speaker Database**: Stores voice profiles for identification

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/process` | POST | Process audio file or recording |
| `/api/speakers` | GET | List all registered speakers |
| `/api/speakers` | POST | Register a new speaker |
| `/api/speakers/<id>` | GET | Get specific speaker details |
| `/api/speakers/<id>` | DELETE | Remove a speaker |

## Adding New Speakers

### Via Web Interface
1. Navigate to "Speaker Management"
2. Click "Add New Speaker"
3. Enter speaker details (name, optional metadata)
4. Upload or record voice samples (minimum 30 seconds recommended)
5. Submit to register the new speaker

### Via API
```python
import requests
import json

speaker_data = {
    'name': 'Rajesh Kumar',
    'language': 'Kannada',
    'metadata': json.dumps({'age': 45, 'gender': 'male'})
}

files = {
    'audio_sample': open('voice_sample.wav', 'rb')
}

response = requests.post('http://localhost:5000/api/speakers', 
                        data=speaker_data,
                        files=files)
```

## Configuration

Key configuration options in `.env`:

```
# Server settings
PORT=5000
DEBUG=True

# Audio processing
MAX_AUDIO_LENGTH_SECONDS=3600
SAMPLE_RATE=16000

# Model paths
ASR_MODEL_PATH=./models/kannada_asr_model
DIARIZATION_MODEL_PATH=./models/diarization_model
SPEAKER_EMBEDDING_MODEL=./models/speaker_embedding_model

# Database
MONGODB_URI=mongodb://localhost:27017/speaker_db
```

## Known Limitations

- Optimal performance for conversations with 2-6 speakers
- Limited accuracy in extremely noisy environments
- Better results with higher quality audio (16kHz+, minimal background noise)
- New speaker registration requires at least 20 seconds of clean speech
- Processing long files (>1 hour) may require significant resources

## Troubleshooting

### Common Issues

**Q: Speech recognition accuracy is low**  
A: Ensure clean audio input. Try using a different microphone or recording in a quieter environment.

**Q: Speaker identification is inaccurate**  
A: Add more voice samples for each speaker. Make sure samples are clear and contain only one speaker.

**Q: Application crashes with long audio files**  
A: Increase the `MAX_MEMORY` setting in `.env` or split large files before processing.

**Q: Models fail to download**  
A: Check your internet connection and firewall settings. You may need to download models manually.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
