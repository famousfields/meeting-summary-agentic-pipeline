Meeting Summary Pipeline

A Python-based pipeline that transcribes and summarizes meetings using speech-to-text and NLP models. Designed to help users automatically generate concise meeting notes from audio recordings.

Features

-Converts meeting audio into text using OpenAI Whisper (CPU/GPU fallback supported).

-Summarizes long transcripts into concise notes using NLP techniques.

-Memory-efficient and easily configurable for local environments.

-Supports multiple audio formats via ffmpeg.

Installation
1) Clone the repository
  git clone https://github.com/famousfields/meeting-summary-agentic-pipeline.git
  cd meeting-summary-agentic-pipeline
2)Set up Python environment (recommended with Poetry):
  poetry install
  poetry shell
3) Install dependencies manually (if not using Poetry):
    pip install torch transformers ffmpeg-python

Usage
1) record audio using 'space key' whenever u want to speak
2) press escape to end audio detection
3) recieve summarized text based on audio input

Dependencies

Python 3.10+
PyTorch
Transformers
FFmpeg

Notes

-GPU acceleration is supported but optional.

-Ensure ffmpeg is installed and available in your system PATH.

-Works best with clean audio for accurate transcription.
