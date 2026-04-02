Meeting Summary Pipeline

This project records meeting audio, transcribes it with Whisper, optionally assigns speaker labels with pyannote, extracts likely action items, and generates a short summary.

Features

- Records microphone input with push-to-talk using the space bar.
- Captures a second input device for system or loopback audio at the same time.
- Resamples, mixes, denoises, and saves audio to `merged_audio.wav`.
- Transcribes speech with Hugging Face Whisper models.
- Optionally runs speaker diarization when `HF_TOKEN` is available.
- Summarizes longer transcripts in chunks so longer meetings are handled more safely.

Installation

1. Clone the repository.
   `git clone https://github.com/famousfields/meeting-summary-agentic-pipeline.git`
   `cd meeting-summary-agentic-pipeline`
2. Install dependencies with Poetry.
   `poetry install`
3. Make sure `ffmpeg` is installed and available on your `PATH`.
4. Optional: create a `.env` file with:
   `HF_TOKEN=your_huggingface_token`
   `MEETING_MIC_DEVICE=2`
   `MEETING_SYS_DEVICE=4`

Usage

1. Run `poetry run python meeting_pipeline.py` or `python3 meeting_pipeline.py`.
2. Hold the space bar when you want to capture microphone input.
3. Press `Esc` to stop recording and run transcription.
4. Review the printed transcript, action items, and summary.

Device Selection

- `MEETING_MIC_DEVICE` is the input device index for your microphone.
- `MEETING_SYS_DEVICE` is the input device index for your loopback/system capture source.
- If either configured device is invalid, the script prints the available input devices and exits with a clear error.

Notes

- Speaker diarization requires a Hugging Face token that can access `pyannote/speaker-diarization`.
- Without `HF_TOKEN`, the script still transcribes and summarizes audio, but speakers are labeled as `UNKNOWN`.
- Clean loopback audio and mic input improve both transcription and summary quality.
