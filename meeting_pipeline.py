import os
import queue
import re
import threading
import time
from dataclasses import dataclass

import librosa
import noisereduce as nr
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.core import Segment
from pynput import keyboard
from transformers import pipeline as hf_pipeline


TARGET_RATE = 16000
DEFAULT_MIC_DEVICE = 2
DEFAULT_SYS_DEVICE = 4
WHISPER_MODEL = "openai/whisper-small"
SUMMARY_MODEL = "t5-small"
OUTPUT_AUDIO_FILE = "merged_audio.wav"
ACTION_VERBS = [
    "need",
    "decide",
    "assign",
    "do",
    "complete",
    "implement",
    "follow up",
    "prepare",
    "approve",
    "confirm",
    "should",
    "must",
]


@dataclass
class AudioEvent:
    # Keep each captured chunk with enough timing metadata to rebuild the meeting timeline.
    timestamp: float
    source: str
    chunk: np.ndarray
    sample_rate: int


def list_input_devices():
    lines = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            lines.append(
                f"{index}: {device['name']} "
                f"(inputs={device['max_input_channels']}, "
                f"default_sr={int(device['default_samplerate'])})"
            )
    return "\n".join(lines)


def resolve_input_device(env_name, fallback_index):
    raw_value = os.getenv(env_name)
    device_index = int(raw_value) if raw_value is not None else fallback_index

    try:
        device_info = sd.query_devices(device_index, "input")
    except Exception as exc:
        available = list_input_devices()
        raise RuntimeError(
            f"Invalid input device for {env_name}: {device_index}\n"
            f"Available input devices:\n{available}"
        ) from exc

    return device_index, int(device_info["default_samplerate"]), device_info["name"]


def record_audio(mic_device, sys_device, mic_rate, sys_rate, channels=1):
    audio_queue = queue.Queue()
    stop_flag = threading.Event()
    mic_active = threading.Event()

    def on_pressed(key):
        if key == keyboard.Key.space:
            mic_active.set()

    def on_released(key):
        if key == keyboard.Key.space:
            mic_active.clear()
        if key == keyboard.Key.esc:
            stop_flag.set()
            return False
        return True

    listener = keyboard.Listener(on_press=on_pressed, on_release=on_released)
    listener.start()

    def mic_thread():
        def mic_callback(indata, frames, t_info, status):
            if status:
                print("mic status:", status)
            if mic_active.is_set():
                audio_queue.put(
                    AudioEvent(
                        timestamp=t_info.inputBufferAdcTime,
                        source="mic",
                        chunk=indata.copy().flatten(),
                        sample_rate=mic_rate,
                    )
                )

        with sd.InputStream(
            device=mic_device,
            channels=channels,
            samplerate=mic_rate,
            callback=mic_callback,
        ):
            while not stop_flag.is_set():
                sd.sleep(50)

    def sys_thread():
        def sys_callback(indata, frames, t_info, status):
            if status:
                print("system status:", status)
            # System audio stays on so overlapping playback is preserved while you speak.
            audio_queue.put(
                AudioEvent(
                    timestamp=t_info.inputBufferAdcTime,
                    source="system",
                    chunk=indata.copy().flatten(),
                    sample_rate=sys_rate,
                )
            )

        with sd.InputStream(
            device=sys_device,
            channels=channels,
            samplerate=sys_rate,
            callback=sys_callback,
        ):
            while not stop_flag.is_set():
                sd.sleep(50)

    threads = [
        threading.Thread(target=mic_thread, daemon=True),
        threading.Thread(target=sys_thread, daemon=True),
    ]
    for thread in threads:
        thread.start()

    print("Recording started. Hold SPACE for mic input, ESC to quit.")
    try:
        while not stop_flag.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_flag.set()
    finally:
        mic_active.clear()
        listener.stop()

    for thread in threads:
        thread.join()

    return audio_queue


def denoise_audio(audio, rate):
    if audio.size == 0:
        return audio

    noise_window = min(len(audio), int(rate * 0.5))
    if noise_window == 0:
        return audio

    noise_sample = audio[:noise_window]
    return nr.reduce_noise(y=audio, sr=rate, y_noise=noise_sample)


def process_audio_queue(audio_queue, target_rate=TARGET_RATE, denoise=True):
    all_events = []
    while not audio_queue.empty():
        all_events.append(audio_queue.get())

    if not all_events:
        return None

    all_events.sort(key=lambda event: event.timestamp)
    t0 = all_events[0].timestamp
    tracks = []

    for event in all_events:
        chunk = event.chunk.astype(np.float32)
        if event.sample_rate != target_rate:
            chunk = librosa.resample(
                chunk, orig_sr=event.sample_rate, target_sr=target_rate
            )
        # Convert each chunk timestamp into a sample offset on the final shared timeline.
        start_sample = max(0, int((event.timestamp - t0) * target_rate))
        tracks.append((start_sample, chunk))

    total_len = max(start + len(chunk) for start, chunk in tracks)
    mixed_audio = np.zeros(total_len, dtype=np.float32)

    for start, chunk in tracks:
        end = start + len(chunk)
        mixed_audio[start:end] += chunk

    if denoise:
        # Denoise after mixing so the noise reduction is applied to the actual output audio.
        mixed_audio = denoise_audio(mixed_audio, target_rate).astype(np.float32)

    peak = np.max(np.abs(mixed_audio))
    if peak > 0:
        mixed_audio /= peak

    sf.write(OUTPUT_AUDIO_FILE, mixed_audio, target_rate)
    return mixed_audio


def build_asr_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return hf_pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        device=device,
    )


def transcribe_file(filepath, whisper_pipe):
    print(f"Transcribing {filepath}...")
    return whisper_pipe(filepath, return_timestamps="sentence")


def diarize_audio(file_path, token):
    diarization = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=token,
    )
    return diarization(file_path)


def align_speakers(audio_chunks, diarization_result):
    final_transcript = []
    for segment in audio_chunks.get("chunks", []):
        seg_start, seg_end = segment.get("timestamp", (None, None))
        text = segment.get("text", "").strip()
        if not text or seg_start is None or seg_end is None:
            continue

        speaker = "UNKNOWN"
        if diarization_result is not None:
            seg_interval = Segment(seg_start, seg_end)
            overlapping = diarization_result.crop(seg_interval)
            if len(overlapping) > 0:
                # Pick the speaker with the longest overlap for each Whisper sentence window.
                speaker = max(
                    overlapping.itertracks(yield_label=True),
                    key=lambda item: item[0].duration,
                )[2]

        final_transcript.append((speaker, text))
    return final_transcript


def remap_speakers(final_transcript):
    speaker_map = {}
    next_label = ord("A")
    remapped = []

    for speaker, text in final_transcript:
        if speaker not in speaker_map and speaker != "UNKNOWN":
            speaker_map[speaker] = chr(next_label)
            next_label += 1
        remapped.append((speaker_map.get(speaker, "UNKNOWN"), text))

    return remapped


def extract_action_items(transcript_text):
    sentences = re.split(r"[.!?]\s+", transcript_text)
    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
        and any(verb in sentence.lower() for verb in ACTION_VERBS)
    ]


def split_text_for_summary(transcript_text, max_words=220):
    sentences = re.split(r"(?<=[.!?])\s+", transcript_text.strip())
    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_words = len(sentence.split())
        if current_chunk and current_words + sentence_words > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_words = sentence_words
        else:
            current_chunk.append(sentence)
            current_words += sentence_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks or [transcript_text]


def summarize_transcript(
    transcript_text,
    summarizer,
    chunk_word_limit=220,
    max_length=80,
    min_length=20,
):
    if not transcript_text.strip():
        return ""

    # Summarize in passes so longer meetings do not overflow the model input window.
    chunks = split_text_for_summary(transcript_text, max_words=chunk_word_limit)
    chunk_summaries = []

    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )[0]["summary_text"]
        chunk_summaries.append(summary.strip())

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined_summary_input = " ".join(chunk_summaries)
    if len(combined_summary_input.split()) <= chunk_word_limit:
        return summarizer(
            combined_summary_input,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )[0]["summary_text"]

    return " ".join(chunk_summaries)


def format_transcript(transcript_rows):
    return "\n".join(f"Speaker {speaker}: {text}" for speaker, text in transcript_rows)


def main():
    load_dotenv()

    mic_device, mic_rate, mic_name = resolve_input_device(
        "MEETING_MIC_DEVICE", DEFAULT_MIC_DEVICE
    )
    sys_device, sys_rate, sys_name = resolve_input_device(
        "MEETING_SYS_DEVICE", DEFAULT_SYS_DEVICE
    )

    print(f"Using mic device {mic_device}: {mic_name}")
    print(f"Using system device {sys_device}: {sys_name}")

    audio_queue = record_audio(mic_device, sys_device, mic_rate, sys_rate)
    mixed_audio = process_audio_queue(audio_queue)

    if mixed_audio is None or mixed_audio.size == 0:
        print("No audio recorded.")
        return

    whisper_pipe = build_asr_pipeline()
    audio_result = transcribe_file(OUTPUT_AUDIO_FILE, whisper_pipe)

    token = os.getenv("HF_TOKEN")
    diarization_result = None
    if token:
        diarization_result = diarize_audio(OUTPUT_AUDIO_FILE, token)
    else:
        print("HF_TOKEN not found. Skipping speaker diarization.")

    final_transcript = align_speakers(audio_result, diarization_result)
    final_transcript = remap_speakers(final_transcript)

    formatted_transcript = format_transcript(final_transcript)
    print("Transcript with speakers:")
    print(formatted_transcript)

    full_transcript = " ".join(text for _, text in final_transcript)
    action_items = extract_action_items(full_transcript)
    print("Action items:", action_items)

    summarizer = hf_pipeline(
        "summarization",
        model=SUMMARY_MODEL,
        device=0 if torch.cuda.is_available() else -1,
    )
    summary_text = summarize_transcript(full_transcript, summarizer)
    print("Summary:", summary_text)


if __name__ == "__main__":
    main()
