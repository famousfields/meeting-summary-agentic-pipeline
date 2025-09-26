import queue, time, numpy as np, sounddevice as sd, librosa, torch, os, threading
from pynput import keyboard
import soundfile as sf
from transformers import pipeline as hf_pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.core import Segment

# --- Audio Recording ---
def record_audio(mic_device, sys_device, mic_rate, sys_rate, channels=1):
    audio_queue = queue.Queue()
    space_held = False
    stop_flag = threading.Event()

    # --- Keyboard control ---
    def on_pressed(key):
        nonlocal space_held
        if key == keyboard.Key.space:
            space_held = True

    def on_released(key):
        nonlocal space_held
        if key == keyboard.Key.space:
            space_held = False
        if key == keyboard.Key.esc:
            stop_flag.set()

    listener = keyboard.Listener(on_press=on_pressed, on_release=on_released)
    listener.start()

    # --- Mic thread ---
    def mic_thread():
        def mic_callback(indata, frames, t_info, status):
            if status:
                print("mic status:", status)
            if space_held:
                ts = t_info.inputBufferAdcTime
                audio_queue.put((ts, "mic", indata.copy().flatten(), mic_rate))
        with sd.InputStream(device=mic_device, channels=channels, samplerate=mic_rate, callback=mic_callback):
            while not stop_flag.is_set():
                sd.sleep(50)

    # --- System thread ---
    def sys_thread():
        def sys_callback(indata, frames, t_info, status):
            if status:
                print("sys status:", status)
            if not space_held:
                ts = t_info.inputBufferAdcTime
                audio_queue.put((ts, "sys", indata.copy().flatten(), sys_rate))
        with sd.InputStream(device=sys_device, channels=channels, samplerate=sys_rate, callback=sys_callback):
            while not stop_flag.is_set():
                sd.sleep(50)

    t1 = threading.Thread(target=mic_thread, daemon=True)
    t2 = threading.Thread(target=sys_thread, daemon=True)
    t1.start(); t2.start()

    print("Recording started. Hold SPACE for mic input, ESC to quit.")
    try:
        while not stop_flag.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_flag.set()
    t1.join(); t2.join(); listener.stop()
    return audio_queue

#--- Remove Noise from audio --- 
import noisereduce as nr
def denoise_chunk(chunk, rate):
    # Estimate noise from the first 0.5s or a quiet section
    noise_sample = chunk[:int(rate * 0.5)]
    reduced = nr.reduce_noise(y=chunk, sr=rate, y_noise=noise_sample)
    return reduced

# --- Save separate tracks + optional merged ---
def process_audio_queue(audio_queue, target_rate=16000):
    all_events = []
    while not audio_queue.empty():
        all_events.append(audio_queue.get())

    if not all_events:
        return None

    # Sort by timestamp to maintain original order
    all_events.sort(key=lambda x: x[0])

    # Determine total length
    t0 = all_events[0][0]
    tracks = []
    for ts, src, chunk, orig_rate in all_events:
        if orig_rate != target_rate:
            chunk = librosa.resample(chunk, orig_sr=orig_rate, target_sr=target_rate)
        start_sample = int((ts - t0) * target_rate)
        tracks.append((start_sample, chunk))

    chunk = denoise_chunk(chunk,target_rate)
    
    total_len = max(start + len(chunk) for start, chunk in tracks)
    mixed_audio = np.zeros(total_len, dtype=np.float32)

    for start, chunk in tracks:
        end = start + len(chunk)
        mixed_audio[start:end] += chunk.astype(np.float32)

    # Normalize to avoid clipping
    mixed_audio /= (np.max(np.abs(mixed_audio)) + 1e-9)
    sf.write("merged_audio.wav", mixed_audio, target_rate)

    return mixed_audio

# --- Transcription ---
def transcribe_file(filepath, whisper_pipe):
    print(f"Transcribing {filepath}...")
    return whisper_pipe(filepath, return_timestamps="sentence")


# --- Diarization ---
def diarize_audio(file_path, token):
    diarization = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=token
    )
    return diarization(file_path)

# --- Align speakers ---
def align_speakers(audio_chunks, diarization_result):
    final_transcript = []
    for seg in audio_chunks["chunks"]:
        seg_start, seg_end = seg.get("timestamp", (None, None))
        text = seg["text"]

        # Skip invalid segments
        if seg_start is None or seg_end is None:
            continue  

        seg_interval = Segment(seg_start, seg_end)
        overlapping = diarization_result.crop(seg_interval)

        speaker = "UNKNOWN"
        if len(overlapping) > 0:
            speaker = max(overlapping.itertracks(yield_label=True), 
                          key=lambda x: x[0].duration)[1]
        final_transcript.append((speaker, text))
    return final_transcript

# --- Remap Speakers if Needed --- 
def remap_speakers(final_transcript):
    speaker_map = {}
    next_label = ord('A')  # Start with 'A'
    remapped = []

    for speaker, text in final_transcript:
        if speaker not in speaker_map and speaker != "UNKNOWN":
            speaker_map[speaker] = chr(next_label)
            next_label += 1
        remapped.append((speaker_map.get(speaker, "UNKNOWN"), text))
    
    return remapped

import re 
action_verbs = [
    "need", "decide", "assign", "do", "complete", "implement", 
    "follow up", "prepare", "approve", "confirm", "should", "must"
]
def extract_action_items(transcript_text):
    sentences = re.split(r'[.!?]\s+', transcript_text)
    action_items = []
    for sentence in sentences:
        if any(verb in sentence.lower() for verb in action_verbs):
            action_items.append(sentence.strip())
    return action_items

# --- Summarize ---
from transformers import pipeline
def summarize_transcript(transcript_text, model_name="t5-small", max_length=50, min_length=15):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(transcript_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


def main():
    # --- Devices ---
    mic_device, sys_device = 2, 4
    mic_rate = int(sd.query_devices(mic_device)['default_samplerate'])
    sys_rate = int(sd.query_devices(sys_device)['default_samplerate'])

    # --- Record audio ---
    audio_queue = record_audio(mic_device, sys_device, mic_rate, sys_rate)

    # --- Process audio queue ---
    mixed_audio = process_audio_queue(audio_queue)

    # --- Save merged_audio.wav if not already done ---
    if mixed_audio is not None and mixed_audio.size > 0:
        merged_file = "merged_audio.wav"
    else:
        print("No audio recorded.")
        return

    # --- Load Whisper ---
    whisper_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # --- Transcription ---
    audio_result = transcribe_file(merged_file, whisper_pipe)
    print("Audio transcription results:", audio_result)

    # --- Diarization ---
    token = os.getenv("HF_TOKEN")
    diarization_result = diarize_audio(merged_file, token)

    # --- Align speakers ---
    final_transcript = align_speakers(audio_result, diarization_result)
    for speaker, text in final_transcript:
        print(f"Speaker {speaker}: {text}")
    final_transcript = remap_speakers(final_transcript)

    # --- Full transcript ---
    full_transcript = " ".join(text for _, text in final_transcript)
    print("Transcript:", full_transcript)

    full_meaningful_transcript = extract_action_items(full_transcript)
    print(full_meaningful_transcript)

    # --- Summarize ---
    summary_text = summarize_transcript(full_transcript)
    print("Summary:", summary_text)

if __name__ == "__main__":
    main()
