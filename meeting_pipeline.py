import queue, time, numpy as np, sounddevice as sd, librosa, torch, os, threading
from pynput import keyboard#gets keyboard inputs

# --- Audio Recording ---

for i, dev in enumerate(sd.query_devices()):
    print(i, dev['name'], "IN:", dev['max_input_channels'], "OUT:", dev['max_output_channels'])
def record_audio(mic_device, sys_device, mic_rate, sys_rate, channels=1):
    audio_queue = queue.Queue()
    space_held = False
    stop_flag = threading.Event()  # to stop both threads gracefully

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
            stop_flag.set()  # stop threads
            

    listener = keyboard.Listener(on_press=on_pressed, on_release=on_released)
    listener.start()

    # --- Mic capture thread ---
    def mic_thread():
        def mic_callback(indata, frames, t_info, status):
            if status:
                print("mic status:", status)
            if space_held:
                frame = indata.copy().flatten()
                audio_queue.put((time.time(), "mic", frame, mic_rate))
        print("mic rate:", mic_rate)
        with sd.InputStream(device = mic_device,
                            channels=channels,
                            samplerate=mic_rate,
                            callback=mic_callback):
            while not stop_flag.is_set():
                sd.sleep(50)

    # --- System audio capture thread ---
    def sys_thread():
        def sys_callback(indata, frames, t_info, status):
            if status:
                print("sys status:", status)
            frame = indata.copy().flatten()
            audio_queue.put((time.time(), "sys", frame, sys_rate))

        with sd.InputStream(device = sys_device,
                            channels=channels,
                            samplerate=sys_rate,
                            callback=sys_callback):
            while not stop_flag.is_set():
                sd.sleep(50)

    # --- Start both threads ---
    t1 = threading.Thread(target=mic_thread, daemon=True)
    t2 = threading.Thread(target=sys_thread, daemon=True)
    t1.start()
    t2.start()

    print("Recording started. Hold SPACE for mic input, ESC to quit.")
    try:
        while not stop_flag.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
        stop_flag.set()

    t1.join()
    t2.join()
    listener.stop()

    return audio_queue

import soundfile as sf
def merge_audio_queue(audio_queue, target_rate, outfile=None):
    all_events = []
    while not audio_queue.empty():
        all_events.append(audio_queue.get())

    # Sort by timestamp
    all_events.sort(key=lambda x: x[0])  # (timestamp, source, chunk, orig_rate)

    if not all_events:
        return np.array([]), []

    # Normalize timestamps to start at 0
    t0 = all_events[0][0]

    # Convert timestamps → sample indices
    tracks = []
    for ts, src, chunk, orig_rate in all_events:
        # Resample if needed
        if orig_rate != target_rate:
            chunk = librosa.resample(chunk, orig_sr=orig_rate, target_sr=target_rate)

        start_sample = int((ts - t0) * target_rate)
        tracks.append((start_sample, chunk))

    # Figure out how long the final track must be
    total_len = max(start + len(chunk) for start, chunk in tracks)

    # Create empty buffer
    mixed_audio = np.zeros(total_len, dtype=np.float32)

    # Add all sources into the buffer (mixing)
    for start, chunk in tracks:
        end = start + len(chunk)
        mixed_audio[start:end] += chunk.astype(np.float32)

    # Normalize to avoid clipping
    mixed_audio = mixed_audio / np.max(np.abs(mixed_audio) + 1e-9)

    if outfile:
        sf.write(outfile, mixed_audio, target_rate)

    return mixed_audio, all_events

# --- Transcription ---
from transformers import pipeline as hf_pipeline
def transcribe_audio(audio, whisper_pipe):
    result = whisper_pipe(audio, chunk_length_s=None, return_timestamps="sentence")
    return result  # contains 'chunks' with timestamps and text

# --- Diarization ---
from pyannote.audio import Pipeline as DiarizationPipeline
def diarize_audio(file_path, token):
    diarization = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=token
    )
    return diarization(file_path)

# --- Align speakers with diarization --- 
from pyannote.core import Segment
def align_speakers(audio_chunks, diarization_result):
    final_transcript = []
    for seg in audio_chunks["chunks"]:
        seg_start, seg_end = seg["timestamp"]
        text = seg["text"]
        seg_interval = Segment(seg_start, seg_end)
        overlapping = diarization_result.crop(seg_interval)
        if len(overlapping) > 0:
            speaker = max(overlapping.itertracks(yield_label=True),
                          key=lambda x: x[0].duration)[1]
        else:
            speaker = "UNKNOWN"
        final_transcript.append((speaker, text))
    return final_transcript

# --- Summarize ---
from transformers import pipeline
def summarize_transcript(transcript_text, model_name="t5-small", max_length=50, min_length=15):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(transcript_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def main():
    # --- Devices ---
    mic_device, sys_device = 2,4
    mic_info = sd.query_devices(2)  # your microphone
    mic_rate = int(mic_info['default_samplerate'])
    sys_rate = int(sd.query_devices(sys_device)['default_samplerate'])

    # --- Record audio ---
    audio_queue = record_audio(mic_device, sys_device, mic_rate, sys_rate)

    # --- Merge Audio --- 
    # all_events- lets us know which chunks came from the mic and which from system output.
    # We could, for example:
    #     Save two separate WAVs (mic.wav and sys.wav).
    #     Run diarization on one stream but not the other.
    #     Analyze talk time per source (how much the mic spoke vs. system audio).
    merged_audio, all_events = merge_audio_queue(audio_queue, target_rate=16000, outfile="merged_audio.wav")
    
    # --- Load Whisper ---
    whisper_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # --- Transcription ---
    audio_result = transcribe_audio(merged_audio, whisper_pipe)
    print("audio results" , audio_result)

    # --- Diarization ---
    token = os.getenv("HF_TOKEN")
    diarization_result = diarize_audio("merged_audio.wav", token)

    # --- Align Speakers ---
    final_transcript = align_speakers(audio_result, diarization_result)
    for speaker, text in final_transcript:
        print(f"Speaker {speaker}: {text}")

    # --- Full transcript ---
    full_transcript = " ".join(text for _, text in final_transcript)
    print("Transcript:", full_transcript)

    # --- Summarize ---
    summary_text = summarize_transcript(full_transcript)
    print("Summary:", summary_text)

if __name__ == "__main__":
    main()
