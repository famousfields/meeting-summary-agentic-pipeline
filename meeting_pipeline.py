import queue, time, numpy as np, sounddevice as sd, librosa, torch, os
from pynput import keyboard#gets keyboard inputs

# --- Audio Recording ---
def record_audio(mic_device, sys_device, mic_rate, sys_rate, channels=1):
    audio_queue = queue.Queue()
    space_held = False

    def on_pressed(key):
        nonlocal space_held
        if key == keyboard.Key.space:
            space_held = True

    def on_released(key):
        nonlocal space_held
        if key == keyboard.Key.space:
            space_held = False
        if key == keyboard.Key.esc:
            return False

    def mic_callback(indata, frames, t_info, status):
        if status:
            print("mic status:", status)
        if space_held:
            frame = indata.copy().flatten()
            frame_16k = librosa.resample(frame, orig_sr=mic_rate, target_sr=16000)
            audio_queue.put((time.time(), "mic", frame_16k))

    def sys_callback(indata, frames, t_info, status):
        if status:
            print("sys status:", status)
        frame = indata.copy().flatten()
        frame_16k = librosa.resample(frame, orig_sr=sys_rate, target_sr=16000)
        audio_queue.put((time.time(), "sys", frame_16k))

    listener = keyboard.Listener(on_press=on_pressed, on_release=on_released)
    listener.start()

    with sd.InputStream(channels=channels, samplerate=mic_rate, callback=mic_callback), \
         sd.InputStream(channels=channels, samplerate=sys_rate, callback=sys_callback):
        print("Hold SPACE to record mic, ESC to quit. System audio recording always on.")
        try:
            while listener.running:
                sd.sleep(100)
        except KeyboardInterrupt:
            print("Exiting...")

    # Retrieve and sort audio chunks
    all_events = []
    while not audio_queue.empty():
        all_events.append(audio_queue.get())
    all_events.sort(key=lambda x: x[0])
    ordered_audio = [chunk for _, _, chunk in all_events]
    merged_audio = np.concatenate(ordered_audio, axis=0)
    return merged_audio

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
    mic_device, sys_device = 2, 4
    mic_rate = int(sd.query_devices(mic_device)['default_samplerate'])
    sys_rate = int(sd.query_devices(sys_device)['default_samplerate'])

    # --- Record audio ---
    merged_audio = record_audio(mic_device, sys_device, mic_rate, sys_rate)

    # --- Save for reference ---
    import soundfile as sf
    RATE = 16000
    sf.write("merged_audio.wav", merged_audio, RATE)

    # --- Load Whisper ---
    whisper_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # --- Transcription ---
    audio_result = transcribe_audio(merged_audio, whisper_pipe)

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
