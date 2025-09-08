#Install Torch CPU and ffmpeg
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import sounddevice as sd
from pynput import keyboard
import time 
from pyannote.audio import Pipeline as DiarizationPipeline
import os
import soundfile as sf

# --- Device Setup ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
token = os.getenv("HF_TOKEN")

# --- Whisper Setup ---
model_id = "openai/whisper-tiny"
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)

speech_model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

whisper_pipe = hf_pipeline(
    "automatic-speech-recognition",
    model=speech_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# --- Summarize text-generation pipeline ---
from transformers import pipeline
 
summarizer = pipeline("summarization", model="t5-small")
# --- Audio capture parameters ---
RATE = 16000       # Whisper expects 16 kHz audio
CHANNELS = 1
device_id = [2,6] #input + output audio device id's
full_transcript = []
space_held = False
recorded_frames = []
sys_frames = []

def on_pressed(key):
    global space_held
    if key == keyboard.Key.space:
        if not space_held:
            print("space pressed -> start recording")
        space_held = True

def on_released(key):
    global space_held
    if key == keyboard.Key.space:
            print("space released -> stop recording")
            space_held = False
    if key == keyboard.Key.esc:
        return False


def callback(indata, frames, time, status):
    if status:
        print("audio status:",status)
    if space_held:
        recorded_frames.append((time.time(),indata.copy()))

def sys_callback(indata, frames, time, status):
    if status:
        print("audio status:",status)
    sys_frames.append((time.time(),indata.copy()))

# --- start keyboard listner --- 
listener = keyboard.Listener(on_press=on_pressed, on_release=on_released)
listener.start()


# --- Start audio capture from microphone + system output ---

with sd.InputStream(channels=CHANNELS, samplerate=RATE, callback=callback), \
     sd.InputStream(channels=CHANNELS, samplerate=RATE, callback=sys_callback):

    print("Hold SPACE to record mic, ESC to quit. System audio recording always on.")
    try:
        while listener.running:
            sd.sleep(100)
    except KeyboardInterrupt:
        print("Exiting...")

# --- concatenate in correct chronological order ---
all_frames = recorded_frames + sys_frames
all_frames.sort(key = lambda x:x[0]) #all frames sorted by timestamp
merged_audio = np.concatenate([f[1] for f in all_frames], axis=0).astype(np.float32).flatten()
sf.write("merged_audio.wav", merged_audio, RATE)

result = whisper_pipe(merged_audio, chunk_length_s=None)
full_transcrupt = result["text"]

diarization = DiarizationPipeline.from_pretrained(
    "pyannote/speaker-diarization", 
    use_auth_token=token
)

#--- Run Diarization ---
diarization_result = diarization("merged_audio.wav")
segments = []

##### --------------- TODO: Use whisper model with timestamp("speaker A said ... at {time}")
for turn, _, speaker in diarization_result.itertracks(yield_label=True):
    # Approximate splitting: divide transcript text proportionally
    # Here we just use the full transcript for every segment
    segments.append(f"Speaker {speaker} ({turn.start:.1f}-{turn.end:.1f}s): {full_transcript}")

# --- print segments ---
for seg in segments:
    print(seg)

# --- Generate summary output after recording audio ---
final_transcript = " ".join(full_transcript)

prompt = f"Summarize this text into two sentences\n {final_transcript}"

#response = llama_generate(final_transcript)
summary = summarizer(final_transcript, max_length=50, min_length=15, do_sample=False)
print("Summarized Text: ", summary[0]['summary_text'])
