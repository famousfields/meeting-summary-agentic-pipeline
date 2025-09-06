#Install Torch CPU and ffmpeg
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import sounddevice as sd
from pynput import keyboard

# --- Device Setup ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# --- Whisper Setup ---
model_id = "openai/whisper-tiny"
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
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
full_text = []
space_held = False
recorded_frames = []

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
        recorded_frames.append(indata.copy())

# --- start keyboard listner --- 
listener = keyboard.Listener(on_press=on_pressed, on_release=on_released)
listener.start()


# --- Start streaming from microphone ---
#TODO: 0) start infinite loop
#      1) Check button press
#      2) Stream audio until release

with sd.InputStream(channels=CHANNELS, samplerate=RATE, callback=callback):
    print("Hold SPACE to record, ESC to quit")
    try:
        while listener.running:
            sd.sleep(100)
    except KeyboardInterrupt:
        print("Exiting...")

# --- Concatenate all frames into a single numpy array ---
if recorded_frames:
    audio_data = np.concatenate(recorded_frames, axis=0).astype(np.float32).flatten()
    print("Audio recorded, length:", audio_data.shape)

    # --- Transcribe entire recording at once ---
    result = whisper_pipe(audio_data, chunk_length_s=None)  # process full audio
    print("Full transcription:", result["text"])
    full_text.append(result["text"])
else:
    print("No audio recorded")

# --- Generate LLaMA output after speech ---
final_transcript = " ".join(full_text)

prompt = f"Summarize this text into two sentences\n {final_transcript}"

#response = llama_generate(final_transcript)
summary = summarizer(final_transcript, max_length=50, min_length=15, do_sample=False)
print("Summarized Text: ", summary[0]['summary_text'])
