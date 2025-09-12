#Install Torch CPU and ffmpeg
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline, AutoTokenizer, AutoModelForCausalLM#imports models from huggingface
import numpy as np
import sounddevice as sd#allows access to microphone and does something with recorded audio
from pynput import keyboard#gets keyboard inputs
import time 
from pyannote.audio import Pipeline as DiarizationPipeline#package to segment speakers from audio file
import os
from dotenv import load_dotenv#hides authentication tokens
import soundfile as sf#create and store sound file from 1D float32 audio array
import librosa #package to resample audio files to a specific sample rate

# --- Device Setup ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
load_dotenv() 
token = os.getenv("HF_TOKEN")

# --- Whisper Setup ---
model_id = "openai/whisper-tiny"
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)

for i, dev in enumerate(sd.query_devices()):
    print(i, dev['name'], "IN:", dev['max_input_channels'], "OUT:", dev['max_output_channels'])

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

# --- Summarized text-generation pipeline ---
from transformers import pipeline
 
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

#--mic audio only
all_frames = []
mic_info = sd.query_devices(2)  # your microphone
mic_rate = int(mic_info['default_samplerate'])

sys_info = sd.query_devices(4)  # your system audio device
sys_rate = int(sys_info['default_samplerate'])

def callback(indata, frames, t_info, status):
    if status:
        print("audio status:",status)
    if space_held:
        frame = indata.copy().flatten()
        frame_16k = librosa.resample(frame, orig_sr=mic_rate, target_sr=16000)
        all_frames.append(frame_16k)

#-- output audio only
def sys_callback(indata, frames, t_info, status):
    if status:
        print("audio status:",status)
    if not space_held:
        frame = indata.copy().flatten()
        frame_16k = librosa.resample(frame, orig_sr=sys_rate, target_sr=16000)
        all_frames.append(frame_16k)

# --- start keyboard listner --- 
listener = keyboard.Listener(on_press=on_pressed, on_release=on_released)
listener.start()


# --- Start audio capture from microphone + system output ---


with sd.InputStream(device = 2,channels=CHANNELS,samplerate=mic_rate, callback=callback), \
     sd.InputStream(device = 4,channels=CHANNELS, samplerate =sys_rate, callback=sys_callback):

    print("Hold SPACE to record mic, ESC to quit. System audio recording always on.")
    try:
        while listener.running:
            sd.sleep(100)
    except KeyboardInterrupt:
        print("Exiting...")

# --- concatenate in correct chronological order ---

#TODO: find a way to ensure callback only captures microphone audio and sys_callback only captures system output audio
#currently: sys_callback captures both output and input audio
#next: 1) store audio in 'merged_audio' variable in the order that it is recieved
#      2) pass 'merged_audio' to diarization pipeline to segment speakers
# --- Resample audio streams separately ---
all_chunks = []  # to store all sentence-level chunks
all_chunks2 = []
all_audio = np.concatenate(all_frames, axis=0)
audio_result =  mic_result = whisper_pipe(all_audio, chunk_length_s=None, return_timestamps="sentence")
all_chunks.extend(audio_result.get("chunks", []))
print("Recording RESULT:", audio_result)


# # Optionally merge audio for saving
# merged_audio = np.concatenate(
#     [mic_audio_16k] + ([sys_audio_16k] if sys_frames else [])
# )


sf.write("merged_audio.wav", all_audio, RATE)

# diarization = DiarizationPipeline.from_pretrained(
#     "pyannote/speaker-diarization", 
#     use_auth_token=token
# )

# #--- Run Diarization ---
# diarization_result = diarization("merged_audio.wav")

# Full transcript from separate chunks
full_transcript = " ".join([c["text"] for c in all_chunks])
# --- Align Whisper sentences with diarization speakers ---
speaker_sentences = []


# --- Generate summary output after recording audio ---
# print("final transcript:",full_transcript)

#response = llama_generate(final_transcript)
summarizer = pipeline("summarization", model="t5-small")
summary = summarizer(full_transcript, max_length=50, min_length=15, do_sample=False)
print("Summarized Text: ", summary[0]['summary_text'])