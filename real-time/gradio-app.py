import gradio as gr
import pytube as pt
import subprocess
import librosa
import time
from models import asr, processor

# task = "transcribe"  # transcribe or translate
# language = "bn"
# asr.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
# asr.model.config.max_new_tokens = 448 #default is 448

def _preprocess(filename):
     audio_name = "audio.wav"
     subprocess.call(['ffmpeg', '-y', '-i', filename, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_name])
     return audio_name

def transcribe(microphone, state=""):
    # time.sleep(2)

    print(f"\n\nFile is: {microphone}\n\n")
    
    # for _preprocess(). No need if name of file provided in string format to asr pipeline as automatically uses ffmeg. 
    # Only required if ndarray given by using librosa.load() to load a file
    start_time = time.time()
    print("Starting Preprocessing")
    # speech_array = _preprocess(filename=microphone)
    filename = _preprocess(filename=microphone)
    speech_array, sample_rate = librosa.load(f"{filename}", sr=16_000)
    print(f"\n Preprocessing COMPLETED in {round(time.time()-start_time, 2)}s \n")
    
    print(f"Starting Inference for {microphone}")
    start_time = time.time()
    text = asr(microphone)["text"]
    state += f"{text} "
    print(f"\n Inference COMPLETED in {round(time.time()-start_time, 2)}s \n")
    
    return state, state

demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(source="microphone", type="filepath", label="Microphone", streaming=True),
        "state",
    ],
    outputs=[
        "text",
        "state",
    ],
    title="Whisper Bangla REAL-TIME Demo: Transcribe Audio",
    description=(
        "Transcribe BANGLA audio input in REAL-TIME with the click of a button!"
    ),
    allow_flagging="never",
    live=True,
)


if __name__ == "__main__":
    
    demo.queue()
    demo.launch(share='True')
