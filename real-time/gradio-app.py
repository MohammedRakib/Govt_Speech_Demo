import gradio as gr
import pytube as pt
import subprocess
import time
from models import asr, processor

task = "transcribe"  # transcribe or translate
# language = "bn"
language = "en"
asr.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
# asr.model.config.max_new_tokens = 448 #default is 448

# def _preprocess(filename):
#      audio_name = "audio.wav"
#      subprocess.call(['ffmpeg', '-y', '-i', filename, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_name])
#      return audio_name

def transcribe(microphone, state="", delay=1.2):
    time.sleep(delay-1)

    print(f"\n\nFile is: {microphone}\n\n")

    text = asr(microphone)["text"]
    state += f"{text} "
    
    print(f"Transcription Done for {microphone}!!!")

    return state, state

delay_slider = gr.Slider(minimum=1, maximum=5, default=1.2, label="Rate of transcription")

demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(source="microphone", type="filepath", label="Microphone", streaming=True),
        "state",
        delay_slider,
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
