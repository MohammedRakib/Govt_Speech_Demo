import gradio as gr
import pytube as pt
import subprocess
from models import asr, processor
import time
import librosa
## details: https://huggingface.co/docs/diffusers/optimization/fp16#automatic-mixed-precision-amp
# from torch import autocast 

# task = "transcribe"  # transcribe or translate
# language = "bn"
# asr.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
# asr.model.config.max_new_tokens = 448 #default is 448

def _preprocess(filename):
     audio_name = "audio.wav"
     subprocess.call(['ffmpeg', '-y', '-i', filename, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', "-loglevel", "quiet", audio_name])
     return audio_name
 
def transcribe(microphone, file_upload):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )

    elif (microphone is None) and (file_upload is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

    file = microphone if microphone is not None else file_upload
    print(f"\n\nFile is: {file}\n\n")
    
    # for _preprocess(). No need if name of file provided in string format to asr pipeline as automatically uses ffmeg. 
    # Only required if ndarray given by using librosa.load() to load a file
    start_time = time.time()
    print("Starting Preprocessing")
    speech_array = _preprocess(filename=file)
    # filename = _preprocess(filename=file)
    # speech_array, sample_rate = librosa.load(f"{filename}", sr=16_000)
    print(f"\n Preprocessing COMPLETED in {round(time.time()-start_time, 2)}s \n")
    
    
    start_time = time.time()
    print("Starting Inference")
    text = asr(speech_array)["text"]
    # with autocast("cuda"):
    #     text = asr(speech_array)["text"]
    print(f"\n Inference COMPLETED in {round(time.time()-start_time, 2)}s \n")

    return warn_output + text

def _return_yt_html_embed(yt_url):
    if "?v=" in yt_url:
        video_id = yt_url.split("?v=")[-1].split('&')[0]
    else:
        video_id = yt_url.split("/")[-1].split('?feature=')[0]
        
    print(f"\n\nYT ID is: {video_id}\n\n")   
    return f'<center><iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe> </center>'


def yt_transcribe(yt_url):
    start_time = time.time()
    yt = pt.YouTube(yt_url)
    html_embed_str = _return_yt_html_embed(yt_url)
    stream = yt.streams.filter(only_audio=True)[0]
    filename = "audio.mp3"
    stream.download(filename=filename)
    print(f"\n YT Audio Downloaded in {round(time.time()-start_time, 2)}s \n")
    
    
    # for _preprocess(). No need if name of file provided in string format to asr pipeline as automatically uses ffmeg. 
    # Only required if ndarray given by using librosa.load() to load a file
    start_time = time.time()
    print("Starting Preprocessing")
    speech_array = _preprocess(filename=filename)
    # filename = _preprocess(filename=filename)
    # speech_array, sample_rate = librosa.load(f"{filename}", sr=16_000)
    print(f"\n Preprocessing COMPLETED in {round(time.time()-start_time, 2)}s \n")
    
    start_time = time.time()
    print("Starting Inference")
    text = asr(speech_array)["text"]
    # with autocast("cuda"):
    #     text = asr(speech_array)["text"]
    print(f"\n Inference COMPLETED in {round(time.time()-start_time, 2)}s \n")
    
    return html_embed_str, text


mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(source="microphone", type="filepath", label="Microphone"),
        gr.Audio(source="upload", type="filepath", label="Upload File"),
    ],
    outputs="text",
    title="Whisper Bangla Demo: Transcribe Audio",
    description=(
        "Transcribe long-form microphone or audio inputs in BANGLA with the click of a button!"
    ),
    allow_flagging="never",
)

yt_transcribe = gr.Interface(
    fn=yt_transcribe,
    inputs=[gr.Textbox(lines=1, placeholder="Paste the URL to a Bangla language YouTube video here", label="YouTube URL")],
    outputs=["html", "text"],
    title="Whisper Bangla Demo: Transcribe YouTube",
    description=(
        "Transcribe long-form YouTube videos in BANGLA with the click of a button!"
    ),
    allow_flagging="never",
)

demo = gr.TabbedInterface([mf_transcribe, yt_transcribe], ["Transcribe Bangla Audio", "Transcribe Bangla YouTube Video"])


if __name__ == "__main__":
    
    demo.queue()
    demo.launch(share='True')
