import whisper
import datetime
import subprocess
import gradio as gr
from pathlib import Path
import pandas as pd
import re
import time
import os 
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from pytube import YouTube
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from scipy.io import wavfile
from gpuinfo import GPUInfo

import wave
import contextlib
from transformers import pipeline
import psutil

from models import asr, processor 
import pydub
import numpy as np
import noisereduce as nr


whisper_models = ["base", "small", "medium", "large"]
source_languages = {
    "en": "English",
}

source_language_list = [key[0] for key in source_languages.items()]

MODEL_NAME = "vumichien/whisper-medium-jp"
lang = "ja"

device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)


pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
def _preprocess(filename):
     audio_name = filename
     subprocess.call(['ffmpeg', '-y', '-i', filename, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', "-loglevel", "quiet", audio_name])
     return audio_name
def pr2(start,end,filename):
    audio_name = filename
    subprocess.call(['ffmpeg', '-y', '-i', filename, 'ffmpeg', '-i', filename, '-ss', '20', '-to', '40' ,'-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', "-loglevel", "quiet", audio_name])
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

    text = pipe(file)["text"]

    return warn_output + text

def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str

def yt_transcribe(yt_url):
    yt = YouTube(yt_url)
    html_embed_str = _return_yt_html_embed(yt_url)
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename="audio.mp3")

    text = pipe("audio.mp3")["text"]

    return html_embed_str, text

def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))

def get_youtube(video_url):
    yt = YouTube(video_url)
    abs_video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    print("Success download video")
    print(abs_video_path)
    return abs_video_path

def speech_to_text(video_file_path, selected_source_lang, whisper_model, num_speakers):
    """
    # Transcribe youtube link using OpenAI Whisper
    1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
    2. Generating speaker embeddings for each segments.
    3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
    
    Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
    Speaker diarization model and pipeline from by https://github.com/pyannote/pyannote-audio
    """
    
    model = whisper.load_model(whisper_model)
    time_start = time.time()
    if(video_file_path == None):
        raise ValueError("Error no video input")
    print(video_file_path)

    try:
        # Read and convert youtube video
        _,file_ending = os.path.splitext(f'{video_file_path}')
        print(f'file enging is {file_ending}')
        audio_file = video_file_path.replace(file_ending, ".wav")
        print("starting conversion to wav")
        os.system(f'ffmpeg -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"')
        
        # Get duration
        with contextlib.closing(wave.open(audio_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"conversion to wav ready, duration of audio file: {duration}")

        # Transcribe audio
        options = dict(language=selected_source_lang, beam_size=16, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        print('speech array')
        #speech_array = _preprocess(filename=audio_file)
        
        asr.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language='bengali', task="transcribe")
        result = model.transcribe(audio_file, **transcribe_options)
        sound_file = pydub.AudioSegment.from_wav(audio_file)+6
        sound_file_Value = np.array(sound_file.get_array_of_samples())
        segments = result["segments"]
        print("starting whisper done with whisper")
    except Exception as e:
        raise RuntimeError("Error converting video to audio")

    try:
        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file, clip)
            
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')

        # Assign speaker label
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        # Make output

        objects = {
            'Start' : [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        print('hello world')
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '
            
        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
        objects['Text'].append(text)
        '''
        s = int(0*10000)
        e = int(26*10000)
        new_file=sound_file_Value[s : e]
        print(len(new_file))
        song = pydub.AudioSegment(new_file.tobytes(), frame_rate=sound_file.frame_rate,sample_width=sound_file.sample_width,channels=1)
        song.export("audio2.wav", format="wav",bitrate="256k")
        filen = "audio2.wav"
        rate, data = wavfile.read("audio2.wav")
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        wavfile.write("audio2.wav", rate, reduced_noise)
        speech_array = _preprocess(filen)
        print(asr(speech_array))
        '''
        print(objects)
        time_end = time.time()
        time_diff = time_end - time_start
        memory = psutil.virtual_memory()
        gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
        gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
        gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0
        system_info = f"""
        *Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB.* 
        *Processing time: {time_diff:.5} seconds.*
        *GPU Utilization: {gpu_utilization}%, GPU Memory: {gpu_memory}MiB.*
        """

        return pd.DataFrame(objects), system_info
    
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)


# ---- Gradio Layout -----
# Inspiration from https://huggingface.co/spaces/RASMUS/Whisper-youtube-crosslingual-subtitles
video_in = gr.Video(label="Video file", mirror_webcam=False)
youtube_url_in = gr.Textbox(label="Youtube url", lines=1, interactive=True)
df_init = pd.DataFrame(columns=['Start', 'End', 'Speaker', 'Text'])
memory = psutil.virtual_memory()
selected_source_lang = gr.Dropdown(choices=source_language_list, type="value", value="en", label="Spoken language in video", interactive=True)
selected_whisper_model = gr.Dropdown(choices=whisper_models, type="value", value="base", label="Selected Whisper model", interactive=True)
number_speakers = gr.Number(precision=0, value=2, label="Selected number of speakers", interactive=True)
system_info = gr.Markdown(f"*Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB*")
transcription_df = gr.DataFrame(value=df_init,label="Transcription dataframe", row_count=(0, "dynamic"), max_rows = 10, wrap=True, overflow_row_behaviour='paginate')
title = "Whisper speaker diarization"
demo = gr.Blocks(title=title)
demo.encrypt = False


with demo:
    with gr.Tab("Whisper speaker diarization"):
        gr.Markdown('''
            <div>
            <h1 style='text-align: center'>Whisper speaker diarization</h1>
            
            </div>
        ''')

        with gr.Row():
            gr.Markdown('''
            ### Transcribe youtube link using OpenAI Whisper
            ##### 1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
            ##### 2. Generating speaker embeddings for each segments.
            ##### 3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
            ''')
            
        with gr.Row():         
            gr.Markdown('''
                ### You can test by following examples:
                ''')
        examples = gr.Examples(examples=
                [ "https://www.youtube.com/watch?v=j7BfEzAFuYc&t=32s", 
                  "https://www.youtube.com/watch?v=-UX0X45sYe4", 
                  "https://www.youtube.com/watch?v=7minSgqi-Gw"],
              label="Examples", inputs=[youtube_url_in])
              

        with gr.Row():
            with gr.Column():
                youtube_url_in.render()
                download_youtube_btn = gr.Button("Download Youtube video")
                download_youtube_btn.click(get_youtube, [youtube_url_in], [
                    video_in])
                print(video_in)
                

        with gr.Row():
            with gr.Column():
                video_in.render()
                with gr.Column():
                    gr.Markdown('''
                    ##### Here you can start the transcription process.
                    ##### Please select the source language for transcription.
                    ##### You should select a number of speakers for getting better results.
                    ''')
                selected_source_lang.render()
                selected_whisper_model.render()
                number_speakers.render()
                transcribe_btn = gr.Button("Transcribe audio and diarization")
                transcribe_btn.click(speech_to_text, [video_in, selected_source_lang, selected_whisper_model, number_speakers], [transcription_df, system_info])

                
        with gr.Row():
            gr.Markdown('''
            ##### Here you will get transcription  output
            ##### ''')
            

        with gr.Row():
            with gr.Column():
                transcription_df.render()
                system_info.render()
                gr.Markdown('''<center><img src='https://visitor-badge.glitch.me/badge?page_id=WhisperDiarizationSpeakers' alt='visitor badge'></center>''')
    
    with gr.Tab("Whisper Transcribe Japanese Audio"):
        gr.Markdown(f'''
              <div>
              <h1 style='text-align: center'>Whisper Transcribe Japanese Audio</h1>
              </div>
              Transcribe long-form microphone or audio inputs with the click of a button! The fine-tuned
              checkpoint <a href='https://huggingface.co/{MODEL_NAME}' target='_blank'><b>{MODEL_NAME}</b></a> to transcribe audio files of arbitrary length.
          ''')
        microphone = gr.inputs.Audio(source="microphone", type="filepath", optional=True)
        upload = gr.inputs.Audio(source="upload", type="filepath", optional=True)
        transcribe_btn = gr.Button("Transcribe Audio")
        text_output = gr.Textbox()
        with gr.Row():         
            gr.Markdown('''
                ### You can test by following examples:
                ''')
        examples = gr.Examples(examples=
              [ "sample1.wav", 
                "sample2.wav", 
                ],
              label="Examples", inputs=[upload])
        transcribe_btn.click(transcribe, [microphone, upload], outputs=text_output)
    
    with gr.Tab("Whisper Transcribe Japanese YouTube"):
        gr.Markdown(f'''
              <div>
              <h1 style='text-align: center'>Whisper Transcribe Japanese YouTube</h1>
              </div>
                Transcribe long-form YouTube videos with the click of a button! The fine-tuned checkpoint:
                <a href='https://huggingface.co/{MODEL_NAME}' target='_blank'><b>{MODEL_NAME}</b></a> to transcribe audio files of arbitrary length.
            ''')
        youtube_link = gr.Textbox(label="Youtube url", lines=1, interactive=True)
        yt_transcribe_btn = gr.Button("Transcribe YouTube")
        text_output2 = gr.Textbox()
        html_output = gr.Markdown()
        yt_transcribe_btn.click(yt_transcribe, [youtube_link], outputs=[html_output, text_output2])

demo.launch(debug=True)