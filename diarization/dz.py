from __future__ import unicode_literals
import youtube_dl
import yt_dlp
from pydub import AudioSegment
from pyannote.audio import Pipeline
import re
import whisper
import os
import ffmpeg
import subprocess
import gradio as gr
import traceback
import json
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
from pydub.effects import speedup
import moviepy.editor as mp
import datetime
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import SpeechBrainPretrainedSpeakerEmbedding #PyannoteAudioPretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import json
from datetime import timedelta
import pytube as pt
from models import asr
from transformers import T5ForConditionalGeneration, T5Tokenizer

__FILES = set()
wispher_models = list(whisper._MODELS.keys())

def _preprocess(filename):
     audio_name = "temp_audio.wav"
     subprocess.call(['ffmpeg', '-y', '-i', filename, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', "-loglevel", "quiet", audio_name])
     return audio_name


def correct_grammar(input_text,num_return_sequences=1):
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = T5Tokenizer.from_pretrained('deep-learning-analytics/GrammarCorrector')
    model = T5ForConditionalGeneration.from_pretrained('deep-learning-analytics/GrammarCorrector').to(torch_device)
    batch = tokenizer([input_text],truncation=True,padding='max_length',max_length=len(input_text), return_tensors="pt").to(torch_device)
    results = model.generate(**batch,max_length=len(input_text),num_beams=2, num_return_sequences=num_return_sequences, temperature=1.5)
    generated_sequences = []
    for generated_sequence_idx, generated_sequence in enumerate(results):
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        generated_sequences.append(text)
    generated_text = "".join(generated_sequences)
    _generated_text = ""
    for idx, _sentence in enumerate(generated_text.split('.'), 0):
        if not idx:
            _generated_text+=_sentence+'.'
        elif _sentence[:1]!=' ':
            _generated_text+=' '+_sentence+'.'
        elif _sentence[:1]=='':
            pass
        else:
            _generated_text+=_sentence+'.'
    return _generated_text

def CreateFile(filename):
    __FILES.add(filename)
    return filename

def RemoveFile(filename):
    if (os.path.isfile(filename)):
        os.remove(filename)

def RemoveAllFiles():
    for file in __FILES:
        if (os.path.isfile(file)):
            os.remove(file)
    
def Transcribe_V1(NumberOfSpeakers, SpeakerNames="", audio="temp_audio.wav"):
    SPEAKER_DICT = {}
    SPEAKERS = [speaker.strip() for speaker in SpeakerNames.split(',') if len(speaker)]
    
    def GetSpeaker(sp):
        speaker = sp
        if sp not in list(SPEAKER_DICT.keys()):
            if len(SPEAKERS):
                t = SPEAKERS.pop(0)
                SPEAKER_DICT[sp] = t
                speaker = SPEAKER_DICT[sp]
        else:
            speaker = SPEAKER_DICT[sp]
        return speaker
        
    def millisec(timeStr):
        spl = timeStr.split(":")
        s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
        return s
    
    def preprocess(audio):
        t1 = 0 * 1000 
        t2 = 20 * 60 * 1000
        newAudio = AudioSegment.from_wav(audio)
        a = newAudio[t1:t2]
        spacermilli = 2000
        spacer = AudioSegment.silent(duration=spacermilli)
        newAudio = spacer.append(a, crossfade=0)
        newAudio.export(audio, format="wav")
        return spacermilli, spacer
    
    def diarization(audio):
        as_audio = AudioSegment.from_wav(audio)
        DEMO_FILE = {'uri': 'blabal', 'audio': audio}
        if NumberOfSpeakers:
            dz = pipeline(DEMO_FILE, num_speakers=NumberOfSpeakers)  
        else:
            dz = pipeline(DEMO_FILE)  
        with open(CreateFile(f"diarization_{audio}.txt"), "w") as text_file:
            text_file.write(str(dz))
        dz = open(CreateFile(f"diarization_{audio}.txt")).read().splitlines()
        dzList = []
        for l in dz:
            start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
            start = millisec(start)
            end = millisec(end)
            lex = GetSpeaker(re.findall('(SPEAKER_[0-9][0-9])', string=l)[0])
            dzList.append([start, end, lex])
        sounds = spacer
        segments = []
        dz = open(CreateFile(f"diarization_{audio}.txt")).read().splitlines()
        for l in dz:
            start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
            start = millisec(start)
            end = millisec(end) 
            segments.append(len(sounds))
            sounds = sounds.append(as_audio[start:end], crossfade=0)
            sounds = sounds.append(spacer, crossfade=0)
        sounds.export(CreateFile(f"dz_{audio}.wav"), format="wav")
        return f"dz_{audio}.wav", dzList, segments
    
    def transcribe(dz_audio):
        model = whisper.load_model("medium")
        result = model.transcribe(dz_audio)
        # for _ in result['segments']:
        #     print(_['start'], _['end'], _['text'])
        captions = [[((caption["start"]*1000)), ((caption["end"]*1000)),  caption["text"]] for caption in result['segments']]
        conversation = []
        for i in range(len(segments)):
            idx = 0
            for idx in range(len(captions)):
                if captions[idx][0] >= (segments[i] - spacermilli):
                    break;
            
            while (idx < (len(captions))) and ((i == len(segments) - 1) or (captions[idx][1] < segments[i+1])):
                  c = captions[idx]  
                  start = dzList[i][0] + (c[0] -segments[i])
                  if start < 0: 
                      start = 0
                  idx += 1
                  if not len(conversation):
                      conversation.append([dzList[i][2], c[2]])
                  elif conversation[-1][0] == dzList[i][2]:
                      conversation[-1][1] +=  c[2]
                  else:
                      conversation.append([dzList[i][2], c[2]])
                  #print(f"[{dzList[i][2]}] {c[2]}")
        return conversation, ("".join([f"{speaker} --> {text}\n" for speaker, text in conversation]))

    spacermilli, spacer = preprocess(audio)
    dz_audio, dzList, segments = diarization(audio)
    conversation, t_text = transcribe(dz_audio)
    RemoveAllFiles()
    return (t_text, ({ "data": [{"speaker": speaker, "text": text} for speaker, text in conversation]}))


def Transcribe_V2(model, num_speakers, speaker_names, audio="temp_audio.wav"):
    model = whisper.load_model(model)
    # embedding_model = SpeechBrainPretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
    
    embedding_model = SpeechBrainPretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    SPEAKER_DICT = {}
    default_speaker_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    SPEAKERS = [speaker.strip() for speaker in speaker_names.split(',') if len(speaker)]
    def GetSpeaker(sp):
        speaker = sp
        if sp not in list(SPEAKER_DICT.keys()):
            if len(SPEAKERS):
                t = SPEAKERS.pop(0)
                SPEAKER_DICT[sp] = t
                speaker = SPEAKER_DICT[sp]
            elif len(default_speaker_names):
                t = default_speaker_names.pop(0)
                SPEAKER_DICT[sp] = t
                speaker = SPEAKER_DICT[sp]
        else:
            speaker = SPEAKER_DICT[sp]
        return speaker
    
    # audio = Audio()
    def diarization(audio):
        def millisec(timeStr):
            spl = timeStr.split(":")
            s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
            return s
        as_audio = AudioSegment.from_wav(audio)
        DEMO_FILE = {'uri': 'blabal', 'audio': audio}
        hparams = pipeline.parameters(instantiated=True)
        hparams["segmentation"]["min_duration_off"] -= 0.25
        pipeline.instantiate(hparams)
        if num_speakers:
            dz = pipeline(DEMO_FILE, num_speakers=num_speakers)  
        else:
            dz = pipeline(DEMO_FILE)  
        with open(CreateFile(f"diarization_{audio}.txt"), "w") as text_file:
            text_file.write(str(dz))
        dz = open(CreateFile(f"diarization_{audio}.txt")).read().splitlines()
        print(dz)
        dzList = []
        for l in dz:
            start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
            start = millisec(start)
            end = millisec(end)
            lex = GetSpeaker(re.findall('(SPEAKER_[0-9][0-9])', string=l)[0])
            dzList.append([start, end, lex])
        return dzList
    
    def get_output(segments):
        # print(segments)
        conversation=[]
        for (i, segment) in enumerate(segments):
            # print(f"{i}, {segment["speaker"]}, {segments[i - 1]["speaker"]}, {}")
            if not len(conversation):
                conversation.append([str(timedelta(seconds=float(segment['start']))),str(timedelta(seconds=float(segment['end']))),GetSpeaker(segment["speaker"]), segment["text"].lstrip()])
            elif conversation[-1][2] == GetSpeaker(segment["speaker"]):
                conversation[-1][3] +=  segment["text"].lstrip()
            else:
                conversation.append([str(timedelta(seconds=float(segment['start']))),str(timedelta(seconds=float(segment['end']))),GetSpeaker(segment["speaker"]), segment["text"].lstrip()])
            # if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            #     if i != 0:
            #         conversation.append([GetSpeaker(segment["speaker"]), segment["text"][1:]]) # segment["speaker"] + ' ' + str(time(segment["start"])) + '\n\n'
            # conversation[-1][1] += segment["text"][1:]
        # return output
        for idx in range(len(conversation)):
            conversation[idx][3] = correct_grammar(conversation[idx][3])
        return ("".join([f"[{start}] - {speaker} \n{text}\n" for start, end, speaker, text in conversation])), ({ "data": [{"start": start, "end":end, "speaker": speaker, "text": text} for start, end, speaker, text in conversation]})

    def get_duration(path):
        with contextlib.closing(wave.open(path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
        return frames / float(rate)

    def make_embeddings(path, segments, duration):
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(path, segment, duration)
        return np.nan_to_num(embeddings)

    def segment_embedding(path, segment, duration):
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])
        print(start)
        print(segment['text'])
        print(end)
        clip = Segment(start, end)
        waveform, sample_rate = Audio().crop(path, clip)
        return embedding_model(waveform[None])

    def add_speaker_labels(segments, embeddings, num_speakers):
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    def time(secs):
        return datetime.timedelta(seconds=round(secs))

    duration = get_duration(audio)
    if duration > 4 * 60 * 60:
        return "Audio duration too long"

    # print(json.dumps(diarization(audio)))
    speech_array = _preprocess(filename='temp_audio.wav')
    print(asr(speech_array)['text'])
    result = model.transcribe(audio)
    # print(json.dumps(result))

    segments = result["segments"]

    num_speakers = min(max(round(num_speakers), 1), len(segments))
    if len(segments) == 1:
        segments[0]['speaker'] = 'SPEAKER 1'
    else:
        embeddings = make_embeddings(audio, segments, duration)
        add_speaker_labels(segments, embeddings, num_speakers)
    return get_output(segments)
    # return output

def AudioTranscribe(NumberOfSpeakers=None, SpeakerNames="", audio="", retries=5, model='base'):
    print(f"{NumberOfSpeakers}, {SpeakerNames}, {retries}")
    if retries:
        # subprocess.call(['ffmpeg', '-i', audio,'temp_audio.wav'])
        try:
            subprocess.call(['ffmpeg', '-i', audio,'temp_audio.wav'])
        except Exception as ex:
            traceback.print_exc()
            return AudioTranscribe(NumberOfSpeakers, SpeakerNames, audio, retries-1)
        if not (os.path.isfile("temp_audio.wav")):
            return AudioTranscribe(NumberOfSpeakers, SpeakerNames, audio, retries-1)
        return Transcribe_V2(model, NumberOfSpeakers, SpeakerNames)
    else:
        raise gr.Error("There is some issue ith Audio Transcriber. Please try again later!")

def VideoTranscribe(NumberOfSpeakers=None, SpeakerNames="", video="", retries=5, model='base'):
    if retries:
        try:
            clip = mp.VideoFileClip(video)
            clip.audio.write_audiofile("temp_audio.wav")
            # command = f"ffmpeg -i {video} -ab 160k -ac 2 -ar 44100 -vn temp_audio.wav"
            # subprocess.call(command, shell=True)
        except Exception as ex:
            traceback.print_exc()
            return VideoTranscribe(NumberOfSpeakers, SpeakerNames, video, retries-1)
        if not (os.path.isfile("temp_audio.wav")):
            return VideoTranscribe(NumberOfSpeakers, SpeakerNames, video, retries-1)
        return Transcribe_V2(model, NumberOfSpeakers, SpeakerNames)
    else:
        raise gr.Error("There is some issue ith Video Transcriber. Please try again later!")

def YoutubeTranscribe(NumberOfSpeakers=None, SpeakerNames="", URL="", retries = 5, model='base'):
    if retries:
        if "youtu" not in URL.lower():
            raise gr.Error(f"{URL} is not a valid youtube URL.")
        else:
            RemoveFile("temp_audio.wav")
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': 'temp_audio.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
            }
            
            yt = pt.YouTube(URL)
            #html_embed_str = _return_yt_html_embed(yt_url)
            stream = yt.streams.filter(only_audio=True)[0]
            filename = "temp_audio.mp3"

            stream.download(filename=filename)
            import subprocess

            subprocess.call(['ffmpeg', '-i', 'temp_audio.mp3',
        'temp_audio.wav'])

            #removeFile("temp_audio.m4a")
            return Transcribe_V2(model, NumberOfSpeakers, SpeakerNames)
    else:
        raise gr.Error(f"Unable to get video from {URL}")
 

with gr.Blocks() as yav_ui:
    with gr.Row():
        with gr.Column():
            with gr.Tab("Youtube", id=1):
                ysz = gr.Dropdown(label="Model Size", choices=wispher_models , value='base')
                yinput_nos = gr.Number(label="Number of Speakers", placeholder="2")
                yinput_sn = gr.Textbox(label="Name of the Speakers (ordered by the time they speak and separated by comma)", placeholder="If Speaker 1 is first to speak followed by Speaker 2 then -> Speaker 1, Speaker 2")
                yinput = gr.Textbox(label="Youtube Link", placeholder="https://www.youtube.com/watch?v=GECcjrYHH8w")
                ybutton_transcribe = gr.Button("Transcribe", show_progress=True, scroll_to_output=True)
            with gr.Tab("Video", id=2):
                vsz = gr.Dropdown(label="Model Size", choices=wispher_models, value='base')
                vinput_nos = gr.Number(label="Number of Speakers", placeholder="2")
                vinput_sn = gr.Textbox(label="Name of the Speakers (ordered by the time they speak and separated by comma)", placeholder="If Speaker 1 is first to speak followed by Speaker 2 then -> Speaker 1, Speaker 2")
                vinput = gr.Video(label="Video")
                vbutton_transcribe = gr.Button("Transcribe", show_progress=True, scroll_to_output=True)
            with gr.Tab("Audio", id=3):
                asz = gr.Dropdown(label="Model Size", choices=wispher_models , value='base')
                ainput_nos = gr.Number(label="Number of Speakers", placeholder="2")
                ainput_sn = gr.Textbox(label="Name of the Speakers (ordered by the time they speak and separated by comma)", placeholder="If Speaker 1 is first to speak followed by Speaker 2 then -> Speaker 1, Speaker 2")
                ainput = gr.Audio(label="Audio", type="filepath")
                abutton_transcribe = gr.Button("Transcribe", show_progress=True, scroll_to_output=True)
        with gr.Column():
            with gr.Tab("Text"):
                output_textbox = gr.Textbox(label="Transcribed Text", lines=15)
            with gr.Tab("JSON"):
                output_json = gr.JSON(label="Transcribed JSON")
    ybutton_transcribe.click(
                fn=YoutubeTranscribe,
                inputs=[yinput_nos,yinput_sn,yinput, ysz],
                outputs=[output_textbox,output_json]
            )
    abutton_transcribe.click(
                fn=AudioTranscribe,
                inputs=[ainput_nos,ainput_sn,ainput, asz],
                outputs=[output_textbox,output_json]
            )
    vbutton_transcribe.click(
                fn=VideoTranscribe,
                inputs=[vinput_nos,vinput_sn,vinput, vsz],
                outputs=[output_textbox,output_json]
            )
yav_ui.launch(debug=True)