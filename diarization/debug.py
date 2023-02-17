import gradio as gr
import pytube as pt
import subprocess
import time
import librosa
from models import asr
import whisper
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


file = ''
speech_array = _preprocess(filename=file)

text = asr(speech_array)
print(text)
 