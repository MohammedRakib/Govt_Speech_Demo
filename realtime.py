import os

abs_path = os.path.abspath('.')
#base_dir = os.path.dirname(os.path.dirname(abs_path))
base_dir = os.path.dirname(abs_path)
os.environ['TRANSFORMERS_CACHE'] = os.path.join(base_dir, 'models_cache')

import datetime
import sys
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

# model_name = "shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice"
model_name = "Rakib/whisper-small-bn-all"

pipe = pipeline("automatic-speech-recognition", model=model_name, device=0)
sampling_rate = pipe.feature_extractor.sampling_rate


start = datetime.datetime.now()

chunk_length_s = 5
stream_chunk_s = 0.1
mic = ffmpeg_microphone_live(
    sampling_rate=sampling_rate,
    chunk_length_s=chunk_length_s,
    stream_chunk_s=stream_chunk_s,
)
print("Start talking...")
for item in pipe(mic):
    sys.stdout.write("\033[K")
    print(item["text"], end="\r")
    if not item["partial"][0]:
        print("")
