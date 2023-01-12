import os

abs_path = os.path.abspath('.')
base_dir = os.path.dirname(abs_path)
os.environ['TRANSFORMERS_CACHE'] = os.path.join(base_dir, 'models_cache')

from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor, AutoConfig, WhisperProcessor, WhisperForConditionalGeneration
from typing import Union, BinaryIO
import torch


task = "transcribe"  # transcribe or translate

# model_name = 'openai/whisper-small'
# model_name = 'openai/whisper-large' 
## v2: trained on more epochs with regularization
# model_name = 'openai/whisper-large-v2' 
#bangla
# model_name = 'anuragshas/whisper-large-v2-bn' 
model_name = 'anuragshas/whisper-small-bn' 

## lets you know the device count: cuda:0 or cuda:1
# print(torch.cuda.device_count())


# device = 0 if torch.cuda.is_available() else -1
device = -1 #Exclusively CPU

print(f"Using device: {'GPU' if device==0 else 'CPU'}")

if device != 0:
    print("[Warning!] Using CPU could hamper performance")

print("Loading Tokenizer for ASR Speech-to-Text Model...\n" + "*" * 100)
# tokenizer = AutoTokenizer.from_pretrained(model_name, language=language, task=task)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading Feature Extractor for ASR Speech-to-Text Model...\n" + "*" * 100)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

print("Loading Config for ASR Speech-to-Text Model...\n" + "*" * 100)
config = AutoConfig.from_pretrained(model_name)

print("Loading Processor for ASR Speech-to-Text Model...\n" + "*" * 100)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

print("Loading WHISPER ASR Speech-to-Text Model...\n" + "*" * 100)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    processor=processor, #no effect see: https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/pipelines/automatic_speech_recognition.py
    config=config, #no effect see: https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/pipelines/automatic_speech_recognition.py
    device=device,  # for gpu 1 for cpu -1
    ## chunk files longer than 30s into shorted samples
    chunk_length_s=30, 
    ## the amount of overlap (in secs) to be discarded while stitching the inferenced chunks
    ## stride_length_s is a tuple of the left and right stride(overlap) length.
    ## With only 1 number, both sides get the same stride, by default
    ## The stride_length on one side is 1/6th of the chunk_length_s if stride_length no provided
    stride_length_s=(5, 5),
    ignore_warning=True,
    # decoder_kwargs={"max_new_tokens": 448},  ##default is 448
)


def transcribe(speech_array: Union[str, BinaryIO], language: str = "en") -> str:
    """
    Transcribes an audio array to text
    Args:
        speech_array (np.ndarray): audio in numpy array format
        language (str): "sv" or "en"
    Returns:
        a string containing transcription
    """
    asr.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    # asr.model.config.max_new_tokens = 448 #default is 448
    
    result = asr(speech_array)

    return str(result["text"])


