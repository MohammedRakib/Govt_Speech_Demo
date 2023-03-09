## Dependencies (run in the venv before running this script)
# pip install git+https://github.com/huggingface/datasets git+https://github.com/huggingface/transformers 
# pip install huggingface_hub ipywidgets librosa evaluate>=0.3.0 jiwer bnunicodenormalizer
# sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
# sudo apt update
# sudo apt install -y ffmpeg
#sudo apt-get install git-lfs


## Run the following commands separately before running the py version of this notebook to connect to HuggningFace Hub!
#   git config --global credential.helper store
#   huggingface-cli login
## Enter your token by visiting: https://huggingface.co/settings/tokens
## Create a repo in the hub:
#   huggingface-cli repo create whisper-small-es OR, huggingface-cli repo create <repo_name/model_name>
## Install git lfs and clone the just created repo
#   git lfs install
#   git clone <repo_link>
## cd to the cloned repo and copy (cp) this python script or everything in training dir to that repo
#   cd <repo_name/model_name>
#   cp /home/mamun/asr_training/Govt_Speech_Demo/training/my-training.py .


#For more details: https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event



## 1. Setting Up Environment Variables & Devices
import os
import torch

abs_path = os.path.abspath('.')
# base_dir = os.path.dirname(os.path.dirname(abs_path))
base_dir = os.path.dirname(abs_path)

os.environ['TRANSFORMERS_CACHE'] = os.path.join(base_dir, 'models_cache')
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_DATASETS_CACHE'] = os.path.join(base_dir, 'datasets_cache')
os.environ['HF_DATASETS_OFFLINE'] = '0'

device = "GPU" if torch.cuda.is_available() else "CPU"
print(f"\n\n Device to be used: {device} \n\n")


## 2. Setting Up Variables
model_name = "openai/whisper-tiny"
# model_name = "openai/whisper-small"
# model_name = "openai/whisper-large-v2"

language = "Bengali"
task = "transcribe" # transcribe or translate
print(f"\n\n Loading {model_name} for {language} to {task}...this might take a while.. \n\n")


## 3. Setting Up Training Args
output_dir = "./" 
overwrite_output_dir = True
# max_steps = 30000 
max_steps = 5
# per_device_train_batch_size = 4 
per_device_train_batch_size = 1 
# per_device_eval_batch_size = 32 
per_device_eval_batch_size = 1 
# gradient_accumulation_steps = 8 
gradient_accumulation_steps = 1 
gradient_checkpointing = False 
evaluation_strategy ="steps" 
eval_steps = 5
# eval_steps = 1000 
save_strategy = "steps" 
# save_steps = 1000 
save_steps = 5
save_total_limit = 5 
learning_rate = 1e-5 
# warmup_steps = 3000 
warmup_steps = 1 
# logging_steps = 25
logging_steps = 1
weight_decay = 0.01 
load_best_model_at_end = True 
metric_for_best_model = "wer" 
greater_is_better = False 
# bf16 = True 
bf16 = False
# tf32 = True 
tf32 = False
generation_max_length = 448
report_to = ["tensorboard"] 
predict_with_generate = True
# push_to_hub = True
push_to_hub = False
freeze_feature_encoder = False 


## 4. Load Datasets
print("\n\n Loading Datasets...this might take a while..\n\n")

from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()
google_fleurs = DatasetDict()
openslr = DatasetDict()
## commonvoice_11.0 + google_fleurs + openslr53
my_dataset = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="train+validation+other", cache_dir=os.path.join(base_dir, 'datasets_cache'))
google_fleurs["train"] = load_dataset("google/fleurs", "bn_in", split="train+validation", cache_dir=os.path.join(base_dir, 'datasets_cache'))
openslr = load_dataset("openslr", "SLR53", cache_dir=os.path.join(base_dir, 'datasets_cache'))

common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="test", cache_dir=os.path.join(base_dir, 'datasets_cache'))
google_fleurs["test"] = load_dataset("google/fleurs", "bn_in", split="test", cache_dir=os.path.join(base_dir, 'datasets_cache'))

# skipFiles = open("corrupt_files.txt").read().splitlines()
# skipFiles = skipFiles[3:]
# length = len(skipFiles)
# first = skipFiles[0]
# last = skipFiles[-1]
# print(f"\n No. of corrupt files: {length}, First: {first}, Last {last}\n")

# print("\n Finding indexes of corrupt files... \n")
# from tqdm import tqdm
# count=0
# error_index = []
# for i in tqdm(range(len(common_voice["train"]))):
#     path = common_voice["train"][i]["path"].split("/")[-1].split(".")[0]
#     if path in skipFiles:
#         # print(path)
#         count+=1
#         error_index.append(i)
# print(f"\n Total Corrupt Files: {count} \n")

# print("\n Removing corrupt files from the Common Voice dataset...\n")
# common_voice["train"] = common_voice["train"].filter(lambda example, idx: idx not in error_index, with_indices=True)

print("\n\n So, the datasets to be trained are: \n\n")
print("\n Common Voice 11.0 - Bangla\n")
print(common_voice)
print("\n Google Fleurs - Bangla \n")
print(google_fleurs)
print("\n OpenSLR-53 - Bangla \n")
print(openslr)
print("\n")


## 5. Small Subset for Testing
common_voice['train']  = common_voice['train'].select(range(50))
common_voice['test']  = common_voice['test'].select(range(50))
google_fleurs['train']  = google_fleurs['train'].select(range(50))
google_fleurs['test']  = google_fleurs['test'].select(range(50))
openslr['train'] = openslr['train'].select(range(50))

print("\n\n For testing, the small subsets are:")
print(common_voice)
print(google_fleurs)
print(openslr)
print("\n")


## 6. Merge Datasets
from datasets import concatenate_datasets, Audio

sampling_rate = 16000

## resample to specified sampling rate
common_voice = common_voice.cast_column("audio", Audio(sampling_rate))
google_fleurs = google_fleurs.cast_column("audio", Audio(sampling_rate))
openslr = openslr.cast_column("audio", Audio(sampling_rate))

## normalise columns to ["audio", "sentence"]
common_voice = common_voice.remove_columns(
    set(common_voice['test'].features.keys()) - {"audio", "sentence"}
)

google_fleurs = google_fleurs.rename_column("raw_transcription", "sentence")
google_fleurs = google_fleurs.remove_columns(
    set(google_fleurs['test'].features.keys()) - {"audio", "sentence"}
)

openslr = openslr.remove_columns(
    set(openslr['train'].features.keys()) - {"audio", "sentence"}
)



## check if all audio are in float32 dtype or not.
## a fix is: https://github.com/huggingface/datasets/issues/5345
print("\n Checking all audio dtype is float32 or not... \n")
print(f'Common Voice Train: {common_voice["train"][0]["audio"]["array"].dtype}')
print(f'Common Voice Test: {common_voice["test"][0]["audio"]["array"].dtype}')
print(f'Google Fleurs Train: {google_fleurs["train"][0]["audio"]["array"].dtype}')
print(f'Google Fleurs Test: {google_fleurs["test"][0]["audio"]["array"].dtype}')
print(f'OpenSlR: {openslr["train"][0]["audio"]["array"].dtype}')
print("\n")


## merge the three datasets
my_dataset['train'] = concatenate_datasets([common_voice['train'], google_fleurs['train'], openslr['train']]) #for linux
# my_dataset['train'] = concatenate_datasets([common_voice['train'], openslr['train']])
# my_dataset['train'] = concatenate_datasets([google_fleurs['train'], openslr['train']]) #for windows no commonvoice as it requires ffmpeg-4
# my_dataset['train'] = google_fleurs['train']
my_dataset['test'] = concatenate_datasets([common_voice['test'], google_fleurs['test']]) #for linux
# my_dataset['test'] = common_voice['test']
# my_dataset['test'] = concatenate_datasets([google_fleurs['test']]) #for windows no commonvoice as it requires ffmpeg-4

#shuffle train set with seed=42
my_dataset['train'] = my_dataset['train'].shuffle(seed=42)

print("\n\n AFTER MERGING, final train and validation sets are: ")
print("\n My FINAL DATASET \n")
print(my_dataset)
print("\n")


## 7. Prepare Feature Extractor, Tokenizer and Processor
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

# tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task, use_fast=True)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)

processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)


## 8. Preprocessing Data
print("\n\n Preprocessing Datasets...this might take a while..\n\n")

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from bnunicodenormalizer import Normalizer

do_lower_case = False
do_remove_punctuation = False
do_bangla_unicode_normalization = True

normalizer = BasicTextNormalizer()
bangla_normalizer = Normalizer(allow_english=True)

def prepare_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # compute input length
    batch["input_length"] = len(batch["audio"])
    
    # optional pre-processing steps
    transcription = batch["sentence"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    if do_bangla_unicode_normalization:
        _words = [bangla_normalizer(word)['normalized'] for word in transcription.split()]
        transcription = " ".join([word for word in _words if word is not None])
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    
     # compute labels length **with** special tokens! -> total label length
    batch["labels_length"] = len(batch["labels"])
    
    return batch

## my_dataset is DatasetDict dictionary whereas my_dataset["train"] is Dataset Object.
## map function parameters for both are different!
## see: https://github.com/huggingface/datasets/issues/2407

## This,
my_dataset = my_dataset.map(prepare_dataset, 
                            num_proc=1, # if num_proc>1, then mapping might get stuck. use num_proc=1 in that case.
                            load_from_cache_file=True, 
                            cache_file_names={
                                "train" : os.path.join(base_dir, 'datasets_cache', 'preprocessed_train_cache.arrow'),
                                "test" : os.path.join(base_dir, 'datasets_cache', 'preprocessed_test_cache.arrow'),
                                }
                            )
## OR this,
# my_dataset["train"] = my_dataset["train"].map(
#                             prepare_dataset, 
#                             num_proc=4, # if num_proc>1, then mapping might get stuck. use num_proc=1 in that case.
#                             load_from_cache_file=True, 
#                             cache_file_name=os.path.join(base_dir, 'datasets_cache', 'preprocessed_train_cache.arrow')
#                             )

# my_dataset["test"] = my_dataset["test"].map(
#                             prepare_dataset, 
#                             num_proc=4, # if num_proc>1, then mapping might get stuck. use num_proc=1 in that case.
#                             load_from_cache_file=True, 
#                             cache_file_name=os.path.join(base_dir, 'datasets_cache', 'preprocessed_test_cache.arrow')
#                             )


## 9. Filter too Short or too Long Audio Files
MAX_DURATION_IN_SECONDS = 30.0
max_input_length = MAX_DURATION_IN_SECONDS * 16000

def filter_inputs(input_length):
    """Filter inputs with zero input length or longer than 30s"""
    return 0 < input_length < max_input_length

my_dataset["train"] = my_dataset["train"].filter(
    filter_inputs,
    input_columns=["input_length"],
)
my_dataset["test"] = my_dataset["test"].filter(
    filter_inputs,
    input_columns=["input_length"],
)


max_label_length = 448 #(Check by doing model.config.max_length. Model not yet initialized, so manually written)

def filter_labels(labels_length):
    """Filter label sequences longer than max length (448)"""
    return labels_length < max_label_length

my_dataset["train"] = my_dataset["train"].filter(
    filter_labels,
    input_columns=["labels_length"],
)
my_dataset["test"] = my_dataset["test"].filter(
    filter_labels,
    input_columns=["labels_length"],
)


import re
def filter_transcripts(transcript):
    """Filter transcripts with empty strings and samples containing English characters & numbers"""
    pattern = r'^.*[a-zA-Z0-9]+.*$'
    match = re.match(pattern, transcript)
    return len(transcript.split(" ")) > 1 and not bool(match)

my_dataset["train"] = my_dataset["train"].filter(
    filter_transcripts,
    input_columns=["sentence"],
)
my_dataset["test"] = my_dataset["test"].filter(
    filter_transcripts,
    input_columns=["sentence"],
)

print("\n\n AFTER FILTERING, final train and validation sets are: ")
print("\n My FINAL DATASET \n")
print(my_dataset)
print("\n")


## 10. Save & Cleanup Cache Files (DON'T save too large datasets..will take up all space!!)
## Only save, if you want it to export it to another PC!! 
## Else, map function stores the cache files via cache_file_name parameter!!

# print("\n\n Saving Preprocessed Dataset to Disk..\n\n")

# my_dataset.save_to_disk(os.path.join(base_dir, "datasets_cache"))

## Removes unused cached files & returns the number of removed cache files
print("\n Removing UNUSED Cache Files: \n")
try:
    print(f"{my_dataset.cleanup_cache_files()} for my_dataset")
    print(f"{common_voice.cleanup_cache_files()} for common_voice")
    print(f"{google_fleurs.cleanup_cache_files()} for google_fleurs")
    print(f"{openslr.cleanup_cache_files()} for openslr")
    
except Exception as e:
    print(f"\n\n UNABLE to REMOVE some Cache files. \n Error: {e} \n\n")


## 11. Load Already Preprocessed Dataset from Disk
## Only load if you have a saved dataset via save_to_disk method!!
## Do Once 4 to 6 and 8 to 10. Then start from 7 and 11. EVERYTIME!!!

# from datasets import load_from_disk
# print("\n\n Loading Preprocessed Dataset from Disk..\n\n")

# my_dataset = load_from_disk(os.path.join(base_dir, "datasets_cache"))


## 12. Define Data Collator
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


## 13. Define Evaluation Metrics
import evaluate

metric = evaluate.load("wer", cache_dir=os.path.join(base_dir, "metrics_cache"))

do_normalize_eval = True

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


## 14. Load a Pre-Trained Checkpoint
print("\n\n Loading Model to Device..\n\n")

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name)


## 15. Override generation arguments
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
if gradient_checkpointing:
    model.config.use_cache = False
if freeze_feature_encoder:
    model.freeze_feature_encoder()


## 16. Define the Training Configuration
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    max_steps=max_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    evaluation_strategy=evaluation_strategy,
    eval_steps=eval_steps,
    save_strategy=save_strategy,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    logging_steps=logging_steps,
    weight_decay=weight_decay,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=greater_is_better,
    bf16=bf16,
    tf32=tf32,
    generation_max_length=generation_max_length,
    report_to=report_to,
    predict_with_generate=predict_with_generate,
    push_to_hub=push_to_hub,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=my_dataset["train"],
    eval_dataset=my_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

## We'll save the processor object once before starting training. Since the processor is not trainable, it won't change over the course of training.
## The checkpoint dirs don't save the processor files: 
## (added_tokens.json, merges.txt, normalizer.json, special_tokens_map.json, tokenizer_config.json, vocab.json)
## So, we save beforehand the processor in the best_model directory. 
## This is done so that if we stop training earlier than expected, 
## then we can copy the above files from the best_model dir to the checkpoint folder 
## to load the processor and run the model from the checkpoint dir.

# No need to create best_model folder as trainer automatically creates it!
# if not os.path.exists("best_model"):
#     os.makedirs("best_model")
processor.save_pretrained("best_model")


## 17. Training
print("\n\n Training STARTED..\n\n")

train_result = trainer.train()

## resume from the latest checkpoint
# train_result = trainer.train(resume_from_checkpoint=True)

## resume training from the specific checkpoint in the directory passed
# train_result = trainer.train(resume_from_checkpoint="checkpoint-4000")

print("\n\n Training COMPLETED...\n\n")


## 18. Evaluating & Saving Metrics & Model
print("\n\n Evaluating Model & Saving Metrics...\n\n")

processor.save_pretrained("best_model")
trainer.save_model("best_model")

metrics = train_result.metrics
trainer.save_metrics("train", metrics)
trainer.save_state()

metrics = trainer.evaluate(
    metric_key_prefix="eval",
    max_length=training_args.generation_max_length,
    num_beams=training_args.generation_num_beams,
)

trainer.save_metrics("eval", metrics)


## 19. Push to Hub
if push_to_hub:
    print("\n\n Pushing to Hub...\n\n")

    kwargs = {
        "dataset_tags": ["mozilla-foundation/common_voice_11_0", "google/fleurs", "openslr"],
        # "dataset_tags": ["mozilla-foundation/common_voice_11_0", "openslr"],
        "dataset": ["common-voice-11", "google-fleurs", "openslr53"],  # a 'pretty' name for the training dataset
        # "dataset": "common-voice-11+openslr53",  # a 'pretty' name for the training dataset
        "language": "bn",
        "model_name": "Whisper Small - Mohammed Rakib",  # a 'pretty' name for your model
        "finetuned_from": "openai/whisper-small",
        "tasks": "automatic-speech-recognition",
        "tags": "whisper-event",
    }

    trainer.push_to_hub(**kwargs)


print("\n\n DONEEEEEE \n\n")
