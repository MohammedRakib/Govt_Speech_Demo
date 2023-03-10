import argparse
import os
from regex import R

abs_path = os.path.abspath('.')
# base_dir = os.path.dirname(os.path.dirname(abs_path))
base_dir = os.path.dirname(abs_path)


os.environ['TRANSFORMERS_CACHE'] = os.path.join(base_dir, 'models_cache')
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_DATASETS_CACHE'] = os.path.join(base_dir, 'datasets_cache')
os.environ['HF_DATASETS_OFFLINE'] = '0'

from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import load_dataset, Audio
from bnunicodenormalizer import Normalizer
import evaluate
import unicodedata

wer_metric = evaluate.load("wer", cache_dir=os.path.join(base_dir, "metrics_cache"))
cer_metric = evaluate.load("cer", cache_dir=os.path.join(base_dir, "metrics_cache"))


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            "Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


whisper_norm = BasicTextNormalizer()
bangla_normalizer = Normalizer(allow_english=True)


def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch

def removeOptionalZW(text):
    """
    Removes all optional occurrences of ZWNJ or ZWJ from Bangla text.
    """
    # Regex for matching zero witdh joiner variations.
    STANDARDIZE_ZW = re.compile(r'(?<=\u09b0)[\u200c\u200d]+(?=\u09cd\u09af)')

    # Regex for removing standardized zero width joiner, except in edge cases.
    DELETE_ZW = re.compile(r'(?<!\u09b0)[\u200c\u200d](?!\u09cd\u09af)')
    
    text = STANDARDIZE_ZW.sub('\u200D', text)
    text = DELETE_ZW.sub('', text)
    return text

def bn_unicode_normalise(batch):
    _words = [bangla_normalizer(word)['normalized'] for word in get_text(batch).split()]
    normalized_text = " ".join([word for word in _words if word is not None])
    normalized_text = normalized_text.replace("\u2047", "-")
    normalized_text = normalized_text.replace(u"\u098c", u"\u09ef")
    normalized_text = unicodedata.normalize("NFC", normalized_text)
    normalized_text = removeOptionalZW(normalized_text)
    batch["norm_text"] = whisper_norm(normalized_text)
    return batch


def data(dataset):
    for item in dataset:
        yield {**item["audio"], "reference": item["norm_text"]}


def main(args):
    batch_size = args.batch_size
    whisper_asr = pipeline(
        "automatic-speech-recognition", model=args.model_id, device=args.device
    )

    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language=args.language, task="transcribe"
        )
    )

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=args.streaming,
        use_auth_token=True,
        cache_dir=os.path.join(base_dir, 'datasets_cache'),
    )

    # Only uncomment for debugging
    dataset = dataset.take(args.max_eval_samples)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    if args.do_bangla_unicode_normalize:
        print("\n\n Doing Unicode Normalization! Make sure you have chosen the Bengali split of your dataset! \n\n")
        dataset = dataset.map(bn_unicode_normalise)
    else:
        dataset = dataset.map(normalise)
            
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    predictions = []
    references = []

    # run streamed inference
    for out in whisper_asr(data(dataset), batch_size=batch_size):
        predictions.append(whisper_norm(out["text"]))
        references.append(out["reference"][0])

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)
    
    cer = cer_metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)

    print(f"\n\n WER: {wer} \n\n")
    print(f"\n\n CER: {cer} \n\n")
    
    evaluate.push_to_hub(
        model_id=args.model_id,
        metric_value=wer,
        metric_type="wer",
        metric_name="WER",
        dataset_name=args.dataset,
        dataset_type=args.dataset,
        dataset_split=args.split,
        dataset_config=args.config,
        task_type="automatic-speech-recognition",
        task_name="Automatic Speech Recognition",
        overwrite=True
    )
    
    evaluate.push_to_hub(
        model_id=args.model_id,
        metric_value=cer,
        metric_type="cer",
        metric_name="CER",
        dataset_name=args.dataset,
        dataset_type=args.dataset,
        dataset_split=args.split,
        dataset_config=args.config,
        task_type="automatic-speech-recognition",
        task_name="Automatic Speech Recognition",
        overwrite=True
    )
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with ???? Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset name to evaluate the `model_id`. Should be loadable with ???? Datasets",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config of the dataset. *E.g.* `'en'` for the English split of Common Voice",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'test'`",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--streaming",
        type=bool,
        default=True,
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--do_bangla_unicode_normalize",
        type=bool,
        default=True,
        help="Choose whether you'd like to perform unicode normalization on your Bengali",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Two letter language code for the transcription language, e.g. use 'en' for English.",
    )
    
    args = parser.parse_args()

    main(args)
