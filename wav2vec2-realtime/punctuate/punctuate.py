# Credit: https://github.com/xashru/punctuation-restoration

import os
import re
import torch

from .models import DeepPunctuation, DeepPunctuationCRF
from .config import *

#language either English(en) or Bangla(bn)
# language = 'en'
language = 'bn'

# model_weigths_path
# model_save_path = os.path.join(os.getcwd(), "punctuate", "xlm-roberta-large-bn.pt")

# tokenizer
# model_name = 'xlm-roberta-large'
# model_name = 'sagorsarker/bangla-bert-base'
model_name = 'ai4bharat/indic-bert'
tokenizer = MODELS[model_name][1].from_pretrained(model_name)
token_style = MODELS[model_name][3]

#sequence length to use when preparing dataset (default 256)
sequence_length = 512

#use crf (conditional random field)
use_crf = False

#hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model
lstm_dim = -1

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if use_crf:
    print("Loading Punctuator Model. WITH CRF..\n" + "*" * 100)
    deep_punctuation = DeepPunctuationCRF(model_name, freeze_bert=False, lstm_dim=lstm_dim)
else:
    print("Loading Punctuator Model. WITHOUT CRF..\n" + "*" * 100)
    deep_punctuation = DeepPunctuation(model_name, freeze_bert=False, lstm_dim=lstm_dim)

deep_punctuation.to(device)
# state_dict = torch.load(model_save_path)

# for key in list(state_dict.keys()):
#     state_dict[key.replace(key, f'bert_lstm.{key}')] = state_dict.pop(key)

# ## strict=False will randomly initialize the missing keys in the locally loaded model
# deep_punctuation.load_state_dict(state_dict, strict=False)
deep_punctuation.eval()


def punctuator(input_text: str):
    text = re.sub(r"[,:\-–.!;?]", '', input_text)
    words_original_case = text.split()
    words = text.lower().split()

    word_pos = 0
    sequence_len = sequence_length
    result = ""
    decode_idx = 0
    punctuation_map = {0: '', 1: ',', 2: '.', 3: '?'}
    if language != 'en':
        punctuation_map[2] = '।'

    while word_pos < len(words):
        x = [TOKEN_IDX[token_style]['START_SEQ']]
        y_mask = [0]

        while len(x) < sequence_len and word_pos < len(words):
            tokens = tokenizer.tokenize(words[word_pos])
            if len(tokens) + len(x) >= sequence_len:
                break
            else:
                for i in range(len(tokens) - 1):
                    x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                    y_mask.append(0)
                x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                y_mask.append(1)
                word_pos += 1
        x.append(TOKEN_IDX[token_style]['END_SEQ'])
        y_mask.append(0)
        if len(x) < sequence_len:
            x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
            y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
        attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

        x = torch.tensor(x).reshape(1,-1)
        y_mask = torch.tensor(y_mask)
        attn_mask = torch.tensor(attn_mask).reshape(1,-1)
        x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)

        with torch.no_grad():
            if use_crf:
                y = torch.zeros(x.shape[0])
                y_predict = deep_punctuation(x, attn_mask, y)
                y_predict = y_predict.view(-1)
            else:
                y_predict = deep_punctuation(x, attn_mask)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx] + punctuation_map[y_predict[i].item()] + ' '
                decode_idx += 1
   
    # print(f'\n\n Punctuated text: {result}\n\n')
    
    return result
