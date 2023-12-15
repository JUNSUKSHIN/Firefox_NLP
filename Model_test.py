from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import gluonnlp as nlp
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

import torch
from torch import nn

# 장치 설정
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

# 모델 및 토크나이저 초기화
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)