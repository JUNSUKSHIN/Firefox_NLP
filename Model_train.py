
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
import pandas as pd
import multiprocessing

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from kobert_tokenizer import KoBERTTokenizer

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type) 

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                pad, pair):
        transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        #transform = nlp.data.BERTSentenceTransform(
        #    tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

if __name__ == '__main__':
    multiprocessing.freeze_support()



    class BERTSentenceTransform:
        r"""BERT style data transformation.

        Parameters
        ----------
        tokenizer : BERTTokenizer.
            Tokenizer for the sentences.
        max_seq_length : int.
            Maximum sequence length of the sentences.
        pad : bool, default True
            Whether to pad the sentences to maximum length.
        pair : bool, default True
            Whether to transform sentences or sentence pairs.
        """

        def __init__(self, tokenizer, max_seq_length,vocab, pad=True, pair=True):
            self._tokenizer = tokenizer
            self._max_seq_length = max_seq_length
            self._pad = pad
            self._pair = pair
            self._vocab = vocab

        def __call__(self, line):
            
            text_a = line[0]
            if self._pair:
                assert len(line) == 2
                text_b = line[1]

            tokens_a = self._tokenizer.tokenize(text_a)
            tokens_b = None

            if self._pair:
                tokens_b = self._tokenizer(text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b,
                                        self._max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > self._max_seq_length - 2:
                    tokens_a = tokens_a[0:(self._max_seq_length - 2)]

            # The embedding vectors for `type=0` and `type=1` were learned during
            # pre-training and are added to the wordpiece embedding vector
            # (and position vector). This is not *strictly* necessary since
            # the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.

            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            #vocab = self._tokenizer.vocab
            vocab = self._vocab
            tokens = []
            tokens.append(vocab.cls_token)
            tokens.extend(tokens_a)
            tokens.append(vocab.sep_token)
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens.extend(tokens_b)
                tokens.append(vocab.sep_token)
                segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

            input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

            # The valid length of sentences. Only real  tokens are attended to.
            valid_length = len(input_ids)

            if self._pad:
                # Zero-pad up to the sequence length.
                padding_length = self._max_seq_length - valid_length
                # use padding tokens for the rest
                input_ids.extend([vocab[vocab.padding_token]] * padding_length)
                segment_ids.extend([0] * padding_length)

            return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
                np.array(segment_ids, dtype='int32')
        
    

    def get_kobert_model(model_path, vocab_file, ctx="cpu"):
        bertmodel = BertModel.from_pretrained(model_path)
        device = torch.device(ctx)
        bertmodel.to(device)
        bertmodel.eval()
        vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                            padding_token='[PAD]')
        return bertmodel, vocab_b_obj

    max_len = 100
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 5
    max_grad_norm = 1
    log_interval = 200
    learning_rate =  5e-5

    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')

    dataset_train = []
    for q, label in zip(train_df['sentence'], train_df['target']):
        data = []
        data.append(q)
        data.append(str(label))

        dataset_train.append(data)

    dataset_test  = []
    for q, label in zip(test_df['sentence'], train_df['target']):
        data = []
        data.append(q)
        data.append(str(label))

        dataset_test .append(data)

    class BERTClassifier(nn.Module):
        def __init__(self,
                    bert,
                    hidden_size = 768,
                    num_classes = 4, 
                    dr_rate = None,
                    params = None):
            super(BERTClassifier, self).__init__()
            self.bert = bert
            self.dr_rate = dr_rate

            self.classifier = nn.Linear(hidden_size , num_classes)
            if dr_rate:
                self.dropout = nn.Dropout(p = dr_rate)

        def gen_attention_mask(self, token_ids, valid_length):
            attention_mask = torch.zeros_like(token_ids)
            for i, v in enumerate(valid_length):
                attention_mask[i][:v] = 1
            return attention_mask.float()

        def forward(self, token_ids, valid_length, segment_ids):
            attention_mask = self.gen_attention_mask(token_ids, valid_length)

            _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict = False)
            if self.dr_rate:
                out = self.dropout(pooler)
            return self.classifier(out)
        
    model = BERTClassifier(bertmodel,  dr_rate = 0.5).to(device)

    data_train = BERTDataset(dataset_train, 0, 1, tokenizer, vocab, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tokenizer, vocab, max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size = batch_size)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size = batch_size)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss() # 다중분류를 위한 loss function

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = t_total)

    def calc_accuracy(X,Y):
        max_vals, max_indices = torch.max(X, 1)
        train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
        return train_acc

    train_history = []
    test_history = []
    loss_history = []

    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)

            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
                train_history.append(train_acc / (batch_id+1))
                loss_history.append(loss.data.cpu().numpy())
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
        test_history.append(test_acc / (batch_id+1))