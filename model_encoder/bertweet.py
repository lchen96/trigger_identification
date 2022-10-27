## Encoder BERTweet
import torch
import torch.nn as nn
from torch.nn import functional as F

import transformers
from transformers import AutoModel, AutoTokenizer
transformers.logging.set_verbosity_error()

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pretrain_file = args.bert_tweet
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrain_file)
        self.bertweet = AutoModel.from_pretrained(self.pretrain_file)
        # temp parameters to obtain model device name
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.sent_size = 768

    def forward(self, mes):
        '''Encode textual mesence into embeddings
        mes(list of str, e.g. ['xxx']): strings of messages
        '''
        # extract mesence representation with pre-trained LM
        inputs = self.tokenizer(
            mes, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.dummy_param.device) for k, v in inputs.items()}
        bert_outputs = self.bertweet(**inputs)
        mes_embed = bert_outputs.last_hidden_state[:, 0, :]
        return mes_embed
        