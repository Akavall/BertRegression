from transformers import BertModel 
import torch
from torch import nn

import sys 
import os 

sys.path.append(os.getcwd())

from src.bert_model import parameters as p

class Regressor(nn.Module):
  def __init__(self):
    super(Regressor, self).__init__()
    self.bert = BertModel.from_pretrained(p.PRE_TRAINED_MODEL_NAME)

    self.drop = nn.Dropout(p=p.DROPOUT_RATE)
    self.out = nn.Linear(self.bert.config.hidden_size, 1)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    return self.out(pooled_output).squeeze()