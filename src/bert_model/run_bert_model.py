
import numpy as np
import pandas as pd
from collections import defaultdict

import torch 
from torch import nn 

import transformers
from transformers import BertTokenizer

from bert_model import Regressor

from copy import deepcopy
import json

import sys 
import os 

sys.path.append(os.getcwd())

from src import data_sources as ds
from src.bert_model import parameters as p
from src.data_tools.cleaning_tools import replace_acronyms, set_num_labels


def train_epoch(
  model,
  encoded_plus_list,
  target_list,
  loss_fn,
  optimizer,
  device,
  n_examples,
  batch_size=16
):

  model = model.train()
  losses = []

  for i in range(len(encoded_plus_list) // batch_size + 1):

    optimizer.zero_grad()

    this_batch = encoded_plus_list[i * batch_size : (i + 1) * batch_size]

    step_1 = [ele["input_ids"] for ele in this_batch]

    if len(step_1) == 0:
        continue

    input_ids = torch.stack(step_1).squeeze()
    input_ids = input_ids.to(device)

    step_1 = [ele["attention_mask"] for ele in this_batch]
    attention_mask = torch.stack(step_1).squeeze()
    attention_mask = attention_mask.to(device)


    targets = target_list[i * batch_size : (i+1) * batch_size]

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    loss = loss_fn(outputs, targets)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

  return np.mean(losses)


def calc_sse(outputs, targets, train_min, train_max):

    outputs_1 = outputs * (train_max - train_min) + train_min
    targets_1 = targets * (train_max - train_min) + train_min

    log_diff = outputs_1.log() - targets_1.log()

    sse = (log_diff ** 2).sum()

    return sse.item(), outputs_1, targets_1


def eval_model(
    model,
    encoded_plus_list,
    target_list,
    loss_fn,
    device,
    n_examples,
    train_min,
    train_max,
    batch_size=128
    ):

  model = model.eval()
  losses = []
  correct_predictions = 0
  predictions_list = []
  total_sse = 0

  with torch.no_grad():

    for i in range(len(encoded_plus_list) // batch_size + 1):

      this_batch = encoded_plus_list[i * batch_size : (i + 1) * batch_size]
      step_1 = [ele["input_ids"] for ele in this_batch]

      if len(step_1) == 0:
          continue

      input_ids = torch.stack(step_1).to(device).squeeze()

      step_1 = [ele["attention_mask"] for ele in this_batch]
      attention_mask = torch.stack(step_1).to(device).squeeze()

      this_batch_targets = target_list[i * batch_size : (i+1) * batch_size]


      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )

      loss = loss_fn(outputs, this_batch_targets)
      sse, outputs_original, targets_original = calc_sse(outputs, this_batch_targets, train_min, train_max)
      total_sse += sse 

      losses.append(loss.item())

    rmse = (total_sse / len(target_list))


  return np.mean(losses), rmse



def build_model_and_get_results(encoded_plus_list, 
                                labels_torch,
                                part,
                                n_parts,
                                device,
                                learning_rate,
                                epochs
                                ):

    val_len = len(encoded_plus_list) // n_parts
    
    start = val_len * part 
    end = val_len * (part + 1)

    encoded_plus_val = encoded_plus_list[start:end]
    labels_val = labels_torch[start:end]

    encoded_plus_train = encoded_plus_list[:start] + encoded_plus_list[end:]
    labels_train = torch.cat([labels_torch[:start], labels_torch[end:]])

    model = Regressor().to(device)

    optimizer = transformers.AdamW(model.parameters(),
                                   lr=learning_rate,
                                   correct_bias=False)

    loss_fn = nn.MSELoss().to(device)
    best_loss = np.inf

    train_min = labels_train.min()
    train_max = labels_train.max()

    labels_train = (labels_train - train_min) / (train_max - train_min)
    labels_train.to(device) # this might be redundant

    labels_val = (labels_val - train_min) / (train_max - train_min)
    labels_val.to(device)

    print(f"epochs: {epochs}")

    for epoch in range(epochs):

        print(f"On epoch {epoch + 1}/{epochs}")

        train_loss = train_epoch(
            model,
            encoded_plus_train,
            labels_train,
            loss_fn,
            optimizer,
            device,
            len(encoded_plus_train)
        )

        print(f'Train loss {train_loss}')

        val_loss, rmse = eval_model(
            model,
            encoded_plus_val,
            labels_val,
            loss_fn,
            device,
            len(encoded_plus_val),
            train_min, 
            train_max
        )

        print(f'Val loss {val_loss}')
        print(f'RMSE: {rmse}')
        # print(f"confusion matrix: {cf_matrix}")


        if val_loss < best_loss:

            best_model_dict = deepcopy(model.state_dict())
            best_loss = val_loss

    torch.save(best_model_dict, p.NEW_MODEL_NAME.format(i))

    with open(p.NORM_PARAMS.format(i), "w") as f:
      json.dump({"train_min": train_min.item(), "train_max": train_max.item()}, f)

    return best_loss


if __name__ == "__main__":

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    print("running with bert uncased...")

    df = pd.read_csv(ds.TRAIN_FILE_PATH)

    df = df.sample(frac=1)

    text_list = df["text"].tolist()

    tokenizer = BertTokenizer.from_pretrained(p.PRE_TRAINED_MODEL_NAME)

    labels_torch = torch.from_numpy(np.array(df["target"])).float().to(device)

    encoded_plus_list = [] 

    for text in text_list:
        encoded_plus = tokenizer.encode_plus(
            text,
            max_length=p.MAX_LENGTH,
            truncation=p.TRUNCATION,
            add_special_tokens=p.ADD_SPECIAL_TOKENS,
            return_token_type_ids=p.RETURN_TOKEN_TYPE_IDS,
            pad_to_max_length=p.PAD_TO_MAX_LENGTH,
            return_attention_mask=p.RETURN_ATTENTION_MASK,
            return_tensors=p.RETURN_TENSORS,
        )

        encoded_plus_list.append(encoded_plus)

    best_losses = []

    for i in range(p.N_PARTS):

        best_loss = build_model_and_get_results(encoded_plus_list, 
                                                                labels_torch,
                                                                i,
                                                                p.N_PARTS,
                                                                device,
                                                                p.LEARNING_RATE,
                                                                p.EPOCHS
                                                                )

        best_losses.append(best_loss.item() / len(encoded_plus_list))

    print("accuracy_scores:")
    print(best_losses)
    print(np.mean(best_losses))