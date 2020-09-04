

import torch
from transformers import BertTokenizer
from bert_model import Regressor

import pandas as pd
import numpy as np
import json
import sys 
import os 

sys.path.append(os.getcwd())

from src.data_tools.cleaning_tools import replace_acronyms
from src.bert_model import parameters as p
from src import data_sources as ds

def predict(model, tokenizer, text_list, device, batch_size=16):

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

    all_scores = []

    for i in range(len(encoded_plus_list) // batch_size + 1):

        this_batch = encoded_plus_list[i * batch_size : (i + 1) * batch_size]

        step_1 = [ele["input_ids"] for ele in this_batch]

        if len(step_1) == 0:
            continue

        input_ids = torch.stack(step_1).squeeze()
        input_ids = input_ids.to(device)

        step_1 = [ele["attention_mask"] for ele in this_batch]
        attention_mask = torch.stack(step_1).squeeze()
        attention_mask = attention_mask.to(device)

        scores = model(
        input_ids=input_ids,
        attention_mask=attention_mask
        )

        scores_numpy = scores.cpu().detach().numpy()

        all_scores.append(scores_numpy)

    all_scores_torch = np.concatenate(all_scores)
    return all_scores_torch


if __name__ == "__main__":

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")


    df_test = pd.read_csv(ds.TEST_FILE_PATH)
    text_list = df_test["text"].tolist()

    tokenizer = BertTokenizer.from_pretrained(p.PRE_TRAINED_MODEL_NAME)

    all_model_scores = []

    for i in range(p.N_PARTS):

        model = Regressor().to(device)
        model.load_state_dict(torch.load(p.NEW_MODEL_NAME.format(i)))

        all_scores_torch = predict(model, tokenizer, text_list, device)

        with open(p.NORM_PARAMS.format(i)) as f:
            norm_params = json.load(f)

        all_scores_original = all_scores_torch * (norm_params["train_max"] - norm_params["train_min"]) + norm_params["train_min"]

        all_model_scores.append(all_scores_original)

    scores_numpy = np.stack(all_model_scores).mean(axis=0)

    df_to_write = pd.DataFrame()
    df_to_write["Id"] = df_test["Id"]
    df_to_write["SalePrice"] = scores_numpy

    df_to_write.to_csv(p.OUTPUT_FILE_PATH, index=False)






