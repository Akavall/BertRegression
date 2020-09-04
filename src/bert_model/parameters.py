
import os

PRE_TRAINED_MODEL_NAME="bert-base-uncased"
DROPOUT_RATE=0.3
LEARNING_RATE=0.00001
EPOCHS=20
N_PARTS=5
MAX_LENGTH=128
TRUNCATION=True
ADD_SPECIAL_TOKENS=True
RETURN_TOKEN_TYPE_IDS=False
PAD_TO_MAX_LENGTH=True
RETURN_ATTENTION_MASK=True
RETURN_TENSORS="pt"
NEW_MODEL_NAME=os.getcwd() + "/src/bert_model/generated_models/model_{}.pt"
NORM_PARAMS=os.getcwd() + "/src/bert_model/generated_models/norm_params_{}.pt"
OUTPUT_FILE_PATH="house_prices_pred.csv"