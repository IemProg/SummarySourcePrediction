import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import json, csv

import torch
import torch.nn as nn

from utils import *

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

root_path = os.getcwd()
data_path = os.path.join(root_path, "Data")

train_path = os.path.join(data_path, "train_set.json")
test_path = os.path.join(data_path, "test_set.json")
documents_path = os.path.join(data_path, "documents.json")

f = open(train_path)
train_set = json.load(f)
f = open(test_path)
test_set = json.load(f)
f = open(documents_path)
documents = json.load(f)

train_df = pd.read_json(train_path)
test_df = pd.read_json(test_path)
documents_df = pd.read_json(documents_path)

X_train = pd.read_json('x_train.json')
X_val = pd.read_json('x_val.json')
X_test = pd.read_json('x_test.json')

# Set random seed and set device to GPU.
torch.manual_seed(17)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')


# Initialize tokenizer.
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", model_max_len=512, truncation=True, padding=True)

# Set tokenizer hyperparameters.
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

save_np = False

def tokenize_df(X_train, max_len = 512, labels = True):
    y_train = None
    if labels:
        y_train = X_train["label"]
        X_train = X_train.drop("label", axis=1)

    X_train_tokenz = []
    for sentence in X_train["summary"].to_list():
            token = tokenizer.encode_plus(sentence, max_length=max_len, truncation=True, padding='max_length',
                                                   add_special_tokens=True, return_attention_mask=True)
            X_train_tokenz.append(token)
    return y_train, X_train_tokenz

def save_feats(lst):
    output_ids, output_att = [], []
    for item in lst:
        output_ids.append(item['input_ids'])
        output_att.append(item['attention_mask'])
    return output_ids, output_att

if save_np:
    y_train, X_train = tokenize_df(X_train)
    print("\t\t training dset has been tokenized !!!")
    y_val, X_val = tokenize_df(X_val)
    print("\t\t validation dset has been tokenized !!!")

    y_test, X_test = tokenize_df(X_test)
    print("\t\t testing dset has been tokenized !!!")

    # save to npy file
    ids, att = save_feats(X_train)
    np.save('X_train_ids.npy', ids)
    np.save('X_train_att.npy', att)

    # save to npy file
    ids, att = save_feats(X_val)
    np.save('X_val_ids.npy', ids)
    np.save('X_val_att.npy', att)

    # save to npy file
    ids, att = save_feats(X_test)
    np.save('X_test_ids.npy', ids)
    np.save('X_test_att.npy', att)
else:
    y_train = X_train["label"].to_numpy()
    X_train.drop("label", axis = 1)

    y_val = X_val["label"].to_numpy()
    X_val.drop("label", axis = 1)

    y_test = X_test["label"].to_numpy()
    X_test.drop("label", axis = 1)

    X_train_ids = np.load('X_train_ids.npy')
    X_train_att = np.load("X_train_att.npy")

    X_val_ids = np.load('X_val_ids.npy')
    X_val_att = np.load("X_val_att.npy")

    X_test_ids = np.load('X_test_ids.npy')
    X_test_att = np.load("X_test_att.npy")

from torch.utils.data import DataLoader

# X_train = [X_train_ids, X_train_att]
# X_val = [X_val_ids, X_val_att]
# X_test = [X_test_ids, X_test_att]

X_train = X_train_ids
X_val = X_val_ids
X_test = X_test_ids

dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).long(), torch.from_numpy(y_train).long())
train_iter = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4)

dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val).long(), torch.from_numpy(y_val).long())
valid_iter = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4)

dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test).long(), torch.from_numpy(y_test).long())
test_iter = DataLoader(dataset=dataset, batch_size=8, shuffle=False, num_workers=4)

# Model with extra layers on top of RoBERTa
class ROBERTAClassifier(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ROBERTAClassifier, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)

        return x


# Training Function
os.makedirs("output", exist_ok=True)
output_path = os.path.join(os.getcwd(), "output")
# Main training loop
NUM_EPOCHS = 6
steps_per_epoch = len(train_iter)

model = ROBERTAClassifier(0.3)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch*1,
                                            num_training_steps=steps_per_epoch*NUM_EPOCHS)

print("======================= Start pretraining ==============================")
#pretrain(model=model, train_iter=train_iter, valid_iter=valid_iter, optimizer=optimizer, scheduler=scheduler, PAD_INDEX = PAD_INDEX, device = device, num_epochs=NUM_EPOCHS, valid_period = len(train_iter))

NUM_EPOCHS = 12
print("======================= Start training =================================")
optimizer = AdamW(model.parameters(), lr=2e-6)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch*2, num_training_steps=steps_per_epoch*NUM_EPOCHS)

#train(model=model, train_iter=train_iter, valid_iter=valid_iter, optimizer=optimizer, PAD_INDEX = PAD_INDEX, device = device,scheduler=scheduler, num_epochs=NUM_EPOCHS, output_path = output_path, valid_period = len(train_iter))

plt.figure(figsize=(10, 8))
train_loss_list, valid_loss_list, global_steps_list = load_metrics(output_path + '/metric.pkl', device)
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=14)
plt.savefig('loss.png', dpi=500)

# Evaluation Function
model = ROBERTAClassifier()
model = model.to(device)

save_path = output_path + '/model.pkl'
load_checkpoint(save_path, model, device)
evaluate(model, test_iter, PAD_INDEX, device)


## Make preditions to test private dataset
_, eval_tokenz = tokenize_df(test_df, max_len = 512, labels = False)
ids, att = save_feats(eval_tokenz)
np.save('X_eval_ids.npy', ids)
np.save('X_eval_att.npy', att)

X_eval_ids = np.load('X_eval_ids.npy')
X_eval_att = np.load("X_eval_att.npy")


dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_eval_ids).long(), torch.ones(X_eval_ids.shape[0]).long())
eval_iter = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=4)

y_pred = []
y_true = []

model.eval()
with torch.no_grad():
    for batch in eval_iter:
        source = batch[0].to(device)
        mask = (source != PAD_INDEX).type(torch.uint8)
        output = model(source, attention_mask=mask)

        y_pred.extend(torch.argmax(output, axis=-1).tolist())

# Write predictions to a file
with open("submission_RobertaImad_512.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(y_pred):
        csv_out.writerow([i, row])
