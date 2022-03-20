import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import json, csv

from utils import *
import torch

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(train_df, train_df["label"], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, X_train["label"], test_size=0.2, random_state=42)

os.makedirs("saved_data_tokenz", exist_ok=True)
output_path = os.path.join(os.getcwd(), "saved_data_tokenz")

save_np = False
if save_np:
    path = os.path.join(output_path, "X_train.json")
    X_train.to_json(path)
    path = os.path.join(output_path, "X_val.json")
    X_val.to_json(path)
    path = os.path.join(output_path, "X_test.json")
    X_test.to_json(path)
else:
    path = os.path.join(output_path, "X_train.json")
    X_train = pd.read_json(path)
    path = os.path.join(output_path, "X_val.json")
    X_val = pd.read_json(path)
    path = os.path.join(output_path, "X_test.json")
    X_test = pd.read_json(path)

# Set random seed and set device to GPU.
torch.manual_seed(17)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')

output_path = os.path.join(os.getcwd(), "saved_data_tokenz")

if save_np:
    y_train, X_train_sum = tokenize_df(tokenizer, X_train, max_len = 256, col = "summary")
    y_train, X_train_doc = tokenize_df(tokenizer, X_train, max_len = 256, col = "document")
    print("\t\t training dset has been tokenized !!!")
    y_val, X_val_sum = tokenize_df(tokenizer, X_val, max_len = 256, col = "summary")
    y_val, X_val_doc = tokenize_df(tokenizer, X_val, max_len = 256, col = "document")
    print("\t\t validation dset has been tokenized !!!")
    y_test, X_test_sum = tokenize_df(tokenizer, X_test, max_len = 256, col = "summary")
    y_test, X_test_doc = tokenize_df(tokenizer, X_test, max_len = 256, col = "document")
    print("\t\t testing dset has been tokenized !!!")

    # save to npy file

    ids, att = save_feats(X_train_sum)
    save_path = os.path.join(output_path, 'X_train_sum_ids.npy')
    np.save(save_path, ids)
    save_path = os.path.join(output_path, 'X_train_sum_att.npy')
    np.save(save_path, att)

    ids, att = save_feats(X_train_doc)
    save_path = os.path.join(output_path, 'X_train_doc_ids.npy')
    np.save(save_path, ids)
    save_path = os.path.join(output_path, 'X_train_doc_att.npy')
    np.save(save_path, att)

    # save to npy file
    ids, att = save_feats(X_val_sum)
    save_path = os.path.join(output_path, 'X_val_sum_ids.npy')
    np.save(save_path, ids)
    save_path = os.path.join(output_path, 'X_val_sum_att.npy')
    np.save(save_path, att)

    ids, att = save_feats(X_val_doc)
    save_path = os.path.join(output_path, 'X_val_doc_ids.npy')
    np.save(save_path, ids)
    save_path = os.path.join(output_path, 'X_val_doc_att.npy')
    np.save(save_path, att)

    # save to npy file
    ids, att = save_feats(X_test_sum)
    save_path = os.path.join(output_path, 'X_test_sum_ids.npy')
    np.save(save_path, ids)
    save_path = os.path.join(output_path, 'X_test_sum_att.npy')
    np.save(save_path, att)

    ids, att = save_feats(X_test_doc)
    save_path = os.path.join(output_path, 'X_test_doc_ids.npy')
    np.save(save_path, ids)
    save_path = os.path.join(output_path, 'X_test_doc_att.npy')
    np.save(save_path, att)


cols = ['text_length',
       'text_proc_length', 'text_proc_length_to_text_length',
       'text_lines_count', 'text_word_count', 'text_proc_word_count',
       'text_proc_word_count_to_text_word_count', 'text_mean_word_length',
       'text_proc_mean_word_length',
       'text_proc_mean_word_length_to_text_mean_word_length',
       'text_proc_alphas_count', 'text_alphas_count',
       'text_proc_alphas_count_to_text_alphas_count', 'text_alphas_percent',
       'text_non_alphas_count', 'text_digits_count', 'text_emoji_count',
       'text_emoji_percent', 'text_has_emoji', 'text_upper_letter_count',
       'text_upper_letter_count_to_word_count',
       'text_upper_letter_count_to_length', 'text_urls_count',
       'text_urls_count_to_words_count', 'text_hashtags_count',
       'text_hashtags_count_to_words_count', 'text_usertag_count',
       'text_usertags_count_to_words_count', 'text_punctuation_count',
       'text_proc_punctuation_count', 'text_punctuation_rate',
       'text_punctuation_to_alpha', 'text_proc_punctuation_to_alpha',
       'text_punctuation_unique_count', 'text_proc_count_!',
       'text_proc_count_?']

def merge_doc_summary_feats(ext_train_df_summary, ext_train_df_document, cols= cols):
    ext_train_df_merged = ext_train_df_summary.copy()
    new_df = pd.DataFrame()
    new_cols = []
    for col in cols:
        col_name = col +"_doc"
        ext_train_df_merged[col_name] = ext_train_df_document[col]

        # Use the difference between the original the summary
        name = "ratio_" + col
        ext_train_df_merged[name] = ext_train_df_merged[col] / ext_train_df_merged[col_name]
        new_df[col] = ext_train_df_merged[col] / ext_train_df_merged[col_name]

    return ext_train_df_merged, new_df

def extend_clean_df(df1):
    ext_train_df_summary = add_text_features(df1, "summary")
    ext_train_df_document = add_text_features(df1, "document")
    ext_train_df_merged, new_df = merge_doc_summary_feats(ext_train_df_summary, ext_train_df_document)
    #new_df["label"] = ext_train_df_merged["label"]

    return ext_train_df_merged

ext_train_df_merged = extend_clean_df(X_train)
train_labels = ext_train_df_merged["label"]
ext_train_df_merged = ext_train_df_merged.drop(["label", 'document', 'summary', 'text', 'text_proc'] , axis = 1)
ext_train_df_merged = ext_train_df_merged.replace([np.inf, -np.inf], np.nan)

ext_val_df_merged = extend_clean_df(X_val)
val_labels = ext_val_df_merged["label"]
ext_val_df_merged = ext_val_df_merged.drop(["label", 'document', 'summary', 'text', 'text_proc'] , axis = 1)
ext_val_df_merged = ext_val_df_merged.replace([np.inf, -np.inf], np.nan)

ext_test_df_merged = extend_clean_df(X_test)
test_labels = ext_test_df_merged["label"]
ext_test_df_merged = ext_test_df_merged.drop(["label", 'document', 'summary', 'text', 'text_proc'] , axis = 1)
ext_test_df_merged = ext_test_df_merged.replace([np.inf, -np.inf], np.nan)


numeric_features = ext_test_df_merged.columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor)])

# Impute missing values
ext_train_df_merged = clf.fit_transform(ext_train_df_merged)
ext_val_df_merged = clf.fit_transform(ext_val_df_merged)
ext_test_df_merged = clf.fit_transform(ext_test_df_merged)

y_train = y_train.to_numpy()
X_train.drop("label", axis = 1)

y_val = y_val.to_numpy()
X_val.drop("label", axis = 1)

y_test = y_test.to_numpy()
X_test.drop("label", axis = 1)

X_train_sum_ids = np.load(output_path +"/"+ 'X_train_sum_ids.npy')
X_train_sum_att = np.load(output_path +"/"+ "X_train_sum_att.npy")
X_train_doc_ids = np.load(output_path +"/"+ 'X_train_doc_ids.npy')
X_train_doc_att = np.load(output_path +"/"+ "X_train_doc_att.npy")

X_val_sum_ids = np.load(output_path +"/"+ 'X_val_sum_ids.npy')
X_val_sum_att = np.load(output_path +"/"+ "X_val_sum_att.npy")
X_val_doc_ids = np.load(output_path +"/"+ 'X_val_doc_ids.npy')
X_val_doc_att = np.load(output_path +"/"+ "X_val_doc_att.npy")

X_test_sum_ids = np.load(output_path +"/"+ 'X_test_sum_ids.npy')
X_test_sum_att = np.load(output_path +"/"+ "X_test_sum_att.npy")
X_test_doc_ids = np.load(output_path +"/"+ 'X_test_doc_ids.npy')
X_test_doc_att = np.load(output_path +"/"+ "X_test_doc_att.npy")

from torch.utils.data import DataLoader

X_train = X_train_sum_ids.astype(np.float)
X_val =  X_val_sum_ids.astype(np.float)
X_test =  X_test_sum_ids.astype(np.float)

print("\t\t X_train: {}, {}, Y_train: {}".format(X_train[0].shape, X_train[1].shape, y_train.shape))
print("\t\t X_val: {}, {},  y_val: {}".format(X_val[0].shape, X_val[1].shape, y_val.shape))
print("\t\t x_test: {}, {}, y_test: {}".format(X_test[0].shape, X_test[1].shape, y_test.shape))

dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).long(), torch.tensor(X_train_doc_ids.astype(np.float)).long(),
                                            torch.tensor(ext_train_df_merged.astype(np.float)).float(), torch.tensor(y_train).long())
train_iter = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4)

dataset = torch.utils.data.TensorDataset(torch.tensor(X_val).long(), torch.tensor(X_val_doc_ids.astype(np.float)).long(),
                                            torch.tensor(ext_val_df_merged.astype(np.float)).float(), torch.tensor(y_val).long())
valid_iter = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4)

dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).long(), torch.tensor(X_test_doc_ids.astype(np.float)).long(),
                                            torch.tensor(ext_test_df_merged.astype(np.float)).float(), torch.tensor(y_test).long())
test_iter = DataLoader(dataset=dataset, batch_size=8, shuffle=False, num_workers=4)

# Initialize tokenizer.
#tokenizer = RobertaTokenizer.from_pretrained("roberta-base", model_max_len=256, truncation=True, padding=True)

# Set tokenizer hyperparameters.
MAX_SEQ_LEN = 256
BATCH_SIZE = 15
#PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
#UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
PAD_INDEX = 1
UNK_INDEX = 3
print("\t\t PAD_INDEX: ", PAD_INDEX)
print("\t\t UNK_INDEX: ", UNK_INDEX)

# Model with extra layers on top of RoBERTa
class ROBERTAClassifier(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ROBERTAClassifier, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)

        self.roberta2 = RobertaModel.from_pretrained('roberta-base')
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(768, 64)
        self.bn2 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)

        # Features
        self.fc1 = nn.Linear(105, 128)
        self.d11 = torch.nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.d12 = torch.nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        # head classifier
        self.head = torch.nn.Linear(64 * 3, 2)

        # Features

    def forward(self, input_ids, attention_mask, input_ids2, attention_mask2, features):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)

        _, x1 = self.roberta2(input_ids=input_ids2, attention_mask=attention_mask2, return_dict=False)
        x1 = self.d2(x1)
        x1 = self.l2(x1)
        x1 = self.bn2(x1)
        x1 = torch.nn.Tanh()(x1)
        x1 = self.d2(x1)


        # features
        x2 = self.fc1(features)
        x2 = self.relu(x2)
        x2 = self.d11(x2)
        x2 = self.fc2(x2)
        x2 = self.relu(x2)
        x2 = self.d12(x2)

        # head
        x = self.head(torch.cat((x, x1.view(x.shape), x2.view(x.shape)), dim=1))
        return x

# Training Function
os.makedirs("output", exist_ok=True)
output_path = os.path.join(os.getcwd(), "output")
# Main training loop
NUM_EPOCHS = 30
steps_per_epoch = len(train_iter)

model = ROBERTAClassifier(0.3)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch*1,
                                            num_training_steps=steps_per_epoch*NUM_EPOCHS)

def pretrain(model, optimizer, train_iter, valid_iter, valid_period, PAD_INDEX, device, scheduler = None, num_epochs = 5):
    # Pretrain linear layers, do not train bert
    for param in model.roberta.parameters():
        param.requires_grad = False

    for param in model.roberta2.parameters():
        param.requires_grad = False

    model.train()

    # Initialize losses and loss histories
    train_loss = 0.0
    valid_loss = 0.0
    global_step = 0

    # Train loop
    for epoch in range(num_epochs):
        for batch in train_iter:
            source_sum, source_doc, feats, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            mask = (source_sum != PAD_INDEX).type(torch.uint8).to(device)
            mask2 = (source_doc != PAD_INDEX).type(torch.uint8).to(device)
            y_pred = model(input_ids=source_sum, attention_mask=mask, input_ids2=source_doc, attention_mask2=mask2, features= feats)
            loss = torch.nn.CrossEntropyLoss()(y_pred, target)

            L1_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    L1_reg = L1_reg + torch.norm(param, 1)

            loss = loss + 10e-4 * L1_reg

            loss.backward()

            # Optimizer and scheduler step
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            # Update train loss and global step
            train_loss += loss.item()
            global_step += 1

            # Validation loop. Save progress and evaluate model performance.
            if global_step % valid_period == 0:
                model.eval()

                with torch.no_grad():
                    for batch in valid_iter:
                        source_sum, source_doc, feats, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                        mask = (source_sum != PAD_INDEX).type(torch.uint8).to(device)
                        mask2 = (source_doc != PAD_INDEX).type(torch.uint8).to(device)
                        y_pred = model(input_ids=source_sum, attention_mask=mask, input_ids2=source_doc, attention_mask2=mask2, features= feats)
                        loss = torch.nn.CrossEntropyLoss()(y_pred, target)

                        L1_reg = torch.tensor(0., requires_grad=True)
                        for name, param in model.named_parameters():
                            if 'weight' in name:
                                L1_reg = L1_reg + torch.norm(param, 1)

                        loss = loss + 10e-4 * L1_reg

                        valid_loss += loss.item()

                # Store train and validation loss history
                train_loss = train_loss / valid_period
                valid_loss = valid_loss / len(valid_iter)

                model.train()

                # print summary
                print('Epoch [{}/{}], global step [{}/{}], PT Loss: {:.4f}, Val Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
                              train_loss, valid_loss))

                train_loss = 0.0
                valid_loss = 0.0

    # Set bert parameters back to trainable
    for param in model.roberta.parameters():
        param.requires_grad = True

    for param in model.roberta2.parameters():
        param.requires_grad = True

    print('Pre-training done!')

def train(model, optimizer, train_iter, valid_iter, output_path, valid_period, PAD_INDEX, device, scheduler = None, num_epochs = 5):
    # Initialize losses and loss histories
    train_loss = 0.0
    valid_loss = 0.0
    train_loss_list = []
    valid_loss_list = []
    best_valid_loss = float('Inf')

    global_step = 0
    global_steps_list = []

    model.train()

    # Train loop
    for epoch in range(num_epochs):
        for batch in train_iter:
            source_sum, source_doc, feats, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            mask = (source_sum != PAD_INDEX).type(torch.uint8).to(device)
            mask2 = (source_doc != PAD_INDEX).type(torch.uint8).to(device)
            y_pred = model(input_ids=source_sum, attention_mask=mask, input_ids2=source_doc, attention_mask2=mask2, features=feats)

            loss = torch.nn.CrossEntropyLoss()(y_pred, target)
            L1_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    L1_reg = L1_reg + torch.norm(param, 1)

            loss = loss + 10e-4 * L1_reg

            loss.backward()

            # Optimizer and scheduler step
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            # Update train loss and global step
            train_loss += loss.item()
            global_step += 1

            # Validation loop. Save progress and evaluate model performance.
            if global_step % valid_period == 0:
                model.eval()

                with torch.no_grad():
                    for batch in valid_iter:
                        source_sum, source_doc, feats, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                        mask = (source_sum != PAD_INDEX).type(torch.uint8).to(device)
                        mask2 = (source_doc != PAD_INDEX).type(torch.uint8).to(device)
                        y_pred = model(input_ids=source_sum, attention_mask=mask, input_ids2=source_doc, attention_mask2=mask2, features=feats)

                        loss = torch.nn.CrossEntropyLoss()(y_pred, target)
                        #loss = output[0]
                        L1_reg = torch.tensor(0., requires_grad=True)
                        for name, param in model.named_parameters():
                            if 'weight' in name:
                                L1_reg = L1_reg + torch.norm(param, 1)

                        loss = loss + 10e-4 * L1_reg

                        valid_loss += loss.item()

                # Store train and validation loss history
                train_loss = train_loss / valid_period
                valid_loss = valid_loss / len(valid_iter)
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                global_steps_list.append(global_step)

                # print summary
                print('Epoch [{}/{}], global step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
                              train_loss, valid_loss))

                # checkpoint
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    save_path = output_path + '/model_all.pkl'
                    save_checkpoint(save_path, model, best_valid_loss)
                    save_path = output_path + '/metric_all.pkl'
                    save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list)

                train_loss = 0.0
                valid_loss = 0.0
                model.train()
    save_path = output_path + '/metric_all.pkl'
    save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list)
    print('Training done!')


print("======================= Start pretraining ==============================")
pretrain(model=model, train_iter=train_iter, valid_iter=valid_iter, optimizer=optimizer, scheduler=scheduler, PAD_INDEX = PAD_INDEX, device = device, num_epochs=NUM_EPOCHS, valid_period = len(train_iter))

NUM_EPOCHS = 15
print("======================= Start training =================================")
optimizer = AdamW(model.parameters(), lr=2e-6)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch*2, num_training_steps=steps_per_epoch*NUM_EPOCHS)

train(model=model, train_iter=train_iter, valid_iter=valid_iter, optimizer=optimizer, PAD_INDEX = PAD_INDEX, device = device, scheduler=scheduler, num_epochs=NUM_EPOCHS, output_path = output_path, valid_period = len(train_iter))

plt.figure(figsize=(10, 8))
train_loss_list, valid_loss_list, global_steps_list = load_metrics(output_path + '/metric_all.pkl', device)
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=14)
plt.savefig('loss_all.png', dpi=500)

# Evaluation Function
def evaluate2(model, test_loader, PAD_INDEX, device):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            source_sum, source_doc, feats, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            mask = (source_sum != PAD_INDEX).type(torch.uint8).to(device)
            mask2 = (source_doc != PAD_INDEX).type(torch.uint8).to(device)
            output = model(input_ids=source_sum, attention_mask=mask, input_ids2=source_doc, attention_mask2=mask2, features=feats)

            y_pred.extend(torch.argmax(output, axis=-1).tolist())
            y_true.extend(target.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()

    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])


model = ROBERTAClassifier()
model = model.to(device)

save_path = output_path + '/model_all.pkl'
load_checkpoint(save_path, model, device)
evaluate2(model, test_iter, PAD_INDEX, device)


## Make preditions to test private dataset
output_path = os.path.join(os.getcwd(), "saved_data_tokenz")

test_path = os.path.join(data_path, "test_set.json")
X_eval = pd.read_json(test_path)

X_eval_ids_sum = np.load(output_path + "/" + 'X_eval_sum_ids_256.npy')
X_eval_att_sum = np.load(output_path + "/" + "X_eval_sum_att_256.npy")
X_eval_ids_doc = np.load(output_path + "/" + 'X_eval_doc_ids_256.npy')
X_eval_att_doc = np.load(output_path + "/" + "X_eval_doc_att_256.npy")


ext_eval_df_merged = extend_clean_df(X_eval)
ext_eval_df_merged = ext_eval_df_merged.drop(['document', 'summary', 'text', 'text_proc'] , axis = 1)
ext_eval_df_merged = ext_eval_df_merged.replace([np.inf, -np.inf], np.nan)

ext_eval_df_merged = clf.fit_transform(ext_eval_df_merged)

dataset = torch.utils.data.TensorDataset(torch.tensor(X_eval_ids_sum).long(), torch.tensor(X_eval_ids_doc).long(),
                                        torch.tensor(ext_eval_df_merged.astype(np.float)).float(), torch.ones(X_eval_ids_sum.shape[0]).long())
eval_iter = DataLoader(dataset=dataset, batch_size=8, shuffle=False, num_workers=4)

y_pred = []
y_true = []

model.eval()
with torch.no_grad():
    for batch in eval_iter:
        source_sum, source_doc, feats, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        mask = (source_sum != PAD_INDEX).type(torch.uint8).to(device)
        mask2 = (source_doc != PAD_INDEX).type(torch.uint8).to(device)
        output = model(input_ids=source_sum, attention_mask=mask, input_ids2=source_doc, attention_mask2=mask2, features=feats)

        y_pred.extend(torch.argmax(output, axis=-1).tolist())

# Write predictions to a file
with open("submission_RobertaImad_all_256_withRL.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(y_pred):
        csv_out.writerow([i, row])
