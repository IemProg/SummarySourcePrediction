# -*- coding: utf-8 -*-
"""
"""



import tensorflow as tf
import tensorflow.keras.backend as K

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv


# Read The data
training_set = pd.read_json('./train_set.json')
test_set = pd.read_json('./test_set.json')
#original_documents = pd.read_json('./documents.json')


from transformers import LongformerTokenizer, TFLongformerModel
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', model_max_len=256, truncation=True, padding=True)


def build_model(max_len=256):
  input_ids_sum = tf.keras.layers.Input(shape=(max_len,), name='input_ids_sum', dtype=tf.int32)
  mask_sum = tf.keras.layers.Input(shape=(max_len,), name='attention_mask_sum', dtype=tf.int32)
  input_ids_doc = tf.keras.layers.Input(shape=(max_len,), name='input_ids_doc', dtype=tf.int32)
  mask_doc = tf.keras.layers.Input(shape=(max_len,), name='attention_mask_doc', dtype=tf.int32)
  
  longformer1 = TFLongformerModel.from_pretrained('allenai/longformer-base-4096')
  longformer2 = TFLongformerModel.from_pretrained('allenai/longformer-base-4096')

  last_bert_hidden_layer_sum= longformer1([input_ids_sum, mask_sum])[1]
  last_bert_hidden_layer_sum = tf.keras.layers.Flatten()(last_bert_hidden_layer_sum)

  last_bert_hidden_layer_doc= longformer2([input_ids_doc, mask_doc])[1]
  last_bert_hidden_layer_doc = tf.keras.layers.Flatten()(last_bert_hidden_layer_doc)

  last_bert_hidden_layer = tf.concat([last_bert_hidden_layer_sum, last_bert_hidden_layer_doc], 1)

  net = tf.keras.layers.Dense(64, activation='relu')(last_bert_hidden_layer)
  net = tf.keras.layers.Dropout(0.2)(net)
  net = tf.keras.layers.Dense(32, activation='relu')(net)
  net = tf.keras.layers.Dropout(0.2)(net)
  out = tf.keras.layers.Dense(1, activation='sigmoid')(net)

  model = tf.keras.models.Model(inputs=[input_ids_doc, mask_doc, input_ids_sum, mask_sum], outputs=out)
  model.compile(tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

  return model

model = build_model()

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(training_set, test_size=0.25, random_state=42)

X_train_doc = df_train['document'].to_list()
X_test_doc = df_test['document'].to_list()
X_train_sum = df_train['summary'].to_list()
X_test_sum = df_test['summary'].to_list()
y_train = df_train['label'].to_numpy()
y_test = df_test['label'].to_numpy()

X_train_doc = tokenizer(X_train_doc,max_length=256,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

X_train_sum = tokenizer(X_train_sum,max_length=256,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

X_test_doc = tokenizer(X_test_doc,max_length=256,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

X_test_sum = tokenizer(X_test_sum,max_length=256,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

X_train = [X_train_doc['input_ids'].numpy(), X_train_doc['attention_mask'].numpy(),X_train_sum['input_ids'].numpy(), X_train_sum['attention_mask'].numpy()]
X_test = [X_test_doc['input_ids'].numpy(), X_test_doc['attention_mask'].numpy(),X_test_sum['input_ids'].numpy(), X_test_sum['attention_mask'].numpy()]

model.summary()

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
checkpoint_filepath = '/tmp/checkpoint2'
callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True, 
    save_freq='epoch', 
    verbose=1)

train_history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=3,
    batch_size=2,
    callbacks=[callback1, callback2],
    verbose=1
  )


y_pred = model.predict(X_test)

import numpy as np
y_pred_r = np.around(y_pred.squeeze())

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_r)

#Submission
X_submission_doc = tokenizer(test_set['document'].to_list(),max_length=256,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

X_submission_sum = tokenizer(test_set['summary'].to_list(),max_length=256,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

X_submission = [X_submission_doc['input_ids'].numpy(), X_submission_doc['attention_mask'].numpy(), X_submission_sum['input_ids'].numpy(), X_submission_sum['attention_mask'].numpy()]

predictions = model.predict(X_submission)

predictions = np.around(predictions.squeeze())

predictions = list(predictions.astype(int))

with open("submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])
