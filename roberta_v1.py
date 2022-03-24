import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv


# Read The data
training_set = pd.read_json('./train_set.json')
test_set = pd.read_json('./test_set.json')
#original_documents = pd.read_json('./documents.json')


from transformers import RobertaTokenizer, TFRobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', model_max_len=512, truncation=True, padding=True)

def build_model(max_len=256):
  input_ids = tf.keras.layers.Input(shape=(max_len,), name='input_ids', dtype=tf.int32)
  mask = tf.keras.layers.Input(shape=(max_len,), name='attention_mask', dtype=tf.int32)
  
  roberta = TFRobertaModel.from_pretrained('roberta-base')

  last_bert_hidden_layer = roberta([input_ids, mask])[1]
  last_bert_hidden_layer = tf.keras.layers.Flatten()(last_bert_hidden_layer)

  net = tf.keras.layers.Dense(64, activation='relu')(last_bert_hidden_layer)
  net = tf.keras.layers.Dropout(0.2)(net)
  net = tf.keras.layers.Dense(32, activation='relu')(net)
  net = tf.keras.layers.Dropout(0.2)(net)
  out = tf.keras.layers.Dense(1, activation='sigmoid')(net)

  model = tf.keras.models.Model(inputs=[input_ids, mask], outputs=out)
  model.compile(tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

  return model

model = build_model()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_set['summary'].to_list(), training_set['label'].to_numpy(), test_size=0.05, random_state=42)

X_train = tokenizer(X_train,max_length=256,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

X_test = tokenizer(X_test,max_length=256,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

X_train = [X_train['input_ids'].numpy(), X_train['attention_mask'].numpy()]
X_test = [X_test['input_ids'].numpy(), X_test['attention_mask'].numpy()]

model.summary()

import tensorflow as tf
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
    epochs=10,
    batch_size=8,
    verbose=1,
    callbacks=[callback1, callback2]
  )

y_pred = model.predict(X_test)

import numpy as np
y_pred_r = np.around(y_pred.squeeze())

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_r)

X_submission = tokenizer(test_set['summary'].to_list(),max_length=256,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

X_submission = [X_submission['input_ids'].numpy(), X_submission['attention_mask'].numpy()]

predictions = model.predict(X_submission)

predictions = np.around(predictions.squeeze())

predictions = list(predictions.astype(int))

with open("submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])
