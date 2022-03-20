from pandas.core.common import random_state
import pandas as pd
import numpy as np
import csv
import random
import transformers

random.seed(1)
from transformers import AutoTokenizer
from datasets import load_metric
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
model_checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'

label_list = [0, 1]
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

lr = 1e-5
batch_size = 16
train_path = './train_set.json'
test_path = './test_set.json'


def tokenize_sentence_preprocess_function(examples):
    return tokenizer(examples["tokens"], truncation=True, padding='max_length', max_length=100,
                     is_split_into_words=False, verbose=False)


def compute_metric_sentence(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return performance_measure(labels, predictions, dp_type='sentence')


def performance_measure(y_true, y_pred, dp_type='token'):
    """
    Compute the performance metrics: TP, FP, FN, TN
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        TP, FP, FN, TN
    """
    performace_dict = dict()
    if dp_type == 'token':
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    performace_dict['TP'] = sum(y_t == y_p == 1 for y_t, y_p in zip(y_true, y_pred))
    performace_dict['FP'] = sum(y_t != y_p for y_t, y_p in zip(y_true, y_pred)
                                if ((y_t == 0) and (y_p == 1)))
    performace_dict['FN'] = sum(((y_t != 0) and (y_p == 0))
                                for y_t, y_p in zip(y_true, y_pred))
    performace_dict['TN'] = sum((y_t == y_p == 0)
                                for y_t, y_p in zip(y_true, y_pred))

    precision = performace_dict['TP'] / (performace_dict['TP'] + performace_dict['FP'])
    recall = performace_dict['TP'] / (performace_dict['TP'] + performace_dict['FN'])
    accuracy = (performace_dict['TP'] + performace_dict['TN']) / (performace_dict['TP'] + performace_dict['TN']
                                                                  + performace_dict['FN'] + performace_dict['FP'])
    f1_score = 2 * ((precision * recall) / (precision + recall))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "accuracy": accuracy,
    }
    # return performace_dict['TP'], performace_dict['FP'], performace_dict['FN'], performace_dict['TN']


def main():
    # Read The data
    training_set = pd.read_json(train_path)
    test_set = pd.read_json(test_path)

    training_set = training_set.rename(columns={'summary': 'tokens'})
    test_set = test_set.rename(columns={'summary': 'tokens'})
    test_set['label'] = 1

    dataset = Dataset.from_pandas(training_set).train_test_split(train_size=0.75)
    dataset['validation'] = dataset['test']
    dataset['test'] = Dataset.from_pandas(test_set)

    tokenized_datasets = dataset.map(tokenize_sentence_preprocess_function, batched=True, num_proc=1,
                                     keep_in_memory=True)

    print(len(tokenized_datasets['train']['input_ids'][0]))
    print(len(tokenized_datasets['validation']['input_ids'][0]))
    print(len(tokenized_datasets['test']['input_ids'][0]))

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(label_list)).to(device)

    args = TrainingArguments(
        # f"sentence_{model_name}-finetuned-{task}",
        'model_roberta',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        metric_for_best_model='accuracy',
        # push_to_hub=True,  # True to save the model on hub
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metric_sentence
    )

    trainer.train()
    trainer.evaluate(tokenized_datasets['validation'])
    temp = trainer.predict(tokenized_datasets['test'])
    print(temp)
    predictions = np.argmax(temp[0], axis=1)

    with open("submission_deberta_got.csv", "w") as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(['id', 'label'])
        for i, row in enumerate(predictions):
            csv_out.writerow([i, row])


if __name__ == "__main__":
    main()