import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from datasets import load_dataset, load_metric

import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

print(transformers.__version__)

dataset_base  = load_dataset("./frenchmedmcqa.py")

dataset_train = dataset_base["train"]
print(len(dataset_train))

dataset_val = dataset_base["validation"]
print(len(dataset_val))

dataset_test = dataset_base["test"]
print(len(dataset_test))

metric = load_metric("accuracy")

task = "frenchmedmcqa"
num_labels = 31

batch_size = 4 # XLM-ROBERTA-BASE
# batch_size = 6 # RTX 2080 Ti
# batch_size = 32 # V100
# batch_size = 24
# batch_size = 16

EPOCHS = 10

# model_checkpoint = "dmis-lab/biobert-base-cased-v1.2"
# model_checkpoint = "dmis-lab/biobert-v1.1"
# model_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# model_checkpoint = "camembert-base"
# model_checkpoint = "xlm-roberta-base"
model_checkpoint = "allenai/scibert_scivocab_uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

def preprocess_function(examples):
    return tokenizer(examples["bert_text"], truncation=True, max_length=model.config.max_position_embeddings) # THE ONE

dataset_train = dataset_train.map(preprocess_function, batched=True)
dataset_val   = dataset_val.map(preprocess_function, batched=True)
dataset_test  = dataset_test.map(preprocess_function, batched=True)

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model,
    args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()
