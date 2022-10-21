import numpy as np
import pandas as pd

from datasets import load_dataset

from transformers import BartTokenizer
from transformers import BartForConditionalGeneration, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# MODEL_NAME = 'facebook/bart-large'
MODEL_NAME = 'facebook/bart-base'

EPOCHS = 30

BATCH_SIZE = 3 # BASE
# BATCH_SIZE = 2 # LARGE

dataset  = load_dataset("./frenchmedmcqa.py")

train_dataset = dataset["train"]
print(len(train_dataset))

val_dataset = dataset["validation"]
print(len(val_dataset))

test_dataset = dataset["test"]
print(len(test_dataset))

# Tokenization
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    tokenized_input = tokenizer(batch['source'], padding='max_length', truncation=True, max_length=900)
    tokenized_label = tokenizer(batch['target'], padding='max_length', truncation=True, max_length=900)
    tokenized_input['labels'] = tokenized_label['input_ids']
    return tokenized_input

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=512)
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

# Training
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
output_dir = 'output'

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_accumulation_steps=1, 
    learning_rate=0.0001,
    evaluation_strategy='steps', 
    remove_unused_columns=True, 
    run_name='bart_health_mcqa', 
    logging_steps=1000, 
    eval_steps=1000,
    adam_beta1=0.7,
    adam_beta2=0.9,
    adam_epsilon=1.8e-8 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
trainer.save_model(output_dir + '/model')

###############################################################################

# Evaluation
model_dir = 'output/model'
output_dir = 'output'
model = BartForConditionalGeneration.from_pretrained(model_dir)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred[0], axis=2)
    accuracy = accuracy_score(y_true=labels[0], y_pred=pred[0])
    recall = recall_score(y_true=labels[0], y_pred=pred[0], average='macro')
    precision = precision_score(y_true=labels[0], y_pred=pred[0], average='macro')
    f1 = f1_score(y_true=labels[0], y_pred=pred[0], average='macro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

pred_args = TrainingArguments(
    output_dir=output_dir,
    per_device_eval_batch_size=BATCH_SIZE,
    remove_unused_columns=True,
    eval_accumulation_steps=1
)

print("""
###############################################################################
#                               Validation                                    #
###############################################################################
""")

trainer = Trainer(model=model, args=pred_args, compute_metrics=compute_metrics)

preds, labels, metrics = trainer.predict(val_dataset)
preds_tokens = preds[0].argmax(axis=2)
print(metrics)

decoded_sources = []
for row in val_dataset:
    decoded_sources.append(tokenizer.decode(row['input_ids']))

decoded_preds = [tokenizer.decode(pred) for pred in preds_tokens]
decoded_labels = [tokenizer.decode(label) for label in labels]

output = pd.DataFrame({'Source Text': decoded_sources, 'Target Text': decoded_labels, 'Generated Text': decoded_preds})
output.to_csv(output_dir + "/predictions_dev.csv")

print("""
###############################################################################
#                                  Test                                       #
###############################################################################
""")

trainer = Trainer(model=model, args=pred_args, compute_metrics=compute_metrics)

preds, labels, metrics = trainer.predict(test_dataset)
preds_tokens = preds[0].argmax(axis=2)
print(metrics)

decoded_sources = []
for row in test_dataset:
    decoded_sources.append(tokenizer.decode(row['input_ids']))

decoded_preds = [tokenizer.decode(pred) for pred in preds_tokens]
decoded_labels = [tokenizer.decode(label) for label in labels]

output = pd.DataFrame({'Source Text': decoded_sources, 'Target Text': decoded_labels, 'Generated Text': decoded_preds})
output.to_csv(output_dir + "/predictions_test.csv")
