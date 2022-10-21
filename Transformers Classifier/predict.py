from collections import Counter

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

from datasets import load_dataset
from transformers import TextClassificationPipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path_model = "scibert_scivocab_uncased-finetuned-frenchmedmcqa/checkpoint-272"

tokenizer = AutoTokenizer.from_pretrained(path_model)
model = AutoModelForSequenceClassification.from_pretrained(path_model)

dataset_base  = load_dataset("./frenchmedmcqa.py")

dataset_val = dataset_base["validation"]
print(len(dataset_val))

dataset_test = dataset_base["test"]
print(len(dataset_test))

pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=0)

LETTERS = ["c","a","e","d","b","be","ae","bc","bd","ab","de","cd","ac","ad","ce","bce","abc","cde","bcd","ace","ade","abe","acd","bde","abd","abde","abcd","bcde","abce","acde","abcde"]

def compute_accuracy_exact_match(preds, refs):
    exact_score = []
    for p, r in zip(preds, refs):
        exact_score.append(sorted(p) == sorted(r))
    return sum(exact_score) / len(exact_score)

def compute_accuracy_hamming(preds, refs):
    corrects = [True for p in preds if p in refs]
    corrects = sum(corrects)
    total_refs = len(list(set(preds + refs)))
    return corrects / total_refs

for subset_name, current_dataset in [("dev", dataset_val), ("test", dataset_test)]:

    print(f"### {subset_name} ###")

    y_pred = []
    y_true = []

    hamming_scores = []

    for e in current_dataset:

        # Identifier
        identifier = int(e["id"])

        # Prediction
        # res = pipeline(e["roberta_text_no_ctx"], truncation=True, max_length=model.config.max_position_embeddings)
        # res = pipeline(e["roberta_text"], truncation=True, max_length=model.config.max_position_embeddings)
        
        # res = pipeline(e["roberta_text_no_ctx"], truncation=True, max_length=512)
        # res = pipeline(e["roberta_text"], truncation=True, max_length=512)

        # res = pipeline(e["bert_text_no_ctx"], truncation=True, max_length=model.config.max_position_embeddings)
        res = pipeline(e["bert_text"], truncation=True, max_length=model.config.max_position_embeddings)

        pred = int(res[0]["label"].split("_")[-1])
        pred = LETTERS[pred]
        y_pred.append(pred)
        splitted_pred = sorted(list(pred))

        # Reference
        true_label = LETTERS[e["label"]]
        y_true.append(true_label)
        splitted_true_label = sorted(list(true_label))

        # Compute hamming score
        score = compute_accuracy_hamming(splitted_pred, splitted_true_label)
        hamming_scores.append(score)

    print(">> hamming_scores")
    hamming_score = sum(hamming_scores) / len(hamming_scores)
    print(hamming_score)

    print(">> exact_match")
    exact_match = compute_accuracy_exact_match(y_true, y_pred)
    print(exact_match)
    
    print()
