import numpy as np
import pandas as pd

# VERSION = f"V3" # SBERT + wiki + with multiple + answers for SBERT + BART-BASE
# VERSION = f"V4" # BART-LARGE - Not aligned
# VERSION = f"V5 - wiki bm25"
# VERSION = f"V6 - no ctx"
# VERSION = f"V7 - output_hal_ctx" # HAL SBERT 
# VERSION = f"V8 - output_hal_ctx_bm25" # HAL BM25 

# ------------------ BARThez ------------------

VERSION = f"V1.0 - HAL BM25 BARThez" # HAL BM25 

LETTERS = ["A","B","C","D","E"]

def compute_accuracy_exact_match(preds, refs):
    exact_score = []
    for p, r in zip(preds, refs):
        exact_score.append(sorted(list(set(list(p)))) == sorted(list(set(list(r)))))
    return sum(exact_score) / len(exact_score)

def compute_accuracy_hamming(preds, refs):

    preds = list(set(list(preds)))
    refs = list(set(list(refs)))

    corrects = [True for p in preds if p in refs]
    corrects = sum(corrects)
    total_refs = len(list(set(preds + refs)))
    return corrects / total_refs

def transform(path_csv):

    y_true = []
    y_pred = []

    hamming_scores = []

    df = pd.read_csv(path_csv, sep=",")

    for index, row in df.iterrows():

        def clean(text):
            text = text.replace("<pad>","").replace("</s>","").replace("<s>","").replace("+","").replace(" ","").upper()
            return list(sorted([t for t in text if t in LETTERS]))
        
        # Target
        splitted_true_label = clean(row['Target Text'])
        y_true.append("".join(splitted_true_label))

        # Generated
        splitted_pred = clean(row['Generated Text'])
        y_pred.append("".join(splitted_pred))

        # Compute hamming score
        score = compute_accuracy_hamming(splitted_pred, splitted_true_label)
        hamming_scores.append(score)

    return y_true, y_pred, hamming_scores

for subset in ["test","dev"]:

    CSV = f"{VERSION}/predictions_{subset}.csv"

    y_true, y_pred, hamming_scores = transform(CSV)

    print(f">> Subset {subset}")

    print(">> hamming_scores")
    hamming_score = sum(hamming_scores) / len(hamming_scores)
    print(hamming_score)

    print(">> exact_match")
    exact_match = compute_accuracy_exact_match(y_true, y_pred)
    print(exact_match)
    print()
