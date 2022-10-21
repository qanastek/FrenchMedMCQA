import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

VERSION = f"V3" # BART-BASE
# VERSION = f"V4" # BART-LARGE - Not aligned

LETTERS = ["A","B","C","D","E"]

def transform(path_csv):

    alignments = []

    y_true = []
    y_pred = []

    y_true_raw = []
    y_pred_raw = []

    df = pd.read_csv(path_csv, sep=",")

    for index, row in df.iterrows():

        def clean(text):
            text = text.replace("<pad>","").replace("</s>","").replace("<s>","").replace("+","").replace(" ","").upper()
            return sorted([t for t in text if t in LETTERS])
        
        def indexes(ids):
            return [LETTERS.index(i) for i in ids]
        
        def getBinaryArray(ids):
            arr = [0 for l in LETTERS]
            for i in ids:
                arr[i] = 1
            return arr

        # Target
        target_letters = clean(row['Target Text'])
        target_ids = indexes(target_letters)
        y_true_raw.append(target_ids)
        target_binary = getBinaryArray(target_ids)
        y_true.append(target_binary)

        # Generated
        generated_letters = clean(row['Generated Text'])
        generated_ids = indexes(generated_letters)
        y_pred_raw.append(generated_ids)
        generated_binary = getBinaryArray(generated_ids)
        y_pred.append(generated_binary)

        alignments.append(abs(len(target_ids) - len(generated_ids)))

    print(">> alignments :")
    print(list(set(alignments)))

    return y_true, y_pred, y_true_raw, y_pred_raw

def SaveCR(text, acc, emr_score, filename):
    text = "Accuracy : " + str(acc) + "\n" + "EMR Score : " + str(emr_score) + "\n" + text
    file_out = open(filename, "w")
    file_out.write(text)
    file_out.close()
    return text

# Exact Match Ratio (EMR)
def EMR(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = len(y_true)
    row_indicators = np.all(y_true == y_pred, axis = 1) # axis = 1 will check for equality along rows.
    exact_match_count = np.sum(row_indicators)
    return exact_match_count

# Micro Averaged Precision
def MicroAvgPrecision(y_true_raw, y_pred_raw):

    refs = []
    [refs.extend(a) for a in y_true_raw]

    preds = []
    [preds.extend(b) for b in y_pred_raw]
    
    res = []

    for r, p in zip(refs, preds):

        if r == p:
            res.append(True)
        else:
            res.append(False)

    return sum(res) / len(res)

for subset in ["test","dev"]:
    CSV = f"{VERSION}/predictions_{subset}.csv"
    CR_OUT = f"{VERSION}/cr_{subset}.txt"
    y_true, y_pred, y_true_raw, y_pred_raw = transform(CSV)
    cr = classification_report(y_true, y_pred, target_names=LETTERS)
    acc = accuracy_score(y_true, y_pred)
    microAvgPrecision = MicroAvgPrecision(y_true_raw, y_pred_raw)
    emrScore = EMR(y_true, y_pred)
    text = SaveCR(cr, microAvgPrecision, emrScore, filename=CR_OUT)
    print(f">> Subset {subset}")
    print(text)
