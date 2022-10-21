import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# VERSION = f"V3" # SBERT + wiki + with multiple + answers for SBERT + BART-BASE
# VERSION = f"V4" # BART-LARGE - Not aligned
VERSION = f"V5 - wiki bm25"
# VERSION = f"V6 - no ctx"
# VERSION = f"V7 - output_hal_ctx" # HAL SBERT 
# VERSION = f"V8 - output_hal_ctx_bm25" # HAL BM25 

LETTERS = ["A","B","C","D","E"]

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

def getName(sigle):
    if sigle == "True":
        return "Correct"
    else:
        return "Wrong"

def getColor(sigle):
    if sigle == "Correct":
        return "#f9d0c8"
    else:
        return "#F0A093"

def getPieChart(y_true, y_pred, length):

    print("Elements of the correct length EMR : ", sum([True for e in y_true if len(list(e)) == length]))

    sames = []

    for a, b in zip(y_true, y_pred):

        if len(a) == length:

            same = sorted(list(a)) == sorted(list(set(b)))

            print(sorted(list(a)))
            print(sorted(list(set(b))))
            print(same)
            print()
            
            sames.append(str(same))

    print("#"*50)

    sames = Counter(sames)

    keys = list(sames.keys())
    keys = [getName(a) for a in keys]
    my_colors = [getColor(a) for a in keys]

    values = list(sames.values())
    values = np.array(values)

    plt.pie(values, labels=keys, startangle=90, autopct='%1.2f%%', textprops={'fontsize': 12}, colors=my_colors)
    os.makedirs(f"../pie_charts/{subset}/", exist_ok=True)
    plt.savefig(f"../pie_charts/{subset}/L{length}.png", bbox_inches='tight')
    plt.clf()

    return sames

def getHammingScore(y_true, y_pred, length):

    # print("Elements of the correct length Hamming : ", sum([True for e in y_true if len(list(e)) == length]))

    all_elements = []

    for a, b in zip(y_true, y_pred):
        
        if len(a) == length:

            predictions = list(set(list(b)))
            reals = list(set(list(a)))

            corrects = [True for p in predictions if p in reals]

            predicted_and_real = sorted(list(set(predictions + reals)))
            local_score = sum(corrects) / len(predicted_and_real)

            # print("Predicted : ", predictions)
            # print("Real : ", reals)
            # print("corrects : ", corrects)
            # print("Both : ", predicted_and_real)
            # print("Score : ", local_score)
            # print()
            # print()

            all_elements.append(local_score)

    # print("#"*50)
    # print(sum(all_elements))
    # print(len(all_elements))
    # print(sum(all_elements) / len(all_elements))
    # print((sum(all_elements) / len(all_elements)) * 100)
    # print()

    return (sum(all_elements) / len(all_elements)) * 100

for subset in ["dev","test"]:

    CSV = f"{VERSION}/predictions_{subset}.csv"

    y_true, y_pred, hamming_scores = transform(CSV)

    xAxis = []
    yAxis = []

    hamming_xAxis = []
    hamming_yAxis = []

    print(f">> Subset {subset}")
    for length in range(1,6):
        print(f"Elements length {subset} : ", Counter([len(list(e)) for e in y_true]))
        sames = getPieChart(y_true, y_pred, length)
        xAxis.append(length)
        percentage = float((sames["True"] / (sames["True"] + sames["False"])) * 100)
        yAxis.append(percentage)

        hamming_xAxis.append(length)
        score = getHammingScore(y_true, y_pred, length)
        hamming_yAxis.append(score)

    plt.plot(xAxis, yAxis, color='#E5988C', marker='o')
    plt.plot(hamming_xAxis, hamming_yAxis, color='#292A42', marker='o')
    # plt.title('Percent of correct answers on test set according to the number of answer(s)')
    plt.ylabel('EMR (light) / Hamming score (dark)')
    plt.xlabel('Number of answer(s)')
    plt.xticks([1,2,3,4,5])

            
    plt.ylim([0, 100])
    plt.xlim([0.9, 5.2])

    for x, y in zip(xAxis, yAxis): 
        plt.text(x+0.1, y+2.4, "%.2f" % y + " %", fontsize=13, color='#c34e3c')

    for x, y in zip(hamming_xAxis, hamming_yAxis):
        plt.text(x+0.1, y+2.4, "%.2f" % y + " %", fontsize=13, color='#292A42')

    # plt.grid(True)
    os.makedirs(f"../line_chart/", exist_ok=True)
    plt.savefig(f"../line_chart/{subset}.png", bbox_inches='tight')
