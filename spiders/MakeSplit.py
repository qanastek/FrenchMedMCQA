import os
import json

from tqdm import tqdm

f = open("collect.json","r")
data = json.load(f)
data = {d["id"]: d for d in data}

data_splits = {
    "train": [],
    "dev": [],
    "test": [],
}

for subset in data_splits.keys():

    errors = []

    # Get identifiers for the current subset
    subset_file = open(f"./splits_indexes/{subset}.txt","r")
    ids = subset_file.read().split("\n")
    subset_file.close()

    # For each identifier of the subset
    for identifier in ids:

        if identifier in data:

            data_splits[subset].append(
                data[identifier]
            )

        else:

            errors.append(identifier)

    print(f">> {subset} - {len(errors)} errors")
    # print(f">> {subset} - error example => {errors[0]}")

OUTPUT_DIR = f"./corpus"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("#"*50)
print("Size of the subsets")
print("#"*50)

sizes = {s: len(data_splits[s]) for s in list(data_splits.keys())}
assert sizes == {'train': 2171, 'dev': 312, 'test': 622}

# For each subset
for s in list(data_splits.keys()):

    OUTPUT_DIR_SPLIT = f"{OUTPUT_DIR}/{s}.json"

    with open(OUTPUT_DIR_SPLIT, 'w') as f:
        json.dump(data_splits[s], f, indent=4)
    
    print(f">> {s} : {len(data_splits[s])}")
