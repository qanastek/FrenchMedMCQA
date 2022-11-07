import json
import hashlib

path = f"./collect.json"

f = open(path)

data = json.load(f)

for d in data:

    # Fix Type
    new_type = "simple"

    if len(d["correct_answers"]) > 1:
        new_type = "multiple"
    
    d["type"] = new_type

    # Fix Identifier
    unmutable_question = {
        "question": d["question"],
        "answers": d["answers"],
        "correct_answers": d["correct_answers"],
        "subject_name": d["subject_name"],
        "type": d["type"],
        "source_url": d["source_url"],
    }

    identifier = hashlib.sha256(str(unmutable_question).encode('utf-8')).hexdigest()

    d["id"] = str(identifier)

f.close()

with open(path, 'w') as f_out:
    json.dump(data, f_out, indent=4)
