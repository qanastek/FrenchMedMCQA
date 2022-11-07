import pandas as pd

def convert(subset_name):

    csv_path = f"./corpus/{subset_name}.json"

    df = pd.read_json(csv_path, lines=False)

    data = []

    for index, row in df.iterrows():

        data.append([
            row['id'],
            row['question'],
            row['answers']['a'],
            row['answers']['b'],
            row['answers']['c'],
            row['answers']['d'],
            row['answers']['e'],
            ";".join(row['correct_answers']),
            row['subject_name'],
            row['type'],
        ])

    df_new = pd.DataFrame(
        data,
        columns = [
            "id",
            "question",
            "answers.a",
            "answers.b",
            "answers.c",
            "answers.d",
            "answers.e",
            "correct_answers",
            "subject_name",
            "type",
        ]
    )

    df_new.to_csv(f"./corpus_csv/{subset_name}.csv", encoding='utf-8', index=False)

[convert(a) for a in ["train","dev","test"]]
