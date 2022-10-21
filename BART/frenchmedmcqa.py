# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FrenchMedMCQA : A French Multi-Choice Question Answering Corpus for Medical domain"""

import os
import json

import pandas as pd

import datasets

_DESCRIPTION = """\
FrenchMedMCQA
"""

_HOMEPAGE = "https://frenchmedmcqa.github.io"

_LICENSE = "Apache License 2.0"

_URL = "https://drive.google.com/uc?export=download&id=XXXXXXXXXXXXXXXXXXX" # BM25 Wikipedia

_CITATION = """\
@InProceedings{FrenchMedMCQA,
  title     = {FrenchMedMCQA : A French Multiple-Choice Question Answering Dataset for Medical domain},
  author    = {Yanis LABRAK, Adrien BAZOGE, Richard DUFOUR, Béatrice DAILLE, Mickael ROUVIER, Emmanuel MORIN and Pierre-Antoine GOURRAUD},
  booktitle = {EMNLP 2022 Workshop - The 13th International Workshop on Health Text Mining and Information Analysis (LOUHI 2022)},
  year      = {2022},
  pdf       = {Coming soon},
  url       = {Coming soon},
  abstract  = {Coming soon}
}
"""

class FrenchMedMCQA(datasets.GeneratorBasedBuilder):
    """FrenchMedMCQA : A French Multiple-Choice Question Answering Dataset for Medical domain"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):

        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answer_a": datasets.Value("string"),
                "answer_b": datasets.Value("string"),
                "answer_c": datasets.Value("string"),
                "answer_d": datasets.Value("string"),
                "answer_e": datasets.Value("string"),
                "label": datasets.ClassLabel(
                    names = ["c","a","e","d","b","be","ae","bc","bd","ab","de","cd","ac","ad","ce","bce","abc","cde","bcd","ace","ade","abe","acd","bde","abd","abde","abcd","bcde","abce","acde","abcde"]
                ),
                "correct_answers": datasets.Sequence(
                    datasets.Value("string"),
                    # datasets.features.ClassLabel(names=["a", "b", "c", "d", "e"]),
                ),
                "type": datasets.Value("string"),
                "subject_name": datasets.Value("string"),
                "context": datasets.Value("string"),
                
                "source": datasets.Value("string"),
                "target": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.csv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.csv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.csv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):

        df = pd.read_csv(filepath, sep=",")

        for key, row in df.iterrows():

            if str(row["context"]) == "nan":
                row["context"] = "aucun contexte"

            question = row['question'].replace("\n"," ")

            correct_answers = " + ".join(sorted(row["correct_answers"].upper().split(";")))

            a = row["answers.a"].replace("\n"," ")
            b = row["answers.b"].replace("\n"," ")
            c = row["answers.c"].replace("\n"," ")
            d = row["answers.d"].replace("\n"," ")
            e = row["answers.e"].replace("\n"," ")

            source = f"{question} \\n (A) {a} (B) {b} (C) {c} (D) {d} (E) {e}\\n{row['context']}"
            target = f"{correct_answers}"

            yield key, {
                "id": row["id"],
                "question": row["question"],
                "answer_a": a,
                "answer_b": b,
                "answer_c": c,
                "answer_d": d,
                "answer_e": e,
                "correct_answers": row["correct_answers"].split(";"),
                "label": "".join(sorted(row["correct_answers"].split(";"))),
                "type": row["type"],
                "subject_name": row["subject_name"],
                "context": row["context"],
                
                "source": source,
                "target": target,
            }
