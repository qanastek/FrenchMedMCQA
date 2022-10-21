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
"""FrenchMedMCQA : A French Multiple-Choice Question Answering Corpus for Medical domain"""

import os
import json

import pandas as pd

import datasets

_DESCRIPTION = """\
FrenchMedMCQA
"""

_HOMEPAGE = "https://frenchmedmcqa.github.io"

_LICENSE = "Apache License 2.0"

########## NEW 

_URL = "https://drive.google.com/uc?export=download&id=XXXXXXXXXXXXXXXX" # BM25 Wikipedia

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
    """FrenchMedMCQA : A French Multiple-Choice Question Answering Corpus for Medical domain"""

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
                
                "roberta_text_v2": datasets.Value("string"),
                "roberta_text_v2_lower": datasets.Value("string"),
                "roberta_text": datasets.Value("string"),
                "roberta_text_no_ctx": datasets.Value("string"),
                
                "bert_text": datasets.Value("string"),
                "bert_text_lower": datasets.Value("string"),
                "bert_text_no_ctx": datasets.Value("string"),
                "bert_text_no_ctx_lower": datasets.Value("string"),
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

        CLS = "<s>"
        BOS = "<s>"
        SEP = "</s>"
        EOS = "</s>"
        
        BERT_CLS = "[CLS]"
        BERT_BOS = ""
        BERT_SEP = "[SEP]"
        BERT_EOS = ""

        df = pd.read_csv(filepath, sep=",")

        for key, d in df.iterrows():

            if str(d["context"]) == "nan":
                d["context"] = "aucune"

            sequence_v2 = CLS + " " + d["context"] + f" {SEP} {SEP} " +  d["question"] + f" {SEP} {SEP} " + f" {SEP} ".join([d[f"answers.{letter}"] for letter in ["a","b","c","d","e"]]) + EOS
            sequence = CLS + " " + d["question"] + f" {SEP} " + f" {SEP} ".join([d[f"answers.{letter}"] for letter in ["a","b","c","d","e"]]) + " " + SEP + d["context"] + EOS
            sequence_no_ctx = CLS + " " + d["question"] + f" {SEP} " + f" {SEP} ".join([d[f"answers.{letter}"] for letter in ["a","b","c","d","e"]]) + " " + EOS

            bert_ctx = sequence.replace(SEP, BERT_SEP).replace(CLS, BERT_CLS).replace(BOS, BERT_BOS).replace(EOS, BERT_EOS)
            bert_ctx_lower = sequence.lower().replace(SEP, BERT_SEP).replace(CLS, BERT_CLS).replace(BOS, BERT_BOS).replace(EOS, BERT_EOS)
            
            bert_no_ctx = sequence_no_ctx.replace(SEP, BERT_SEP).replace(CLS, BERT_CLS).replace(BOS, BERT_BOS).replace(EOS, BERT_EOS)
            bert_no_ctx_lower = sequence_no_ctx.lower().replace(SEP, BERT_SEP).replace(CLS, BERT_CLS).replace(BOS, BERT_BOS).replace(EOS, BERT_EOS)

            yield key, {
                "id": d["id"],
                "question": d["question"],
                "answer_a": d["answers.a"],
                "answer_b": d["answers.b"],
                "answer_c": d["answers.c"],
                "answer_d": d["answers.d"],
                "answer_e": d["answers.e"],
                "correct_answers": d["correct_answers"].split(";"),
                "label": "".join(sorted(d["correct_answers"].split(";"))),
                "type": d["type"],
                "subject_name": d["subject_name"],
                "context": d["context"],
                
                "roberta_text": sequence,
                "roberta_text_v2": sequence_v2,
                "roberta_text_v2_lower": sequence_v2.lower(),
                "roberta_text_no_ctx": sequence_no_ctx,
                
                "bert_text": bert_ctx,
                "bert_text_lower": bert_ctx_lower,
                "bert_text_no_ctx": bert_no_ctx,
                "bert_text_no_ctx_lower": bert_no_ctx_lower,
            }
