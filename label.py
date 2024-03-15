from src.labeling import text2features
from src.utils import load_data
from tqdm import tqdm

import benepar, spacy
import pandas as pd


def label():
    nlp_pipeline = spacy.load('en_core_web_md')
    nlp_pipeline.add_pipe("benepar", config={"model": "/proj/mechanistic.shadow/mrofin/tinylinguist/models/benepar_en3"})

    data = load_data("/proj/mechanistic.shadow/mrofin/tinylinguist/data")

    sentences = []
    bad_parses = 0
    for text in tqdm(data["valid"]["text"]):
        text = text.replace("\n", " ")
        words = text.split(" ")
        text = " ".join(w for w in words if w != "")

        try:
            sentences += text2features(text, nlp_pipeline)
        except:
            print("Bad parse!")
            bad_parses += 1

    print(f"Total bad parses: {bad_parses}")

    pd_data = pd.DataFrame(sentences)
    pd_data.to_csv("/proj/mechanistic.shadow/mrofin/tinylinguist/data/valid_sentences.csv", index=False, sep="\t")


if __name__ == "__main__":
    label()
