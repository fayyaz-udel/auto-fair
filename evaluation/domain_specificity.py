import numpy as np
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
import pandas as pd


model = SentenceTransformer("neuml/pubmedbert-base-embeddings")


def cosine_similarity(vecs):
    vec1 = vecs[0]
    vec2 = vecs[1]
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def embed(sentences):
    return model.encode(sentences)


if __name__ == "__main__":
    sims_r = []
    sims_c = []
    file_path = "../output/obesity stigma_gpt-4o_2024_07_17_18_57_00_wocontext/vignettes__metrics.xlsx"
    for index, row in pd.read_excel(file_path).iterrows():
        sims_c.append(cosine_similarity(embed([row["Question"], "Obesity Stigma"])))
        sims_r.append(cosine_similarity(embed([row["Question"], row["Reference"]])))


    print(np.mean(sims_c), np.std(sims_c))
    print(np.mean(sims_r), np.std(sims_r))
