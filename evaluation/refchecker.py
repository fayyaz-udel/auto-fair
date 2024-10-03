import os

import pandas as pd

from refchecker import LLMExtractor, LLMChecker

from utils import chat


def refchecker(responses, references):
    extraction_results = extractor.extract(batch_responses=responses, batch_questions=[], max_new_tokens=1000)
    batch_claims = [[c.content for c in res.claims] for res in extraction_results]
    references = references
    batch_labels = checker.check(batch_claims=batch_claims, batch_references=references, max_reference_segment_length=0)
    c = sum('Contradiction' in labels for labels in batch_labels[0])
    n = sum('Neutral' in labels for labels in batch_labels[0])
    e = sum('Entailment' in labels for labels in batch_labels[0])
    return batch_claims, batch_labels, c, n, e


def run(file_path):
    df = pd.read_excel(file_path)
    new_df_rows = []
    for index, row in df.iterrows():
        r = row.to_dict()
        q = chat(f"Here is a question and it's related answer.\\n\\n{r['Question']} \\n\\n{r['Answer']}\\n\\n Considering the answer, transform the question into a claim.")
        claim, label, c, n, e = refchecker([q], [r['Reference']])
        aggregated_dict = {**{'refcheck_claim': claim, 'refcheck_label': label, 'refcheck_c': c, 'refcheck_n': n, 'refcheck_e': e, 'claim': q}, **r}
        new_df_rows.append(aggregated_dict)
        print(aggregated_dict)

    new_df = pd.DataFrame(new_df_rows)
    new_df.to_excel(file_path, index=False)


if __name__ == "__main__":
    model = 'gpt-4o'
    extractor = LLMExtractor(model=model, batch_size=8)
    checker = LLMChecker(model=model, batch_size=8)

    folder = "../output/obesity stigma_gpt-4o-2024-05-13_2024_10_03__13_22_09/"
    path = folder + "vignettes_.xlsx"
    run(path)
