import os
import re

import pandas as pd


class Vignette:
    def __init__(self, number, question, answer, reference):
        self.number = number
        self.question = question
        self.answer = answer
        self.reference = reference

    def __repr__(self):
        return f"Vignette(number={self.number}, question={self.question}, answer={self.answer}, reference={self.reference})"


def parse_vignettes(text):
    pattern = re.compile(
        r"# Vignette (\d+):\n\n## Question:\n(.+?)\n\n## Answer:\n(.+?)\n\n## Reference:\n(.+?)(?=\n# Vignette|\Z)",
        re.DOTALL)

    # For vignettes without refrence
    # pattern = re.compile(
    #     r"# Vignette (\d+):\n\n## Question:\n(.+?)\n\n## Answer:\n(.+?)(?=\n# Vignette|\Z)",
    #     re.DOTALL)
    vignettes = []

    for match in pattern.finditer(text):
        number = int(match.group(1))
        question = match.group(2).strip()
        answer = match.group(3).strip()
        reference = match.group(4).strip()
        vignettes.append(Vignette(number, question, answer, reference))

    return vignettes


def vignettes_to_dataframe(vignettes):
    data = {
        "Number": [v.number for v in vignettes],
        "Question": [v.question for v in vignettes],
        "Answer": [v.answer for v in vignettes],
        "Reference": [v.reference for v in vignettes],
    }
    df = pd.DataFrame(data)
    return df


def aggegate_vignettes(folder_path):
    all_dfs = []
    for file_name in os.listdir(folder_path + 'vignettes'):
        if file_name.startswith('vignette') and file_name.endswith('.txt'):
            file_path = folder_path + "vignettes/" +file_name
            with open(file_path, 'r', encoding='utf8') as f:
                vignette_text = f.read()
                vignettes = parse_vignettes(vignette_text)
                df = vignettes_to_dataframe(vignettes)
                df['doc_id'] = file_name.split('_')[1]
                df['pmid'] = file_name.split('_')[2].split('.')[0]
                all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_excel(f'{folder_path}vignettes_.xlsx', index=False)