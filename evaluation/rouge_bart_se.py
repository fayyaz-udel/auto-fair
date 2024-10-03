import pandas as pd
from nltk.tokenize import word_tokenize
from rouge import Rouge

from bart_score import BARTScorer
from semantic_entropy import calculate_semantic_entropy
from utils import calculate_avg_std

bart_scorer = BARTScorer(device='cpu', checkpoint='facebook/bart-large-cnn')


def count_tokens(text):
    return len(set([token.lower() for token in word_tokenize(text)]))


def calculate_metrics(generated_text, reference_text):
    rouge = Rouge()
    # rouge_scores = rouge.get_scores(generated_text, reference_text)
    bart_score = bart_scorer.score([reference_text], [generated_text], batch_size=4)[0]
    entrpoy, smiliarity = calculate_semantic_entropy(generated_text, reference_text)
    measurement_dict = {
        "Token Count": count_tokens(generated_text),

        # "ROUGE-1 Recall": rouge_scores[0]["rouge-1"]["r"],
        # "ROUGE-1 Precision": rouge_scores[0]["rouge-1"]["p"],
        # "ROUGE-1 F1": rouge_scores[0]["rouge-1"]["f"],
        #
        # "ROUGE-2 Recall": rouge_scores[0]["rouge-2"]["r"],
        # "ROUGE-2 Precision": rouge_scores[0]["rouge-2"]["p"],
        # "ROUGE-2 F1": rouge_scores[0]["rouge-2"]["f"],
        #
        # "ROUGE-L Recall": rouge_scores[0]["rouge-l"]["r"],
        # "ROUGE-L Precision": rouge_scores[0]["rouge-l"]["p"],
        # "ROUGE-L F1": rouge_scores[0]["rouge-l"]["f"],
        "BARTScore": bart_score,
        "Semantic Entropy": entrpoy,
        "Semantic Similarity": smiliarity
    }
    return measurement_dict


def run(file_path):
    df = pd.read_excel(file_path)
    new_df_rows = []
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        # aggregated_dict = {**calculate_metrics(row_dict["Question"], "None"), **row_dict} # For vignettes without refrence
        aggregated_dict = {**calculate_metrics(row_dict["Question"], row_dict["Reference"]), **row_dict}
        new_df_rows.append(aggregated_dict)
        print(aggregated_dict)

    new_df = pd.DataFrame(new_df_rows)
    new_df.to_excel(file_path[:-5] + "_metrics.xlsx", index=False)
    calculate_avg_std(new_df, file_path[:-5] + "_metrics.txt")


if __name__ == "__main__":
    folder = "./output/obesity stigma_claude_2024_10_02__03_11_56/"
    path = folder + "vignettes_.xlsx"
    run(path)
