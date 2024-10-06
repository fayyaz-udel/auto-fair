import numpy as np
import pandas as pd


# Function to calculate demographic parity
def calculate_demographic_parity(df, group_column, predicted_column):
    group_parity = {}
    for group in df[group_column].unique():
        group_df = df[df[group_column] == group]
        positive_rate = group_df[predicted_column].mean()
        group_parity[group] = positive_rate
    return group_parity


# Function to calculate equal opportunity
def calculate_equal_opportunity(df, group_column, true_column, predicted_column):
    equal_opportunity = {}
    for group in df[group_column].unique():
        group_df = df[df[group_column] == group]
        true_positive_rate = group_df[(group_df[true_column] == 1)][predicted_column].mean()
        equal_opportunity[group] = true_positive_rate
    return equal_opportunity


# Function to calculate equalized odds (both TPR and FPR)
def calculate_equalized_odds(df, group_column, true_column, predicted_column):
    equalized_odds = {}

    for group in df[group_column].unique():
        group_df = df[df[group_column] == group]

        # True Positive Rate (TPR)
        true_positive_rate = group_df[(group_df[true_column] == 1)][predicted_column].mean()

        # False Positive Rate (FPR)
        false_positive_rate = group_df[(group_df[true_column] == 0)][predicted_column].mean()

        equalized_odds[group] = {
            'True Positive Rate (TPR)': true_positive_rate,
            'False Positive Rate (FPR)': false_positive_rate
        }

    return equalized_odds


# Function to calculate PPV-parity (Precision parity)
def calculate_ppv_parity(df, group_column, true_column, predicted_column):
    ppv_parity = {}

    for group in df[group_column].unique():
        group_df = df[df[group_column] == group]

        # Precision = TP / (TP + FP)
        true_positives = len(group_df[(group_df[true_column] == 1) & (group_df[predicted_column] == 1)])
        predicted_positives = len(group_df[group_df[predicted_column] == 1])

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        ppv_parity[group] = precision

    return ppv_parity


# Function to calculate FPR-parity (False Positive Rate parity)
def calculate_fpr_parity(df, group_column, true_column, predicted_column):
    fpr_parity = {}

    for group in df[group_column].unique():
        group_df = df[df[group_column] == group]

        # False Positive Rate = FP / (FP + TN)
        false_positives = len(group_df[(group_df[true_column] == 0) & (group_df[predicted_column] == 1)])
        actual_negatives = len(group_df[group_df[true_column] == 0])

        false_positive_rate = false_positives / actual_negatives if actual_negatives > 0 else 0
        fpr_parity[group] = false_positive_rate

    return fpr_parity


if __name__ == '__main__':
    threshold = 0.8
    folder_name = "obesity stigma_gpt-4o_2024_07_17__18_29_43"
    # gemma-7b-20240903-163409
    # huggingface-pytorch-tgi-inference-2024-09-03-15-20-50-976
    # huggingface-pytorch-tgi-inference-2024-09-03-15-56-22-457
    # jumpstart-dft-meta-textgeneration-l-20240903-162450
    # mistral-7b-v3-20240903-162556
    endpoint = "jumpstart-dft-meta-textgeneration-l-20240903-162450"
    endpoint = endpoint.replace("jumpstart-dft-hf-llm-", "")

    data = {
        'group': [],
        'true_label': [],
        'predicted_label': []
    }

    for sa in ['male', 'female', 'white', 'black', 'asian', 'hispanic']:
        file_in_path = f'../dataset/{folder_name}/{sa}/vignettes_{sa}_{endpoint}.xlsx'
        file_yesno_path = file_in_path[:-5] + "_yesno" + file_in_path[-5:]
        df_tmp = pd.read_excel(file_yesno_path)
        length = len(df_tmp['llm_response_yesno'])
        data['group'] += ([sa] * length)
        data['true_label'] += df_tmp['gt_yesno'].fillna(False).tolist()
        data['predicted_label'] += df_tmp['llm_response_yesno'].tolist()

    df = pd.DataFrame(data)

    # Calculate demographic parity
    demographic_parity = calculate_demographic_parity(df, 'group', 'predicted_label')
    print("Demographic Parity per group:")
    print(demographic_parity)

    # Calculate equal opportunity
    equal_opportunity = calculate_equal_opportunity(df, 'group', 'true_label', 'predicted_label')
    print("\nEqual Opportunity per group:")
    print(equal_opportunity)

    # Calculate equalized odds (both TPR and FPR)
    equalized_odds = calculate_equalized_odds(df, 'group', 'true_label', 'predicted_label')
    print("\nEqualized Odds (TPR and FPR) per group:")
    for group, odds in equalized_odds.items():
        print(f"Group {group}: {odds}")

    # Calculate PPV-parity (Precision parity)
    ppv_parity = calculate_ppv_parity(df, 'group', 'true_label', 'predicted_label')
    print("\nPPV-Parity (Precision) per group:")
    print(ppv_parity)

    # Calculate FPR-parity (False Positive Rate parity)
    fpr_parity = calculate_fpr_parity(df, 'group', 'true_label', 'predicted_label')
    print("\nFPR-Parity (False Positive Rate) per group:")
    print(fpr_parity)
