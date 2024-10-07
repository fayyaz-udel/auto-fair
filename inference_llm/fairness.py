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
        true_positive_rate = group_df[group_df[true_column] == 1][predicted_column].mean()
        equal_opportunity[group] = true_positive_rate
    return equal_opportunity


# Function to calculate equalized odds (both TPR and FPR)
def calculate_equalized_odds(df, group_column, true_column, predicted_column):
    equalized_odds = {}

    for group in df[group_column].unique():
        group_df = df[df[group_column] == group]

        # True Positive Rate (TPR)
        true_positive_rate = group_df[group_df[true_column] == 1][predicted_column].mean()

        # False Positive Rate (FPR)
        false_positive_rate = group_df[group_df[true_column] == 0][predicted_column].mean()

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


def run(folder_name, endpoint):
    threshold = 0.8
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
        data['group'] += [sa] * length
        data['true_label'] += df_tmp['gt_yesno'].fillna(False).tolist()
        data['predicted_label'] += df_tmp['llm_response_yesno'].tolist()

    df = pd.DataFrame(data)

    # Calculate metrics
    demographic_parity = calculate_demographic_parity(df, 'group', 'predicted_label')
    equal_opportunity = calculate_equal_opportunity(df, 'group', 'true_label', 'predicted_label')
    equalized_odds = calculate_equalized_odds(df, 'group', 'true_label', 'predicted_label')
    ppv_parity = calculate_ppv_parity(df, 'group', 'true_label', 'predicted_label')
    fpr_parity = calculate_fpr_parity(df, 'group', 'true_label', 'predicted_label')

    # Aggregate metrics into a DataFrame with metrics as rows and groups as columns
    groups = df['group'].unique()

    # Prepare a dictionary where keys are metrics and values are dictionaries of group: value
    metrics_dict = {
        'Demographic Parity': demographic_parity,
        'Equal Opportunity': equal_opportunity,
        'True Positive Rate (TPR)': {group: equalized_odds[group]['True Positive Rate (TPR)'] for group in groups},
        'False Positive Rate (FPR)': {group: equalized_odds[group]['False Positive Rate (FPR)'] for group in groups},
        'PPV-Parity (Precision)': ppv_parity,
        'FPR-Parity': fpr_parity
    }

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df = metrics_df.transpose()

    # Convert DataFrame to LaTeX table format
    latex_table = metrics_df.to_latex(float_format="%.2f", index=True)

    print(latex_table)


if __name__ == "__main__":
    folder_name = "obesity stigma_gpt-4o_2024_07_17__18_29_43"
    endpoint_list = [
        "gemma-7b-20240903-163409",
        "jumpstart-dft-meta-textgeneration-l-20240903-162450",
        "jumpstart-dft-meta-textgeneration-l-20240903-162450",
        "mistral-7b-v3-20240903-162556",
        "huggingface-pytorch-tgi-inference-2024-09-03-15-20-50-976",
        "huggingface-pytorch-tgi-inference-2024-09-03-15-56-22-457",
        ]

    for endpoint in endpoint_list:
        print("\multicolumn{6}{c}{" + endpoint + "} \\\\")
        run(folder_name, endpoint)
