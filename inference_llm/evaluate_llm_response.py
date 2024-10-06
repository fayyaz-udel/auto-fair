import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, mutual_info_score, accuracy_score
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import KFold

import config


def find_yes_no(text, seperated=True):
    # Tokenize the text to split into words, considering lowercasing to ensure case insensitivity
    if seperated:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text.lower())
    else:
        tokens = text.lower().split('*#*')

    # Initialize positions to a high value
    pos_yes = float('inf')
    pos_no = float('inf')

    # Iterate through tokens to find positions of 'yes' and 'no'
    for i, token in enumerate(tokens):
        if token.strip() == 'yes' and pos_yes == float('inf'):
            pos_yes = i
        elif token.strip() == 'no' and pos_no == float('inf'):
            pos_no = i

    # Determine the outcome based on the positions of 'yes' and 'no'
    if pos_yes != float('inf') and pos_no != float('inf'):
        if pos_yes < pos_no:
            return True, pos_yes  # Yes comes first, return True and its position
        else:
            return False, pos_no  # No comes first, return False and its position
    elif pos_yes != float('inf'):
        return True, pos_yes  # Only Yes is found
    elif pos_no != float('inf'):
        return False, pos_no  # Only No is found
    else:
        return None, -1  # Neither Yes nor No is found


def binarize_response(file_in_path, file_out_path):
    df = pd.read_excel(file_in_path)
    new_df_rows = []
    for index, row in df.iterrows():
        rowd = row.to_dict()
        llm_response_yesno, answer_idx = find_yes_no(rowd['llm_response_seperated'], seperated=False)
        response = {'llm_response_yesno': llm_response_yesno,
                    'gt_yesno': find_yes_no(rowd['Answer'])[0],
                    'token_prob': float(rowd['llm_log_prob'].split('*#*')[answer_idx]) if answer_idx != -1 else None}
        if response['llm_response_yesno'] is None:
            response['llm_response_yesno'] = not response['gt_yesno']
        aggregated_dict = {**response, **rowd}
        new_df_rows.append(aggregated_dict)

    new_df = pd.DataFrame(new_df_rows)
    new_df.to_excel(file_out_path, index=False)


def evaluate_llm_response(file_path, file_out_path, sa, threshold=None):
    df = pd.read_excel(file_path)
    if threshold is not None:
        df = df[df['geval'] >= threshold]

    y_pred = df['llm_response_yesno'].tolist()
    y_true = df['gt_yesno'].fillna(False).tolist()
    y_prob = np.exp(df['token_prob'].fillna(0.5).tolist())

    for i in range(len(y_prob)):
        if y_pred[i] == False:
            y_prob[i] = 1 - y_prob[i]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mi_scores = []
    accuracy_scores = []

    for train_index, test_index in kf.split(y_true):
        y_true_test = y_true[test_index]
        y_pred_test = y_pred[test_index]

        mi = mutual_info_score(y_true_test, y_pred_test)
        accuracy = accuracy_score(y_true_test, y_pred_test)

        mi_scores.append(mi)
        accuracy_scores.append(accuracy)

    t_stat, p_value = ttest_1samp(accuracy_scores, 0.5)

    # Print or return the t-test results along with other metrics
    print(f"T-test results -- Statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

    kl = np.mean(mi_scores)
    accuracy = np.mean(accuracy_scores)
    kl_std = np.std(mi_scores)
    accuracy_std = np.std(accuracy_scores)



    with open(file_out_path, "a") as f:
        f.write(f"{sa} & {kl:.2f}({kl_std:.2f}) & {accuracy:.2f}({accuracy_std: .2f}) \\\\")
        f.write("\n")
        f.flush()
        f.close()

    print(f"{sa} & {kl:.2f}({kl_std:.2f}) & {accuracy:.2f}({accuracy_std: .2f}) \\\\\\n")


def file_initiate(out_path, folder_name, endpoint):
    with open(out_path, "w") as f:
        f.write(f"{folder_name}")
        f.write("\n")
        f.write(f"{endpoint}")
        f.write("\n")
        f.write("Sensitive Attribute & KL & ACC \\\\\n")
        f.flush()
        f.close()


if __name__ == "__main__":
    threshold = 0.8
    folder_name = "obesity stigma_gpt-4o_2024_07_17__18_29_43"
    # gemma-7b-20240903-163409
    # huggingface-pytorch-tgi-inference-2024-09-03-15-20-50-976
    # huggingface-pytorch-tgi-inference-2024-09-03-15-56-22-457
    # jumpstart-dft-meta-textgeneration-l-20240903-162450
    # mistral-7b-v3-20240903-162556
    endpoint = "jumpstart-dft-meta-textgeneration-l-20240903-162450"
    endpoint = endpoint.replace("jumpstart-dft-hf-llm-", "")
    out_path = f'../output/{folder_name}/fairness_{endpoint}_{threshold:0.2f}.txt'
    file_initiate(out_path, folder_name, endpoint)

    for sa in config.sensitive_attribute_list + ['']:
        file_in_path = f'../output/{folder_name}/{sa}/vignettes_{sa}_{endpoint}.xlsx'
        file_yesno_path = file_in_path[:-5] + "_yesno" + file_in_path[-5:]

        binarize_response(file_in_path, file_yesno_path)
        ############################################################################
        df = pd.read_excel(file_yesno_path)
        df_eval = pd.read_excel(f'../output/{folder_name}/vignettes__metrics.xlsx')
        pd.merge(df, df_eval[['doc_id', 'Number', 'geval']], how='left', left_on=['doc_id', 'Number'], right_on=['doc_id', 'Number']).to_excel(file_yesno_path, index=False)
        ############################################################################
        # evaluate_llm_response(file_yesno_path, out_path, sa, threshold)

        y_pred = df['llm_response_yesno'].tolist()
        y_true = df['gt_yesno'].fillna(False).tolist()
        y_prob = np.exp(df['token_prob'].fillna(0.5).tolist())

        for i in range(len(y_prob)):
            if y_pred[i] == False:
                y_prob[i] = 1 - y_prob[i]
