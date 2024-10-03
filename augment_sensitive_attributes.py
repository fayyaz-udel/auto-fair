import os
import re

import pandas as pd
sensitive_attribute_list = ["male", "female", "white", "black", "asian", "hispanic"]

def add_sensitive_to_question(question, sensitive_attribute):
    pattern_s = r'\b(patient|person|individual)\b'
    pattern_p = r'\b(patients|people|individuals)\b'

    if re.search(pattern_s, question):
        return re.sub(pattern_s, f"{sensitive_attribute} patient", question)
    elif re.search(pattern_p, question):
        return re.sub(pattern_p, f"{sensitive_attribute} patients", question)
    else:
        return None


def add_sensitive_attributes(file_in_path, file_out_path, sensitive_attribute):
    new_df_rows = []
    df = pd.read_excel(file_in_path)

    for index, row in df.iterrows():
        row_dict = row.to_dict()
        new_question = add_sensitive_to_question(row_dict["Question"].lower(), sensitive_attribute)
        if new_question:
            row_dict["Question"] = new_question
        else:
            print(row_dict["Question"])
            continue
        new_df_rows.append(row_dict)

    new_df = pd.DataFrame(new_df_rows)
    new_df.to_excel(file_out_path, index=False)


def run_step2(folder_path):
    file_name = "vignettes_.xlsx"
    sensitive_attribute = sensitive_attribute_list
    for sn in sensitive_attribute:
        out_file_path = folder_path + f"{sn}/" + file_name[:-5] + f"{sn}.xlsx"
        in_file_path = folder_path + file_name
        os.mkdir(folder_path + f"{sn}")
        add_sensitive_attributes(in_file_path, out_file_path, sn)


if __name__ == "__main__":
    path = "./output/obesity stigma_gpt-4o-2024-05-13_2024_10_03__13_22_09/"
    run_step2(path)
