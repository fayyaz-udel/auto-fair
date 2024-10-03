import pandas as pd
from sagemaker.predictor import retrieve_default

import config


def inference_aws(endpoint_name, query, model_name=None):
    response_text = ""
    response_text_seperated = ""
    log_prob = ""
    predictor = retrieve_default(endpoint_name)

    payload = {
        "inputs": query,
        "parameters": {
            "max_new_tokens": 15,
            "decoder_input_details": True,
            "details": True

        }
    }

    response = predictor.predict(payload)

    for t in response[0]['details']['tokens']:
        response_text += t['text']
        response_text_seperated += t['text']
        response_text_seperated += "*#*"
        log_prob += str(t['logprob'])
        log_prob += "*#*"
    return response_text, response_text_seperated, log_prob  # (response[0]['generated_text'])[len(payload['inputs']):].strip()


def ask_from_llm(endpoint_n, file_in_path, file_out_path, model_name=None):
    df = pd.read_excel(file_in_path)
    new_df_rows = []
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        qest = f'''Question: {row_dict['Question']} \n\n Answer (Yes or No):'''
        response_text, response_text_seperated, log_prob = inference_aws(endpoint_n, qest, model_name)
        response = {'llm_response': response_text,
                    'llm_response_seperated': response_text_seperated,
                    'llm_log_prob': log_prob}

        aggregated_dict = {**response, **row_dict}
        new_df_rows.append(aggregated_dict)
        print(aggregated_dict)

    new_df = pd.DataFrame(new_df_rows)
    new_df.to_excel(file_out_path, index=False)


def run_step3(folder, endpoint, model_name=None):
    endpoint_short = endpoint.replace("jumpstart-dft-hf-llm-", "")

    for sa in [''] + config.sensitive_attribute_list:
        print("hamed   " + sa)
        file_in_path = folder + f'{sa}/vignettes_{sa}.xlsx'
        file_out_path = folder + f'{sa}/vignettes_{sa}_{endpoint_short}.xlsx'
        ask_from_llm(endpoint, file_in_path, file_out_path, model_name)


if __name__ == "__main__":
    folder = './output/obesity stigma_gpt-4o_2024_07_17__18_29_43/'
    endpoint = "jumpstart-dft-hf-llm-mistral-7b-v3-20240903-162556"
    run_step3(folder, endpoint, None)
