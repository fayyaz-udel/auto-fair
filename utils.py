import json
import os

import boto3
from botocore.exceptions import ClientError
from nltk import word_tokenize
from openai import OpenAI

from config import *
from prompts import *


def calculate_avg_std(df, file_path):
    f = open(file_path, "w")

    concated_text = df['Question'].str.cat(sep=' ')
    token_count = len(set([token.lower() for token in word_tokenize(concated_text)]))
    print(f"All Tokens count, {token_count}\n")
    f.write(f"All Tokens count, {token_count}\n")
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            avg = df[column].mean()
            std = df[column].std()
            formatted_avg = f"{avg:.2f}"
            formatted_std = f"{std:.2f}"
            print(f"{column}, {formatted_avg},{formatted_std}")
            f.write(f"{column}, {formatted_avg},{formatted_std}\n")
    f.flush()
    f.close()


def chat(prompt: str, model: str, system: str = None) -> str:
    ############## GPT ##############
    if "gpt" in model:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        msg = [{"role": "user", "content": prompt}]
        if system:
            msg.append({"role": "system", "content": system})
        chat_completion = client.chat.completions.create(messages=msg, model=model)
        return chat_completion.choices[0].message.content

    ############## Claude ##############
    elif model == "claude":
        bedrock = boto3.client(service_name="bedrock-runtime")
        body = json.dumps({
            "max_tokens": 256000,
            "messages": [{"role": "user", "content": prompt}],
            "anthropic_version": "bedrock-2023-05-31"
        })

        response = bedrock.invoke_model(body=body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0")

        response_body = json.loads(response.get("body").read())
        return response_body.get("content")[0]['text']

    ############## llama3 ##############
    elif model == "llama3":
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        model_id = "meta.llama3-70b-instruct-v1:0"
        formatted_prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": 2048,
            "temperature": 0.5,
        }

        request = json.dumps(native_request)

        try:
            response = client.invoke_model(modelId=model_id, body=request)

        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)
        model_response = json.loads(response["body"].read())
        response_text = model_response["generation"]
        return response_text
    else:
        raise Exception("Invalid model version")


def generate_vignettes(disease, context, model, has_context=True):
    if has_context:
        vignette_prompt = brief_vignette_template.format(disease=disease, context=context)
    else:
        vignette_prompt = brief_vignette_template_wo_context

    vignettes = chat(vignette_prompt, model)
    return vignettes


# def truncate_text(input_text: str, limit=9830) -> str:
#     tokenizer = tiktoken.encoding_for_model(model)
#     tokens = tokenizer.encode(input_text)
#     print("No of tokens:" + str(len(tokens)))
#
#     if len(tokens) <= limit:
#         return input_text
#
#     truncated_tokens = tokens[:limit]
#     truncated_text = tokenizer.decode(truncated_tokens)
#     return truncated_text


def generate_pubmed_query(disease):
    query = ""
    for word in disease.split():
        query += f'AND ({word}[Title]) \n\n'

    return query[3:] + pubmed_query
