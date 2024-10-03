# __import__('pysqlite3')
# import sys
#
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import dotenv
import pandas as pd

from augment_sensitive_attributes import run_step2

dotenv.load_dotenv()

import streamlit as st
from generation.generate_vignettes import generate

st.markdown("<h1 style='text-align: center;'>Vignette Generator</h1>", unsafe_allow_html=True)
openai_key = st.text_input('OpenAI API Key:', type='password')
condition = st.text_input('Condition/Disease:', 'Obesity')
count = st.select_slider('Count:', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], value=3)
model = st.selectbox('Model', ["gpt-4o", "gpt-4o-mini"])
sensitive_attribute = st.selectbox('Sensitive Attribute', ["gender", "race"])


def generate_vignette(params):
    if params["sensitive_attribute"] == 'gender':
        sensitive_attribute_list = ["male", "female"]
    else:
        sensitive_attribute_list = ["white", "black", "asian", "hispanic"]
    file_name = "vignettes_.xlsx"
    folder_path = generate(params)
    run_step2(folder_path)

    dfs = []
    for sn in sensitive_attribute_list:
        out_file_path = folder_path + f"{sn}/" + file_name[:-5] + f"{sn}.xlsx"
        dfs.append(pd.read_excel(out_file_path))
    return pd.concat(dfs).loc[:, ['pmid', 'Question', 'Answer', 'Reference']]


result = None
if st.button('Generate Vignettes'):
    params = {
        "disease": condition,
        "count": count,
        "sensitive_attribute": sensitive_attribute,
        "model": model}

    os.environ["OPENAI_API_KEY"] = openai_key
    result = generate_vignette(params)
    st.dataframe(result)

# Run the app using: streamlit run frontend.py
