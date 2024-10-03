import os

import pandas as pd
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

from utils import calculate_avg_std


def g_eval(actual_output, context):
    correctness_metric = GEval(
        model="gpt-4o",
        name="faithfulness",
        criteria="Determine whether the 'actual output' correctly represent a question from the given context.",
        evaluation_params=[LLMTestCaseParams.CONTEXT, LLMTestCaseParams.ACTUAL_OUTPUT],
        verbose_mode=True
    )
    test_case = LLMTestCase(input='extract a question from the given context', actual_output=actual_output,
                            context=[context])
    correctness_metric.measure(test_case)
    return correctness_metric.score


def run(file_path):
    df = pd.read_excel(file_path)
    new_df_rows = []
    for index, row in df.iterrows():
        r = row.to_dict()
        aggregated_dict = {**{'geval': g_eval(r['Question'], r['Reference'])}, **r}
        new_df_rows.append(aggregated_dict)
        print(aggregated_dict)

    new_df = pd.DataFrame(new_df_rows)
    new_df.to_excel(file_path, index=False)
    calculate_avg_std(new_df, file_path[:-5] + "_metrics.txt")


if __name__ == "__main__":
    # os.environ["OPENAI_API_KEY"] =
    folder = "../output/obesity stigma_gpt-4o-2024-05-13_2024_10_03__13_22_09/"
    path = folder + "vignettes_.xlsx"
    run(path)
