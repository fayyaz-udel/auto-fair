import pandas as pd

from utils import calculate_avg_std

if __name__ == '__main__':
    folder = '../dataset/'
    file = "vignettes_our_gpt4o.xlsx"
    df = pd.read_excel(folder + file)

    calculate_avg_std(df, folder + file[:-5] + "_metrics.txt")
    print("#" * 50)
    df = df[df['geval'] > 0.8]
    df = df[df['refcheck_c'] == 0]

    calculate_avg_std(df, folder + file[:-5] + "_metrics_filter.txt")
