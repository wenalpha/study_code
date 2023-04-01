# coding: utf-8

import pandas as pd
def process(file):
    data = pd.read_csv(file, sep = '\t', header = None, usecols = [0, 1])
    # print(data.head())
    out_file = file.split('.')[0] +'.txt'
    with open(out_file, 'w') as out:
        for i in range(data.shape[0]):
            label = str(data.iloc[i, 0])
            text = data.iloc[i, 1]
            out.writelines(text)
            out.writelines("\n")
            out.writelines(label)
            out.writelines("\n")

file = 'test.tsv'
process(file)