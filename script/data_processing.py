import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split, KFold

def load_data(csv_file, module_file, iteration_number):
    ADMC = list(pd.read_csv(csv_file, nrows=0).columns)
    with open(module_file, 'r') as file:
        contents = file.read()
        module = ast.literal_eval(contents)
    myList_pop = module[iteration_number]
    myList = [x for x in ADMC if x not in myList_pop]
    N_in = len(myList) - 1
    df = pd.read_csv(csv_file, usecols=myList)
    y_csv = df["DX"]
    x_csv = df.drop('DX', axis=1)
    x = x_csv.iloc[:,]
    y = y_csv.to_numpy()
    X = x.values.reshape(x.shape[0], x.shape[1], 1)
    return X, y, N_in, module