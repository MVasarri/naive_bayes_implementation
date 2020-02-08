import numpy as np
import pandas as pd
#from scipy import stats #ho dovuto importare la mibreria scipy per calcolare la moda
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder


def sistemaDS(index):
    file = ['mushroom_dataset.csv', 'monk_dataset.csv',
            'csv_result-PhishingData.csv']
    df = pd.read_csv(file[index])
    print('dataset N°: ', index)
    print(df)

    df = df.apply(LabelEncoder().fit_transform)
    print(df)
    row = df.shape[0]
    coll = df.shape[1]
    if index == 2:        
        df.drop('id', axis=1, inplace=True)
        coll = coll-1
        for i in range(row):
            if df[str(coll - 1)].iloc[i] == 2:
                df[str(coll - 1)].iloc[i] = 1
    print(df)
    print(df[str(coll - 1)].value_counts())
    return df

def limMaxKPar(df):
    maxVar = []
    coll = df.shape[1]
    for i in range(coll):
        maxVar.append(df[str(i)].values.max())
    return maxVar

def limMinKPar(df):
    minVar = []
    coll = df.shape[1]
    for i in range(coll):
        minVar.append(df[str(i)].values.min())
    return minVar