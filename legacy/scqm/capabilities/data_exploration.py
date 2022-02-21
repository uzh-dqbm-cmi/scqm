import numpy as np
import pandas as pd
import csv


def column_content(df: pd.DataFrame):
    # Get unique values of each column
    temp = df.apply(lambda col: col.unique())
    for i in range(len(temp)):
        # if the number of unique values are larger than 10 - replace it with the label of 'too many'
        if len(temp[i]) > 10:
            temp[i] = 'too many'

    return temp

def nan_content_per_column(df):
    # na_content = list()
    na_content = pd.DataFrame(data=np.zeros((1, len(df.columns))), columns=df.columns)
    length = len(df)
    for i in range(len(df.columns)):
        temp = sum((df.iloc[:, i]).isnull() * 1) / length * 100
        # na_content.append(temp)
        na_content.iloc[0, i] = temp

    return na_content

from typing import Dict
def table_profiling(tables: Dict) -> None:

    patient_id_names = ['p', 'v', 'm', 'hi', 'mny', 'rau', 'soa', 'sor', 'as', 'bf', 'bd', 'd', 'eq', 'h', 'ps', 'ra', 'sf',
                        'se']
    for ix, key in enumerate(tables):
        df = tables[key]
        # header = df.columns.to_numpy()
        types = df.dtypes
        values = column_content(df)
        na_content = nan_content_per_column(df).T
        ddf = pd.concat([types, values, na_content], axis=1)
        ddf.columns = ['Type', 'Unique_Values', 'NA content']
        file = patient_id_names[ix] + '_table.csv'

        ddf.to_csv(file)



