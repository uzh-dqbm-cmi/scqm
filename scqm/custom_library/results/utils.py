import numpy as np


def naive_baseline(df):
    df['naive'][1:] = df.targets[:-1]
    return df

def get_naive_baseline(result_df):
    tmp = result_df.copy()
    tmp['naive'] = np.nan
    tmp = tmp.groupby('patient_id').apply(naive_baseline).dropna()
    mse = sum((tmp.targets - tmp.naive) ** 2) / len(tmp)
    return mse