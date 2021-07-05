import pandas as pd
import numpy as np

def conditional_identification(df, dic):
    mask = pd.DataFrame([df[key] == val for key, val in dic.items()]).T.all(axis=1)
    return df[mask]


def dropHighCorr(data, threshold):
    corr = data.corr()
    df_not_correlated = ~(corr.mask(np.tril(np.ones([len(corr)] * 2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = data[un_corr_idx]
    return df_out
