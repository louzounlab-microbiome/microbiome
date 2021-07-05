import pickle
from pathlib import Path
from Plot.plot_rho import draw_rhos_calculation_figure
from os.path import join
import os
from pandas.api.types import is_numeric_dtype
import pandas as pd

with open(Path('../data/used_data/otumf_data/otumf'), 'rb') as otumf_file:
    otumf = pickle.load(otumf_file)

categorical_limit = 3
for col_name in otumf.extra_features_df:
    folder_name = col_name
    folder_path = join(Path('../visualization/correlation'), folder_name)

    column = otumf.extra_features_df[col_name]
    column_no_nan = column.dropna()
    column_no_nan=pd.to_numeric(column_no_nan,errors='ignore')
    if is_numeric_dtype(column_no_nan):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        try:
            draw_rhos_calculation_figure(otumf.extra_features_df[col_name], otumf.otu_features_df, title=col_name,
                                         taxnomy_level=6, save_folder=folder_path)
        except Exception:
            pass
    elif len(column_no_nan.unique()) <= categorical_limit:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for categorical_value in column_no_nan.unique():
            numeric_series = column_no_nan.apply(lambda x: 1 if x == categorical_value else 0)

            try:
                draw_rhos_calculation_figure(numeric_series, otumf.otu_features_df,
                                             title='{categorical}_{col_name}'.format(categorical=categorical_value,
                                                                                     col_name=col_name),
                                             taxnomy_level=6, save_folder=folder_path)
            except Exception:
                pass
    else:
        pass
