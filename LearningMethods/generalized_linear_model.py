import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


class GLM_coefficients:
    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame, x_relevant_column_names: list, shuffle_times: int
                 , real_coefficient_column_name: str = "Real_coefficient",
                 **kwargs):
        self.x_df = x_df
        self.y_df = y_df
        self.x_relevant_column_names = x_relevant_column_names
        self.shuffle_times = shuffle_times
        self.args_for_regression = kwargs
        self.coeff_dict = {}
        self.real_coefficients_name= real_coefficient_column_name
        for x_relevant_column_name in x_relevant_column_names:
            relevant_column_index = list(self.x_df.columns).index(x_relevant_column_name)
            # find the original coefficients of the column
            original_coefficients_series = self._get_coefficients(relevant_column_index)
            # find the coefficients of the table with the shuffled column
            shuffled_coefficients = []
            for shuffle_number in range(shuffle_times):
                shuffled_df = self._shuffle(x_relevant_column_name)
                shuffled_coefficients.append(self._get_coefficients(relevant_column_index))
            coefficients_df = pd.DataFrame(shuffled_coefficients, columns=list(range(shuffle_number)),
                                           index=self.y_df.index)
            coefficients_df[self.real_coefficients_name] = original_coefficients_series
            self.coeff_dict[x_relevant_column_name].append(coefficients_df)
            shuffled_df = self.x_df

    def _get_coefficients(self, relevant_column_index: int, save_index=True):
        coeff = LinearRegression(**self.args_for_regression).fit(self.x_df, self.y_df).coef_
        if save_index:
            return pd.Series(coeff[:relevant_column_index], index=self.y_df.index)
        else:
            return coeff

    def _shuffle(self, column_name_to_shuffle: str):
        df_with_shuffled_column = self.x_df.copy()
        df_with_shuffled_column[column_name_to_shuffle] = np.random.permutation(
            df_with_shuffled_column[column_name_to_shuffle].values)
        return df_with_shuffled_column

    def get_most_significant_coefficients(self,x_column_name,percentile):
        coefficients_df = self.coeff_dict[x_column_name]
        all_coefficients = coefficients_df.values.flatten()
        upper_bound = np.percentile(all_coefficients, 100 - percentile)
        lower_bound = np.percentile(all_coefficients, percentile)
        real_upper_significant_coefficients = coefficients_df[self.real_coefficients_name] >= upper_bound
        real_lower_significant_coefficients = coefficients_df[self.real_coefficients_name] <= upper_bound

        all_real_significant_coefficients = pd.concat([real_upper_significant_coefficients,real_lower_significant_coefficients])
        return all_real_significant_coefficients





