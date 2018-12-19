import pandas as pd
import numpy as np

def cluster_based_on_time(time_series, k=2):
    time_series_sorted = time_series.sort_values()
    return np.array_split(time_series_sorted, k)