import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def cluster_based_on_time(time_series, k=2):
    time_series_sorted = time_series.sort_values()
    delta = time_series_sorted.shape[0]/k