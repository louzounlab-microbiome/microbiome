import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_distance(uncensored_data, censored_data, beta=None, visualize=False):

    def calculate_mse_per_row(row):
        return np.linalg.norm(row)

    def calculate_dist_per_row(row):
        intermidate_matrix = uncensored_data - row
        return intermidate_matrix.apply(calculate_mse_per_row, axis=1)

    dist_matrix = {}
    for subject_id, subject_data in censored_data.items():
        dist_matrix[subject_id] = subject_data.apply(calculate_dist_per_row, axis=1)


    full_dist_matrix = pd.DataFrame()
    for dist_matrix_per_subject in dist_matrix.values():
        full_dist_matrix = full_dist_matrix.append(dist_matrix_per_subject)

    beta = 1 / full_dist_matrix.stack().std() if beta is None else beta

    K = np.exp(-1 * beta * full_dist_matrix)
    if visualize:
        flatten_data = full_dist_matrix.values.flatten()
        first_pop = flatten_data[flatten_data<30]
        plt.hist(first_pop, bins=100)
        plt.title(f'Mean of distances is: {np.mean(first_pop)}')
        plt.xlabel('Distance[microbiome_space]')
        plt.ylabel('Count')

    return K, full_dist_matrix
