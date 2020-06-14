
import matplotlib.pyplot as plt

from Preprocess.time_series_analsys import compute_time_for_censored_using_similarity_matrix, time_series_analysis_rnn, stats_input

import numpy as np
import pickle
from allergy.prepare_data import prepare_data as prep_data

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RECORD = True
USE_SIMILARITY = False
USE_CENSORED = False
record_inputs = False
use_recorded = True
n_components = 20


def main(use_censored=USE_CENSORED, use_similarity=USE_SIMILARITY, grid_results_folder='rnn_grid_search_no_censored'):
    if not use_recorded:
        x_for_deep, y_for_deep, x_for_deep_censored, y_for_deep_censored, censored_data, not_censored, otu_after_pca_wo_taxonomy, OtuMf = prep_data(n_components)
    else:
        x_for_deep = pickle.load(open(os.path.join(SCRIPT_DIR, "x_for_deep.p"), "rb"))
        y_for_deep = pickle.load(open(os.path.join(SCRIPT_DIR, "y_for_deep.p"), "rb"))
        x_for_deep_censored = pickle.load(open(os.path.join(SCRIPT_DIR, "x_for_deep_censored.p"), "rb"))
        y_for_deep_censored = pickle.load(open(os.path.join(SCRIPT_DIR, "y_for_deep_censored.p"), "rb"))
        censored_data = pickle.load(open(os.path.join(SCRIPT_DIR, "censored_data.p"), "rb"))
        not_censored = pickle.load(open(os.path.join(SCRIPT_DIR, "not_censored.p"), "rb"))
        otu_after_pca_wo_taxonomy = pickle.load(open(os.path.join(SCRIPT_DIR, "otu_after_pca_wo_taxonomy.p"), "rb"))
        OtuMf = pickle.load(open(os.path.join(SCRIPT_DIR, "OtuMf.p"), "rb"))

    if record_inputs:
        pickle.dump(x_for_deep, open(os.path.join(SCRIPT_DIR, "x_for_deep.p"), "wb"))
        pickle.dump(y_for_deep, open(os.path.join(SCRIPT_DIR, "y_for_deep.p"), "wb"))
        pickle.dump(x_for_deep_censored, open(os.path.join(SCRIPT_DIR, "x_for_deep_censored.p"), "wb"))
        pickle.dump(y_for_deep_censored, open(os.path.join(SCRIPT_DIR, "y_for_deep_censored.p"), "wb"))
        pickle.dump(censored_data, open(os.path.join(SCRIPT_DIR, "censored_data.p"), "wb"))
        pickle.dump(not_censored, open(os.path.join(SCRIPT_DIR, "not_censored.p"), "wb"))
        pickle.dump(otu_after_pca_wo_taxonomy, open(os.path.join(SCRIPT_DIR, "otu_after_pca_wo_taxonomy.p"), "wb"))
        pickle.dump(OtuMf, open(os.path.join(SCRIPT_DIR, "OtuMf.p"), "wb"))

    if use_similarity:
        betas_list = [1, 10, 100]
    else:
        betas_list = [None]  # just a list of one element so that the for loop will run only once

    for beta in betas_list:
        censored_mse_fraction_factor = None

        if use_censored:
            y_for_deep_censored['mse_coeff'] = 0

        if use_similarity:
            censored_mse_fraction_factor = 2

            ##### Similarity algo ####
            not_censored_for_similarity = not_censored.join(otu_after_pca_wo_taxonomy)

            censored_data_with_time = compute_time_for_censored_using_similarity_matrix(not_censored_for_similarity,
                                                                                        censored_data,
                                                                                        n_components,
                                                                                        OtuMf,
                                                                                        otu_after_pca_wo_taxonomy,
                                                                                        beta=beta,
                                                                                        remove_outliers=True,
                                                                                        th_value=None)

            # combine the x_censored and the syntethic time
            x_for_deep_censored['time_for_the_event'][censored_data_with_time['time_for_the_event'].index] = \
                censored_data_with_time['time_for_the_event']
            y_for_deep_censored['delta_time'][censored_data_with_time['time_for_the_event'].index] = censored_data_with_time['time_for_the_event']

            # change the MSE coeff for the last sample of censored (its just prep, the actual value will be set within the algo)
            y_for_deep_censored['mse_coeff'][censored_data_with_time['time_for_the_event'].index] = 'last_censored'

            ##### END Similarity algo ####


        starting_col = np.argwhere(x_for_deep.columns == 0).tolist()[0][0]
        X = x_for_deep.iloc[:, starting_col:starting_col + n_components]
        X['groupby'] = x_for_deep['PatientNumber210119']
        y = y_for_deep  # ['delta_time']

        starting_col = np.argwhere(x_for_deep_censored.columns == 0).tolist()[0][0]
        X_train_censored = x_for_deep_censored.iloc[:, starting_col:starting_col + n_components]
        X_train_censored['groupby'] = x_for_deep_censored['PatientNumber210119']
        y_train_censored = y_for_deep_censored
        number_samples_censored = y_train_censored.shape[0]
        print(f'Number of censored subjects: {number_samples_censored}')

        # remove outliers
        before_removal = y.shape[0]
        std = y['delta_time'].values.std()
        th = std * 5

        outlier_mask = y['delta_time'] < th
        y = y.loc[outlier_mask]
        X = X.loc[outlier_mask]

        after_removal = y.shape[0]
        print(f'{before_removal-after_removal} outlier/s were removed')

        stats_input(y, y_train_censored)

        PLOT_INPUT_TO_NN_STATS = False
        if PLOT_INPUT_TO_NN_STATS:
            plt.hist(y['delta_time'].values, bins=150)
            b = y['delta_time'].values.copy()
            b.sort()
            med = b[int(len(b)/2)]
            std = y['delta_time'].values.std()
            mean = y['delta_time'].values.mean()

            plt.title(f'STD={std}, MED={med}, Mean={mean}')

        epochs_list = [20, 100, 1000]#['MAX', 20, 100, 1000]
        mse_factor_list = [0.1, 10, 1000] # np.arange(0.005, 1, 0.005)

        if not use_similarity:
            if not use_censored:
                mse_factor_list = [1]
                X_train_censored = None
                y_train_censored = None



        l2_lambda_list = [1, 20]
        #np.logspace(0, 2, 5) #  0.01, 0.1, 1, 10, 100
        number_layers_list = [1, 2, 3]
        number_neurons_per_layer_list = [20, 50]

        l2_lambda_list = [0.1, 1, 10, 100]
        dropout_list = [0, 0.2, 0.6]  # np.arange(0, 0.8, 0.1)
        epochs_list = [1000]
        number_layers_list = [1, 2, 3]
        number_neurons_per_layer_list = [10, 30]

    train_res, test_res  = time_series_analysis_rnn(X, y,
                                                    n_components,
                                                    l2_lambda_list,
                                                    dropout_list,
                                                    mse_factor_list,
                                                    number_layers_list,
                                                    number_neurons_per_layer_list,
                                                    epochs_list,
                                                    cross_val_number=5,
                                                    X_train_censored=X_train_censored,
                                                    y_train_censored=y_train_censored,
                                                    record=RECORD,
                                                    grid_search_dir=grid_results_folder,
                                                    beta_for_similarity=beta,
                                                    censored_mse_fraction_factor=censored_mse_fraction_factor,
                                                    early_stop_fraction=None, #0.05,
                                                    min_epochs=30)

    total_num_of_configs = len(dropout_list) * \
                           len(l2_lambda_list) * \
                           len(number_layers_list) * \
                           len(number_neurons_per_layer_list) * \
                           len(betas_list)
    print(f'Total number of configuration that were checked: {total_num_of_configs}')

if __name__ == '__main__':
    grid_results_folder = r'C:\Users\Bar\Desktop\testing\allergy_lstm_naive_after_reduction'
    for idx in range(1):
        # for cv in range(5):
        main(USE_CENSORED, USE_SIMILARITY, f'{grid_results_folder}_iter_{idx}')