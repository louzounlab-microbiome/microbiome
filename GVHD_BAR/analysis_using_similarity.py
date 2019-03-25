from infra_functions.time_series_analsys import compute_time_for_censored_using_similarity_matrix, time_series_using_xgboost, stats_input

import numpy as np
import pickle
from GVHD_BAR.prepare_data import prepare_data as prep_data

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RECORD = True
USE_SIMILARITY = True
REMOVE_OUTLIERS = True
record_inputs = False
use_recorded = True
n_components = 20


def main(use_similarity=USE_SIMILARITY, grid_results_folder='grid_search_xgboost_with_censored'):
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
        if use_similarity:
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

            # change the MSE coeff for the last sample of censored
            y_for_deep_censored['mse_coeff'][censored_data_with_time['time_for_the_event'].index] = 5

            ##### END Similarity algo ####


        starting_col = np.argwhere(x_for_deep.columns == 0).tolist()[0][0]
        X = x_for_deep.iloc[:, starting_col:starting_col + n_components]
        y = y_for_deep['delta_time']

        starting_col = np.argwhere(x_for_deep_censored.columns == 0).tolist()[0][0]
        X_train_censored = x_for_deep_censored.iloc[:, starting_col:starting_col + n_components]
        y_train_censored = y_for_deep_censored['delta_time']
        number_samples_censored = y_train_censored.shape[0]
        print(f'Number of censored subjects: {number_samples_censored}')

        if REMOVE_OUTLIERS:
            # remove outliers
            before_removal = y.shape[0]
            std = y.values.std()
            th = std * 5

            outlier_mask = y < th
            y = y.loc[outlier_mask]
            X = X.loc[outlier_mask]

            after_removal = y.shape[0]
            print(f'{before_removal - after_removal} outlier/s were removed')


        alpha_list = [0.01, 20, 50, 100]
        n_estimators_list = [5, 10, 20]
        min_child_weight_list = [0.1, 1, 10, 20]
        reg_lambda_list = [0, 10, 20]
        max_depth_list = [3, 5, 10]

        # alpha_list = [0.01]
        # n_estimators_list = [5]
        # min_child_weight_list = [0.1]
        # reg_lambda_list = [0]
        # max_depth_list = [3]

        if not use_similarity:
            X_train_censored = None
            y_train_censored = None

        train_res, test_res = time_series_using_xgboost(X, y,
                                                        alpha_list,
                                                        n_estimators_list,
                                                        min_child_weight_list,
                                                        reg_lambda_list,
                                                        max_depth_list,
                                                        cross_val_number=5,
                                                        X_train_censored=X_train_censored,
                                                        y_train_censored=y_train_censored,
                                                        record=RECORD,
                                                        grid_search_dir=grid_results_folder,
                                                        deep_verbose=False,
                                                        beta_for_similarity=beta,
                                                        use_random_time=True)


    total_num_of_configs = len(alpha_list) *\
                           len(n_estimators_list) *\
                           len(min_child_weight_list) *\
                           len(reg_lambda_list) *\
                           len(max_depth_list) *\
                           len(betas_list)
    print(f'Total number of configuration that were checked: {total_num_of_configs}')

if __name__ == '__main__':
    use_similiarity = USE_SIMILARITY
    grid_results_folder = 'grid_search_xgboost_with_censored_test'
    main(use_similiarity, grid_results_folder)
