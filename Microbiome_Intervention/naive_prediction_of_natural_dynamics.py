from pathlib import Path
import pandas as pd
import os
import copy
from sklearn.model_selection import train_test_split
from LearningMethods.Regression.regressors import calc_linear_regression, get_significant_bacteria_for_model_using_b
from LearningMethods.Regression.regressors import calc_ridge_regression, calc_ard_regression, svr_regression
from LearningMethods.Regression.regressors import decision_tree_regressor, random_forest_regressor
from Microbiome_Intervention.create_learning_data_from_data_set import get_X_y_from_file_path
from Microbiome_Intervention.significant_bacteria import get_significant_beta_from_file, \
    check_if_bacteria_correlation_is_significant


def conclude_results(tax, df):
    # algorithms_list = ["linear regression", "ridge regression", "ard regression", "svr", "decision tree",
    #                    "random forest"]
    algorithms_list = ["ard regression", "random forest"]
    with open(os.path.join(tax, "best_models_conclusion.csv"), "w") as conclusion:
        conclusion.write("algorithm,rhos mean,rhos median,rhos std,p value mean,rmse mean,rmse median,rmse std\n")
        for a in algorithms_list:
            rhos_mean = df[df["ALGORITHM"] == a]['RHO'].mean()
            rhos_median = df[df["ALGORITHM"] == a]['RHO'].median()
            rhos_std = df[df["ALGORITHM"] == a]['RHO'].std()
            p_val_mean = df[df["ALGORITHM"] == a]['P_VALUE'].mean()
            rmse_mean = df[df["ALGORITHM"] == a]['RMSE'].mean()
            rmse_median = df[df["ALGORITHM"] == a]['RMSE'].median()
            rmse_std = df[df["ALGORITHM"] == a]['RMSE'].std()
            random_rhos_mean = df[df["ALGORITHM"] == a]['RANDOM_RHO'].mean()
            random_rhos_median = df[df["ALGORITHM"] == a]['RANDOM_RHO'].median()
            random_rhos_std = df[df["ALGORITHM"] == a]['RANDOM_RHO'].std()
            random_p_val_mean = df[df["ALGORITHM"] == a]['RANDOM_P_VALUE'].mean()
            random_rmse_mean = df[df["ALGORITHM"] == a]['RANDOM_RMSE'].mean()
            random_rmse_median = df[df["ALGORITHM"] == a]['RANDOM_RMSE'].median()
            random_rmse_std = df[df["ALGORITHM"] == a]['RANDOM_RMSE'].std()

            conclusion.write(a + " - real," + str(round(rhos_mean, 5)) + "," + str(round(rhos_median, 5)) + "," + str(round(rhos_std, 5)) +
                             "," + str(round(p_val_mean, 5)) + "," + str(round(rmse_mean, 5)) +
                             "," + str(round(rmse_median, 5)) + "," + str(round(rmse_std, 5))

                             + "\n" + a + " - random," + str(round(random_rhos_mean, 5)) + "," + str(round(random_rhos_median, 5))
                             + "," + str(round(random_rhos_std, 5)) + "," + str(round(random_p_val_mean, 5))
                             + "," + str(round(random_rmse_mean, 5)) + "," + str(round(random_rmse_median, 5))
                             + "," + str(round(random_rmse_std, 5)) + "\n")


def create_data_frames(all_res_path):
    all_times_all_bacteria_all_models_results_df = Path(all_res_path)
    if not all_times_all_bacteria_all_models_results_df.exists():
        all_times_all_bacteria_all_models_results_df = pd.DataFrame(columns=['BACTERIA_NUMBER', 'BACTERIA', 'ALGORITHM',
                                                                             'RHO', 'RANDOM_RHO', 'P_VALUE',
                                                                             'RANDOM_P_VALUE', 'RMSE', 'RANDOM_RMSE',
                                                                             'PARAMS', 'BETA'])
        all_times_all_bacteria_all_models_results_df.to_csv(all_res_path, index=False)

def run_all_types_of_regression(X, y, test_size, bacteria, i, tax,
                                all_times_all_bacteria_all_models_results_df, all_times_all_bact_results_path,
                                bact):
    # devide to train and test-
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    bacteria_list = copy.deepcopy(bacteria)

    # linear regression
    rho, pvalue, b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse = calc_linear_regression(X_train, X_test, y_train, y_test)
    b = list(b_1).__str__()
    b = b.replace(", ", ";")[1:-1]
    all_times_all_bacteria_all_models_results_df.loc[len(all_times_all_bacteria_all_models_results_df)] = \
        [i, bact, "linear regression", rho, mixed_rho, pvalue, mixed_pvalues, rmse, mixed_rmse, " ", b]


    # ridge regression
    rho, pvalue, b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse = calc_ridge_regression(X_train, X_test, y_train, y_test)
    b = list(b_1).__str__()
    b = b.replace(", ", ";")[1:-1]
    all_times_all_bacteria_all_models_results_df.loc[len(all_times_all_bacteria_all_models_results_df)] = \
        [i, bact, "ridge regression", rho, mixed_rho, pvalue, mixed_pvalues, rmse, mixed_rmse, " ", b]


    # ard regression
    rho, pvalue, b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse = calc_ard_regression(X_train, X_test, y_train, y_test)
    b = list(b_1).__str__()
    b = b.replace(", ", ";")[1:-1]
    all_times_all_bacteria_all_models_results_df.loc[len(all_times_all_bacteria_all_models_results_df)] = \
        [i, bact, "ard regression", rho, mixed_rho, pvalue, mixed_pvalues, rmse, mixed_rmse, " ", b]

    # SVR
    svr_tuned_parameters = {'kernel': 'linear',  # , 'rbf', 'poly', 'sigmoid'],
                            'gamma': 'auto',  # , 'scale'],
                            'C': 0.01}  # , 0.1, 1, 10, 100, 1000]}
    rho, pvalue, b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse = svr_regression(X, y, svr_tuned_parameters, test_size=0.2, cross_validation=5)
    b = list(b_1[0]).__str__()
    b = b.replace(", ", ";")[1:-1]
    all_times_all_bacteria_all_models_results_df.loc[len(all_times_all_bacteria_all_models_results_df)] = \
        [i, bact, "svr", rho, mixed_rho, pvalue, mixed_pvalues, rmse, mixed_rmse, str(svr_tuned_parameters).replace(",", ";"), b]


    # decision trees
    decision_tree_params = {"max_features": 2, "min_samples_split": 4,
                            "n_estimators": 50, "min_samples_leaf": 2}
    rho, pvalue, b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse = decision_tree_regressor(X, y, decision_tree_params, test_size=0.2, cross_validation=5)
    all_times_all_bacteria_all_models_results_df.loc[len(all_times_all_bacteria_all_models_results_df)] = \
        [i, bact, "decision tree", rho, mixed_rho, pvalue, mixed_pvalues, rmse, mixed_rmse, str(decision_tree_params).replace(",", ";"), b_1]

    # random_forest_regressor
    random_forest_params = {"max_depth": 5, "min_samples_split": 4,
                            "n_estimators": 50, "min_samples_leaf": 2}
    rho, pvalue, b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse = random_forest_regressor(X, y, random_forest_params, test_size=0.2, cross_validation=5)
    all_times_all_bacteria_all_models_results_df.loc[len(all_times_all_bacteria_all_models_results_df)] = \
        [i, bact, "random forest", rho, mixed_rho, pvalue, mixed_pvalues, rmse, mixed_rmse, str(random_forest_params).replace(",", ";"), b_1]

    # update results
    all_times_all_bacteria_all_models_results_df.to_csv(all_times_all_bact_results_path, index=False)


def preform_learning(tax, bacteria, X_y_files_list_path, test_size=0.2):
    # create data frames
    all_times_all_bact_results_path = os.path.join(tax, "all_times_all_bacteria_best_models_results_df.csv")

    create_data_frames(all_res_path=all_times_all_bact_results_path)

    with open(os.path.join(tax, X_y_files_list_path), "r") as file:
        paths = file.readlines()
        paths = [p.strip('\n') for p in paths]

    for i, [bact, path] in enumerate(zip(bacteria, paths)):
        all_times_all_bacteria_all_models_results_df = pd.read_csv(all_times_all_bact_results_path)
        X, y, name = get_X_y_from_file_path(tax, path)

        """
        run_all_types_of_regression(X, y, test_size, bacteria, i, tax,
                                    all_times_all_bacteria_all_models_results_df, all_times_all_bact_results_path, bact)
        """
        run_all_params_combinations_ard_random_forest(X, y, test_size, i, all_times_all_bacteria_all_models_results_df,
                                    all_times_all_bact_results_path, bact)
    # process results -
    all_times_all_bacteria_all_models_results_df = pd.read_csv(all_times_all_bact_results_path)
    conclude_results(tax, all_times_all_bacteria_all_models_results_df)
    print(all_times_all_bacteria_all_models_results_df.head())


def run_all_params_combinations_ard_random_forest(X, y, test_size, i, results_df, results_path, bact):
    # devide to train and test-
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    # ard regression
    params = ""
    rho, pvalue, b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse = calc_ard_regression(X_train, X_test, y_train, y_test)
    b = list(b_1).__str__()
    b = b.replace(", ", ";")[1:-1]
    results_df.loc[len(results_df)] = \
        [i, bact, "ard regression", rho,  mixed_rho, pvalue, mixed_pvalues, rmse, mixed_rmse, params, b]

    # random_forest_regressor
    random_forest_params = {"max_depth": 5, "min_samples_split": 4,
                            "n_estimators": 50, "min_samples_leaf": 2}
    rho, pvalue, b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse = random_forest_regressor(X, y, random_forest_params, test_size=0.2,
                                                                         cross_validation=5)
    results_df.loc[len(results_df)] = \
        [i, bact, "random forest", rho,  mixed_rho, pvalue, mixed_pvalues, rmse, mixed_rmse,
         str(random_forest_params).replace(",", ";"), b_1]


    # update results
    results_df.to_csv(results_path, index=False)
    return results_df


"""
def preform_ard_10_fold_cross_validation(tax, bacteria, X_y_files_list_path, test_size=0.2):
    # create data frames
    results_path = os.path.join(tax, "ard_regression_results_df.csv")
    significant_bacteria_path = os.path.join(tax, "ard_regression_significant_bacteria_df.csv")

    results_df = Path(results_path)
    if not results_df.exists():
        results_df = pd.DataFrame(columns=['BACTERIA_NUMBER', 'BACTERIA', 'ALGORITHM',
                                                                             'RHO', 'P_VALUE', 'RANDOM_RHO',
                                                                             'RANDOM_P_VALUE', 'RHOS_DIFFERANCE',
                                                                             'PARAMS', 'BETA'])
        results_df.to_csv(results_path, index=False)


    with open(os.path.join(tax, X_y_files_list_path), "r") as file:
        paths = file.readlines()
        paths = [p.strip('\n') for p in paths]
    for i, [bact, path] in enumerate(zip(bacteria, paths)):
        all_times_all_bacteria_all_models_results_df = pd.read_csv(results_path)
        X, y, name = get_X_y_from_file_path(tax, path)

        results_df = run_all_params_combinations(X, y, test_size, bacteria, i, tax,
                                                 results_df, results_path, bact)

    # proccess results -
    all_times_all_bacteria_all_models_results_df = conclude_results(tax, results_df)
    print(all_times_all_bacteria_all_models_results_df.head())

    print("!")
"""








