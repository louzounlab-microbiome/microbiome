from pathlib import Path
import pandas as pd
import numpy as np
import os
import copy
from sklearn.metrics import roc_auc_score
from LearningMethods.Regression.regressors import calc_linear_regression
from LearningMethods.Regression.regressors import calc_ridge_regression, calc_ard_regression, svr_regression
from LearningMethods.Regression.regressors import decision_tree_regressor, random_forest_regressor
from Microbiome_Intervention.create_learning_data_from_data_set import get_X_y_from_file_path, \
    get_adapted_X_y_for_wanted_learning_task


def conclude_results(df, auc, conclusionss_path):
    # algorithms_list = ["linear regression", "ridge regression", "ard regression", "svr", "decision tree",
    #                    "random forest"]
    algorithms_list = ["ard regression"]
    with open(conclusionss_path, "w") as conclusion:
        conclusion.write("algorithm,rhos mean,rhos median,rhos std,p value mean,rmse mean,rmse median,rmse std,auc\n")
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
                             "," + str(round(rmse_median, 5)) + "," + str(round(rmse_std, 5)) +
                             "," + str(round(auc, 5))

                             + "\n" + a + " - random," + str(round(random_rhos_mean, 5)) + "," + str(round(random_rhos_median, 5))
                             + "," + str(round(random_rhos_std, 5)) + "," + str(round(random_p_val_mean, 5))
                             + "," + str(round(random_rmse_mean, 5)) + "," + str(round(random_rmse_median, 5))
                             + "," + str(round(random_rmse_std, 5)) +"\n")


def create_data_frames(all_res_path, important_bacteria_reults_path):
    all_times_all_bacteria_all_models_results_df = Path(all_res_path)
    if not all_times_all_bacteria_all_models_results_df.exists():
        all_times_all_bacteria_all_models_results_df = pd.DataFrame(columns=['BACTERIA_NUMBER', 'BACTERIA', 'ALGORITHM',
                                                                             'RHO', 'RANDOM_RHO', 'P_VALUE',
                                                                             'RANDOM_P_VALUE', 'RMSE', 'RANDOM_RMSE',
                                                                             'PARAMS', 'BETA'])
        all_times_all_bacteria_all_models_results_df.to_csv(all_res_path, index=False)

    important_bacteria_reults_df = Path(important_bacteria_reults_path)
    if not important_bacteria_reults_df.exists():
        important_bacteria_reults_df = pd.DataFrame(columns=['bacteria', 'auc',
                                                             'true_positive', 'true_negative',
                                                             'false_positive', 'false_negative',
                                                             'acc', 'precision', 'recall',
                                                             'specificity', 'sensitivity', 'balanced acc', 'F1'])
        important_bacteria_reults_df.to_csv(important_bacteria_reults_path, index=False)


def run_all_types_of_regression(X_trains, X_tests, y_trains, y_tests, bacteria, i, tax,
                                all_times_all_bacteria_all_models_results_df, all_times_all_bact_results_path,
                                bact):
    bacteria_list = copy.deepcopy(bacteria)

    for X_train, X_test, y_train, y_test in zip(X_trains, X_tests, y_trains, y_tests):
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


def preform_reggression_learning(tax, bacteria, task, X_y_files_list_path, k_fold, test_size):
    # create data frames

    all_times_all_bact_results_path = os.path.join(tax, task + "_" + str(k_fold) +
                                                   "_fold_test_size_" + str(test_size)
                                                   + "_results_df.csv")
    important_bacteria_reults_path = os.path.join(tax,  task + "_" + str(k_fold) +
                                                  "_fold_test_size_" + str(test_size)
                                                  + "_significant_bacteria_prediction_results_df.csv")
    conclusionss_path = os.path.join(tax,  task + "_" + str(k_fold) +
                                     "_fold_test_size_" + str(test_size) + "_conclusions.csv")

    with open(os.path.join(tax, "bacteria.txt"), "r") as b_file:
        bacteria = b_file.readlines()
        bacteria = [b.rstrip() for b in bacteria]

    create_data_frames(all_res_path=all_times_all_bact_results_path, important_bacteria_reults_path=important_bacteria_reults_path)

    with open(os.path.join(tax, X_y_files_list_path), "r") as file:
        paths = file.readlines()
        paths = [p.strip('\n') for p in paths]

    train_binary_significant_from_all_bacteria = []
    test_b_list_from_all_bacteria = []
    for i, [bact, path] in enumerate(zip(bacteria, paths)):
        all_times_all_bacteria_all_models_results_df = pd.read_csv(all_times_all_bact_results_path)
        important_bacteria_reults_df = pd.read_csv(important_bacteria_reults_path)

        # optional tasks!!
        # 1) run multiple regressors to select the most successful regressor, Xt is used to forecast Xt+1
        if task == "run_all_types_of_regression":
            X_trains, X_tests, y_trains, y_tests, name = \
                get_adapted_X_y_for_wanted_learning_task(tax, path, "regular", k_fold, test_size)
            run_all_types_of_regression(X_trains, X_tests, y_trains, y_tests, bacteria, i, tax,
                                        all_times_all_bacteria_all_models_results_df,
                                        all_times_all_bact_results_path, bact)

        else:
            # 2) prediction of interaction network structure, Xt is used to forecast Xt+1
            if task == "interaction_network_structure":
                X_trains, X_tests, y_trains, y_tests, name = \
                    get_adapted_X_y_for_wanted_learning_task(tax, path, "regular", k_fold, test_size)

            # 3) prediction of interaction network structure while hiding 6 time points from train set and using them
            # as test set, Xt is used to forecast Xt+1.
            if task == "hidden_measurements":
                X_trains, X_tests, y_trains, y_tests, name = \
                    get_adapted_X_y_for_wanted_learning_task(tax, path, "hidden_measurements", k_fold, test_size)

            results_df, train_binary_significant_list, test_b_list = \
                predict_interaction_network_structure([X_trains], [X_tests], [y_trains], [y_tests], i,
                                                         all_times_all_bacteria_all_models_results_df,
                                                         all_times_all_bact_results_path, important_bacteria_reults_df,
                                                         important_bacteria_reults_path, bact, bacteria)
            train_binary_significant_from_all_bacteria.append(train_binary_significant_list)
            test_b_list_from_all_bacteria.append(test_b_list)

            # process results -
            train_binary_significant_from_all_bacteria = list(np.array(train_binary_significant_from_all_bacteria).flat)
            test_b_list_from_all_bacteria = list(np.array(test_b_list_from_all_bacteria).flat)
            total_auc = roc_auc_score(y_true=train_binary_significant_from_all_bacteria,
                                      y_score=test_b_list_from_all_bacteria,
                                      average='micro')

            all_times_all_bacteria_all_models_results_df = pd.read_csv(all_times_all_bact_results_path)
            conclude_results(all_times_all_bacteria_all_models_results_df, total_auc, conclusionss_path)
            print(all_times_all_bacteria_all_models_results_df.head())





def predict_interaction_network_structure(X_trains, X_tests, y_trains, y_tests, i, results_df, results_path,
                                          important_bacteria_reults_df, important_bacteria_reults_path,
                                          bact, bacteria_list):

    # create a df that represents the binary significant of bacteria - 0/1 according to train set
    train_binary_significant_df = pd.DataFrame(index=range(len(X_trains)), columns=bacteria_list)
    # create a df that saves the continuous b value of each bacteria according to test set
    test_b_df = pd.DataFrame(index=range(len(X_trains)), columns=bacteria_list)

    for cross_val_i, [X_train, X_test, y_train, y_test] in enumerate(zip(X_trains, X_tests, y_trains, y_tests)):
        true_positive = 0  # train -> significant + test -> significant
        false_negative = 0  # train -> significant + test -> not significant
        false_positive = 0  # train -> not significant + test -> significant
        true_negative = 0  # train -> not significant + test -> not significant

        # TRAIN
        params = ""
        rho, pvalue, train_b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse, y_pred = \
            calc_ard_regression(X_train, X_test, y_train, y_test)
        b = list(train_b_1).__str__()
        b = b.replace(", ", ";")[1:-1]
        results_df.loc[len(results_df)] = \
            [i, bact, "ard regression", rho,  mixed_rho, pvalue, mixed_pvalues, rmse, mixed_rmse, params, b]

        mean = np.mean(train_b_1)
        std = np.std(train_b_1)
        important_b_list = []
        for b_i, b in enumerate(train_b_1):  # iterate rhos
            if b > mean + (2 * std) or b < mean - (2 * std):
                important_b_list.append(1)
            else:
                important_b_list.append(0)
        train_binary_significant_df.loc[cross_val_i] = important_b_list

        # TEST
        rho, pvalue, test_b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse, y_pred = \
            calc_ard_regression(X_test, X_train, y_test, y_train)
        test_b_df.loc[cross_val_i] = test_b_1

        mean = np.mean(test_b_1)
        std = np.std(test_b_1)
        for b_i, b in enumerate(test_b_1):  # iterate rhos
            if b > mean + (2 * std) or b < mean - (2 * std):
                if important_b_list[b_i]:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if important_b_list[b_i]:
                    false_negative += 1
                else:
                    true_negative += 1

        # SCORE
        acc = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        try:
            precision = true_positive / (true_positive + false_positive)
        except:
            precision = 0
        try:
            recall = true_positive / (true_positive + false_negative)
        except:
            recall = 0
        try:
            specificity = true_negative / (true_negative + false_positive)  # actual negatives that are correctly identified
        except:
            specificity = 0
        try:
            sensitivity = true_positive / (true_negative + false_negative)  # actual positives that are correctly identified
        except:
            sensitivity = 0
        balanced_acc = (specificity + sensitivity) / 2
        try:
            F1 = 2 * ((precision * recall) / (precision + recall))
        except:
            F1 = 0
        try:
            auc = roc_auc_score(y_true=important_b_list, y_score=abs(test_b_1), average='micro')
        except:
            auc = 0
            print("ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.")
        important_bacteria_reults_df.loc[len(important_bacteria_reults_df)] = [bact, auc, true_positive, true_negative,
                                                                               false_positive, false_negative,
                                                                               acc, precision, recall,
                                                                               specificity, sensitivity, balanced_acc,
                                                                               F1]

    # claculate AUC - y_true = 0/1 significant or not according to train set
    # y_score = b values according to test set
    total_auc = roc_auc_score(y_true=train_binary_significant_df.values, y_score=abs(test_b_df.values), average='micro')
    print(results_path + " auc=" + str(total_auc))
    important_bacteria_reults_df.loc[len(important_bacteria_reults_df)] = \
        [bact, total_auc,
         important_bacteria_reults_df['true_positive'].mean(),
         important_bacteria_reults_df['true_negative'].mean(),
         important_bacteria_reults_df['false_positive'].mean(),
         important_bacteria_reults_df['false_negative'].mean(),
         important_bacteria_reults_df['acc'].mean(),
         important_bacteria_reults_df['precision'].mean(),
         important_bacteria_reults_df['recall'].mean(),
         important_bacteria_reults_df['specificity'].mean(),
         important_bacteria_reults_df['sensitivity'].mean(),
         important_bacteria_reults_df['balanced acc'].mean(),
         important_bacteria_reults_df['F1'].mean()]
    # update results
    important_bacteria_reults_df.to_csv(important_bacteria_reults_path, index=False)
    results_df.to_csv(results_path, index=False)
    return results_df, train_binary_significant_df.values, abs(test_b_df.values)







