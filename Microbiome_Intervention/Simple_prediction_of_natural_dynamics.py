from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import os
import copy

from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, accuracy_score
from LearningMethods.Regression.regressors import calc_linear_regression, calc_lasso_regression, \
    calc_bayesian_ridge_regression
from LearningMethods.Regression.regressors import calc_ridge_regression, calc_ard_regression, svr_regression
from LearningMethods.Regression.regressors import decision_tree_regressor, random_forest_regressor
from Microbiome_Intervention.Create_learning_data_from_data_set import get_X_y_from_file_path, \
    get_adapted_X_y_for_wanted_learning_task

reg_name_to_func_map = {
    "decision tree regression": decision_tree_regressor,
    "random forest regression": random_forest_regressor,
    "ard regression": calc_ard_regression,
    "lasso regression": calc_lasso_regression,
    "linear regression": calc_linear_regression,
    "ridge regression": calc_ridge_regression,
    "bayesian ridge regression": calc_bayesian_ridge_regression,
    "svr regression": svr_regression
}


def conclude_results(df, auc, conclusions_path):
    """
    :param df: (DataFrame) in path 'all_times_all_bact_results_path' filled by the run_all_types_of_regression' function
    :param auc: (float) auc score
    :param conclusions_path: path to save the results.
    :return:
    """
    algorithms_list = ["linear regression", "ridge regression", "ard regression",
                       "lasso regression",  "bayesian ridge regression", "svr regression",
                       "decision tree regression", "random forest regression"]
    print("concluding")
    with open(conclusions_path, "w") as conclusion:
        conclusion.write("Model,rhos mean,rhos median,rhos std,p value mean,p_val_std,mse mean,mse median,mse std,auc\n")
        for a in algorithms_list:
            rhos_mean = df[df["ALGORITHM"] == a]['RHO'].mean()
            rhos_median = df[df["ALGORITHM"] == a]['RHO'].median()
            rhos_std = df[df["ALGORITHM"] == a]['RHO'].std()
            p_val_mean = df[df["ALGORITHM"] == a]['P_VALUE'].mean()
            p_val_std = df[df["ALGORITHM"] == a]['P_VALUE'].std()
            mse_mean = df[df["ALGORITHM"] == a]['MSE'].mean()
            mse_median = df[df["ALGORITHM"] == a]['MSE'].median()
            mse_std = df[df["ALGORITHM"] == a]['MSE'].std()
            random_rhos_mean = df[df["ALGORITHM"] == a]['RANDOM_RHO'].mean()
            random_rhos_median = df[df["ALGORITHM"] == a]['RANDOM_RHO'].median()
            random_rhos_std = df[df["ALGORITHM"] == a]['RANDOM_RHO'].std()
            random_p_val_mean = df[df["ALGORITHM"] == a]['RANDOM_P_VALUE'].mean()
            random_p_val_std = df[df["ALGORITHM"] == a]['RANDOM_P_VALUE'].std()
            random_mse_mean = df[df["ALGORITHM"] == a]['RANDOM_MSE'].mean()
            random_mse_median = df[df["ALGORITHM"] == a]['RANDOM_MSE'].median()
            random_mse_std = df[df["ALGORITHM"] == a]['RANDOM_MSE'].std()

            conclusion.write(a + "," + str(round(rhos_mean, 5)) + "," + str(round(rhos_median, 5)) + "," + str(round(rhos_std, 5)) +
                             "," + str(round(p_val_mean, 5)) + "," + str(round(p_val_std, 5)) +
                             "," + str(round(mse_mean, 5)) + "," + str(round(mse_median, 5)) +
                             "," + str(round(mse_std, 5)) + "," + str(round(auc, 5))

                             + "\n" + a + " - random," + str(round(random_rhos_mean, 5)) + "," + str(round(random_rhos_median, 5))
                             + "," + str(round(random_rhos_std, 5)) + "," + str(round(random_p_val_mean, 5))
                             + str(round(random_p_val_std, 5)) + "," + str(round(random_mse_mean, 5))
                             + "," + str(round(random_mse_median, 5)) + "," + str(round(random_mse_std, 5)) +"\n")

    df = df.groupby(["BACTERIA"]).mean()
    df["MSE_DIFFERENCE"] = df["MSE"] - df["RANDOM_MSE"]
    df = df.sort_values(by=['MSE_DIFFERENCE'])
    df.to_csv(conclusions_path.replace("conclusions", "bacteria_conclusions"))


def create_data_frames(all_res_path, important_bacteria_reults_path):
    """
    An auxiliary function for creating prime data in the provided paths.
    :param all_res_path: (string) path for all results for each bacteria data frame
    :param important_bacteria_reults_path: (string) path for bacteria AUC score data frame
    """
    all_times_all_bacteria_all_models_results_df = Path(all_res_path)
    if not all_times_all_bacteria_all_models_results_df.exists():
        all_times_all_bacteria_all_models_results_df = pd.DataFrame(columns=['BACTERIA_NUMBER', 'BACTERIA', 'ALGORITHM',
                                                                             'RHO', 'RANDOM_RHO', 'P_VALUE',
                                                                             'RANDOM_P_VALUE', 'MSE', 'RANDOM_MSE',
                                                                             'BETA'])
        all_times_all_bacteria_all_models_results_df.to_csv(all_res_path, index=False)

    important_bacteria_reults_df = Path(important_bacteria_reults_path)
    if not important_bacteria_reults_df.exists():
        important_bacteria_reults_df = pd.DataFrame(columns=['bacteria', 'auc'])
        important_bacteria_reults_df.to_csv(important_bacteria_reults_path, index=False)


def run_all_types_of_regression(X_trains, X_tests, y_trains, y_tests, i,
                                all_times_all_bacteria_all_models_results_df, all_times_all_bact_results_path,
                                bact):
    """
    For the requested bacterium, for each X, y fold, run regression for any existing regression type and write data in
    data frame.
    :param X_trains: (list) list of numpy Array shaped: (number of train samples, number of bacteria)
    :param X_tests: (list) list of numpy Array shaped: (number of test samples, number of bacteria)
    :param y_trains: (list) list of numpy Array shaped: (number of train samples)
    :param y_tests: (list) list of numpy Array shaped: (number of test samples)
    :param i: (int) bacteria number.
    :param all_times_all_bacteria_all_models_results_df:
    :param all_times_all_bact_results_path:
    :param bact: (string) bacteria name.
    :return:  doesn't return an object, add bacteria i results to csv file at path 'all_times_all_bact_results_path'
    after filling the file for each bacteria - calculate the average scores using 'conclude_results' function
    """
    for X_train, X_test, y_train, y_test in zip(X_trains, X_tests, y_trains, y_tests):
        for model_name, model_clf in reg_name_to_func_map.items():
            rho, pvalue, b_1, mixed_rho, mixed_pvalues, mse, mixed_mse, y_pred = \
                model_clf(X_train, X_test, y_train, y_test)
            b = list(b_1).__str__()
            b = b.replace(", ", ";")[1:-1]
            all_times_all_bacteria_all_models_results_df.loc[len(all_times_all_bacteria_all_models_results_df)] = \
                [i, bact, model_name, rho, mixed_rho, pvalue, mixed_pvalues, mse, mixed_mse, " ", b]
        # update results
        all_times_all_bacteria_all_models_results_df.to_csv(all_times_all_bact_results_path, index=False)


def predict_interaction_network_structure_using_coeffs(X_trains, X_tests, y_trains, y_tests, i, results_df, results_path,
                                                       important_bacteria_reults_df, important_bacteria_reults_path,
                                                       bact, bacteria_list, reg_type):
    """
    Simple model - feature importance calculation-
    In linear models, it is easy to describe the relationship between the two bacteria using the model coefficients
    size and sign.
    The more extreme the coefficient, the more the respective feature of the coefficient have an effect on the
    prediction and hence it can be concluded that there is an interaction between the two.
    Given a regressor that predicts a specific bacterial change based on all the bacteria present in the sample,
    the regressor 's coeffinces can be extracted and used to infer the importance of each bacterium in the prediction
    task, and thus indicate interactions between bacteria- competitive or symbiosis.
    Evaluation-
    In order to evaluate the prediction of the bacterial interaction network, we tested whether a relationship between
    bacteria identified in the training set was correctly predicted in the test set.
    Therefore we needed to the extreme b values that belong to bacteria who have a significant - positive or negative
    role in the change over time in the bacteria being studied.
    In order to do that, we split the data into two sets, train and test.
    For the train set, we calculated the mean and std of the coefficients.
    The coefficient values which were 2 std step up or down from the mean concluded significant.
    For the test set, we trained the model on the new unseen data and saved the coefficient value which were obtained
    from the model.
    The AUC calculation will compare the binary value learned from the training set (significant\non-significant) and
    the predicted continuous interaction value from the test set and output the area under the curve.
    the AUC in calculated for each fold and for all the fold together. returned the train binary significant values and
    the absolute values of test b.

    :param X_trains: (list) list of numpy Array shaped: (number of train samples, number of bacteria)
    :param X_tests: (list) list of numpy Array shaped: (number of test samples, number of bacteria)
    :param y_trains: (list) list of numpy Array shaped: (number of train samples)
    :param y_tests: (list) list of numpy Array shaped: (number of test samples)
    :param i: (int) bacteria number.
    :param results_df: (DataFrame) data frame to add to the regression train information.
    :param results_path: path the save the updated data frame 'results_df'.
    :param important_bacteria_reults_df: (DataFrame) data frame to add to the AUC of the prediction of interaction for
    each bacterua.
    :param important_bacteria_reults_path:  path the save the updated data frame 'important_bacteria_reults_df'.
    :param bact: (string) bacteria name.
    :param bacteria_list: (list) list of all bacteria names.
    :param reg_type: regression type - must have coefficients.
    :return:  results_df, train_binary_significant_df.values, abs(test_b_df.values)
    """
    # create a df that represents the binary significant of bacteria - 0/1 according to train set
    train_binary_significant_df = pd.DataFrame(index=range(len(X_trains)), columns=bacteria_list)
    # create a df that saves the continuous b value of each bacteria according to test set
    test_b_df = pd.DataFrame(index=range(len(X_trains)), columns=bacteria_list)

    for cross_val_i, [X_train, X_test, y_train, y_test] in enumerate(zip(X_trains, X_tests, y_trains, y_tests)):
        # TRAIN
        rho, pvalue, train_b_1, mixed_rho, mixed_pvalues, mse, mixed_mse, y_pred = \
            reg_name_to_func_map[reg_type](X_train, X_test, y_train, y_test)

        b = list(train_b_1).__str__()
        b = b.replace(", ", ";")[1:-1]
        results_df.loc[len(results_df)] = \
            [i, bact, reg_type.replace("_", " "), rho,  mixed_rho, pvalue, mixed_pvalues, mse, mixed_mse, b]

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
        rho, pvalue, test_b_1, mixed_rho, mixed_pvalues, mse, mixed_mse, y_pred = \
            reg_name_to_func_map[reg_type](X_test, X_train, y_test, y_train)
        test_b_df.loc[cross_val_i] = test_b_1

        # print(round(roc_auc_score(y_true=important_b_list, y_score=abs(test_b_1)), 4))

    # claculate AUC - y_true = 0/1 significant or not according to train set
    # y_score = b values according to test set
    total_auc = roc_auc_score(y_true=list(train_binary_significant_df.values.flatten()),
                              y_score=list(abs(test_b_df.values).flatten()))

    print("auc=" + str(round(total_auc, 4)))
    important_bacteria_reults_df.loc[len(important_bacteria_reults_df)] = [bact, total_auc]
    # update results
    important_bacteria_reults_df.to_csv(important_bacteria_reults_path, index=False)
    results_df.to_csv(results_path, index=False)
    return results_df, train_binary_significant_df.values, abs(test_b_df.values)


def predict_interaction_network_structure_using_change_in_data_auc_calc_trail(bacteria_list, folder, data_set_name, CHANGE = 0.5):
    k_fold = 1
    test_size = 0.3
    p_value = 0.01
    bacteria_number_list = list(range(len(bacteria_list)))

    for regressor_name, regressor_clf in reg_name_to_func_map.items():
        # create a df that saves a binary value 1/0 => interaction/no interaction according to the train set
        train_binary_significant_df = pd.DataFrame(columns=bacteria_list)
        # create a df that saves the continuous b value of each bacteria according to the test set
        test_b_df = pd.DataFrame(columns=bacteria_list)

        print(regressor_name)
        df_title = os.path.join(folder, regressor_name.replace(" ", "_") + "_interaction_network_change_in_data_df.csv")
        df = pd.DataFrame(columns=["BACTERIA", "CHANGED_BACTERIA", "CHANGE", "Y"])
        df.to_csv(df_title, index=False)

        for b_i, bacteria_num in enumerate(bacteria_number_list):  # for each bacteria
            print(str(b_i) + " / " + str(len(bacteria_list)))
            train_binary_significant_for_b_i = []
            test_1_u_score_for_b_i = []
            df = pd.read_csv(df_title)
            path = "X_y_for_bacteria_number_" + str(bacteria_num) + ".csv"
            X_trains, X_tests, y_trains, y_tests, name = \
                get_adapted_X_y_for_wanted_learning_task(folder, path, "regular", k_fold, test_size)

            for bacteria_to_change_num in bacteria_number_list:
                X_train, X_test, y_train, y_test = X_trains[0], X_tests[0], y_trains[0], y_tests[0]
                # TRAIN
                X_test_positive_change = copy.deepcopy(X_test)
                X_test_negative_change = copy.deepcopy(X_test)

                for s_i, sample in enumerate(X_test_positive_change):
                    X_test_positive_change[s_i][bacteria_to_change_num] += CHANGE
                for s_i, sample in enumerate(X_test_negative_change):
                    X_test_negative_change[s_i][bacteria_to_change_num] -= CHANGE

                # regression
                _, _, _, _, _, _, _, y_pred_no_change = regressor_clf(X_train, X_test, y_train, y_test)
                y_str = ""
                for val in y_pred_no_change:
                    y_str += str(val) + " "
                df.loc[len(df)] = [int(bacteria_num), int(-1), "no change", y_str]
                _, _, _, _, _, _, _, y_pred_pos_change = regressor_clf(X_train, X_test_positive_change, y_train, y_test)
                y_str = ""
                for val in y_pred_pos_change:
                    y_str += str(val) + " "
                df.loc[len(df)] = [int(bacteria_num), int(bacteria_to_change_num), "plus " + str(CHANGE), y_str]
                _, _, _, _, _, _, _, y_pred_neg_change = regressor_clf(X_train, X_test_negative_change, y_train, y_test)
                y_str = ""
                for val in y_pred_neg_change:
                    y_str += str(val) + " "
                df.loc[len(df)] = [int(bacteria_num), int(bacteria_to_change_num), "minus " + str(CHANGE), y_str]

                pos_u, pos_u_test_p_val = mannwhitneyu(y_pred_no_change, y_pred_pos_change)
                neg_u, neg_u_test_p_val = mannwhitneyu(y_pred_no_change, y_pred_neg_change)

                if pos_u_test_p_val < p_value and neg_u_test_p_val < p_value:
                    train_binary_significant_for_b_i.append(1)
                else:
                    train_binary_significant_for_b_i.append(0)

                # TEST
                X_train_positive_change = copy.deepcopy(X_train)
                X_train_negative_change = copy.deepcopy(X_train)

                for s_i, sample in enumerate(X_train_positive_change):
                    X_train_positive_change[s_i][bacteria_to_change_num] += CHANGE
                for s_i, sample in enumerate(X_train_negative_change):
                    X_train_negative_change[s_i][bacteria_to_change_num] -= CHANGE

                # regression
                _, _, _, _, _, _, _, y_pred_no_change = regressor_clf(X_test, X_train, y_test, y_train)

                _, _, _, _, _, _, _, y_pred_pos_change = regressor_clf(X_test, X_train_positive_change, y_test, y_train)

                _, _, _, _, _, _, _, y_pred_neg_change = regressor_clf(X_test, X_train_negative_change, y_test, y_train)


                pos_u, pos_u_test_p_val = mannwhitneyu(y_pred_no_change, y_pred_pos_change)
                neg_u, neg_u_test_p_val = mannwhitneyu(y_pred_no_change, y_pred_neg_change)

                test_1_u_score_for_b_i.append((1 / pos_u, 1 / neg_u))

            # save bacteria b_i results
            df.to_csv(df_title, index=False)
            train_binary_significant_df.loc[len(train_binary_significant_df)] = train_binary_significant_for_b_i
            test_b_df.loc[len(test_b_df)] = test_1_u_score_for_b_i

        # calculate AUC on the flatten data frame
        # positive change tuple[0]
        pos_b = []
        neg_b = []
        for row in test_b_df.values:
            for val in row:
                pos_b.append(float(val[0]))
                neg_b.append(float(val[1]))
        pos_b = np.array(pos_b)
        neg_b = np.array(neg_b)

        train_binary_significant_values = []
        for val in np.array(train_binary_significant_df.values).flatten():
            train_binary_significant_values.append(val)

        train_binary_significant_values = np.array(train_binary_significant_values)
        try:
            pos_auc = roc_auc_score(train_binary_significant_values, pos_b)
            neg_auc = roc_auc_score(train_binary_significant_values, neg_b)

            Networks_AUC_df = pd.read_csv("all_Networks_AUC.csv")
            Networks_AUC_df.loc[len(Networks_AUC_df)] = ["positive change", regressor_name, data_set_name, test_size,
                                                         k_fold,
                                                         pos_auc, datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S")]
            Networks_AUC_df.loc[len(Networks_AUC_df)] = ["negative change", regressor_name, data_set_name, test_size,
                                                         k_fold,
                                                         neg_auc, datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S")]
            Networks_AUC_df.to_csv("all_Networks_AUC.csv", index=False)


        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(len(train_binary_significant_values))
            print(set(train_binary_significant_values))
            print(len(pos_b))
            print(len(set(pos_b)))
            print(len(neg_b))
            print(len(set(neg_b)))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def predict_interaction_network_structure_using_change_in_data(bacteria_list, folder, CHANGE=0.5):
    """
    Complex models - feature importance calculation-
    NN is complex model which function as a black box that cannot be clearly deduced from the contribution of each
    feature.
    A different approach for inferring the importance of each bacterium in these models is required.

    Initially we trained the model to predict the change in each of the bacteria, we used the original data as an input.
    Now, although the equivalent to ‘coefficencts‘ is not visible to us, every bacterium has an effect, between none and
    extreme on the prediction.
    Therefore, we could estimate this effect by introducing modified input into the model and examining its effects on
    prediction.

    We chose to examine the relationship between each bacterial pair by using the existing fixed model that was trained
    for the ‘first’ bacterial prediction, and forward it an input that was modified only for the ‘second’ bacterium.
    U test can be used to investigate whether two independent samples were selected from populations having the same
    distribution. That is, the test can tell whether the distribution of predictions has changed significantly in light
    of the change in input - indicating interacting.
    Comparing the original prediction and the modified data’s prediction distributions, if the change between the two is
    significant according to U test, we conclude that there is interaction between the bacterial pair.

    The type of interaction will be determined by the obtained change- increasing or decreasing the count of the
     bacterium at a fixed size, and its effect, increase or decrease in the prediction of the count of bacteria.

    The change will be by the constant 'CHANGE'
    :param bacteria_list: (list) list of bacteria names.
    :param folder: (string) main dataset folder "DATASET/tax=x"
    :param CHANGE: (float) size of change in bacteria values.
    :return:
    """
    k_fold = 1
    test_size = 0.3
    bacteria_number_list = list(range(len(bacteria_list)))

    for regressor_name, regressor_clf in reg_name_to_func_map.items():
        print(regressor_name)
        df_title = os.path.join(folder, regressor_name.replace(" ", "_") + "_interaction_network_change_in_data_df.csv")
        df = pd.DataFrame(columns=["BACTERIA", "CHANGED_BACTERIA", "CHANGE", "Y"])
        df.to_csv(df_title, index=False)

        for b_i, bacteria_num in enumerate(bacteria_number_list):  # for each bacteria
            print(str(b_i) + " / " + str(len(bacteria_list)))
            df = pd.read_csv(df_title)
            path = "X_y_for_bacteria_number_" + str(bacteria_num) + ".csv"
            X_trains, X_tests, y_trains, y_tests, name = \
                get_adapted_X_y_for_wanted_learning_task(folder, path, "regular", k_fold, test_size)

            for bacteria_to_change_num in bacteria_number_list:
                for X_train, X_test, y_train, y_test in zip(X_trains, X_tests, y_trains, y_tests):
                    X_positive_change = copy.deepcopy(X_test)
                    X_negative_change = copy.deepcopy(X_test)

                    for s_i, sample in enumerate(X_positive_change):
                        X_positive_change[s_i][bacteria_to_change_num] += CHANGE
                    for s_i, sample in enumerate(X_negative_change):
                        X_negative_change[s_i][bacteria_to_change_num] -= CHANGE

                    # regression
                    _, _, _, _, _, _, _, y_pred_no_change = regressor_clf(X_train, X_test, y_train, y_test)
                    y_str = ""
                    for val in y_pred_no_change:
                        y_str += str(val) + " "
                    df.loc[len(df)] = [int(bacteria_num), int(-1), "no change", y_str]
                    _, _, _, _, _, _, _, y_pred_pos_change = regressor_clf(X_train, X_positive_change, y_train, y_test)
                    y_str = ""
                    for val in y_pred_pos_change:
                        y_str += str(val) + " "
                    df.loc[len(df)] = [int(bacteria_num), int(bacteria_to_change_num), "plus " + str(CHANGE), y_str]
                    _, _, _, _, _, _, _, y_pred_neg_change = regressor_clf(X_train, X_negative_change, y_train, y_test)
                    y_str = ""
                    for val in y_pred_neg_change:
                        y_str += str(val) + " "
                    df.loc[len(df)] = [int(bacteria_num), int(bacteria_to_change_num), "minus " + str(CHANGE), y_str]

            df.to_csv(df_title, index=False)
            # update results for each bacteria
