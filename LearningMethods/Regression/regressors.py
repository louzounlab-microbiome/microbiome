import os
import random
import numpy as np
from scipy.stats import stats, spearmanr
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, ARDRegression, LogisticRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
from LearningMethods.general_functions import shorten_bact_names, shorten_single_bact_name


# -------- general learning functions --------

# main learning loop - same for all basic algorithms
def learning_cross_val_loop(clf, cross_validation, X, y, test_size):
    # Split the data set
    X_trains, X_tests, y_trains, y_tests, des_tree_coefs = [], [], [], [], []
    y_test_from_all_iter, y_score_from_all_iter = np.array([]), np.array([])
    y_pred_from_all_iter, class_report_from_all_iter = np.array([]), np.array([])
    train_errors, test_errors, y_train_preds, \
    y_test_preds = [], [], [], []

    for i in range(cross_validation):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

    for iter_num in range(cross_validation):
        # FIT
        clf.fit(X_trains[iter_num], y_trains[iter_num])
        # GET RESULTS
        y_pred = clf.predict(X_tests[iter_num])
        y_test_preds.append(y_pred)

        # SAVE y_test AND y_score
        y_test_from_all_iter = np.append(y_test_from_all_iter, y_tests[iter_num])
        y_pred_from_all_iter = np.append(y_pred_from_all_iter, list(y_pred))

    return y_tests, y_test_preds


def calc_corr_on_joined_results(cross_validation, y_tests, y_test_preds):
    all_y_train, all_predictions_train, all_test_real_tags, all_test_pred_tags = [], [], [], []

    for i in range(cross_validation):
        all_test_real_tags = all_test_real_tags + list(y_tests[i])
        all_test_pred_tags = all_test_pred_tags + list(y_test_preds[i])
    all_test_pred_tags = np.array(all_test_pred_tags)
    test_rho, test_p_value = stats.spearmanr(all_test_real_tags, all_test_pred_tags)
    test_rmse = mean_squared_error(all_test_real_tags, all_test_pred_tags)

    mixed_rho, mixed_pvalues, mixed_rmse = calc_evaluation_on_mix_predictions(all_test_pred_tags, all_test_real_tags)

    return test_rho, test_p_value, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse


# not working
def get_significant_bacteria_for_model_using_b(b_1, bacteria, task_num, reg_type, significant_bacteria_all_models_df, tax):
    # the most significant b values are those in the edges
    model_bact = shorten_single_bact_name(bacteria[task_num])
    upper_bound = np.percentile(b_1, 95)
    lower_bound = np.percentile(b_1, 5)
    df = pd.DataFrame(columns=["bacteria", "beta value"])
    df["bacteria"] = bacteria
    df["beta value"] = b_1
    df.sort_values(by=["beta value"])

    short_feature_names, bacterias = shorten_bact_names(bacteria)
    significant_bacteria_and_rhos = []

    for i, bact in enumerate(bacterias):
        if b_1[i] < lower_bound or b_1[i] > upper_bound:  # significant
            significant_bacteria_and_rhos.append([bact, b_1[i]])

    for couple in significant_bacteria_and_rhos:
        significant_bacteria_all_models_df.loc[len(significant_bacteria_all_models_df)] = [task_num, couple[0], couple[1]]

    left_padding = 0.4
    fig, ax = plt.subplots()
    y_pos = np.arange(len(significant_bacteria_and_rhos))
    c = [s[1] for s in significant_bacteria_and_rhos]
    coeff_color = []
    for x in c:
        if x >= 0:
            coeff_color.append('green')
        else:
            coeff_color.append('red')
    ax.barh(y_pos, c, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_feature_names)
    plt.yticks(fontsize=10)
    plt.title("Bacteria Number " + str(task_num) + " Model" + "\n" + model_bact +
              "\nSignificant Bacteria According to " + reg_type + " Regression Coefficients")
    ax.set_xlabel("Coeff value")
    fig.subplots_adjust(left=left_padding)
    # plt.show()
    plt.savefig(os.path.join(tax, "bacteria_number_" + str(task_num) + "_model_bacteria_correlation.svg"),
                bbox_inches='tight', format='svg')


def calc_evaluation_on_mix_predictions(y_pred, y_test):
    mixed_y_list, mixed_rhos, mixed_pvalues, mixed_rmse = [], [], [], []
    for num in range(10):  # run 10 times to avoid accidental results
        mixed_y_fred = y_pred.copy()
        random.shuffle(mixed_y_fred)
        mixed_y_list.append(mixed_y_fred)
        rho_, pvalue_ = spearmanr(y_test, mixed_y_fred, axis=None)
        mixed_rhos.append(rho_)
        mixed_pvalues.append(pvalue_)
        mixed_rmse.append(mean_squared_error(y_test, mixed_y_fred))
    return np.array(mixed_rhos).mean(), np.array(mixed_pvalues).mean(), np.array(mixed_rmse).mean()


def calc_spearmanr_from_regressor(reg, X_test, y_test):
    b_1 = reg.coef_
    b_n = reg.intercept_

    # use the b value to decide with bacteria have influence on the tested bacteria
    y_pred = []
    for x in X_test:
        reg_y = np.dot(x, b_1) + b_n
        y_pred.append(reg_y)

    # check if the bacteria change can be predicted according to the spearman correlation
    rho, p_val = spearmanr(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    # check if the predictions are random
    mixed_rho, mixed_pvalues, mixed_rmse = calc_evaluation_on_mix_predictions(y_pred, y_test)

    return rho, p_val, b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse


# -------- many kind of regressors implementations --------
def calc_linear_regression(X_train, X_test, y_train, y_test):
    reg = LinearRegression().fit(X_train, y_train)
    reg.score(X_train, y_train)
    return calc_spearmanr_from_regressor(reg, X_test, y_test)


def calc_ridge_regression(X_train, X_test, y_train, y_test):
    reg = Ridge().fit(X_train, y_train)
    reg.score(X_train, y_train)
    return calc_spearmanr_from_regressor(reg, X_test, y_test)


def calc_ard_regression(X_train, X_test, y_train, y_test):
    reg = ARDRegression().fit(X_train, y_train)
    reg.score(X_train, y_train)
    return calc_spearmanr_from_regressor(reg, X_test, y_test)


def svr_regression(X, y, params, test_size, cross_validation):
    clf = svm.SVR(C=params['C'], kernel=params['kernel'], gamma=params['gamma'])
    y_tests, y_test_preds = learning_cross_val_loop(clf, cross_validation, X, y, test_size)
    b_1 = clf.coef_
    rho, pvalue, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse = calc_corr_on_joined_results(cross_validation, y_tests, y_test_preds)
    return rho, pvalue, b_1, mixed_rho, mixed_pvalues


def decision_tree_regressor(X, y, params, test_size=0.2, cross_validation=5):
    clf = DecisionTreeRegressor(max_features=params["max_features"], min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"])

    y_tests, y_test_preds = learning_cross_val_loop(clf, cross_validation, X, y, test_size)

    rho, pvalue, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse = calc_corr_on_joined_results(cross_validation, y_tests, y_test_preds)
    return rho, pvalue, " ", mixed_rho, mixed_pvalues, test_rmse, mixed_rmse


def random_forest_regressor(X, y, params, test_size=0.2, cross_validation=5):

    clf = RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                min_samples_split=params["min_samples_split"], min_samples_leaf=params["min_samples_leaf"])

    y_tests, y_test_preds = learning_cross_val_loop(clf, cross_validation, X, y, test_size)

    rho, pvalue, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse = calc_corr_on_joined_results(cross_validation, y_tests, y_test_preds)
    return rho, pvalue, " ", mixed_rho, mixed_pvalues, test_rmse, mixed_rmse


def calc_bayesian_ridge_regression(X_train, X_test, y_train, y_test):
    reg = BayesianRidge().fit(X_train, y_train)
    reg.score(X_train, y_train)
    return calc_spearmanr_from_regressor(reg, X_test, y_test)


# -------- start grid searching the right params for each data set --------
def calc_ard_regression_with_params(X_train, X_test, y_train, y_test, alpha):
    reg = ARDRegression().fit(X_train, y_train)
    reg.score(X_train, y_train)
    return calc_spearmanr_from_regressor(reg, X_test, y_test)

