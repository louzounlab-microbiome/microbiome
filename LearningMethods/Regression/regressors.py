import os
import random
import numpy as np
from scipy.stats import stats, spearmanr
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, ARDRegression, LogisticRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
from LearningMethods.general_functions import shorten_bact_names, shorten_single_bact_name


# -------- general learning functions --------

# main learning loop - same for all basic algorithms
def learning_cross_val_loop(clf, cross_validation, X, y, test_size):
    # Split the data set
    y_test_from_all_iter = np.array([])
    train_errors, test_errors, y_train_preds, y_test_preds, des_tree_coefs = [], [], [], [], []

    # devide to train and test-
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    if type(cross_validation) == int:
        for n in range(cross_validation):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
            X_trains.append(X_train)
            X_tests.append(X_test)
            y_trains.append(y_trains)
            y_tests.append(y_tests)
    elif cross_validation == "loo":  # hold-one-subject-out cross-validation
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            X_trains.append(X_train)
            X_tests.append(X_test)
            y_trains.append(y_trains)
            y_tests.append(y_tests)

    for X_train, X_test, y_train, y_test in zip(X_trains, X_tests, y_trains, y_tests):
        # FIT
        clf.fit(X_train, y_train)
        # GET RESULTS
        y_pred = clf.predict(X_test)
        y_test_preds.append(y_pred)

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

    return rho, p_val, b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse, y_pred


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
