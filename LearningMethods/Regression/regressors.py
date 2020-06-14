import random
import numpy as np
from scipy.stats import stats, spearmanr
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, ARDRegression, LogisticRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.tree import DecisionTreeRegressor

svr_default_params = {'kernel': 'linear', 'gamma': 'auto', 'C': 0.01}

decision_tree_default_params = {"max_features": 2, "min_samples_split": 4,
                                "n_estimators": 50, "min_samples_leaf": 2}

random_forest_default_params ={"max_depth": 5, "min_samples_split": 4,
                                "n_estimators": 50, "min_samples_leaf": 2}

# -------- general learning functions --------

# main learning loop - same for all basic algorithms
def learning_cross_val_loop(clf, X_train, X_test, y_train, y_test):
    # Split the data set
    train_errors, test_errors, y_train_preds, y_test_preds, des_tree_coefs = [], [], [], [], []
    # FIT
    clf.fit(X_train, y_train)
    # GET RESULTS
    y_pred = clf.predict(X_test)
    return y_test, y_pred


def calc_corr_on_joined_results(y_test, y_pred):
    test_rho, test_p_value = stats.spearmanr(y_test, y_pred)
    test_rmse = mean_squared_error(y_test, y_pred)
    mixed_rho, mixed_pvalues, mixed_rmse = calc_evaluation_on_mix_predictions(y_test, y_pred)

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


def calc_lasso_regression(X_train, X_test, y_train, y_test):
    reg = linear_model.Lasso(alpha=0.01).fit(X_train, y_train)
    reg.score(X_train, y_train)
    return calc_spearmanr_from_regressor(reg, X_test, y_test)


def calc_bayesian_ridge_regression(X_train, X_test, y_train, y_test):
    reg = BayesianRidge().fit(X_train, y_train)
    reg.score(X_train, y_train)
    return calc_spearmanr_from_regressor(reg, X_test, y_test)


def svr_regression(X_train, X_test, y_train, y_test, params=svr_default_params):
    clf = svm.SVR(C=params['C'], kernel=params['kernel'], gamma=params['gamma'])
    y_test, y_test_pred = learning_cross_val_loop(clf, X_train, X_test, y_train, y_test)
    b_1 = clf.coef_[0]
    rho, pvalue, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse = calc_corr_on_joined_results(y_test, y_test_pred)
    return rho, pvalue, b_1, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse, y_test_pred


def decision_tree_regressor(X_train, X_test, y_train, y_test, params=decision_tree_default_params):
    clf = DecisionTreeRegressor(max_features=params["max_features"], min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"])

    y_test, y_test_pred = learning_cross_val_loop(clf, X_train, X_test, y_train, y_test)

    rho, pvalue, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse = calc_corr_on_joined_results(y_test, y_test_pred)
    return rho, pvalue, "    ", mixed_rho, mixed_pvalues, test_rmse, mixed_rmse, y_test_pred


def random_forest_regressor(X_train, X_test, y_train, y_test, params=random_forest_default_params):

    clf = RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                min_samples_split=params["min_samples_split"], min_samples_leaf=params["min_samples_leaf"])

    y_test, y_test_pred = learning_cross_val_loop(clf, X_train, X_test, y_train, y_test)

    rho, pvalue, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse = calc_corr_on_joined_results(y_test, y_test_pred)
    return rho, pvalue, "    ", mixed_rho, mixed_pvalues, test_rmse, mixed_rmse, y_test_pred

