import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

def visualize_prediction(folder_to_use, config_to_use, ax_title, fig_title=None, intercept=None):
    folder_to_use = os.path.join(folder_to_use, config_to_use)

    pass

    if not os.path.exists(os.path.join(folder_to_use,'y_train_values.npy')):
        train_y = np.empty(0)
        train_y_predicted = np.empty(0)
        test_y = np.empty(0)
        test_y_predicted = np.empty(0)
        for iteration in os.listdir(folder_to_use):
            train_y = np.append(train_y, np.load(os.path.join(folder_to_use, iteration, 'y_train_values.npy')))
            train_y_predicted = np.append(train_y_predicted, np.load(os.path.join(folder_to_use, iteration, 'y_train_predicted_values.npy')))
            test_y = np.append(test_y, np.load(os.path.join(folder_to_use, iteration, 'y_test_values.npy')))
            test_y_predicted = np.append(test_y_predicted, np.load(os.path.join(folder_to_use, iteration, 'y_test_predicted_values.npy')))

    else:
        train_y = np.load(os.path.join(folder_to_use,'y_train_values.npy'))
        train_y_predicted = np.load(os.path.join(folder_to_use,'y_train_predicted_values.npy'))
        test_y = np.load(os.path.join(folder_to_use,'y_test_values.npy'))
        test_y_predicted = np.load(os.path.join(folder_to_use,'y_test_predicted_values.npy'))


    mask = remove_outliers(train_y)
    train_y = train_y[mask]
    train_y_predicted = train_y_predicted[mask]

    mask = remove_outliers(train_y_predicted)
    train_y = train_y[mask]
    train_y_predicted = train_y_predicted[mask]

    mask = remove_outliers(test_y)
    test_y = test_y[mask]
    test_y_predicted = test_y_predicted[mask]

    mask = remove_outliers(test_y_predicted)
    test_y = test_y[mask]
    test_y_predicted = test_y_predicted[mask]


    fig, ax = plt.subplots(1, 1)
    ax.scatter(train_y, train_y_predicted, label='Train', color='red')
    ax.scatter(test_y, test_y_predicted, label='Test', color='blue')

    config = ' '.join(os.path.basename(folder_to_use).split('^'))
    if fig_title:
        fig.suptitle(f'{fig_title}\nConfig - {config}\n', fontsize=10)
    else:
        fig.suptitle(f'Config - {config}\n', fontsize=10)
    ax.set_xlabel('Real values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(ax_title)
    # ax.axis('equal')
    ax.legend()

    x_regressor, y_regressor = get_regressor_input(train_y, train_y_predicted, precenteage_for_regressor=20)
    x_pred, y_pred, lower, upper, prediction_lower, prediction_upper = fit_line_with_confidence(x_regressor, y_regressor, intercept=True)
    ax.plot(x_pred, y_pred, '-', color='darkred', linewidth=2)
    ax.fill_between(x_pred, lower, upper, color='lightcoral', alpha=0.6)
    ax.fill_between(x_pred, prediction_lower, prediction_upper, color='lightcoral', alpha=0.2)

    x_regressor, y_regressor = get_regressor_input(test_y, test_y_predicted, precenteage_for_regressor=20)
    x_pred, y_pred, lower, upper, prediction_lower, prediction_upper  = fit_line_with_confidence(x_regressor, y_regressor, intercept=True)
    ax.plot(x_pred, y_pred, '-', color='navy', linewidth=2)
    ax.fill_between(x_pred, lower, upper, color='cornflowerblue', alpha=0.6)
    ax.fill_between(x_pred, prediction_lower, prediction_upper, color='cornflowerblue', alpha=0.2)

    plt.show()

def get_regressor_input(x, y, precenteage_for_regressor=20):
    number_of_samples = len(x)
    p = np.random.permutation(number_of_samples)
    number_of_samples_to_use = number_of_samples // (100 // precenteage_for_regressor)
    idx_to_use=list(range(number_of_samples_to_use-1))
    x_regressor = x[p[idx_to_use]]
    y_regressor = y[p[idx_to_use]]

    return  np.append(x_regressor, [x[x.argmin()], x[x.argmax()]]), np.append(y_regressor,[y[x.argmin()], y[x.argmax()]])

def remove_outliers(input, th=5):
    th_val = np.std(input) * th
    return abs(input)<th_val

def fit_line_with_confidence(x, y, one_sided_precent=2.5, intercept=True):

    number_of_samples = len(x)


    if intercept:
        x = sm.add_constant(x)  # constant intercept term
    else:
        x = sm.add_constant(x)
        x[:, 0] = 0

    model = sm.OLS(y, x)

    fitted = model.fit()

    # x_pred = np.linspace(min(x.min(), 0), x.max(), 50)
    x_pred = np.linspace(x.min(), x.max(), 50)

    if intercept:
        x_pred2 = sm.add_constant(x_pred)
    else:
        x_pred2 = sm.add_constant(x_pred)
        x_pred2[:, 0] = 0


    sdev, prediction_lower, prediction_upper = wls_prediction_std(fitted, exog=x_pred2, alpha=0.05)

    y_pred = fitted.predict(x_pred2)

    y_hat = fitted.predict(x)



    y_err = y - y_hat

    mean_x = x.T[1].mean()

    n = len(x)

    dof = n - fitted.df_model - 1

    from scipy import stats

    t = stats.t.ppf(1 - one_sided_precent/100, df=dof)

    s_err = np.sum(np.power(y_err, 2))

    conf = t * np.sqrt((s_err / (n - 2)) * (1.0 / n + (np.power((x_pred - mean_x), 2) / ((np.sum(np.power(x_pred, 2))) - n * (np.power(mean_x, 2))))))
    upper = y_pred + abs(conf)

    lower = y_pred - abs(conf)


    return x_pred, y_pred, lower, upper, prediction_lower, prediction_upper

##### similiarity #####
##MUCO##
### best rho ###
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\gvhd_multi_grid_xgboost_without_similiarity',r'alpha=0.01^n_estimators=5^min_child_weight=20^reg_lambda=20^max_depth=5', 'Naive Algo')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\gvhd_multi_grid_xgboost_with_similiarity','alpha=100^n_estimators=20^min_child_weight=0.1^reg_lambda=10^max_depth=3^beta_for_similarity=10', 'Similiarity')

### best mse ###
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\gvhd_multi_grid_xgboost_without_similiarity',r'alpha=0.01^n_estimators=5^min_child_weight=20^reg_lambda=20^max_depth=3', 'Naive Algo')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\gvhd_multi_grid_xgboost_with_similiarity','alpha=20^n_estimators=20^min_child_weight=10^reg_lambda=20^max_depth=5^beta_for_similarity=100', 'Similiarity')

##ALLERGY##
### best rho ###
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\allergy_multi_grid_xgboost_without_similiarity',r'alpha=20^n_estimators=20^min_child_weight=10^reg_lambda=20^max_depth=5', 'Naive Algo')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\allergy_multi_grid_xgboost_with_similiarity','alpha=20^n_estimators=10^min_child_weight=0.1^reg_lambda=0^max_depth=3^beta_for_similarity=10', 'Similiarity')
#
# ## best mse ###
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\allergy_multi_grid_xgboost_without_similiarity',r'alpha=50^n_estimators=20^min_child_weight=10^reg_lambda=0^max_depth=5', 'Naive Algo')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\allergy_multi_grid_xgboost_with_similiarity','alpha=100^n_estimators=20^min_child_weight=10^reg_lambda=0^max_depth=5^beta_for_similarity=100', 'Similiarity')

##### FNN #####
##MUCO##
### best rho ###

# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\gvhd_multi_grid_tf_wo_censor_wo_similiarity',r'l2=10^dropout=0.2^factor=1^epochs=80^number_iterations=5^number_layers=1^neurons_per_layer=20', 'Naive Algo')
# visualize_prediction(r'C:\gvhd_multi_grid_tf_with_censored_without_similiarity_fixed','l2=1^dropout=0.6^factor=1^epochs=80^number_iterations=5^number_layers=1^neurons_per_layer=50', 'TS')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\gvhd_multi_grid_tf_with_similiarity','l2=1^dropout=0^factor=1^epochs=20^number_iterations=5^number_layers=1^neurons_per_layer=20^censored_mse_factor=0.5^beta_for_similarity=1', 'TS+Similiarity')

### best mse ###
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\gvhd_multi_grid_tf_wo_censor_wo_similiarity',r'l2=10^dropout=0^factor=1^epochs=80^number_iterations=5^number_layers=3^neurons_per_layer=50', 'Naive Algo')
# visualize_prediction(r'C:\gvhd_multi_grid_tf_with_censored_without_similiarity_fixed','l2=1^dropout=0.6^factor=1^epochs=80^number_iterations=5^number_layers=1^neurons_per_layer=20', 'TS')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\gvhd_multi_grid_tf_with_similiarity','l2=1^dropout=0.2^factor=0.1^epochs=20^number_iterations=5^number_layers=1^neurons_per_layer=50^censored_mse_factor=0.05^beta_for_similarity=100', 'TS+Similiarity')


##ALLERGY##
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\allergy_multi_grid_tf_wo_censor_wo_similiarity',r'l2=20^dropout=0.6^factor=1^epochs=80^number_iterations=5^number_layers=1^neurons_per_layer=50', 'Naive Algo')
# visualize_prediction(r'C:\allergy_multi_grid_tf_with_censored_without_similiarity_fixed','l2=20^dropout=0.6^factor=1^epochs=20^number_iterations=5^number_layers=1^neurons_per_layer=50', 'TS')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\allergy_multi_grid_tf_with_similiarity','l2=10^dropout=0^factor=100^epochs=80^number_iterations=5^number_layers=2^neurons_per_layer=50^censored_mse_factor=50.0^beta_for_similarity=100', 'TS+Similiarity')

### best mse ###
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\allergy_multi_grid_tf_wo_censor_wo_similiarity',r'l2=100^dropout=0.2^factor=1^epochs=20^number_iterations=5^number_layers=3^neurons_per_layer=50', 'Naive Algo')
# visualize_prediction(r'C:\allergy_multi_grid_tf_with_censored_without_similiarity_fixed','l2=1^dropout=0.2^factor=1^epochs=20^number_iterations=5^number_layers=1^neurons_per_layer=20', 'TS')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\allergy_multi_grid_tf_with_similiarity','l2=100^dropout=0.2^factor=100^epochs=20^number_iterations=5^number_layers=1^neurons_per_layer=50^censored_mse_factor=50.0^beta_for_similarity=10', 'TS+Similiarity')

##### LSTM #####
##MUCO##
### best rho ###
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\lstm_naive\gvhd_multi_grid_rnn_wo_censor_wo_similiarity',r'l2=10^dropout=0.6^factor=1^epochs=1000^number_iterations=5^number_layers=2^neurons_per_layer=50', 'Naive LSTM')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\gvhd_multi_grid_rnn_with_censored_without_similiarity_less_params',r'l2=1^dropout=0.6^factor=1^epochs=100^number_iterations=5^number_layers=2^neurons_per_layer=20', 'LSTM+TS')

### best mse ###
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\lstm_naive\gvhd_multi_grid_rnn_wo_censor_wo_similiarity',r'l2=1^dropout=0^factor=1^epochs=1000^number_iterations=5^number_layers=2^neurons_per_layer=20', 'Naive LSTM')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\gvhd_multi_grid_rnn_with_censored_without_similiarity_less_params',r'l2=1^dropout=0.2^factor=1^epochs=100^number_iterations=5^number_layers=2^neurons_per_layer=50', 'LSTM+TS')


##### LSTM #####
##MUCO##
### best rho ###
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\lstm_naive\allergy_multi_grid_rnn_wo_censor_wo_similiarity',r'l2=1^dropout=0.6^factor=1^epochs=100^number_iterations=5^number_layers=2^neurons_per_layer=50', 'Naive LSTM')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\allergy_multi_grid_rnn_with_censored_without_similiarity_less_params',r'l2=20^dropout=0.6^factor=1^epochs=1000^number_iterations=5^number_layers=2^neurons_per_layer=20', 'LSTM+TS')

### best mse ###
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\lstm_naive\gvhd_multi_grid_rnn_wo_censor_wo_similiarity',r'l2=1^dropout=0^factor=1^epochs=1000^number_iterations=5^number_layers=2^neurons_per_layer=20', 'Naive LSTM')
# visualize_prediction(r'C:\Users\Bar\Desktop\reports\reports_new\gvhd_multi_grid_rnn_with_censored_without_similiarity_less_params',r'l2=1^dropout=0.2^factor=1^epochs=100^number_iterations=5^number_layers=2^neurons_per_layer=50', 'LSTM+TS')



####### FNN WITH EARLY STOP ###

######### GVHD  ########
# visualize_prediction(r'z:\GVHD_FNN_AGAIN',r'l2=1^dropout=0.2^factor=1^epochs=1000^number_iterations=5^number_layers=1^neurons_per_layer=20', 'GVHD FNN')
# visualize_prediction(r'C:\Users\Bar\Desktop\testing\gvhd_FNN_best_config_iter_0',r'l2=1^dropout=0.2^factor=1^epochs=1000^number_iterations=10^number_layers=1^neurons_per_layer=20',  'GVHD FNN')
