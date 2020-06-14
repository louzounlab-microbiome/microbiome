from Projects.GVHD_BAR.load_merge_otu_mf import OtuMfHandler
from Preprocess.preprocess import preprocess_data
from Preprocess.general import apply_pca, use_spearmanr
from Preprocess.fit import fit_SVR, fit_random_forest
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

DRAW_FIT = False

def plot_fit(x, y, name):
    plt.scatter(x, y)
    plt.title('Age in days using \n' + name)
    plt.xlabel('Real')
    plt.ylabel('Predicted')

def plot_spearman_vs_params(spearman_values, label=None):
    x_values = []
    y_values = []
    for i, spearman_value in enumerate(spearman_values):
        x_values.append(i)
        y_values.append(1-spearman_value['spearman_rho'])
    plt.plot(x_values, y_values, label=label, linewidth=0.5)
    plt.title(r'$1-\rho$ vs params.json')
    plt.xlabel('sample #')
    plt.ylabel(r'$1-\rho$ value')

def predict_get_spearman_value(test_set, regressor):
    test_df = pd.DataFrame(test_set['age_in_days'])
    test_df['predicted'] = regressor.predict(test_set.loc[:, test_set.columns != 'age_in_days'])
    spearman_values = use_spearmanr(test_set['age_in_days'].values, test_df['predicted'].values)
    return test_df, spearman_values

if __name__ == "__main__":
    OtuMf = OtuMfHandler('aging_otu_table.csv', 'mf.csv', from_QIIME=True)
    preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=True, taxnomy_level=5)
    otu_after_pca_wo_taxonomy, _ = apply_pca(preproccessed_data, n_components=80)
    # otu_after_pca = OtuMf.add_taxonomy_col_to_new_otu_data(otu_after_pca_wo_taxonomy)
    # merged_data_after_pca = OtuMf.merge_mf_with_new_otu_data(otu_after_pca_wo_taxonomy)
    merged_data_with_age = otu_after_pca_wo_taxonomy.join(OtuMf.mapping_file['age_in_days'])
    merged_data_with_age = merged_data_with_age[merged_data_with_age.age_in_days.notnull()] # remove NaN days

    # create train set and test set
    merged_data_with_age = merged_data_with_age.sample(frac=1)
    train_size = math.ceil(merged_data_with_age.shape[0] * 0.85)
    train_set = merged_data_with_age.iloc[0:train_size]
    test_set = merged_data_with_age.iloc[train_size+1:]

    train_x_data = train_set.loc[:, train_set.columns != 'age_in_days']
    train_y_values = train_set['age_in_days']
    test_x_data = test_set.loc[:, test_set.columns != 'age_in_days']
    test_y_values = test_set['age_in_days']
    # SVR
    # c_values = [1, 10, 100, 1e3, 1e4, 1e5]
    # gamma_values = ['auto',0.1 , 0.5, 1, 10, 100]
    # for i, c in enumerate(c_values):
    #     plt.figure(2*i-1)
    #     for j, gamma in enumerate(gamma_values):
    #         regressor = fit_SVR(train_set.loc[:, train_set.columns != 'age_in_days'], train_set['age_in_days'], C=c, gamma=gamma)
    #         test_df, spearman_values= predict_get_spearman_value(test_set, regressor)
    #         plt.subplot(math.ceil(len(gamma_values)/2), 2, j+1)
    #         plot_fit(test_df['age_in_days'], test_df['predicted'],'SVR C={} gamma={} - spearman rho={}'.format(c, gamma, spearman_values['rho']))
    #     plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # random forest
    n_estimators_list = range(1, 1000 , 50)
    max_features_list = range(80, 1 , -5)
    min_samples_leaf_list = range(30, 1, -5)
    best_params = {'mse': {'params.json': {}, 'mse': 999999, 'spearman_rho': -2}, 'spearman_rho': {'params.json': {}, 'mse': 999999, 'spearman_rho': -2}}
    spearman_train_values = []
    spearman_test_values = []
    count = 0
    for i, n_estimators in enumerate(n_estimators_list):
        for j, max_features in enumerate(max_features_list):
            if DRAW_FIT:
                print(count)
                plt.figure(count)
                count += 1
            for t, min_samples_leaf in enumerate(min_samples_leaf_list):
                current_params = {'n_estimators': n_estimators, 'max_features': max_features, 'min_samples_leaf': min_samples_leaf}
                regressor = fit_random_forest(train_x_data, train_y_values, **current_params)
                train_predicted_df, spearman_value = predict_get_spearman_value(train_set, regressor)

                mse = mean_squared_error(train_predicted_df['age_in_days'], train_predicted_df['predicted'])
                if DRAW_FIT:
                    plt.subplot(len(min_samples_leaf_list), 2, 2*t+1)
                    plot_fit(train_predicted_df['age_in_days'], train_predicted_df['predicted'],
                             'Random Forest - Train \n n_estimator={}, max_features={}, min_samples_leaf={} \n spearman rho={}, MSE={}'
                             .format(current_params['n_estimators'], current_params['max_features'], current_params['min_samples_leaf'], spearman_value['rho'], mse))
                if spearman_value['rho'] > best_params['spearman_rho']['spearman_rho']:
                    best_params['spearman_rho']['spearman_rho'] = spearman_value['rho']
                    best_params['spearman_rho']['params.json'] = current_params
                    best_params['spearman_rho']['mse'] = mse

                spearman_train_values.append({'params.json': current_params, 'mse': mse, 'spearman_rho': spearman_value['rho']})
                test_predicted_df, spearman_value = predict_get_spearman_value(test_set, regressor)
                mse = mean_squared_error(test_predicted_df['age_in_days'], test_predicted_df['predicted'])
                spearman_test_values.append({'params.json': current_params, 'mse': mse, 'spearman_rho': spearman_value['rho']})
                if DRAW_FIT:
                    plt.subplot(len(min_samples_leaf_list), 2, 2*t+1+1)
                    plot_fit(train_predicted_df['age_in_days'], train_predicted_df['predicted'],
                             'Random Forest - Test \n n_estimator={}, max_features={}, min_samples_leaf={} \n spearman rho={}, MSE={}'
                             .format(current_params['n_estimators'], current_params['max_features'], current_params['min_samples_leaf'], spearman_value['rho'], mse))
                plt.subplots_adjust(hspace=0.5, wspace=0.5)
            plt.subplots_adjust(hspace=1.2, wspace=0.5)

    print(best_params)

    plt.figure(count)
    plot_spearman_vs_params(spearman_train_values, label='Train')
    plot_spearman_vs_params(spearman_test_values, label='Test')
    plt.legend()
    plt.show()




