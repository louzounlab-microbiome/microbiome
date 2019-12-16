from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
from infra_functions.general import apply_pca, use_spearmanr, use_pearsonr
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pickle
from sklearn.metrics import mean_squared_error

from GVHD_BAR.show_data import calc_results_and_plot

import xgboost as xgb
import datetime
from GVHD_BAR.show_data import calc_results
from GVHD_BAR.calculate_distances import calculate_distance
from GVHD_BAR.cluster_time_events import cluster_based_on_time
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RECORD = False

USE_SIMILARITY = True
USE_CLUSTER = False
USE_CLOSEST_NEIGHBOR = False
USE_CERTAINTY = False

def predict_get_spearman_value(X, y, regressor):
    predicted_y = regressor.predict(X)
    spearman_values = use_spearmanr(y, predicted_y)
    pearson_values = use_pearsonr(y, predicted_y)
    return predicted_y, spearman_values, pearson_values


def plot_fit(x, y, name):
    plt.scatter(x, y)
    plt.title('Age in days using \n' + name)
    plt.xlabel('Real')
    plt.ylabel('Predicted')


def plot_spearman_vs_params(spearman_values, label=None, plot=True):
    x_values = []
    y_values = []
    for i, spearman_value in enumerate(spearman_values):
        x_values.append(i)
        y_values.append(1 - spearman_value['spearman_rho'])
    if plot:
        plt.plot(x_values, y_values, label=label, linewidth=0.5)
        plt.title(r'$1-\rho$ vs params.json')
        plt.xlabel('sample #')
        plt.ylabel(r'$1-\rho$ value')
    return x_values, y_values


def get_datetime(date_str):
    if pd.isnull(date_str):
        date_str = '01/01/1900'
    return datetime.datetime.strptime(date_str, '%d/%m/%Y')


def get_days(days_datetime):
    return days_datetime.days


n_components = 20

OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR,'saliva_samples_231018.csv'), os.path.join(SCRIPT_DIR, 'saliva_samples_mapping_file_231018.csv'), from_QIIME=True)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=True, taxnomy_level=5)
otu_after_pca_wo_taxonomy, _ = apply_pca(preproccessed_data, n_components=n_components, visualize=False)
# otu_after_pca = OtuMf.add_taxonomy_col_to_new_otu_data(otu_after_pca_wo_taxonomy)
# merged_data_after_pca = OtuMf.merge_mf_with_new_otu_data(otu_after_pca_wo_taxonomy)
# merged_data_with_age = otu_after_pca_wo_taxonomy.join(OtuMf.mapping_file['age_in_days'])
# merged_data_with_age = merged_data_with_age[merged_data_with_age.age_in_days.notnull()] # remove NaN days
# merged_data_with_age_group = otu_after_pca_wo_taxonomy.join(OtuMf.mapping_file[['age_group', 'age_in_days','MouseNumber']])
# merged_data_with_age_group = merged_data_with_age_group[merged_data_with_age_group.age_group.notnull()] # remove NaN days

# OtuMf.mapping_file.apply(lambda x: -999 if x['Mucositis_Start'] is None else (datetime.datetime.strptime(x['DATE'], '%d/%m/%Y') - datetime.datetime.strptime(x['Mucositis_Start'], '%d/%m/%Y')).days)


OtuMf.mapping_file['DATE_datetime'] = OtuMf.mapping_file['DATE'].apply(get_datetime)
OtuMf.mapping_file['Mocosities_start_datetime'] = OtuMf.mapping_file['Mucositis_Start'].apply(get_datetime)
OtuMf.mapping_file['TIME_BEFORE_MOCO_START'] = OtuMf.mapping_file['Mocosities_start_datetime'] - OtuMf.mapping_file[
    'DATE_datetime']

OtuMf.mapping_file['time_for_the_event'] = OtuMf.mapping_file['TIME_BEFORE_MOCO_START'].apply(get_days)

OtuMf.mapping_file['time_for_the_event'][
    OtuMf.mapping_file['Mocosities_start_datetime'] == datetime.datetime.strptime('01/01/1900', '%d/%m/%Y')] = 9999

# create groups
data_grouped = OtuMf.mapping_file.groupby('Personal_ID')
censored_data = {}
dilated_df = pd.DataFrame()
for subject_id, subject_data in data_grouped:
    if 9999 in subject_data['time_for_the_event'].values:  # censored
        tmp_data = subject_data.join(otu_after_pca_wo_taxonomy)
        tmp_data_only_valid = tmp_data.loc[tmp_data[0].notnull()]
        if not tmp_data_only_valid.empty:
            # get only the last sample
            censored_data[subject_id] = tmp_data_only_valid.loc[
                tmp_data_only_valid['TIME_BEFORE_MOCO_START'] == min(tmp_data_only_valid['TIME_BEFORE_MOCO_START'])]

    else:  # not censored
        before_event_mask = subject_data['time_for_the_event'] > 0
        dilated_df = dilated_df.append(subject_data.loc[before_event_mask])
print(censored_data)
# entire_data_with_the_event[~entire_data_with_the_event['MouseNumber'].isin(censored_subject)]

dilated_df = dilated_df.join(otu_after_pca_wo_taxonomy)

# remove subects with no data in mocrobiome
dilated_df = dilated_df.loc[dilated_df[0].notnull()]

# remove too far data
before_event_mask = dilated_df['time_for_the_event'] < 30

dilated_df = dilated_df.loc[before_event_mask]



if USE_CLUSTER:
    # cluster based on time
    number_of_classes=2
    clusterd_data = cluster_based_on_time(dilated_df['time_for_the_event'], k=number_of_classes)
    dilated_df_clusterd = pd.DataFrame()

    for idx, cluster in enumerate(clusterd_data):
        tmp_df = dilated_df.loc[cluster.index.tolist()]
        tmp_df['cluster_number'] = idx
        dilated_df_clusterd = dilated_df_clusterd.append(tmp_df)


    # create knn classifier
    starting_col = np.argwhere(dilated_df.columns == 0).tolist()[0][0]

    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=7)
    X= dilated_df_clusterd.iloc[:, starting_col:starting_col + n_components]
    y=dilated_df_clusterd['cluster_number']
    neigh.fit(X, y)

# print(neigh.predict([[1.1]]))



betas_values = {}
betas = [9]  # np.logspace(-3, 3, 50)
censored_time = {}
removed_rows = {}


if USE_CLOSEST_NEIGHBOR:
    betas = [1]

for beta in betas:
    #### use all samples ####

    if ~USE_CLOSEST_NEIGHBOR:
        print('beta = ' + str(beta))


        def multiply_by_time_for_the_event(col):
            return col.apply(lambda x: x * col['time_for_the_event'])

        inputs = {subject_id: subject_data[list(range(n_components))] for subject_id, subject_data in
                                censored_data.items()}
        if USE_CLUSTER:
            # use cluster
            class_prediction = {subject_id: neigh.predict(subject_data)[0] for subject_id, subject_data in inputs.items()}

            inputs_per_class = {}
            for class_number in range(number_of_classes):
                inputs_per_class[class_number] ={}

            for (subject_id, class_predicted), subject_data in zip(class_prediction.items(), inputs.values()):
                inputs_per_class[class_predicted].update({subject_id: subject_data})

            K={}
            K_t_time={}
            K_t_time_multiplied_time_for_the_event={}
            censored_data_with_time = pd.DataFrame()
            for class_number in range(number_of_classes):
                a=dilated_df.loc[dilated_df_clusterd['cluster_number'] == class_number]
                K[class_number], _ = calculate_distance(a[list(range(20))], inputs_per_class[class_number], beta, visualize=False)
                K_t_time[class_number] = K[class_number].transpose().join(dilated_df['time_for_the_event'])
                K_t_time_multiplied_time_for_the_event[class_number] = K_t_time[class_number].apply(multiply_by_time_for_the_event, axis=1)
                denominator = K_t_time[class_number].sum()
                nominator = K_t_time_multiplied_time_for_the_event[class_number].sum()
                tmp = nominator / denominator
                censored_data_with_time_per_class = OtuMf.mapping_file.loc[tmp.index[:-1].tolist()]
                censored_data_with_time_per_class['time_for_the_event'] = tmp
                censored_data_with_time = censored_data_with_time.append(censored_data_with_time_per_class)
        else:
            if USE_CERTAINTY:
                certinty_factor = 0.5
                factor_coeff = True
                updated_df = dilated_df.copy()
                censored_data_with_time = pd.DataFrame()
                certainty_matrix = pd.DataFrame(np.ones((len(dilated_df.index),1)), index=dilated_df.index)
                while True:
                    K, dist_matrix = calculate_distance(updated_df[list(range(n_components))], inputs, beta, visualize=False)

                    ### choose the best censored subject and update K to contain only it###
                    certainties = np.dot(1/dist_matrix.values, certainty_matrix.values)
                    best_candidate_idx = np.argmax(certainties)
                    best_candidate = dist_matrix.index[best_candidate_idx]
                    certainty_coefficent = certinty_factor * sigmoid(certainties[best_candidate_idx])
                    K = K.loc[best_candidate].to_frame()
                    K_t_time = K.join(updated_df['time_for_the_event'])

                    if factor_coeff:
                        K_t_time[best_candidate] = K_t_time[best_candidate] * certainty_matrix[0]
                    K_t_time_multiplied_time_for_the_event = K_t_time.apply(multiply_by_time_for_the_event, axis=1)
                    denominator = K_t_time.sum()
                    nominator = K_t_time_multiplied_time_for_the_event.sum()
                    tmp = nominator / denominator
                    censored_data_with_time_per_subject = OtuMf.mapping_file.loc[tmp.index[:-1].tolist()]
                    censored_data_with_time_per_subject['time_for_the_event'] = tmp
                    censored_data_with_time = censored_data_with_time.append(censored_data_with_time_per_subject)

                    # update the inputs for next iter
                    certainty_matrix = certainty_matrix.append(pd.DataFrame(certainty_coefficent.tolist(), index=[best_candidate]))
                    updated_df = updated_df.append(censored_data_with_time_per_subject.join(otu_after_pca_wo_taxonomy))
                    inputs.pop(list(inputs.keys())[best_candidate_idx])
                    if len(inputs) == 0: # means we finished iterating over the censored
                        break

            else:
                K, _ = calculate_distance(dilated_df[list(range(n_components))], inputs, beta, visualize=False)
                K_t_time = K.transpose().join(dilated_df['time_for_the_event'])

                K_t_time_multiplied_time_for_the_event = K_t_time.apply(multiply_by_time_for_the_event, axis=1)

                denominator = K_t_time.sum()
                nominator = K_t_time_multiplied_time_for_the_event.sum()
                censored_time[beta] = nominator / denominator
                censored_data_with_time = OtuMf.mapping_file.loc[censored_time[beta].index[:-1].tolist()]
                censored_data_with_time['time_for_the_event'] = censored_time[beta]
    else:
        #### use closest neighbour ####
        K, _ = calculate_distance(dilated_df[list(range(n_components))],
                               {subject_id: subject_data[list(range(20))] for subject_id, subject_data in
                                censored_data.items()}, 1, visualize=False)


        def get_closest_neighbour(row):
            a = row.sort_values()
            return a.index[0]


        K_closest_neighbour = K.apply(get_closest_neighbour, axis=1).to_frame()
        censored_data_with_time = OtuMf.mapping_file.loc[K_closest_neighbour.index.tolist()]
        censored_data_with_time['time_for_the_event'] = dilated_df['time_for_the_event'][K_closest_neighbour[0]].values

    # for both closest and distance based

    censored_data_with_time = censored_data_with_time.join(otu_after_pca_wo_taxonomy)
    number_of_rows_before_removal = censored_data_with_time.shape[0]
    # remove subects with no data in mocrobiome
    censored_data_with_time = censored_data_with_time.loc[censored_data_with_time[0].notnull()]

    # remove subjects that are unable to calculate the syntethic time for the event
    censored_data_with_time = censored_data_with_time.loc[censored_data_with_time['time_for_the_event'].notnull()]
    number_of_rows_after_removal = censored_data_with_time.shape[0]
    removed_rows[beta] = number_of_rows_before_removal - number_of_rows_after_removal
    # create train set and test set
    stats = {'svm': {'test': {'wrong': [], 'size': [], 'score': [], 'roc_auc': []},
                     'train': {'wrong': [], 'size': [], 'score': [], 'roc_auc': []}}
        , 'lda': {'test': {'wrong': [], 'size': [], 'score': [], 'roc_auc': []},
                  'train': {'wrong': [], 'size': [], 'score': [], 'roc_auc': []}}}
    # coefficent_df = {'svm': {'coef': [], 'class': []}, 'lda': {'coef': [], 'class': []}}

    y_train_spearman_values = []
    y_test_spearman_values = []

    y_train_values = []
    y_train_predicted_values = []
    y_test_values = []
    y_test_predicted_values = []

    y_train_dict = {}
    y_test_dict = {}

    cross_val_number = 4
    plt.figure(1000)
    total_train_mse = 0
    total_test_mse = 0

    test_iter = []
    train_iter = []

    starting_col = np.argwhere(dilated_df.columns == 0).tolist()[0][0]
    X = dilated_df.iloc[:, starting_col:starting_col + n_components]
    y = dilated_df.loc[:, 'time_for_the_event']

    if USE_SIMILARITY:
        X_train_censored = censored_data_with_time.iloc[:, starting_col:starting_col + n_components]
        y_train_censored = censored_data_with_time.loc[:, 'time_for_the_event']

    for k in range(1):
        # for i in range(cross_val_number): # crossvalidation
        for i in range(len(X)):
            print('\nIteration number: ', str(i + 1))
            spearman_train_values = []
            spearman_test_values = []

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # used in cross validation
            X_test = X.iloc[i].to_frame().transpose()
            y_test = pd.Series(y.iloc[i], [y.index[i]])
            X_train = X.drop(X.index[i])
            y_train = y.drop(y.index[i])
            if USE_SIMILARITY:
                X_train = X_train.append(X_train_censored)
                y_train = y_train.append(y_train_censored)

            # shuffle
            idx = np.random.permutation(X_train.index)
            X_train = X_train.reindex(idx)
            y_train = y_train.reindex(idx)

            ### xgboost #####
            algo_name = 'xgboost'
            # _alpha = [0.01, 20, 50, 100]
            # _n_estimators = [5, 10, 20]
            # _reg_lambda = [0, 10, 20]
            # _max_depth = [3, 5, 10]
            # _min_child_weight = [0.1, 1, 10, 20]

            _alpha = [0.01]
            _n_estimators = [20]
            _reg_lambda = [20]
            _max_depth = [5]
            _min_child_weight = [20]

            count = 0
            sample_num = 0
            for alpha in _alpha:
                for n_estimators in _n_estimators:
                    for min_child_weight in _min_child_weight:
                        for idx, reg_lambda in enumerate(_reg_lambda):
                            count += 1
                            # plt.figure(count)
                            for idx2, max_depth in enumerate(_max_depth):
                                print(
                                    f'***** sample num = {sample_num} *****\n params.json: alpha={alpha}, n_estimators={n_estimators}, reg_lambda={reg_lambda}, max_depth={max_depth}, min_child_weight={min_child_weight}')

                                xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=1, learning_rate=0.1,
                                                          reg_lambda=reg_lambda,
                                                          max_depth=max_depth, alpha=alpha, n_estimators=n_estimators,
                                                          min_child_weight=min_child_weight)
                                xg_reg.fit(X_train, y_train)

                                predicted_y, spearman_value, pearson_value = predict_get_spearman_value(X_train,
                                                                                                        y_train,
                                                                                                        xg_reg)
                                y_train_predicted_values += predicted_y.tolist()
                                y_train_values += y_train.values.tolist()
                                mse = mean_squared_error(y_train.values, predicted_y)
                                spearman_train_values.append(
                                    {'mse': mse, 'spearman_rho': spearman_value['rho'],
                                     'pearson_rho': pearson_value['rho']})
                                print(f'train results: {spearman_value, pearson_value,  mse}')

                                key = '|'.join([x + '=' + str(eval(x)) for x in
                                                ['alpha', 'n_estimators', 'min_child_weight', 'reg_lambda',
                                                 'max_depth']])
                                if key not in y_train_dict.keys():
                                    y_train_dict[key] = {}
                                if 'y_train_values' not in y_train_dict[key].keys():
                                    y_train_dict[key]['y_train_values'] = y_train.values.tolist()
                                    y_train_dict[key]['y_train_predicted_values'] = predicted_y.tolist()
                                else:
                                    y_train_dict[key]['y_train_values'] += y_train.values.tolist()
                                    y_train_dict[key]['y_train_predicted_values'] += predicted_y.tolist()

                                # plt.subplot(3,2,(idx2+1)*2-1)
                                # plot_fit(y_train.values, predicted_y,
                                #          'xgboost - Train \n alpha={}, n_estimator={}, reg_lambda={}, max_depth={} \n spearman rho={}, MSE={}'
                                #          .format(alpha, n_estimators, reg_lambda, max_depth, spearman_value['rho'], mse))

                                predicted_y, spearman_value, pearson_value = predict_get_spearman_value(X_test, y_test,
                                                                                                        xg_reg)
                                mse = mean_squared_error(y_test.values, predicted_y)
                                y_test_values += y_test.values.tolist()

                                y_test_predicted_values += predicted_y.tolist()
                                spearman_test_values.append(
                                    {'mse': mse, 'spearman_rho': spearman_value['rho'],
                                     'pearson_rho': pearson_value['rho']})

                                print(f'test results: {spearman_value, mse, pearson_value}\n')
                                sample_num += 1

                                if key not in y_test_dict.keys():
                                    y_test_dict[key] = {}
                                if 'y_test_values' not in y_test_dict[key].keys():
                                    y_test_dict[key]['y_test_values'] = y_test.values.tolist()
                                    y_test_dict[key]['y_test_predicted_values'] = predicted_y.tolist()
                                else:
                                    y_test_dict[key]['y_test_values'] += y_test.values.tolist()
                                    y_test_dict[key]['y_test_predicted_values'] += predicted_y.tolist()

                                # if cross_val_number == 1:
                                #     plt.subplot(3,2,(idx2+1)*2)
                                #     plot_fit(y_test.values, predicted_y,
                                #              'xgboost - Train \n alpha={}, n_estimator={}, reg_lambda={}, max_depth={} \n spearman rho={}, MSE={}'
                                #              .format(alpha, n_estimators, reg_lambda, max_depth, spearman_value['rho'], mse))

            #### random forest #####
            # algo_name = 'Random_Forest'
            # n_estimators_list = [91]
            # max_features_list = [88]
            # min_samples_leaf_list = [5]
            # # n_estimators_list = range(1, 50, 15)
            # # max_features_list = range(n_components, 1, -2)
            # # min_samples_leaf_list = range(20, 1, -3)
            # best_params = {'mse': {'params.json': {}, 'mse': 999999, 'spearman_rho': -2},
            #                'spearman_rho': {'params.json': {}, 'mse': 999999, 'spearman_rho': -2}}
            # spearman_train_values = []
            # spearman_test_values = []
            # count = 0
            # DRAW_FIT = False
            # option_number = 0
            # for i, n_estimators in enumerate(n_estimators_list):
            #     for j, max_features in enumerate(max_features_list):
            #         if DRAW_FIT:
            #             count += 1
            #             print(count)
            #             # plt.figure(count)
            #         for t, min_samples_leaf in enumerate(min_samples_leaf_list):
            #             option_number += 1
            #             current_params = {'n_estimators': n_estimators, 'max_features': max_features,
            #                               'min_samples_leaf': min_samples_leaf}
            #             regressor = fit_random_forest(X_train, y_train, **current_params)
            #             predicted_y, spearman_value = predict_get_spearman_value(X_train, y_train, regressor)
            #
            #             mse = mean_squared_error(y_train.values, predicted_y)
            #             y_train_values += y_train.values.tolist()
            #             y_train_predicted_values += predicted_y.tolist()
            #             if DRAW_FIT:
            #                 plt.subplot(len(min_samples_leaf_list), 2, 2 * t + 1)
            #                 plot_fit(y_train.values, predicted_y,
            #                          'Random Forest - Train \n n_estimator={}, max_features={}, min_samples_leaf={} \n spearman rho={}, MSE={}'
            #                          .format(current_params['n_estimators'], current_params['max_features'],
            #                                  current_params['min_samples_leaf'], spearman_value['rho'], mse))
            #             if spearman_value['rho'] > best_params['spearman_rho']['spearman_rho']:
            #                 best_params['spearman_rho']['spearman_rho'] = spearman_value['rho']
            #                 best_params['spearman_rho']['params.json'] = current_params
            #                 best_params['spearman_rho']['mse'] = mse
            #
            #             key = '_'.join([x + '_' + str(eval(x)) for x in
            #                             ['n_estimators', 'max_features', 'min_samples_leaf']])
            #             if key not in y_train_dict.keys():
            #                 y_train_dict[key] = {}
            #             if 'y_train_values' not in y_train_dict[key].keys():
            #                 y_train_dict[key]['y_train_values'] = y_train.values.tolist()
            #                 y_train_dict[key]['y_train_predicted_values'] = predicted_y.tolist()
            #             else:
            #                 y_train_dict[key]['y_train_values'] += y_train.values.tolist()
            #                 y_train_dict[key]['y_train_predicted_values'] += predicted_y.tolist()
            #
            #             spearman_train_values.append(
            #                 {'params.json': current_params, 'mse': mse, 'spearman_rho': spearman_value['rho']})
            #             predicted_y, spearman_value = predict_get_spearman_value(X_test, y_test, regressor)
            #             mse = mean_squared_error(y_test.values, predicted_y)
            #             y_test_values += y_test.values.tolist()
            #             y_test_predicted_values += predicted_y.tolist()
            #             spearman_test_values.append(
            #                 {'params.json': current_params, 'mse': mse, 'spearman_rho': spearman_value['rho']})
            #
            #             if key not in y_test_dict.keys():
            #                 y_test_dict[key] = {}
            #             if 'y_test_values' not in y_test_dict[key].keys():
            #                 y_test_dict[key]['y_test_values'] = y_test.values.tolist()
            #                 y_test_dict[key]['y_test_predicted_values'] = predicted_y.tolist()
            #             else:
            #                 y_test_dict[key]['y_test_values'] += y_test.values.tolist()
            #                 y_test_dict[key]['y_test_predicted_values'] += predicted_y.tolist()
            #
            #
            #
            #             if DRAW_FIT:
            #                 plt.subplot(len(min_samples_leaf_list), 2, 2 * t + 1 + 1)
            #                 plot_fit(y_test.values, predicted_y,
            #                          'Random Forest - Test \n n_estimator={}, max_features={}, min_samples_leaf={} \n spearman rho={}, MSE={}'
            #                          .format(current_params['n_estimators'], current_params['max_features'],
            #                                  current_params['min_samples_leaf'], spearman_value['rho'], mse))
            #             plt.subplots_adjust(hspace=0.5, wspace=0.5)
            #
            #             print(option_number, current_params)
            #
            #         plt.subplots_adjust(hspace=1.2, wspace=0.5)
            #
            # print(best_params)
            ### end random forest

            # plt.figure(i)
        #     total_train_mse += spearman_train_values[0]['mse']
        #     total_test_mse += spearman_test_values[0]['mse']
        #     x_train_values, current_y_train_values = plot_spearman_vs_params(spearman_train_values,
        #                                                                      label='Train: iteration #' + str(i), plot=False)
        #     if y_train_spearman_values:
        #         y_train_spearman_values = [a + b for a, b in zip(y_train_spearman_values, current_y_train_values)]
        #     else:
        #         y_train_spearman_values = current_y_train_values
        #
        #     x_test_values, current_y_test_values = plot_spearman_vs_params(spearman_test_values,
        #                                                                    label='Test iteration #' + str(i), plot=False)
        #     if y_test_spearman_values:
        #         y_test_spearman_values = [a + b for a, b in zip(y_test_spearman_values, current_y_test_values)]
        #     else:
        #         y_test_spearman_values = current_y_test_values
        #
        #     # plt.show()
        #     # xgb.plot_importance(xg_reg)
        #     # plt.show()
        #
        # plt.figure(100)
        # y_train_spearman_values = [a / cross_val_number for a in y_train_spearman_values]
        # y_test_spearman_values = [a / cross_val_number for a in y_test_spearman_values]
        # #
        # mean_train_mse = total_train_mse / cross_val_number
        # mean_test_mse = total_test_mse / cross_val_number
        # if cross_val_number > 1:
        #     plt.plot(x_train_values, y_train_spearman_values, label='Train', linewidth=0.5)
        #     plt.plot(x_test_values, y_test_spearman_values, label='Test', linewidth=0.5)
        #     plt.title(r'$1-\rho$ vs params.json')
        #     plt.xlabel('sample #')
        #     plt.ylabel(r'$1-\rho$ value')

    if RECORD:
        pickle.dump(y_train_dict, open(algo_name + "_train_data.p", "wb"))
        pickle.dump(y_test_dict, open(algo_name + "_test_data.p", "wb"))
    betas_values[beta] = calc_results(y_train_dict, y_test_dict, algo_name, visualize=False)

y_train_predicted_values = y_train_dict['alpha=0.01|n_estimators=20|min_child_weight=20|reg_lambda=20|max_depth=5']['y_train_predicted_values']
y_train_values = y_train_dict['alpha=0.01|n_estimators=20|min_child_weight=20|reg_lambda=20|max_depth=5']['y_train_values']
y_test_values = y_test_dict['alpha=0.01|n_estimators=20|min_child_weight=20|reg_lambda=20|max_depth=5']['y_test_values']
y_test_predicted_values = y_test_dict['alpha=0.01|n_estimators=20|min_child_weight=20|reg_lambda=20|max_depth=5']['y_test_predicted_values']
calc_results_and_plot(y_train_values, y_train_predicted_values, y_test_values, y_test_predicted_values,'NewAlg\n alpha=0.01|n_estimators=20|min_child_weight=20|reg_lambda=20|max_depth=5', visualize=True, title='NewAlg')
print('**************\n')
print(betas_values)
print(removed_rows)


# betas_as_x = []
# y_train_rho = []
# y_test_rho = []
# for beta, value in betas_values.items():
#     betas_as_x.append(beta)
#     train = value[0]
#     test = value[1]
#     y_train_rho.append(
#         train['alpha=0.01|n_estimators=20|min_child_weight=20|reg_lambda=20|max_depth=5']['spearman']['rho'])
#     y_test_rho.append(
#         test['alpha=0.01|n_estimators=20|min_child_weight=20|reg_lambda=20|max_depth=5']['spearman']['rho'])
#
# best_beta_idx = y_test_rho.index(max(y_test_rho))
# plt.figure(756)
# plt.scatter(np.log10(betas_as_x), y_test_rho, label='Test', linewidth=0.3)
# plt.scatter(np.log10(betas_as_x), y_train_rho, label='Train', linewidth=0.3)
# plt.legend()
# plt.title(
#     'Spearman' + f'$\\rho$ vs $\\beta$ \n best: $\\rho$ = {max(y_test_rho)} , $\\beta$ = {betas_as_x[best_beta_idx]}')
# plt.xlabel(r'log($\beta$)')
# plt.ylabel(r'$\rho$ value')
# plt.show()

#
# for i in range(3):
#     print('\nIteration number: ', str(i + 1))
#     physoligcal_data_with_age_group = physoligcal_data_with_age_group.sample(frac=1)
#     train_size = math.ceil(physoligcal_data_with_age_group.shape[0] * 0.8)
#     train_set = physoligcal_data_with_age_group.iloc[0:train_size]
#     test_set = physoligcal_data_with_age_group.iloc[train_size+1:]
#
#     train_x_data = train_set.loc[:, train_set.columns != 'age_group']
#     train_y_values = train_set['age_group']
#     train_y_values = train_y_values.str.replace('old', '0')
#     train_y_values = train_y_values.str.replace('young', '1')
#     train_y_values = pd.to_numeric(train_y_values)
#
#     test_x_data = test_set.loc[:, test_set.columns != 'age_group']
#     test_y_values = test_set['age_group']
#     test_y_values = test_y_values.str.replace('old', '0')
#     test_y_values = test_y_values.str.replace('young', '1')
#     test_y_values = pd.to_numeric(test_y_values)

# clf = svm.SVC(kernel='linear')
# print('SVM fit')
# clf.fit(train_x_data, train_y_values)
# print('SVM prediction')
#
# # test accuracy on the test set
# train_set_df = pd.DataFrame(train_y_values)
# train_set_df['predicted'] = clf.predict(train_x_data.values)
# train_set_df['wrong'] = train_set_df['predicted'] != train_set_df['age_group']
# stats['svm']['train']['wrong'].append(train_set_df['wrong'].sum())
# stats['svm']['train']['size'].append(train_set_df['wrong'].size)
# stats['svm']['train']['score'].append(clf.score(train_x_data, train_y_values))
# # coefficent_df['svm']['coef'].append(clf.coef_)
# # coefficent_df['svm']['class'].append(clf.classes_)
#
# false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y_values.values,
#                                                                 clf.decision_function(train_x_data))
# roc_auc = auc(false_positive_rate, true_positive_rate)
# stats['svm']['train']['roc_auc'].append(roc_auc)
# plt.subplot(2,1,1)
# plt.plot(false_positive_rate, true_positive_rate, 'b',label='Train AUC = %0.2f' % roc_auc)
# plt.title('Train set')
# print('SVM-Train - Total wrong predictions : {}, out of: {}, accuracy: {}, auc: {}'.format(train_set_df['wrong'].sum(),
#                                                                                            train_set_df['wrong'].size,
#                                                                                            clf.score(train_x_data, train_y_values),
#                                                                                            roc_auc))
# # test accuracy on the test set
# test_set_df = pd.DataFrame(test_y_values)
# test_set_df['predicted'] = clf.predict(test_x_data.values)
# test_set_df['wrong'] = test_set_df['predicted'] != test_set_df['age_group']
# stats['svm']['test']['wrong'].append(test_set_df['wrong'].sum())
# stats['svm']['test']['size'].append(test_set_df['wrong'].size)
# stats['svm']['test']['score'].append(clf.score(test_x_data, test_y_values))
# # coefficent_df['svm']['coef'].append(clf.coef_)
# # coefficent_df['svm']['class'].append(clf.classes_)
# false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y_values.values,
#                                                                 clf.decision_function(test_x_data))
# roc_auc = auc(false_positive_rate, true_positive_rate)
# stats['svm']['test']['roc_auc'].append(roc_auc)
# plt.subplot(2, 1, 2)
# plt.plot(false_positive_rate, true_positive_rate, 'b', label='Test AUC = %0.2f' % roc_auc)
# plt.title('Test set')
# print('SVM-Test - Total wrong predictions : {}, out of: {}, accuracy: {}, auc: {}'.format(test_set_df['wrong'].sum(),
#                                                                                           test_set_df['wrong'].size,
#                                                                                           clf.score(test_x_data,
#                                                                                                     test_y_values),
#                                                                                           roc_auc))
#     clf = LinearDiscriminantAnalysis()
#     print('LDA fit')
#     clf.fit(train_x_data, train_y_values)
#     print('LDA prediction')
#
#     # test accuracy on the test set
#     train_set_df = pd.DataFrame(train_y_values)
#     train_set_df['predicted'] = clf.predict(train_x_data.values)
#     train_set_df['wrong'] = train_set_df['predicted'] != train_set_df['age_group']
#     stats['lda']['train']['wrong'].append(train_set_df['wrong'].sum())
#     stats['lda']['train']['size'].append(train_set_df['wrong'].size)
#     stats['lda']['train']['score'].append(clf.score(train_x_data, train_y_values))
#     # coefficent_df['svm']['coef'].append(clf.coef_)
#     # coefficent_df['svm']['class'].append(clf.classes_)
#
#     false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y_values.values,
#                                                                     clf.decision_function(train_x_data))
#     roc_auc = auc(false_positive_rate, true_positive_rate)
#     stats['lda']['train']['roc_auc'].append(roc_auc)
#     plt.subplot(2, 1, 1)
#     plt.plot(false_positive_rate, true_positive_rate, 'b', label='Train AUC = %0.2f' % roc_auc)
#     plt.title('Train set')
#     print('LDA-Train - Total wrong predictions : {}, out of: {}, accuracy: {}, auc: {}'.format(
#         train_set_df['wrong'].sum(),
#         train_set_df['wrong'].size,
#         clf.score(train_x_data, train_y_values),
#         roc_auc))
#     # test accuracy on the test set
#     test_set_df = pd.DataFrame(test_y_values)
#     test_set_df['predicted'] = clf.predict(test_x_data.values)
#     test_set_df['wrong'] = test_set_df['predicted'] != test_set_df['age_group']
#     stats['lda']['test']['wrong'].append(test_set_df['wrong'].sum())
#     stats['lda']['test']['size'].append(test_set_df['wrong'].size)
#     stats['lda']['test']['score'].append(clf.score(test_x_data, test_y_values))
#     # coefficent_df['svm']['coef'].append(clf.coef_)
#     # coefficent_df['svm']['class'].append(clf.classes_)
#     false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y_values.values,
#                                                                     clf.decision_function(test_x_data))
#     roc_auc = auc(false_positive_rate, true_positive_rate)
#     stats['lda']['test']['roc_auc'].append(roc_auc)
#     plt.subplot(2, 1, 2)
#     plt.plot(false_positive_rate, true_positive_rate, 'b', label='Test AUC = %0.2f' % roc_auc)
#     plt.title('Test set')
#     print(
#         'LDA-Test - Total wrong predictions : {}, out of: {}, accuracy: {}, auc: {}'.format(test_set_df['wrong'].sum(),
#                                                                                             test_set_df['wrong'].size,
#                                                                                             clf.score(test_x_data,
#                                                                                                       test_y_values),
#                                                                                             roc_auc))
#
#     if i == 0:
#         plt.show()
#
# # print('\nSVM - Train - Total wrong predictions : {}, out of: {}, accuracy: {}, AUC: {}'.format(np.sum(stats['svm']['train']['wrong']), np.sum(stats['svm']['train']['size']),
# #                                                                             np.mean(stats['svm']['train']['score']), np.mean(stats['svm']['train']['roc_auc'])))
# #
# # print('SVM - Test - Total wrong predictions : {}, out of: {}, accuracy: {}, AUC: {}'.format(np.sum(stats['svm']['test']['wrong']), np.sum(stats['svm']['test']['size']),
# #                                                                             np.mean(stats['svm']['test']['score']), np.mean(stats['svm']['test']['roc_auc'])))
#
# print('\nLDA - Train - Total wrong predictions : {}, out of: {}, accuracy: {}, AUC: {}'.format(np.sum(stats['lda']['train']['wrong']), np.sum(stats['lda']['train']['size']),
#                                                                             np.mean(stats['lda']['train']['score']), np.mean(stats['lda']['train']['roc_auc'])))
#
# print('LDA - Test - Total wrong predictions : {}, out of: {}, accuracy: {}, AUC: {}'.format(np.sum(stats['lda']['test']['wrong']), np.sum(stats['lda']['test']['size']),
#                                                                             np.mean(stats['lda']['test']['score']), np.mean(stats['lda']['test']['roc_auc'])))
#
# # print(coefficent_df)
