from Projects.GVHD_BAR.load_merge_otu_mf import OtuMfHandler
from Preprocess.preprocess import preprocess_data
import tensorflow as tf

tf.enable_eager_execution()
from tensorflow.contrib import autograph
from tensorflow.python.keras import optimizers, regularizers, callbacks
from Preprocess import tf_analaysis
from tensorflow.python.keras.losses import mean_squared_error
# from Preprocess.generate_N_colors import getDistinctColors, rgb2hex
from Preprocess.general import apply_pca, use_spearmanr, use_pearsonr  # sigmoid
from Preprocess.fit import fit_SVR, fit_random_forest
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from scipy.stats import pearsonr
import numpy as np
import pickle
from sklearn import svm
# from sklearn.svm import SV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import datetime
from GVHD_BAR.show_data import calc_results_and_plot
from GVHD_BAR.calculate_distances import calculate_distance
from GVHD_BAR.cluster_time_events import cluster_based_on_time
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RECORD = True
PLOT = False
USE_SIMILARITY = False
USE_CLUSTER = False
USE_CLOSEST_NEIGHBOR = False
USE_CERTAINTY = False

callbacks_ = callbacks.EarlyStopping(monitor='my_mse_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
callbacks_=None

# @autograph.convert()
def my_loss(y_true, y_pred):
    mse_loss = my_mse_loss(y_true, y_pred)

    time_sense_loss = y_true[:, 2] - y_pred[:, 1]
    tsls = tf.square(time_sense_loss)

    return y_true[:, 4]*tsls + y_true[:, 3] * mse_loss

def my_mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1]))

    return mse_loss



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

OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, 'saliva_samples_231018.csv'),
                     os.path.join(SCRIPT_DIR, 'saliva_samples_mapping_file_231018.csv'), from_QIIME=True)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=5)
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
y_for_deep = pd.DataFrame()
x_for_deep = pd.DataFrame()
x_for_deep_censored = pd.DataFrame()
y_for_deep_censored = pd.DataFrame()


def calculate_y_for_deep_per_row(row):
    a = row.sort_values()
    return a.index[0]


for subject_id, subject_data in data_grouped:
    if 9999 in subject_data['time_for_the_event'].values:  # censored
        tmp_data = subject_data.join(otu_after_pca_wo_taxonomy)
        tmp_data_only_valid = tmp_data.loc[tmp_data[0].notnull()]
        if not tmp_data_only_valid.empty:
            x_for_deep_censored = x_for_deep_censored.append(subject_data)

            tmp_data_only_valid['time_before_moco_start_days'] = tmp_data_only_valid[
                'TIME_BEFORE_MOCO_START'].apply(get_days)
            tmp_data_only_valid.sort_index(by='time_before_moco_start_days', ascending=False, inplace=True)
            tmp_data_only_valid['relative_start_date'] = tmp_data_only_valid['time_before_moco_start_days'].iloc[
                                                             0] - tmp_data_only_valid[
                                                             'time_before_moco_start_days']
            tmp_data_only_valid['relative_max_date'] = tmp_data_only_valid['relative_start_date'][-1] - \
                                                       tmp_data_only_valid['relative_start_date']
            tmp_data_only_valid['delta_time'] = -1
            tmp_data_only_valid['mse_coeff'] = 0
            tmp_data_only_valid['time_sense_coeff'] = 1
            y_for_deep_censored = y_for_deep_censored.append(
                tmp_data_only_valid[['relative_start_date', 'delta_time', 'relative_max_date', 'mse_coeff', 'time_sense_coeff']])

            # get only the last sample
            censored_data[subject_id] = tmp_data_only_valid.loc[
                tmp_data_only_valid['TIME_BEFORE_MOCO_START'] == min(tmp_data_only_valid['TIME_BEFORE_MOCO_START'])]

    else:  # not censored
        before_event_mask = subject_data['time_for_the_event'] > 0
        before_event_subjects = subject_data.loc[before_event_mask]
        if not before_event_subjects.empty:
            dilated_df = dilated_df.append(before_event_subjects)

            x_for_deep = x_for_deep.append(before_event_subjects)
            before_event_subjects['time_before_moco_start_days'] = before_event_subjects[
                'TIME_BEFORE_MOCO_START'].apply(get_days)
            before_event_subjects.sort_index(by='time_before_moco_start_days', ascending=False, inplace=True)
            before_event_subjects['relative_start_date'] = before_event_subjects['time_before_moco_start_days'].iloc[
                                                               0] - before_event_subjects['time_before_moco_start_days']
            before_event_subjects['relative_max_date'] = before_event_subjects['relative_start_date'] + \
                                                         before_event_subjects['time_before_moco_start_days']
            before_event_subjects['delta_time'] = before_event_subjects['time_for_the_event']
            before_event_subjects['mse_coeff'] = 1
            before_event_subjects['time_sense_coeff'] = 0
            y_for_deep = y_for_deep.append(
                before_event_subjects[['relative_start_date', 'delta_time', 'relative_max_date', 'mse_coeff', 'time_sense_coeff']])

x_for_deep = x_for_deep.join(otu_after_pca_wo_taxonomy)
x_for_deep = x_for_deep.loc[x_for_deep[0].notnull()]
y_for_deep = y_for_deep.loc[x_for_deep.index]

x_for_deep_censored = x_for_deep_censored.join(otu_after_pca_wo_taxonomy)
x_for_deep_censored = x_for_deep_censored.loc[x_for_deep_censored[0].notnull()]
y_for_deep_censored = y_for_deep_censored.loc[x_for_deep_censored.index]

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
    number_of_classes = 2
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
    X = dilated_df_clusterd.iloc[:, starting_col:starting_col + n_components]
    y = dilated_df_clusterd['cluster_number']
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
            class_prediction = {subject_id: neigh.predict(subject_data)[0] for subject_id, subject_data in
                                inputs.items()}

            inputs_per_class = {}
            for class_number in range(number_of_classes):
                inputs_per_class[class_number] = {}

            for (subject_id, class_predicted), subject_data in zip(class_prediction.items(), inputs.values()):
                inputs_per_class[class_predicted].update({subject_id: subject_data})

            K = {}
            K_t_time = {}
            K_t_time_multiplied_time_for_the_event = {}
            censored_data_with_time = pd.DataFrame()
            for class_number in range(number_of_classes):
                a = dilated_df.loc[dilated_df_clusterd['cluster_number'] == class_number]
                K[class_number], _ = calculate_distance(a[list(range(20))], inputs_per_class[class_number], beta,
                                                        visualize=False)
                K_t_time[class_number] = K[class_number].transpose().join(dilated_df['time_for_the_event'])
                K_t_time_multiplied_time_for_the_event[class_number] = K_t_time[class_number].apply(
                    multiply_by_time_for_the_event, axis=1)
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
                certainty_matrix = pd.DataFrame(np.ones((len(dilated_df.index), 1)), index=dilated_df.index)
                while True:
                    K, dist_matrix = calculate_distance(updated_df[list(range(n_components))], inputs, beta,
                                                        visualize=False)

                    ### choose the best censored subject and update K to contain only it###
                    certainties = np.dot(1 / dist_matrix.values, certainty_matrix.values)
                    best_candidate_idx = np.argmax(certainties)
                    best_candidate = dist_matrix.index[best_candidate_idx]
                    certainty_coefficent = certinty_factor  # * sigmoid(certainties[best_candidate_idx])
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
                    certainty_matrix = certainty_matrix.append(
                        pd.DataFrame(certainty_coefficent.tolist(), index=[best_candidate]))
                    updated_df = updated_df.append(censored_data_with_time_per_subject.join(otu_after_pca_wo_taxonomy))
                    inputs.pop(list(inputs.keys())[best_candidate_idx])
                    if len(inputs) == 0:  # means we finished iterating over the censored
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

    y_train_spearman_values = []
    y_test_spearman_values = []

    starting_col = np.argwhere(dilated_df.columns == 0).tolist()[0][0]
    X = dilated_df.iloc[:, starting_col:starting_col + n_components]
    y = dilated_df.loc[:, 'time_for_the_event']

    #
    starting_col = np.argwhere(x_for_deep.columns == 0).tolist()[0][0]
    X = x_for_deep.iloc[:, starting_col:starting_col + n_components]
    y = y_for_deep  # ['delta_time']

    if USE_SIMILARITY:
        X_train_censored = censored_data_with_time.iloc[:, starting_col:starting_col + n_components]
        y_train_censored = censored_data_with_time.loc[:, 'time_for_the_event']

    epochs_list = list(range(10,100,15))# + list(range(100,1100,200))
    lr_list = np.arange(0.01, 1, 0.015)
    dropout_list = np.arange(0.05, 0.8, 0.05)
    l2_lambda_list = np.arange(0, 2, 0.1)
    train_res, test_res = [], []
    for l2_lambda in l2_lambda_list:
        for dropput in dropout_list:
            for lr in lr_list:
                for epochs in epochs_list:
                    y_train_values = []
                    y_train_predicted_values = []

                    y_test_values = []
                    y_test_predicted_values = []

                    USE_CROSS_VAL = True
                    USE_LLO = False

                    if USE_CROSS_VAL:
                        number_iterations = 5
                    elif USE_LLO:
                        number_iterations = int(len(X))

                    current_configuration = {'l2': l2_lambda, 'dropout': dropput, 'lr': lr, 'epochs': epochs, 'number_iterations': number_iterations}
                    current_configuration_str = '^'.join(
                        [str(key) + '=' + str(value) for key, value in current_configuration.items()])
                    print(current_configuration)

                    for i in range(number_iterations):
                        print('\nIteration number: ', str(i + 1))
                        spearman_train_values = []
                        spearman_test_values = []
                        if USE_CROSS_VAL:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                            starting_col = np.argwhere(x_for_deep_censored.columns == 0).tolist()[0][0]
                            X_train_censored = x_for_deep_censored.iloc[:, starting_col:starting_col + n_components]
                            y_train_censored = y_for_deep_censored

                            # X_train = X_train.append(X_train_censored)
                            # y_train = y_train.append(y_train_censored)

                        elif USE_LLO:
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

                        algo_name = 'Neural Network'

                        test_model = tf_analaysis.nn_model()
                        regularizer = regularizers.l2(l2_lambda)
                        test_model.build_nn_model(hidden_layer_structure=[{'units': n_components, 'activation': tf.nn.relu, 'kernel_regularizer': regularizer},
                                                                          {'units': 50, 'activation': tf.nn.relu, 'kernel_regularizer': regularizer},
                                                                          ({'rate': dropput}, 'dropout'),
                                                                          {'units': 20, 'activation': tf.nn.relu,'kernel_regularizer': regularizer},
                                                                          ({'rate': dropput}, 'dropout'),
                                                                          {'units': 4, 'kernel_regularizer': regularizer}])
                        # sgd = tf.train.GradientDescentOptimizer(lr)
                        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                        test_model.compile_nn_model(loss=my_mse_loss, metrics=[my_mse_loss], optimizer=optimizer)
                        test_model.train_model(X_train.values, y_train.values.astype(np.float), epochs=epochs, callbacks=[callbacks_])
                        test_model.evaluate_model(X_test.values, y_test.values.astype(np.float))

                        y_train_values.append(y_train.values[:, 1])
                        y_train_predicted_values.append(test_model.predict(X_train.values)[:, 1])

                        y_test_values.append(y_test.values[:, 1])
                        y_test_predicted_values.append(test_model.predict(X_test.values)[:, 0])

                    #### END OF CONFIGURATION OPTION  ####
                    y_train_values = [item for sublist in y_train_values for item in sublist]
                    y_train_predicted_values = [item for sublist in y_train_predicted_values for item in sublist]

                    y_test_values = [item for sublist in y_test_values for item in sublist]
                    y_test_predicted_values = [item for sublist in y_test_predicted_values for item in sublist]

                    current_train_res, current_test_res = calc_results_and_plot(y_train_values, y_train_predicted_values,
                                                                                y_test_values,
                                                                                y_test_predicted_values, algo_name='NeuralNetwork',
                                                                                visualize=PLOT,
                                                                                title=f'Epochs: {epochs}, Validation iterations: {number_iterations}',
                                                                                show=False)

                    # print(current_train_res)
                    # print(current_test_res)
                    if RECORD:
                        GRID_SEARCH_DIRECTORY = 'GridSearch'
                        if not os.path.exists(GRID_SEARCH_DIRECTORY):
                            os.mkdir(GRID_SEARCH_DIRECTORY)
                        if not os.path.exists(GRID_SEARCH_DIRECTORY+'/'+current_configuration_str):
                            os.mkdir(GRID_SEARCH_DIRECTORY+'/'+current_configuration_str)
                        np.save(GRID_SEARCH_DIRECTORY+'/'+current_configuration_str+'/y_train_values.npy', y_train_values)
                        np.save(GRID_SEARCH_DIRECTORY+'/'+current_configuration_str+'/y_train_predicted_values.npy', y_train_predicted_values)
                        np.save(GRID_SEARCH_DIRECTORY+'/'+current_configuration_str+'/y_test_values.npy', y_test_values)
                        np.save(GRID_SEARCH_DIRECTORY+'/'+current_configuration_str+'/y_test_predicted_values.npy', y_test_predicted_values)
                    with open(GRID_SEARCH_DIRECTORY+'/'+'grid_search_results.txt', 'a') as f:
                        f.writelines(['\n',current_configuration_str, '\n','Train\n ' ,str(current_train_res),'\nTest\n ' ,str(current_test_res), '\n'])
                    with open(GRID_SEARCH_DIRECTORY+'/'+current_configuration_str+'/'+ 'grid_search_results.txt', 'a') as f:
                        f.writelines( ['Train\n ', str(current_train_res), '\nTest\n ',str(current_test_res), '\n'])


                    train_res.append(current_train_res)
                    test_res.append(current_test_res)

train_rhos = []
test_rhos = []
for train_rho, test_rho in zip(train_res, test_res):
    train_rhos.append(train_rho['spearman']['rho'])
    test_rhos.append(test_rho['spearman']['rho'])
plt.figure()
plt.scatter(list(epochs_list), train_rhos, label='Train')
plt.scatter(list(epochs_list), test_rhos, label='Test')
plt.legend()
plt.show()
