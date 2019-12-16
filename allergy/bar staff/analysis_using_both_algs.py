from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
import tensorflow as tf
import warnings
tf.enable_eager_execution()

from infra_functions.general import apply_pca, use_spearmanr, use_pearsonr  #
from infra_functions.time_series_analsys import compute_time_for_censored_using_similarity_matrix, time_series_analysis_tf
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pickle

import datetime

import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RECORD = True
PLOT = False
USE_SIMILARITY = False
PLOT_INPUT_TO_NN_STATS = False


def print_recursive(object_to_print, count=0):
    replace_char_list = ['_', ' ']
    spaces = count * ' '
    if type(object_to_print) is dict:
        count += 1
        for key, val in object_to_print.items():
            if key in ['mean_time_to_event', 'samples_number', 'squared_mean']:
                print(f'{spaces}{key.replace(replace_char_list[0], replace_char_list[1])}: {val}')
            else:
                print(f'{spaces}{key.replace(replace_char_list[0], replace_char_list[1])}')
                print_recursive(val, count=count)

    else:
        # spaces = (count-1) * '\t'+ ' '
        print(f'{spaces}{object_to_print}')

def stats_input(uncensored, censored, verbose=True):
    stats = {'uncensored': {'mean_time_to_event': uncensored['delta_time'].mean(),
                            'squared_mean': uncensored['delta_time'].mean() * uncensored['delta_time'].mean(),
                            'samples_number': uncensored.shape[0]},
             'censored': {'samples_number': censored.shape[0]}}
    if verbose:
        print('\n\nStats of subjects (Uncensored and Censored)\n'
              '-------------------------------------------')
        print_recursive(stats)
    return stats

# @autograph.convert()
def my_loss(y_true, y_pred):
    mse_loss = my_mse_loss(y_true, y_pred)

    time_sense_loss = y_true[:, 2] - y_pred[:, 1]  # Max_delta - predicted_delta should be negative
    tsls = time_sense_loss #tf.square(time_sense_loss)

    return y_true[:, 4] * tsls + y_true[:, 3] * mse_loss

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


def get_datetime(date_str, time_format='%d/%m/%Y'):
    if pd.isnull(date_str):
        date_str = '01/01/1900'
        return datetime.datetime.strptime(date_str, '%d/%m/%Y')
    try:
        return datetime.datetime.strptime(date_str, time_format)
    except ValueError:
        try:
            # use %Y
            time_format = time_format.split('/')
            time_format[-1] = '%Y'
            time_format = '/'.join(time_format)
            return datetime.datetime.strptime(date_str, time_format)
        except ValueError:
            try:
                # use %y
                time_format = time_format.split('/')
                time_format[-1] = '%y'
                time_format = '/'.join(time_format)
                return datetime.datetime.strptime(date_str, time_format)
            except:
                warnings.warn(f'{date_str} is not a valid date, sample will be ignored')
                date_str = '01/01/1800'
                return datetime.datetime.strptime(date_str, '%d/%m/%Y')


def get_days(days_datetime):
    return days_datetime.days


n_components = 20
use_recorded = False

if not use_recorded:
    OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'),
                         os.path.join(SCRIPT_DIR, 'mf_merge_ok84_ok93_ok66_69_merged_by_RestoredSampleCode_as_ID_290119.csv'),
                         from_QIIME=True, id_col='Feature ID', taxonomy_col='Taxonomy')
    preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=5, taxonomy_col='Taxonomy',
                                         preform_taxnomy_group=True)
    otu_after_pca_wo_taxonomy, _, _ = apply_pca(preproccessed_data, n_components=n_components, visualize=False)

    ######## Pre process (Remove control group) ########
    column_to_use_for_filter = 'AllergyTypeData131118'
    OtuMf.mapping_file = OtuMf.mapping_file.loc[OtuMf.mapping_file['AllergyTypeData131118'] != 'Con']

    ######## get date of sample in date format ########
    date_of_sample_col = 'Date'
    OtuMf.mapping_file['Date_of_sample'] = OtuMf.mapping_file[date_of_sample_col].apply(get_datetime, time_format='%m/%d/%y')

    ######## remove invalid subjects (those who had samples with no dates or bad dates) ########
    # bad dates
    tmp = OtuMf.mapping_file.loc[OtuMf.mapping_file['Date_of_sample'].isin(['1800-01-01', '1900-01-01'])]
    patients_with_bad_date = tmp['PatientNumber210119'].unique()
    # remove bad dates
    OtuMf.mapping_file = OtuMf.mapping_file.loc[~OtuMf.mapping_file['PatientNumber210119'].isin(patients_with_bad_date)]

    ######## Calculate time for event ########
    OtuMf.mapping_file['time_for_the_event'] = 9999
    col_to_group_by = 'PatientNumber210119'
    data_grouped = OtuMf.mapping_file.groupby(col_to_group_by)

    for subject_id, subject_data in data_grouped:
        if any(subject_data['SuccessDescription'] == 'A1'):  # Uncensored
            date_of_event = subject_data['Date_of_sample'].max()
            time_for_the_event = date_of_event-subject_data['Date_of_sample']
            tmp_df = OtuMf.mapping_file.loc[subject_data.index]
            tmp_df['time_for_the_event'] = time_for_the_event.apply(get_days)
            OtuMf.mapping_file.update(tmp_df)
        else:  # Censored
            pass

    ######## Filter alergies ########
    # allergy types ['Sesame', 'Peanut', 'Egg', 'Non', 'Walnut', 'Milk', 'Cashew', 'Hazelnut']
    # OtuMf.mapping_file['AllergyTypeData131118'].value_counts()
    # Peanut    134
    # Milk    112
    # Sesame    80
    # Walnut    72
    # Egg    28
    # Cashew    18
    # Hazelnut    9
    # Non    9
    allergy_to_use = ['Peanut']
    OtuMf.mapping_file = OtuMf.mapping_file[OtuMf.mapping_file['AllergyTypeData131118'].isin(allergy_to_use)]


    ######## Create inputs ########

    # create groups
    col_to_group_by = 'PatientNumber210119'
    data_grouped = OtuMf.mapping_file.groupby(col_to_group_by)
    censored_data = {}
    not_censored = pd.DataFrame()
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

                tmp_data_only_valid.sort_index(by='Date_of_sample', ascending=True, inplace=True)
                tmp_data_only_valid['relative_start_date'] = (tmp_data_only_valid['Date_of_sample']- tmp_data_only_valid['Date_of_sample'].iloc[0]).apply(get_days)
                tmp_data_only_valid['relative_max_date'] = (tmp_data_only_valid['Date_of_sample'].iloc[-1] - tmp_data_only_valid['Date_of_sample']).apply(get_days)
                tmp_data_only_valid['delta_time'] = -1
                tmp_data_only_valid['mse_coeff'] = 0
                tmp_data_only_valid['time_sense_coeff'] = 1
                y_for_deep_censored = y_for_deep_censored.append(
                    tmp_data_only_valid[['relative_start_date', 'delta_time', 'relative_max_date', 'mse_coeff', 'time_sense_coeff']])

                # get only the last sample
                censored_data[subject_id] = tmp_data_only_valid.loc[tmp_data_only_valid['relative_max_date'] == min(tmp_data_only_valid['relative_max_date'])]

        else:  # not censored
            before_event_mask = subject_data['time_for_the_event'] > 0
            before_event_subjects = subject_data.loc[before_event_mask]
            if not before_event_subjects.empty:
                not_censored = not_censored.append(before_event_subjects)

                x_for_deep = x_for_deep.append(before_event_subjects)
                before_event_subjects.sort_index(by='time_for_the_event', ascending=False, inplace=True)
                before_event_subjects['relative_start_date'] = before_event_subjects['time_for_the_event'].iloc[
                                                                   0] - before_event_subjects['time_for_the_event']
                before_event_subjects['relative_max_date'] = before_event_subjects['time_for_the_event']
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
else:
    x_for_deep = pickle.load(open("x_for_deep.p", "rb"))
    y_for_deep = pickle.load(open("y_for_deep.p", "rb"))
    x_for_deep_censored = pickle.load(open("x_for_deep_censored.p", "rb"))
    y_for_deep_censored = pickle.load(open("y_for_deep_censored.p", "rb"))
    censored_data = pickle.load(open("censored_data.p", "rb"))


record_for_deep_inputs_before_similarity = False
if record_for_deep_inputs_before_similarity:
    pickle.dump(x_for_deep, open("x_for_deep.p", "wb" ))
    pickle.dump(y_for_deep, open("y_for_deep.p", "wb" ))
    pickle.dump(x_for_deep_censored, open("x_for_deep_censored.p", "wb" ))
    pickle.dump(y_for_deep_censored, open("y_for_deep_censored.p", "wb" ))
    pickle.dump(censored_data, open("censored_data.p", "wb" ))


if USE_SIMILARITY:
    ##### Similarity algo ####
    not_censored = not_censored.join(otu_after_pca_wo_taxonomy)

    censored_data_with_time = compute_time_for_censored_using_similarity_matrix(not_censored,
                                                                                censored_data,
                                                                                n_components,
                                                                                OtuMf,
                                                                                otu_after_pca_wo_taxonomy,
                                                                                beta=9,
                                                                                remove_outliers=True,
                                                                                th_value=None)

    # combine the x_censored and the syntethic time
    x_for_deep_censored['time_for_the_event'][censored_data_with_time['time_for_the_event'].index] = \
        censored_data_with_time['time_for_the_event']
    y_for_deep_censored['delta_time'][censored_data_with_time['time_for_the_event'].index] = censored_data_with_time['time_for_the_event']

    # change the MSE coeff for the last sample of censored
    y_for_deep_censored['mse_coeff'][censored_data_with_time['time_for_the_event'].index] = 5

    ##### END Similarity algo ####


starting_col = np.argwhere(x_for_deep.columns == 0).tolist()[0][0]
X = x_for_deep.iloc[:, starting_col:starting_col + n_components]
y = y_for_deep  # ['delta_time']

starting_col = np.argwhere(x_for_deep_censored.columns == 0).tolist()[0][0]
X_train_censored = x_for_deep_censored.iloc[:, starting_col:starting_col + n_components]
y_train_censored = y_for_deep_censored
number_samples_censored = y_train_censored.shape[0]
print(f'Number of censored subjects: {number_samples_censored}')

before_removal = y.shape[0]
# remove outliers
std = y['delta_time'].values.std()
th = std * 5

outlier_mask = y['delta_time'] < th
y = y.loc[outlier_mask]
X = X.loc[outlier_mask]

after_removal = y.shape[0]
print(f'{before_removal-after_removal} outlier/s were removed')

stats_of_input = stats_input(y, y_train_censored)

if PLOT_INPUT_TO_NN_STATS:
    plt.hist(y['delta_time'].values, bins=150)
    b = y['delta_time'].values.copy()
    b.sort()
    med = b[int(len(b)/2)]
    std = y['delta_time'].values.std()
    mean = y['delta_time'].values.mean()

    plt.title(f'STD={std}, MED={med}, Mean={mean}')

# remove outliers

epochs_list = [20, 80]#[10, 50, 100] #list(range(10,100,20)) + list(range(100,200,30))
mse_factor_list = [0.1, 1, 10, 100, 1000] # np.arange(0.005, 1, 0.005)
if not USE_SIMILARITY:
    mse_factor_list = [1]
    X_train_censored = None
    y_train_censored = None
dropout_list = [0, 0.2, 0.6] #np.arange(0, 0.8, 0.1)
l2_lambda_list = [1, 10, 20, 100]
#np.logspace(0, 2, 5) #  0.01, 0.1, 1, 10, 100
number_layers_list = [1, 2, 3]
number_neurons_per_layer_list = [20, 50]

#best conf
# l2=10^dropout=0.6^factor=10^epochs=20^number_iterations=5^number_layers=1^neurons_per_layer=50
# epochs_list = [20]#[10, 50, 100] #list(range(10,100,20)) + list(range(100,200,30))
# mse_factor_list = [10] # np.arange(0.005, 1, 0.005)
# dropout_list = [0.6] #np.arange(0, 0.8, 0.1)
# l2_lambda_list = [10]
# #np.logspace(0, 2, 5) #  0.01, 0.1, 1, 10, 100
# number_layers_list = [1]
# number_neurons_per_layer_list = [50]


train_res, test_res  = time_series_analysis_tf(X, y,
                                               n_components,
                                               l2_lambda_list,
                                               dropout_list,
                                               mse_factor_list,
                                               number_layers_list,
                                               number_neurons_per_layer_list,
                                               epochs_list,
                                               cross_val_number=5,
                                               X_train_censored=X_train_censored,
                                               y_train_censored=y_train_censored,
                                               record=RECORD,
                                               grid_search_dir='grid_search_no_censored')

# train_rhos = []
# test_rhos = []
# for train_rho, test_rho in zip(train_res, test_res):
#     train_rhos.append(train_rho['spearman']['rho'])
#     test_rhos.append(test_rho['spearman']['rho'])
# plt.figure()
# plt.scatter(list(epochs_list), train_rhos, label='Train')
# plt.scatter(list(epochs_list), test_rhos, label='Test')
# plt.legend()
# plt.show()




