
import tensorflow as tf
import random
import time
from tensorflow.python.keras import backend as K

tf.enable_eager_execution()
from tensorflow.python.keras import optimizers, regularizers, callbacks
from infra_functions import tf_analaysis

import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut

from GVHD_BAR.show_data import calc_results_and_plot
from GVHD_BAR.calculate_distances import calculate_distance
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RECORD = True
PLOT = False
USE_SIMILARITY = False
PLOT_INPUT_TO_NN_STATS = False


PADDED_VALUE = -999

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
             'censored': {'samples_number': 'Not using' if censored is None else censored.shape[0]}}
    if verbose:
        print('\n\nStats of subjects (Uncensored and Censored)\n'
              '-------------------------------------------')
        print_recursive(stats)
    return stats

def my_loss(y_true, y_pred):
    mse_loss = my_mse_loss(y_true, y_pred)

    time_sense_loss = y_true[:, 2] - y_pred[:, 1]  # Max_delta - predicted_delta should be negative
    tsls = time_sense_loss #tf.square(time_sense_loss)

    return y_true[:, 4] * tsls + y_true[:, 3] * mse_loss


def my_loss_batch(y_true, y_pred):
    batch_size = y_pred.shape[0]
    steps_size = y_pred.shape[1]

    total_loss = 0
    for batch_idx in range(batch_size):
        single_y_true = y_true[batch_idx, :, :]
        single_y_pred = y_pred[batch_idx, :, :]
        tmp = np.array(single_y_true)
        mask = np.all(tmp!=PADDED_VALUE, axis=1)
        single_y_true = tmp[mask,:].reshape(-1, tmp.shape[1])

        tmp = np.array(single_y_pred)
        single_y_pred = tmp[mask,:].reshape(-1, tmp.shape[1])

        total_loss += sum(my_loss(single_y_true, single_y_pred))

    return total_loss / (batch_size*steps_size).value


def my_mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1]))

    return mse_loss

def time_series_using_xgboost(X, y,
                              alpha_list,
                              n_estimators_list,
                              min_child_weight_list,
                              reg_lambda_list,
                              max_depth_list,
                              cross_val_number=5,
                              X_train_censored=None,
                              y_train_censored=None,
                              record=RECORD,
                              grid_search_dir='grid_search_xgboost',
                              deep_verbose=False,
                              beta_for_similarity=None,
                              use_random_time=None):

    if len(y.shape) == 1:
        y = y.to_frame(name='delta_time')
        y_train_censored = y_train_censored if y_train_censored is None else y_train_censored.to_frame(name='delta_time')

    print(f'\nUsing xgboost analysis\n')
    stats_of_input = stats_input(y, y_train_censored, verbose=True)
    train_res, test_res = {}, {}
    for alpha in alpha_list:
        for n_estimators in n_estimators_list:
            for min_child_weight in min_child_weight_list:
                for reg_lambda in reg_lambda_list:
                    for max_depth in max_depth_list:
                        y_train_values = []
                        y_train_predicted_values = []

                        y_test_values = []
                        y_test_predicted_values = []

                        USE_CROSS_VAL = True
                        USE_LLO = False

                        if USE_CROSS_VAL:
                            number_iterations = cross_val_number
                        elif USE_LLO:
                            number_iterations = int(len(X))

                        current_configuration = {'alpha': alpha, 'n_estimators': n_estimators, 'min_child_weight': min_child_weight,
                                                 'reg_lambda': reg_lambda,'max_depth': max_depth}
                        if beta_for_similarity is not None:
                            current_configuration.update({'beta_for_similarity': beta_for_similarity})

                        current_configuration_str = '^'.join(
                            [str(key) + '=' + str(value) for key, value in current_configuration.items()])
                        print(f'Current config: {current_configuration}')

                        hist=[]
                        for i in range(number_iterations):

                            # sleep for random time to avoid collisions :O
                            # if use_random_time:
                            #     random_time_to_wait = random.random()
                            #     time.sleep(random_time_to_wait)

                            print('\nIteration number: ', str(i + 1))
                            if USE_CROSS_VAL:
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                            elif USE_LLO:
                                X_test = X.iloc[i].to_frame().transpose()
                                y_test = pd.Series(y.iloc[i], [y.index[i]])
                                X_train = X.drop(X.index[i])
                                y_train = y.drop(y.index[i])

                            # add censored
                            X_train = X_train.append(X_train_censored)
                            y_train = y_train.append(y_train_censored)

                            # shuffle
                            idx = np.random.permutation(X_train.index)
                            X_train = X_train.reindex(idx)
                            y_train = y_train.reindex(idx)


                            ### the actual regressor ###
                            xg_reg = XGBRegressor(objective='reg:linear', colsample_bytree=1, learning_rate=0.1,
                                                  reg_lambda=reg_lambda,
                                                  max_depth=max_depth, alpha=alpha, n_estimators=n_estimators,
                                                  min_child_weight=min_child_weight)
                            eval_set = [(X_train, y_train), (X_test, y_test)]
                            eval_metric = ["rmse"]

                            xg_reg.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=deep_verbose)

                            tmp_eval = xg_reg.evals_result()
                            hist.append({'train': tmp_eval['validation_0']['rmse'][-1],
                                         'test': tmp_eval['validation_1']['rmse'][-1]})

                            y_train_values.append(y_train.values.ravel())
                            y_train_predicted_values.append(xg_reg.predict(X_train))

                            y_test_values.append(y_test.values.ravel())
                            y_test_predicted_values.append(xg_reg.predict(X_test))

                        #### END OF CONFIGURATION OPTION  ####
                        y_train_values = [item for sublist in y_train_values for item in sublist]
                        y_train_predicted_values = [item for sublist in y_train_predicted_values for item in sublist]

                        # remove the -1 values (the ones that are censored)
                        tmp = [i for i in zip(y_train_values, y_train_predicted_values) if int(i[0]) != -1]
                        y_train_values = [i[0] for i in tmp]
                        y_train_predicted_values = [i[1] for i in tmp]

                        y_test_values = [item for sublist in y_test_values for item in sublist]
                        y_test_predicted_values = [item for sublist in y_test_predicted_values for item in sublist]

                        current_train_res, current_test_res = calc_results_and_plot(y_train_values,
                                                                                    y_train_predicted_values,
                                                                                    y_test_values,
                                                                                    y_test_predicted_values,
                                                                                    algo_name='XGBoost',
                                                                                    visualize=PLOT,
                                                                                    title=f'Validation iterations: {number_iterations}',
                                                                                    show=False)

                        # print(current_train_res)
                        # print(current_test_res)
                        if record:
                            record_results(grid_search_dir,
                                           current_configuration_str,
                                           y_train_values,
                                           y_train_predicted_values,
                                           y_test_values,
                                           y_test_predicted_values,
                                           stats_of_input,
                                           current_train_res,
                                           current_test_res,
                                           hist)

                        train_res.update({current_configuration_str: current_train_res})
                        test_res.update({current_configuration_str: current_test_res})

    return train_res, test_res


def compute_time_for_censored_using_similarity_matrix(not_censored_data,
                                                      censored_data,
                                                      number_pca_used_in_data,
                                                      OtuMf,
                                                      otu_after_pca_wo_taxonomy,
                                                      beta,
                                                      remove_outliers=True,
                                                      th_value=None):

    print(f'Using similarity with beta = {beta}')

    # remove subects with no data in mocrobiome
    not_censored_data = not_censored_data.loc[not_censored_data[0].notnull()]
    before_removal = not_censored_data.shape[0]

    # remove outliers
    if remove_outliers:
        std = not_censored_data['time_for_the_event'].values.std()
        th = std * 5 if th_value is None else th_value

        outlier_mask = not_censored_data['time_for_the_event'] < th
        not_censored_data = not_censored_data.loc[outlier_mask]

        after_removal = not_censored_data.shape[0]
        print(f'Similarity outlier removal: {before_removal - after_removal} outlier/s were removed')

    def multiply_by_time_for_the_event(col):
        return col.apply(lambda x: x * col['time_for_the_event'])

    inputs = {subject_id: subject_data[list(range(number_pca_used_in_data))] for subject_id, subject_data in
              censored_data.items()}

    K, _ = calculate_distance(not_censored_data[list(range(number_pca_used_in_data))], inputs, beta, visualize=False)
    K_t_time = K.transpose().join(not_censored_data['time_for_the_event'])

    K_t_time_multiplied_time_for_the_event = K_t_time.apply(multiply_by_time_for_the_event, axis=1)

    denominator = K_t_time.sum()
    nominator = K_t_time_multiplied_time_for_the_event.sum()
    censored_time = nominator / denominator
    censored_data_with_time = OtuMf.mapping_file.loc[censored_time.index[:-1].tolist()]
    censored_data_with_time['time_for_the_event'] = censored_time

    censored_data_with_time = censored_data_with_time.join(otu_after_pca_wo_taxonomy)
    number_of_rows_before_removal = censored_data_with_time.shape[0]
    # remove subects with no data in mocrobiome
    censored_data_with_time = censored_data_with_time.loc[censored_data_with_time[0].notnull()]

    # remove subjects that are unable to calculate the synthetic time for the event
    censored_data_with_time = censored_data_with_time.loc[censored_data_with_time['time_for_the_event'].notnull()]
    number_of_rows_after_removal = censored_data_with_time.shape[0]
    removed_rows = number_of_rows_before_removal - number_of_rows_after_removal
    print(f'Removed {removed_rows} due to a problem with calculating synthetic time')

    return censored_data_with_time

def time_series_analysis_tf(X, y,
                            input_size,
                            l2_lambda_list,
                            dropout_list,
                            mse_factor_list,
                            number_layers_list,
                            number_neurons_per_layer_list,
                            epochs_list,
                            cross_val_number=5,
                            X_train_censored=None,
                            y_train_censored=None,
                            record=RECORD,
                            grid_search_dir='grid_search_tf',
                            beta_for_similarity=None,
                            censored_mse_fraction_factor=None):

    print(f'\nUsing tf analysis\n')
    stats_of_input = stats_input(y, y_train_censored, verbose=True)
    train_res, test_res = {}, {}
    for l2_lambda in l2_lambda_list:
        for dropout in dropout_list:
            for factor in mse_factor_list:
                for number_layers in number_layers_list:
                    for number_neurons_per_layer in number_neurons_per_layer_list:
                        for epochs in epochs_list:
                            # clear the model
                            K.clear_session()

                            y_train_values = []
                            y_train_predicted_values = []

                            y_test_values = []
                            y_test_predicted_values = []

                            USE_CROSS_VAL = True
                            USE_LLO = False

                            if USE_CROSS_VAL:
                                number_iterations = cross_val_number
                            elif USE_LLO:
                                number_iterations = int(len(X))

                            current_configuration = {'l2': l2_lambda, 'dropout': dropout, 'factor': factor, 'epochs': epochs,
                                                     'number_iterations': number_iterations, 'number_layers': number_layers, 'neurons_per_layer': number_neurons_per_layer}

                            if censored_mse_fraction_factor is not None:
                                # use mse factor of censored_mse_fraction_factor of the uncensored for the censored samples
                                y_train_censored['mse_coeff'].loc[y_train_censored[
                                                                      'mse_coeff'] == 'last_censored'] = factor / censored_mse_fraction_factor
                                current_configuration.update({'censored_mse_factor': factor / censored_mse_fraction_factor})

                            if beta_for_similarity is not None:
                                current_configuration.update({'beta_for_similarity': beta_for_similarity})

                            current_configuration_str = '^'.join(
                                [str(key) + '=' + str(value) for key, value in current_configuration.items()])
                            print(f'Current config: {current_configuration}')

                            for i in range(number_iterations):
                                print('\nIteration number: ', str(i + 1))
                                if USE_CROSS_VAL:
                                    y['mse_coeff'] = y['mse_coeff'].astype(float)
                                    y['mse_coeff'] = factor
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                                elif USE_LLO:
                                    X_test = X.iloc[i].to_frame().transpose()
                                    y_test = pd.Series(y.iloc[i], [y.index[i]])
                                    X_train = X.drop(X.index[i])
                                    y_train = y.drop(y.index[i])

                                # add censored
                                X_train = X_train.append(X_train_censored)
                                y_train = y_train.append(y_train_censored)

                                # shuffle
                                idx = np.random.permutation(X_train.index)
                                X_train = X_train.reindex(idx)
                                y_train = y_train.reindex(idx)

                                algo_name = 'Neural Network'

                                test_model = tf_analaysis.nn_model()
                                regularizer = regularizers.l2(l2_lambda)

                                model_structure = [{'units': input_size, 'activation': tf.nn.relu, 'kernel_regularizer': regularizer}]
                                for layer_idx in range(number_layers):
                                    model_structure.append({'units': number_neurons_per_layer, 'activation': tf.nn.relu, 'kernel_regularizer': regularizer})
                                    model_structure.append(({'rate': dropout}, 'dropout'))

                                model_structure.append({'units': 4, 'kernel_regularizer': regularizer})
                                test_model.build_nn_model(hidden_layer_structure=model_structure)

                                test_model.compile_nn_model(loss=my_loss, metrics=[my_loss])
                                hist = test_model.train_model(X_train.values, y_train.values.astype(np.float), epochs=epochs, verbose=False)
                                # plt.plot(hist.history['loss'])
                                test_model.evaluate_model(X_test.values, y_test.values.astype(np.float))

                                y_train_values.append(y_train.values[:, 1])
                                y_train_predicted_values.append(test_model.predict(X_train.values)[:, 1])

                                y_test_values.append(y_test.values[:, 1])
                                y_test_predicted_values.append(test_model.predict(X_test.values)[:, 1])

                            #### END OF CONFIGURATION OPTION  ####
                            y_train_values = [item for sublist in y_train_values for item in sublist]
                            y_train_predicted_values = [item for sublist in y_train_predicted_values for item in sublist]

                            # remove the -1 values (the ones that are censored)
                            tmp = [i for i in zip(y_train_values, y_train_predicted_values) if int(i[0]) != -1]
                            y_train_values = [i[0] for i in tmp]
                            y_train_predicted_values = [i[1] for i in tmp]

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
                            if record:
                                record_results(grid_search_dir,
                                               current_configuration_str,
                                               y_train_values,
                                               y_train_predicted_values,
                                               y_test_values,
                                               y_test_predicted_values,
                                               stats_of_input,
                                               current_train_res,
                                               current_test_res,
                                               hist.history)

                            train_res.update({current_configuration_str: current_train_res})
                            test_res.update({current_configuration_str: current_test_res})

    return train_res, test_res


def time_series_analysis_rnn(X, y,
                            input_size,
                            l2_lambda_list,
                            dropout_list,
                            mse_factor_list,
                            number_layers_list,
                            number_neurons_per_layer_list,
                            epochs_list,
                            cross_val_number=5,
                            X_train_censored=None,
                            y_train_censored=None,
                            record=RECORD,
                            grid_search_dir='grid_search_rnn',
                            beta_for_similarity=None,
                            censored_mse_fraction_factor=None):

    print(f'\nUsing lstm analysis\n')
    stats_of_input = stats_input(y, y_train_censored, verbose=True)
    train_res, test_res = {}, {}
    for l2_lambda in l2_lambda_list:
        for dropout in dropout_list:
            for factor in mse_factor_list:
                for number_layers in number_layers_list:
                    for number_neurons_per_layer in number_neurons_per_layer_list:
                        for epochs in epochs_list:
                            # clear the model
                            K.clear_session()

                            y_train_values = []
                            y_train_predicted_values = []

                            y_test_values = []
                            y_test_predicted_values = []

                            USE_CROSS_VAL = True
                            USE_LLO = False

                            if USE_CROSS_VAL:
                                number_iterations = cross_val_number
                            elif USE_LLO:
                                number_iterations = int(len(X))

                            current_configuration = {'l2': l2_lambda, 'dropout': dropout, 'factor': factor, 'epochs': epochs,
                                                     'number_iterations': number_iterations, 'number_layers': number_layers, 'neurons_per_layer': number_neurons_per_layer}

                            if censored_mse_fraction_factor is not None:
                                # use mse factor of censored_mse_fraction_factor of the uncensored for the censored samples
                                y_train_censored['mse_coeff'].loc[y_train_censored[
                                                                      'mse_coeff'] == 'last_censored'] = factor / censored_mse_fraction_factor
                                current_configuration.update({'censored_mse_factor': factor / censored_mse_fraction_factor})

                            if beta_for_similarity is not None:
                                current_configuration.update({'beta_for_similarity': beta_for_similarity})

                            current_configuration_str = '^'.join(
                                [str(key) + '=' + str(value) for key, value in current_configuration.items()])
                            print(f'Current config: {current_configuration}')

                            for i in range(number_iterations):
                                print('\nIteration number: ', str(i + 1))
                                if USE_CROSS_VAL:
                                    y['mse_coeff'] = y['mse_coeff'].astype(float)
                                    y['mse_coeff'] = factor


                                    # split the data such that a sample is only in one group, or the train or the test
                                    data_grouped = X.groupby('groupby')

                                    groups = list(data_grouped.groups.keys())

                                    shuffled_idx = list(np.random.permutation(len(groups)))
                                    X_train = pd.DataFrame()
                                    min_x_train_len = np.ceil(0.8 * len(X))
                                    for idx in shuffled_idx:
                                        group_name_to_take = groups[idx]
                                        shuffled_idx.pop()
                                        group_to_take = data_grouped.get_group(group_name_to_take)
                                        X_train = X_train.append(group_to_take)
                                        if len(X_train) > min_x_train_len:
                                            break
                                    y_train = y.loc[X_train.index]

                                    X_test = pd.DataFrame()
                                    for idx in shuffled_idx:
                                        group_name_to_take = groups[idx]
                                        shuffled_idx.pop()
                                        group_to_take = data_grouped.get_group(group_name_to_take)
                                        X_test = X_test.append(group_to_take)
                                    y_test = y.loc[X_test.index]

                                # add censored
                                X_train = X_train.append(X_train_censored)
                                y_train = y_train.append(y_train_censored)



                                algo_name = 'LSTM Network'

                                test_model = tf_analaysis.nn_model()
                                regularizer = regularizers.l2(l2_lambda)

                                model_structure = [({'units': 2*input_size, 'input_shape': (None, input_size), 'return_sequences': True}, 'LSTM')]
                                for layer_idx in range(number_layers):
                                    model_structure.append({'units': number_neurons_per_layer, 'activation': tf.nn.relu, 'kernel_regularizer': regularizer})
                                    model_structure.append(({'rate': dropout}, 'dropout'))


                                model_structure.append({'units': 4, 'kernel_regularizer': regularizer})
                                test_model.build_nn_model(hidden_layer_structure=model_structure)

                                test_model.compile_nn_model(loss=my_loss_batch, metrics=[my_loss_batch])

                                def sample_generator(inputs, targets, batch_size=1):
                                    data_grouped = inputs.groupby('groupby')

                                    keys, values = [], []
                                    for key, value in data_grouped.groups.items():
                                        keys.append(key)
                                        values.append(value)

                                    timestep_in_group = [len(x) for x in list(values)]
                                    max_timestep_in_group = [np.max(x) for x in np.array_split(timestep_in_group, np.ceil(len(timestep_in_group)/batch_size))]
                                    batches = np.array_split(keys, np.ceil(len(keys)/batch_size))

                                    subject_data = data_grouped.get_group(keys[0])
                                    x_time_step = subject_data.drop('groupby', axis=1)
                                    sample_targets = targets.loc[x_time_step.index].values

                                    for batch, max in zip(batches, max_timestep_in_group):
                                        x_batch_to_return = np.ndarray((batch_size, max, x_time_step.shape[1]))
                                        target_batch_to_return = np.ndarray((batch_size, max, sample_targets.shape[1]))
                                        for idx, group_in_batch  in enumerate(batch):
                                            subject_data = data_grouped.get_group(group_in_batch)
                                            x_time_step = subject_data.drop('groupby', axis=1)
                                            sample_targets = targets.loc[x_time_step.index].values
                                            x_time_step=x_time_step.values
                                            number_of_zero_rows = max - len(x_time_step)
                                            rows_to_add = np.zeros((number_of_zero_rows, x_time_step.shape[1]))
                                            x_time_step = np.vstack([x_time_step, rows_to_add])
                                            x_batch_to_return[idx, :, :] = x_time_step


                                            rows_to_add = PADDED_VALUE * np.ones((number_of_zero_rows, sample_targets.shape[1]))
                                            sample_targets = np.vstack([sample_targets, rows_to_add])
                                            target_batch_to_return[idx, :, :] = sample_targets.astype(np.float)

                                        yield x_batch_to_return, target_batch_to_return

                                    # for subject_id, subject_data in data_grouped:
                                    #     # print(subject_data)
                                    #     x_time_step = subject_data.drop('groupby', axis=1)
                                    #     sample_targets = targets.loc[x_time_step.index]
                                    #     input_shape = x_time_step.values.shape
                                    #     target_shape = sample_targets.shape
                                    #
                                    #     x_time_step = x_time_step.values.reshape(1, input_shape[0], input_shape[1])
                                    #     target = sample_targets.values.reshape(1, target_shape[0], target_shape[1]).astype(
                                    #         np.float)
                                    # yield x_time_step, target

                                train_samples = sample_generator(X_train, y_train, batch_size=3)
                                for input_sample, target_sample in train_samples:
                                    hist = test_model.train_model(input_sample, target_sample, epochs=epochs, verbose=False, batch_size=3)

                                # plt.plot(hist.history['loss'])

                                # # test the model
                                # test_samples = sample_generator(X_test, y_test)
                                # for input_sample, target_sample in test_samples:
                                #     test_model.evaluate_model(input_sample, target_sample)

                                y_train_values.append(y_train.values[:, 1])

                                # y_train_predicted_values.append(test_model.predict(X_train.values)[:, 1])
                                train_samples = sample_generator(X_train, y_train)
                                for input_sample, target_sample in train_samples:
                                    predicted_val = test_model.predict(input_sample)
                                    y_train_predicted_values.append(predicted_val[:,:,1])

                                y_test_values.append(y_test.values[:, 1])

                                # y_test_predicted_values.append(test_model.predict(X_test.values)[:, 1])
                                # test the model
                                test_samples = sample_generator(X_test, y_test)
                                for input_sample, target_sample in test_samples:
                                    predicted_val = test_model.predict(input_sample)
                                    y_test_predicted_values.append(predicted_val[:,:,1])

                            #### END OF CONFIGURATION OPTION  ####
                            y_train_values = [item for sublist in y_train_values for item in sublist]
                            y_train_predicted_values = [item for sublist in y_train_predicted_values for item in sublist]
                            y_train_predicted_values = [item for sublist in y_train_predicted_values for item in sublist]

                            # remove the -1 values (the ones that are censored)
                            tmp = [i for i in zip(y_train_values, y_train_predicted_values) if int(i[0]) != -1]
                            y_train_values = [i[0] for i in tmp]
                            y_train_predicted_values = [i[1] for i in tmp]

                            y_test_values = [item for sublist in y_test_values for item in sublist]
                            y_test_predicted_values = [item for sublist in y_test_predicted_values for item in sublist]
                            y_test_predicted_values = [item for sublist in y_test_predicted_values for item in sublist]

                            current_train_res, current_test_res = calc_results_and_plot(y_train_values, y_train_predicted_values,
                                                                                        y_test_values,
                                                                                        y_test_predicted_values, algo_name='NeuralNetwork',
                                                                                        visualize=PLOT,
                                                                                        title=f'Epochs: {epochs}, Validation iterations: {number_iterations}',
                                                                                        show=False)

                            # print(current_train_res)
                            # print(current_test_res)
                            if record:
                                record_results(grid_search_dir,
                                               current_configuration_str,
                                               y_train_values,
                                               y_train_predicted_values,
                                               y_test_values,
                                               y_test_predicted_values,
                                               stats_of_input,
                                               current_train_res,
                                               current_test_res,
                                               hist.history)

                            train_res.update({current_configuration_str: current_train_res})
                            test_res.update({current_configuration_str: current_test_res})

    return train_res, test_res


def record_results(grid_search_dir,
                   current_configuration_str,
                   y_train_values,
                   y_train_predicted_values,
                   y_test_values,
                   y_test_predicted_values,
                   stats_of_input,
                   current_train_res,
                   current_test_res,
                   hist=None):

    config_dir = grid_search_dir + '/' + current_configuration_str
    if not os.path.exists(grid_search_dir):
        os.mkdir(grid_search_dir)
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    np.save(config_dir + '/y_train_values.npy', y_train_values)
    np.save(config_dir + '/y_train_predicted_values.npy', y_train_predicted_values)
    np.save(config_dir + '/y_test_values.npy', y_test_values)
    np.save(config_dir + '/y_test_predicted_values.npy', y_test_predicted_values)
    np.save(grid_search_dir + '/stats.npy', stats_of_input)
    if hist is not None:
        with open(config_dir + '/' + 'hist_res.p', 'wb') as f:
            pickle.dump(hist, f)
    with open(grid_search_dir + '/' + 'grid_search_results.txt', 'a') as f:
        f.writelines(['\n', current_configuration_str, '\n', 'Train\n ', str(current_train_res), '\nTest\n ',
                      str(current_test_res), '\n'])
    with open(config_dir + '/' + 'grid_search_results.txt', 'a') as f:
        f.writelines(['Train\n ', str(current_train_res), '\nTest\n ', str(current_test_res), '\n'])