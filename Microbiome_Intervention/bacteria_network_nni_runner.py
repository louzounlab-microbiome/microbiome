import copy
import os
import sys
from datetime import datetime
from random import shuffle

import pandas as pd
import numpy as np
import nni
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import Conv1d, Linear
from torch.nn.functional import leaky_relu

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LearningMethods.intervention_nn import run_NN, SequenceModule
from LearningMethods.intervention_rnn import run_RNN
from Microbiome_Intervention.Create_learning_data_from_data_set import get_adapted_X_y_for_wanted_learning_task

# the change in value of the bacteria
CHANGE = 0.5


class SequenceModuleClf(nn.Module):

    def __init__(self, params, timesteps=10):
        """
        Create model structure replica in order to load the trained model weights and bias from file
        :param params: model dimensions - have to same as the loaded model dimensions for it to work
        :param timesteps: irrelevant
        """
        super(SequenceModuleClf, self).__init__()
        self.fc1 = Linear(params["NN_input_dim"], params["NN_hidden_dim_1"])
        self.fc2 = Linear(params["NN_hidden_dim_1"], params["NN_output_dim"])
        self.conv = Conv1d(2, timesteps, 1)

    def forward(self, x):
        x = leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


class bacteria_network_clf(nn.Module):
    def __init__(self, params):
        """
        Create model to be loaded from file
        :param params: params for the SequenceModuleClf
        """
        super(bacteria_network_clf, self).__init__()
        self._sequence_nn = SequenceModuleClf(params)

    def forward(self, x):
        x = leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def predict(self, bacteria):
        outputs = self._sequence_nn(bacteria)
        return outputs


def run_nn_bacteria_network(tax, data_set_name):
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

    The change will be by the constant 'CHANGE', you can alter it

    :param tax: (string) main dataset folder "DATASET/tax=x"
    :return: doesn't return an object, create a csv file with all the results
    csv name = "interaction_network_" + params.__str__().replace(" ", "").replace("'", "") + "_df.csv"
    To determine which bacteria have interactions and create a visual graph use "built_network_form_file.py"
    sent the results csv path
    """

    # sub folder path for network results,
    folder = os.path.join(tax, "interaction_network")
    if not os.path.exists(folder):
        os.mkdir(folder)
    # ------------------------------------ decide on mission ------------------------------------
    nni_ = False  # check model or run nni for real
    GPU = True if nni_ else False
    report_loss = True
    report_correlation = not report_loss
    single_bacteria = True  # learn the change in single bacteria or all in once
    k_fold = False  # run k fold
    p_value = 0.001
    test_size = 0.3

    if nni_:
        params = nni.get_next_parameter()
    else:
        params = {"STRUCTURE": "001L200H",
                  "TRAIN_TEST_SPLIT": 0.7,
                  "EPOCHS": 70,
                  "LEARNING_RATE": 1e-3,
                  "OPTIMIZER": "Adam",
                  "REGULARIZATION": 0.01,
                  "DROPOUT": 0.1}

    with open(os.path.join(tax, "bacteria.txt"), "r") as b_file:
        bacteria = b_file.readlines()
        bacteria = [b.rstrip() for b in bacteria]

    bacteria_number_list = range(len(bacteria))
    # ------------------------------------ data loading ------------------------------------
    # run a prediction of a single bacteria at a time
    # consider the average loss and correlation of all runs as the performance measurement
    df_title = os.path.join(folder, "interaction_network_" + params.__str__().replace(" ", "").replace("'", "") + "_df.csv")

    df = pd.DataFrame(columns=["BACTERIA", "CHANGED_BACTERIA", "CHANGE", "Y"])
    df.to_csv(df_title, index=False)

    # create a df that saves a binary value 1/0 => interaction/no interaction according to the train set
    train_binary_significant_df = pd.DataFrame(columns=bacteria)
    # create a df that saves the continuous b value of each bacteria according to the test set
    test_b_df = pd.DataFrame(columns=bacteria)

    for b_i, bacteria_num in enumerate(bacteria_number_list):  # for each bacteria
        df = pd.read_csv(df_title)
        path = "X_y_for_bacteria_number_" + str(b_i) + ".csv"
        X_trains, X_tests, y_trains, y_tests, name = \
            get_adapted_X_y_for_wanted_learning_task(tax, path, "regular", k_fold=1)
        X_train, X_test, y_train, y_test = X_trains[0], X_tests[0], y_trains[0], y_tests[0]
        NUMBER_OF_SAMPLES = X_train.shape[0]
        NUMBER_OF_BACTERIA = X_train.shape[1]
        NUMBER_OF_TIME_POINTS = None
        missing_values = np.array(
            [1 for j in range(NUMBER_OF_SAMPLES)])

        # split to train and test
        """
        split_list = [1 - test_size, test_size]
        split_list = np.multiply(np.cumsum(split_list), len(X)).astype("int").tolist()

        # list of shuffled indices to sample randomly
        shuffled_idx = []
        shuffle(person_indexes)
        for arr in person_indexes:
            for val in arr:
                shuffled_idx.append(val)

        # split the data itself
        X_train = X[shuffled_idx[:split_list[0]]]
        y_train = y[shuffled_idx[:split_list[0]]]

        X_test = X[shuffled_idx[split_list[0]:split_list[1]]]
        y_test = y[shuffled_idx[split_list[0]:split_list[1]]]
        """

        train_binary_significant_for_b_i = []
        test_1_u_score_for_b_i = []
        """
        path = "time_serie_X_y_for_bacteria_number_" + str(bacteria_num) + ".csv"
        X, y, missing_values, name = get_adapted_X_y_for_wanted_learning_task(tax, path, "time_serie")
        NUMBER_OF_SAMPLES = X.shape[0]
        NUMBER_OF_TIME_POINTS = X.shape[1]
        NUMBER_OF_BACTERIA = X.shape[2]

        flat_time_points_values_num = NUMBER_OF_SAMPLES * NUMBER_OF_TIME_POINTS

        X = X.reshape(flat_time_points_values_num, NUMBER_OF_BACTERIA)
        y = y.reshape(flat_time_points_values_num)
        missing_values = missing_values.reshape(flat_time_points_values_num)

        person_indexes = np.linspace(0, flat_time_points_values_num - 1, flat_time_points_values_num). \
            reshape(NUMBER_OF_SAMPLES, NUMBER_OF_TIME_POINTS).astype(int).tolist()
        """
        # TRAIN
        # run the model one time with no change, then save it
        res_map = run_NN(X_train, y_train, missing_values, params, name, folder,
                         NUMBER_OF_SAMPLES, NUMBER_OF_TIME_POINTS, NUMBER_OF_BACTERIA,
                         save_model=True, GPU_flag=GPU, k_fold=k_fold,
                         task_id="base_" + str(b_i) + "_model",
                         person_indexes=None)

        model_path = os.path.join(folder, "trained_models", params.__str__().replace(" ", "").replace("'", "") + "_base_" + str(b_i) + "_model_model")
        out_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]  # else NUMBER_OF_BACTERIA
        structure = params["STRUCTURE"]
        layer_num = int(structure[0:3])
        hid_dim_1 = int(structure[4:7])
        hid_dim_2 = int(structure[8:11]) if len(structure) > 10 else None

        clf_params = {"NN_input_dim": X_train.shape[1], "NN_hidden_dim_1": hid_dim_1, "NN_output_dim": out_dim}

        clf = bacteria_network_clf(clf_params)
        clf.load(model_path)
        y_pred_no_change = clf.predict(torch.FloatTensor(X_train))
        y_str = ""
        for val in y_pred_no_change:
            y_str += str(val.detach().numpy()[0]) + " "
        df.loc[len(df)] = [int(bacteria_num), int(-1), "no change", y_str]

        # ------------------------------------ send to network ------------------------------------
        for bacteria_to_change_num in bacteria_number_list:  # change each bacteria

            # change X, y for only bacteria_to_change_num
            X_positive_change = copy.deepcopy(X_train)
            for s_i, sample in enumerate(X_positive_change):  # 0.9459053900000001 -0.05409460999999999
                X_positive_change[s_i][bacteria_to_change_num] += CHANGE

            y_pred_pos_change = clf.predict(torch.FloatTensor(X_positive_change))
            y_str = ""
            for val in y_pred_pos_change:
                y_str += str(val.detach().numpy()[0]) + " "

            df.loc[len(df)] = [int(bacteria_num), int(bacteria_to_change_num), "plus " + str(CHANGE), y_str]

            X_negative_change = copy.deepcopy(X_train)
            for s_i, sample in enumerate(X_negative_change):
                X_negative_change[s_i][bacteria_to_change_num] -= CHANGE

            y_pred_neg_change = clf.predict(torch.FloatTensor(X_negative_change))
            y_str = ""
            for val in y_pred_neg_change:
                y_str += str(val.detach().numpy()[0]) + " "

            df.loc[len(df)] = [int(bacteria_num), int(bacteria_to_change_num), "minus " + str(CHANGE), y_str]

            pos_u, pos_u_test_p_val = mannwhitneyu(y_pred_no_change.detach().numpy(), y_pred_pos_change.detach().numpy())
            neg_u, neg_u_test_p_val = mannwhitneyu(y_pred_no_change.detach().numpy(), y_pred_neg_change.detach().numpy())

            if pos_u_test_p_val < p_value and neg_u_test_p_val < p_value:
                train_binary_significant_for_b_i.append(1)
            else:
                train_binary_significant_for_b_i.append(0)

        # TEST
        # run the model one time with no change, then save it
        NUMBER_OF_SAMPLES = X_test.shape[0]
        NUMBER_OF_BACTERIA = X_test.shape[1]
        NUMBER_OF_TIME_POINTS = None
        missing_values = np.array(
            [1 for j in range(NUMBER_OF_SAMPLES)])
        res_map = run_NN(X_test, y_test, missing_values, params, name, folder,
                         NUMBER_OF_SAMPLES, NUMBER_OF_TIME_POINTS, NUMBER_OF_BACTERIA,
                         save_model=True, GPU_flag=GPU, k_fold=k_fold,
                         task_id="base_" + str(b_i) + "_model",
                         person_indexes=None)

        model_path = os.path.join(folder, "trained_models", params.__str__().replace(" ", "").replace("'", "") + "_base_" + str(
            b_i) + "_model_model")
        out_dim = 1 if len(y_test.shape) == 1 else y_test.shape[1]  # else NUMBER_OF_BACTERIA
        structure = params["STRUCTURE"]
        layer_num = int(structure[0:3])
        hid_dim_1 = int(structure[4:7])
        hid_dim_2 = int(structure[8:11]) if len(structure) > 10 else None

        clf_params = {"NN_input_dim": X_test.shape[1], "NN_hidden_dim_1": hid_dim_1, "NN_output_dim": out_dim}

        clf = bacteria_network_clf(clf_params)
        clf.load(model_path)
        y_pred_no_change = clf.predict(torch.FloatTensor(X_test))
        # ------------------------------------ send to network ------------------------------------
        for bacteria_to_change_num in bacteria_number_list:  # change each bacteria

            # change X, y for only bacteria_to_change_num
            X_positive_change = copy.deepcopy(X_test)
            for s_i, sample in enumerate(X_positive_change):
                X_positive_change[s_i][bacteria_to_change_num] += CHANGE

            y_pred_pos_change = clf.predict(torch.FloatTensor(X_positive_change))

            X_negative_change = copy.deepcopy(X_test)
            for s_i, sample in enumerate(X_negative_change):
                X_negative_change[s_i][bacteria_to_change_num] -= CHANGE

            y_pred_neg_change = clf.predict(torch.FloatTensor(X_negative_change))

            pos_u, pos_u_test_p_val = mannwhitneyu(y_pred_no_change.detach().numpy(), y_pred_pos_change.detach().numpy())
            neg_u, neg_u_test_p_val = mannwhitneyu(y_pred_no_change.detach().numpy(), y_pred_neg_change.detach().numpy())

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
        Networks_AUC_df.loc[len(Networks_AUC_df)] = ["positive change", "neural network " + params.__str__(),
                                                     data_set_name, test_size,k_fold,
                                                     pos_auc, datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S")]
        Networks_AUC_df.loc[len(Networks_AUC_df)] = ["negative change", "neural network " + params.__str__(),
                                                     data_set_name, test_size, k_fold,
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


if __name__ == "__main__":
    run_simulation_data = False
    run_gdm = True
    run_vitaminA = True

    if run_simulation_data:
        tax = os.path.join("Simulations", "3")
        run_nn_bacteria_network(tax, "simualation 1")

    if run_gdm:
        tax = os.path.join('GDM', 'tax=5')
        run_nn_bacteria_network(tax, "GDM")

    if run_vitaminA:
        tax = os.path.join('VitamineA', 'tax=5')
        run_nn_bacteria_network(tax, "VitamineA")
