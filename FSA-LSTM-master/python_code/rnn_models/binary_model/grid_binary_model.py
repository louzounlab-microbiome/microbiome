import sys
import os
import torch

sys.path.insert(0, os.path.join(".."))
sys.path.insert(0, os.path.join("..", ".."))
sys.path.insert(0, os.path.join("..", "..", "finit_state_machine"))
import ast
import os
from itertools import product
from time import strftime, gmtime
from binary_params import BinaryFSTParams, BinaryModuleParams, BinaryActivatorParams
from binary_rnn_activator import binaryActivator
from binary_rnn_model import BinaryModule
from fst_dataset import FstDataset
import numpy as np
import csv
RESULTS_DIR = "Grid_Results"
RAW_RESULTS_DIR = "Raw_results"
PARSED_RESULTS_DIR = "Parsed_results"
BINARY_RESULT_DIR = "Binary_model"


class GridSearch:
    def __init__(self, repeat=1):
        self._base_dir = os.path.join("..")
        self._build_result_dir()
        self._repeat = repeat

    def _build_result_dir(self):
        self._results_dir = os.path.join(self._base_dir, RESULTS_DIR)
        self._binary_model_results_dir = os.path.join(self._base_dir, RESULTS_DIR, BINARY_RESULT_DIR)
        self._raw_results_dir = os.path.join(self._binary_model_results_dir, RAW_RESULTS_DIR)
        self._parsed_results_dir = os.path.join(self._binary_model_results_dir, PARSED_RESULTS_DIR)

        if RESULTS_DIR not in os.listdir(self._base_dir):
            os.mkdir(self._results_dir)
        if BINARY_RESULT_DIR not in os.listdir(self._results_dir):
            os.mkdir(self._binary_model_results_dir)
        if RAW_RESULTS_DIR not in os.listdir(self._binary_model_results_dir):
            os.mkdir(self._raw_results_dir)
        if PARSED_RESULTS_DIR not in os.listdir(self._binary_model_results_dir):
            os.mkdir(self._parsed_results_dir)

    def _all_configurations(self):
        """
        set grid parameters here
        """
        size_alphabet_iter = list(range(5, 11))
        num_states_iter = list(range(15, 21))
        num_accept_states_iter = [2]
        epochs_iter = [200]
        out_lstm_dim_iter = [60]
        configurations = list(product(size_alphabet_iter, num_states_iter, num_accept_states_iter, epochs_iter,
                                      out_lstm_dim_iter))

        # prepare param objects
        for size_alphabet, num_states, num_accept_states, epochs, out_lstm in configurations:
            for _ in range(self._repeat):
                # str for configuration
                config_str = "|".join([str(size_alphabet), str(num_states), str(num_accept_states),
                                       str(epochs), str(out_lstm)])

                # dataset
                ds_params = BinaryFSTParams()
                ds_params.FST_ALPHABET_SIZE = size_alphabet
                ds_params.FST_STATES_SIZE = num_states
                ds_params.FST_ACCEPT_STATES_SIZE = num_accept_states
                dataset = FstDataset(ds_params)

                # model
                model_params = BinaryModuleParams(alphabet_size=len(dataset.chr_embed), lstm_out_dim=out_lstm)

                # activator
                activator_params = BinaryActivatorParams()
                activator_params.EPOCHS = epochs

                yield dataset, model_params, activator_params, config_str

    def _check_configuration(self, dataset: FstDataset, model_params: BinaryModuleParams,
                             activator_params: BinaryActivatorParams):

        model = BinaryModule(model_params)
        activator = binaryActivator(model, activator_params, dataset)
        activator.train(show_plot=False)
        del model

        return activator.accuracy_train_vec, activator.auc_train_vec, activator.loss_train_vec, \
               activator.accuracy_dev_vec, activator.auc_dev_vec, activator.loss_dev_vec

    def go(self, name=""):
        time = strftime("%m%d%H%M%S", gmtime())
        file_name = os.path.join(self._raw_results_dir, "binary_grid_" + time + "_" + name + ".txt")
        out_res = open(file_name, "wt")

        out_res.write("line0: config, " +
                      "line1: acc_train, line2: auc_train, line3: loss_train, " +
                      "line4: acc_dev, line5: auc_dev, line6: loss_dev\n")

        for dataset, model_params, activator_params, config_str in self._all_configurations():
            print(config_str)
            print(file_name)
            acc_train, auc_train, loss_train, acc_dev, auc_dev, loss_dev = \
                self._check_configuration(dataset, model_params, activator_params)
            out_res.write(config_str + "\n"
                          + str(acc_train) + "\n"
                          + str(auc_train) + "\n"
                          + str(loss_train) + "\n"
                          + str(acc_dev) + "\n"
                          + str(auc_dev) + "\n"
                          + str(loss_dev) + "\n")
            sys.stdout.flush()
            torch.cuda.empty_cache()

    def parse(self, res_file):
        file_name = sorted(os.listdir(self._raw_results_dir))[res_file] if type(res_file) is int else res_file
        res_file = open(os.path.join(self._raw_results_dir, file_name), "rt")
        res_file.readline()                 # skip header

        res_list = [["best_train_acc", "best_train_auc", "best_train_loss", "best_dev_acc", "best_dev_auc",
                     "best_dev_loss", 'size_alphabet', "num_states",  "num_accept_states", "epochs", "lstm_out_dim"]]

        while True:
            config = res_file.readline()
            acc_train = res_file.readline()
            auc_train = res_file.readline()
            loss_train = res_file.readline()
            acc_dev = res_file.readline()
            auc_dev = res_file.readline()
            loss_dev = res_file.readline()

            if not loss_dev:
                break

            config = config.replace("\n", "").split('|')

            best_result_idx = np.argmax(ast.literal_eval(auc_train))

            acc_train = ast.literal_eval(acc_train)[best_result_idx]
            auc_train = ast.literal_eval(auc_train)[best_result_idx]
            loss_train = ast.literal_eval(loss_train)[best_result_idx]
            acc_dev = ast.literal_eval(acc_dev)[best_result_idx]
            auc_dev = ast.literal_eval(auc_dev)[best_result_idx]
            loss_dev = ast.literal_eval(loss_dev)[best_result_idx]

            config_line = [str(acc_train)] + [str(auc_train)] + [str(loss_train)] + \
                          [str(acc_dev)] + [str(auc_dev)] + [str(loss_dev)] + config
            res_list.append(config_line)

        with open(os.path.join(self._parsed_results_dir, file_name.strip(".txt") + "_analyzed.csv"), "wt", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(res_list)


if __name__ == "__main__":
    name = "first_grid_alp_5_to_10__stat_15_to_20"
    if len(sys.argv) > 1:
        name = sys.argv[1]
    # GridSearch(repeat=3).go(name)
    GridSearch().parse("binary_grid_0820180055_first_grid_alp_5_to_10__stat_15_to_20.txt")
