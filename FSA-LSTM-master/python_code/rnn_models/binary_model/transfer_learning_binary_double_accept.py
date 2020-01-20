import ast
import sys
import os

from bokeh.models import Legend

sys.path.insert(0, os.path.join(".."))
sys.path.insert(0, os.path.join("..", ".."))
sys.path.insert(0, os.path.join("..", "..", "finit_state_machine"))

from copy import deepcopy
from time import strftime, gmtime
import os
from binary_params import BinaryFSTParams, BinaryModuleParams, BinaryActivatorParams
from binary_rnn_activator import binaryActivator
from binary_rnn_model import BinaryModule
from fst_double_accept_dataset import FstDoubleAcceptDataset
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot

NUM_STATES = 15
SIZE_ALPHABET = 5
# SUB_DATA_SIZES = (50, 100)
SUB_DATA_SIZES = (100, 200)
# MAIN_DATA_SIZE = 10000
MAIN_DATA_SIZE = 20000


def transfer_double_accept(main_data_set_siza=MAIN_DATA_SIZE, sub_data_sizes=SUB_DATA_SIZES,
                           num_states=NUM_STATES, size_alphabet=SIZE_ALPHABET):
    time = strftime("%m%d%H%M%S", gmtime())
    out_res = open(os.path.join("transfer_binary_double_accept_" + time + ".txt"), "wt")
    out_res.write("line0: config, line1: acc_train, line2: auc_train, line3: loss_train, line4: acc_dev, "
                  "line5: auc_dev, line6: loss_dev\n")

    ds_params = BinaryFSTParams()
    ds_params.DATASET_SIZE = main_data_set_siza
    ds_params.FST_ACCEPT_STATES_SIZE = 2
    ds_params.FST_STATES_SIZE = num_states
    ds_params.FST_ALPHABET_SIZE = size_alphabet
    ds_params.NEGATIVE_SAMPLES = True

    ds = FstDoubleAcceptDataset(ds_params)
    # data is for accept state one
    ds.mode_one()

    # train base model with 10000 samples
    base_model = BinaryModule(BinaryModuleParams(alphabet_size=ds_params.FST_ALPHABET_SIZE, lstm_out_dim=100))

    activator = binaryActivator(base_model, BinaryActivatorParams(), ds)
    activator.train(show_plot=False)
    out_res.write(str(ds_params.DATASET_SIZE) + ",base" + "\n"
                  + str(activator.accuracy_train_vec) + "\n" + str(activator.auc_train_vec) + "\n"
                  + str(activator.loss_train_vec) + "\n" + str(activator.accuracy_dev_vec) + "\n"
                  + str(activator.auc_dev_vec) + "\n" + str(activator.loss_dev_vec) + "\n")

    # data is for accept state two
    ds.mode_two()
    for data_size in sub_data_sizes:  # , 500, 1000, 10000]:
        ds.resize(data_size)

        # train without transfer
        solo_model = BinaryModule(BinaryModuleParams(alphabet_size=ds_params.FST_ALPHABET_SIZE, lstm_out_dim=100))
        activator = binaryActivator(solo_model, BinaryActivatorParams(), ds)
        activator.train(show_plot=False)
        out_res.write(str(data_size) + ",solo" + "\n"
                      + str(activator.accuracy_train_vec) + "\n" + str(activator.auc_train_vec) + "\n"
                      + str(activator.loss_train_vec) + "\n" + str(activator.accuracy_dev_vec) + "\n"
                      + str(activator.auc_dev_vec) + "\n" + str(activator.loss_dev_vec) + "\n")

        # train with transfer
        transfer_model = deepcopy(base_model)
        activator = binaryActivator(transfer_model, BinaryActivatorParams(), ds)
        activator.train(show_plot=False)
        out_res.write(str(data_size) + ",transfer" + "\n"
                      + str(activator.accuracy_train_vec) + "\n" + str(activator.auc_train_vec) + "\n"
                      + str(activator.loss_train_vec) + "\n" + str(activator.accuracy_dev_vec) + "\n"
                      + str(activator.auc_dev_vec) + "\n" + str(activator.loss_dev_vec) + "\n")
    out_res.close()


def plot_transfer(name):
    res_file = open(os.path.join(name), "rt")
    res_file.readline()                             # skip header

    data = {}
    while True:
        config = res_file.readline().strip().split(",")
        acc_train = res_file.readline()
        auc_train = res_file.readline()
        loss_train = res_file.readline()
        acc_dev = res_file.readline()
        auc_dev = res_file.readline()
        loss_dev = res_file.readline()

        if not loss_dev:
            break
        data_size, model_type = config

        if model_type not in data:
            data[model_type] = {}
        data[model_type][int(data_size)] = {
            "acc_train": ast.literal_eval(acc_train),
            "auc_train": ast.literal_eval(auc_train),
            "loss_train": ast.literal_eval(loss_train),
            "acc_dev": ast.literal_eval(acc_dev),
            "auc_dev": ast.literal_eval(auc_dev),
            "loss_dev": ast.literal_eval(loss_dev)
        }

        # data = { base: {one_size: vectors}, solo: {size: vectors}, transfer: {size: vector}
    col_plots = []
    for graph_type, vector_type in [("Acc Train", "acc_train"), ("AUC Train", "auc_train"), ("Loss Train", "loss_train"),
                                    ("Acc Dev", "acc_dev"), ("AUC Dev", "auc_dev"), ("Loss Dev", "loss_dev")]:
        p = figure(plot_width=600, plot_height=250,
                   title=graph_type + " , base: " + str(MAIN_DATA_SIZE) + "samples, alphabet=" + str(SIZE_ALPHABET) +
                         ", states=" + str(NUM_STATES) + ", train_split=0.5",
                   x_axis_label="epochs", y_axis_label=graph_type)
        color_base, color_transfer_50, color_transfer_100, color_solo_50, color_solo_100 = \
            ("black", "green", "blue", "orange", "red")

        y_axis_base = data["base"][MAIN_DATA_SIZE][vector_type]
        y_axis_tran_50 = data["transfer"][SUB_DATA_SIZES[0]][vector_type]
        y_axis_tran_100 = data["transfer"][SUB_DATA_SIZES[1]][vector_type]
        y_axis_solo_50 = data["solo"][SUB_DATA_SIZES[0]][vector_type]
        y_axis_solo_100 = data["solo"][SUB_DATA_SIZES[1]][vector_type]
        x_axis = list(range(len(y_axis_solo_100)))

        p.line(x_axis, y_axis_base, line_color=color_base)
        p.line(x_axis, y_axis_tran_50, line_color=color_transfer_50)
        p.line(x_axis, y_axis_tran_100, line_color=color_transfer_100)
        p.line(x_axis, y_axis_solo_50, line_color=color_solo_50)
        p.line(x_axis, y_axis_solo_100, line_color=color_solo_100)
        col_plots.append([p])

    p = figure(plot_width=600, plot_height=200, title=graph_type + "Legend",
               x_axis_label="", y_axis_label="")
    p.line([0], [0], line_color=color_base, legend="base_data")
    p.line([0], [0], line_color=color_transfer_50, legend="sub_data_transfer_50_samples")
    p.line([0], [0], line_color=color_transfer_100, legend="sub_data_transfer_100_samples")
    p.line([0], [0], line_color=color_solo_50, legend="sub_data_solo_50_samples")
    p.line([0], [0], line_color=color_solo_100, legend="sub_data_solo_100_samples")

    show(gridplot([[p]] + col_plots))


if __name__ == "__main__":
    # transfer_double_accept()
    # plot_transfer("transfer_binary_double_accept_0601123659.txt")
    plot_transfer("transfer_binary_double_accept_0601180228.txt")
