import os
from collections import Counter
from random import shuffle
from sys import stdout
import pandas as pd
import numpy as np
import nni
import torch
from bokeh.models import Title
from matplotlib.pyplot import figure
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import Module, Linear, Conv1d
from torch.nn.functional import mse_loss, relu, sigmoid, leaky_relu

from bokeh.io import output_file, save, export_png
from bokeh.plotting import figure, show
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from LearningMethods.model_params import NNModuleParams, NNActivatorParams, MicrobiomeDataset, split_microbiome_dataset

TRAIN_JOB = "TRAIN"
DEV_JOB = "DEV"
VALIDATE_JOB = "VALIDATE"
LOSS_PLOT = "loss"
CORR_PLOT = "correlation"
ACCURACY_PLOT = "accuracy"
R2_PLOT = "R^2"
CALC_CORR = True
CALC_R2 = True
CALC_ACC = False
AUC_PLOT = "ROC-AUC"
CONV_LAYER = False
SHUFFLE = True
DIM = 1
SAVE_RUN_RESULTS = True

NUMBER_OF_BACTERIA = 0
NUMBER_OF_TIME_POINTS = 0
NUMBER_OF_SAMPLES = 0

PRINT_PROGRESS = False
PRINT_INFO = False


# ----------------------------------------------- models -----------------------------------------------

class MicrobiomeModule(Module):
    def __init__(self, params: NNModuleParams):
        super(MicrobiomeModule, self).__init__()
        # useful info in forward function
        self.params = params
        self._sequence_nn = SequenceModule(params.SEQUENCE_PARAMS)
        self.optimizer = self.set_optimizer(params.LEARNING_RATE, params.OPTIMIZER, params.REGULARIZATION)

    def set_optimizer(self, lr, opt, l2_reg):
        return opt(self.parameters(), lr=lr, weight_decay=l2_reg)

    def forward(self, x):
        """
        print(self._sequence_nn)
        print("nn dim:")
        print(x.shape)
        """
        x = self._sequence_nn(x)
        return x.float()


class SequenceModule(Module):
    def __init__(self, params):
        super(SequenceModule, self).__init__()
        self._layer_num = params.NN_layers
        self.conv = Conv1d(2, params.timesteps, 1)
        if self._layer_num == 1:
            # HOW TO MAKE params.NN_layers a usable param?
            self.fc1 = Linear(params.NN_input_dim, params.NN_hidden_dim_1)
            self.fc2 = Linear(params.NN_hidden_dim_1, params.NN_output_dim)

        elif self._layer_num == 2:
            # HOW TO MAKE params.NN_layers a usable param?
            self.fc1 = Linear(params.NN_input_dim, params.NN_hidden_dim_1)
            self.fc2 = Linear(params.NN_hidden_dim_1, params.NN_hidden_dim_2)
            self.fc3 = Linear(params.NN_hidden_dim_2, params.NN_output_dim)

    def forward(self, x):
        # 3 layers NN
        if CONV_LAYER:
            x = self.conv(x)
        if self._layer_num == 1:
            x = leaky_relu(self.fc1(x))
            x = self.fc2(x)
        elif self._layer_num == 2:
            x = leaky_relu(self.fc1(x))
            x = leaky_relu(self.fc2(x))
            x = self.fc3(x)

        return x

# ----------------------------------------------- activate model -----------------------------------------------
class Activator:
    def __init__(self, model: MicrobiomeModule, params: NNActivatorParams, data: MicrobiomeDataset, splitter, person_indexes):
        self._model = model.cuda() if params.GPU else model
        self._gpu = params.GPU
        self._epochs = int(params.EPOCHS)
        self._batch_size = params.BATCH_SIZE
        self._loss_func = params.LOSS
        self._corr_func = params.CORR
        self._r2_func = params.R2
        self._early_stop = params.EARLY_STOP
        self.shuffle = model.params.SHUFFLE
        self._load_data(data, params.TRAIN_TEST_SPLIT, params.BATCH_SIZE, splitter, person_indexes)
        self._init_loss_and_acc_vec()
        self._init_print_att()
        self._dim = model.params.DIM

    # init loss and accuracy vectors (as function of epochs)
    def _init_loss_and_acc_vec(self):
        self._loss_vec_train = []
        self._loss_vec_dev = []
        self._corr_vec_train = []
        self._corr_vec_dev = []
        self._r2_vec_train = []
        self._r2_vec_dev = []
        self._r2_vec_for_each_target_value_train = []
        self._r2_vec_for_each_target_value_dev = []
        self._accuracy_vec_train = []
        self._accuracy_vec_dev = []
        self._auc_vec_train = []
        self._auc_vec_dev = []
        self._train_label_and_output = None
        self._test_label_and_output = None


    # init variables that holds the last update for loss and accuracy
    def _init_print_att(self):
        self._print_train_loss = 0
        self._print_dev_loss = 0
        self._print_train_corr = 0
        self._print_dev_corr = 0
        self._print_train_r2 = 0
        self._print_dev_r2 = 0
        self._print_train_accuracy = 0
        self._print_train_auc = 0
        self._print_dev_accuracy = 0
        self._print_dev_auc = 0

    # update loss after validating
    def _update_loss(self, loss, job=TRAIN_JOB):
        if job == TRAIN_JOB:
            self._loss_vec_train.append(loss)
            self._print_train_loss = loss
        elif job == DEV_JOB:
            self._loss_vec_dev.append(loss)
            self._print_dev_loss = loss

    # update correlation after validating
    def _update_corr(self, corr, job=TRAIN_JOB):
        if job == TRAIN_JOB:
            self._corr_vec_train.append(corr)
            self._print_train_corr = corr
        elif job == DEV_JOB:
            self._corr_vec_dev.append(corr)
            self._print_dev_corr = corr

    # update r2 after validating
    def _update_r2(self, r2, job=TRAIN_JOB):
        if job == TRAIN_JOB:
            self._r2_vec_train.append(r2)
            self._print_train_r2 = r2
        elif job == DEV_JOB:
            self._r2_vec_dev.append(r2)
            self._print_dev_r2 = r2

    # update r2 after validating
    def _update_r2_for_each_target_value(self, r2, job=TRAIN_JOB):
        if job == TRAIN_JOB:
            self._r2_vec_for_each_target_value_train.append(r2)
        elif job == DEV_JOB:
            self._r2_vec_for_each_target_value_dev.append(r2)

    # update accuracy after validating
    def _update_accuracy(self, pred, true, job=TRAIN_JOB):
        # calculate acc
        acc = sum([1 if round(i) == int(j) else 0 for i, j in zip(pred, true)]) / len(pred)
        if job == TRAIN_JOB:
            self._print_train_accuracy = acc
            self._accuracy_vec_train.append(acc)
            return acc
        elif job == DEV_JOB:
            self._print_dev_accuracy = acc
            self._accuracy_vec_dev.append(acc)
            return acc

    # update auc after validating
    def _update_auc(self, pred, true, job=TRAIN_JOB):
        # pred_ = [-1 if np.isnan(x) else x for x in pred]
        # if there is only one class in the batch
        num_classes = len(Counter(true))
        if num_classes < 2:
            auc = 0.5
        # calculate auc
        else:
            auc = roc_auc_score(true, pred)
        if job == TRAIN_JOB:
            self._print_train_auc = auc
            self._auc_vec_train.append(auc)
            return auc
        elif job == DEV_JOB:
            self._print_dev_auc = auc
            self._auc_vec_dev.append(auc)
            return auc

    # print progress of a single epoch as a percentage
    def _print_progress(self, batch_index, len_data, job=""):
        if PRINT_PROGRESS:
            prog = int(100 * (batch_index + 1) / len_data)
            stdout.write("\r\r\r\r\r\r\r\r" + job + " %d" % prog + "%")
            print("", end="\n" if prog == 100 else "")
            stdout.flush()

    # print last loss and accuracy
    def _print_info(self, jobs=()):
        if PRINT_INFO:
            if TRAIN_JOB in jobs:
                print("Loss_Train: " + '{:{width}.{prec}f}'.format(self._print_train_loss, width=6, prec=4),
                      end=" || ")
                if CALC_CORR:
                    print("Corr_Train: " + '{:{width}.{prec}f}'.format(self._print_train_corr, width=6, prec=4),
                      end=" || ")
                if CALC_R2:
                    print("R2_Train: " + '{:{width}.{prec}f}'.format(self._print_train_r2, width=6, prec=4),
                      end=" || ")
                if CALC_ACC:
                    print("Acc_Train: " + '{:{width}.{prec}f}'.format(self._print_train_accuracy, width=6, prec=4) +
                      " || AUC_Train: " + '{:{width}.{prec}f}'.format(self._print_train_auc, width=6, prec=4) +
                      " || ")

            if DEV_JOB in jobs:
                print("Loss_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_loss, width=6, prec=4),
                      end=" || ")
                if CALC_CORR:
                    print("Corr_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_corr, width=6, prec=4),
                      end=" || ")
                if CALC_R2:
                    print("R2_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_r2, width=6, prec=4),
                      end=" || ")
                if CALC_ACC:
                    print("Acc_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_accuracy, width=6, prec=4) +
                      " || AUC_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_auc, width=6, prec=4) +
                      " || ")
            print("")

    # plot loss / accuracy graph
    def plot_line(self, title, save_name, job=LOSS_PLOT):



        """
        t1, t2 = title.split("\n")
        p = figure(plot_width=600, plot_height=250,  # , title=title
                   x_axis_label="epochs", y_axis_label=job)
        p.add_layout(Title(text=t2), 'above')
        p.add_layout(Title(text=t1), 'above')

        color1, color2 = ("orange", "red") if job == LOSS_PLOT else ("green", "blue")
        x_axis = list(range(len(y_axis_dev)))
        p.line(x_axis, y_axis_train, line_color=color1, legend_label="train")
        p.line(x_axis, y_axis_dev, line_color=color2, legend_label="dev")
        output_file(save_name + " " + job + "_fig.html")
        #save(p)
        export_png(p, filename=save_name + " " + job + "_png.png")
        # show(p)
        """


        if job == LOSS_PLOT:
            y_axis_train = self._loss_vec_train  # if job == LOSS_PLOT else self._accuracy_vec_train
            y_axis_dev = self._loss_vec_dev  # if job == LOSS_PLOT else self._accuracy_vec_dev
            c1 = "red"
            c2 = "blue"
        elif job == CORR_PLOT:
            y_axis_train = self._corr_vec_train
            y_axis_dev = self._corr_vec_dev
            c1 = "c"
            c2 = "m"
        elif job == R2_PLOT:
            y_axis_train = self._r2_vec_train
            y_axis_dev = self._r2_vec_dev
            c1 = "green"
            c2 = "orange"

        elif job == ACCURACY_PLOT:
            y_axis_train = self._accuracy_vec_train
            y_axis_dev = self._accuracy_vec_dev
            c1 = "brown"
            c2 = "pink"

        elif job == AUC_PLOT:
            y_axis_train = self._auc_vec_train
            y_axis_dev = self._auc_vec_dev
            c1 = "gray"
            c2 = "olive"

        fig, ax = plt.subplots()
        plt.title(title)
        y = np.linspace(0, len(y_axis_train), len(y_axis_train))
        plt.plot(y, y_axis_train, label="train", color=c1)
        plt.plot(y, y_axis_dev, label="test", color=c2)
        plt.xlabel("epochs")
        plt.xticks(np.arange(50, len(y_axis_train) + 1, step=50))
        plt.ylabel(job)
        plt.legend()
        plt.savefig(save_name + " " + job + ".png")
        # plt.show()

    def _plot_acc_dev(self):
        self.plot_line(title="loss", save_name="plot", job=LOSS_PLOT)
        self.plot_line(title="R2", save_name="plot", job=R2_PLOT)
        self.plot_line(title="accurecy", save_name="plot", job=ACCURACY_PLOT)
        self.plot_line(title="auc", save_name="plot", job=AUC_PLOT)

    @property
    def model(self):
        return self._model

    @property
    def loss_train_vec(self):
        return self._loss_vec_train

    @property
    def loss_dev_vec(self):
        return self._loss_vec_dev

    @property
    def corr_train_vec(self):
        return self._corr_vec_train

    @property
    def corr_dev_vec(self):
        return self._corr_vec_dev

    @property
    def r2_train_vec(self):
        return self._r2_vec_train

    @property
    def r2_dev_vec(self):
        return self._r2_vec_dev

    @property
    def r2_for_each_target_value_vec_train(self):
        return self._r2_vec_for_each_target_value_train

    @property
    def r2_for_each_target_value_dev_vec(self):
        return self._r2_vec_for_each_target_value_dev

    @property
    def accuracy_train_vec(self):
        return self._accuracy_vec_train

    @property
    def auc_train_vec(self):
        return self._auc_vec_train


    @property
    def accuracy_dev_vec(self):
        return self._accuracy_vec_dev

    @property
    def auc_dev_vec(self):
        return self._auc_vec_dev


    # load dataset
    def _load_data(self, train_dataset, train_split, batch_size, splitter, person_indexes):
        # split dataset
        train, dev = splitter(train_dataset, [train_split, 1-train_split], person_indexes)
        # set train loader
        self._train_loader = DataLoader(
            train,
            batch_size=batch_size,
            # collate_fn=train.collate_fn,
            shuffle=SHUFFLE,
            pin_memory=True,
            num_workers=8
        )

        self._train_valid_loader = DataLoader(
            train,
            batch_size=batch_size,
            # collate_fn=train.collate_fn,
            shuffle=SHUFFLE,
            pin_memory=True,
            num_workers=8
        )

        # set validation loader
        self._dev_loader = DataLoader(
            dev,
            batch_size=batch_size,
            # collate_fn=dev.collate_fn,
            shuffle=SHUFFLE,
            pin_memory=True,
            num_workers=8
        )

    def _to_gpu(self, x, l, m):
        x = x.cuda() if self._gpu else x
        l = l.cuda() if self._gpu else l
        m = m.cuda() if self._gpu else m
        return x, l, m

    # train a model, input is the enum of the model type
    def train(self, show_plot=False, apply_nni=False, validate_rate=10):
        self._init_loss_and_acc_vec()
        # calc number of iteration in current epoch
        len_data = len(self._train_loader)
        last_epoch = list(range(self._epochs))[-1]
        for epoch_num in range(self._epochs):
            # calc number of iteration in current epoch
            for batch_index, (sequence, label, missing_values) in enumerate(self._train_loader):
                sequence, label, missing_values = self._to_gpu(sequence, label, missing_values)
                # print progress
                self._model.train()
                output = self._model(sequence.float())
                """
                print("label:")
                print(label.shape)
                print("seq:")
                print(sequence.shape)
                print("output:")
                print(output.shape)
                print(output.squeeze(dim=2).shape)
                print(label.float().squeeze(dim=1).shape)
                """
                loss = self._loss_func(output.squeeze(dim=self._dim).float(), label.float(), missing_values)  # calculate loss
                #print("\n" + str(loss))
                loss.backward()                                 # back propagation
                self._model.optimizer.step()                    # update weights
                self._model.zero_grad()                         # zero gradients

                if PRINT_PROGRESS:
                    self._print_progress(batch_index, len_data, job=TRAIN_JOB)

                # self._train_label_and_output = (label, output)
            # validate and print progress

            # /----------------------  FOR NNI  -------------------------
            if epoch_num % validate_rate == 0:
                # validate on dev set anyway
                save_true_and_pred = True
                self._validate(self._train_loader, save_true_and_pred, job=TRAIN_JOB)
                self._validate(self._dev_loader, save_true_and_pred, job=DEV_JOB)
                torch.cuda.empty_cache()
                # report dev result as am intermediate result
                if apply_nni:
                    test_loss = self._print_dev_loss
                    nni.report_intermediate_result(test_loss)
                # validate on train set as well and display results
                else:
                    torch.cuda.empty_cache()
                    self._print_info(jobs=[TRAIN_JOB, DEV_JOB])

            if self._early_stop and epoch_num > 30 and self._print_dev_loss > np.max(self._loss_vec_dev[-30:]):
                break

        # report final results
        if apply_nni:
            test_loss = np.max(self._print_dev_loss)
            nni.report_final_result(test_loss)

        if show_plot:
            self._plot_acc_dev()

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader, save_true_and_pred, job=""):
        # for calculating total loss and accuracy
        loss_count = 0
        corr_count = 0
        r2_count = 0
        r2_count_list = []
        true_labels = []
        pred = []

        self._model.eval()
        # calc number of iteration in current epoch
        len_data = len(data_loader)
        for batch_index, (sequence, label, missing_values) in enumerate(data_loader):
            sequence, label, missing_values = self._to_gpu(sequence, label, missing_values)
            # print progress
            if PRINT_PROGRESS:
                self._print_progress(batch_index, len_data, job=VALIDATE_JOB)
            output = self._model(sequence.float())
            # calculate total loss
            loss_count += self._loss_func(output.squeeze(dim=self._dim).float(), label.float(), missing_values)  # calculate loss
            if CALC_CORR:
                corr_count += self._corr_func(output.squeeze(dim=self._dim).float(), label.float(), missing_values)  # calculate corr
            if CALC_R2:
                if len(label.shape) == 1:
                    output = output.squeeze(dim=self._dim).float()
                    r2_count += self._r2_func(output.float(), label.float(), missing_values)  # calculate r^2
                elif len(label.shape) == 2:
                    r2_count_list.append(self._r2_func(output.float(), label.float(), missing_values))  # calculate r^2


            true_labels += label.tolist()
            pred += output.squeeze().tolist()

            if job == TRAIN_JOB:
                self._train_label_and_output = (label, output)

            elif job == DEV_JOB:
                self._test_label_and_output = (label, output)

        # update loss accuracy
        loss = float(loss_count / len(data_loader))
        self._update_loss(loss, job=job)

        if CALC_CORR:
            corr = float(corr_count / len(data_loader))
            self._update_corr(corr, job=job)

        if CALC_R2:
            if r2_count != 0:
                # print(r2_count)
                r2 = float(r2_count / len(data_loader))
                self._update_r2(r2, job=job)
            else:
                # print(r2_count_list)
                r2_avarage_count = np.mean((np.array(r2_count_list)), axis=0)
                self._update_r2_for_each_target_value(r2_avarage_count, job=job)
                r2 = float(np.mean(r2_avarage_count) / len(data_loader))
                # print(np.mean(r2_avarage_count))
                self._update_r2(r2, job=job)



        """
        self._update_accuracy(pred, true_labels, job=job)
        self._update_auc(pred, true_labels, job=job)
        """
        return loss

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)

    def scatter_results(self, fig_title, folder_and_name):
        train_l, train_o = self._train_label_and_output
        test_l, test_o = self._test_label_and_output
        train_l, train_o = train_l.flatten().detach().numpy(), train_o.flatten().detach().numpy()
        test_l, test_o = test_l.flatten().detach().numpy(), test_o.flatten().detach().numpy()
        # min_val = min(min(np.min(train_l), np.min(train_o)), min(np.min(test_l), np.min(test_o)))
        # max_val = max(max(np.max(train_l), np.max(train_o)), max(np.max(test_l), np.max(test_o)))

        #min_val_per = min(min(np.percentile(train_l, 0.5), np.percentile(train_o, 0.5)), min(np.percentile(test_l, 0.5), np.percentile(test_o, 0.5)))
        min_val = min(min(train_l.min(), train_o.min()), min(test_l.min(), test_o.min()))
        #max_val_per = max(max(np.percentile(train_l, 99.5), np.percentile(train_o, 99.5)), max(np.percentile(test_l, 99.5), np.percentile(test_o, 99.5)))
        max_val = max(max(train_l.max(), train_o.max()), max(test_l.max(), test_o.max()))
        limit = max(abs(min_val), abs(max_val))

        fig, ax = plt.subplots()
        plt.ylim((-limit, limit))
        plt.xlim((-limit, limit))
        x = np.linspace(-limit, limit, 1000)
        plt.plot(x, x + 0, '--k', alpha=0.5)
        plt.scatter(train_l, train_o,
                    label='Train', color='red', s=3, alpha=0.3)
        plt.scatter(test_l, test_o,
                    label='Test', color='blue', s=3, alpha=0.3)
        plt.title(fig_title, fontsize=10)
        plt.xlabel('Real values')
        plt.ylabel('Predicted Values')
        plt.legend(loc='upper left')
        print(folder_and_name)
        plt.savefig(folder_and_name)
        # plt.show()

    def scatter_results_by_time_point(self, fig_title, folder_and_name):
        """
        a = np.array(
            [[[1, 2, 3, 4], [5, 6, 7, 8]],
             [[9, 10, 11, 12], [13, 14, 15, 16]],
             [[17, 18, 19, 20], [21, 22, 23, 24]]])
        t = np.split(a, indices_or_sections=2, axis=1)
        b = np.split(a, indices_or_sections=4, axis=2)
        """
        if self._train_label_and_output and self._test_label_and_output:
            train_l, train_o = self._train_label_and_output
            test_l, test_o = self._test_label_and_output

            train_l, train_o = train_l.detach().numpy(), train_o.detach().numpy()
            test_l, test_o = test_l.detach().numpy(), test_o.detach().numpy()
            wanted_indices = train_l.shape[1]
            wanted_axis = 1
            train_l_by_time_point = np.split(train_l, indices_or_sections=wanted_indices, axis=wanted_axis)
            train_o_by_time_point = np.split(train_o, indices_or_sections=wanted_indices, axis=wanted_axis)
            test_l_by_time_point = np.split(test_l, indices_or_sections=wanted_indices, axis=wanted_axis)
            test_o_by_time_point = np.split(test_o, indices_or_sections=wanted_indices, axis=wanted_axis)

            # train_l_by_bacteria = np.split(train_l, indices_or_sections=train_l.shape[2], axis=2)

            for i, (train_l, train_o, test_l, test_o) in enumerate(zip(train_l_by_time_point, train_o_by_time_point,
                                              test_l_by_time_point, test_o_by_time_point)):
                train_l, train_o = train_l.flatten(), train_o.flatten()
                test_l, test_o = test_l.flatten(), test_o.flatten()

                min_val = min(min(np.min(train_l), np.min(train_o)), min(np.min(test_l), np.min(test_o)))
                max_val = max(max(np.max(train_l), np.max(train_o)), max(np.max(test_l), np.max(test_o)))
                fig, ax = plt.subplots()
                plt.ylim((min_val, max_val))
                plt.xlim((min_val, max_val))
                plt.scatter(train_l, train_o,
                            label='Train', color='red', s=3, alpha=0.3)
                plt.scatter(test_l, test_o,
                            label='Test', color='blue', s=3, alpha=0.3)
                plt.title(fig_title + " - Time Point " + str(i+1), fontsize=10)
                plt.xlabel('Real values')
                plt.ylabel('Predicted Values')
                plt.legend(loc='upper left')
                print(folder_and_name)
                plt.savefig(folder_and_name + "_time_Point_" + str(i+1) + ".png")
                # plt.show()

# ----------------------------------------------- run model -----------------------------------------------


def run_nn_experiment(X, y, missing_values, params, folder, LOSS, person_indexes, save_model, GPU_flag = False,
                      LSTM_LAYER_NUM = 2, k_fold=False, task_id=""):
    out_dim = 1 if len(y.shape) == 1 else y.shape[1]  # else NUMBER_OF_BACTERIA
    model_dim = 1
    CORR_FUNC = "single_bacteria_custom_corr_for_missing_values" if len(y.shape) == 1 else "multi_bacteria_custom_corr_for_missing_values"
    EARLY_STOP = 0
    BATCH_SIZE = NUMBER_OF_SAMPLES
    structure = params["STRUCTURE"]
    layer_num = int(structure[0:3])
    hid_dim_1 = int(structure[4:7])
    hid_dim_2 = int(structure[8:11]) if len(structure) > 10 else None
    params_str = params.__str__().replace(" ", "").replace("'", "") + "_" + task_id

    microbiome_dataset = MicrobiomeDataset(X, y, missing_values)
    activator_params = NNActivatorParams(TRAIN_TEST_SPLIT=params["TRAIN_TEST_SPLIT"],
                                         LOSS=LOSS,
                                         CORR=CORR_FUNC,
                                         BATCH_SIZE=BATCH_SIZE,
                                         GPU=GPU_flag,
                                         EPOCHS=params["EPOCHS"],
                                         EARLY_STOP=EARLY_STOP)
    if k_fold:
        pass
        # run_nn_fold(5, params, out_dim, activator_params, microbiome_dataset, folder)

    else:
        activator = Activator(MicrobiomeModule(NNModuleParams(NUMBER_OF_BACTERIA=NUMBER_OF_BACTERIA,
                                                              nn_hid_dim_1=hid_dim_1,
                                                              nn_hid_dim_2=hid_dim_2,
                                                              nn_output_dim=out_dim,
                                                              NN_LAYER_NUM=layer_num,
                                                              DROPOUT=params["DROPOUT"],
                                                              LEARNING_RATE=params["LEARNING_RATE"],
                                                              OPTIMIZER=params["OPTIMIZER"],
                                                              REGULARIZATION=params["REGULARIZATION"],
                                                              DIM=model_dim, SHUFFLE=SHUFFLE)),
                              activator_params, microbiome_dataset, split_microbiome_dataset, person_indexes)


        activator.train(validate_rate=1)
        if save_model:
            if not os.path.exists(os.path.join(folder, "trained_models")):
                os.makedirs(os.path.join(folder, "trained_models"))
            activator.save_model(os.path.join(folder, "trained_models", params_str + "_model"))

        print(params_str)
        results_sub_folder = os.path.join(folder, "NN_RESULTS")
        if not os.path.exists(results_sub_folder):
            os.makedirs(results_sub_folder)
        if False:
            r2_for_each_target_value_vec_train = [np.mean(traget_r2_values[-10:]) for traget_r2_values in
                                                  np.transpose(activator.r2_for_each_target_value_vec_train)]
            r2_for_each_target_value_vec_dev = [np.mean(traget_r2_values[-10:]) for traget_r2_values in
                                                np.transpose(activator.r2_for_each_target_value_dev_vec)]
            df = pd.DataFrame(index=range(1, len(r2_for_each_target_value_vec_dev) + 1), columns=["train", "dev"])
            df["train"] = r2_for_each_target_value_vec_train
            df["dev"] = r2_for_each_target_value_vec_dev
            df.to_csv(os.path.join(folder, "NNI", params_str + "_r2_for_each_target_value.csv"))



            with open(os.path.join(folder, "NNI", params_str + "_score.txt"), "w") as s:
                s.write("loss," + ",".join([str(v) for v in range(len(activator.loss_train_vec))]) + "\n")
                s.write("train_loss," + ",".join([str(v) for v in activator.loss_train_vec]) + "\n")
                s.write("dev_loss," + ",".join([str(v) for v in activator.loss_dev_vec]) + "\n")

            df = pd.read_csv(os.path.join(folder, "NNI", params_str + "_score.txt"))
            df.to_csv(os.path.join(folder, "NNI", params_str + "_score.csv"), index=False)
            os.remove(os.path.join(folder, "NNI", params_str + "_score.txt"))

            with open(os.path.join(folder, "NNI", params_str + "_r2.txt"), "w") as s:
                s.write("r2," + ",".join([str(v) for v in range(len(activator.r2_train_vec))]) + "\n")
                s.write("train_r2," + ",".join([str(v) for v in activator.r2_train_vec]) + "\n")
                s.write("dev_r2," + ",".join([str(v) for v in activator.r2_dev_vec]) + "\n")

            df = pd.read_csv(os.path.join(folder, "NNI", params_str + "_r2.txt"))
            df.to_csv(os.path.join(folder, "NNI", params_str + "_r2.csv"), index=False)
            os.remove(os.path.join(folder, "NNI", params_str + "_r2.txt"))

        title = "NN STRUCTURE: " + str(NUMBER_OF_BACTERIA) + "-"
        if layer_num == 1:
            title += str(hid_dim_1) + "-" + str(out_dim)
        if layer_num == 2:
            title += str(hid_dim_1) + "-" + str(hid_dim_2) + "-" + str(out_dim)

        title += "  BATCH: " + str(BATCH_SIZE) + " {" + task_id + "}"

        title += "\nREG: " + str(params["REGULARIZATION"]) + " DRO: " + str(params["DROPOUT"])
        title += " LR: " + str(params["LEARNING_RATE"]) + " OPT: " + str(params["OPTIMIZER"])
        if SAVE_RUN_RESULTS:
            activator.plot_line(title, os.path.join(results_sub_folder, params_str + "_fig"), job=LOSS_PLOT)
        dev_avg_loss = np.mean(activator.loss_dev_vec[-10:])
        train_avg_loss = np.mean(activator.loss_train_vec[-10:])

        dev_avg_corr = None
        train_avg_corr = None
        if CALC_CORR:
            if SAVE_RUN_RESULTS:
                activator.plot_line(title, os.path.join(results_sub_folder, params_str + "_fig"), job=CORR_PLOT)
            dev_avg_corr = np.mean(activator.corr_dev_vec[-10:])
            train_avg_corr = np.mean(activator.corr_dev_vec[-10:])

        dev_avg_r2 = None
        train_avg_r2 = None
        if CALC_R2:
            if SAVE_RUN_RESULTS:
                activator.plot_line(title, os.path.join(results_sub_folder, params_str + "_fig"), job=R2_PLOT)
            dev_avg_r2 = np.mean(activator.r2_dev_vec[-10:])
            train_avg_r2 = np.mean(activator.r2_train_vec[-10:])


        if SAVE_RUN_RESULTS:
            activator.scatter_results("Scatter Plot of NN Results\n" + title,
                                  os.path.join(results_sub_folder, params_str + "_Scatter_Plot.png"))

    result_map = {"TRAIN": {"loss": train_avg_loss,
                            "corr": train_avg_corr,
                            "r2": train_avg_r2},
                  "TEST": {"loss": dev_avg_loss,
                            "corr": dev_avg_corr,
                            "r2": dev_avg_r2}
                  }
    r_df = pd.DataFrame(result_map)
    if SAVE_RUN_RESULTS:
        r_df.to_csv(os.path.join(results_sub_folder, params_str + "_results_df.csv"))
    return result_map


def run_NN(X, y, missing_values, params, name, folder, number_of_samples, number_of_time_points, number_of_bacteria,
           save_model=False,  person_indexes=None, Loss = "custom_rmse_for_missing_value", add_conv_layer=False,
           GPU_flag=False, k_fold=False, task_id=""):
    print(name)
    global NUMBER_OF_SAMPLES
    global NUMBER_OF_TIME_POINTS
    global NUMBER_OF_BACTERIA
    global CONV_LAYER
    NUMBER_OF_SAMPLES = number_of_samples
    NUMBER_OF_TIME_POINTS = number_of_time_points
    NUMBER_OF_BACTERIA = number_of_bacteria
    CONV_LAYER = add_conv_layer

    return run_nn_experiment(X, y, missing_values, params, folder, Loss, person_indexes, save_model, GPU_flag=GPU_flag,
                             k_fold=k_fold, task_id=task_id)



if __name__ == "__main__":
    pass
    # run_NN(X, y, missing_values, params, name, folder, number_of_samples, number_of_time_points, number_of_bacteria, GPU_flag=False, k_fold=False, task_id="")
"""
def run_nn_fold(k, params, out_dim, activator_params, microbiome_dataset, folder):
    dim = 1
    title = "NET STRUCTURE: " + str(NUMBER_OF_BACTERIA) + "-"
    if params["NN_LAYER_NUM"] == 1:
        title += str(params["HID_DIM_1"]) + "-" + str(out_dim)
    if params["NN_LAYER_NUM"] == 2:
        title += str(params["HID_DIM_1"]) + "-" + str(params["HID_DIM_2"]) + "-" + str(out_dim)

    title += "  BATCH: " + str(params["BATCH_SIZE"])

    title += "\nLEARNING_RATE: " + str(params["LEARNING_RATE"]) + " OPTIMIZER: " + str(params["OPTIMIZER"]) + \
             " REGULARIZATION: " + str(params["REGULARIZATION"]) + " DROPOUT: " + str(params["DROPOUT"])

    t1, t2 = title.split("\n")
    job = LOSS_PLOT
    p = figure(plot_width=600, plot_height=250, x_axis_label="epochs", y_axis_label=job)
    p.add_layout(Title(text=t2), 'above')
    p.add_layout(Title(text=t1), 'above')

    color1, color2 = ("orange", "red") if job == LOSS_PLOT else ("green", "blue")

    # ----------------------------------------K-FOLD----------------------------------------
    for k in range(1, 6):
        activator = Activator(MicrobiomeModule(NNModuleParams(NUMBER_OF_BACTERIA=NUMBER_OF_BACTERIA,
                                                              nn_hid_dim_1=params["HID_DIM_1"],
                                                              nn_hid_dim_2=params["HID_DIM_2"],
                                                              nn_output_dim=out_dim,
                                                              NN_LAYER_NUM=params["NN_LAYER_NUM"],
                                                              DROPOUT=params["DROPOUT"],
                                                              LEARNING_RATE=params["LEARNING_RATE"],
                                                              OPTIMIZER=params["OPTIMIZER"],
                                                              REGULARIZATION=params["REGULARIZATION"])),
                              activator_params, microbiome_dataset, split_microbiome_dataset, dim, SHUFFLE)

        activator.train(validate_rate=1)

        print(params.__str__().replace(" ", ""))

        with open(os.path.join(folder, "NNI", params.__str__().replace(" ", "") + "_score_" + str(k) + ".txt"), "w") as s:
            s.write("loss," + ",".join([str(v) for v in range(len(activator.loss_train_vec))]) + "\n")
            s.write("train_loss," + ",".join([str(v) for v in activator.loss_train_vec]) + "\n")
            s.write("dev_loss," + ",".join([str(v) for v in activator.loss_dev_vec]) + "\n")

        df = pd.read_csv(os.path.join(folder, "NNI", params.__str__().replace(" ", "") + "_score_" + str(k) + ".txt"))
        df.to_csv(os.path.join(folder, "NNI", params.__str__().replace(" ", "") + "_score_" + str(k) + ".csv"), index=False)
        os.remove(os.path.join(folder, "NNI", params.__str__().replace(" ", "") + "_score_" + str(k) + ".txt"))


        with open(os.path.join(folder, "NNI", params.__str__().replace(" ", "") + "_score.txt"), "w") as s:
            s.write("loss," + ",".join([str(v) for v in range(len(activator.loss_train_vec))]) + "\n")
            s.write("train_loss," + ",".join([str(v) for v in activator.loss_train_vec]) + "\n")
            s.write("dev_loss," + ",".join([str(v) for v in activator.loss_dev_vec]) + "\n")

        with open(os.path.join(folder, "NNI", params.__str__().replace(" ", "") + "_r2.txt"), "w") as s:
            s.write("r2," + ",".join([str(v) for v in range(len(activator.r2_train_vec))]) + "\n")
            s.write("train_r2," + ",".join([str(v) for v in activator.r2_train_vec]) + "\n")
            s.write("dev_r2," + ",".join([str(v) for v in activator.r2_dev_vec]) + "\n")

        df = pd.read_csv(os.path.join(folder, "NNI", params.__str__().replace(" ", "") + "_score.txt"))
        df.to_csv(os.path.join(folder, "NNI", params.__str__().replace(" ", "") + "_score.csv"), index=False)
        os.remove(os.path.join(folder, "NNI", params.__str__().replace(" ", "") + "_score.txt"))


        y_axis_train = activator.loss_train_vec
        y_axis_dev = activator.loss_dev_vec

        x_axis = list(range(len(y_axis_dev)))
        p.line(x_axis, y_axis_train, line_color=color1, legend_label="train")
        p.line(x_axis, y_axis_dev, line_color=color2, legend_label="dev")
    output_file(os.path.join(folder, params.__str__().replace(" ", "") + " " + job + "_fig.html"))
    save(p)
    show(p)
"""