from collections import Counter
from sys import stdout

from bokeh.io import output_file, save
from bokeh.plotting import figure, show
from sklearn.metrics import roc_auc_score
from torch.nn import DataParallel
from torch.utils.data import DataLoader, random_split
import numpy as np
from binary_params import BinaryActivatorParams
from binary_rnn_model import BinaryModule
from fst_dataset import FstDataset, split_fst_dataset
import torch
import nni
TRAIN_JOB = "TRAIN"
DEV_JOB = "DEV"
VALIDATE_JOB = "VALIDATE"
LOSS_PLOT = "loss"
ACCURACY_PLOT = "accuracy"
AUC_PLOT = "ROC-AUC"


class binaryActivator:
    def __init__(self, model: BinaryModule, params: BinaryActivatorParams, data: FstDataset, splitter):
        self._model = (model).cuda() if params.GPU else model
        self._gpu = params.GPU
        self._epochs = params.EPOCHS
        self._batch_size = params.BATCH_SIZE
        self._loss_func = params.LOSS
        self._load_data(data, params.TRAIN_TEST_SPLIT, params.BATCH_SIZE, splitter)
        self._init_loss_and_acc_vec()
        self._init_print_att()

    # init loss and accuracy vectors (as function of epochs)
    def _init_loss_and_acc_vec(self):
        self._loss_vec_train = []
        self._loss_vec_dev = []
        self._accuracy_vec_train = []
        self._accuracy_vec_dev = []
        self._auc_vec_train = []
        self._auc_vec_dev = []

    # init variables that holds the last update for loss and accuracy
    def _init_print_att(self):
        self._print_train_accuracy = 0
        self._print_train_loss = 0
        self._print_train_auc = 0
        self._print_dev_accuracy = 0
        self._print_dev_loss = 0
        self._print_dev_auc = 0

    # update loss after validating
    def _update_loss(self, loss, job=TRAIN_JOB):
        if job == TRAIN_JOB:
            self._loss_vec_train.append(loss)
            self._print_train_loss = loss
        elif job == DEV_JOB:
            self._loss_vec_dev.append(loss)
            self._print_dev_loss = loss

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
        prog = int(100 * (batch_index + 1) / len_data)
        stdout.write("\r\r\r\r\r\r\r\r" + job + " %d" % prog + "%")
        print("", end="\n" if prog == 100 else "")
        stdout.flush()

    # print last loss and accuracy
    def _print_info(self, jobs=()):
        if TRAIN_JOB in jobs:
            print("Acc_Train: " + '{:{width}.{prec}f}'.format(self._print_train_accuracy, width=6, prec=4) +
                  " || AUC_Train: " + '{:{width}.{prec}f}'.format(self._print_train_auc, width=6, prec=4) +
                  " || Loss_Train: " + '{:{width}.{prec}f}'.format(self._print_train_loss, width=6, prec=4),
                  end=" || ")
        if DEV_JOB in jobs:
            print("Acc_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_accuracy, width=6, prec=4) +
                  " || AUC_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_auc, width=6, prec=4) +
                  " || Loss_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_loss, width=6, prec=4),
                  end=" || ")
        print("")

    # plot loss / accuracy graph
    def plot_line(self, job=LOSS_PLOT):
        p = figure(plot_width=600, plot_height=250, title="Rand_FST - Dataset " + job,
                   x_axis_label="epochs", y_axis_label=job)
        color1, color2 = ("orange", "red") if job == LOSS_PLOT else ("green", "blue")

        if job == LOSS_PLOT:
            y_axis_train = self._loss_vec_train if job == LOSS_PLOT else self._accuracy_vec_train
            y_axis_dev = self._loss_vec_dev if job == LOSS_PLOT else self._accuracy_vec_dev
        elif job == ACCURACY_PLOT:
            y_axis_train = self._accuracy_vec_train
            y_axis_dev = self._accuracy_vec_dev
        elif job == AUC_PLOT:
            y_axis_train = self._auc_vec_train
            y_axis_dev = self._auc_vec_dev

        x_axis = list(range(len(y_axis_dev)))
        p.line(x_axis, y_axis_train, line_color=color1, legend="train")
        p.line(x_axis, y_axis_dev, line_color=color2, legend="dev")
        output_file(job + "_fig.html")
        save(p)
        show(p)

    def _plot_acc_dev(self):
        self.plot_line(LOSS_PLOT)
        self.plot_line(ACCURACY_PLOT)
        self.plot_line(AUC_PLOT)

    @property
    def model(self):
        return self._model

    @property
    def loss_train_vec(self):
        return self._loss_vec_train

    @property
    def accuracy_train_vec(self):
        return self._accuracy_vec_train

    @property
    def auc_train_vec(self):
        return self._auc_vec_train

    @property
    def loss_dev_vec(self):
        return self._loss_vec_dev

    @property
    def accuracy_dev_vec(self):
        return self._accuracy_vec_dev

    @property
    def auc_dev_vec(self):
        return self._auc_vec_dev

    # load dataset
    def _load_data(self, train_dataset, train_split, batch_size, splitter):
        # split dataset
        train, dev = splitter(train_dataset, [train_split, 1-train_split])
        # set train loader
        self._train_loader = DataLoader(
            train,
            batch_size=batch_size,
            collate_fn=train.collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=8
        )

        self._train_valid_loader = DataLoader(
            train,
            batch_size=100,
            collate_fn=train.collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=8
        )

        # set validation loader
        self._dev_loader = DataLoader(
            dev,
            batch_size=100,
            collate_fn=dev.collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=8
        )

    def _to_gpu(self, x, l):
        x = x.cuda() if self._gpu else x
        l = l.cuda() if self._gpu else l
        return x, l

    # train a model, input is the enum of the model type
    def train(self, show_plot=True, apply_nni=False, validate_rate=10, early_stop=False):
        self._init_loss_and_acc_vec()
        # calc number of iteration in current epoch
        len_data = len(self._train_loader)
        for epoch_num in range(self._epochs):
            # calc number of iteration in current epoch
            for batch_index, (sequence, label) in enumerate(self._train_loader):
                sequence, label = self._to_gpu(sequence, label)
                # print progress
                self._model.train()

                output = self._model(sequence)                  # calc output of current model on the current batch
                loss = self._loss_func(output.squeeze(dim=1), label.float())  # calculate loss
                loss.backward()                                 # back propagation
                self._model.optimizer.step()                    # update weights
                self._model.zero_grad()                         # zero gradients

                self._print_progress(batch_index, len_data, job=TRAIN_JOB)
            # validate and print progress

            # /----------------------  FOR NNI  -------------------------
            if epoch_num % validate_rate == 0:
                # validate on dev set anyway
                self._validate(self._dev_loader, job=DEV_JOB)
                torch.cuda.empty_cache()
                # report dev result as am intermediate result
                if apply_nni:
                    test_auc = self._print_dev_auc
                    nni.report_intermediate_result(test_auc)
                # validate on train set as well and display results
                else:
                    torch.cuda.empty_cache()
                    self._validate(self._train_valid_loader, job=TRAIN_JOB)
                    self._print_info(jobs=[TRAIN_JOB, DEV_JOB])

            if early_stop and epoch_num > 30 and self._print_dev_loss > np.max(self._loss_vec_dev[-30:]):
                break

        # report final results
        if apply_nni:
            test_auc = np.max(self._print_dev_accuracy)
            nni.report_final_result(test_auc)

        # -----------------------  FOR NNI  -------------------------/

        if show_plot:
            self._plot_acc_dev()

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader, job=""):
        # for calculating total loss and accuracy
        loss_count = 0
        true_labels = []
        pred = []

        self._model.eval()
        # calc number of iteration in current epoch
        len_data = len(data_loader)
        for batch_index, (sequence, label) in enumerate(data_loader):
            sequence, label = self._to_gpu(sequence, label)
            # print progress
            self._print_progress(batch_index, len_data, job=VALIDATE_JOB)
            output = self._model(sequence)
            # calculate total loss
            loss_count += self._loss_func(output.squeeze(dim=1), label.float())
            true_labels += label.tolist()
            pred += output.squeeze().tolist()

        # update loss accuracy
        loss = float(loss_count / len(data_loader))
        self._update_loss(loss, job=job)
        self._update_accuracy(pred, true_labels, job=job)
        self._update_auc(pred, true_labels, job=job)
        return loss


if __name__ == '__main__':
    from binary_params import BinaryFSTParams, BinaryModuleParams
    fst_dataset = FstDataset(BinaryFSTParams())
    activator = binaryActivator(BinaryModule(BinaryModuleParams(alphabet_size=len(fst_dataset.chr_embed))),
                                BinaryActivatorParams(), fst_dataset, split_fst_dataset)
    activator.train()
