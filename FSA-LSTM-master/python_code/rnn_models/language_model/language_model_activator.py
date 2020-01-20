import sys
import os
from numpy import average
sys.path.insert(0, os.path.join(".."))
sys.path.insert(0, os.path.join("..", ".."))
sys.path.insert(0, os.path.join("..", "..", "finit_state_machine"))
sys.path.insert(0, os.path.join("..", "..", "rnn_models"))

from sys import stdout
from bokeh.plotting import figure, show
from torch.utils.data import DataLoader
from fat_language_model_dataset import FstLanguageModuleDataset, split_fst_language_model_dataset
from language_model import LanguageModule
from language_model_params import LanguageModelFSTParams, LanguageModelParams, LanguageModelActivatorParams
import torch


TRAIN_JOB = "TRAIN"
DEV_JOB = "DEV"
VALIDATE_JOB = "VALIDATE"
LOSS_PLOT = "loss"
ACCURACY_PLOT = "accuracy"


class LanguageModuleActivator:
    def __init__(self, model: LanguageModule, params: LanguageModelActivatorParams, data: FstLanguageModuleDataset,
                 report_intermediate=None, splitter=None):
        self._model = model.cuda() if params.GPU else model
        self._gpu = params.GPU
        self._epochs = params.EPOCHS
        self._batch_size = params.BATCH_SIZE
        self._loss_func, self._loss_kargs = params.LOSS
        self._load_data(data, params.TRAIN_TEST_SPLIT, params.BATCH_SIZE, splitter)
        self._init_loss_and_acc_vec()
        self._init_print_att()
        self._report_intermediate = report_intermediate

    def _hyper_parameter_tuning_intermediate_res(self, id):
        if self._report_intermediate is None:
            return
        measures = {
            "Acc_Train":            self._print_train_accuracy,
            "Acceptor_Acc_Train":   self._print_train_acceptor_acc,
            "Loss_Train":           self._print_train_loss,
            "Acc_Dev":              self._print_dev_accuracy,
            "Acceptor_Acc_Dev":     self._print_dev_acceptor_acc,
            "Loss_Dev":             self._print_dev_loss
        }
        self._report_intermediate(id, measures)


    # init loss and accuracy vectors (as function of epochs)
    def _init_loss_and_acc_vec(self):
        self._loss_vec_train = []
        self._loss_vec_dev = []
        self._accuracy_vec_train = []
        self._accuracy_vec_dev = []
        self._acceptor_acc_vec_train = []
        self._acceptor_acc_vec_dev = []

    # init variables that holds the last update for loss and accuracy
    def _init_print_att(self):
        self._print_train_accuracy = 0
        self._print_train_loss = 0
        self._print_train_acceptor_acc = 0
        self._print_dev_accuracy = 0
        self._print_dev_loss = 0
        self._print_dev_acceptor_acc = 0

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
        true_without_pad = list(filter(lambda a: a != self._loss_kargs.get("ignore_index", -100), true))
        acc = sum([1 if int(i) == int(j) and int(j) != self._loss_kargs.get("ignore_index", -100)
                   else 0 for i, j in zip(pred, true)]) / len(true_without_pad)
        if job == TRAIN_JOB:
            self._print_train_accuracy = acc
            self._accuracy_vec_train.append(acc)
            return acc
        elif job == DEV_JOB:
            self._print_dev_accuracy = acc
            self._accuracy_vec_dev.append(acc)
            return acc

    # update accuracy after validating
    def _update_acceptor_accuracy(self, pred, true, job=TRAIN_JOB):
        # calculate acc
        end_idx = self._train_loader.dataset.end_idx

        acc_vec = []
        for p, t in zip(pred, true):
            if t == end_idx:
                acc_vec.append(1 if p == end_idx else 0)
        acc = average(acc_vec)

        if job == TRAIN_JOB:
            self._print_train_acceptor_acc = acc
            self._acceptor_acc_vec_train.append(acc)
            return acc
        elif job == DEV_JOB:
            self._print_dev_acceptor_acc = acc
            self._acceptor_acc_vec_dev.append(acc)
            return acc

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
                  " || Acceptor_Acc_Train: " + '{:{width}.{prec}f}'.format(self._print_train_acceptor_acc, width=6, prec=4) +
                  " || Loss_Train: " + '{:{width}.{prec}f}'.format(self._print_train_loss, width=6, prec=4),
                  end=" || ")
        if DEV_JOB in jobs:
            print("Acc_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_accuracy, width=6, prec=4) +
                  " || Acceptor_Acc_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_acceptor_acc, width=6, prec=4) +
                  " || Loss_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_loss, width=6, prec=4),
                  end=" || ")
        print("")

    # plot loss / accuracy graph
    def plot_line(self, job=LOSS_PLOT):
        p = figure(plot_width=600, plot_height=250, title="Rand_FST - Dataset " + job,
                   x_axis_label="epochs", y_axis_label=job)
        color1, color2 = ("orange", "red") if job == LOSS_PLOT else ("green", "blue")
        y_axis_train = self._loss_vec_train if job == LOSS_PLOT else self._accuracy_vec_train
        y_axis_dev = self._loss_vec_dev if job == LOSS_PLOT else self._accuracy_vec_dev
        x_axis = list(range(len(y_axis_dev)))
        p.line(x_axis, y_axis_train, line_color=color1, legend="train")
        p.line(x_axis, y_axis_dev, line_color=color2, legend="dev")
        show(p)

    def _plot_acc_dev(self):
        self.plot_line(LOSS_PLOT)
        self.plot_line(ACCURACY_PLOT)

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
    def acceptor_acc_train_vec(self):
        return self._acceptor_acc_vec_train

    @property
    def loss_dev_vec(self):
        return self._loss_vec_dev

    @property
    def accuracy_dev_vec(self):
        return self._accuracy_vec_dev

    @property
    def acceptor_acc_dev_vec(self):
        return self._acceptor_acc_vec_dev

    # load dataset
    def _load_data(self, train_dataset, train_split, batch_size, splitter):
        # split dataset
        splitter = split_fst_language_model_dataset if splitter is None else splitter
        train, dev = splitter(train_dataset, [train_split, 1 - train_split])
        # set train loader
        self._train_loader = DataLoader(
            train,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )

        self._train_valid_loader = DataLoader(
            train,
            batch_size=len(train),
            collate_fn=train_dataset.collate_fn,
            shuffle=False
        )

        # set validation loader
        self._dev_loader = DataLoader(
            dev,
            batch_size=len(dev),
            collate_fn=dev.collate_fn,
            shuffle=False
        )

    def _to_gpu(self, x, l, h):
        x = x.cuda() if self._gpu else x                    # input
        l = l.cuda() if self._gpu else l                    # label
        h = (h[0].cuda(), h[1].cuda()) if self._gpu else h  # hidden
        return x, l, h

    # train a model, input is the enum of the model type
    def train(self, show_plot=True, valid_rate=1):
        self._init_loss_and_acc_vec()
        # calc number of iteration in current epoch
        len_data = len(self._train_loader)
        for epoch_num in range(self._epochs):
            self._model.train()
            # calc number of iteration in current epoch

            for batch_index, (sequence, label) in enumerate(self._train_loader):
                self._print_progress(batch_index, len_data, job=TRAIN_JOB)
                hidden = (torch.zeros((self.model.lstm_layers, sequence.shape[0], self.model.dim_hidden_lstm)),
                          # dim = (bach, len_seq=1, hidden_dim)
                          torch.zeros((self.model.lstm_layers, sequence.shape[0], self.model.dim_hidden_lstm)))
                sequence, label, hidden = self._to_gpu(sequence, label, hidden)
                output, hidden = self.model(sequence, hidden)

                loss = self._loss_func(output.reshape(-1, output.shape[2]), label.reshape(-1), **self._loss_kargs)
                loss.backward()
                self._model.optimizer.step()                    # update weights
                self._model.zero_grad()                         # zero gradients

            # validate and print progress
            if epoch_num % valid_rate == 0 or epoch_num + 1 == self._epochs:
                self._validate(self._train_loader, job=TRAIN_JOB)
                self._validate(self._dev_loader, job=DEV_JOB)
                self._hyper_parameter_tuning_intermediate_res(epoch_num)
                self._print_info(jobs=[TRAIN_JOB, DEV_JOB])

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
            hidden = (torch.zeros((self.model.lstm_layers, sequence.shape[0], self.model.dim_hidden_lstm)),
                      # dim = (bach, len_seq=1, hidden_dim)
                      torch.zeros((self.model.lstm_layers, sequence.shape[0], self.model.dim_hidden_lstm)))
            sequence, label, hidden = self._to_gpu(sequence, label, hidden)

            self._print_progress(batch_index, len_data, job=VALIDATE_JOB)
            output, hidden = self.model(sequence, hidden)

            # calculate total loss
            loss_count += self._loss_func(output.reshape(-1, output.shape[2]), label.reshape(-1), **self._loss_kargs)
            true_labels += label.reshape(-1).tolist()
            pred += output.reshape(-1, output.shape[2]).argmax(dim=1).tolist()

        # update loss accuracy
        loss = float(loss_count / len(data_loader))
        self._update_loss(loss, job=job)
        self._update_accuracy(pred, true_labels, job=job)
        self._update_acceptor_accuracy(pred, true_labels, job=job)
        return loss


if __name__ == '__main__':
    _ds_params = LanguageModelFSTParams()
    _ds = FstLanguageModuleDataset(_ds_params)
    activator = LanguageModuleActivator(LanguageModule(LanguageModelParams(alphabet_size=_ds_params.FST_ALPHABET_SIZE)),
                                        LanguageModelActivatorParams(ignore_index=_ds.pad_idx), _ds)
    activator.train()
