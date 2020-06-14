import os
from random import shuffle
from sys import stdout

import numpy as np
import nni
import torch
from matplotlib.pyplot import figure
from sklearn import metrics
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD, adagrad
from torch.nn.functional import mse_loss
from bokeh.io import output_file, save
from bokeh.plotting import figure, show
from torch.utils.data import DataLoader
TRAIN_JOB = "TRAIN"
DEV_JOB = "DEV"
VALIDATE_JOB = "VALIDATE"
LOSS_PLOT = "loss"
ACCURACY_PLOT = "accuracy"
AUC_PLOT = "ROC-AUC"

NUMBER_OF_BACTERIA = 0
NUMBER_OF_TIME_POINTS = 0
NUMBER_OF_SAMPLES = 0


# loss_name_to_function_map = {"custom_rmse_for_missing_value": custom_rmse_for_missing_values}

optimizer_name_to_function_map = {"Adam": Adam, "SGD": SGD, "Adagrad": adagrad}


def custom_rmse_for_missing_values(input, target, missing_values):
    # which option us better? torch.mean => average loss or torch.sum => total loss
    print(target.shape)
    print(input.shape)
    return torch.mean(torch.norm(torch.mul(torch.sub(target, input), missing_values.float()), dim=1))


class ActivatorParams:
    def __init__(self, TRAIN_TEST_SPLIT=0.8, LOSS="custom_rmse_for_missing_value", BATCH_SIZE=100,
                 GPU=False, EPOCHS=50, EARLY_STOP=0):

        self.TRAIN_TEST_SPLIT = TRAIN_TEST_SPLIT
        self.LOSS = custom_rmse_for_missing_values
        self.BATCH_SIZE = BATCH_SIZE
        self.GPU = GPU
        self.EPOCHS = EPOCHS
        self.EARLY_STOP = EARLY_STOP


class ModuleParams:
    def __init__(self, lstm_out_dim=NUMBER_OF_BACTERIA, LSTM_LAYER_NUM=1, DROPOUT=0, LEARNING_RATE=1e-3,
                 OPTIMIZER="Adam", REGULARIZATION=1e-4):
        self.SEQUENCE_PARAMS = SequenceParams(out_dim=lstm_out_dim,
                                                     LSTM_LAYER_NUM=LSTM_LAYER_NUM,
                                                     DROPOUT=DROPOUT)
        self.LINEAR_PARAMS = MLPParams(in_dim=NUMBER_OF_TIME_POINTS, )
        self.LEARNING_RATE = LEARNING_RATE
        self.OPTIMIZER = optimizer_name_to_function_map[OPTIMIZER]
        self.REGULARIZATION = REGULARIZATION


class SequenceParams:
    def __init__(self, out_dim, LSTM_LAYER_NUM, DROPOUT):
        self.LSTM_hidden_dim = out_dim
        self.LSTM_layers = LSTM_LAYER_NUM
        self.LSTM_dropout = DROPOUT


class MLPParams:
    def __init__(self, in_dim):
        self.LINEAR_in_dim = in_dim
        self.LINEAR_out_dim = in_dim  # 1


class MicrobiomeModule(Module):
    def __init__(self, params: ModuleParams):
        super(MicrobiomeModule, self).__init__()
        # useful info in forward function
        self._sequence_lstm = SequenceModule(params.SEQUENCE_PARAMS)
        # self._mlp = MLPModule(params.LINEAR_PARAMS)
        self.optimizer = self.set_optimizer(params.LEARNING_RATE, params.OPTIMIZER, params.REGULARIZATION)

    def set_optimizer(self, lr, opt, l2_reg):
        return opt(self.parameters(), lr=lr, weight_decay=l2_reg)

    def forward(self, x):
        x = self._sequence_lstm(x)
        # x = self._mlp(x)  # .permute(0, 2, 1))!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return x


class SequenceModule(Module):
    def __init__(self, params):
        super(SequenceModule, self).__init__()
        self._lstm = LSTM(params.LSTM_hidden_dim, params.LSTM_layers, batch_first=True,
                          bidirectional=False, dropout=params.LSTM_dropout)

    def forward(self, x):
        # 3 layers LSTM
        output_seq, _ = self._lstm(x.float())
        print(self._lstm)
        print(output_seq.shape)
        return output_seq   # for single bacteria model .transpose(1, 2)


class MLPModule(Module):
    def __init__(self, params: MLPParams):
        super(MLPModule, self).__init__()
        # useful info in forward function
        self._linear = Linear(params.LINEAR_in_dim, params.LINEAR_out_dim)

    def forward(self, x):
        x = self._linear(x)
        return x


class MicrobiomeDataset(Dataset):
    def __init__(self, X, y, missing_values):
        self._X = X
        self._y = y
        self._missing_values = missing_values

    def __getitem__(self, index):
        return self._X[index], self._y[index], self._missing_values[index]

    def __len__(self):
        return len(self._X)

TRAIN_JOB = "TRAIN"
DEV_JOB = "DEV"
VALIDATE_JOB = "VALIDATE"
LOSS_PLOT = "loss"
ACCURACY_PLOT = "accuracy"
AUC_PLOT = "ROC-AUC"


def split_microbiome_dataset(dataset: MicrobiomeDataset, split_list):
    """
    this function splits a data-set into n = len(split_list) disjointed data-sets
    """
    import numpy as np
    # create a list of lengths [0.1, 0.4, 0.5] -> [100, 500, 1000(=len_data)]
    split_list = np.multiply(np.cumsum(split_list), len(dataset)).astype("int").tolist()
    # list of shuffled indices to sample randomly
    shuffled_idx = list(range(len(dataset)))
    shuffle(shuffled_idx)
    # split the data itself
    new_data_X = [[] for _ in range(len(split_list))]
    new_data_y = [[] for _ in range(len(split_list))]
    new_data_missing_values = [[] for _ in range(len(split_list))]
    for sub_data_idx, (start, end) in enumerate(zip([0] + split_list[:-1], split_list)):
        for i in range(start, end):
            X, y, missing_values = dataset.__getitem__(shuffled_idx[i])
            new_data_X[sub_data_idx].append(np.array(X))
            new_data_y[sub_data_idx].append(np.array(y))
            new_data_missing_values[sub_data_idx].append(np.array(missing_values))
    # create sub sets
    # new_data_X = np.array(new_data_X)
    # new_data_y = np.array(new_data_y)
    sub_datasets = []
    for i in range(len(new_data_X)):
        sub_datasets.append(MicrobiomeDataset(np.array(new_data_X[i]),
                                              np.array(new_data_y[i]),
                                              np.array(new_data_missing_values[i])))
    return sub_datasets


class Activator:
    def __init__(self, model: MicrobiomeModule, params: ActivatorParams, data: MicrobiomeDataset, splitter):
        self._model = (model).cuda() if params.GPU else model
        self._gpu = params.GPU
        self._epochs = params.EPOCHS
        self._batch_size = params.BATCH_SIZE
        self._loss_func = params.LOSS
        self._early_stop = params.EARLY_STOP
        self._load_data(data, params.TRAIN_TEST_SPLIT, params.BATCH_SIZE, splitter)
        self._init_loss_and_acc_vec()
        self._init_print_att()

    # init loss and accuracy vectors (as function of epochs)
    def _init_loss_and_acc_vec(self):
        self._loss_vec_train = []
        self._loss_vec_dev = []
        """
        self._accuracy_vec_train = []
        self._accuracy_vec_dev = []
        self._auc_vec_train = []
        self._auc_vec_dev = []
        """

    # init variables that holds the last update for loss and accuracy
    def _init_print_att(self):
        self._print_train_loss = 0
        self._print_dev_loss = 0
        """
        self._print_train_accuracy = 0
        self._print_train_auc = 0
        self._print_dev_accuracy = 0
        self._print_dev_auc = 0
        """

    # update loss after validating
    def _update_loss(self, loss, job=TRAIN_JOB):
        if job == TRAIN_JOB:
            self._loss_vec_train.append(loss)
            self._print_train_loss = loss
        elif job == DEV_JOB:
            self._loss_vec_dev.append(loss)
            self._print_dev_loss = loss
    """
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
    """
    # print progress of a single epoch as a percentage
    def _print_progress(self, batch_index, len_data, job=""):
        prog = int(100 * (batch_index + 1) / len_data)
        stdout.write("\r\r\r\r\r\r\r\r" + job + " %d" % prog + "%")
        print("", end="\n" if prog == 100 else "")
        stdout.flush()

    # print last loss and accuracy
    def _print_info(self, jobs=()):
        if TRAIN_JOB in jobs:
            """
            print("Acc_Train: " + '{:{width}.{prec}f}'.format(self._print_train_accuracy, width=6, prec=4) +
                  " || AUC_Train: " + '{:{width}.{prec}f}'.format(self._print_train_auc, width=6, prec=4) +
                  " || ") """
            print("Loss_Train: " + '{:{width}.{prec}f}'.format(self._print_train_loss, width=6, prec=4),
                  end=" || ")
        if DEV_JOB in jobs:
            """
            print("Acc_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_accuracy, width=6, prec=4) +
                  " || AUC_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_auc, width=6, prec=4) +
                  " || ")"""
            print("Loss_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_loss, width=6, prec=4),
                  end=" || ")
        print("")

    # plot loss / accuracy graph
    def plot_line(self, title, folder, job=LOSS_PLOT):
        p = figure(plot_width=600, plot_height=250, title=title + " " + job,
                   x_axis_label="epochs", y_axis_label=job)
        color1, color2 = ("orange", "red") if job == LOSS_PLOT else ("green", "blue")

        # if job == LOSS_PLOT:
        y_axis_train = self._loss_vec_train  # if job == LOSS_PLOT else self._accuracy_vec_train
        y_axis_dev = self._loss_vec_dev # if job == LOSS_PLOT else self._accuracy_vec_dev
        """
        elif job == ACCURACY_PLOT:
            y_axis_train = self._accuracy_vec_train
            y_axis_dev = self._accuracy_vec_dev
        elif job == AUC_PLOT:
            y_axis_train = self._auc_vec_train
            y_axis_dev = self._auc_vec_dev
        """

        x_axis = list(range(len(y_axis_dev)))
        p.line(x_axis, y_axis_train, line_color=color1, legend_label="train")
        p.line(x_axis, y_axis_dev, line_color=color2, legend_label="dev")
        output_file(os.path.join(folder, title + " " + job + "_fig.html"))
        save(p)
        show(p)
    """
    def _plot_acc_dev(self):
        self.plot_line(LOSS_PLOT)
        self.plot_line(ACCURACY_PLOT)
        self.plot_line(AUC_PLOT)
    """
    @property
    def model(self):
        return self._model

    @property
    def loss_train_vec(self):
        return self._loss_vec_train

    @property
    def loss_dev_vec(self):
        return self._loss_vec_dev

    """
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
    """

    # load dataset
    def _load_data(self, train_dataset, train_split, batch_size, splitter):
        # split dataset
        train, dev = splitter(train_dataset, [train_split, 1-train_split])
        # set train loader
        self._train_loader = DataLoader(
            train,
            batch_size=batch_size,
            # collate_fn=train.collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=8
        )

        self._train_valid_loader = DataLoader(
            train,
            batch_size=100,
            # collate_fn=train.collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=8
        )

        # set validation loader
        self._dev_loader = DataLoader(
            dev,
            batch_size=100,
            # collate_fn=dev.collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=8
        )

    def _to_gpu(self, x, l):
        x = x.cuda() if self._gpu else x
        l = l.cuda() if self._gpu else l
        return x, l

    # train a model, input is the enum of the model type
    def train(self, show_plot=True, apply_nni=False, validate_rate=10):
        self._init_loss_and_acc_vec()
        # calc number of iteration in current epoch
        len_data = len(self._train_loader)
        for epoch_num in range(self._epochs):
            # calc number of iteration in current epoch
            for batch_index, (sequence, label, missing_values) in enumerate(self._train_loader):
                # sequence, label = self._to_gpu(sequence, label)
                # print progress
                self._model.train()

                output = self._model(sequence)                  # calc output of current model on the current batch
                loss = self._loss_func(output.squeeze(dim=1), label.float(), missing_values)  # calculate loss
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

            if self._early_stop and epoch_num > 30 and self._print_dev_loss > np.max(self._loss_vec_dev[-30:]):
                break

        # report final results
        if apply_nni:
            test_auc = np.max(self._print_dev_accuracy)
            nni.report_final_result(test_auc)

        # -----------------------  FOR NNI  -------------------------/
        """
        if show_plot:
            self._plot_acc_dev()
        """
    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader, job=""):
        # for calculating total loss and accuracy
        loss_count = 0
        true_labels = []
        pred = []

        self._model.eval()
        # calc number of iteration in current epoch
        len_data = len(data_loader)
        for batch_index, (sequence, label, missing_values) in enumerate(data_loader):
            sequence, label = self._to_gpu(sequence, label)
            # print progress
            self._print_progress(batch_index, len_data, job=VALIDATE_JOB)
            output = self._model(sequence)
            # calculate total loss
            loss_count += self._loss_func(output.squeeze(dim=1), label.float(), missing_values)  # calculate loss

            true_labels += label.tolist()
            pred += output.squeeze().tolist()

        # update loss accuracy
        loss = float(loss_count / len(data_loader))
        self._update_loss(loss, job=job)
        """
        self._update_accuracy(pred, true_labels, job=job)
        self._update_auc(pred, true_labels, job=job)
        """
        return loss


def run_experiment(X, y, missing_values, name, folder):
    score = open(os.path.join(folder, name + "_score.csv"), "wt")
    for k in range(1, 21):
        k *= 10
        microbiome_dataset = MicrobiomeDataset(X, y, missing_values)
        split_list = [0.8, 0.2]


        activator_params = ActivatorParams()
        activator = Activator(MicrobiomeModule(ModuleParams(lstm_out_dim=NUMBER_OF_BACTERIA)),
                                    activator_params, microbiome_dataset, split_microbiome_dataset)

        activator.train(validate_rate=10)

        score.write(str(k) + ",train_loss," + ",".join([str(v) for v in activator.loss_train_vec]) + "\n")
        # score.write(str(k) + "train_acc," + ",".join([str(v) for v in activator.accuracy_train_vec]) + "\n")
        # score.write(str(k) + "train_auc," + ",".join([str(v) for v in activator.auc_train_vec]) + "\n")
        score.write(str(k) + ",dev_loss," + ",".join([str(v) for v in activator.loss_dev_vec]) + "\n")
        # score.write(str(k) + "dev_acc," + ",".join([str(v) for v in activator.accuracy_dev_vec]) + "\n")
        # score.write(str(k) + "dev_auc," + ",".join([str(v) for v in activator.auc_dev_vec]) + "\n")

    activator.plot_line(name, folder)
    score.close()
    activator.plot_line(name, folder)


def run_nni_experiment(X, y, missing_values, params, name, folder):
    # for k in range(1, 21):
    #     k *= 10
    global NUMBER_OF_SAMPLES
    global NUMBER_OF_TIME_POINTS
    global NUMBER_OF_BACTERIA
    NUMBER_OF_SAMPLES = X.shape[0]
    NUMBER_OF_TIME_POINTS = X.shape[1]
    NUMBER_OF_BACTERIA = X.shape[2]

    score = open(os.path.join(folder, "NNI", name + "_score.csv"), "wt")
    microbiome_dataset = MicrobiomeDataset(X, y, missing_values)
    activator_params = ActivatorParams(TRAIN_TEST_SPLIT=params["TRAIN_TEST_SPLIT"],
                                       LOSS=params["LOSS"],
                                       BATCH_SIZE=params["BATCH_SIZE"],
                                       GPU=False,
                                       EPOCHS=params["EPOCHS"],
                                       EARLY_STOP=params["EARLY_STOP"])

    activator = Activator(MicrobiomeModule(ModuleParams(lstm_out_dim=NUMBER_OF_BACTERIA,
                                                        LSTM_LAYER_NUM=params["LSTM_LAYER_NUM"],
                                                        DROPOUT=params["DROPOUT"],
                                                        LEARNING_RATE=params["LEARNING_RATE"],
                                                        OPTIMIZER=params["OPTIMIZER"],
                                                        REGULARIZATION=params["REGULARIZATION"])),
                          activator_params, microbiome_dataset, split_microbiome_dataset)

    activator.train(validate_rate=1)

    score.write("train_loss," + ",".join([str(v) for v in activator.loss_train_vec]) + "\n")
    score.write("dev_loss," + ",".join([str(v) for v in activator.loss_dev_vec]) + "\n")
    activator.plot_line(name, os.path.join(folder, "NNI"))
    score.close()

    return activator.loss_dev_vec[-1]


def run_RNN(X, y, missing_values, name, folder):
    print(name)
    global NUMBER_OF_SAMPLES
    global NUMBER_OF_TIME_POINTS
    global NUMBER_OF_BACTERIA
    NUMBER_OF_SAMPLES = X.shape[0]
    NUMBER_OF_TIME_POINTS = X.shape[1]
    NUMBER_OF_BACTERIA = X.shape[2]
    run_experiment(X, y, missing_values, name, folder)



    """
    _ds = MicrobiomeDataset(X, y)
    _dl = DataLoader(dataset=_ds, batch_size=2)
    _binary_module = MicrobiomeModule(ModuleParams(lstm_out_dim=NUMBER_OF_BACTERIA))
    mse_list = []
    for _i, (_sequence, _label) in enumerate(_dl):
        _out = _binary_module(_sequence.float())
        mse = metrics.mean_squared_error(_out.detach().numpy().flat, _label.detach().numpy().flat)
        print(mse)
        mse_list.append(mse)

    print("average mse: " + str(np.mean(mse_list)))

    """