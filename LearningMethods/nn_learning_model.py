import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
import warnings
from LearningMethods.deep_learning_model import data_set, Plotter

warnings.filterwarnings("ignore")


def create_optimizer_loss(net, params, device):

    if params["optimizer"] == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=params["lr"], weight_decay=params["reg"])
    elif params["optimizer"] == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=params["lr"])
    elif params["optimizer"] == "Adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=params["lr"], weight_decay=params["reg"])
    else:
        optimizer = optim.SGD(net.parameters(), lr=params["lr"])

    if params["loss"] == "BCE":
        criterion = nn.BCELoss().to(device)
    elif params["loss"] == "MSE":
        criterion = nn.MSELoss().to(device)
    else:
        criterion = nn.BCELoss().to(device)
    return criterion, optimizer


def create_train_test(X, y,params):
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=params["test_size"],
                                                        random_state=42)
    training_len = len(X_train)
    validation_len = len(X_test)

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    return X_train, X_test, y_train, y_test, training_len, validation_len


def generator(X_arr, y_arr,  params):
    net_params = {'batch_size': params["batch_size"],
                  'shuffle': params["shuffle"],
                  'num_workers': params["num_workers"]}
    set = data_set(X_arr, y_arr)
    generator = DataLoader(set, **net_params)

    return generator


def nn_main(X, y, params, title, Net, plot=False ):
    if torch.cuda.is_available():
        device = torch.device('cuda')  # maybe 'cuda:0' or any index
    else:
        device = torch.device('cpu')

    net = Net(list(X.shape)[1], int(params["hid_dim_0"]), int(params["hid_dim_1"]), 1).to(device)

    criterion, optimizer = create_optimizer_loss(net, params, device)

    X_train, X_test, y_train, y_test, training_len, validation_len = create_train_test(X, y, params)

    training_generator = generator(X_train, y_train, params)
    validation_generator = generator(X_test, y_test, params)

    if plot:
        plotter = Plotter(title=title, save_to_filepath=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    "loss_plot.svg"), param_dict=params)

    for i in range(int(params["epochs"])):
        net.train()
        train_running_loss = .0
        train_running_acc = []
        y_all = np.array([])
        y_score_all = np.array([])
        for x, y in training_generator:
            weight_dict = {1: (1. / sum(y).float()).data.item(), 0: (1. / (len(y) - sum(y).float())).data.item()}
            optimizer.zero_grad()
            y_all = np.append(y_all, y)
            y_score = net(x.float())
            y_score_all = np.append(y_score_all, y_score.detach().numpy()[:, 0])
            # criterion = self.build_weighted_loss(y, weight_dict)
            loss = criterion(y_score, y.unsqueeze(dim=1).float())
            train_running_loss += loss.item()
            y_pred = [np.argmax(i) for i in y_score.detach().numpy()]
            acc = accuracy_score(y_pred, y.detach().numpy())
            train_running_acc.append(acc)
            loss.backward()
            optimizer.step()

        train_running_loss /= training_len
        train_running_acc = np.mean(np.array(train_running_acc))

        if i % 25 == 0:
            # fpr, tpr, thresholds = roc_curve(y_all.astype(int), np.array(y_score_all))
            # epoch_auc = auc(fpr, tpr)
            info = "TRAIN - epoch {:5}  loss: {:7}  accuracy: {:7}".format(i, round(train_running_loss, 5),
                                                                           round(train_running_acc, 5))
            print(info)

        net.eval()
        test_running_loss = .0
        test_running_acc = []
        y_all = np.array([])
        y_score_all = np.array([])
        with torch.no_grad():
            for x, y in validation_generator:
                weight_dict = {1: (len(y) / sum(y).float()).data.item(), 0: (len(y) / (len(y) - sum(y).float())).data.item()}
                y_all = np.append(y_all, y)
                y_score = net(x.float())
                y_score_all = np.append(y_score_all, y_score[:, 0])
                # criterion = self.build_weighted_loss(y, weight_dict)
                loss = criterion(y_score, y.unsqueeze(dim=1).float())
                test_running_loss += loss.item()
                y_pred = [np.argmax(i) for i in y_score.detach().numpy()]
                acc = accuracy_score(y_pred, y.detach().numpy())
                test_running_acc.append(acc)

            test_running_loss /= validation_len
            test_running_acc = np.mean(np.array(test_running_acc))
            if i % 25 == 0:
                fpr, tpr, thresholds = roc_curve(y_all.astype(int), np.array(y_score_all))
                epoch_auc = auc(fpr, tpr)
                info = "TEST  - epoch {:5}  loss: {:7}  auc: {:7} accuracy: {:7}".format(i, round(test_running_loss, 5),
                                                                                         round(epoch_auc, 5),
                                                                                         round(test_running_acc, 5))
                print(info)
            if plot:
                plotter.add_values(i, loss_train=train_running_loss, acc_train=train_running_acc,
                        loss_test=test_running_loss, acc_test=test_running_acc)
    if plot:
        plotter.plot()

    info = "TRAIN - epoch {:5}  loss: {:7}  accuracy: {:7}".format(i, round(train_running_loss, 5),                                                               round(train_running_acc, 5))
    print(info)
    info = "TEST  - epoch {:5}  loss: {:7}  auc: {:7} accuracy: {:7}".format(i, round(test_running_loss, 5),
                                                                             round(epoch_auc, 5),
                                                                             round(test_running_acc, 5))
    print(info)
    print(net)
    return epoch_auc, test_running_acc
