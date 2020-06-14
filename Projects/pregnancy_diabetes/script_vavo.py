import nni
import sys
import os
import pandas as pd
import warnings
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_curve, auc
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LearningMethods.nn_models import *
from LearningMethods.nn_learning_model import nn_main, nn_learn, create_optimizer_loss, generator
from LearningMethods.deep_learning_model import data_set, Plotter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

models_nn = {'relu_b':nn_2hl_relu_b_model, 'tanh_b':nn_2hl_tanh_b_model,
             'leaky_b':nn_2hl_leaky_b_model, 'sigmoid_b':nn_2hl_sigmoid_b_model,
             'relu_mul':nn_2hl_relu_mul_model, 'tanh_mul':nn_2hl_tanh_mul_model,
             'leaky_mul':nn_2hl_leaky_mul_model, 'sigmoid_mul':nn_2hl_sigmoid_mul_model}

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
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=params["test_size"], stratify=np.array(y))
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


def nn_learn(net, training_generator, validation_generator, criterion, optimizer, plot, params, training_len, validation_len, title):
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
            x = torch.tensor(x, device=device, dtype=torch.float)
            #weight_dict = {1: (1. / sum(y).float()).data.item(), 0: (1. / (len(y) - sum(y).float())).data.item()}
            optimizer.zero_grad()
            y_all = np.append(y_all, y)
            y=y.to(device)
            y_score, _ = net(x)
            y_score_all = np.append(y_score_all, y_score.data.cpu().numpy()[:, 0])
            # criterion = self.build_weighted_loss(y, weight_dict)
            loss = criterion(y_score, y.unsqueeze(dim=1).float())
            train_running_loss += loss.item()
            y_pred = [np.argmax(i) for i in y_score.data.cpu().numpy()]
            acc = accuracy_score(y_pred, y.data.cpu().numpy())
            train_running_acc.append(acc)
            loss.backward()
            optimizer.step()

        train_running_loss /= training_len
        train_running_acc = np.mean(np.array(train_running_acc))

        if i % 25 == 0:
            fpr, tpr, thresholds = roc_curve(y_all.astype(int), np.array(y_score_all))
            tarin_epoch_auc = auc(fpr, tpr)
            info = "TRAIN - epoch {:5}  loss: {:7}  auc: {:7} accuracy: {:7}".format(i, round(train_running_loss, 5),
                                                                            round(tarin_epoch_auc, 5),
                                                                           round(train_running_acc, 5))
            print(info)

        net.eval()
        test_running_loss = .0
        test_running_acc = []
        y_all = np.array([])
        y_score_all = np.array([])
        with torch.no_grad():
            for x, y in validation_generator:
                x = torch.tensor(x, device=device, dtype=torch.float)
                #weight_dict = {1: (len(y) / sum(y).float()).data.item(), 0: (len(y) / (len(y) - sum(y).float())).data.item()}
                y_all = np.append(y_all, y)
                y=y.to(device)
                y_score, _ = net(x)
                y_score_all = np.append(y_score_all, y_score.data.cpu().numpy()[:, 0])
                # criterion = self.build_weighted_loss(y, weight_dict)
                loss = criterion(y_score, y.unsqueeze(dim=1).float())
                test_running_loss += loss.item()
                y_pred = [np.argmax(i) for i in y_score.data.cpu().numpy()]
                acc = accuracy_score(y_pred, y.data.cpu().numpy())
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

    info = "TRAIN - epoch {:5}  loss: {:7}  auc: {:7} accuracy: {:7}".format(i, round(train_running_loss, 5),
                                                                             round(tarin_epoch_auc, 5),
                                                                            round(train_running_acc, 5))
    print(info)
    info = "TEST  - epoch {:5}  loss: {:7}  auc: {:7} accuracy: {:7}".format(i, round(test_running_loss, 5),
                                                                             round(epoch_auc, 5),
                                                                             round(test_running_acc, 5))
    print(info)

    return epoch_auc, test_running_acc, net


def nn_main(X_train, y_train, X_test, y_test, plot=False, k_fold = 5):
    auc_mean = 0
    acc_mean = 0
    params = {
        "model": 'sigmoid_b',
        "hid_dim_0": 40,
        "hid_dim_1": 15,
        "reg": 0.73,
        "dims": [20, 40, 60, 2],
        "lr": 0.05,
        "test_size": 0.2,
        "batch_size": 8,
        "shuffle": 1,
        "num_workers": 4,
        "epochs": 250,
        "optimizer": 'SGD',
        "loss": 'BCE'
    }
    training_len = len(X_train)
    validation_len = len(X_test)
    second_Net = models_nn[params["model"]]
    for i in range(k_fold):
        net_second = second_Net(list(X_train.shape)[1], int(params["hid_dim_0"]), int(params["hid_dim_1"]), 1).to(device)
        criterion_second_model, optimizer_second_model = create_optimizer_loss(net_second, params, device)
      
        X_train = torch.from_numpy(np.array(X_train))
        X_test = torch.from_numpy(np.array(X_test))
        y_train = torch.from_numpy(np.array(y_train)).long()
        y_test = torch.from_numpy(np.array(y_test)).long()
      
        training_generator_ens = generator(X_train, y_train, params)
        validation_generator_ens = generator(X_test, y_test, params)
      
        epoch_auc, test_running_acc, net = nn_learn(net_second, training_generator_ens, validation_generator_ens, criterion_second_model, optimizer_second_model, plot, params, training_len, validation_len, 'hi')
        auc_mean += epoch_auc

    print(net)
    return auc_mean/k_fold, acc_mean/k_fold


if __name__ == "__main__":
    df_concat = pd.read_csv('check_csv_concatenate_train.csv')
    df_concat = df_concat.set_index('ID')
    X_train = df_concat.iloc[:, :-1]
    y_train = df_concat.iloc[:, -1]
    
    df_concat = pd.read_csv('check_csv_concatenate_val.csv')
    df_concat = df_concat.set_index('ID')
    X_val = df_concat.iloc[:, :-1]
    y_val = df_concat.iloc[:, -1]
    
    nn_main(X_train, y_train, X_val, y_val, plot=False, k_fold = 5)
