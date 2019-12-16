import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import Dataset, DataLoader
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")


class Net(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_dim, mid_dim_1)
        self.fc2 = nn.Linear(mid_dim_1, mid_dim_2)
        self.fc3 = nn.Linear(mid_dim_2, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x  #F.softmax(x, dim=-1) for multi class


class data_set(Dataset):
    def __init__(self, X, y):
        self._X = X
        self._y = y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        return self._X[idx], self._y[idx]
        
        
class Plotter():
    def __init__(self, title, save_to_filepath, param_dict):
        self. title = title
        self.save_to_filepath = save_to_filepath
        self.param = param_dict
        self.epochs = []
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_loss_list = []
        self.test_acc_list = []

    def add_values(self, i, loss_train, acc_train, loss_test, acc_test):
        self.epochs.append(i)
        self.train_loss_list.append(loss_train)
        self.train_acc_list.append(acc_train)
        self.test_loss_list.append(loss_test)
        self.test_acc_list.append(acc_test)

    def plot(self):
        plt.plot(self.train_loss_list)
        plt.plot(self.test_loss_list)
        # plt.plot(self.train_acc_list)
        # plt.plot(self.test_acc_list)
        title = 'Model Loss\n'
        for i, (key, val) in enumerate(self.param.items()):
            title = title + key.capitalize() + "=" + str(val) + " "
            if (i+1) % 3 == 0:
                title = title + "\n"

        plt.title(title)
        plt.xlabel('epoch')
        plt.xticks(range(0, len(self.epochs), int(len(self.epochs) / 5)))

        plt.legend(['train loss', 'test loss'], loc='upper left')  # 'train acc', 'test acc'], loc='upper left')
        plt.savefig(self.save_to_filepath, bbox_inches="tight", format='svg')
        plt.show()

    def plot_y_diff_plot(self, title, y, y_score):
        plt.figure(figsize=(5, 5))
        m = max(y.max(), y_score.max())
        data = {'a': np.array(y),
                'b': np.array(y_score)}
        plt.scatter('a', 'b', data=data)  # , c='c', s='d'
        plt.axis([-0.05, m + 0.05, -0.05, m + 0.05])
        plt.xlabel('real size')
        plt.ylabel('predicted size')
        plt.title(title)
        # plt.show()
        plt.savefig(title.replace(" ", "_") + ".svg", bbox_inches="tight", format='svg')


def nn_nni_main(X, y, params, title, folder):
    if torch.cuda.is_available():
        device = torch.device('cuda')  # maybe 'cuda:0' or any index
    else:
        device = torch.device('cpu')
    # params = in_dim, mid_dim_1, mid_dim_2, out_dim, lr=0.001, test_size=0.2, batch_size=4, shuffle=True,
    #             num_workers=4, epochs=500
    param_dict = {"lr": params["lr"], "test_size": params["test_size"], "batch_size": params["batch_size"], "shuffle": params["shuffle"],
            "num_workers": params["num_workers"], "epochs": params["epochs"]}
    input_size = list(X.shape)[1]
    net = Net(input_size, int(params["hid_dim_0"]), int(params["hid_dim_1"]), 1).to(device)

    def _build_weighted_loss(self, labels):
        weights_list = []
        for i in range(labels.shape[0]):
            weights_list.append(self._weights_dict[labels[i].data.item()])
        weights_tensor = torch.DoubleTensor(weights_list).to(self._device)
        self._criterion = torch.nn.BCELoss(weight=weights_tensor).to(self._device)
    

    if  params["optimizer"] == "Adam":
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
    plotter = Plotter(title=title, save_to_filepath=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "loss_plot.svg"),  param_dict=param_dict)

    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=params["test_size"], random_state=42)
    training_len = len(X_train)
    validation_len = len(X_test)

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    net_params = {'batch_size': params["batch_size"],
              'shuffle': params["shuffle"],
              'num_workers': params["num_workers"]}

    training_set = data_set(X_train, y_train)
    training_generator = DataLoader(training_set, **net_params)

    validation_set = data_set(X_test, y_test)
    validation_generator = DataLoader(validation_set, **net_params)
    
    def build_weighted_loss(labels, weights_dict):
        weights_list = []
        for i in range(labels.shape[0]):
            weights_list.append(weights_dict[labels[i].data.item()])
        weights_tensor = torch.DoubleTensor(weights_list).float().unsqueeze(dim=1)
        return torch.nn.BCELoss(weight=weights_tensor)
    

    for i in range(int(params["epochs"])):
        net.train()
        train_running_loss = .0
        train_running_acc = []
        y_all = np.array([])
        y_score_all = np.array([])
        for x, y in training_generator:
            weight_dict = {1: (1. / sum(y).float()).data.item(), 0: (1. / (len(y)-sum(y).float())).data.item()}
            optimizer.zero_grad()
            y_all = np.append(y_all, y)
            y_score = net(x.float())
            y_score_all = np.append(y_score_all, y_score.detach().numpy()[:, 0])
            #criterion = build_weighted_loss(y, weight_dict)
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
            #fpr, tpr, thresholds = roc_curve(y_all.astype(int), np.array(y_score_all))
            #epoch_auc = auc(fpr, tpr)
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
                weight_dict = {1: (len(y) / sum(y).float()).data.item(), 0: (len(y) / (len(y)-sum(y).float())).data.item()}
                y_all = np.append(y_all, y)
                y_score = net(x.float())
                y_score_all = np.append(y_score_all, y_score[:, 0])
                #criterion = build_weighted_loss(y, weight_dict)
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
                                                                      round(epoch_auc, 5),round(test_running_acc, 5))
                print(info)
            #plotter.add_values(i, loss_train=train_running_loss, acc_train=train_running_acc,
            #                   loss_test=test_running_loss, acc_test=test_running_acc)
    #plotter.plot()
    print(net)
    return epoch_auc, test_running_loss
