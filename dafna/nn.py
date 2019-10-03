import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_curve, auc
# from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from dafna.plot_auc import roc_auc
from dafna.plot_loss import LossAccPlotter
import pickle
import warnings
warnings.filterwarnings("ignore")


class Best_Net_Info():
    def __init__(self, folder, in_dim, mid_dim_1, mid_dim_2, out_dim, lr, test_size, batch_size, shuffle, epochs):
        self.description = ("in_dim:" + str(in_dim) + "\n" + "mid_dim_1:" + str(mid_dim_1) + "\n" + "mid_dim_2:" + str(mid_dim_2) +
               "\n" + "out_dim:" + str(out_dim) + "\n" + "lr:" + str(lr) + "\n" + "test_size:" + str(test_size) +
               "\n" + "batch_size:" + str(batch_size) + "\n" + "shuffle:" + str(shuffle) + "\n" + "epochs:"
               + str(epochs) + "\n")

        with open("best_model_description" + self.description.replace("\n", "_") + " .txt", "w") as f:
            f.write(self.description)

        self.folder = folder
        self.best_model = None
        self.best_train_loss = np.Inf
        self.best_train_acc = np.Inf
        self.best_test_loss = np.Inf
        self.best_test_acc = np.Inf
        self.y = None
        self.y_score = None

    def check_if_better(self, model, best_train_loss, best_train_acc, best_test_loss, best_test_acc, y, y_score):
        if best_train_loss < self.best_train_loss and best_test_loss < self.best_test_loss:
            # update
            self.best_model = model
            self.best_train_loss = best_train_loss
            self.best_test_loss = best_test_loss
            self.best_train_acc = best_train_acc
            self.best_test_acc = best_test_acc
            self.y = y
            self.y_score = y_score

    def save_model_state(self):
        torch.save(self.best_model.state_dict(), os.path.join(self.folder, self.description.replace("\n", "_") + ".pt"))

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
        x = self.fc3(x)
        return x  #F.softmax(x, dim=-1) for multi class


class data_set(Dataset):
    def __init__(self, X, y):
        self._X = X
        self._y = y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        return self._X[idx], self._y[idx]


def nn_main(X, y, title, folder, in_dim, mid_dim_1, mid_dim_2, out_dim, lr=0.001, test_size=0.2, batch_size=4, shuffle=True,
            num_workers=4, epochs=500):
    # make general, send params from file!
    param_dict = { "lr": lr, "test_size": test_size, "batch_size": batch_size, "shuffle": shuffle,
            "num_workers": num_workers, "epochs": epochs}
    net = Net(in_dim, mid_dim_1, mid_dim_2, out_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1)
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    plotter = Plotter(title=title, save_to_filepath=os.path.join(folder, "loss_plot.svg"), param_dict=param_dict)

    # one_hot_mat = np.zeros((y.shape[0], len(set(y))))
    # one_hot_mat[np.arange(y.shape[0]), y] = 1

    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=42)
    training_len = len(X_train)
    validation_len = len(X_test)

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'num_workers': num_workers}

    training_set = data_set(X_train, y_train)
    training_generator = DataLoader(training_set, **params)

    validation_set = data_set(X_test, y_test)
    validation_generator = DataLoader(validation_set, **params)

    best_net_info = Best_Net_Info(folder, in_dim, mid_dim_1, mid_dim_2, out_dim, lr, test_size, batch_size, shuffle, epochs)
    for i in range(epochs):
        net.train()
        train_running_loss = .0
        train_running_acc = []
        y_all = np.array([])
        y_score_all = np.array([])
        for x, y in training_generator:
            optimizer.zero_grad()
            y_all = np.append(y_all, y)
            y_score = net(x.float())
            y_score_all = np.append(y_score_all, y_score.detach().numpy()[:, 0])
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
            #epoch_auc = auc(fpr, tpr)
            print("TRAIN - epoch {:5}  loss: {:7}  accuracy: {:7}".format(i, round(train_running_loss, 5),
                                                                   round(train_running_acc, 5)))  #, round(epoch_auc, 5)))

        net.eval()
        test_running_loss = .0
        test_running_acc = []
        y_all = np.array([])
        y_score_all = np.array([])
        with torch.no_grad():
            for x, y in validation_generator:
                y_all = np.append(y_all, y)
                y_score = net(x.float())
                y_score_all = np.append(y_score_all, y_score[:, 0])
                loss = criterion(y_score, y.unsqueeze(dim=1).float())
                test_running_loss += loss.item()
                y_pred = [np.argmax(i) for i in y_score.detach().numpy()]
                acc = accuracy_score(y_pred, y.detach().numpy())
                test_running_acc.append(acc)

            test_running_loss /= validation_len
            test_running_acc = np.mean(np.array(test_running_acc))
            if i % 25 == 0:
                # fpr, tpr, thresholds = roc_curve(y_all.astype(int), np.array(y_score_all))
                # epoch_auc = auc(fpr, tpr)
                print("TEST  - epoch {:5}  loss: {:7}  accuracy: {:7}".format(i, round(test_running_loss, 5),
                                                                      round(test_running_acc, 5)))  #  auc: {:7}, round(epoch_auc, 5)))

            best_net_info.check_if_better(net, train_running_loss, train_running_acc, test_running_loss, test_running_acc, y_all, y_score_all)


            plotter.add_values(i, loss_train=train_running_loss, acc_train=train_running_acc,
                               loss_test=test_running_loss, acc_test=test_running_acc)

    #roc_auc(y_all.astype(int), y_score_all, visualize=True, graph_title='ROC Curve\n epoch ' + str(i), save=True,
    #        folder=folder, fontsize=17)

    best_net_info.save_model_state()
    plotter.plot()
    plotter.plot_y_diff_plot("Real Tumer Sizes vs. Predicted Sizes (test set)", best_net_info.y, best_net_info.y_score)
    print(net)

"""
iris = datasets.load_iris()
X = iris.data
y = iris.target

if False:
    X_2 = iris.data[:, :2]  # we only take the first three features.
    plot_data_3d(X, X_2, y, data_name="Iris Data Set", save=True)

nn_main(X, y, 4, 10, 20, 3)
"""