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
from dafna.plot_auc import roc_auc
from dafna.plot_loss import LossAccPlotter


class Net(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        """
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 3)
        """
        self.fc1 = nn.Linear(in_dim, mid_dim_1)
        self.fc2 = nn.Linear(mid_dim_1, mid_dim_2)
        self.fc3 = nn.Linear(mid_dim_2, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class data_set(Dataset):
    def __init__(self, X, y):
        self._X = X
        self._y = y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        return self._X[idx], self._y[idx]


def nn_main(X, y, title, folder, in_dim, mid_dim_1, mid_dim_2, out_dim, lr=0.05, test_size=0.2, batch_size=8, shuffle=True,
            num_workers=4, epochs=20):
    # make general, send params from file!
    net = Net(in_dim, mid_dim_1, mid_dim_2, out_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    plotter = LossAccPlotter(title=title, save_to_filepath=os.path.join(folder, "loss_acc_plot.svg"))
    # writer = SummaryWriter()

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
            y_score_all = np.append(y_score_all, y_score.detach().numpy()[:, 1])
            loss = criterion(y_score, y)
            train_running_loss += loss.item()
            y_pred = [np.argmax(i) for i in y_score.detach().numpy()]
            acc = accuracy_score(y_pred, y.detach().numpy())
            train_running_acc.append(acc)
            loss.backward()
            optimizer.step()

        train_running_loss /= training_len
        train_running_acc = np.mean(np.array(train_running_acc))
        # writer.add_scalar('Loss/train', running_loss, i)
        # writer.add_scalar('Accuracy/train', running_acc, i)

        if i % 50 == 0:
            fpr, tpr, thresholds = roc_curve(y_all.astype(int), np.array(y_score_all))
            epoch_auc = auc(fpr, tpr)
            print("TRAIN - epoch {:5}  loss: {:7}  accuracy: {:7}  auc: {:7}".format(i, round(train_running_loss, 5),
                                                                   round(train_running_acc, 5), round(epoch_auc, 5)))

        net.eval()
        test_running_loss = .0
        test_running_acc = []
        y_all = np.array([])
        y_score_all = np.array([])
        with torch.no_grad():
            for x, y in validation_generator:
                y_all = np.append(y_all, y)
                y_score = net(x.float())
                y_score_all = np.append(y_score_all, y_score[:, 1])
                loss = criterion(y_score, y)  #torch.tensor(y))
                test_running_loss += loss.item()
                y_pred = [np.argmax(i) for i in y_score.detach().numpy()]
                acc = accuracy_score(y_pred, y.detach().numpy())
                test_running_acc.append(acc)

            test_running_loss /= validation_len
            test_running_acc = np.mean(np.array(test_running_acc))
            # writer.add_scalar('Loss/test', running_loss, i)
            # writer.add_scalar('Accuracy/test', running_acc, i)
            if i % 50 == 0:
                fpr, tpr, thresholds = roc_curve(y_all.astype(int), np.array(y_score_all))
                epoch_auc = auc(fpr, tpr)
                print("TEST  - epoch {:5}  loss: {:7}  accuracy: {:7}  auc: {:7}".format(i, round(test_running_loss, 5),
                                                                      round(test_running_acc, 5), round(epoch_auc, 5)))
            plotter.add_values(i, loss_train=train_running_loss, acc_train=train_running_acc,
                               loss_val=test_running_loss, acc_val=test_running_acc)


    roc_auc(y_all.astype(int), y_score_all, visualize=True, graph_title='ROC Curve\n epoch ' + str(i), save=True,
            folder=folder, fontsize=17)

        # writer.add_scalars('Loss', {'Train-loss': train_running_loss, 'Test-loss': test_running_loss}, i)
        # writer.add_scalars('Accuracy', {'Train-accuracy': train_running_acc, 'Test-accuracy': test_running_acc}, i)

        # writer.add_scalar('Loss/test', np.random.random(), n_iter)
        # writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    plotter.block()
    plotter.fig.savefig(os.path.join(folder, "nn_summary_plot.svg"), bbox_inches="tight", format='svg')
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