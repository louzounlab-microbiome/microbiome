import torch
import numpy as np
import torch.nn as nn
from LearningMethods.abstract_learning_model import AbstractLearningModel
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class DeepLearningModel(AbstractLearningModel):
    def __init__(self):
        super().__init__()

    def _build_weighted_loss(self, labels):
        weights_list = []
        for i in range(labels.shape[0]):
            weights_list.append(self._weights_dict[labels[i].data.item()])
        weights_tensor = torch.DoubleTensor(weights_list).to(self._device)
        self._criterion = torch.nn.BCELoss(weight=weights_tensor).to(self._device)



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
        self.title = title
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
            if (i + 1) % 3 == 0:
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

