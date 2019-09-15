from sys import stdout
import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
from torch.optim import RMSprop
from torch.utils.data import DataLoader, WeightedRandomSampler

from allergy.allergy_data_loader import AllergyDataLoader


class NeuralNetParams(): 
    def __init__(self):
        self.INPUT_DIM = 20
        self.H1_DIM = 100
        self.OUTPUT_DIM = 1
        self.LR = 1e-3
        self.ACTIVATION = torch.tanh
        self.OPTIMIZER = RMSprop


class NeuralNet(nn.Module):
    def __init__(self, params: NeuralNetParams):
        super(NeuralNet, self).__init__()
        self._params = params
        self._input = nn.Linear(params.INPUT_DIM, params.H1_DIM)
        self._layer1 = nn.Linear(params.H1_DIM, params.OUTPUT_DIM)
        self._activation = params.ACTIVATION
        # set optimizer
        self.optimizer = self.set_optimizer()

    # init optimizer with RMS_prop
    def set_optimizer(self):
        return self._params.OPTIMIZER(self.parameters(), lr=self._params.LR)

    def forward(self, x):  # לדבג ולסדר מימדים
        x = self._input(x)
        x = torch.tanh(x)
        x = self._layer1(x)
        x = torch.sigmoid(x)  # if multi class adjust din and change to softmax
        return x


if __name__ == '__main__':

    task = 'success task'
    allergy_dataset = AllergyDataLoader(task, False, False, False)
    if task == 'success task':
        target = list(allergy_dataset.get_id_to_binary_success_tag_map.values())

    print('target train 0/1: {}/{}'.format(
        len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))

    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # target = torch.from_numpy(target).long()
    # train_dataset = torch.utils.data.TensorDataset(data, target)

    data_loader = DataLoader(
        allergy_dataset,
        batch_size=64, sampler=sampler  # shuffle=True
    )

    for i, (data, target) in enumerate(data_loader):
        print
        "batch index {}, 0/1: {}/{}".format(
            i,
            len(np.where(np.array(target) == 0)[0]),
            len(np.where(np.array(target) == 1)[0]))

    for batch_index, (data, label) in enumerate(data_loader):
        stdout.write("\r\r\r%d" % int(100 * ((batch_index + 1) / len(data_loader))) + "%")
        stdout.flush()

    nn = NeuralNet(NeuralNetParams())
    for batch_index, (data, label) in enumerate(data_loader):
        stdout.write("\r\r\r%d" % int(100 * ((batch_index + 1) / len(data_loader))) + "%")
        stdout.flush()
        x = nn(data)
        print(x)

    # dl = DataLoader(os.path.join("..", "data", "pos", "train"), suf_pref=True)
    # voc_size = dl.vocab_size  # 100232
    # pre_size = dl.vocabulary.len_pref()
    # suf_size = dl.vocabulary.len_suf()
    # embed_dim = 50
    # out1 = int(dl.win_size * embed_dim * 0.66)
    # out2 = int(dl.win_size * embed_dim * 0.33)
    # out3 = int(dl.pos_dim)
    # layers_dimensions = (dl.win_size, out1, out2, out3)
    # NN = PrefSufNet(layers_dimensions, voc_size, pre_size, suf_size, embedding_dim=embed_dim)
    # x, p, s, l = dl.__getitem__(0)
    # NN(x, p, s)
    # e = 0
