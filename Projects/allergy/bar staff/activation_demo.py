import torch.nn.functional as F
import random
import torch
from torch.autograd import Variable
from sys import stdout
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from allergy.allergy_data_loader import AllergyDataLoader
from allergy.nn_models import NeuralNet, NeuralNetParams
from allergy.loggers import *
import numpy as np


class AllergyActivatorParams:
    def __init__(self):
        self.LOSS = F.binary_cross_entropy
        self.BATCH_SIZE = 64
        self.GPU = True
        self.EPOCHS = 30
        self.VALIDATION_RATE = 20


class AllergyActivator:
    def __init__(self, model: NeuralNet, params: AllergyActivatorParams, data: AllergyDataLoader = None):
        self._model = model
        self._epochs = params.EPOCHS
        self._validation_rate = params.VALIDATION_RATE
        self._batch_size = params.BATCH_SIZE
        self._loss_func = params.LOSS
        self._load_data(data)
        self._loss_vec = []

    @property
    def model(self):
        return self._model

    def get_loss(self):
        return self._loss_vec

    # load dataset
    def _load_data(self, train_dataset):
        task = 'success task'
        if task == 'success task':
            target = list(train_dataset.get_id_to_binary_success_tag_map.values())

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

        # set validation loader
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            sampler=sampler
        )

    # train a model, input is the enum of the model type
    def train(self):
        self._loss_vec = []
        for epoch_num in range(self._epochs):
            # calc number of iteration in current epoch
            for batch_index, (data, label) in enumerate(self._train_loader):

                self._model.train()
                self._model.zero_grad()                         # zero gradients
                output = self._model(data)                      # calc output of current model on the current batch
                loss = self._loss_func(output, label.float())   # calculate loss
                loss.backward()                                 # back propagation
                self._model.optimizer.step()                    # update weights

                self._loss_vec.append(self._validate(self._train_loader))

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader):
        loss_count = 0
        self._model.eval()
        len_data = len(data_loader)
        for batch_index, (data, label) in enumerate(self._train_loader):
            stdout.write("\r\r\r%d" % int(100 * (batch_index + 1) / len_data) + "%")
            stdout.flush()

            output = self._model(data)
            # calculate total loss
            loss_count += self._loss_func(output, label.float())

        loss = float(loss_count / len(data_loader))
        return loss


if __name__ == '__main__':
    task = 'success task'
    allergy_dataset = AllergyDataLoader(task, False, False, False)
    activator = AllergyActivator(NeuralNet(NeuralNetParams()), AllergyActivatorParams(), allergy_dataset)
    activator.train()
    print(activator.get_loss())

