import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import matplotlib
from aging.analysis import train_x_data, train_y_values, test_x_data, test_y_values

class FirstNet(nn.Module):
    def __init__(self, feature_size):
        super(FirstNet, self).__init__()
        self.feature_size = feature_size
        self.fc0 = nn.Linear(feature_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.feature_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.mse_loss(x, dim=1)


def train(model, optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, labels)
        loss.backward()
        optimizer.step()

def test_model(model, x, y):
    model.eval()
    test_loss = 0
    for data, target in zip(x, y):
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
    test_loss /= len(x)

    print('Validation set: Average loss: {:.4f}'.format(test_loss))
    return test_loss


def train_test(model, optimizer):
    train(model, optimizer)
    avg_train_loss = test_model(model, train_x_data, train_y_values)
    avg_test_loss = test_model(model, test_x_data, test_y_values)
    return avg_train_loss, avg_test_loss


models = [FirstNet(feature_size=train_x_data.shape())]
optimizers = [ optim.SGD(models[0].parameters(), lr=0.05)]
model_graph = [{'test': [], 'train': []}]

epoch_numbers = 10
for epoch in range(1, epoch_numbers + 1):
    print('\nepoch: {}'.format(epoch))
    for model in range(len(models)):
        print('---------\n '+ 'model ' + str(model+1) +'\n---------')
        avg_train_loss, avg_test_loss = train_test(models[model], optimizers[model])
        model_graph[model]['train'].append(avg_train_loss)
        model_graph[model]['test'].append(avg_test_loss)

x_values = list(range(1, epoch_numbers+1))
for model in range(len(models)):
    val_y_test = model_graph[model]['test']
    val_y_train = model_graph[model]['train']
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(x_values, val_y_test, label='Test')
    ax1.plot(x_values, val_y_train, label='Train')
    ax1.legend(['Validation', 'Train'])
    plt.title('Model '+ str(model+1))
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
plt.show()