from load_merge_otu_mf import OtuMfHandler
from Preprocess import preprocess_data
from pca import *
from plot_confusion_matrix import *
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

otu = 'C:/Users/Anna/Desktop/docs/otu_psc2.csv'
mapping = 'C:/Users/Anna/Desktop/docs/mapping_psc.csv'
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
#visualize_pca(preproccessed_data)

otu_after_pca = apply_pca(preproccessed_data, n_components=30)
merged_data = otu_after_pca.join(OtuMf.mapping_file['DiagnosisGroup'])

merged_data.fillna(0)

mapping_disease_for_labels = {'Control':0,'Cirrhosis/HCC':1, 'PSC/PSC+IBD':2}
mapping_disease = {'Control':0,'Cirrhosis ':1, 'HCC':1, 'PSC+IBD':2,'PSC':2}
merged_data['DiagnosisGroup'] = merged_data['DiagnosisGroup'].map(mapping_disease)
merged_data = merged_data.join(OtuMf.mapping_file[['Age', 'BMI', 'FattyLiver','RegularExercise', 'Smoking']])
mappin_boolean = {'yes' :1, 'no': 0, 'Control': 0, '0':0, '1':1}
merged_data['FattyLiver'] = merged_data['FattyLiver'].map(mappin_boolean)
merged_data['RegularExercise'] = merged_data['RegularExercise'].map(mappin_boolean)
merged_data['Smoking'] = merged_data['Smoking'].map(mappin_boolean)
#print(merged_data.tail())

X,y = merged_data.loc[:, merged_data.columns != 'DiagnosisGroup'], merged_data['DiagnosisGroup']

train_target = torch.tensor(y.values.astype(np.float32))
train = torch.tensor(X.values.astype(np.float32))
train_tensor = torch.utils.data.TensorDataset(train, train_target)

num_train = len(X)
indices = list(range(num_train))
split = int(num_train*0.1)
validation_idx = np.random.choice(indices, size=split, replace=False)
train_idx = list(set(indices) - set(validation_idx))
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

train_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size =8, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size =8,  sampler=validation_sampler)
#test_target = torch.tensor(y_test.values.astype(np.float32))
#test = torch.tensor(X_test.values.astype(np.float32))
#test_tensor = torch.utils.data.TensorDataset(test, test_target)
#test_loader = torch.utils.data.DataLoader(dataset = test_tensor, batch_size = 8, shuffle = True)

print (num_train)
print (len(validation_idx))
print (len(train_idx))
print(len(train_loader))
print(len(validation_loader))

class FirstNet(nn.Module):
    def __init__(self,size):
        super(FirstNet, self).__init__()
        self.size = size
        self.fc0 = nn.Linear(size, 20)
        self.fc1 = nn.Linear(20, 3)

    def forward(self, x):
        x = x.view(-1, self.size)
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
        #x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

model = FirstNet(X.shape[1])
print(model)
optimizer = optim.Adagrad(model.parameters(), lr=0.3)

def train(epoch, model):
    model.train()
    t_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        labels = labels.long()
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        #if batch_idx % 2 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_sampler),
        #               100. * batch_idx / len(train_sampler), loss.item()))
        t_loss += loss.item()
    return t_loss/len(train_loader), 100. * correct / len(train_sampler)

def validation():
    model.eval()
   # for name, param in model.named_parameters():
    #        print(name, param.data)
    val_loss = 0
    correct = 0
    for data, target in validation_loader:
        output = model(data)
        target = target.long()
        #validation_loss += F.nll_loss(output, target, size_average=False).item()
        val_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        val_loss /= len(validation_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_sampler),
        100. * correct / len(validation_sampler)))
    return val_loss, 100. * correct / len(validation_sampler)

epochs =[]
train_loss=[]
control_loss =[]
train_accuracy = []
test_accuracy = []

for epoch in range(1, 10):
    epochs.append(epoch)
    train(epoch,model)
    #train_loss.append(t_l)
    #train_accuracy.append(t_a)
    validation()
    #control_loss.append(validation()[0])

print('train loss ' + str(sum(train_loss) / float(len(train_loss))))
print('test loss ' + str(sum(control_loss) / float(len(control_loss))))
#print('test accuracy ' +  str(sum(train_accuracy) / float(len(control_loss))))
