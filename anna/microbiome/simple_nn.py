from load_merge_otu_mf import OtuMfHandler
from preprocess import preprocess_data
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
import numpy as np

#  2-layer neural network (single hidden layer)
import matplotlib.pyplot as plt

# define initial parameters
n_classes = 3
validation_precentage = 0.1
sigmoid = lambda x: 1 / (1 + np.exp(-x))
rellu = lambda x: np.maximum(x, 0)
softmax = lambda x: np.exp(x) / np.sum(np.exp(x))


def relu_derivative(x):
    return (x > 0) * 1


# convert to one hot vector
def oneHot(vector, num_classes):
    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


# Forward propagation
def fprop(x, y, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    h1 = sigmoid(z1)
    # rellu
    # h1= rellu(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    loss = -np.dot(y.T, np.log(h2))
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


# Back propagation
def bprop(fprop_cache):
    x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    dz2 = h2 - y
    dW2 = np.dot(dz2, h1.T)
    db2 = dz2
    dz1 = np.dot(fprop_cache['W2'].T,
                 dz2) * sigmoid(z1) * (1 - sigmoid(z1))
    # rellu defivation
    # dz1 = np.dot(fprop_cache['W2'].T,
    #             dz2) * relu_derivative(z1)
    dW1 = np.dot(dz1, x.T)
    db1 = dz1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


# shuffle 2 vectors
def shuffled_copies(x, y):
    t = np.random.permutation(len(x))
    return x[t], y[t]


# the main function that builds the model
def build_model(train_x, train_y, validation_x, validation_y, hidden_layer_size, batch_size, num_epochs, learning_rate,
                n_classes=3, val=True, plot_loss_graph=True):
    W1 = np.zeros((hidden_layer_size, train_x.shape[1]))
    b1 = np.random.rand(hidden_layer_size, 1)
    W2 = np.zeros((n_classes, hidden_layer_size))
    b2 = np.random.rand(n_classes, 1)
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    # parameters for graph
    loss_list = []
    loss_val_list = []
    epochs = []
    # loop for epochs
    for num_epoch in range(num_epochs):
        train_x, train_y = shuffled_copies(train_x, train_y)
        loss = 0
        for k in range(int(train_x.shape[0] / batch_size)):
            from_batch = k * batch_size
            to_batch = (k + 1) * batch_size
            batch_x = train_x[from_batch:to_batch]
            batch_y = train_y[from_batch:to_batch]
            bprop_dict = {'b1': 0, 'W1': 0, 'b2': 0, 'W2': 0}
            # loop for batches
            for b_x, b_y in zip(batch_x, batch_y):
                fprop_cache = fprop(b_x, b_y, params)
                loss = loss + fprop_cache['loss']
                bprop_cache = bprop(fprop_cache)
                print(params)
                for key in bprop_cache:
                    bprop_dict[key] = bprop_dict[key] + bprop_cache[key] / float(batch_size)
            for key in params:
                params[key] = params[key] - learning_rate * bprop_dict[key]
            # to check on validation set
        if val:
            count = 0
            print(count)
            loss_val = 0
            for v_x, v_y in zip(validation_x, validation_y):
                fprop_cache = fprop(v_x, v_y, params)
                loss_val = loss_val + fprop_cache['loss']
                label = np.argmax(v_y)
                y_hat = np.argmax(fprop_cache['h2'])
                if label == y_hat:
                    count += 1
            print(count)
            print(len(validation_y))
            print("validation accuracy %f" %(count/len(validation_y)))
            loss_list.append(loss[0][0] / train_x.shape[0])
            loss_val_list.append(loss_val[0][0] / validation_x.shape[0])
            epochs.append(num_epoch)
    # to plot loss function graph
    if val and plot_loss_graph:
        x2, y2 = zip(*(zip(epochs, loss_val_list)))
        x3, y3 = zip(*(zip(epochs, loss_list)))
        plt.plot(x2, y2, lw=2, color='blue', label='Validation')
        plt.plot(x3, y3, lw=2, color='red', label='Train')
        plt.legend(loc=1, ncol=1)
        plt.title('Loss per epoch')
        plt.xlim(0, num_epochs - 1)
        plt.show()
    return params


# return model's prediction
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = np.dot(W1, x) + b1
    h1 = sigmoid(z1)
    # h1=rellu(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    return np.argmax(h2)


# load data
train_x = X.values
train_y = y.values.astype(int)
train_y = oneHot(train_y, n_classes)

x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
y = train_y.reshape(train_y.shape[0], train_y.shape[1], 1)

# define train and validation set
train_x, train_y = shuffled_copies(x, y)
validation_x = train_x[int((1 - validation_precentage) * len(train_x)):]
validation_y = train_y[int((1 - validation_precentage) * len(train_y)):]
train_x = train_x[:int((1 - validation_precentage) * len(train_x))]
train_y = train_y[:int((1 - validation_precentage) * len(train_y))]

# build_model(train_x, train_y,validation_x,validation_y, 100, 32, 120, 0.05,10,True,False)
model = build_model(train_x, train_y, validation_x, validation_y, 20, 8, 100, 0.05, 3, True, True)
#test_x = np.loadtxt("test_x")
#tst_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)
#test_x = tst_x / 255.0
#result = []

# run test examples
#for d in test_x:
#    result.append(predict(model, d))
#f = open('test.pred', 'w+')
#for elem in result:
#    f.write(str(elem) + "\n")
#f.close()
