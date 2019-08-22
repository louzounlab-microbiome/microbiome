from allergies import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


#df, mapping_file = allergies(perform_distance=False,level =3)
df, mapping_file = allergies(perform_distance=True,level =5)
otu_after_pca, _ = apply_pca(df, n_components=29)
otu_after_pca = otu_after_pca.drop(otu_after_pca.columns[[6, 8, 9, 17, 18]], axis=1)
merged_data = otu_after_pca.join(mapping_file)

X = merged_data.drop(['AllergyType'], axis =1)
y = merged_data['AllergyType']

train_target = torch.tensor(y.values.astype(np.float32))
train = torch.tensor(X.values.astype(np.float32))
train_tensor = torch.utils.data.TensorDataset(train, train_target) #, shuffle=True)

num_train = len(X)
indices = list(range(num_train))
train_idx,validation_idx = train_test_split(indices,test_size=0.2, random_state=42)

#validation_idx = np.random.choice(indices, size=split, replace=False)
#train_idx = list(set(indices) - set(validation_idx))
print(validation_idx)
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

size = X.shape[1]
#train_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size =8,  sampler=train_sampler)
#validation_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size =8,  sampler=validation_sampler)
#test_target = torch.tensor(y_test.values.astype(np.float32))
#test = torch.tensor(X_test.values.astype(np.float32))
#test_tensor = torch.utils.data.TensorDataset(test, test_target)
#test_loader = torch.utils.data.DataLoader(dataset = test_tensor, batch_size = 8, shuffle = True)

# print (num_train)
# print (len(validation_idx))
# print (len(train_idx))
# print(len(train_loader))
# print(len(validation_loader))

class NNet(nn.Module):
    def __init__(self,size, dropout):
        super(NNet, self).__init__()
        n_features = size
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 60),
            #nn.BatchNorm1d(60, eps=1e-04, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(60, 10),
            #nn.BatchNorm1d(30, eps=1e-04, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        self.out = nn.Sequential(
            nn.Linear(10, n_out)
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
       # x = self.hidden2(x)
        x = self.out(x)
        return x


#print(model)
sig =nn.Sigmoid()
def train(epoch, model,optimizer):
    model.train()
    t_loss = 0
    correct = 0
    y_score =[]
    y_target =[]
    train_loader = torch.utils.data.DataLoader(dataset=train_tensor, batch_size=64, sampler=train_sampler)
    #validation_loader = torch.utils.data.DataLoader(dataset=train_tensor, batch_size=8, sampler=validation_sampler)
    for batch_idx, (data, labels) in enumerate(train_loader):
        if (np.sum(labels.data.numpy()==1)>0) and (np.sum(labels.data.numpy()==0)>0):
            pos_weight = np.sum(labels.data.numpy()==0)/np.sum(labels.data.numpy()==1)
            criterion =nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight))
        else:
            criterion = nn.BCEWithLogitsLoss()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels.view(-1, 1))
        #labels = labels.long()
        #loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        pred = torch.tensor(sig(output)).data.numpy().tolist()  # get the index of the max log-probability
        flat_list = [item for sublist in pred for item in sublist]
        #correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        y_score.append(flat_list)
        y_target.append(labels.data.numpy().tolist())
        #pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        #correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        #if batch_idx % 2 == 0:
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_sampler),
        #                100. * batch_idx / len(train_sampler), loss.item()))
        t_loss += loss.item()
    auc = metrics.roc_auc_score([item for sublist in y_target for item in sublist], [item for sublist in y_score for item in sublist])
    t_loss /= len(train_loader)

    #print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch,t_loss))
    return t_loss, auc

def validation(model):
    y_pred =[]
    y_label =[]
    model.eval()
   # for name, param in model.named_parameters():
    #        print(name, param.data)
    val_loss = 0
    correct = 0
    validation_loader = torch.utils.data.DataLoader(dataset=train_tensor, batch_size=32, sampler=validation_sampler)
    for data, target in validation_loader:
        #print(data)
        output = model(data)
        if (np.sum(target.data.numpy()==1)>0) and (np.sum(target.data.numpy()==0)>0):
            pos_weight = np.sum(target.data.numpy()==0)/np.sum(target.data.numpy()==1)
            criterion =nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight))
        else:
            criterion = nn.BCEWithLogitsLoss()
        #target = target.long()
        #validation_loss += F.nll_loss(output, target, size_average=False).item()
        val_loss += criterion(output, target.view(-1, 1)).item()  # sum up batch loss

        pred = torch.tensor(sig(output)).data.numpy().tolist()  # get the index of the max log-probability
        flat_list = [item for sublist in pred for item in sublist]
        # correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        y_pred.append(flat_list)
        y_label.append(target.data.numpy().tolist())
    auc = metrics.roc_auc_score([item for sublist in y_label for item in sublist], [item for sublist in y_pred for item in sublist])
    val_loss /= len(validation_loader)
    #print('\nTest set: Average loss: {:.4f}, AUC: {:.2f}\n'.format(
    #    val_loss,  auc))
    return val_loss, auc



def train_and_eval(size, lr, weight_decay,dropout=0.35):
    #optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
    model = NNet(size,dropout)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = []
    train_loss = []
    control_loss = []
    #train_accuracy = []
    test_accuracy = []
    prediction =[]
    data = []
    for epoch in range(1, 2500):
    #for data, label in validation_loader.dataset:
    #    print(label.data)
        epochs.append(epoch)
        t_l,_ =train(epoch,model, optimizer)
        train_loss.append(t_l)
        #train_accuracy.append(t_a)
        v_l,v_a =validation(model)
        # if v_a == 85:
        #     for data,target in train_tensor:
        #         #print(data)
        #         output = model(data)
        #         pred = output.data.max(1, keepdim=True)[1]
        #         prediction.append(pred)
        #         data.append(data)
        test_accuracy.append(v_a)
        control_loss.append(v_l)
        if len(test_accuracy)>500:
            m_t = max(test_accuracy)
            if all(i < m_t-0.02 for i in test_accuracy[-50:]):
                break

    for name, param in model.named_parameters():
        if name in ['hidden0.0.weight']:
            W0 = param.data
        elif name in ['hidden1.0.weight']:
            W1 = param.data
        elif  name in ['out.0.weight']:
            W2 = param.data
    res = np.dot(np.dot(W2, W1), W0)
    columns_to_delete = []
    for i in res[0]:
        if abs(i) < 0.1:
            columns_to_delete.append(np.where(res[0] == i)[0][0])
    print(columns_to_delete)

    print('done')

#print('train loss ' + str(sum(train_loss) / float(len(train_loss))))
#print('test loss ' + str(sum(control_loss) / float(len(control_loss))))
    m_t_a = max(test_accuracy)
    print('LR : {}, WD : {}, Dropout : {:.3f} test accuracy/ auc : {} '.format(lr, weight_decay, dropout,m_t_a))

# plot graphs
    x2,y2 = zip(*(zip(epochs,control_loss)))
    x3,y3 = zip(*(zip(epochs,train_loss)))
    font = {'size': 10}
    plt.rc('font', **font)
    plt.plot(x2,y2,lw=2, color='blue', label='Validation')
    plt.plot(x3,y3,lw=2, color='red', label='Train')
    plt.legend( loc=1,ncol=1)
    plt.title('LpE:lr- %s,weight_decay - %s, dropout - %s, acc - %s ' %(lr, weight_decay,dropout, m_t_a.item()))
    plt.xlim(0, 2500)
    #plt.ylim(0, 2)
    plt.show()

# for lr in range (4000, 0,-1000):
#     for wd in range(31,0,-10):
#         for do in range(20,55,5):
#             train_and_eval(size, lr/10000, wd/1000, do/100)

train_and_eval(size, 0.3,0.011, 0.35)

print('done')
