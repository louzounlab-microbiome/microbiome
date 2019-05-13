
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

#normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('./data', train=True, download=True, transform=transforms),
                batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('./data', train=False, transform=transforms),
                batch_size=32, shuffle=True)

print (train_loader)
print (len(test_loader))

class FirstNet(nn.Module):
    def __init__(self,image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

model = FirstNet(image_size=28*28)
print(model)
optimizer = optim.SGD(model.parameters(), lr=0.02)
def train(epoch, model):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
for epoch in range(1, 10):
    train(epoch,model)
    test()