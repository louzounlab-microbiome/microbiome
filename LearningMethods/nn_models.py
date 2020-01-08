import torch.nn as nn
import torch.nn.functional as F


class nn_2hl_relu_b_model(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim):
        super(nn_2hl_relu_b_model, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_dim, mid_dim_1)
        self.fc2 = nn.Linear(mid_dim_1, mid_dim_2)
        self.fc3 = nn.Linear(mid_dim_2, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class nn_2hl_sigmoid_b_model(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim):
        super(nn_2hl_sigmoid_b_model, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_dim, mid_dim_1)
        self.fc2 = nn.Linear(mid_dim_1, mid_dim_2)
        self.fc3 = nn.Linear(mid_dim_2, out_dim)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


class nn_2hl_tanh_b_model(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim):
        super(nn_2hl_tanh_b_model, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_dim, mid_dim_1)
        self.fc2 = nn.Linear(mid_dim_1, mid_dim_2)
        self.fc3 = nn.Linear(mid_dim_2, out_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class nn_2hl_leaky_b_model(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim):
        super(nn_2hl_leaky_b_model, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_dim, mid_dim_1)
        self.fc2 = nn.Linear(mid_dim_1, mid_dim_2)
        self.fc3 = nn.Linear(mid_dim_2, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class nn_2hl_relu_mul_model(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim):
        super(nn_2hl_relu_mul_model, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_dim, mid_dim_1)
        self.fc2 = nn.Linear(mid_dim_1, mid_dim_2)
        self.fc3 = nn.Linear(mid_dim_2, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class nn_2hl_sigmoid_mul_model(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim):
        super(nn_2hl_sigmoid_mul_model, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_dim, mid_dim_1)
        self.fc2 = nn.Linear(mid_dim_1, mid_dim_2)
        self.fc3 = nn.Linear(mid_dim_2, out_dim)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class nn_2hl_tanh_mul_model(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim):
        super(nn_2hl_tanh_mul_model, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_dim, mid_dim_1)
        self.fc2 = nn.Linear(mid_dim_1, mid_dim_2)
        self.fc3 = nn.Linear(mid_dim_2, out_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class nn_2hl_leaky_mul_model(nn.Module):
    def __init__(self, in_dim, mid_dim_1, mid_dim_2, out_dim):
        super(nn_2hl_leaky_mul_model, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_dim, mid_dim_1)
        self.fc2 = nn.Linear(mid_dim_1, mid_dim_2)
        self.fc3 = nn.Linear(mid_dim_2, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x