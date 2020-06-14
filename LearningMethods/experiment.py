import torch
from torch.nn import Module, Linear, Conv1d
from torch.nn.functional import leaky_relu


class SequenceParams:
    def __init__(self, NN_input_dim, NN_hidden_dim_1, NN_hidden_dim_2, NN_output_dim, NN_LAYER_NUM, DROPOUT, timesteps):
        self.NN_input_dim = NN_input_dim
        self.NN_hidden_dim_1 = NN_hidden_dim_1
        self.NN_hidden_dim_2 = NN_hidden_dim_2
        self.NN_output_dim = NN_output_dim
        self.NN_layers = NN_LAYER_NUM
        self.NN_dropout = DROPOUT
        self.timesteps = timesteps


class SequenceModule(Module):
    def __init__(self, params):
        super(SequenceModule, self).__init__()
        self._layer_num = params.NN_layers
        self.timesteps = params.timesteps
        self.conv = Conv1d(2, self.timesteps, 1)
        if self._layer_num == 1:
            # HOW TO MAKE params.NN_layers a usable param?
            self.fc1 = Linear(params.NN_input_dim, params.NN_hidden_dim_1)
            self.fc2 = Linear(params.NN_hidden_dim_1, params.NN_output_dim)
        elif self._layer_num == 2:
            # HOW TO MAKE params.NN_layers a usable param?
            self.fc1 = Linear(params.NN_input_dim, params.NN_hidden_dim_1)
            self.fc2 = Linear(params.NN_hidden_dim_1, params.NN_hidden_dim_2)
            self.fc3 = Linear(params.NN_hidden_dim_2, params.NN_output_dim)

    def forward(self, x):
        # 3 layers NN
        x = self.conv(x)
        if self._layer_num == 1:
            x = leaky_relu(self.fc1(x))
            x = self.fc2(x)
        elif self._layer_num == 2:
            x = leaky_relu(self.fc1(x))
            x = leaky_relu(self.fc2(x))
            x = self.fc3(x)
        return x


def check():
    params = SequenceParams(100, 500, 50, 5, 2, 0.5, timesteps=10)
    model = SequenceModule(params)
    input = torch.rand([200, 2, 100])
    print(input.size())
    out = model(input)
    print(out.size())
    pass

check()