import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, List,Iterable


class learning_model(nn.Module):
    """A simple NN which receives as an input an iterable of integers which describes the layers structure except the
    output layer. Additionally, the constructor gets the output size of the model in order to construct the output layer
     and the activation function which will be used in ll internal layers."""
    def __init__(self, structure_list: Iterable[int], out_size:int, activation_fn=torch.tanh):
        super(learning_model, self).__init__()
        # construct the layers of the model.
        self.linears = nn.ModuleList(
            [nn.Linear(structure_list[i], structure_list[i + 1]) for i in range(0, len(structure_list) - 1)])
        # create the output layer.
        self.out = nn.Linear(structure_list[-1], out_size)
        self.activation_fn = activation_fn
    """The forward function, forwards the input while using the activation function in all layers, except the last 
    one. """
    def forward(self, input):
        for layer in self.linears:
            input = self.activation_fn(layer(input))
        output = self.out(input)
        return output

    def predict(self, x, threshold=0.5):
        """
        x is the result received after the forward function.
        the function applies softmax on x and then decides its label according to the result.
        if threshold is not given, the label is decided based on the dimension with the highest probability.
        otherwise, the prediction is considered to be binary, and is been made according to the threshold.
        """
        # Apply softmax to output.
        is_binary = self.out.out_features == 2
        # dim=1 solved me an issue, I think its suits our datasets, but if not check the documentation and alter it.
        pred = F.softmax(x,dim=1)
        ans = []
        # Pick the class with maximum weight
        for t in pred:
            if is_binary:
                if t[1] >= threshold:
                    ans.append(1)
                else:
                    ans.append(0)
            else:
                ans.append(t.argmax())
        return ans

    def predict_prob(self, input):
        """A function that runs the input through the net and provides the probabilities of each class."""
        # dim=1 solved me an issue, I think its suits our datasets, but if not check the documentation and alter it.

        return F.softmax(self(input),dim=1)


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(yhat, y)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


def get_weights_out_of_target(target_series, weights_fn=None):
    if weights_fn is None:
        weights_fn = lambda input: 1 / input
    target_unique_elements = sorted(list(target_series.unique()))
    quantity_of_target_unique_elements = [list(target_series).count(i) for i in target_unique_elements]
    normedWeights = list(map(weights_fn, quantity_of_target_unique_elements))
    normedWeights = torch.FloatTensor(normedWeights)
    return normedWeights


def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def best_threshold(precision_arr, recall_arr, thresholds_arr, index_function=None, maximize=True):
    if index_function is None:
        index_function = calculate_f1_score
    index_values = [index_function(precision, recall) for precision, recall in zip(precision_arr, recall_arr)]
    if maximize:
        best_value = max(index_values)
    else:
        best_value = min(index_values)
    return thresholds_arr[index_values.index(best_value)], best_value


def early_stopping(history, patience=2, ascending=True):
    if len(history) <= patience:
        return False
    if ascending:
        return history[-patience - 1] == max(history[-patience - 1:])
    else:
        return history[-patience - 1] == min(history[-patience - 1:])
