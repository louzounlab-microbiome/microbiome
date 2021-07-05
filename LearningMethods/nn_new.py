import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, List, Iterable
import pytorch_lightning as pl
from torch import nn, optim
from pytorch_lightning.metrics.functional import auc, accuracy, f1
from sklearn.metrics import roc_auc_score

def _dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
# Global variables which translate the names into the corresponding functions and objects
activation_fn_dict = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid,
                      'leaky relu': nn.LeakyReLU}
optimizers_dict = {'adam': optim.Adam, 'Adadelta': optim.Adadelta, 'Adagrad': optim.Adagrad}


class BinaryNeuralNetwork(pl.LightningModule):
    """A simple NN which receives as an input an iterable of integers which describes the layers structure except the
    output layer. Additionally, the constructor gets the output size of the model in order to construct the output layer
     and the activation function which will be used in ll internal layers."""

    def __init__(self, input_size: int, internal_list_structure: Iterable[int], activation_fn_name: str = 'relu',
                 optimizer_name: str = 'adam',
                 learning_rate: float = 1e-3):
        super(BinaryNeuralNetwork, self).__init__()
        # construct the layers of the model.

        self.activation_fn = activation_fn_dict[activation_fn_name]()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.internal_list_structure = internal_list_structure
        # Insert the first layer
        modules = self.block(input_size, self.internal_list_structure[0])
        # Insert the internal layers
        for i in range(0, len(self.internal_list_structure) - 1):
            modules.extend(self.block(self.internal_list_structure[i], self.internal_list_structure[i + 1]))

        # Insert the last layer together with the sigmoid function
        modules.extend(self.block(self.internal_list_structure[-1], 1, activation=False))
        modules.append(nn.Sigmoid())

        self.modules_list = nn.ModuleList(modules)
        self.loss_function = nn.BCELoss()

    def block(self, in_feat, out_feat, activation=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if activation:
            layers.append(self.activation_fn)
        return layers

    """The forward function, forwards the input while using the activation function in all layers, except the last 
    one. """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BinaryNeuralNetwork")
        parser.add_argument('--input_size', type=int)
        parser.add_argument('--internal_list_structure', type=int, nargs='+')
        parser.add_argument('--activation_fn_name', type=str, default='relu')
        parser.add_argument('--optimizer_name', type=str, default='adam')
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parent_parser

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x

    def configure_optimizers(self):
        return optimizers_dict[self.optimizer_name](self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = torch.squeeze(self(x))
        if len(logits.shape) == 0:
            logits = torch.unsqueeze(logits,dim=0)
        loss = self.loss_function(logits, y.type(torch.float))
        acc = accuracy(logits, y)
        self.log('train_accuracy',acc,prog_bar=True,on_step=False,on_epoch=True)
        return {'loss': loss, 'train_acc': acc}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results['validation_acc'] = results['train_acc']
        del results['train_acc']
        return results

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_val_loss = torch.tensor([output['loss'] for output in outputs]).mean()
        avg_val_acc = torch.tensor([output['validation_acc'] for output in outputs]).mean()

        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log('validation_acc', avg_val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # please notice that the test step is assuming that the whole test is passed as a unit.
        # Use LearningDataSet located in the simple_dataset module in order to achieve that.

        x, y = batch
        logits = torch.squeeze(self(x))
        acc = accuracy(logits, y)
        f1_score = f1(logits, y, 1)
        auc_score = roc_auc_score(y, logits)
        metrics_dict = {'acc': acc.item(), 'f1_score': f1_score.item(), 'auc_score': auc_score}
        self.test_predictions = logits
        self.log_dict(metrics_dict)


if __name__ == '__main__':
    # The purpose of this script is to allow multiple runs of the new_nn module on different datasets
    # and to compare the results.
    # Specific import for the script, not for the module.
    from argparse import ArgumentParser
    from subprocess import check_output
    import yaml
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.style.use('ggplot')
    parser = ArgumentParser()
    # add PROGRAM level args
    # A file which includes the command line arguments needed for the nn_classification script for all different trail.
    # Each in a different line
    # Check microbiome/Projects/GDM/experiment1/code/learning_code/config.txt for instance.
    parser.add_argument('--arguments_file', '-a', type=str)
    parser.add_argument('--cwd', '-c', type=str,default='.')
    # A path to a directory that the nn_classification script will be executed from.
    parser.add_argument('--experiments_names', '-n', type=str, nargs='+')
    parser.add_argument('--title','-t',type=str,default='Performance comparison')
    parser.add_argument('--cross_validation','-cv', default= 1,type=int)
    # The different trail names.
    args = parser.parse_args()

    test_dict_list = []
    arguments_file = open(args.arguments_file, 'r')
    command_line_argument_list = arguments_file.readlines()
    # Read the commands line by line
    for command_line_arguments in command_line_argument_list:
        if command_line_arguments == '\n':
            break
        else:
            cross_validation_test_dict_list = []
            for k in range(args.cross_validation):
                # execute a trail using the command line arguments inserted.
                output = check_output(['python3', 'nn_classification_runner.py','--seed',str(k)] + command_line_arguments.split(' '), stdin=None,
                                      cwd=args.cwd).decode("utf-8")

                # The next two lines are an ugly way to extract the test results out of the output
                output = output.split('--------------------------------------------------------------------------------')[1]
                test_str = output.split('TEST RESULTS')[1]
                test_dict = yaml.load(test_str)
                cross_validation_test_dict_list.append(test_dict)
            final_test_dict = _dict_mean(cross_validation_test_dict_list)
            test_dict_list.append(final_test_dict)

    # Plot the trails results
    exp_result_df = pd.DataFrame(test_dict_list, index=args.experiments_names).transpose()
    ax = exp_result_df.plot.bar(rot=0)
    ax.set_title(args.title)
    # add annotation
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.show()

# --arguments_file /home/sharon200102/Documents/second_degree/Research/microbiome_projects/microbiome/Projects/GDM/experiment1/code/learning_code/config.txt -n GAN_results Saliva_results Stool_results