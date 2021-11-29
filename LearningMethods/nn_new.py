import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, List, Iterable
import pytorch_lightning as pl
from torch import nn, optim
from pytorch_lightning.metrics.functional import auc, accuracy, f1,r2score
from sklearn.metrics import roc_auc_score
from scipy import stats
from torch.optim.lr_scheduler import StepLR
from general_functions import _dict_mean

# Global variables which translate the names into the corresponding functions and objects
activation_fn_dict = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid,
                      'leaky relu': nn.LeakyReLU,'elu':nn.ELU}
optimizers_dict = {'adam': optim.Adam, 'Adadelta': optim.Adadelta, 'Adagrad': optim.Adagrad}


class NeuralNetwork(pl.LightningModule):
    """A simple NN which receives as an input an iterable of integers which describes the layers structure except the
    output layer. Additionally, the constructor gets the output size of the model in order to construct the output layer
     and the activation function which will be used in ll internal layers."""

    def __init__(self, input_size: int, internal_list_structure: Iterable[int], activation_fn_name: str = 'relu',
                 optimizer_name: str = 'adam',
                 learning_rate: float = 1e-3,goal = 'binary',weight_decay = 1e-3
                 ,num_of_classes = None,pos_weight = None,lr_step_size = None,gamma=None):
        super(NeuralNetwork, self).__init__()
        # construct the layers of the model.

        self.activation_fn = activation_fn_dict[activation_fn_name]()
        self.optimizer_name = optimizer_name
        '''@nni.variable(nni.uniform(1e-4,1e-1), name=self.learning_rate)'''
        self.learning_rate = learning_rate
        self.goal = goal
        self.internal_list_structure = internal_list_structure
        '''@nni.variable(nni.uniform(1e-4,1e-1), name=self.weight_decay)'''
        self.weight_decay = weight_decay

        '''@nni.variable(nni.uniform(5,30), name=self.lr_step_size)'''
        self.lr_step_size = lr_step_size
        '''@nni.variable(nni.uniform(1e-2,8e-1), name=self.gamma)'''
        self.gamma = gamma
        if self.lr_step_size is not None and self.gamma is not None:
            self.lr_scheduler_flag = True
        else:
            self.lr_scheduler_flag = False

        self.prediction_flag = True if self.goal == 'binary' or self.goal == 'multi_class' else False
        # num of classes cannot be none while goal is multi class
        if goal == 'multi_class' and num_of_classes is None:
            raise AssertionError('num of classes cannot be none while goal is multi class')
        else:
            output_layer_size = num_of_classes if goal == 'multi_class' else 1
        # Insert the first layer
        modules = self.block(input_size, self.internal_list_structure[0])
        # Insert the internal layers
        for i in range(0, len(self.internal_list_structure) - 1):
            modules.extend(self.block(self.internal_list_structure[i], self.internal_list_structure[i + 1]))

        # Insert the last layer
        modules.extend(self.block(self.internal_list_structure[-1], output_layer_size, activation=False))

        self.modules_list = nn.ModuleList(modules)
        if self.goal == 'binary':
            if pos_weight is not None:
                self.loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
            else:
                self.loss_function = nn.BCEWithLogitsLoss()

            self.probability_fn = nn.Sigmoid()

        elif self.goal == 'multi_class':
            self.loss_function = nn.CrossEntropyLoss()
            self.probability_fn = nn.Softmax(dim=1)

        else:
            self.loss_function = nn.MSELoss()



    def block(self, in_feat, out_feat, activation=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if activation:
            layers.append(self.activation_fn)
        return layers

    """The forward function, forwards the input while using the activation function in all layers, except the last 
    one. """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NeuralNetwork")
        parser.add_argument('--input_size', type=int)
        parser.add_argument('--internal_list_structure', type=int, nargs='+')
        parser.add_argument('--activation_fn_name', type=str, default='relu')
        parser.add_argument('--optimizer_name', type=str, default='adam')
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--goal', type=str, default='binary',choices = ['binary','multi_class','regression'])
        parser.add_argument('--weight_decay', type=float, default=1e-2)
        parser.add_argument('--num_of_classes', type=int,default = None)
        parser.add_argument('--pos_weight', type=float,default = None)
        parser.add_argument('--lr_step_size','-ls', type=int,default = None)
        parser.add_argument('--gamma', type=float,default = None)






        return parent_parser

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x

    def configure_optimizers(self):
        optimizer = optimizers_dict[self.optimizer_name](self.parameters(), lr=self.learning_rate,
                                                        weight_decay = self.weight_decay)
        if self.lr_scheduler_flag:
            lr_scheduler = StepLR(optimizer,self.lr_step_size,self.gamma)
            return [optimizer],[lr_scheduler]

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = torch.squeeze(self(x))
        if len(logits.shape) == 0:
            logits = torch.unsqueeze(logits,dim=0)
        loss = self.loss_function(logits, y.type(torch.float))
        if self.prediction_flag:
            preds = self.probability_fn(logits)
            # Log additional statistics in case of prediction
            acc = accuracy(preds, y)
            self.log('train_accuracy',acc,prog_bar=True,on_step=False,on_epoch=True)
            return {'loss': loss, 'train_acc': acc}

        else:
            r2 = r2score(logits,y)
            return {'loss': loss,'train_r2':r2}



    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        if self.prediction_flag:
            # in case of prediction modify the name of the training statistics to validation statistics
            results['validation_acc'] = results['train_acc']
            del results['train_acc']
        else:
            results['validation_r2'] = results['train_r2']
            del results['train_r2']
        return results

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_val_loss = torch.tensor([output['loss'] for output in outputs]).mean()
        if self.prediction_flag:
            avg_val_acc = torch.tensor([output['validation_acc'] for output in outputs]).mean()
            self.log('validation_acc', avg_val_acc, prog_bar=True)
        else:
            avg_val_r2 = torch.tensor([output['validation_r2'] for output in outputs]).mean()
            self.log('validation_r2', avg_val_r2, prog_bar=True)

        self.log('val_loss', avg_val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # please notice that the test step is assuming that the whole test is passed as a unit.
        # Use LearningDataSet located in the simple_dataset module in order to achieve that.

        x, y = batch
        logits = torch.squeeze(self(x))
        if self.goal == 'binary':
            preds = self.probability_fn(logits)
            acc = accuracy(preds, y)
            f1_score = f1(preds, y, 1)
            auc_score = roc_auc_score(y, preds)
            metrics_dict = {'acc': acc.item(), 'f1_score': f1_score.item(), 'auc_score': auc_score}
        elif self.goal == 'multi_class':
            preds = self.probability_fn(logits)
            acc = accuracy(preds, y)
            metrics_dict = {'acc': acc.item()}
        else:
            r2 = r2score(logits,y)
            rho = stats.spearmanr(logits,y)[0]
            metrics_dict = {'r2': r2.item(),'correlation':rho}




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
    # The different trail names.
    args = parser.parse_args()

    test_dict_list = []
    arguments_file = open(args.arguments_file, 'r')
    command_line_argument_list = arguments_file.readlines()
    # Read the commands line by line
    for command_line_arguments_str in command_line_argument_list:
        if command_line_arguments_str == '\n':
            break
        else:
            # strip thw whole command
            command_line_arguments_str.strip('\n')
            # strip each argument in the command
            command_line_arguments = list(map(lambda x: x.strip('\n'), command_line_arguments_str.split(' ')))

            output = check_output(['python3', 'nn_classification_runner.py','-t',str(k)] + command_line_arguments, stdin=None,
                                  cwd=args.cwd).decode("utf-8")



"""
    # Plot the trails results
    exp_result_df = pd.DataFrame(test_dict_list, index=args.experiments_names).transpose()
    ax = exp_result_df.plot.bar(rot=0)
    ax.set_title(args.title)
    # add annotation
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.show()
"""
# --arguments_file /home/sharon200102/Documents/second_degree/Research/microbiome_projects/microbiome/Projects/GDM/experiment1/code/learning_code/config.txt -n GAN_results Saliva_results Stool_results
# -a /home/sharon200102/Documents/second_degree/Research/microbiome_projects/microbiome/Projects/BGU_fatty_liver/Code/learning_code/complete_samples_config.txt -n VAE_complete_samples metabolomics microbiome -cv 20
