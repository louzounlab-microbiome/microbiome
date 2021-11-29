import pickle
from argparse import ArgumentParser
import argparse
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os.path

from LearningMethods.nn_new import NeuralNetwork
from Datasets.simple_dataset import LearningDataSet, ToTensor
import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np
import random
from LearningMethods.general_functions import _dict_mean
import nni

if __name__ == '__main__':
    """ This script aims to construct a neural network according to the architecture inserted
    and to classify the data."""
    parser = ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--tag_path', type=str)
    parser.add_argument('--results_path', type=str, default='.')
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--group_id_path', '-g', help='If the train test split should consider an external division',
                        default=None)
    parser.add_argument('--seed','-s', default=0,type=int)
    parser.add_argument('--repeat','-r',type=int,default=1)
    parser.add_argument('--report_metric','-rm',type=str,default=None)
    parser.add_argument('--nni','-n', action='store_true')





    # Add data and model specific args
    parser = LearningDataSet.add_model_specific_args(parser)
    parser = NeuralNetwork.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    if args.nni:
        trail_id = nni.get_trial_id()
        results_folder = os.path.join(args.results_path, args.project_name,trail_id)
        tuned_params = nni.get_next_parameter()
    else:
        results_folder = os.path.join(args.results_path, args.project_name)
        tuned_params = None

    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    arg_groups = {}

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    dm_parameters = dict(vars(arg_groups['LearningDataSet']))
    model_parameters = dict(vars(arg_groups['NeuralNetwork']))
    if tuned_params is not None:
        model_parameters.update(tuned_params)

    monitor = 'val_loss'
    logs_path = os.path.join(results_folder, 'LOGS')
    # Init the models with the args inserted

    data_df = pd.read_csv(args.data_path, index_col=0)
    tag_series = pd.read_csv(args.tag_path, index_col=0, squeeze=True)
    if args.group_id_path is not None:
        group_id = pd.read_csv(args.group_id_path,index_col=0,squeeze=True)
    else:
        group_id = None

    results_list = []
    for rep in range(args.repeat):
        checkpoint_path = os.path.join(results_folder, 'CHECKPOINTS', str(rep))
        dm = LearningDataSet(data_df, tag_series,group_id=group_id,transform=ToTensor([torch.float32, torch.tensor(tag_series.values).type()]), **dm_parameters)
        model = NeuralNetwork(**model_parameters)
        checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=checkpoint_path)
        callbacks = [EarlyStopping(monitor=monitor, patience=20), checkpoint_callback]
        logger = TensorBoardLogger(save_dir=logs_path)

        trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
        trainer.fit(model=model, datamodule=dm)
        result = trainer.test(ckpt_path="best")[0]
        results_list.append(result)

    final_results = _dict_mean(results_list)
    with open(os.path.join(results_folder,'Results'),'wb') as file:
        pickle.dump(final_results,file)

    if args.report_metric is not None and args.nni:
        nni.report_final_result(final_results[args.report_metric])

