from argparse import ArgumentParser
import argparse
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from LearningMethods.nn_new import BinaryNeuralNetwork
from Datasets.simple_dataset import LearningDataSet, ToTensor
import pandas as pd
import pytorch_lightning as pl
import os.path
import torch
import numpy as np
import random



torch.manual_seed(1)
if __name__ == '__main__':
    """ This script aims to construct a neural network according to the architecture inserted
    and to classify the data."""
    parser = ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--tag_path', type=str)
    parser.add_argument('--results_path', type=str, default='.')
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--save_test', default=None)
    parser.add_argument('--group_id_path', '-g', help='If the train test split should consider an external division',
                        default=None)
    parser.add_argument('--seed','-s', default=None,type=int)

    # Add data and model specific args
    parser = LearningDataSet.add_model_specific_args(parser)
    parser = BinaryNeuralNetwork.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
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
    model_parameters = dict(vars(arg_groups['BinaryNeuralNetwork']))

    monitor = 'val_loss'
    logs_path = os.path.join(args.results_path, args.project_name, 'LOGS')
    checkpoint_path = os.path.join(args.results_path, args.project_name, 'CHECKPOINTS')
    # Init the models with the args inserted

    data_df = pd.read_csv(args.data_path, index_col=0)
    tag_series = pd.read_csv(args.tag_path, index_col=0, squeeze=True)
    if args.group_id_path is not None:
        group_id = pd.read_csv(args.group_id_path,index_col=0)
    else:
        group_id = None

    dm = LearningDataSet(data_df, tag_series,group_id=group_id,transform=ToTensor([torch.float32, torch.long]), **dm_parameters)
    model = BinaryNeuralNetwork(**model_parameters)
    callbacks = [EarlyStopping(monitor=monitor, patience=5), ModelCheckpoint(monitor=monitor, dirpath=checkpoint_path)]
    logger = TensorBoardLogger(save_dir=logs_path)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)
    if args.save_test is not None:
        test_df = data_df.iloc[dm.test_data.indices]
        test_df = test_df.assign(pred=model.test_predictions, label=tag_series.iloc[dm.test_data.indices])
        test_df.to_csv(os.path.join(args.results_path, args.project_name, args.save_test))
