import sys
import os
import nni
import logging
from torch.optim import SGD, Adam

sys.path.insert(1, os.path.join(".."))
sys.path.insert(1, os.path.join("..", ".."))
sys.path.insert(1, os.path.join("..", "..", ".."))
sys.path.insert(1, os.path.join("..", "..", "..", "finit_state_machine"))
logger = logging.getLogger("NNI_logger")

from binary_params import BinaryFSTParams, BinaryModuleParams, BinaryActivatorParams
from binary_rnn_activator import binaryActivator
from binary_rnn_model import BinaryModule
from fst_dataset import FstDataset


def run_trial(params):
    # collect configuration
    alphabet_size = int(params["alphabet"])
    states_size = int(params["states"])
    embed_dim = int(params["embed_dim"])
    lstm_layers = int(params["lstm_layers"])
    lstm_dropout = params["lstm_dropout"]
    lstm_out = int(params["lstm_out"])
    batch_size = int(params["batch_size"])
    opt = Adam if params["optimizer"] == "ADAM" else SGD
    lr = params["learning_rate"]
    l2_reg = params["regularization"]
    epochs = int(params["epochs"])

    # define data-set
    ds_params = BinaryFSTParams()
    ds_params.FST_ALPHABET_SIZE = alphabet_size
    ds_params.FST_STATES_SIZE = states_size
    ds_params.FST_ACCEPT_STATES_SIZE = 2
    dataset = FstDataset(ds_params)

    # define model
    model_params = BinaryModuleParams(alphabet_size=len(dataset.chr_embed), embed_dim=embed_dim,
                                      lstm_layers=lstm_layers, lstm_dropout=lstm_dropout, lstm_out_dim=lstm_out)
    model_params.OPTIMIZER = opt
    model_params.REGULARIZATION = l2_reg
    model_params.LEARNING_RATE = lr

    # define activator
    activator_params = BinaryActivatorParams()
    activator_params.EPOCHS = epochs
    activator_params.BATCH_SIZE = batch_size
    activator_params.EPOCHS = epochs

    model = BinaryModule(model_params)
    activator = binaryActivator(model, activator_params, dataset)
    activator.train(show_plot=False, apply_nni=True, early_stop=True)


def main():
    try:
        # get parameters form tuner
        params = nni.get_next_parameter()
        logger.debug(params)
        run_trial(params)
    except Exception as exception:
        logger.error(exception)
        raise


if __name__ == "__main__":
    main()
