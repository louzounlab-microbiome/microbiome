import os
import sys


sys.path.insert(1, os.path.join(".."))
sys.path.insert(1, os.path.join("..", ".."))
sys.path.insert(1, os.path.join("..", "..", ".."))
sys.path.insert(1, os.path.join("..", "..", "..", "finit_state_machine"))


from torch.optim import Adam
from binary_params import BinaryModuleParams, BinaryActivatorParams
from binary_rnn_activator import binaryActivator
from binary_rnn_model import BinaryModule
from experiment_oxford_sentiment.oxford_sentiment_dataset import OxfordSentimentDataset, oxford_sentiment_data_split
import os
from torch.utils.data import DataLoader

if __name__ == "__main__":
    sources = [
        # os.path.join("sentiment labelled sentences", "amazon_cells_labelled.txt")
        # os.path.join("sentiment labelled sentences", "imdb_labelled.txt")
        os.path.join("sentiment labelled sentences", "yelp_labelled.txt")
    ]
    ds = OxfordSentimentDataset(sources)

    # define model
    model_params = BinaryModuleParams(alphabet_size=ds.num_words, embed_dim=30,
                                      lstm_layers=1, lstm_dropout=0.5, lstm_out_dim=100)
    model_params.OPTIMIZER = Adam
    model_params.REGULARIZATION = 18e-3
    model_params.LEARNING_RATE = 1e-3

    # define activator
    activator_params = BinaryActivatorParams()
    activator_params.EPOCHS = 200
    activator_params.BATCH_SIZE = 16

    model = BinaryModule(model_params)
    activator = binaryActivator(model, activator_params, ds, oxford_sentiment_data_split)
    activator.train()
