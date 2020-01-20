import os
import sys
sys.path.insert(1, os.path.join(".."))
sys.path.insert(1, os.path.join("..", ".."))
sys.path.insert(1, os.path.join("..", "..", ".."))
sys.path.insert(1, os.path.join("..", "..", "..", "finit_state_machine"))


from yelp_sentiment_dataset import YelpSentimentDataset, yelp_sentiment_data_split
from torch.optim import Adam
from binary_params import BinaryModuleParams, BinaryActivatorParams
from binary_rnn_activator import binaryActivator
from binary_rnn_model import BinaryModule
import os
from torch.utils.data import DataLoader

if __name__ == "__main__":
    sources = [
        os.path.join("yelp_review_polarity_csv", "train.csv"),
        os.path.join("yelp_review_polarity_csv", "test.csv"),
    ]
    ds = YelpSentimentDataset(sources)

    # define model
    model_params = BinaryModuleParams(alphabet_size=ds.num_words, embed_dim=10,
                                      lstm_layers=1, lstm_dropout=0, lstm_out_dim=50)
    model_params.OPTIMIZER = Adam
    model_params.REGULARIZATION = 1e-3
    model_params.LEARNING_RATE = 1e-3

    # define activator
    activator_params = BinaryActivatorParams()
    activator_params.EPOCHS = 200
    activator_params.BATCH_SIZE = 64
    activator_params.TRAIN_TEST_SPLIT = 0.93

    model = BinaryModule(model_params)
    activator = binaryActivator(model, activator_params, ds, yelp_sentiment_data_split)
    activator.train()
