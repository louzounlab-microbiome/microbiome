from torch.nn.functional import relu, cross_entropy
from torch.optim import Adam
from dataset_params import FSTParams


ALPHABET_SIZE = 2
class LanguageModelFSTParams(FSTParams):
    def __init__(self):
        super().__init__()
        self.DATASET_SIZE = 1000
        self.NEGATIVE_SAMPLES = True
        self.FST_ALPHABET_SIZE = ALPHABET_SIZE
        self.FST_STATES_SIZE = 10
        self.FST_ACCEPT_STATES_SIZE = 1


class LanguageModelParams:
    def __init__(self, alphabet_size, lstm_out_dim=100):
        self.RNN_EMBED_dim = 10
        self.RNN_EMBED_vocab_dim = alphabet_size + 3    # +1 for _PAD_ + _END_ + _START_
        self.RNN_LSTM_hidden_dim = lstm_out_dim
        self.RNN_LSTM_layers = 3
        self.RNN_LSTM_dropout = 0.1

        self.MLP_LINEAR_in_dim = lstm_out_dim
        self.MLP_LINEAR_hidden_dim = 50
        self.MLP_LINEAR_out_dim = alphabet_size + 1  # ALPHABET + END
        self.MLP_Activation = relu

        self.LEARNING_RATE = 1e-3
        self.OPTIMIZER = Adam


class LanguageModelActivatorParams:
    def __init__(self, **kargs):
        self.TRAIN_TEST_SPLIT = 0.5
        self.LOSS = (cross_entropy, {"ignore_index": kargs.get("ignore_index", -100)})
        self.BATCH_SIZE = 64
        self.GPU = True
        self.EPOCHS = 250
