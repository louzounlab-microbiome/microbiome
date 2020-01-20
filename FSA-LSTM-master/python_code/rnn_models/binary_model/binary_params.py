from torch.nn.functional import relu, binary_cross_entropy
from torch.optim import Adam
from dataset_params import FSTParams


# 3 - 5
# 5 - 10
# 2 - 10
ALPHABET_SIZE = 10
class BinaryFSTParams(FSTParams):
    def __init__(self):
        super().__init__()
        self.DATASET_SIZE = 20000
        self.NEGATIVE_SAMPLES = True
        self.FST_ALPHABET_SIZE = ALPHABET_SIZE
        self.FST_STATES_SIZE = 20
        self.FST_ACCEPT_STATES_SIZE = 2


LSTM_OUT_DIM = 50
EMBED_DIM = 8
class SequenceEncoderParams:
    def __init__(self, vocab_dim, out_dim, embed_dim, lstm_layers, lstm_dropout):
        self.EMBED_dim = embed_dim
        self.EMBED_vocab_dim = vocab_dim    # +1 for _PAD_
        self.LSTM_hidden_dim = out_dim
        self.LSTM_layers = lstm_layers
        self.LSTM_dropout = lstm_dropout


class MLPParams:
    def __init__(self, in_dim):
        self.LINEAR_in_dim = in_dim
        self.LINEAR_out_dim = 1


class BinaryModuleParams:
    def __init__(self, alphabet_size=ALPHABET_SIZE, lstm_out_dim=LSTM_OUT_DIM, embed_dim=EMBED_DIM, lstm_layers=1, lstm_dropout=0.5):
        self.SEQUENCE_PARAMS = SequenceEncoderParams(vocab_dim=alphabet_size + 1, out_dim=lstm_out_dim,
                                                     embed_dim=embed_dim, lstm_layers=lstm_layers, lstm_dropout=lstm_dropout)
        self.LINEAR_PARAMS = MLPParams(in_dim=lstm_out_dim)
        self.LEARNING_RATE = 1e-3
        self.OPTIMIZER = Adam
        self.REGULARIZATION = 1e-4


class BinaryActivatorParams:
    def __init__(self):
        self.TRAIN_TEST_SPLIT = 0.5
        self.LOSS = binary_cross_entropy
        self.BATCH_SIZE = 64
        self.GPU = True
        self.EPOCHS = 200
