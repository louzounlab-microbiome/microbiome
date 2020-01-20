from torch.nn import Module, LSTM, Dropout, Embedding, AvgPool1d, Linear, MaxPool1d
from torch import sigmoid, softmax
from fat_language_model_dataset import FstLanguageModuleDataset
from language_model_params import LanguageModelParams


class LanguageModuleLSTM(Module):
    def __init__(self, embed_dim, vocab_dim, lstm_hidden_dim, lstm_layers, lstm_dropout):
        super(LanguageModuleLSTM, self).__init__()
        # word embed layer
        self._embeddings = Embedding(vocab_dim, embed_dim)
        # Bi-LSTM layers
        self._lstm = LSTM(embed_dim, lstm_hidden_dim, lstm_layers, batch_first=True, bidirectional=False)
        self._dropout = Dropout(p=lstm_dropout)

    @property
    def lstm(self):
        return self._lstm

    def forward(self, words_embed, hidden):
        x = self._embeddings(words_embed)
        output_seq, hidden_seq = self._lstm(self._dropout(x), hidden)
        return output_seq, hidden_seq


class LanguageModuleMLP(Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation):
        super(LanguageModuleMLP, self).__init__()
        # useful info in forward function
        self._layer0 = Linear(in_dim, hidden_dim)
        self._layer1 = Linear(hidden_dim, out_dim)
        self._activation = activation

    def forward(self, x):
        x = self._layer0(x)
        x = self._activation(x)
        x = self._layer1(x)
        x = softmax(x, dim=1)
        return x


class LanguageModule(Module):
    def __init__(self, params: LanguageModelParams):
        super(LanguageModule, self).__init__()
        # useful info in forward function
        self._dim_hidden_lstm = params.RNN_LSTM_hidden_dim
        self._num_layers_lstm = params.RNN_LSTM_layers
        self._sequence_lstm = LanguageModuleLSTM(params.RNN_EMBED_dim, params.RNN_EMBED_vocab_dim,
                                                 params.RNN_LSTM_hidden_dim, params.RNN_LSTM_layers,
                                                 params.RNN_LSTM_dropout)
        self._mlp = LanguageModuleMLP(params.MLP_LINEAR_in_dim, params.MLP_LINEAR_hidden_dim, params.MLP_LINEAR_out_dim,
                                      params.MLP_Activation)
        self.optimizer = self.set_optimizer(params.LEARNING_RATE, params.OPTIMIZER)

    @property
    def lstm_module(self):
        return self._sequence_lstm.lstm

    @property
    def dim_hidden_lstm(self):
        return self._dim_hidden_lstm

    @property
    def lstm_layers(self):
        return self._num_layers_lstm

    def set_optimizer(self, lr, opt):
        return opt(self.parameters(), lr=lr)

    def forward(self, x, hidden):
        x, hidden = self._sequence_lstm(x, hidden)
        x = self._mlp(x)
        return x, hidden


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset_params import FSTParams
    from fst_dataset import FstDataset
    import torch
    _ds_params = FSTParams()
    _ds = FstLanguageModuleDataset(_ds_params)
    _dl = DataLoader(
        dataset=_ds,
        batch_size=2,
        collate_fn=_ds.collate_fn
    )

    _lm = LanguageModule(LanguageModelParams(alphabet_size=_ds_params.FST_ALPHABET_SIZE))
    for _i, (_sequence, _label) in enumerate(_dl):
        # ( lstm_layers, batch_size, x_input_dim )
        _hidden = (torch.zeros((_lm.lstm_layers, _sequence.shape[0], _lm.dim_hidden_lstm)),  # dim = (bach, len_seq=1, hidden_dim)
                   torch.zeros((_lm.lstm_layers, _sequence.shape[0], _lm.dim_hidden_lstm)))

        for _sym_idx in range(_sequence.shape[1]):
            word_col = _sequence[:, _sym_idx].unsqueeze(dim=1)
            _out, _hidden = _lm(word_col, _hidden)
            print(_out)
        e = 0
