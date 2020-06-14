from torch.optim import Adam, SGD, Adagrad
from torch.utils.data import Dataset
from random import shuffle


from LearningMethods.model_metrics import custom_rmse_for_missing_values, nn_custom_r2_for_missing_values, \
    single_bacteria_custom_corr_for_missing_values, multi_bacteria_custom_corr_for_missing_values

name_of_function_to_function_map = {"custom_rmse_for_missing_values": custom_rmse_for_missing_values,
                                    "nn_custom_r2_for_missing_values": nn_custom_r2_for_missing_values,
                                    "single_bacteria_custom_corr_for_missing_values": single_bacteria_custom_corr_for_missing_values,
                                    "multi_bacteria_custom_corr_for_missing_values": multi_bacteria_custom_corr_for_missing_values}

optimizer_name_to_function_map = {"Adam": Adam, "SGD": SGD, "Adagrad": Adagrad}

# ----------------------------------------------- params -----------------------------------------------

class NNActivatorParams:
    def __init__(self, TRAIN_TEST_SPLIT=0.8, LOSS="nn_custom_rmse_for_missing_value",
                 CORR="single_bacteria_custom_corr_for_missing_values", BATCH_SIZE=100,
                 GPU=False, EPOCHS=50, EARLY_STOP=0):

        self.TRAIN_TEST_SPLIT = TRAIN_TEST_SPLIT
        self.LOSS = name_of_function_to_function_map[LOSS]
        self.CORR = name_of_function_to_function_map[CORR]
        self.R2 = nn_custom_r2_for_missing_values
        self.BATCH_SIZE = BATCH_SIZE
        self.GPU = GPU
        self.EPOCHS = EPOCHS
        self.EARLY_STOP = EARLY_STOP



class NNModuleParams:
    def __init__(self, NUMBER_OF_BACTERIA, nn_hid_dim_1, nn_hid_dim_2,
                 nn_output_dim, NN_LAYER_NUM=1, DROPOUT=0, LEARNING_RATE=1e-3,
                 OPTIMIZER="Adam", REGULARIZATION=1e-4, DIM=1, SHUFFLE=True):
        self.SEQUENCE_PARAMS = NNSequenceParams(NN_input_dim=NUMBER_OF_BACTERIA,
                                                NN_hidden_dim_1=nn_hid_dim_1, NN_hidden_dim_2=nn_hid_dim_2,
                                                NN_output_dim=nn_output_dim,
                                                NN_LAYER_NUM=NN_LAYER_NUM, DROPOUT=DROPOUT)
        self.LEARNING_RATE = LEARNING_RATE
        self.OPTIMIZER = optimizer_name_to_function_map[OPTIMIZER]
        self.REGULARIZATION = REGULARIZATION
        self.DIM = DIM
        self.SHUFFLE = SHUFFLE


class NNSequenceParams:
    def __init__(self, NN_input_dim, NN_hidden_dim_1, NN_hidden_dim_2, NN_output_dim, NN_LAYER_NUM, DROPOUT, timesteps=10):
        self.NN_input_dim = NN_input_dim
        self.NN_hidden_dim_1 = NN_hidden_dim_1
        self.NN_hidden_dim_2 = NN_hidden_dim_2
        self.NN_output_dim = NN_output_dim
        self.NN_layers = NN_LAYER_NUM
        self.NN_dropout = DROPOUT
        self.timesteps = timesteps


class RNNActivatorParams:
    def __init__(self, TRAIN_TEST_SPLIT=0.8, LOSS="rnn_custom_rmse_for_missing_values", CORR="single_bacteria_custom_corr_for_missing_values", BATCH_SIZE=100,
                 GPU=False, EPOCHS=50, EARLY_STOP=0):
        self.TRAIN_TEST_SPLIT = TRAIN_TEST_SPLIT
        self.LOSS = name_of_function_to_function_map[LOSS]
        self.CORR = name_of_function_to_function_map[CORR]
        self.R2 = nn_custom_r2_for_missing_values
        self.BATCH_SIZE = BATCH_SIZE
        self.GPU = GPU
        self.EPOCHS = EPOCHS
        self.EARLY_STOP = EARLY_STOP



class RNNModuleParams:
    def __init__(self, NUMBER_OF_BACTERIA, lstm_hidden_dim, mlp_out_dim, LSTM_LAYER_NUM=1, DROPOUT=0, LEARNING_RATE=1e-3,
                 OPTIMIZER="Adam", REGULARIZATION=1e-4, DIM=1, SHUFFLE=False):
        self.SEQUENCE_PARAMS = RNNSequenceParams(LSTM_input_dim=NUMBER_OF_BACTERIA, LSTM_hidden_dim=lstm_hidden_dim,
                                                 LSTM_LAYER_NUM=LSTM_LAYER_NUM, DROPOUT=DROPOUT)
        self.LINEAR_PARAMS = MLPParams(in_dim=lstm_hidden_dim, out_dim=mlp_out_dim)
        self.LEARNING_RATE = LEARNING_RATE
        self.OPTIMIZER = optimizer_name_to_function_map[OPTIMIZER]
        self.REGULARIZATION = REGULARIZATION
        self.DIM = DIM
        self.SHUFFLE = SHUFFLE


class RNNSequenceParams:
    def __init__(self, LSTM_input_dim, LSTM_hidden_dim, LSTM_LAYER_NUM, DROPOUT):
        self.LSTM_input_dim = LSTM_input_dim
        self.LSTM_hidden_dim = LSTM_hidden_dim
        self.LSTM_layers = LSTM_LAYER_NUM
        self.LSTM_dropout = DROPOUT


class MLPParams:
    def __init__(self, in_dim, out_dim):
        self.LINEAR_in_dim = in_dim
        self.LINEAR_out_dim = out_dim

# ----------------------------------------------- data sets -----------------------------------------------
class MicrobiomeDataset(Dataset):
    def __init__(self, X, y, missing_values):
        self._X = X
        self._y = y
        self._missing_values = missing_values

    def __getitem__(self, index):
        return self._X[index], self._y[index], self._missing_values[index]

    def __len__(self):
        return len(self._X)


def split_microbiome_dataset(dataset: MicrobiomeDataset, split_list, person_indexes=None):
    """
    this function splits a data-set into n = len(split_list) disjointed data-sets
    """
    import numpy as np
    # create a list of lengths [0.1, 0.4, 0.5] -> [100, 500, 1000(=len_data)]
    split_list = np.multiply(np.cumsum(split_list), len(dataset)).astype("int").tolist()
    # list of shuffled indices to sample randomly
    shuffled_idx = []
    if person_indexes:
        shuffle(person_indexes)
        for arr in person_indexes:
            for val in arr:
                shuffled_idx.append(val)
    else:
        shuffled_idx = list(range(len(dataset)))
        shuffle(shuffled_idx)
    # split the data itself
    new_data_X = [[] for _ in range(len(split_list))]
    new_data_y = [[] for _ in range(len(split_list))]
    new_data_missing_values = [[] for _ in range(len(split_list))]
    for sub_data_idx, (start, end) in enumerate(zip([0] + split_list[:-1], split_list)):
        for i in range(start, end):
            X, y, missing_values = dataset.__getitem__(shuffled_idx[i])
            new_data_X[sub_data_idx].append(np.array(X))
            new_data_y[sub_data_idx].append(np.array(y))
            new_data_missing_values[sub_data_idx].append(np.array(missing_values))
    # create sub sets
    # new_data_X = np.array(new_data_X)
    # new_data_y = np.array(new_data_y)
    sub_datasets = []
    for i in range(len(new_data_X)):
        sub_datasets.append(MicrobiomeDataset(np.array(new_data_X[i]),
                                              np.array(new_data_y[i]),
                                              np.array(new_data_missing_values[i])))
    return sub_datasets



