from binary_params import BinaryActivatorParams, BinaryModuleParams, BinaryFSTParams
from binary_rnn_activator import binaryActivator
from binary_rnn_model import BinaryModule
from dataset_params import FSTParams
from experiment_fst_simplified_json_base_line.simplified_json_fst_params import SIMPLIFIED_JSON_ALPHABET, \
    SIMPLIFIED_JSON_STATES, SIMPLIFIED_JSON_ACCEPT_STATES, SIMPLIFIED_JSON_FST, SIMPLIFIED_JSON_TRANSITIONS, \
    SIMPLIFIED_JSON_INIT_STATE
from experiment_fst_simplified_json_base_line.tools import load_fst_by_nodes_to_add
from fst import FST
from fst_dataset import FstDataset, split_fst_dataset


class AddEdgesParams(FSTParams):
    def __init__(self):
        super().__init__()
        self.DATASET_SIZE = 1000
        self.NEGATIVE_SAMPLES = True
        self.FST_ALPHABET_SIZE = None
        self.FST_STATES_SIZE = None
        self.FST_ACCEPT_STATES_SIZE = None


def run_experiment():
    score = open("score.csv", "wt")
    for k in range(1, 21):
        k *= 10
        if k == 0:
            alphabet, states, init_state, accept_states, transitions = \
                SIMPLIFIED_JSON_ALPHABET, SIMPLIFIED_JSON_STATES, SIMPLIFIED_JSON_INIT_STATE, \
                SIMPLIFIED_JSON_ACCEPT_STATES, SIMPLIFIED_JSON_TRANSITIONS
        else:
            alphabet, states, init_state, accept_states, transitions = load_fst_by_nodes_to_add(k)
            init_state = init_state[0]

        fst = FST(alphabet, states, init_state, accept_states, transitions)
        fst_dataset = FstDataset(BinaryFSTParams(), fst=fst)

        activator_params = BinaryActivatorParams()
        activator_params.EPOCHS = 100

        activator = binaryActivator(BinaryModule(BinaryModuleParams(alphabet_size=len(fst_dataset.chr_embed))),
                                    activator_params, fst_dataset, split_fst_dataset)
        activator.train(validate_rate=10)

        score.write(str(k) + "train_loss," + ",".join([str(v) for v in activator.loss_train_vec]) + "\n")
        score.write(str(k) + "train_acc," + ",".join([str(v) for v in activator.accuracy_train_vec]) + "\n")
        score.write(str(k) + "train_auc," + ",".join([str(v) for v in activator.auc_train_vec]) + "\n")
        score.write(str(k) + "dev_loss," + ",".join([str(v) for v in activator.loss_dev_vec]) + "\n")
        score.write(str(k) + "dev_acc," + ",".join([str(v) for v in activator.accuracy_dev_vec]) + "\n")
        score.write(str(k) + "dev_auc," + ",".join([str(v) for v in activator.auc_dev_vec]) + "\n")


if __name__ == '__main__':
    run_experiment()
