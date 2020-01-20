from binary_params import BinaryActivatorParams, BinaryModuleParams
from binary_rnn_activator import binaryActivator
from binary_rnn_model import BinaryModule
from language_model import LanguageModule
from language_model_activator import LanguageModuleActivator
from language_model_params import LanguageModelParams, LanguageModelActivatorParams
from python_code.experiment_simplified_json.simplified_json_dataset import SimpleJsonAcceptorDataset, \
    split_simplified_json_acceptor_dataset, SimpleJsonLanguageModelDataset, split_simple_json_language_model_dataset


def train_acceptor():
    json_dataset = SimpleJsonAcceptorDataset(size=1000)
    activator_params = BinaryActivatorParams()
    activator_params.EPOCHS = 50
    activator = binaryActivator(BinaryModule(BinaryModuleParams(alphabet_size=len(json_dataset.chr_embed))),
                                activator_params, json_dataset, split_simplified_json_acceptor_dataset)
    activator.train(validate_rate=10)


def train_language_model():
    json_dataset = SimpleJsonLanguageModelDataset(size=10000)
    activator_params = LanguageModelActivatorParams(ignore_index=json_dataset.pad_idx)
    activator_params.EPOCHS = 250
    activator = LanguageModuleActivator(LanguageModule(LanguageModelParams(alphabet_size=len(json_dataset._idx_to_chr))),
                                        activator_params, json_dataset, splitter=split_simple_json_language_model_dataset)

    activator.train(valid_rate=10)


if __name__ == '__main__':
    train_acceptor()
    # train_language_model()
"""
TRAIN 100%
VALIDATE 100%
VALIDATE 100%
Acc_Train: 0.8500 || AUC_Train: 0.9521 || Loss_Train: 0.5165 || Acc_Dev: 0.8460 || AUC_Dev: 0.9467 || Loss_Dev: 0.5217 || 
TRAIN 100%
VALIDATE 100%
VALIDATE 100%
Acc_Train: 0.8460 || AUC_Train: 0.9667 || Loss_Train: 0.4871 || Acc_Dev: 0.8420 || AUC_Dev: 0.9671 || Loss_Dev: 0.4920 || 
TRAIN 100%
VALIDATE 100%
VALIDATE 100%
Acc_Train: 0.8380 || AUC_Train: 0.9759 || Loss_Train: 0.4518 || Acc_Dev: 0.8300 || AUC_Dev: 0.9765 || Loss_Dev: 0.4559 || 
"""