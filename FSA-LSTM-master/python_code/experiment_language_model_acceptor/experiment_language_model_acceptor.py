

def _run_experiment(config, report_intermediate_res):
    from fat_language_model_dataset import FstLanguageModuleDataset
    from language_model import LanguageModule
    from language_model_activator import LanguageModuleActivator
    from language_model_params import LanguageModelFSTParams, LanguageModelActivatorParams, LanguageModelParams
    import numpy as np

    alphabet_size = config["alphabet_size"]
    state_size = config["state_size"]

    ds_params = LanguageModelFSTParams()
    ds_params.FST_ACCEPT_STATES_SIZE = state_size
    ds_params.FST_ALPHABET_SIZE = alphabet_size

    ds = FstLanguageModuleDataset(ds_params)
    activator = LanguageModuleActivator(LanguageModule(LanguageModelParams(alphabet_size=ds_params.FST_ALPHABET_SIZE)),
                                        LanguageModelActivatorParams(ignore_index=ds.pad_idx), ds,
                                        report_intermediate=report_intermediate_res)
    activator.train(valid_rate=10)

    best_score = int(np.argmin(activator.loss_train_vec))
    measures = {
        "Acc_Train": activator.accuracy_train_vec[best_score],
        "Acceptor_Acc_Train": activator.accuracy_train_vec[best_score],
        "Loss_Train": activator.loss_train_vec[best_score],
        "Acc_Dev": activator.acceptor_acc_dev_vec[best_score],
        "Acceptor_Acc_Dev": activator.acceptor_acc_dev_vec[best_score],
        "Loss_Dev": activator.loss_dev_vec[best_score]
    }
    return measures


def debug():
    def ret_inter(id_, res):
        print(id_, res)

    _run_experiment({
        "alphabet_size": 5,
        "state_size": 15
    }, ret_inter)


if __name__ == '__main__':
    debug()

    # from python_code.hyper_parameter_tuning.src.hype import Hype
    # search_space_ = {
    #         "alphabet_size":           {"_type": "choice",   "_value": [8, 9, 10]},
    #         "state_size":      {"_type": "choice",  "_value": [18, 19, 20]},
    # }
    # Hype("lang_model_acceptor", search_space_, _run_experiment).run_test()
