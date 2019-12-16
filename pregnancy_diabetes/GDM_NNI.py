import nni
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pregnancy_diabetes.GDMDataLoader import GDMDataLoader
from LearningMethods.nn_for_nni import nn_nni_main


def main_nni(X, y, title, folder, result_type):
    #params = nni.get_next_parameter()    
    params = {
        "hid_dim_0": 120,
        "hid_dim_1": 160,
        "reg": 0.68,
        "dims": [20, 40, 60, 2],
        "lr": 0.001,
        "test_size": 0.1,
        "batch_size": 32,
        "shuffle": 1,
        "num_workers": 4,
        "epochs": 150,
        "optimizer": 'SGD',
        "loss": 'MSE'
    }

    print(params)
    auc, loss = nn_nni_main(X, y, params, title, folder)
    
    if result_type == "loss":
        nni.report_final_result(loss)
    if result_type == "auc":
        nni.report_final_result(auc)
    else:
        raise Exception


if __name__ == "__main__":
    task = "prognostic_diabetes_task"
    bactria_as_feature_file = 'merged_GDM_tables_w_tax.csv'
    samples_data_file = 'ok.csv'
    tax = int(sys.argv[1])
    folder = "pregnancy_diabetes"
    data_loader = GDMDataLoader( title=task, bactria_as_feature_file=bactria_as_feature_file,
                                                   samples_data_file=samples_data_file, taxnomy_level=tax,
                                                   allow_printing=True, perform_anna_preprocess=False)
    X, y = data_loader.get_X_y_for_nni(task)
    main_nni(X, y, task, folder, result_type="auc")

