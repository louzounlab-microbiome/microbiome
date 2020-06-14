import nni
import sys
import os
import pandas as pd

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pregnancy_diabetes.GDMDataLoader import GDMDataLoader
from LearningMethods.nn_for_nni import nn_nni_main


def main_nni(X, y, title, folder, result_type):
    # params = nni.get_next_parameter()
    params = {
        "hid_dim_0": 100,
        "hid_dim_1": 10,
        "reg": 0.68,
        "dims": [20, 40, 60, 2],
        "lr": 0.001,
        "test_size": 0.15,
        "batch_size": 4,
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
    task = "prognostic_PTSD_task"
    data_file = 'merged_GDM_tables_w_tax.csv'
    tag_file = 'ok.csv'
    folder = "sderot_anxiety"
    X = pd.read_csv(data_file).set_index('ID').to_numpy()
    y = pd.read_csv(tag_file).set_index('ID').to_numpy()

    main_nni(X, y, task, folder, result_type="auc")

