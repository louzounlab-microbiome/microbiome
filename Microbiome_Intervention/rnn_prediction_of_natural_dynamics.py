import os
from pathlib import Path

import pandas as pd
from Microbiome_Intervention.create_learning_data_from_data_set import get_adapted_X_y_for_wanted_learning_task
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def preform_time_serie_learning(tax, task, X_y_files_list_path):
    all_times_all_bact_results_path = os.path.join(tax, task + "_results_df.csv")
    important_bacteria_reults_path = os.path.join(tax, task + "_significant_bacteria_prediction_results_df.csv")
    conclusionss_path = os.path.join(tax, task + "_conclusions.csv")

    with open(os.path.join(tax, "bacteria.txt"), "r") as b_file:
        bacteria = b_file.readlines()
        bacteria = [b.rstrip() for b in bacteria]

    all_times_all_bacteria_all_models_results_df = Path(all_times_all_bact_results_path)
    if not all_times_all_bacteria_all_models_results_df.exists():
        all_times_all_bacteria_all_models_results_df = pd.DataFrame(columns=['BACTERIA_NUMBER', 'BACTERIA', 'ALGORITHM',
                                                                             'RHO', 'RANDOM_RHO', 'P_VALUE',
                                                                             'RANDOM_P_VALUE', 'RMSE', 'RANDOM_RMSE',
                                                                             'PARAMS', 'BETA'])
        all_times_all_bacteria_all_models_results_df.to_csv(all_times_all_bact_results_path, index=False)

    important_bacteria_reults_df = Path(important_bacteria_reults_path)
    if not important_bacteria_reults_df.exists():
        important_bacteria_reults_df = pd.DataFrame(columns=['bacteria', 'auc',
                                                             'true_positive', 'true_negative',
                                                             'false_positive', 'false_negative',
                                                             'acc', 'precision', 'recall',
                                                             'specificity', 'sensitivity', 'balanced acc', 'F1'])
        important_bacteria_reults_df.to_csv(important_bacteria_reults_path, index=False)

    with open(os.path.join(tax, X_y_files_list_path), "r") as file:
        paths = file.readlines()
        paths = [p.strip('\n') for p in paths]

    for i, [bact, path] in enumerate(zip(bacteria, paths)):
        all_times_all_bacteria_all_models_results_df = pd.read_csv(all_times_all_bact_results_path)
        important_bacteria_reults_df = pd.read_csv(important_bacteria_reults_path)
        X, y, name = get_adapted_X_y_for_wanted_learning_task(tax, path, "time_serie")
        predict_interaction_network_structure_RNN(X, y, name, i, bact, bacteria)


def predict_interaction_network_structure_RNN(X_serie, y_serie, name, i, bact, bacteria):
    pass


if __name__ == "__main__":
    # load time serie date from files
    tax = os.path.join("MDSINE_data_cdiff", "tax=7")
    task = "RNN"
    X_y_files_list_path = "time_serie_files_names.txt"
    preform_time_serie_learning(tax, task, X_y_files_list_path)
