import os
import pandas as pd
import numpy as np
import nni
from sklearn.model_selection import train_test_split
from LearningMethods.svm_learning_model import SVMLearningModel
from LearningMethods.xgb_learning_model import XGBLearningModel
from Plot import pickle
from LearningMethods.nn_models import *
from LearningMethods.nn_learning_model import nn_main
from collections import Counter

models_nn = {'relu_b': nn_2hl_relu_b_model, 'tanh_b': nn_2hl_tanh_b_model,
             'leaky_b': nn_2hl_leaky_b_model, 'sigmoid_b': nn_2hl_sigmoid_b_model,
             'relu_mul': nn_2hl_relu_mul_model, 'tanh_mul': nn_2hl_tanh_mul_model,
             'leaky_mul': nn_2hl_leaky_mul_model, 'sigmoid_mul': nn_2hl_sigmoid_mul_model}


def read_otu_and_mapping_files(otu_path, mapping_path):
    otu_file = pd.read_csv(otu_path)
    id_col = otu_file.columns[0]
    otu_file = otu_file.set_index(id_col)
    mapping_file = pd.read_csv(mapping_path)
    id_col = mapping_file.columns[0]
    mapping_file = mapping_file.set_index(id_col)


    otu_ids = otu_file.index
    map_ids = mapping_file.index
    mutual_ids = [id for id in otu_ids if id in map_ids]
    X = otu_file.loc[mutual_ids]
    y_ = mapping_file.loc[mutual_ids]

    n = [i for i, item in zip(y_.index, y_["Tag"]) if pd.isna(item)]
    X = X.drop(n).values
    y = y_.drop(n)["Tag"].astype(int)

    print(Counter(y))
    return np.array(X), np.array(y)


def get_weights(y):
    classes_sum = [np.sum(np.array(y) == unique_class) for unique_class in
                   np.unique(np.array(y))]
    classes_ratio = [1 - (a / sum(classes_sum)) for a in classes_sum]
    weights_map = {a: classes_ratio[a] for a in set(y)}
    return weights_map


def multi_model_learning_main(folder, otu_path, mapping_path, pca_path, dict):
    # step 1: get data from files
    X, y = read_otu_and_mapping_files(otu_path, mapping_path)
    pca_obj = pickle.load(open(pca_path, "rb"))
    weights = get_weights(y)
    with open(os.path.join(folder, "bacteria_" + dict["TASK_TITLE"] + "_task_tax_level_" + str(dict["TAX_LEVEL"]) + ".txt"), "r") as f:
        bacteria = f.readlines()
        bacteria = [b.strip("\n") for b in bacteria]

    # step 2: create data for simple models
    X_trains, X_tests, y_trains, y_tests = [], [], [], []

    for n in range(dict["K_FOLD"]):
        X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X), pd.Series(y), test_size=dict["TEST_SIZE"], shuffle=True)
        X_trains.append(X_train.index)
        X_tests.append(X_test.index)
        y_trains.append(y_train.index)
        y_tests.append(y_test.index)

    # step 3: create models according to parameters
    # send the place to change dir to relative to current path
    if dict["SVM"]:
        svm_model = SVMLearningModel()
        svm_model.fit(pd.DataFrame(X), pd.Series(y), X_trains, X_tests, y_trains, y_tests, dict["SVM_params"], weights,
                      bacteria, task_name_title=dict["TASK_TITLE"], relative_path_to_save_results=folder, pca_obj=pca_obj)
    if dict["XGB"]:
        xgb_model = XGBLearningModel()
        xgb_model.fit(pd.DataFrame(X), pd.Series(y), X_trains, X_tests, y_trains, y_tests, dict["XGB_params"], bacteria,
                      task_name_title=dict["TASK_TITLE"], relative_path_to_save_results=folder, pca_obj=pca_obj)
    if dict["NN"]:
        Net = models_nn[dict['NN_params']["model"]]
        _, _ = nn_main(X, y, dict['NN_params'], dict["TASK_TITLE"], Net, plot=True)
    if dict["NNI"]:
        params = nni.get_next_parameter()
        Net = models_nn[params["model"]]
        auc, acc = nn_main(X, y, params, dict["TASK_TITLE"], Net, plot=True)
        if dict["NNI_params"]["result_type"] == "acc":
            nni.report_final_result(acc)
        if dict["NNI_params"]["result_type"] == "auc":
            nni.report_final_result(auc)
        else:
            raise Exception

    os.chdir('..')


if __name__ == "__main__":
    folder = os.path.join("sderot_anxiety", 'ptsd_5_tax_5_csv_files')  # the name of the project folder

    project_folder_and_task = os.path.join('..', folder)  # adjust acorrding to runing folder
    otu_path = os.path.join(project_folder_and_task, 'OTU_merged_PTST_task.csv')
    mapping_path = os.path.join(project_folder_and_task, 'Tag_file_PTST_task.csv')
    pca_path = os.path.join('project_folder_and_task', 'Pca_obj_PTST_task.pkl')
    tax = str(5)
    k_fold = 17
    test_size = 0.2
    names = ["no anxiety", "anxiety"]
    # get params dictionary from file / create it here
    dict = {"TASK_TITLE": "PTSD time 5",  # the name of the task for plots titles...
            "FOLDER_TITLE": project_folder_and_task,  # creates the folder for the task we want to do, save results in it
            "TAX_LEVEL": tax,
            "CLASSES_NAMES": names,
            "SVM": True,
            "SVM_params": {'kernel': ['linear'],
                           'gamma': ['auto', 'scale'],
                           'C': [0.01, 0.1, 1, 10, 100, 1000],
                           "create_coeff_plots": True,
                           "CLASSES_NAMES": names,
                           "K_FOLD": k_fold,
                           "TEST_SIZE": test_size,
                           "TASK_TITLE": "sderot_anxiety"
                           },
            # if single option for each param -> single run, otherwise -> grid search.
            "XGB": True,
            "XGB_params": {'learning_rate': [0.1],
                           'objective': ['binary:logistic'],
                           'n_estimators': [1000],
                           'max_depth': [7],
                           'min_child_weight': [1],
                           'gamma': [1],
                           "create_coeff_plots": True,
                           "CLASSES_NAMES": names,
                           "K_FOLD": k_fold,
                           "TEST_SIZE": test_size,
                           "TASK_TITLE": "sderot_anxiety"
                           },  # if single option for each param -> single run, otherwise -> grid search.
            "NN": True,
            "NN_params": {
                        "hid_dim_0": 120,
                        "hid_dim_1": 160,
                        "reg": 0.68,
                        "lr": 0.001,
                        "test_size": 0.1,
                        "batch_size": 32,
                        "shuffle": 1,
                        "num_workers": 4,
                        "epochs": 150,
                        "optimizer": 'SGD',
                        "loss": 'MSE',
                        "model": 'tanh_b'
            },  # if single option for each param -> single run, otherwise -> grid search.
            "NNI": False,
            "NNI_params": {
                        "result_type": 'auc'
            },
            # enter to model params?  might want to change for different models..
            "K_FOLD": k_fold,
            "TEST_SIZE": test_size,
            #  ...... add whatever
            }
    multi_model_learning_main(folder, otu_path, mapping_path, pca_path, dict)
