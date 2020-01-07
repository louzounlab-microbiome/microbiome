import os

from sklearn.model_selection import train_test_split

from LearningMethods.svm_learning_model import SVMLearningModel
from LearningMethods.xgb_learning_model import XGBLearningModel

import pandas as pd
import numpy as np

from Plot import draw_X_y_rhos_calculation_figure, PCA_t_test, plot_data_3d, plot_data_2d, pickle


def read_otu_and_mapping_files(otu_path, mapping_path):
    otu_file = pd.read_csv(otu_path)
    mapping_file = pd.read_csv(mapping_path)
    X = otu_file.set_index("ID").values
    y = mapping_file["Tag"]
    return np.array(X), np.array(y)


def get_weights(y):
    classes_sum = [np.sum(np.array(y) == unique_class) for unique_class in
                   np.unique(np.array(y))]
    classes_ratio = [1 - (a / sum(classes_sum)) for a in classes_sum]
    weights_map = {a: classes_ratio[a] for a in set(y)}
    return weights_map


def main(folder, otu_path, mapping_path, pca_path, dict):
    # step 1: get data from files
    X, y = read_otu_and_mapping_files(otu_path, mapping_path)
    pca_obj = pickle.load(open(pca_path, "rb"))
    weights = get_weights(y)
    with open(os.path.join("..", folder, "bacteria_tax_level_" + str(dict["TAX_LEVEL"]) + ".txt"), "r") as f:
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
    if dict["SVM"]:
        svm_model = SVMLearningModel()
        svm_model.fit(pd.DataFrame(X), pd.Series(y), X_trains, X_tests, y_trains, y_tests, dict["SVM_params"], weights,
                      pca_obj, bacteria, task_name_folder=dict["FOLDER_TITLE"] + dict["TAX_LEVEL"], project_folder=folder)
    if dict["XGB"]:
        xgb_model = XGBLearningModel()
        xgb_model.fit(pd.DataFrame(X), pd.Series(y), X_trains, X_tests, y_trains, y_tests, dict["SVM_params"], weights,
                      pca_obj, bacteria, task_name_folder="sderot_anxiety_tax_level"+dict["TAX_LEVEL"])

    if dict["NN"]:
        # nn_model = NNLearningModel()
        # nn_model.fit(X, y, dict["NN_params"])
        pass

    os.chdir('..')


if __name__ == "__main__":
    folder = 'sderot_anxiety' # the name of the project folder
    otu_path = os.path.join('..', folder, 'OTU_merged_prognostic_PTSD_task_tax_level_6_pca_2.csv')
    mapping_path = os.path.join('..', folder, 'Tag_file_prognostic_PTSD_task_tax_level_6_pca_2.csv')
    pca_path = os.path.join('..', folder, 'Pca_obj_prognostic_PTSD_task_tax_level_6_pca_2.pkl')
    k_fold = 17
    test_size = 0.2
    names = ["no anxiety", "anxiety"]
    # get params dictionary from file / create it here
    dict = {"TASK_TITLE": "sderot_anxiety",  # the name of the task for plots titles...
            "FOLDER_TITLE": folder,  # creates the folder for the task we want to do, save results in it
            "TAX_LEVEL": str(6),
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
            "XGB_params": {},  # if single option for each param -> single run, otherwise -> grid search.
            "NN": True,
            "NN_params": {},  # if single option for each param -> single run, otherwise -> grid search.
            "NNI": True,
            "NNI_params": {},

            # enter to model params?  might want to change for different models..
            "K_FOLD": k_fold,
            "TEST_SIZE": test_size,
            #  ...... add whatever
            }
    main(folder, otu_path, mapping_path, pca_path, dict)
