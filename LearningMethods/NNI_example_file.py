import nni
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pregnancy_diabetes.GDMDataLoader import GDMDataLoader
from LearningMethods.nn_for_nni import nn_nni_main


def main_nni(X, y, title, folder, result_type):
    #enable if you want to run nni, disable if not!
    #params = nni.get_next_parameter()
    
    #default parameter 
    params = {
        "hid_dim_0": 80,
        "hid_dim_1": 140,
        "reg": 0.01,
        "dims": [20, 40, 60, 2],
        "lr": 0.001,
        "test_size": 0.2,
        "batch_size": 32,
        "shuffle": 1,
        "num_workers": 4,
        "epochs": 100,
        "optimizer": 'default',
        "loss": 'default'
    }

    print(params)
    #run the default neural network architechture in LearningMethods.nn_for_nni
    auc, loss = nn_nni_main(X, y, params, title, folder)
    
    if result_type == "loss":
        nni.report_final_result(loss)
    if result_type == "auc":
        nni.report_final_result(auc)
    else:
        raise Exception


if __name__ == "__main__":
    #task name
    task = "prognostic_diabetes_task"
    #files name
    bactria_as_feature_file = '../pregnancy_diabetes/merged_GDM_tables_w_tax.csv'
    samples_data_file = '../pregnancy_diabetes/ok.csv'
    #taxonomy level (input as paramter if needed)
    tax = int(sys.argv[1])
    #specified the folder run from 
    folder = "LearningMethods"
    #create data loader of the corresponding task
    data_loader = GDMDataLoader( title=task, bactria_as_feature_file=bactria_as_feature_file,
                                                   samples_data_file=samples_data_file, taxnomy_level=tax,
                                                   allow_printing=True, perform_anna_preprocess=False)
    #extract inputs and label
    X, y = data_loader.get_X_y_for_nni(task)
    #run main function
    main_nni(X, y, task, folder, result_type="auc")

