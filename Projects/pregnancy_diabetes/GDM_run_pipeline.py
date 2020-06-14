import sys
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LearningMethods.multi_model_learning import main
from GDMDataLoader import GDMDataLoader
from LearningMethods.learning_pipeline import learn

def learning_pipeline(otu_path, mapping_path, pca_path, tax, site, trimester):
    folder = 'pregnancy_diabetes' + '_'+ trimester + '_'+ site # the name of the project folder
    k_fold = 5
    test_size = 0.2
    names = ["Control", "GDM"]
    # get params dictionary from file / create it here
    dict = {"TASK_TITLE": 'GDM_tax_level_'+ str(tax)+'from_'+ site+'trimester_'+trimester,  # the name of the task for plots titles...
            "FOLDER_TITLE": folder,  # creates the folder for the task we want to do, save results in it
            "TAX_LEVEL": str(tax),
            "SITE": site,
            "CLASSES_NAMES": names,
            "SVM": True,
            "SVM_params": {'kernel': ['linear'],
                           'gamma': ['scale'],
                           'C': [0.01, 0.1, 1, 10, 100, 1000],
                           "create_coeff_plots": False,
                           "CLASSES_NAMES": names,
                           "K_FOLD": k_fold,
                           "TEST_SIZE": test_size,
                           "TASK_TITLE": "pregnancy_diabetes_" + trimester + '_'+ site 
                           },
            # if single option for each param -> single run, otherwise -> grid search.
            "XGB": False,
            "XGB_params": {'learning_rate': [0.01, 0.05, 0.1],
                           'objective': ['binary:logistic'],
                           'n_estimators': [1000],
                           'max_depth': [3,5,7,9],
                           'min_child_weight': [1,5,9],
                           'gamma': [0.0, 0.5, 1, 5, 9],
                           "create_coeff_plots": True,
                           "CLASSES_NAMES": names,
                           "K_FOLD": k_fold,
                            "TEST_SIZE": test_size,
                           "TASK_TITLE": "pregnancy_diabetess_" + trimester + '_'+ site
                           },  # if single option for each param -> single run, otherwise -> grid search.
            "NN": False,
            "NN_params": {
                        "hid_dim_0": 120,
                        "hid_dim_1": 160,
                        "reg": 0.5,
                        "lr": 0.001,
                        "test_size": 0.1,
                        "batch_size": 32,
                        "shuffle": 1,
                        "num_workers": 4,
                        "epochs": 150,
                        "optimizer": 'SGD',
                        "loss": 'MSE',
                        "model": 'sigmoid_b'
            },  # if single option for each param -> single run, otherwise -> grid search.
            "NNI": True,
            "NNI_params": {
                        "result_type": 'auc'
            },
            # enter to model params?  might want to change for different models..
            "K_FOLD": k_fold,
            "TEST_SIZE": test_size,
            #  ...... add whatever
            }
    main(folder, otu_path, mapping_path, pca_path, dict)


if __name__ == "__main__":
    learning_tasks = 'prognostic_diabetes_T1_task_SALIVA_tax_level_'
    bactria_as_feature_file = 'DB/GDM_taxonomy.csv'
    samples_data_file = 'DB/samples_metadata.csv'
    if len(sys.argv)==2:
        #only one taxonomy level 
        tax = int(sys.argv[1])
        taxonomy_levels = [tax]
    else:
        #insert list of taxonomy levels 
        taxonomy_levels = [5,6]
    for tax in taxonomy_levels:
        # using the task name create  different data sets
        data_loader = GDMDataLoader(title=str(learning_tasks+str(tax)), bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file, taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)

        #mark the learning task wanted         
        learn(title=str(learning_tasks+str(tax)), data_loader=data_loader,  
              allow_printing=False, calculate_rhos=False,
              SVM=True, XGBOOST=True, NN=False,
              cross_validation=7, create_coeff_plots=False,
              check_all_parameters=True,
              svm_parameters={'kernel': 'linear', 'C': 100, 'gamma': 'auto'},
              xgb_parameters={'learning_rate': 0.1,
                              'objective': 'binary:logistic',
                              'n_estimators': 1000,
                              'max_depth': 7,
                              'min_child_weight': 1,
                              'gamma': 1},
              create_pca_plots=False,
              test_size=0.15,
              edge_percent=1,
              BINARY=True)