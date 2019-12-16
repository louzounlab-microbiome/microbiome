import sys
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LearningMethods.learning_pipeline import learn
from pregnancy_diabetes.GDMDataLoader import GDMDataLoader

if __name__ == "__main__":
    learning_tasks = 'prognostic_diabetes_task_STOOL_tax_level_'
    bactria_as_feature_file = 'merged_GDM_tables_w_tax.csv'
    samples_data_file = 'ok.csv'
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