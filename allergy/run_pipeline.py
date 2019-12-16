from allergy.alg_before_treatment_data_set import AlgBeforeTreatmentDataLoader
from LearningMethods.learning_pipeline import learn
from allergy.alg_data_set import AlgDataLoader


def RunAlgBeforeTreatmentDataLoader(learning_tasks, bactria_as_feature_file, samples_data_file, tax = 6):
    # data_loader = should be a specific type of data set, not the abstract
    for task in learning_tasks:
        # using the task name create  different data sets
        data_loader = AlgBeforeTreatmentDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file, taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)

        learn(title=task, data_loader=data_loader,
              allow_printing=True, calculate_rhos=True,
              SVM=True, XGBOOST=True, NN=False,
              cross_validation=5, create_coeff_plots=True,
              check_all_parameters=True,
              svm_parameters={'kernel': 'linear', 'C': 1, 'gamma': 'scale'},
              xgb_parameters={'learning_rate': 0.01,
                                         'objective': 'binary:logistic',
                                         'n_estimators': 1000,
                                         'max_depth': 5,
                                         'min_child_weight': 5,
                                         'gamma': 1},
              create_pca_plots=True,
              test_size=0.2,
              edge_percent=1,
              BINARY=True)


if __name__ == "__main__":
    task = 'diagnostics task'
    bactria_as_feature_file = 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'
    samples_data_file = 'mf_merge_ok84_ok93_ok66_69_merged_by_RestoredSampleCode_as_ID_290119.csv'
    tax = 6

    allergy_dataset = AlgDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file,  taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)

    learn(title=task, data_loader=allergy_dataset,
          allow_printing=True, calculate_rhos=False,
          SVM=True, XGBOOST=False, NN=False,
          cross_validation=5, create_coeff_plots=True,
          check_all_parameters=True,
          svm_parameters={'kernel': 'linear', 'C': 10, 'gamma': 'auto'},
          xgb_parameters={'learning_rate': 0.01,
                          'objective': 'binary:logistic',
                          'n_estimators': 1000,
                          'max_depth': 5,
                          'min_child_weight': 5,
                          'gamma': 1},
          create_pca_plots=False,
          test_size=0.2,
          edge_percent=1,
          BINARY=True)


    learning_tasks = ["allergy_type_before_treatment_task"]  # ["health_before_treatment_task"]  #
    bactria_as_feature_file = 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'
    samples_data_file = 'mf_MG17_070519_No_Eggs_150919_for_dafna.csv'
    tax = 6

    for task in learning_tasks:
        # using the task name create  different data sets
        data_loader = AlgBeforeTreatmentDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file, taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)

        learn(title=task, data_loader=data_loader,
              allow_printing=True, calculate_rhos=False,
              SVM=True, XGBOOST=False, NN=False,
              cross_validation=5, create_coeff_plots=True,
              check_all_parameters=False,
              svm_parameters={'kernel': 'linear', 'C': 10, 'gamma': 'auto'},
              xgb_parameters={'learning_rate': 0.01,
                                         'objective': 'binary:logistic',
                                         'n_estimators': 1000,
                                         'max_depth': 5,
                                         'min_child_weight': 5,
                                         'gamma': 1},
              create_pca_plots=False,
              test_size=0.1,
              edge_percent=1,
              BINARY=False)
    """
    learning_tasks = ["health_before_treatment_task"]  #  ["allergy_type_before_treatment_task"]  #
    bactria_as_feature_file = 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'
    samples_data_file = 'mf_MG17_070519_No_Eggs_150919_for_dafna.csv'
    tax = 7

    for task in learning_tasks:
        # using the task name create  different data sets
        data_loader = AlgBeforeTreatmentDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file, taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)

        learn(title=task, data_loader=data_loader,
              allow_printing=True, calculate_rhos=True,
              SVM=True, XGBOOST=True, NN=False,
              cross_validation=5, create_coeff_plots=True,
              check_all_parameters=True,
              svm_parameters={'kernel': 'linear', 'C': 1, 'gamma': 'scale'},
              xgb_parameters={'learning_rate': 0.01,
                                         'objective': 'binary:logistic',
                                         'n_estimators': 1000,
                                         'max_depth': 5,
                                         'min_child_weight': 5,
                                         'gamma': 1},
              create_pca_plots=True,
              BINARY=True)
    """

    """
    learning_tasks = ["health_before_treatment_task"]
    bactria_as_feature_file = 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'
    samples_data_file = 'mf_merge_ok84_ok93_ok66_69_merged_by_RestoredSampleCode_as_ID_290119.csv'
    tax = 6
    # run(learning_tasks, bactria_as_feature_file, samples_data_file, tax)
    """
    # Pre process - mostly not needed
    allow_printing = False  # do you want printing to screen?
    perform_regression = False  # run normalization to drew plots
    calculate_rhos = False  # calculate correlation between reaction to treatment to the tags
    """

    # data_loader = should be a specific type of data set, not the abstract
    for task in learning_tasks:
        # using the task name create  different data sets
        data_loader = AlgDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file, taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)

        learn(title=task, data_loader=data_loader,
              allow_printing=True, calculate_rhos=True,
              SVM=True, XGBOOST=True, NN=False,
              cross_validation=5, create_coeff_plots=True,
              check_all_parameters=True,
              svm_parameters={'kernel': 'linear', 'C': 1, 'gamma': 'scale'},
              xgb_parameters={'learning_rate': 0.01,
                                         'objective': 'binary:logistic',
                                         'n_estimators': 1000,
                                         'max_depth': 5,
                                         'min_child_weight': 5,
                                         'gamma': 1},
              create_pca_plots=False)
    """
