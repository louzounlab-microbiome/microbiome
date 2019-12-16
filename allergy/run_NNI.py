from allergy.alg_before_treatment_data_set import AlgBeforeTreatmentDataLoader
from LearningMethods.NNI_example_file import main_nni

if __name__ == "__main__":
    task = "health_before_treatment_task"
    bactria_as_feature_file = 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'
    samples_data_file = 'mf_MG17_070519_No_Eggs_150919_for_dafna.csv'
    tax = 6
    folder = "NNI"

    data_loader = AlgBeforeTreatmentDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                               samples_data_file=samples_data_file, taxnomy_level=tax,
                                               allow_printing=False, perform_anna_preprocess=False)
    X, y = data_loader.get_X_y_for_nni(task)
    main_nni(X, y, task, folder, result_type="auc")

