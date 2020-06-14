import os
import pandas as pd
from Microbiome_Intervention.TimeSerieDataSet import TimeSerieDataLoader
from Microbiome_Intervention.Bulit_network_from_file import create_G, simulation_auc
from Microbiome_Intervention.Significant_bacteria import check_if_bacteria_correlation_is_significant, \
    get_significant_beta_from_file
from Microbiome_Intervention.single_bacteria_nni_runner import run_single_bacteria


def main():
    """
    Run the data processing and machine learning of various types according to the selected parameters.
    """
    # parameters
    tax = 5
    test_size = 0.3
    k_fold = 20

    # create data for learning tasks
    create_regression_data = False
    create_regression_multi_bacteria_data = False
    create_time_serie_data = False  # maybe dont need
    create_multi_bacteria_time_serie_data = False

    # learning options
    run_all_types_of_regression = False
    run_interaction_network_structure_coef = False
    run_interaction_network_structure_change_in_data = False
    run_nn = True
    run_lstm = False
    run_nn_net = False

    # preprocces params
    preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1, 'normalization': 'log',
                       'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': (0, 'PCA')}

    # datasets files paths (files are organized by the needed format)
    datasets_to_files_maps = {
        "VitamineA": {"otu": os.path.join('..', 'Datasets', 'VitamineA', 'ok16_va_otu_table.csv'),
                      "map":  os.path.join('..', 'Datasets', 'VitamineA', 'metadata_ok16_va.csv')},

        "GDM": {"otu": os.path.join('..', 'Datasets', 'GDM', 'stool_otu_T.csv'),
                "map": os.path.join('..', 'Datasets', 'GDM', 'GDM_tables_stool_no_dups.csv')},

        "Allergy": {"otu": os.path.join('..', 'Datasets', 'Allergy', 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'),
                    "map": os.path.join('..', 'Datasets', 'Allergy', 'mf_merge_ok84_ok93_ok66_69_merged_by_RestoredSampleCode_as_ID_290119.csv')}
    }

    """
    "CRC": {"otu": "../Datasets/CRC/exported_feature-table_for_YoramL.csv",
                "map": '../Datasets/CRC/mapping file with data Baniyahs Merge.csv'},
    """
    # create file to save all AUC scores for all tasks
    auc_score_file_path = "all_Networks_AUC.csv"
    if not os.path.exists(auc_score_file_path):
        auc_score_file_df = pd.DataFrame(columns=["Network type", "Model", "Dataset", "Test size", "K fold", "AUC", "Time"])
        auc_score_file_df.to_csv(auc_score_file_path, index=False)

    for data_set in datasets_to_files_maps.keys():
        print(data_set)
        #if data_set != "Allergy": #"VitamineA" or data_set == "GDM":
        #    continue
        # step 1 - create data files for learning
        bactria_as_feature_file = datasets_to_files_maps[data_set]["otu"]
        samples_data_file = datasets_to_files_maps[data_set]["map"]
        data = TimeSerieDataLoader(title=data_set, taxnomy_level=tax)
        if create_regression_data or create_time_serie_data or \
                create_multi_bacteria_time_serie_data or create_regression_multi_bacteria_data:

            data.read_file(data_set, bactria_as_feature_file, samples_data_file, preprocess_prms, tax)

        if create_regression_data:
            data.create_reg_data()
        if create_regression_multi_bacteria_data:
            data.create_multi_bacteria_reg_data()
        if create_time_serie_data:
            data.create_time_serie_data()
        if create_multi_bacteria_time_serie_data:
            data.create_multi_bacteria_time_serie_data()

        # step 2 - regression

        if run_all_types_of_regression:
            #if data_set == "VitamineA" or data_set == "GDM":
            #    continue
            data.run_multi_type_regression(k_fold, test_size)

        if run_interaction_network_structure_coef:

            algorithms_list = ["linear regression", "ridge regression", "ard regression",
                               "lasso regression", "bayesian ridge regression", "svr regression"]  # "decision tree regression", "random forest regression"
            for reg_type in algorithms_list:
                print(reg_type)
                data.run_regression_coef_net(reg_type, k_fold, test_size)


                folder = os.path.join(data.load_and_save_path, "Significant_bacteria")
                if not os.path.exists(folder):
                    os.mkdir(folder)
                bacteria_path = os.path.join(data.load_and_save_path, "bacteria.txt")
                results_path = os.path.join(data.load_and_save_path, "run_all_types_of_regression_" +
                                           str(k_fold) + "_fold_test_size_" + str(test_size) + "_results_df.csv")
                # significant_bacteria = check_if_bacteria_correlation_is_significant(results_df_path, reg_type)
                return_path = get_significant_beta_from_file(results_path, reg_type, folder, bacteria_path)

        if run_interaction_network_structure_change_in_data:
            if data_set == "Allergy":
                data.run_regression_change_net()
                folder = data.load_and_save_path

                algorithms_list = ["linear regression", "ridge regression", "ard regression",
                                   "lasso regression", "bayesian ridge regression",
                                   "svr regression", "decision tree regression", "random forest regression"]

                for reg_type in algorithms_list:
                    task_title = data_set + "_" + reg_type.replace(" ", "_")
                    model_name = reg_type.replace(" ", "_") + "_"
                    df_title = os.path.join(folder, model_name + "interaction_network_change_in_data_df.csv")
                    pvalue = 0.001
                    create_G(task_title, df_title, folder, p_value=pvalue, simulation=True)

        # step 3 - neural networks
        if run_nn:
            data.run_nn(multi_or_single="multi")
            data.run_nn(multi_or_single="single")

        # step 4 - long short term memory networks
        if run_lstm:
            data.run_lstm(multi_or_single="multi")
            data.run_lstm(multi_or_single="single")

        if run_nn_net:
            data.run_nn_network()


if __name__ == "__main__":
    main()
