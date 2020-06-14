import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from Microbiome_Intervention.Create_learning_data_from_data_set import create_data_for_single_bacteria_model_learning, \
    create_data_as_time_serie_for_signal_bacteria_model_learning, \
    create_data_as_time_serie_for_multi_bacteria_model_learning, get_adapted_X_y_for_wanted_learning_task, \
    create_data_for_multi_bacteria_model_learning
from Microbiome_Intervention.bacteria_network_nni_runner import run_nn_bacteria_network
from Microbiome_Intervention.multi_bacteria_nni_runner import run_multi_bacteria
from Microbiome_Intervention.Simple_prediction_of_natural_dynamics import create_data_frames, \
    predict_interaction_network_structure_using_change_in_data, run_all_types_of_regression, \
    predict_interaction_network_structure_using_coeffs, conclude_results, \
    predict_interaction_network_structure_using_change_in_data_auc_calc_trail
from Microbiome_Intervention.single_bacteria_nni_runner import run_single_bacteria


class TimeSerieDataLoader:
    def __init__(self, title, taxnomy_level):
        self._data_set_folder = title
        self._taxnomy_level = taxnomy_level
        self.load_and_save_path = os.path.join("Microbiome_Intervention", title, "tax=" + str(taxnomy_level))
        if not os.path.exists(self.load_and_save_path):
            os.makedirs(self.load_and_save_path)
        self.reg_X_y_files_pkl_path = "X_y_files_path.pkl"
        self.time_serie_X_y_for_all_bacteria_path = 'time_serie_X_y_for_all_bacteria.csv'
        self.reg_X_y_for_all_bacteria_path = 'X_y_for_all_bacteria.csv'
        self.time_serie_X_y_files_pkl_path = "time_serie_X_y_files_path.pkl"
        self.multi_bacteria_time_serie_X_y_files_pkl_path = "multi_bacteria_time_serie_files_names.pkl"
        self.bacteria_list_path = "bacteria.txt"

    def read_file(self, title, bactria_as_feature_file, samples_data_file, preprocess_prms, tax):
        otu_and_mapping_file = CreateOtuAndMappingFiles(bactria_as_feature_file, samples_data_file)
        otu_and_mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=False)
        preproccessed_data = otu_and_mapping_file.otu_features_df
        tag_file = otu_and_mapping_file.tags_df.join(otu_and_mapping_file.extra_features_df, how='outer')

        otu_path, tag_path, pca_path = otu_and_mapping_file.csv_to_learn(title + '_task', os.path.join('..', "Datasets", title),
                                                               tax=tax, pca_n=otu_and_mapping_file.pca)
        preproccessed_data.index = [str(id) for id in preproccessed_data.index]
        tag_file.index = [str(id) for id in tag_file.index]

        index_to_id_map = {}
        id_to_features_map = {}
        for i, row in enumerate(preproccessed_data.values):
            id_to_features_map[preproccessed_data.index[i]] = row
            index_to_id_map[i] = preproccessed_data.index[i]

        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map

        data_ids_list = preproccessed_data.index.tolist()
        tag_ids_list = tag_file.index.tolist()
        ids_list = [id for id in data_ids_list if id in tag_ids_list]
        self._ids_list = ids_list

        patient_column = 'Patient'
        id_to_patient_map = {}
        for sample in ids_list:
            child_num = tag_file.loc[sample, patient_column]
            id_to_patient_map[sample] = child_num

        # create time series for each child
        child_to_ids_map = {child: [] for child in set(id_to_patient_map.values())}
        for key, val in id_to_patient_map.items():
            child_to_ids_map[val].append(key)
        for serie in child_to_ids_map.values():
            serie.sort()

        time_column = 'Time'
        id_to_time_map = {}
        for sample in ids_list:
            period = tag_file.loc[sample, time_column]
            id_to_time_map[sample] = period
        periods = list(set(id_to_time_map.values()))
        periods.sort()
        period_to_index = {p: i for i, p in enumerate(periods)}


        ids_list = []
        features_list = []
        for key, val in id_to_features_map.items():
            ids_list.append(key[:-1])
            features_list.append(list(val))

        with open(os.path.join(self.load_and_save_path, "bacteria.txt"), "w") as b_file:
            for b in otu_and_mapping_file.bacteria:
                b_file.write(b + "\n")

        self.mapping_file = tag_file
        self.ids_list = ids_list
        self.features_list = features_list
        self.data_set_tax_path = tax
        self.bacteria = otu_and_mapping_file.bacteria
        self.period_column = time_column
        self.id_to_period_map = id_to_time_map
        self.id_to_participant_map = id_to_patient_map
        self.id_to_features_map = id_to_features_map
        self.participant_to_ids_map = child_to_ids_map
        self.period_to_index = period_to_index

    def create_reg_data(self):
        X_y_files_path = \
            create_data_for_single_bacteria_model_learning(self.mapping_file, self.bacteria,
                                                           self.period_column, self.id_to_period_map,
                                                           self.id_to_participant_map, self.id_to_features_map,
                                                           self.participant_to_ids_map, self.period_to_index,
                                                           self.ids_list, self.features_list, self.load_and_save_path)

        pickle.dump(X_y_files_path, open(os.path.join(self.load_and_save_path, self.reg_X_y_files_pkl_path), "wb"))

    def create_multi_bacteria_reg_data(self):
        X_y_for_all_bacteria_path = \
            create_data_for_multi_bacteria_model_learning(self.mapping_file, self.data_set_tax_path, self.bacteria,
                                                           self.period_column, self.id_to_period_map,
                                                           self.id_to_participant_map, self.id_to_features_map,
                                                           self.participant_to_ids_map, self.period_to_index,
                                                           self.ids_list, self.features_list, self.load_and_save_path)

    def create_time_serie_data(self):
        X_y_files_path = \
            create_data_as_time_serie_for_signal_bacteria_model_learning(self.mapping_file, self.data_set_tax_path, self.bacteria,
                                                           self.period_column, self.id_to_period_map,
                                                           self.id_to_participant_map, self.id_to_features_map,
                                                           self.participant_to_ids_map, self.period_to_index,
                                                                         self.load_and_save_path)

        pickle.dump(X_y_files_path, open(os.path.join(self.load_and_save_path, self.time_serie_X_y_files_pkl_path), "wb"))

    def create_multi_bacteria_time_serie_data(self):
        X_y_files_path = \
            create_data_as_time_serie_for_multi_bacteria_model_learning(self.mapping_file, self.data_set_tax_path,
                                                                         self.bacteria,
                                                                         self.period_column, self.id_to_period_map,
                                                                         self.id_to_participant_map,
                                                                         self.id_to_features_map,
                                                                         self.participant_to_ids_map,
                                                                         self.period_to_index,
                                                                        self.load_and_save_path)

        pickle.dump(X_y_files_path, open(os.path.join(self.load_and_save_path, self.multi_bacteria_time_serie_X_y_files_pkl_path), "wb"))

    def run_multi_type_regression(self, k_fold, test_size):
        task = "run_all_types_of_regression"
        X_y_files_path = pickle.load(open(os.path.join(self.load_and_save_path, self.reg_X_y_files_pkl_path), "rb"))

        all_times_all_bact_results_path = os.path.join(self.load_and_save_path, task + "_" + str(k_fold) +
                                                       "_fold_test_size_" + str(test_size)
                                                       + "_results_df.csv")
        important_bacteria_reults_path = os.path.join(self.load_and_save_path, task + "_" + str(k_fold) +
                                                      "_fold_test_size_" + str(test_size)
                                                      + "_significant_bacteria_prediction_results_df.csv")
        conclusionss_path = os.path.join(self.load_and_save_path, task + "_" + str(k_fold) +
                                         "_fold_test_size_" + str(test_size) + "_conclusions.csv")

        # create data frames
        with open(os.path.join(self.load_and_save_path, "bacteria.txt"), "r") as b_file:
            bacteria = b_file.readlines()
            bacteria = [b.rstrip() for b in bacteria]

        create_data_frames(all_res_path=all_times_all_bact_results_path,
                           important_bacteria_reults_path=important_bacteria_reults_path)

        with open(os.path.join(self.load_and_save_path, X_y_files_path), "r") as file:
            paths = file.readlines()
            paths = [p.strip('\n') for p in paths]


        for i, [bact, path] in enumerate(zip(bacteria, paths)):
            print(str(i) + " / " + str(len(bacteria)))
            all_times_all_bacteria_all_models_results_df = pd.read_csv(all_times_all_bact_results_path)

            X_trains, X_tests, y_trains, y_tests, name = \
                get_adapted_X_y_for_wanted_learning_task(self.load_and_save_path, path, "regular", k_fold, test_size)
            run_all_types_of_regression(X_trains, X_tests, y_trains, y_tests, i,
                                        all_times_all_bacteria_all_models_results_df,
                                        all_times_all_bact_results_path, bact)

        conclude_results(all_times_all_bacteria_all_models_results_df, 0, conclusionss_path)

    def run_regression_coef_net(self, reg_type, k_fold, test_size):
        task = "interaction_network_structure_coef"
        X_y_files_path = pickle.load(open(os.path.join(self.load_and_save_path, self.reg_X_y_files_pkl_path), "rb"))
        all_times_all_bact_results_path = os.path.join(self.load_and_save_path,
                                                       reg_type.replace(" ", "_") + "_" + task + "_" + str(k_fold) +
                                                       "_fold_test_size_" + str(test_size)
                                                       + "_results_df.csv")
        important_bacteria_reults_path = os.path.join(self.load_and_save_path, reg_type.replace(" ", "_") + "_" + task + "_" + str(k_fold) +
                                                      "_fold_test_size_" + str(test_size)
                                                      + "_significant_bacteria_prediction_results_df.csv")

        with open(os.path.join(self.load_and_save_path, "bacteria.txt"), "r") as b_file:
            bacteria = b_file.readlines()
            bacteria = [b.rstrip() for b in bacteria]

        create_data_frames(all_res_path=all_times_all_bact_results_path,
                           important_bacteria_reults_path=important_bacteria_reults_path)

        with open(os.path.join(self.load_and_save_path, X_y_files_path), "r") as file:
            paths = file.readlines()
            paths = [p.strip('\n') for p in paths]

        train_binary_significant_from_all_bacteria = []
        test_b_list_from_all_bacteria = []

        for i, [bact, path] in enumerate(zip(bacteria, paths)):
            print(str(i) + " / " + str(len(bacteria)))

            all_times_all_bacteria_all_models_results_df = pd.read_csv(all_times_all_bact_results_path)
            important_bacteria_reults_df = pd.read_csv(important_bacteria_reults_path)
            X_trains, X_tests, y_trains, y_tests, name = \
                get_adapted_X_y_for_wanted_learning_task(self.load_and_save_path, path, "regular", k_fold, test_size)

            results_df, train_binary_significant_list, test_b_list = \
                predict_interaction_network_structure_using_coeffs(X_trains, X_tests, y_trains, y_tests, i,
                                                                   all_times_all_bacteria_all_models_results_df,
                                                                   all_times_all_bact_results_path,
                                                                   important_bacteria_reults_df,
                                                                   important_bacteria_reults_path, bact, bacteria,
                                                                   reg_type)
            # save bacteria y true nd y pred
            train_binary_significant_from_all_bacteria.append(list(np.array(train_binary_significant_list).flat))
            test_b_list_from_all_bacteria.append(list(np.array(test_b_list).flat))

        train_binary_significant_from_all_bacteria = list(np.array(train_binary_significant_from_all_bacteria).flat)
        test_b_list_from_all_bacteria = list(np.array(test_b_list_from_all_bacteria).flat)
        total_auc = roc_auc_score(y_true=train_binary_significant_from_all_bacteria,
                                  y_score=test_b_list_from_all_bacteria)

        Networks_AUC_df = pd.read_csv("all_Networks_AUC.csv")
        data_set = self.load_and_save_path.split(os.path.sep)[0]
        Networks_AUC_df.loc[len(Networks_AUC_df)] = ["coefficients", reg_type, data_set, test_size, k_fold, total_auc,
                                                     datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S")]
        Networks_AUC_df.to_csv("all_Networks_AUC.csv", index=False)

    def run_regression_change_net(self):
        task = "interaction_network_structure_change_in_data"
        with open(os.path.join(self.load_and_save_path, "bacteria.txt"), "r") as b_file:
            bacteria = b_file.readlines()
            bacteria = [b.rstrip() for b in bacteria]

        # predict_interaction_network_structure_using_change_in_data(bacteria, self.load_and_save_path)
        predict_interaction_network_structure_using_change_in_data_auc_calc_trail(bacteria, self.load_and_save_path, self._data_set_folder)

    def run_nn(self, multi_or_single):
        if multi_or_single == "multi":
            run_multi_bacteria(self.load_and_save_path, self.time_serie_X_y_for_all_bacteria_path,
                               self.reg_X_y_for_all_bacteria_path, NN_or_RNN="NN",
                               results_df_title="multi_bacteria_grid_search_df")
        elif multi_or_single == "single":
            bacteria_sorted_by_mse = "bacteria_sorted_by_mse.csv"
            best_bacteria_path = "run_all_types_of_regression_10_fold_test_size_0.5_bacteria_conclusions.csv"
            X_y_files_list_path = 'multi_bacteria_time_serie_files_names.txt'
            run_single_bacteria(self.load_and_save_path, bacteria_sorted_by_mse, best_bacteria_path, X_y_files_list_path,
                                NN_or_RNN="NN", results_df_title="single_bacteria_grid_search_df")

    def run_lstm(self, multi_or_single):
        if multi_or_single == "multi":
            run_multi_bacteria(self.load_and_save_path, self.time_serie_X_y_for_all_bacteria_path,
                               self.reg_X_y_for_all_bacteria_path, NN_or_RNN="RNN",
                               results_df_title="multi_bacteria_grid_search_df")
        elif multi_or_single == "single":
            bacteria_sorted_by_mse = "bacteria_sorted_by_mse.csv"
            best_bacteria_path = "run_all_types_of_regression_10_fold_test_size_0.5_bacteria_conclusions.csv"
            X_y_files_list_path = 'multi_bacteria_time_serie_files_names.txt'
            run_single_bacteria(self.load_and_save_path, bacteria_sorted_by_mse, best_bacteria_path,
                                X_y_files_list_path,
                                NN_or_RNN="RNN", results_df_title="single_bacteria_grid_search_df")

    def run_nn_network(self):
        run_nn_bacteria_network(self.load_and_save_path, self._data_set_folder)
