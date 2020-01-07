import os
import pickle

from Microbiome_Intervention.create_learning_data_from_data_set import create_data_for_signal_bacteria_model_learning, \
    create_data_as_time_serie_for_signal_bacteria_model_learning
from Microbiome_Intervention.naive_prediction_of_natural_dynamics import preform_reggression_learning
from Microbiome_Intervention.rnn_prediction_of_natural_dynamics import preform_time_serie_learning


class TimeSerieDataLoader:
    def __init__(self, title, taxnomy_level):
        self._data_set_folder = title
        self._taxnomy_level = taxnomy_level
        self.load_and_save_path = os.path.join(title, "tax=" + str(taxnomy_level))
        self.reg_X_y_files_pkl_path = "X_y_files_path.pkl"
        self.time_serie_X_y_files_pkl_path = "time_serie_X_y_files_path.pkl"
        self.bacteria_list_path = "bacteria.txt"

    def _read_file(self, title, bactria_as_feature_file, samples_data_file):
        """
        # create
        self.otu_mf
        self.data_set_tax_path
        self.bacteria
        self.period_column
        self.id_to_period_map
        self.id_to_participant_map
        self.id_to_features_map
        self.participant_to_ids_map
        self.period_to_index
        """
        raise NotImplemented

    def create_reg_data(self):
        X_y_files_path = \
            create_data_for_signal_bacteria_model_learning(self.otu_mf, self.data_set_tax_path, self.bacteria,
                                                           self.period_column, self.id_to_period_map,
                                                           self.id_to_participant_map, self.id_to_features_map,
                                                           self.participant_to_ids_map, self.period_to_index)

        pickle.dump(X_y_files_path, open(os.path.join(self.load_and_save_path, self.reg_X_y_files_pkl_path), "wb"))

    def create_time_serie_data(self):
        X_y_files_path = \
            create_data_as_time_serie_for_signal_bacteria_model_learning(self.otu_mf, self.data_set_tax_path, self.bacteria,
                                                           self.period_column, self.id_to_period_map,
                                                           self.id_to_participant_map, self.id_to_features_map,
                                                           self.participant_to_ids_map, self.period_to_index)

        pickle.dump(X_y_files_path, open(os.path.join(self.load_and_save_path, self.time_serie_X_y_files_pkl_path), "wb"))

    def run(self, run_regression, run_rnn, run_lstm, cross_validation, test_size):
        with open(os.path.join(self.load_and_save_path, self.bacteria_list_path), "r") as b_file:
            bacteria = b_file.readlines()
            bacteria = [b.rstrip() for b in bacteria]

        if run_regression:
            # create all models for each bacterium to predict its change, through the
            # previous general state (all bacteria values)
            # 1-"run_all_types_of_regression" 2-"interaction_network_structure" 3-"hidden_measurements"
            task = "hidden_measurements"
            X_y_files_path = pickle.load(open(os.path.join(self.load_and_save_path, self.reg_X_y_files_pkl_path), "rb"))
            preform_reggression_learning(self.load_and_save_path, bacteria, task, X_y_files_path, cross_validation, test_size)
        if run_rnn:
            X_y_files_path = pickle.load(open(os.path.join(self.load_and_save_path, self.time_serie_X_y_files_pkl_path), "rb"))
            preform_time_serie_learning(self.load_and_save_path, bacteria, X_y_files_path, cross_validation, test_size)
        if run_lstm:
            X_y_files_path = pickle.load(open(os.path.join(self.load_and_save_path, self.time_serie_X_y_files_pkl_path), "rb"))
            preform_time_serie_learning(self.load_and_save_path, bacteria, X_y_files_path, cross_validation, test_size)