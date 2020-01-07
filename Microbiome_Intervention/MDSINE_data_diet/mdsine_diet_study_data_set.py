import pickle
from Microbiome_Intervention import TimeSerieDataLoader
from Microbiome_Intervention.create_learning_data_from_data_set import create_data_for_markob_model_learning, \
    create_data_for_signal_bacteria_model_learning
from Microbiome_Intervention.naive_prediction_of_natural_dynamics import preform_reggression_learning
from Microbiome_Intervention.significant_bacteria import check_if_bacteria_correlation_is_significant, \
    get_significant_beta_from_file
from infra_functions.load_merge_otu_mf import OtuMfHandler
import os
from infra_functions.preprocess import preprocess_data

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
n_components = 20


class MDSINE_Study_DataLoader(TimeSerieDataLoader):
    def __init__(self, data_name, bactria_as_feature_file, samples_data_file, taxnomy_level,
                 create_regression_data, create_time_serie_data):
        super().__init__(data_name, taxnomy_level)
        if create_regression_data or create_time_serie_data:
            self._read_file(data_name, bactria_as_feature_file, samples_data_file)
        if create_regression_data:
            self.create_reg_data()
        if create_time_serie_data:
            self.create_time_serie_data()

    def _read_file(self, title, bactria_as_feature_file, samples_data_file):
        tax = self.load_and_save_path
        if not os.path.exists(tax):
            os.mkdir(tax)

        OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, bactria_as_feature_file),
                             os.path.join(SCRIPT_DIR, samples_data_file),
                             from_QIIME=False, id_col='#SampleID', taxonomy_col='taxonomy')

        preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=self._taxnomy_level,
                                             taxonomy_col='taxonomy',
                                             preform_taxnomy_group=False)
        self._preproccessed_data = preproccessed_data

        bacteria = list(preproccessed_data.columns)
        with open(os.path.join(tax, "bacteria.txt"), "w") as file:
            for b in bacteria:
                file.write(b + '\n')

        preproccessed_data_ids_list = preproccessed_data.index.tolist()[1:]
        mapping_file_ids_list = OtuMf.mapping_file.index.tolist()
        ids_list = mapping_file_ids_list
        self._ids_list = ids_list

        index_to_id_map = {}
        id_to_features_map = {}
        for i, row in enumerate(preproccessed_data.values):
            id_to_features_map[preproccessed_data.index[i]] = row
            index_to_id_map[i] = preproccessed_data.index[i]

        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map



        period_column = "measurementID"
        id_to_period_map = {}
        for sample in ids_list:
            period = OtuMf.mapping_file.loc[sample, period_column]
            id_to_period_map[sample] = period
        periods = list(set(id_to_period_map.values()))
        periods.sort()
        period_to_index = {p: i for i, p in enumerate(periods)}

        patient_column = "subjectID"
        id_to_patient_map = {}
        for sample in ids_list:
            patient_num = OtuMf.mapping_file.loc[sample, patient_column]
            id_to_patient_map[sample] = patient_num

        # create time series for each child
        patient_to_ids_map = {child: [] for child in set(id_to_patient_map.values())}
        for key, val in id_to_patient_map.items():
            patient_to_ids_map[val].append(key)
        for serie in patient_to_ids_map.values():
            serie.sort()

        self.otu_mf = OtuMf
        self.data_set_tax_path = tax
        self.bacteria = bacteria
        self.period_column = period_column
        self.id_to_period_map = id_to_period_map
        self.id_to_participant_map = id_to_patient_map
        self.id_to_features_map = id_to_features_map
        self.participant_to_ids_map = patient_to_ids_map
        self.period_to_index = period_to_index


if __name__ == "__main__":
    data = "MDSINE_data_diet"
    bactria_as_feature_file = 'counts.csv'
    samples_data_file = 'metadata.csv'
    tax = 7
    create_regression_data = True
    create_time_serie_data = True
    run_regression = False
    run_rnn = True
    run_lstm = True
    cross_val = 10
    test_size = 0.5

    cdiff_dataset = MDSINE_Study_DataLoader(data, bactria_as_feature_file, samples_data_file, tax,
                                    create_regression_data, create_time_serie_data)
    cdiff_dataset.run(run_regression, run_rnn, run_lstm, cross_validation=10, test_size=0.5)

    calc_significant_bacteria = True
    all_times_all_bact_results_path = "all_times_all_bacteria_best_models_results_df.csv"
    # if we want to run on a single model, change to a different csv file with the same structure
    if calc_significant_bacteria:
        folder = os.path.join("tax=" + str(tax), "Significant_bacteria")
        for algo in ["ard regression"]:
            results_df_path = os.path.join("tax=" + str(tax), all_times_all_bact_results_path)
            significant_bacteria = check_if_bacteria_correlation_is_significant(results_df_path, algo)
            get_significant_beta_from_file(results_df_path, algo, significant_bacteria, folder)
