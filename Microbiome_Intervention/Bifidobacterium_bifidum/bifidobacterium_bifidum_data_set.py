import pickle
from Microbiome_Intervention import TimeSerieDataLoader
from Microbiome_Intervention.create_learning_data_from_data_set import create_data_for_markob_model_learning, \
    create_data_for_signal_bacteria_model_learning
from Microbiome_Intervention.naive_prediction_of_natural_dynamics import preform_learning
from Microbiome_Intervention.significant_bacteria import check_if_bacteria_correlation_is_significant, \
    get_significant_beta_from_file
from infra_functions.load_merge_otu_mf import OtuMfHandler
import os
from infra_functions.preprocess import preprocess_data

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
n_components = 20


class Bifidobacterium_bifidum_DataLoader(TimeSerieDataLoader):
    def __init__(self, title, bactria_as_feature_file, samples_data_file, taxnomy_level, created_data):
        super().__init__(title, bactria_as_feature_file, samples_data_file, taxnomy_level, created_data)

    def _read_file(self, title, bactria_as_feature_file, samples_data_file, created_data):
        tax = 'tax=' + str(self._taxnomy_level)
        if not os.path.exists(tax):
            os.mkdir(tax)

        OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, bactria_as_feature_file),
                             os.path.join(SCRIPT_DIR, samples_data_file),
                             from_QIIME=False, id_col='#OTU ID', taxonomy_col='taxonomy')

        preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=self._taxnomy_level,
                                             taxonomy_col='taxonomy',
                                             preform_taxnomy_group=True)
        self._preproccessed_data = preproccessed_data

        bacteria = preproccessed_data.columns
        with open(os.path.join(tax, "bacteria.txt"), "w") as file:
            for b in bacteria:
                file.write(b + '\n')

        index_to_id_map = {}
        id_to_features_map = {}
        for i, row in enumerate(preproccessed_data.values):
            id_to_features_map[preproccessed_data.index[i]] = row
            index_to_id_map[i] = preproccessed_data.index[i]

        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map
        ids_list = preproccessed_data.index.tolist()
        self._ids_list = ids_list

        """
        id_to_T_map = {}
        id_to_P_map = {}
        id_to_S_map = {}
        stage_column = 'Stage'
        for sample in OtuMf.mapping_file.index:
            info = OtuMf.mapping_file.loc[sample, stage_column]
            t = int(info[info.find('T') + 1])
            p = int(info[info.find('P') + 1])
            s = int(info[info.find('S') + 1:])
            id_to_T_map[sample] = t
            id_to_P_map[sample] = p
            id_to_S_map[sample] = s

        T_list = []
        S_list = []
        for sample in OtuMf.mapping_file.index:
            T_list.append(id_to_T_map[sample])
            S_list.append(id_to_S_map[sample])
        OtuMf.mapping_file['T'] = T_list
        OtuMf.mapping_file['S'] = S_list
        """
        column = 'S'
        id_to_S_map = {}
        for sample in OtuMf.mapping_file.index:
            s = OtuMf.mapping_file.loc[sample, column]
            id_to_S_map[sample] = s


        column = 'T'
        id_to_T_map = {}
        for sample in OtuMf.mapping_file.index:
            t = OtuMf.mapping_file.loc[sample, column]
            id_to_T_map[sample] = t

        column = 'Description'
        id_to_description_map = {}
        for sample in OtuMf.mapping_file.index:
            d = OtuMf.mapping_file.loc[sample, column]
            id_to_description_map[sample] = d


        # create time series for each child
        patient_to_ids_map = {patient: [] for patient in set(id_to_S_map.values())}
        for key, val in id_to_S_map.items():
            patient_to_ids_map[val].append(key)
        for serie in patient_to_ids_map.values():
            serie.sort()

        times = list(set(id_to_T_map.values()))
        times.sort()
        times_to_index = {p: i for i, p in enumerate(times)}
        if not created_data:
            bact_list, X_y_files_path = create_data_for_signal_bacteria_model_learning(OtuMf, tax, bacteria,
                                                                                       'T',
                                                                                       id_to_T_map,
                                                                                       id_to_S_map,
                                                                                       id_to_features_map,
                                                                                       patient_to_ids_map,
                                                                                       times_to_index)
            pickle.dump(bact_list, open(os.path.join(tax, "time_points_list.pkl"), "wb"))
            pickle.dump(X_y_files_path, open(os.path.join(tax, "X_y_files_path.pkl"), "wb"))
        else:
            # time_points_list = pickle.load(open(os.path.join(tax, "time_points_list.pkl"), "rb"))
            X_y_files_path = pickle.load(open(os.path.join(tax, "X_y_files_path.pkl"), "rb"))

        # create all models for each bacterium to predict its change, through the
        # previous general state (all bacteria values)
        preform_learning(tax, list(bacteria), X_y_files_path)


if __name__ == "__main__":
    task = 'task'
    bactria_as_feature_file = 'otu_table_P2_4times.csv'
    samples_data_file = 'Fasting_map2.csv'
    tax = 6

    create_data = True
    if create_data:
        bifidum_dataset = Bifidobacterium_bifidum_DataLoader(title=task,
                                                             bactria_as_feature_file=bactria_as_feature_file,
                                                             samples_data_file=samples_data_file, taxnomy_level=tax,
                                                             created_data=True)

    calc_significant_bacteria = True
    all_times_all_bact_results_path = "all_times_all_bacteria_best_models_results_df.csv"
    # if we want to run on a single model, change to a different csv file with the same structure
    if calc_significant_bacteria:
        folder = os.path.join("tax=" + str(tax), "Significant_bacteria")
        for algo in ["ard regression"]:
            results_df_path = os.path.join("tax=" + str(tax), all_times_all_bact_results_path)
            significant_bacteria = check_if_bacteria_correlation_is_significant(results_df_path, algo)
            get_significant_beta_from_file(results_df_path, algo, significant_bacteria, folder)

