#from anna.microbiome.distance_learning_func import distance_learning
from LearningMethods.abstract_data_set import AbstractDataLoader
from anna.microbiome.distance_learning_func import distance_learning
from Projects.GVHD_BAR.load_merge_otu_mf import OtuMfHandler
from Preprocess.preprocess import preprocess_data
import os
import pandas as pd
import numpy as np
from Preprocess.general import apply_pca

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

n_components = 20


class AlgBeforeTreatmentDataLoader(AbstractDataLoader):
    """
    Fill _read_file function according to specific data set in order to achieve mutual behavior
    create all of the following members:
        self._preproccessed_data = preproccessed_data
        self._pca_obj = pca_obj
        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map
        self._ids_list = ids_list
        make your own -> self._id_to_tag_map = _id_to_tag_map
        self._weight_map = classes_ratio
        self._feature_list = feature_list
    """
    def __init__(self, title, bactria_as_feature_file, samples_data_file, taxnomy_level, allow_printing, perform_anna_preprocess):
        super().__init__(title, bactria_as_feature_file, samples_data_file, taxnomy_level, allow_printing, perform_anna_preprocess)

    def _read_file(self, title, bactria_as_feature_file, samples_data_file, allow_printing, perform_anna_preprocess):
        features = pd.read_csv(bactria_as_feature_file, header=1)
        cols = list(features.columns)
        # remove non-numeric values
        cols.remove('Feature ID')
        cols.remove('Taxonomy')

        OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, bactria_as_feature_file),
                             os.path.join(SCRIPT_DIR, samples_data_file),
                             from_QIIME=True, id_col='Feature ID', taxonomy_col='Taxonomy')

        if perform_anna_preprocess:
            preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxonomy_col='Taxonomy',
                                                 taxnomy_level=6)
            mapping_file = OtuMf.mapping_file['XXXXX']
            mapping_disease = {'a': 0,
                               'b': 1,  # 'Cashew' + 'Hazelnut' + 'Walnut'
                               'c': 2,
                               'd': 3}
            mapping_file = mapping_file.map(mapping_disease)
            preproccessed_data, mapping_file = distance_learning(perform_distance=True, level=self._taxnomy_level,
                                                                 preproccessed_data=preproccessed_data,
                                                                 mapping_file=mapping_file)
            self._preproccessed_data = preproccessed_data
        else:
            preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=self._taxnomy_level,
                                                 taxonomy_col='Taxonomy',
                                                 preform_taxnomy_group=True)

            self._preproccessed_data = preproccessed_data
            # drow_data(preproccessed_data)
            # otu_after_pca_wo_taxonomy, _, _ = apply_pca(data_after_log_zcore, n_components=40, visualize=False)

        otu_after_pca_wo_taxonomy, pca_obj, _ = apply_pca(preproccessed_data, n_components=n_components,
                                                          visualize=False)
        self._pca_obj = pca_obj

        index_to_id_map = {}
        id_to_features_map = {}
        for i, row in enumerate(otu_after_pca_wo_taxonomy.values):
            id_to_features_map[otu_after_pca_wo_taxonomy.index[i]] = row
            index_to_id_map[i] = otu_after_pca_wo_taxonomy.index[i]
        ids_who_has_features = list(id_to_features_map.keys())


        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map
        ids_list = otu_after_pca_wo_taxonomy.index.tolist()

        sample_id_to_sample_code_map = {}
        sample_ids = [s for s in OtuMf.mapping_file.index] #  if not s.startswith("Con")]
        sample_code = [s for s in OtuMf.mapping_file["SampleCode"]] #  if s != "Control"]
        for id, code in zip(sample_ids, sample_code):
            if not id.startswith("Con"):
                sample_id_to_sample_code_map[id] = code

        # ------------------------------------ each TASK creates different tag map --------------------------------
        before_ids = []
        id_to_tag_map = {}

        for id in OtuMf.mapping_file.index:
            before = OtuMf.mapping_file.loc[id, "TreatmentPoint"]
            if before == "before":
                if sample_id_to_sample_code_map[id] in id_to_features_map.keys():
                    before_ids.append(id)
                else:
                    print(code + " not in id_to_features_map")
            elif before == "Control":
                before_ids.append(id)

        code_list = []
        for id in before_ids:
            if id in sample_id_to_sample_code_map.keys():
                code_list.append(sample_id_to_sample_code_map[id])
            else:
                code_list.append(id)

        # HEALTH_BEFORE_TREATMENT_TASK
        if self._task == "health_before_treatment_task":
            for id, code in zip(before_ids, code_list):
                before = OtuMf.mapping_file.loc[id, "TreatmentPoint"]
                if before == "before":
                    if code in id_to_features_map.keys():
                        id_to_tag_map[code] = 1
                elif before == "Control":
                    id_to_tag_map[id] = 0

            self._ids_list = list(id_to_tag_map.keys())
            """
            # before_ids.remove("382954")
            # before_ids.remove("386137")
            # before_ids.remove("386100")
                    if self._task == "health_before_treatment_task":
            for id in OtuMf.mapping_file.index:
                before = OtuMf.mapping_file.loc[id, "TreatmentPoint"]
                if before == "before":
                    code = sample_id_to_sample_code_map[id]
                    if code in id_to_features_map.keys():
                        before_ids.append(code)
                        id_to_tag_map[code] = 1
                    else:
                        print(code + " not in id_to_features_map")

                elif before == "Control":
                        before_ids.append(id)
                        id_to_tag_map[id] = 0
                else:
                    print(before + " error")
            """

        # ALLERGY_TYPE_BEFORE_TREATMENT_TASK
        elif self._task == "allergy_type_before_treatment_task":
            tag_to_allergy_type_map = {0: 'Milk',
                                       1: 'Tree_nut',  # 'Cashew' + 'Hazelnut' + 'Walnut'
                                       2: 'Peanut',
                                       3: 'Sesame'}  # removed 'Egg' samples
            for sample, code in zip(before_ids, code_list):
                a = OtuMf.mapping_file.loc[sample, 'AllergyType']
                if a == 'Milk' or a == 'Milk_suspected' or a == 'milk':
                    id_to_tag_map[code] = 0
                elif a == 'Cashew' or a == 'Cashew ' or a == 'Hazelnut' or a == 'Walnut' or a == 'Nuts':
                    id_to_tag_map[code] = 1
                elif a == 'Peanut':
                    id_to_tag_map[code] = 2
                elif a == 'Sesame':
                    id_to_tag_map[code] = 3
            self._ids_list = [id for id in code_list if not id.startswith("Con")]

        self._id_to_tag_map = id_to_tag_map

        # -------------------------------------------- weights !--------------------------------------------
        # calculate weights
        y = list(id_to_tag_map.values())
        classes_sum = [np.sum(np.array(y) == unique_class) for unique_class in
                       np.unique(np.array(y))]
        classes_ratio = [1 - (a / sum(classes_sum)) for a in classes_sum]
        weights = [classes_ratio[a] for a in np.array(y)]
        self._weight_map = {i: classes_ratio[i] for i in range(len(classes_ratio))}

        # return the list of features and the list of ids in the same order
        feature_list = []
        for id in self._ids_list:
            if id in sample_id_to_sample_code_map.keys():
                feature_list.append(id_to_features_map[sample_id_to_sample_code_map[id]])
            else:
                id_to_features_map[id]
        self._feature_list = feature_list


    def get_confusin_matrix_names(self):
        if self._task == "health_before_treatment_task":
            names = ['Healthy', 'Allergic']
        elif self._task == "allergy_type_before_treatment_task":
           names = ['Milk', 'Tree_nut', 'Peanut', 'Sesame']
        return names


if __name__ == "__main__":
    task = 'success task'
    bactria_as_feature_file = 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'
    samples_data_file = 'mf_MG17_070519_No_Eggs_150919_for_dafna.csv'
    tax = 6
    """
    allergy_dataset = AlgBeforeTreatmentDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file,  taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)
    

    RunAlgBeforeTreatmentDataLoader(learning_tasks=["health_before_treatment_task"],
                                    bactria_as_feature_file=bactria_as_feature_file,
                                    samples_data_file=samples_data_file, tax=6)
    """
