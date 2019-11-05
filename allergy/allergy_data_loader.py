from sys import stdout
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
from anna.microbiome.distance_learning_func import distance_learning
from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
import os
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from infra_functions.general import apply_pca, use_spearmanr, use_pearsonr, roc_auc, convert_pca_back_orig, draw_horizontal_bar_chart  # sigmoid
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
from sklearn.utils import class_weight
n_components = 20


class AllergyDataLoader(Dataset):
    def __init__(self, TITLE, PRINT, REG, WEIGHTS, ANNA_PREPROCESS):
        self._task = TITLE
        self._id_list, self._id_wo_con_list, self._feature_list = self._read_file(TITLE, PRINT, REG ,WEIGHTS, ANNA_PREPROCESS)   # read file to structured data

    @property
    def get_index_to_id_map(self):
        return self._index_to_id_map

    @property
    def get_id_to_features_map(self):
        return self._id_to_features_map

    def __len__(self):
        return len(self._id_list)

    def __getitem__(self, index):
        if self._task == "success task":
            out = self._id_to_features_map[self._id_wo_con_list[index]],\
                   self._id_to_binary_success_tag_map[self._id_wo_con_list[index]]

        elif self._task == "health task":
            out = self._feature_list[index], self._id_to_binary_health_tag_map[self._id_list[index]]

        elif self._task == "prognostic task":
            out = self._id_to_features_map[self._id_wo_con_list[index]], \
                  self._id_to_binary_success_tag_map[self._id_wo_con_list[index]]

        elif self._task == "allergy type task":
            out = self._id_to_features_map[self._id_wo_con_list[index]], \
                  self._id_to_allergy_number_type_tag_map[self._id_wo_con_list[index]]

        return torch.Tensor(out[0]), out[1]

    @property
    def get_ids_list_w_con(self):
        return self._ids_list_w_con

    @property
    def get_ids_list_wo_con(self):
        return self._ids_list_wo_con

    @property
    def get_ids_list_wo_multiple(self):
        return self._ids_list_wo_multiple

    @property
    def get_id_wo_non_and_egg_allergy_type_list(self):
        return self._id_wo_non_and_egg_allergy_type_list

    @property
    def get_stage_0_ids(self):
        return self._stage_0_ids

    @property
    def get_id_to_binary_health_tag_map(self):
        return self._id_to_binary_health_tag_map

    @property
    def get_id_to_success_tag_map(self):
        return self._id_to_success_tag_map

    @property
    def get_id_to_stage_map(self):
        return self._id_to_stage_map

    @property
    def get_id_to_binary_success_tag_map(self):
        return self._id_to_binary_success_tag_map

    @property
    def get_tag_to_allergy_type_map(self):
        return self._tag_to_allergy_type_map

    @property
    def get_allergy_type_to_instances_map(self):
        return self._allergy_type_to_instances_map

    @property
    def get_allergy_type_to_weight_map(self):
        return self._allergy_type_to_weight_map

    @property
    def get_milk_vs_other_allergy_weight_map(self):
        return self._milk_vs_other_allergy_weight_map

    @property
    def get_healthy_vs_allergic_weight_map(self):
        return self._healthy_vs_allergic_weight_map

    @property
    def get_responding_vs_not_weight_map(self):
        return self._responding_vs_not_weight_map

    @property
    def get_prognostic_responding_vs_not_weight_map(self):
        return self._prognostic_responding_vs_not_weight_map

    @property
    def get_id_to_allergy_type_tag_map(self):
        return self._id_to_allergy_type_tag_map

    @property
    def get_id_to_allergy_number_type_tag_map(self):
        return self._id_to_allergy_number_type_tag_map

    @property
    def get_id_to_milk_allergy_tag_map(self):
        return self._id_to_milk_allergy_tag_map

    @property
    def get_preproccessed_data(self):
        return self._preproccessed_data

    @property
    def get_pca_obj(self):
        return self._pca_obj

    @property
    def get_id_to_single_or_multiple_allergy_map(self):
        return self.id_to_single_or_multiple_allergy_map

    def lineplot2y(self, x_data, x_label, y1_data, y1_color, y1_label, y2_data, y2_color, y2_label, title):
        # Each variable will actually have its own plot object but they
        # will be displayed in just one plot
        # Create the first plot object and draw the line
        _, ax1 = plt.subplots()
        ax1.plot(x_data, y1_data, color=y1_color)
        # Label axes
        ax1.set_ylabel(y1_label, color=y1_color)
        ax1.set_xlabel(x_label)
        ax1.set_title(title)

        # Create the second plot object, telling matplotlib that the two
        # objects have the same x-axis
        ax2 = ax1.twinx()
        ax2.plot(x_data, y2_data, color=y2_color)
        ax2.set_ylabel(y2_label, color=y2_color)
        # Show right frame line
        ax2.spines['right'].set_visible(True)
        _.show()
        _.savefig(title)

    def reg(self, features, cols):
        means = [features[cols][col].mean() for col in cols]
        error = [features[cols][col].std(ddof=0) for col in cols]
        plt.imshow(features[cols])
        plt.imshow(features[cols].apply(zscore))
        l = (features[cols] + 0.1).apply(np.log10)
        plt.imshow(l)
        plt.imshow(l.apply(zscore))

        features_no_id = features[cols].apply(zscore)
        plt.imshow(features_no_id)
        normal_means = [features_no_id[col].mean() for col in cols]
        normal_error = [features_no_id[col].std(ddof=0) for col in cols]

        for col in cols:
            print(col + " mean=" + str(features_no_id[col].mean()) + " std=" + str(features_no_id[col].std(ddof=0)))
        print("total z_score mean=" + str(features_no_id.mean()) + " std=" + str(features_no_id.std(ddof=0)))
        # print(features)

        """
        # Call the function to create plot
        lineplot2y(x_data = cols # bacterias
                   , x_label = 'Bacteria'
                   , y1_data = means
                   , y1_color = '#539caf'
                   , y1_label = 'Means'
                   , y2_data = normal_means
                   , y2_color = '#7663b0'
                   , y2_label = 'Normalized means'
                   , title = 'before_and_after_z_score_per_person_bar_plot')
    
        """

    def drow_data(self, preproccessed_data):
        nan_vals = preproccessed_data.isnull().sum().sum()
        plt.imshow(preproccessed_data)
        plt.imshow(preproccessed_data.apply(zscore))
        data_after_log = (preproccessed_data + 0.1).apply(np.log10)
        plt.imshow(data_after_log)
        plt.imshow(data_after_log.apply(zscore))
        data_after_log_zcore = data_after_log.apply(zscore)
        # plt.imsave('preproccessed_data____.png', (data))

        for i, x in enumerate(preproccessed_data.iterrows()):
            plt.hist(np.array(x[1]), normed=True, bins=30)
            plt.title("sample " + str(i) + ": " + x[0])
            plt.xlabel('Bacteria value')
            plt.ylabel('Probability')
            plt.show()
            plt.savefig('preproccessed_data_hist_' + str(i) + '.png')

    def _read_file(self, TITLE, PRINT, REG, WEIGHTS, ANNA_PREPROCESS):
        bactria_as_feature_file = 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'

        features = pd.read_csv(bactria_as_feature_file, header=1)
        cols = list(features.columns)
        # remove non-numeric values
        cols.remove('Feature ID')
        cols.remove('Taxonomy')

        if REG:
            self.reg(features, cols)
        # get single\multiple information
        multiple_samples_info_path = 'mf_merge_ok84_ok93_ok66_69_TreeNuts_controls_271118_040219 post     MG17 07.05.19.csv'
        multiple_samples_info_df = pd.read_csv(multiple_samples_info_path)
        single_or_multiple_list = multiple_samples_info_df['Michael_4_Single_Multiple']
        single_or_multiple_id_list = multiple_samples_info_df['SampleCode']
        single_or_multiple_map = {}
        for id, s_or_m in zip(single_or_multiple_id_list, single_or_multiple_list):
            single_or_multiple_map[id] = s_or_m
        ids_list_wo_multiple = [key for key, val in single_or_multiple_map.items() if val == 'Single']
        ids_of_multiple = [key for key, val in single_or_multiple_map.items() if val == 'Multiple']
        id_to_single_or_multiple_allergy_map = {}
        for id in ids_list_wo_multiple:
            id_to_single_or_multiple_allergy_map[id] = 0
        for id in ids_of_multiple:
            id_to_single_or_multiple_allergy_map[id] = 1
        self.id_to_single_or_multiple_allergy_map = id_to_single_or_multiple_allergy_map


        # mf_merge_ok84_ok93_ok66_69_TreeNuts_controls_271118_040219 post     MG17 07.05.19.xlsx
        samples_data_file = 'mf_merge_ok84_ok93_ok66_69_merged_by_RestoredSampleCode_as_ID_290119.csv'


        OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, bactria_as_feature_file),
                             os.path.join(SCRIPT_DIR, samples_data_file),
                             from_QIIME=True, id_col='Feature ID', taxonomy_col='Taxonomy')

        if ANNA_PREPROCESS:
            preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxonomy_col='Taxonomy', taxnomy_level=6)
            # if we want to remove certain type of data according to the features
            # preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['AllergyType', 'SuccessDescription']], how='inner')
            # preproccessed_data = preproccessed_data.loc[
            #    (preproccessed_data['AllergyType'] == 'Milk') | ((preproccessed_data['AllergyType'] == 'Peanut'))]
            # preproccessed_data = preproccessed_data.drop(['AllergyType', 'SuccessDescription'], axis=1)
            # mapping_file = OtuMf.mapping_file.loc[(OtuMf.mapping_file['AllergyType'] == 'Milk') | (OtuMf.mapping_file['AllergyType'] == 'Peanut')]

            mapping_file = OtuMf.mapping_file['AllergyType']
            mapping_disease = {'Milk': 0,
                              'Tree_nut': 1,  # 'Cashew' + 'Hazelnut' + 'Walnut'
                              'Peanut': 2,
                              'Sesame': 3}
            mapping_file = mapping_file.map(mapping_disease)
            preproccessed_data, mapping_file = distance_learning(perform_distance=True, level=3, preproccessed_data=preproccessed_data, mapping_file=mapping_file)
            self._preproccessed_data = preproccessed_data


        else:
            preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6, taxonomy_col='Taxonomy',
                                                 preform_taxnomy_group=True)

            self._preproccessed_data = preproccessed_data
            # drow_data(preproccessed_data)
            # otu_after_pca_wo_taxonomy, _, _ = apply_pca(data_after_log_zcore, n_components=40, visualize=False)

        otu_after_pca_wo_taxonomy, pca_obj, _ = apply_pca(preproccessed_data, n_components=n_components, visualize=False)
        control = otu_after_pca_wo_taxonomy.index[0:62]  # 'Con'
        self._pca_obj = pca_obj
        # if we want to remove the healthy samples that are used for control
        # otu_after_pca_wo_taxonomy = otu_after_pca_wo_taxonomy.drop(preproccessed_data.index[0:62])

        index_to_id_map = {}
        id_to_features_map = {}
        for i, row in enumerate(otu_after_pca_wo_taxonomy.values):
                id_to_features_map[otu_after_pca_wo_taxonomy.index[i]] = row
                index_to_id_map[i] = otu_after_pca_wo_taxonomy.index[i]

        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map

        success_tag_column = 'SuccessDescription'
        stages_column = 'TreatmentTimePoint'
        allergan_column = 'AllergyType'
        code_column = 'ParticipentCode'
        ids_list_w_con = otu_after_pca_wo_taxonomy.index.tolist()
        ids_list_wo_con = otu_after_pca_wo_taxonomy.index.drop(otu_after_pca_wo_taxonomy.index[0:62])

        self._ids_list_w_con = ids_list_w_con
        self._ids_list_wo_con = ids_list_wo_con
        self._ids_list_wo_multiple = [id for id in ids_list_wo_multiple if id in ids_list_w_con]

        stages = []

        # ##### separate samples by allergic and healthy==>'Con'
        id_to_binary_health_tag_map = {}
        for sample in ids_list_w_con:
            if sample.startswith('Con'):
                id_to_binary_health_tag_map[sample] = 1
            else:
                id_to_binary_health_tag_map[sample] = 0

        self._id_to_binary_health_tag_map = id_to_binary_health_tag_map

        # ##### separate samples by stage, success of treatment and allergen type
        id_to_success_tag_map = {}
        id_to_stage_map = {}
        id_to_binary_success_tag_map = {}
        id_to_allergy_type_tag_map = {}
        id_to_allergy_number_type_tag_map = {}
        id_to_milk_allergy_tag_map = {}
        allergan_types = set()

        tag_to_allergy_type_map = {0: 'Milk',
                                   1: 'Tree_nut',  # 'Cashew' + 'Hazelnut' + 'Walnut'
                                   2: 'Peanut',
                                   3: 'Sesame'}  # removed 'Egg' samples

        allergy_type_to_instances_map = {'Milk': 0,
                                         'Tree_nut': 0,
                                         'Peanut': 0,
                                         'Sesame': 0}  # 'Non': 9 samples, 'Egg': 35 samples
        """
        nuts_samples_list = []
        for sample in ids_list_wo_con:
             a = OtuMf.mapping_file.loc[sample, allergan_column]
             if a == 'Nuts':
                nuts_samples_list.append(sample)
        with open("nuts_samples.txt", "w") as file:
            for l in nuts_samples_list:
                 file.write(l + "\n")
    """
        non_allergy_type_ids = []
        egg_allergy_type_ids = []
        for sample in ids_list_wo_con:
            s = OtuMf.mapping_file.loc[sample, stages_column]
            # stages
            stages.append(s)
            id_to_stage_map[sample] = s
            stage_0_ids = [key for key in id_to_stage_map if id_to_stage_map[key] == '0_before']
            self._stage_0_ids = stage_0_ids

            # success
            t = OtuMf.mapping_file.loc[sample, success_tag_column]
            id_to_success_tag_map[sample] = t
            # save tags from k-classes as success(A1)->1 and failure(the rest)->0
            if t == 'A1':
                id_to_binary_success_tag_map[sample] = 1
            else:
                id_to_binary_success_tag_map[sample] = 0

            # allergy type
            a = OtuMf.mapping_file.loc[sample, allergan_column]
            allergan_types.add(a)
            id_to_allergy_type_tag_map[sample] = a

            if a == 'Milk' or a == 'Milk_suspected' or a == 'milk':
                id_to_allergy_number_type_tag_map[sample] = 0
                id_to_milk_allergy_tag_map[sample] = 1
                allergy_type_to_instances_map['Milk'] = allergy_type_to_instances_map.get('Milk') + 1
            elif a == 'Cashew' or a == 'Cashew ' or a == 'Hazelnut' or a == 'Walnut' or a == 'Nuts':
                id_to_allergy_number_type_tag_map[sample] = 1
                id_to_milk_allergy_tag_map[sample] = 0
                allergy_type_to_instances_map['Tree_nut'] = allergy_type_to_instances_map.get('Tree_nut') + 1
            elif a == 'Peanut':
                id_to_allergy_number_type_tag_map[sample] = 2
                id_to_milk_allergy_tag_map[sample] = 0
                allergy_type_to_instances_map['Peanut'] = allergy_type_to_instances_map.get('Peanut') + 1
            elif a == 'Sesame':
                id_to_allergy_number_type_tag_map[sample] = 3
                id_to_milk_allergy_tag_map[sample] = 0
                allergy_type_to_instances_map['Sesame'] = allergy_type_to_instances_map.get('Sesame') + 1
            elif a == 'Egg':
                egg_allergy_type_ids.append(sample)
                # id_to_allergy_number_type_tag_map[sample] = 1
                # id_to_milk_allergy_tag_map[sample] = 0
                # allergy_type_to_instances_map['Egg'] = allergy_type_to_instances_map.get('Egg') + 1
            elif a == 'Non':
                non_allergy_type_ids.append(sample)
                # id_to_allergy_number_type_tag_map[sample] = None
                # id_to_milk_allergy_tag_map[sample] = None
                # allergy_type_to_instances_map['Non'] = allergy_type_to_instances_map.get('Non') + 1
            else:
                print("error in allergy type " + str(sample))

        self._id_wo_non_and_egg_allergy_type_list = [x for x in self._ids_list_wo_con if x not in non_allergy_type_ids + egg_allergy_type_ids]
        self._tag_to_allergy_type_map = tag_to_allergy_type_map
        self._allergy_type_to_instances_map = allergy_type_to_instances_map
        self._id_to_success_tag_map = id_to_success_tag_map
        self._id_to_stage_map = id_to_stage_map
        self._id_to_binary_success_tag_map = id_to_binary_success_tag_map
        self._id_to_allergy_type_tag_map = id_to_allergy_type_tag_map
        self._id_to_allergy_number_type_tag_map = id_to_allergy_number_type_tag_map
        self._id_to_milk_allergy_tag_map = id_to_milk_allergy_tag_map

        self._ids_list_wo_multiple = [id for id in ids_list_wo_multiple if id in id_to_allergy_number_type_tag_map.keys()]

        # -------------------------------------------- weights !--------------------------------------------
        # calculate weights for types of allergy
        if WEIGHTS:
            total_sum = sum(list(allergy_type_to_instances_map.values()))
            types = list(allergy_type_to_instances_map.keys())
            allergy_type_to_weight_map = {}
            for t in types:
                allergy_type_to_weight_map[t] = total_sum / allergy_type_to_instances_map[t]

            # normalize
            max_weight = max(list(allergy_type_to_weight_map.values()))
            for t in types:
                allergy_type_to_weight_map[t] = allergy_type_to_weight_map.get(t) / max_weight

            # calculate weights for milk vs. other types of allergy
            milk_vs_other_allergy_weight_map = {'Other': total_sum / (total_sum - allergy_type_to_instances_map.get("Milk")),
                                               'Milk': total_sum / allergy_type_to_instances_map.get("Milk")}
            # normalize
            max_weight = max(list(milk_vs_other_allergy_weight_map.values()))
            for t in ['Other', 'Milk']:
                milk_vs_other_allergy_weight_map[t] = milk_vs_other_allergy_weight_map.get(t) / max_weight

            # calculate weights for healthy and allergic
            healthy_vs_allergic_weight_map = {
                'Allergic': (len(ids_list_w_con)) / len(ids_list_wo_con),
                'Healthy': (len(ids_list_w_con)) / (len(ids_list_w_con) - len(ids_list_wo_con))}

            # normalize
            max_weight = max(list(healthy_vs_allergic_weight_map.values()))
            for t in ['Allergic', 'Healthy']:
                healthy_vs_allergic_weight_map[t] = healthy_vs_allergic_weight_map.get(t) / max_weight

            # calculate weights for responding and not (success)
            no_response = list(id_to_binary_success_tag_map.values()).count(0)
            yes_response = list(id_to_binary_success_tag_map.values()).count(1)

            responding_vs_not_weight_map = {
                'No': (len(ids_list_wo_con)) / no_response,
                'Yes': (len(ids_list_wo_con) / yes_response)}

            # normalize
            max_weight = max(list(responding_vs_not_weight_map.values()))
            for t in ['No', 'Yes']:
                responding_vs_not_weight_map[t] = responding_vs_not_weight_map.get(t) / max_weight

            # calculate weights for responding and not (prognostic)
            tags = []
            for i in stage_0_ids:
                tags.append(id_to_binary_success_tag_map.get(i))

            no_response = tags.count(0)
            yes_response = tags.count(1)

            prognostic_responding_vs_not_weight_map = {
                'No': (len(stage_0_ids)) / no_response,
                'Yes': (len(stage_0_ids) / yes_response)}

            # normalize
            max_weight = max(list(prognostic_responding_vs_not_weight_map.values()))
            for t in ['No', 'Yes']:
                prognostic_responding_vs_not_weight_map[t] = prognostic_responding_vs_not_weight_map.get(t) / max_weight

            self._allergy_type_to_weight_map = allergy_type_to_weight_map
            self._milk_vs_other_allergy_weight_map = milk_vs_other_allergy_weight_map
            self._healthy_vs_allergic_weight_map = healthy_vs_allergic_weight_map
            self._responding_vs_not_weight_map = responding_vs_not_weight_map
            self._prognostic_responding_vs_not_weight_map = prognostic_responding_vs_not_weight_map



        """    # count tags in all vs. stage_0
        all_tags = list(id_to_binary_success_tag_map.values())
        print("tags total len: " + str(len(all_tags)) + " pos tags: " + str(all_tags.count(1))
              + " percent: " + str(all_tags.count(1)/len(all_tags)))
        stage_0_tags = [id_to_binary_success_tag_map[id] for id in stage_0_ids if id in id_to_binary_success_tag_map.keys()]
        print("stage 0 tags total len: " + str(len(stage_0_tags)) + " pos tags: " + str(stage_0_tags.count(1))
              + " percent: " + str(stage_0_tags.count(1)/len(stage_0_tags)))
        """

        # return the list of features and the list of ids in the same order
        feature_list = [id_to_features_map[id] for id in ids_list_w_con]
        return ids_list_w_con, ids_list_wo_con, feature_list


if __name__ == "__main__":
    # dl_dev = TrainDataLoader(os.path.join("..", "data", "pos", "dev"), vocab=dl_train.vocabulary)
    # d = [dl_train.__getitem__(i) for i in range(len(dl_train))]
    # d = [dl_dev.__getitem__(i) for i in range(len(dl_dev))]
    task = 'success task'
    allergy_dataset = AllergyDataLoader(TITLE=task, PRINT=False, REG=False, RHOS=True)

    if task == 'success task':
        target = list(allergy_dataset.get_id_to_binary_success_tag_map.values())

    print('target train 0/1: {}/{}'.format(
        len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))

    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # target = torch.from_numpy(target).long()
    #train_dataset = torch.utils.data.TensorDataset(data, target)

    data_loader = DataLoader(
        allergy_dataset,
        batch_size=64, sampler=sampler  # shuffle=True
    )

    for i, (data, target) in enumerate(data_loader):
        print
        "batch index {}, 0/1: {}/{}".format(
            i,
            len(np.where(np.array(target) == 0)[0]),
            len(np.where(np.array(target) == 1)[0]))

    for batch_index, (data, label) in enumerate(data_loader):
        stdout.write("\r\r\r%d" % int(100 * ((batch_index + 1) / len(data_loader))) + "%")
        stdout.flush()

        #print(data)
        # print(label)

    # dl_test = TestDataLoader(os.path.join("..", "data", "pos", "test"), dl_train.vocabulary, labeled=False)
    # dl_test.vocabulary.learn_distribution(os.path.join("..", "data", "pos", "test"), labeled=False)
    # dl_test.load_pos_map(dl_train.pos_map)
    #
    # data_loader = DataLoader(
    #     dl_test,
    #     batch_size=64, shuffle=True
    # )
    # for i, (word, vec, (is_start, is_end)) in enumerate(data_loader.dataset):
    #     print(i, word, vec, is_start, is_end)

    # d = [dl_test.__getitem__(i) for i in range(len(dl_test))]
    # e = 1
