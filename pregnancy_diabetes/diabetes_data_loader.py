import csv
import math
import pickle
from copy import deepcopy
from enum import Enum
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore, spearmanr
# from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm, metrics
from collections import Counter
from allergy.dafna import calc_auc_on_joined_results, edit_confusion_matrix, print_confusion_matrix
from anna.microbiome.distance_learning_func import distance_learning
from infra_functions.general import apply_pca, draw_rhos_calculation_figure, roc_auc, convert_pca_back_orig, \
    draw_dynamics_rhos_calculation_figure
from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, MeanShift, KMeans, AffinityPropagation

n_components = 20


class DiabetesDataLoader:
    def __init__(self, microbial_file_path, mapping_file_path, to_csv, merge_files, tax_level=5, add_tax_row=False,
                 draw_self_corr_mat=False, from_pkl=False, load_heavy_otu_and_preprocces_from_pickle=True,
                 ANNA_PREPROCESS=False):
        self.microbial_file_path = microbial_file_path
        self.mapping_file_path = mapping_file_path
        self.tax_level = tax_level

        if to_csv:
            self._convert_files_to_csv()
        if merge_files:
            self._merge_files()

        self._read_file(tax_level, add_tax_row, draw_self_corr_mat, from_pkl, load_heavy_otu_and_preprocces_from_pickle,
                        ANNA_PREPROCESS)

    def _convert_files_to_csv(self):
        directory = 'GDM_tables'
        for filename in os.listdir(directory):
            with open(directory + '/' + filename) as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                data_dict = {}
                for i, row in enumerate(reader):
                    # print(row)
                    data_dict[i] = row
            data_dict.pop(0)
            df = pd.DataFrame(data_dict)
            # df.drop(columns=['B', 'C'])
            df.to_csv('GDM_tables_as_csv/' + filename.split(".")[0] + '.csv')

        directory = 'mapping_files'
        for filename in os.listdir(directory):
            path = directory + '/' + filename
            df = pd.read_excel(path)
            df.to_csv(path.split('.')[0] + '.csv')

    def _merge_files(self):
        df_list = []
        map_df_list = []
        directory = 'GDM_tables_as_csv/'
        for i, filename in enumerate(os.listdir(directory)):
            with open(directory + filename) as file:
                df = pd.read_csv(file)
                df.columns = df.loc[0]
                df = df.drop([0])
                if i != 0:
                    df = df.drop(df.shape[0])
                df_list.append(df)

            samples_data_file = 'mapping_files/' + filename.split("_")[0] + '.csv'
            with open(samples_data_file) as file:
                map_df = pd.read_csv(file)
                # map_df.columns = map_df.loc[0]
                map_df = map_df.drop([0])
                map_df_list.append(map_df)

        final_df = pd.concat(reversed(df_list), join='inner')
        print("NA in final df => " + str(final_df.isnull().values.any()))
        # final_df = reduce(lambda x, y: pd.merge(x, y), df_list)
        final_df.index = range(final_df.index.shape[0])
        final_df.to_csv('merged_ok_table_w_taxonomy_files.csv')

        final_map_df = pd.concat(reversed(map_df_list), join='inner')
        final_map_df.index = range(final_map_df.index.shape[0])

        final_map_df.to_csv('mapping_file.csv')

    def _read_files_separately(self):
        all_bacterias = []
        ids = []
        bact_id_to_name_dict = {}
        with open("merged_df_columns.txt", "r") as f:
            columns = f.readlines()
            columns = [c.strip() for c in columns]
        columns_no_otu_id = columns[1:]

        # fill df

        directory = 'GDM_tables_as_csv/'
        for filename in os.listdir(directory):
            if filename == "ignore":
                continue
            path = directory + filename
            samples_data_file = 'mapping_files/' + filename.split("_")[0] + '.csv'
            merged_df = pd.DataFrame(columns=columns)
            print(filename.split("_")[0])

            with open(path) as csvfile:
                with open(samples_data_file) as mapfile:
                    features = pd.read_csv(csvfile, header=1)
                    mapping = pd.read_csv(samples_data_file, header=1)

                    bacterias = list(features.columns)
                    bacterias.pop(0)
                    names = list(features.iloc[len(features)-1])
                    names.pop(0)

                    for i, bact in enumerate(bacterias):
                        bact_id_to_name_dict[bact] = names[i]

                    all_bacterias = all_bacterias + bacterias
                    file_ids = list(features["#OTU ID"])
                    file_ids.pop(-1)
                    ids = ids + file_ids

                    for j, id in enumerate(file_ids):
                        next_row = [0 for j in range(len(columns_no_otu_id))]
                        for b, bact in enumerate(columns_no_otu_id):
                            if bact in bacterias:
                                next_row[b] = features[bact][j]
                        merged_df.loc[len(merged_df)] = [id] + next_row

                    merged_df.to_csv('merged_df_' + filename.split("_")[0] + '.csv')

        print("number of bacterias before removing doubles " + str(len(all_bacterias)))
        all_bacterias = set(all_bacterias)
        print("number of bacterias after removing doubles " + str(len(all_bacterias)))

        print("number of ids before removing doubles " + str(len(ids)))
        all_bacterias = set(all_bacterias)
        print("number of ids after removing doubles " + str(len(ids)))

        columns = ["#OTU ID"] + list(all_bacterias)
        with open("bact_id_to_name_dict.txt", "w") as f:
            for key in list(bact_id_to_name_dict.keys()):
                f.write(key + "," + bact_id_to_name_dict[key] + "\n")

        # with open("merged_df_columns.txt" "w") as f:
        #     for c in columns:
        #         f.write(c + "\n")
        number_of_samples = len(ids)



        """
        OtuMf = OtuMfHandler(path, samples_data_file, from_QIIME=True, id_col='#OTU ID',
                             taxonomy_col='taxonomy', tax_in_col=True)

        preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=True, taxnomy_level=6,
                                             taxonomy_col='taxonomy', preform_taxnomy_group=True)

        self._preproccessed_data = preproccessed_data
        # drow_data(preproccessed_data)
        # otu_after_pca_wo_taxonomy, _, _ = apply_pca(data_after_log_zcore, n_components=40, visualize=False)

        otu_after_pca_wo_taxonomy, pca_obj, _ = apply_pca(preproccessed_data, n_components=n_components,
                                                          visualize=True)
        self._pca_obj = pca_obj

        index_to_id_map = {}
        id_to_features_map = {}
        for i, row in enumerate(otu_after_pca_wo_taxonomy.values):
            id_to_features_map[otu_after_pca_wo_taxonomy.index[i]] = row
            index_to_id_map[i] = otu_after_pca_wo_taxonomy.index[i]
        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map

        ids_list = otu_after_pca_wo_taxonomy.index.tolist()
        self._ids_list = ids_list
        success_tag_column = 'SuccessDescription'
        """

    def find_rare_bacteria(self, OtuMf):
        bact_to_num_of_non_zeros_values_map = {}
        otu = OtuMf.otu_file
        bacteria = otu.columns[1:]
        num_of_samples = len(otu.index) - 1
        for bact in bacteria:
            values = otu[bact]
            count_map = Counter(values)
            zeros = 0
            if 0 in count_map.keys():
                zeros += count_map[0]
            if '0' in count_map.keys():
                zeros += count_map['0']

            bact_to_num_of_non_zeros_values_map[bact] = num_of_samples - zeros

        rare_bacteria = []
        for key, val in bact_to_num_of_non_zeros_values_map.items():
            if val < 5:
                rare_bacteria.append(key)
        return rare_bacteria

    def _read_file(self, tax_level, add_tax_row, draw_self_corr_mat, from_pkl, load_heavy_otu_and_preprocces_from_pickle, ANNA_PREPROCESS):
        SampleID_to_womanno_map = {}
        SampleID_to_trimester_map = {}
        SampleID_to_body_site_map = {}
        SampleID_to_Control_or_GDM_map = {}
        SampleID_to_Run_map = {}
        SampleID_to_Plate_map = {}
        SampleID_to_Well_map = {}
        SampleID_to_Type_map = {}
        SampleID_to_Sample_map = {}
        Woman_to_all_SampleID_map = {}


        """
        # create Sample_id_to_file_map
        Sample_id_to_file_map = {}
        directory = 'GDM_tables_as_csv/'
        for filename in os.listdir(directory):
            print(filename)
            if filename.startswith("merged"):
                df = pd.read_csv(os.path.join(directory, filename))
                file_ids = list(df["#OTU ID"])
                for id in file_ids:
                    Sample_id_to_file_map[id] = filename.split("ok")[1].split(".")[0]


        with open("Sample_id_to_file_map.txt",  "w") as file:
            for key, val in Sample_id_to_file_map.items():
                file.write(key + "," + val + "\n")
        """
        # load Sample_id_to_file_map
        Sample_id_to_file_map = {}
        with open("Sample_id_to_file_map.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                id_file = line.strip().split(",")
                Sample_id_to_file_map[id_file[0]] = id_file[1]
        self.SampleID_to_file_map = Sample_id_to_file_map

        if from_pkl:
            # pickle.dump(features, open("features.pkl", "wb"))
            # pickle.dump(mapping, open("mapping.pkl", "wb"))
            features = pickle.load(open("features.pkl", "rb"))
            mapping = pickle.load(open("mapping.pkl", "rb"))
        else:
            features = pd.read_csv(self.microbial_file_path, header=0)
            mapping = pd.read_csv(self.mapping_file_path, header=0)

        bact_id_to_name_dict = {}
        with open("bact_id_to_name_dict.txt", "r") as f:
            lines = f.readlines()
            lines = [l.strip().split(",") for l in lines]
            for bact in lines:
                bact_id_to_name_dict[bact[0]] = bact[1]

        if add_tax_row:
            cols = features.columns[1:]
            tax_row = [bact_id_to_name_dict[bact] for bact in cols]
            features.loc[len(features)] = ["taxonomy"] + tax_row
            features.to_csv("merged_GDM_tables_w_tax.csv")

        self.bact_id_to_name_dict = bact_id_to_name_dict
        mapping = mapping[1:]
        mapping.index = mapping["#SampleID"]
        ids = list(features["#OTU ID"])
        ids.remove('taxonomy')
        mapping = mapping.drop(columns=['Unnamed: 0', "#SampleID"])
        mapping_cols = list(mapping.columns)
        nan_ids = ['HHSEH-STOOL-T0-rep-1', 'HHSHK-STOOL-T0-rep-1', 'HHSTB-STOOL-T0-rep-1', 'HHSYG-STOOL-T0-rep-1']
        for sample in nan_ids:
            ids.remove(sample)
            mapping = mapping.drop(index=sample)

        self.ids_list = ids
        """
        for i, sample in enumerate(ids):
            if mapping.iloc[i]['Control/GDM'] not in ['Control', 'GDM']:
                nan_ids.append(sample)
                ids.remove(sample)
                mapping = mapping.drop(index=sample)

        
        woman = set(mapping['womanno.'])
        for w in woman:
            Woman_to_all_SampleID_map[w] = [mapping.index[m] for m in range(len(ids)) if mapping.iloc[m]['womanno.'] == w]
                                            
        pickle.dump(Woman_to_all_SampleID_map, open("Woman_to_all_SampleID_map.pkl", "wb"))
        """

        Woman_to_all_SampleID_map = pickle.load(open("Woman_to_all_SampleID_map.pkl", "rb"))

        body_site = Enum('body_site', 'SALIVA STOOL Meconium')
        Control_or_GDM = Enum('Control_or_GDM', 'Control GDM')  # nan !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        type = Enum('type', 'Baby Mother Nurse')


        #for i, sample in zip(mapping.index, ids):
        for row in mapping.iterrows():
            if row[0] in ids:
                SampleID_to_womanno_map[row[0]] = row[1]['womanno.']
                SampleID_to_trimester_map[row[0]] = int(row[1]['trimester'])
                SampleID_to_body_site_map[row[0]] = body_site[row[1]['body_site']].value
                SampleID_to_Control_or_GDM_map[row[0]] = Control_or_GDM[row[1]['Control/GDM']].value
                SampleID_to_Run_map[row[0]] = row[1]['Run']
                SampleID_to_Plate_map[row[0]] = row[1]['Plate']
                SampleID_to_Well_map[row[0]] = row[1]['Well']
                SampleID_to_Type_map[row[0]] = type[row[1]['Type']].value
                SampleID_to_Sample_map[row[0]] = row[1]['Sample']

        self.SampleID_to_womanno_map = SampleID_to_womanno_map
        self.SampleID_to_trimester_map = SampleID_to_trimester_map
        self.body_site_enum = body_site
        self.SampleID_to_body_site_map = SampleID_to_body_site_map
        self.Control_or_GDM_enum = Control_or_GDM
        self.SampleID_to_Control_or_GDM_map = SampleID_to_Control_or_GDM_map
        self.SampleID_to_Run_map = SampleID_to_Run_map
        self.SampleID_to_Plate_map = SampleID_to_Plate_map
        self.SampleID_to_Well_map = SampleID_to_Well_map
        self.type_enum = type
        self.SampleID_to_Type_map = SampleID_to_Type_map
        self.SampleID_to_Sample_map = SampleID_to_Sample_map
        self.Woman_to_all_SampleID_map = Woman_to_all_SampleID_map

        OtuMf = OtuMfHandler(self.microbial_file_path, self.mapping_file_path, from_QIIME=False, id_col='#OTU ID',
                             taxonomy_col='taxonomy', tax_in_col=True, make_merge=False)

        rare_bacteria = self.find_rare_bacteria(OtuMf)

        OtuMf = OtuMfHandler(self.microbial_file_path, self.mapping_file_path, from_QIIME=False, id_col='#OTU ID',
                                 taxonomy_col='taxonomy', tax_in_col=True, make_merge=False, drop_rare_bacteria=rare_bacteria)


        if ANNA_PREPROCESS:
            preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxonomy_col='taxonomy', taxnomy_level=5)

            # if we want to remove certain type of data according to the features
            # preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['AllergyType', 'SuccessDescription']], how='inner')
            # preproccessed_data = preproccessed_data.loc[
            #    (preproccessed_data['AllergyType'] == 'Milk') | ((preproccessed_data['AllergyType'] == 'Peanut'))]
            # preproccessed_data = preproccessed_data.drop(['AllergyType', 'SuccessDescription'], axis=1)
            # mapping_file = OtuMf.mapping_file.loc[(OtuMf.mapping_file['AllergyType'] == 'Milk') | (OtuMf.mapping_file['AllergyType'] == 'Peanut')]

            mapping_file = OtuMf.mapping_file['Control/GDM']
            mapping_disease = {'Control': 0,
                              'GDM': 1}
            mapping_file = mapping_file.map(mapping_disease)
            preproccessed_data, mapping_file = distance_learning(perform_distance=True, level=3, preproccessed_data=preproccessed_data, mapping_file=mapping_file)


        else:
            preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=tax_level,
                                                 taxonomy_col='taxonomy', preform_taxnomy_group=True)
        self.OtuMf = OtuMf
        self.preproccessed_data = preproccessed_data
        if draw_self_corr_mat:
            correlation_matrix(preproccessed_data, 12)

        otu_after_pca_wo_taxonomy, pca_obj, _ = apply_pca(preproccessed_data, n_components=n_components,
                                                          visualize=False)
        self.pca_obj = pca_obj

        index_to_id_map = {}
        id_to_features_map = {}
        for i, row in enumerate(otu_after_pca_wo_taxonomy.values):
            id_to_features_map[otu_after_pca_wo_taxonomy.index[i]] = row
            index_to_id_map[i] = otu_after_pca_wo_taxonomy.index[i]
        self.index_to_id_map = index_to_id_map
        self.id_to_features_map = id_to_features_map

        # predict GDM using first trimester bacteria
        tri_1_ids = [i for i in ids if 'T1' in i]
        self.tri_1_ids = tri_1_ids

    @ property
    def get_diabetic_healthy_task_data(self):
        ids = self.ids_list
        tag_map = ddl.SampleID_to_Control_or_GDM_map
        id_to_features_map = self.id_to_features_map
        task_name = "diabetic_healthy"
        return ids, tag_map, id_to_features_map, task_name

    @property
    def get_ids_sub_sets_and_states(self):
        all_ids = self.ids_list
        body_site_map = ddl.SampleID_to_body_site_map
        trimester_map = ddl.SampleID_to_trimester_map
        # SALIVA = 0 STOOL = 1
        saliva_ids = [id for id in all_ids if body_site_map[id] == 1]
        stool_ids = [id for id in all_ids if body_site_map[id] == 2]
        trimester_1_ids = [id for id in all_ids if trimester_map[id] == 1]
        trimester_2_ids = [id for id in all_ids if trimester_map[id] == 2]
        trimester_3_ids = [id for id in all_ids if trimester_map[id] == 3]
        trimester_4_ids = [id for id in all_ids if trimester_map[id] == 4]


        saliva_1_2_ids = [id for id in saliva_ids if trimester_map[id] == 1]
        saliva_2_3_ids = [id for id in saliva_ids if trimester_map[id] == 2]
        saliva_partum_ids = [id for id in saliva_ids if trimester_map[id] == 3]

        stool_1_2_ids = [id for id in stool_ids if trimester_map[id] == 1]
        stool_2_3_ids = [id for id in stool_ids if trimester_map[id] == 2]
        stool_partum_ids = [id for id in stool_ids if trimester_map[id] == 3]

        meconium_petus_ids = [id for id in all_ids if trimester_map[id] == 4]



        return [saliva_ids, stool_ids, trimester_1_ids, trimester_2_ids, trimester_3_ids, trimester_4_ids,
                saliva_1_2_ids, saliva_2_3_ids, saliva_partum_ids,
                stool_1_2_ids, stool_2_3_ids, stool_partum_ids, meconium_petus_ids],\
               ["Saliva", "Stool", "Trimester_1", "Trimester_2", "Postpartum", "Meconium",
                "Saliva_1", "Saliva_2", "Saliva_Postpartum",
                "Stool_1", "Stool_2", "Stool_Postpartum", "Meconium_Petus"]




def correlation_matrix(df, number):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    #ax1.grid(True)
    plt.title('Samples Feature Correlation')
    features = list(df.index)
    ax1.set_xticklabels(features, fontsize=6)
    ax1.set_yticklabels(features, fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax) # , ticks=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    plt.show()
    plt.savefig("Samples_features_correlation_" + str(number) + ".png")


def calc_saliva_stool_pairs(samples, id_to_Type_map, id_to_trimester_map, id_to_body_site_map, body_site_enum):
    pairs = []
    df = pd.DataFrame(columns=['id', 'type', 'trimester', 'body_site'])
    for i, sample in enumerate(samples):  # check for the same type and trimester
        try:
            type = id_to_Type_map[sample]
            trimester = id_to_trimester_map[sample]
            body_site = id_to_body_site_map[sample]
            df.loc[i] = [sample] + [type, trimester, body_site]
        except KeyError:
            pass

    sal = body_site_enum['SALIVA'].value
    sto = body_site_enum['STOOL'].value
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if df.iloc[i]['type'] == df.iloc[j]['type'] and df.iloc[i]['trimester'] == df.iloc[j]['trimester'] \
                    and df.iloc[i]['body_site'] != df.iloc[j]['body_site']:
                if sal in [df.iloc[i]['body_site'], df.iloc[j]['body_site']]  \
                        and sto in [df.iloc[i]['body_site'], df.iloc[j]['body_site']]:
                    if df.iloc[i]['body_site'] == sal:
                        pairs.append([df.iloc[i]['id'], df.iloc[j]['id']])
                    else:
                        pairs.append([df.iloc[j]['id'], df.iloc[i]['id']])

    return pairs


def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df


def calc_saliva_stool_correlation(ddl):
    dump_rho = False
    calc_adge_rows = False
    draw_corr = False
    draw_vals = False


    all_woman_pairs = []
    print(ddl.mapping_file_path)
    ids = deepcopy(ddl.ids_list)
    id_to_body_site_map = deepcopy(ddl.SampleID_to_body_site_map)
    index_to_id_map = deepcopy(ddl.index_to_id_map)
    woman_to_all_ids_map = deepcopy(ddl.Woman_to_all_SampleID_map)
    id_to_Type_map = deepcopy(ddl.SampleID_to_Type_map)
    id_to_trimester_map = deepcopy(ddl.SampleID_to_trimester_map)
    body_site_enum = deepcopy(ddl.body_site_enum)
    for woman, samples in woman_to_all_ids_map.items():
        pairs = calc_saliva_stool_pairs(samples, id_to_Type_map, id_to_trimester_map, id_to_body_site_map, body_site_enum)
        all_woman_pairs = all_woman_pairs + pairs

    # iterate all bacteria and create ds of values of SALIVA vs. STOOL
    bacteria = ddl.bact_id_to_name_dict
    data = deepcopy(ddl.preproccessed_data)
    tax_level = ddl.tax_level
    # create df of features - saliva vs. stool

    # bact_df = pd.DataFrame(columns=['id', 'saliva', 'stool'])
    saliva_ids = [id[0] for id in all_woman_pairs]
    stool_ids = [id[1] for id in all_woman_pairs]

    saliva_df = data.loc[saliva_ids, :]
    saliva_df = saliva_df.drop(columns=['Unassigned'])
    # saliva_df.to_csv('saliva_df.csv')
    stool_df = data.loc[stool_ids, :]
    stool_df = stool_df.drop(columns=['Unassigned'])
    # stool_df.to_csv('stool_df.csv')
    df_len = stool_df.shape[1]
    real_rho, real_pval = spearmanr(saliva_df.T, stool_df.T, axis=1)

    if dump_rho:
        pickle.dump(real_rho, open("real_rho_" + str(tax_level) + ".pkl", "wb"))
        pickle.dump(real_pval, open("real_pval_" + str(tax_level) + ".pkl", "wb"))
        real_rho = pickle.load(open("real_rho" + str(tax_level) + ".pkl", "rb"))

    real_rho = real_rho[:df_len, df_len:]

    # mix_saliva_df = saliva_df.sample(frac=1).reset_index(drop=True)
    # mix_stool_df = stool_df.sample(frac=1).reset_index(drop=True)
    mix_saliva_df = shuffle(saliva_df)
    # each column should have its own scramble
    mix_stool_df = shuffle(stool_df)

    mix_rho, mix_pval = spearmanr(mix_saliva_df.T, mix_stool_df.T, axis=1)

    if dump_rho:
        pickle.dump(mix_rho, open("mix_rho_" + str(tax_level) + ".pkl", "wb"))
        pickle.dump(mix_pval, open("mix_pval_" + str(tax_level) + ".pkl", "wb"))
        mix_rho = pickle.load(open("mix_rho" + str(tax_level) + ".pkl", "rb"))

    mix_rho = mix_rho[:df_len, df_len:]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(real_rho, interpolation="nearest", cmap=cmap)
    plt.title('Saliva vs. Stool real features correlation-\n Taxanomy level = ' + str(tax_level))
    fig.colorbar(cax)
    # plt.show()
    plt.savefig("Saliva_vs_Stool_real_features_spearmanr_correlation_" + str(tax_level) + ".png")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(mix_rho, interpolation="nearest", cmap=cmap)
    plt.title('Saliva vs. Stool mix features correlation-\n Taxanomy level = ' + str(tax_level))
    fig.colorbar(cax)
    # plt.show()
    plt.savefig("Saliva_vs_Stool_mix_features_spearmanr_correlation_" + str(tax_level) + ".png")
    print("Saliva_vs_Stool_mix_features_spearmanr_correlation")

    # for correlation in [real_rho, mix_rho]:
    color_list = ["#e8e425", "#eda1da", "#71C1F6", "#9927E7", "#BEF49A", "#81FBD4"]

    if calc_adge_rows:
        correlation = real_rho
        upper_bound = np.percentile(correlation, 96)
        lower_bound = np.percentile(correlation, 3)
        high_corr_idx = {i: [] for i in range(len(correlation))}
        low_corr_idx = {i: [] for i in range(len(correlation))}
        for i in range(df_len):
            for j in range(df_len):
                if correlation[i][j] > upper_bound:
                    l = high_corr_idx[i]
                    l.append(j)
                    high_corr_idx[i] = l

                elif correlation[i][j] < lower_bound:
                    l = low_corr_idx[i]
                    l.append(j)
                    low_corr_idx[i] = l

        high_rows = []
        low_rows = []
        for i in range(len(correlation)):
            high = list(high_corr_idx[i])
            low = list(low_corr_idx[i])
            if len(high) > len(correlation)/2:
                high_rows.append(i)
            elif len(low) > len(correlation)/2:
                low_rows.append(i)

    if draw_corr:
        # check row distribution
        for title, edge_rows in zip(["high", "low"], [high_rows, low_rows]):
            for i, c in zip(range(len(edge_rows)), color_list[0:len(edge_rows)]):

                # ser = pd.Series(correlation[high_rows[i]])
                # ser_des = ser.describe()
                [count, bins] = np.histogram(correlation[edge_rows[i]], 50)
                plt.bar(bins[:-1], count / 10, width=0.8 * (bins[1] - bins[0]), alpha=0.4,
                        label="real " + title + " values for " + str(edge_rows[i]), color=c)

            plt.title("Real " + title + " values")
            plt.xlabel('Rho value')
            plt.ylabel('Number of bacteria')
            plt.legend()
            # print("Real tags_vs_Mixed_tags_at_" + title + "_combined.png")
            # plt.show()
            plt.savefig("Real_" + title + "_values.png")
            plt.close()

    if draw_vals:
        # check row distribution
        for title, edge_rows in zip(["high", "low"], [high_rows, low_rows]):
            for i, c in zip(range(len(edge_rows)), color_list[0:len(edge_rows)]):
                # ser = pd.Series(correlation[high_rows[i]])
                # ser_des = ser.describe()
                [count, bins] = np.histogram(saliva_df.iloc[edge_rows[i]], 50)
                plt.bar(bins[:-1], count / 10, width=0.8 * (bins[1] - bins[0]), alpha=0.4,
                        label="real " + title + " values for " + str(edge_rows[i]), color=c)

            plt.title("Saliva " + title + " values")
            plt.xlabel('Value')
            plt.ylabel('Number of bacteria')
            plt.legend()
            # print("Real tags_vs_Mixed_tags_at_" + title + "_combined.png")
            # plt.show()
            plt.savefig("Saliva_" + title + "_values.png")
            plt.close()

        for title, edge_rows in zip(["high", "low"], [high_rows, low_rows]):
            for i, c in zip(range(len(edge_rows)), color_list[0:len(edge_rows)]):
                # ser = pd.Series(correlation[high_rows[i]])
                # ser_des = ser.describe()
                [count, bins] = np.histogram(stool_df.iloc[edge_rows[i]], 50)
                plt.bar(bins[:-1], count / 10, width=0.8 * (bins[1] - bins[0]), alpha=0.4,
                        label="real " + title + " values for " + str(edge_rows[i]), color=c)

            plt.title("Stool " + title + " values")
            plt.xlabel('Value')
            plt.ylabel('Number of bacteria')
            plt.legend()
            # print("Real tags_vs_Mixed_tags_at_" + title + "_combined.png")
            # plt.show()
            plt.savefig("Stool_" + title + "_values.png")
            plt.close()


def calc_diabetic_healty_correlation(ddl, sub_set_of_ids, state_of_ids):
    preproccessed_data = ddl.preproccessed_data
    id_to_binary_tag_map = ddl.SampleID_to_Control_or_GDM_map
    draw_rhos_calculation_figure(id_to_binary_tag_map, preproccessed_data, 'Diabetic_Healthy_on_' + state_of_ids,
                                 save_folder='diabetic_healty_correlation', ids_list=sub_set_of_ids)


def calc_progress_pairs(t, samples, id_to_Type_map, id_to_trimester_map, id_to_body_site_map, id_to_Control_or_GDM_map, type, health):
    if health == 'Control':
        health = 1
    elif health == 'GDM':
        health = 2

    if type == 'SALIVA':
        type = 1
    elif type == 'STOOL':
        type = 2

    pairs = []
    df = pd.DataFrame(columns=['id', 'type', 'trimester', 'body_site', 'health'])
    for i, sample in enumerate(samples):  # check for the same type and trimester
        try:
            ty = id_to_Type_map[sample]
            tri = id_to_trimester_map[sample]
            b_site = id_to_body_site_map[sample]
            he = id_to_Control_or_GDM_map[sample]
            df.loc[i] = [sample] + [ty, tri, b_site, he]
        except KeyError:
            pass
    if len(df) > 1:
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                first_tri = df.iloc[i]['trimester']
                second_tri = df.iloc[j]['trimester']
                pair_health = df.iloc[i]['health']
                pair_type = df.iloc[i]['body_site']
                if df.iloc[i]['type'] == df.iloc[j]['type'] and df.iloc[i]['body_site'] == df.iloc[j]['body_site']\
                        and first_tri != second_tri:
                    # add [t, t+1] pairs
                    if first_tri - second_tri == -1:
                        if first_tri == t and pair_health == health and pair_type == type:
                            pairs.append([df.iloc[i]['id'], df.iloc[j]['id']])
                    elif first_tri - second_tri == 1:
                        if second_tri == t and pair_health == health and pair_type == type:
                            pairs.append([df.iloc[j]['id'], df.iloc[i]['id']])

    return pairs


def get_extreme_bact_in_corr(margin, val_to_comp):
    sub_set_map = {}
    all_pairs = []
    for type in ["SALIVA", 'STOOL']:
        for health in ["Control", "GDM"]:
            for t in [1, 2]:
                if t == 1:
                    tri = "1_2"
                if t == 2:
                    tri = "2_3"
                file = pickle.load(open(os.path.join("dynamics_rhos", "out_of_bound_corr_idx_" + tri + "_" + type + "_"
                                                     + health + ".pkl"), "rb"))
                pairs = [row[0] + "#" + row[1] for row in file]
                all_pairs += pairs
                pairs_map = {row[0] + "#" + row[1]: {"rho_score": row[2], "rho": row[3]} for row in file}
                # df = pd.DataFrame(file)
                # df.columns = ["bacteria_1", "bacteria_2", "rho_score", "rho"]
                sub_set_map[tri + "_" + type + "_" + health] = pairs_map

    # we want to find those bactrial who has a very different score and influnce in different states
    # create all pairs set to compare between
    all_pairs = list(set(all_pairs))
    to_remove = []
    for i, p in enumerate(all_pairs):
        p = p.split("#")
        if p[1] + "#" + p[0] in all_pairs[i:]:
            to_remove.append(p[1] + "#" + p[0])

    for r in to_remove:
        all_pairs.remove(r)

    dynamics_changed_bact_pairs = {}
    for pair in all_pairs:  # get score for each state
        pair_scores = {}
        for key, val in sub_set_map.items():
            if pair in val.keys():
                pair_scores[key] = val[pair][val_to_comp]  #'rho' or 'rho_score'
        p_s_items = list(pair_scores.items())
        for i, [state, score] in enumerate(p_s_items):
            for j in range(i+1, len(p_s_items)):
                difference = abs(score - p_s_items[j][1])  # 0 = rho, 1 = rho_score  !!!!!!!!!!!
                if difference > margin:
                    if pair in dynamics_changed_bact_pairs.keys():  # add
                        dynamics_changed_bact_pairs[pair].append(
                            {"state_1": state, "state_2": p_s_items[j][0], "difference": difference})
                    else:  # create
                        dynamics_changed_bact_pairs[pair] = [{"state_1": state, "state_2": p_s_items[j][0], "difference": difference }]

    with open(os.path.join("dynamics_rhos", val_to_comp + "_dynamics_changed_bacterial_pairs_margin-" + str(margin) + ".txt"), "w") as results_file:
        results_file.write("bacteria-1,bacteria-2,state-1,rho score-1,state-2,rho score-2,difference\n")
        for key, val in dynamics_changed_bact_pairs.items():
            two_bact = key.split("#")
            for m in val:
                one = sub_set_map[m['state_1']][key]['rho']
                two = sub_set_map[m['state_2']][key]['rho']
                results_file.write(two_bact[0] + "," + two_bact[1] + "," + m['state_1'] + "," + str(one) + "," +
                                   m['state_2'] + "," + str(two) + "," + str(abs(one - two)) + "\n")
    return 0


def visualize_extreme_bact_in_corr(margin, val_to_comp):
    labels = ["1_2_SALIVA_Control",  "1_2_SALIVA_GDM",
              "1_2_STOOL_Control",   "1_2_STOOL_GDM",
              "2_3_SALIVA_Control",   "2_3_SALIVA_GDM",
              "2_3_STOOL_Control",   "2_3_STOOL_GDM"]

    state_to_index_map = {"1_2_SALIVA_Control": 1,
                          "1_2_SALIVA_GDM": 2,
                          "1_2_STOOL_Control": 3,
                          "1_2_STOOL_GDM": 4,
                          "2_3_SALIVA_Control": 5,
                          "2_3_SALIVA_GDM": 6,
                          "2_3_STOOL_Control": 7,
                          "2_3_STOOL_GDM": 8}

    state_to_index_map = {"1_2_SALIVA_Control": 1,
                          "1_2_SALIVA_GDM": 2,
                          "2_3_SALIVA_Control": 3,
                          "2_3_SALIVA_GDM": 4,
                          "1_2_STOOL_Control": 5,
                          "1_2_STOOL_GDM": 6,
                          "2_3_STOOL_Control": 7,
                          "2_3_STOOL_GDM": 8}

    results_file = pd.read_csv((os.path.join("dynamics_rhos", val_to_comp + "_dynamics_changed_bacterial_pairs_margin-"
                                            + str(margin) + ".csv")))
    X = results_file['state-1']
    Y = results_file['state-2']
    X = [state_to_index_map[val] for val in X]  #  - 0.5
    Y = [state_to_index_map[val] for val in Y]
    X_Y = [(x, y) for x, y in zip(X, Y)]
    from collections import Counter
    point_to_occur = Counter(X_Y)
    X = []
    Y = []
    sizes = []
    for key, val in point_to_occur.items():
        X.append(key[0])
        Y.append(key[1])
        sizes.append(val)

    sizes = [s / max(sizes)*1000 for s in sizes]
    plt.title("Dynamics changed bacterial pairs")
    plt.axis([0, 8, 0, 10])
    plt.xticks(np.arange(0, 8), labels, rotation=45)
    plt.yticks(np.arange(1, 9), labels)
    plt.scatter(X, Y, c=sizes, s=sizes, alpha=0.3, cmap='viridis')
    plt.colorbar()
    plt.show()

    plt.savefig(os.path.join("dynamics_rhos", val_to_comp + "_dynamics_changed_bacterial_pairs_margin-"
                             + str(margin) + "_plot.png"))


def plt_delta_tri_corr(corr, t, type, health, linkage=False):
    pickle.dump(corr, open(os.path.join("saliva_stool", "Delta_trimester_" + str(t) + "_" + str(t+1) + '_' + type +
                                        '_' + health + "_correlation_df.pkl"), "wb"))
    bacterial = corr.index
    short_feature_names = []
    for f in bacterial:
        i = 1
        while len(f.split(";")[-i]) < 5:  # meaningless name
            i += 1
        short_feature_names.append(f.split(";")[-i])
    corr.index = list(range(len(short_feature_names)))
    corr.columns = list(range(len(short_feature_names)))

    for m in ["single", "ward"]:
        sns.set(color_codes=True)
        g = sns.clustermap(corr, method=m)
        if not type and not health:
            g.fig.suptitle('Delta trimester ' + str(t) + ' to ' + str(t + 1) + ' _cluster map (method=' + m + ')')
            g.savefig(os.path.join("saliva_stool",
                                   "Delta_trimester_" + str(t) + "_" + str(t + 1) + "_clustermap_method_" + m + ".png"))
        else:
            g.fig.suptitle('Delta trimester ' + str(t) + ' to ' + str(t+1) + ' ' + type + ' ' + health +
                           ' _cluster map (method=' + m + ')')
            g.savefig(os.path.join("saliva_stool", "Delta_trimester_" + str(t) + "_" + str(t+1) + '_' + type + '_' + health +
                                     "_clustermap_method_" + m + ".png"))

        """
        g.fig.suptitle('Delta trimester ' + str(t) + ' to ' + str(t + 1) + ' '  + ' cluster map (method=' + m + ')')
        g.savefig(os.path.join("saliva_stool", "Delta_trimester_" + str(t) + "_" + str(t + 1) + "_clustermap_method_" + m + ".png"))
        """
    if linkage:
        Z = linkage(corr, 'ward')
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        plt.savefig(os.path.join("saliva_stool", "Delta_trimester_" + str(t) + "_" + str(t+1) + '_' + type + '_' + health +
                                 "_ward_linkage.png"))

        Z = linkage(corr, 'single')
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        plt.savefig(os.path.join("saliva_stool", "Delta_trimester_" + str(t) + "_" + str(t+1) + '_' + type + '_' + health +
                                 "_single_linkage.png"))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(corr, interpolation="nearest", cmap=cmap)
    plt.title('Delta trimester ' + str(t) + ' to ' + str(t+1) + ' ' + type + ' ' + health + ' feature correlation')
    # plt.xlabel('Saliva')
    # plt.ylabel('Stool')
    fig.colorbar(cax)
    plt.savefig(os.path.join("saliva_stool", "Delta_trimester_" + str(t) + "_" + str(t+1) + '_' + type + '_' + health +
                             "_features_correlation.png"))


def split_ids(pairs):
    ids_1 = [id[0] for id in pairs]
    ids_2 = [id[1] for id in pairs]
    ids_len = len(pairs)
    return ids_1, ids_2, ids_len


def subtract_df(data, ids_1, ids_2, ids_len):
    delta_1_df = data.loc[ids_1, :]
    delta_1_df = delta_1_df.drop(columns=['Unassigned'])
    delta_2_df = data.loc[ids_2, :]
    delta_2_df = delta_2_df.drop(columns=['Unassigned'])
    delta_1_df.index = list(range(ids_len))
    delta_2_df.index = list(range(ids_len))
    delta_1_2_df = delta_2_df.subtract(delta_1_df)
    return delta_1_2_df


def calc_corr_from_pair_set_of_ids(pairs, data, bacterial, t, tri_to_tri, b_type, health, plt, high_low_idx):
    ids_1, ids_2, ids_len = split_ids(pairs)
    delta_df = subtract_df(data, ids_1, ids_2, ids_len)
    delta_df.index = ids_1
    delta_df_corr = delta_df.corr()

    if plt:
        plt_delta_tri_corr(delta_df_corr, t=t, type=b_type, health=health)
    high_low_idx = True


def calc_dynamics(ddl, from_pkl=False, plt=False):
    id_to_body_site_map = deepcopy(ddl.SampleID_to_body_site_map)
    woman_to_all_ids_map = deepcopy(ddl.Woman_to_all_SampleID_map)
    id_to_Type_map = deepcopy(ddl.SampleID_to_Type_map)
    id_to_trimester_map = deepcopy(ddl.SampleID_to_trimester_map)
    id_to_Control_or_GDM_map = deepcopy(ddl.SampleID_to_Control_or_GDM_map)

    # iterate all bacteria and create ds of values of t vs. t+1
    data = deepcopy(ddl.preproccessed_data)
    bacterial = data.columns[1:]
    tax_level = ddl.tax_level


    if from_pkl:
        # pickle.dump(all_1_2_woman_pairs, open("all_1_2_woman_pairs.pkl", "wb"))
        # pickle.dump(all_2_3_woman_pairs, open("all_2_3_woman_pairs.pkl", "wb"))
        all_1_2_woman_pairs = pickle.load(open("all_1_2_woman_pairs.pkl", "rb"))
        all_2_3_woman_pairs = pickle.load(open("all_2_3_woman_pairs.pkl", "rb"))
    else:
        if False:  # calculate clusters between all samples from 1-2 and 2-3
            all_1_2_woman_pairs = pickle.load(open("all_1_2_woman_pairs.pkl", "rb"))
            ids_1, ids_2, ids_len = split_ids(all_1_2_woman_pairs)
            delta_df = subtract_df(data, ids_1, ids_2, ids_len)
            delta_df_corr = delta_df.corr()
            plt_delta_tri_corr(delta_df_corr, t=1, type=None, health=None)


            all_2_3_woman_pairs = pickle.load(open("all_2_3_woman_pairs.pkl", "rb"))
            ids_1, ids_2, ids_len = split_ids(all_2_3_woman_pairs)
            delta_df = subtract_df(data, ids_1, ids_2, ids_len)
            delta_df_corr = delta_df.corr()
            plt_delta_tri_corr(delta_df_corr, t=2, type=None, health=None)

        t = 1
        tri_to_tri = "1_2"
        high_low_idx = True
        SALIVA_Control_1_2_woman_pairs = []
        for woman, samples in woman_to_all_ids_map.items():
            SALIVA_pairs = calc_progress_pairs(t, samples, id_to_Type_map, id_to_trimester_map, id_to_body_site_map, id_to_Control_or_GDM_map,
                                               type="SALIVA", health="Control")
            SALIVA_Control_1_2_woman_pairs = SALIVA_Control_1_2_woman_pairs + SALIVA_pairs

        calc_corr_from_pair_set_of_ids(SALIVA_Control_1_2_woman_pairs, data, bacterial, t, tri_to_tri, "SALIVA",
                                       "Control", plt, high_low_idx=high_low_idx)

        STOOL_Control_1_2_woman_pairs = []
        for woman, samples in woman_to_all_ids_map.items():
            STOOL_pairs = calc_progress_pairs(t, samples, id_to_Type_map, id_to_trimester_map, id_to_body_site_map, id_to_Control_or_GDM_map,
                                               type="STOOL", health="Control")
            STOOL_Control_1_2_woman_pairs = STOOL_Control_1_2_woman_pairs + STOOL_pairs
        calc_corr_from_pair_set_of_ids(STOOL_Control_1_2_woman_pairs, data, bacterial, t, tri_to_tri, "STOOL",
                                       "Control", plt, high_low_idx=high_low_idx)

        SALIVA_GDM_1_2_woman_pairs = []
        for woman, samples in woman_to_all_ids_map.items():
            SALIVA_pairs = calc_progress_pairs(t, samples, id_to_Type_map, id_to_trimester_map, id_to_body_site_map, id_to_Control_or_GDM_map,
                                               type="SALIVA", health="GDM")
            SALIVA_GDM_1_2_woman_pairs = SALIVA_GDM_1_2_woman_pairs + SALIVA_pairs
        calc_corr_from_pair_set_of_ids(SALIVA_GDM_1_2_woman_pairs, data, bacterial, t, tri_to_tri, "SALIVA",
                                       "GDM", plt, high_low_idx=high_low_idx)

        STOOL_GDM_1_2_woman_pairs = []
        for woman, samples in woman_to_all_ids_map.items():
            STOOL_pairs = calc_progress_pairs(t, samples, id_to_Type_map, id_to_trimester_map, id_to_body_site_map, id_to_Control_or_GDM_map,
                                               type="STOOL", health="GDM")
            STOOL_GDM_1_2_woman_pairs = STOOL_GDM_1_2_woman_pairs + STOOL_pairs
        calc_corr_from_pair_set_of_ids(STOOL_GDM_1_2_woman_pairs, data, bacterial, t, tri_to_tri, "STOOL",
                                       "GDM", plt, high_low_idx=high_low_idx)

        t = 2
        tri_to_tri = "2_3"
        SALIVA_Control_2_3_woman_pairs = []
        for woman, samples in woman_to_all_ids_map.items():
            SALIVA_pairs = calc_progress_pairs(t, samples, id_to_Type_map, id_to_trimester_map, id_to_body_site_map, id_to_Control_or_GDM_map,
                                               type="SALIVA", health="Control")
            SALIVA_Control_2_3_woman_pairs = SALIVA_Control_2_3_woman_pairs + SALIVA_pairs
        calc_corr_from_pair_set_of_ids(SALIVA_Control_2_3_woman_pairs, data, bacterial, t, tri_to_tri, "SALIVA",
                                       "Control", plt, high_low_idx=high_low_idx)

        STOOL_Control_2_3_woman_pairs = []
        for woman, samples in woman_to_all_ids_map.items():
            STOOL_pairs = calc_progress_pairs(t, samples, id_to_Type_map, id_to_trimester_map, id_to_body_site_map, id_to_Control_or_GDM_map,
                                              type="STOOL", health="Control")
            STOOL_Control_2_3_woman_pairs = STOOL_Control_2_3_woman_pairs + STOOL_pairs
        calc_corr_from_pair_set_of_ids(STOOL_Control_2_3_woman_pairs, data, bacterial, t, tri_to_tri, "STOOL",
                                       "Control", plt, high_low_idx=high_low_idx)

        SALIVA_GDM_2_3_woman_pairs = []
        for woman, samples in woman_to_all_ids_map.items():
            SALIVA_pairs = calc_progress_pairs(t, samples, id_to_Type_map, id_to_trimester_map, id_to_body_site_map, id_to_Control_or_GDM_map,
                                               type="SALIVA", health="GDM")
            SALIVA_GDM_2_3_woman_pairs = SALIVA_GDM_2_3_woman_pairs + SALIVA_pairs
        calc_corr_from_pair_set_of_ids(SALIVA_GDM_2_3_woman_pairs, data, bacterial, t, tri_to_tri, "SALIVA",
                                       "GDM", plt, high_low_idx=high_low_idx)

        STOOL_GDM_2_3_woman_pairs = []
        for woman, samples in woman_to_all_ids_map.items():
            STOOL_pairs = calc_progress_pairs(t, samples, id_to_Type_map, id_to_trimester_map, id_to_body_site_map, id_to_Control_or_GDM_map,
                                              type="STOOL", health="GDM")
            STOOL_GDM_2_3_woman_pairs = STOOL_GDM_2_3_woman_pairs + STOOL_pairs
        calc_corr_from_pair_set_of_ids(STOOL_GDM_2_3_woman_pairs, data, bacterial, t, tri_to_tri, "STOOL",
                                       "GDM", plt, high_low_idx=high_low_idx)


def get_svm_clf(title):
    if title == 'diabetic_healthy':
        clf = svm.SVC(kernel='linear', C=1000, gamma='auto', class_weight='balanced')

    return clf


def predict_GMM(ddl, TUNED_PAREMETERS=False, Cross_validation=10):
    ids, tag_map, id_to_features_map, task_name = ddl.get_diabetic_healthy_task_data
    # get trimester 1 sub set of ids
    ids = ddl.tri_1_ids
    X = [id_to_features_map[id] for id in ids]
    y = [tag_map[id] for id in ids]

    print("SVM...")
    if TUNED_PAREMETERS:
        svm_tuned_parameters = [{'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                                 'gamma': ['auto', 'scale'],
                                 'C': [0.01, 0.1, 1, 10, 100, 1000]}]

        svm_clf = GridSearchCV(svm.SVC(class_weight='balanced'), svm_tuned_parameters, cv=5,
                               scoring='roc_auc', return_train_score=True)

        svm_clf.fit(X, y)
        print(svm_clf.best_params_)
        print(svm_clf.best_score_)
        print(svm_clf.cv_results_)

        svm_results = pd.DataFrame(svm_clf.cv_results_)
        svm_results.to_csv("svm_all_results_df_" + task_name + ".csv")
        pickle.dump(svm_results, open("svm_all_results_df_" + task_name + ".pkl", 'wb'))



    # Split the data set
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []
    svm_y_test_from_all_iter = []
    svm_y_score_from_all_iter = []
    svm_y_pred_from_all_iter = []
    svm_class_report_from_all_iter = []
    svm_coefs = []

    train_accuracies = []
    test_accuracies = []
    confusion_matrixes = []
    y_train_preds = []
    y_test_preds = []

    for i in range(Cross_validation):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

    bacteria_average = []
    bacteria_coeff_average = []

    for iter_num in range(Cross_validation):
        print(f'------------------------------\nIteration number {iter_num}')
        # SVM
        clf = get_svm_clf(task_name)

        clf.fit(X_trains[iter_num], y_trains[iter_num])
        y_score = clf.decision_function(X_tests[iter_num])  # what is this for?
        y_pred = clf.predict(X_tests[iter_num])
        y_test_preds.append(y_pred)
        svm_class_report = classification_report(y_tests[iter_num], y_pred).split("\n")
        train_pred = clf.predict(X_trains[iter_num])
        y_train_preds.append(train_pred)
        train_accuracies.append(accuracy_score(y_trains[iter_num], train_pred))
        test_accuracies.append(accuracy_score(y_tests[iter_num], y_pred))  # same as - clf.score(X_test, y_test)

        confusion_matrixes.append(confusion_matrix(y_tests[iter_num], y_pred))

        try:
            _, _, _, svm_roc_auc = roc_auc(y_tests[iter_num], y_pred, verbose=True, visualize=False,
                                           graph_title='SVM\n' + str(iter_num), save=True)
        except ValueError:
            print(task_name + " multiclass format is not supported at roc auc calculation")

        # save the y_test and y_score
        svm_y_test_from_all_iter.append(y_tests[iter_num])  # .values)
        svm_y_score_from_all_iter.append(list(y_score))
        svm_y_pred_from_all_iter.append(list(y_pred))
        svm_class_report_from_all_iter.append(svm_class_report)


    all_y_train, all_predictions_train, all_test_real_tags, all_test_pred_tags, train_auc, test_auc, train_rho, \
    test_rho = calc_auc_on_joined_results(Cross_validation, y_trains, y_train_preds, y_tests, y_test_preds)

    print("\n------------------------------\n")
    try:
        _, _, _, svm_roc_auc = roc_auc(all_test_real_tags, all_test_pred_tags, verbose=False, visualize=True,
                                       graph_title='SVM\n' + task_name + " AUC on all iterations", save=True)
    except ValueError:
        print(task_name + "multiclass format is not supported at roc auc calculation")

    confusion_matrix_average, confusion_matrix_acc, confusion_matrix_indexes = \
        edit_confusion_matrix(task_name, confusion_matrixes, ddl, "SVM")
    print_confusion_matrix(confusion_matrix_average, confusion_matrix_indexes, confusion_matrix_acc, "SVM",
                           task_name)

    print("svm final results: " + task_name)
    print("train_auc: " + str(train_auc))
    print("test_auc: " + str(test_auc))
    print("train_rho: " + str(train_rho))
    print("test_rho: " + str(test_rho))
    print("confusion_matrix_average: ")
    print(confusion_matrix_average)
    print("confusion_matrix_acc: ")
    print(confusion_matrix_acc)
    confusion_matrix_average.to_csv("svm_confusion_matrix_average_on_" + task_name + ".txt")

    with open("svm_AUC_on_" + task_name + ".txt", "w") as file:
        file.write("train_auc: " + str(train_auc) + "\n")
        file.write("test_auc: " + str(test_auc) + "\n")
        file.write("train_rho: " + str(train_rho) + "\n")
        file.write("test_rho: " + str(test_rho) + "\n")

    # save results to data frame
    results = pd.DataFrame(
        {
            "train accuracy": train_accuracies,
            "test accuracy": test_accuracies,
            "y score": svm_y_score_from_all_iter,
            "class report": svm_class_report_from_all_iter,
            "y test": svm_y_test_from_all_iter,
            "y pred": svm_y_pred_from_all_iter,
        }
    )

    results.append([np.mean(train_accuracies), np.mean(test_accuracies), None, None, None, None])
    pickle.dump(results, open("svm_clf_results_" + task_name + ".pkl", 'wb'))
    results.to_csv("svm_clf_results_" + task_name + ".csv")
    pickle.dump(confusion_matrix_average, open("svm_clf_confusion_matrix_results_" + task_name + ".pkl", 'wb'))
    confusion_matrix_average.to_csv("svm_clf_confusion_matrix_results_" + task_name + ".csv")

    return 0

    # learn


def clac_distance_between_samples(id_to_features_map, sample_1, sample_2):
    features_1 = np.array(id_to_features_map[sample_1])
    features_2 = np.array(id_to_features_map[sample_2])
    dst = np.linalg.norm(features_1 - features_2)
    return dst


def clac_distance_between_all_samples_and_plot(ddl):
    id_to_features_map = ddl.id_to_features_map
    ids = id_to_features_map.keys()
    saliva = [z for z in ids if 'SALIVA' in z]
    stool = [z for z in ids if 'STOOL' in z]
    meconium = [z for z in ids if 'Meconium' in z]
    ids = saliva + stool + meconium
    if False:
        df = pd.DataFrame(columns=ids, index=ids)
        for id_1 in ids:
            for id_2 in ids:
                distance = clac_distance_between_samples(id_to_features_map, id_1, id_2)
                df[id_1][id_2] = distance
                df[id_2][id_1] = distance
        df.to_csv("norm_2_distance_all_samples_sorted_by_body_part.csv")

    df = pd.read_csv("norm_2_distance_all_samples.csv")
    df_array = df.to_numpy()
    ax = sns.heatmap(df_array.astype(int))
    plt.show()

def calc_clusters(ddl, folder="clustering"):
    # distance_matrix = pd.read_csv("norm_2_distance_all_samples.csv")
    # distance_matrix[i][j] = np.exp(- distance_matrix[i][j] ** 2)

    bad_bacteria = ['H019PS-SALIVA-T1rep-2', 'C009AZ-Meconium-T4-rep-1', 'H033RB-SALIVA-T1rep-1',
                    'B020MS-SALIVA-T2-rep-1', 'H049LH-SALIVA-T3-rep-1', 'H092RN-Meconium-T4-rep-1',
                    'R050BL-SALIVA-T2-rep-1', 'B019AV-Meconium-T4-rep-1', 'B2031EB-Meconium-T4-rep-1',
                    'H087MA-SALIVA-T3-rep-1', 'B2019MD-Meconium-T4-rep-1', 'B2030RM-Meconium-T4-rep-1',
                    'N000NN-Meconium-T4-rep-1', 'C001LK-Meconium-T4-rep-1', 'H230SA-STOOL-T1-rep-1',
                    'R036DS-Meconium-T4-rep-1']

    ids, tag_map, id_to_features_map, task_name = ddl.get_diabetic_healthy_task_data

    # for bad in bad_bacteria:
    #     ids.remove(bad)
    # mec_ids = [id for id in ids if 'Meconium' in id ]
    X = np.array([id_to_features_map[id] for id in ids])
    y = [tag_map[id] for id in ids]

    for num_of_clusters in [4]:
        print("\n-----------------------------------\n" + str(num_of_clusters) + " clusters:")
        spectral_clustering = SpectralClustering(affinity='rbf', assign_labels='discretize', coef0=1,
                           degree=3, eigen_solver=None, eigen_tol=0.0, gamma=1.0,
                           kernel_params=None, n_clusters=num_of_clusters, n_init=10, n_jobs=None,
                           n_neighbors=10, random_state=0)

        agglomerative_clustering = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                                connectivity=None, linkage='average', memory=None, n_clusters=num_of_clusters,
                                pooling_func='deprecated')

        mean_shift_clustering = MeanShift(bandwidth=num_of_clusters, bin_seeding=False, cluster_all=True, min_bin_freq=1,
                  n_jobs=None, seeds=None)

        kmean_clustering = KMeans(n_clusters=num_of_clusters, random_state=0).fit(X)

        affinity_propagation_clustering = AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
                            damping=0.5, max_iter=200, preference=None, verbose=False)

        for clust in [agglomerative_clustering]:
            clust.fit(X)
            division_to_clusters = clust.labels_
            distribution = Counter(division_to_clusters)

        body_part_cluster_df = pd.DataFrame(index=range(1, num_of_clusters + 1),
                                            columns=["saliva", "stool", "meconium"])
        GDM_cluster_df = pd.DataFrame(index=range(1, num_of_clusters + 1), columns=["GDM+", "GDM-"])
        trimester_cluster_df = pd.DataFrame(index=range(1, num_of_clusters + 1),
                                            columns=["1", "2", "postpartum", "petus"])
        file_cluster_df = pd.DataFrame(index=range(1, num_of_clusters + 1), columns=["file_52", "file_58", "file_59",
                                                                                     "file_73", "file_86", "file_89",
                                                                                     "file_90", "file_102", "file_104"])
        for c in range(num_of_clusters):

            # fill data frames using maps
            SampleID_to_trimester_map = ddl.SampleID_to_trimester_map
            SampleID_to_body_site_map = ddl.SampleID_to_body_site_map
            SampleID_to_Control_or_GDM_map = ddl.SampleID_to_Control_or_GDM_map
            SampleID_to_file_map = ddl.SampleID_to_file_map

            sub_group = [ids[i] for i, val in enumerate(division_to_clusters) if val == c]
            sub_g_size = len(sub_group)
            num_of_samples = len(y)
            saliva_counter = len([0 for z in sub_group if 'SALIVA' in z])
            stool_counter = len([0 for z in sub_group if 'STOOL' in z])
            meconium_counter = len([0 for z in sub_group if 'Meconium' in z])

            body_part_cluster_df.iloc[c] = [round(saliva_counter/num_of_samples, 5),
                                            round(stool_counter/num_of_samples, 5),
                                            round(meconium_counter/num_of_samples, 5)]


            t1_counter = len([0 for z in sub_group if 'T1' in z])
            t2_counter = len([0 for z in sub_group if 'T2' in z])
            t3_counter = len([0 for z in sub_group if 'T3' in z])
            t4_counter = len([0 for z in sub_group if 'T4' in z])

            trimester_cluster_df.iloc[c] = [round(t1_counter/num_of_samples, 5), round(t2_counter/num_of_samples, 5),
                                            round(t3_counter/num_of_samples, 5), round(t4_counter/num_of_samples, 5)]

            GDM_plus_counter = len([0 for z in sub_group if SampleID_to_Control_or_GDM_map[z] == 1])
            GDM_minus_counter = len([0 for z in sub_group if SampleID_to_Control_or_GDM_map[z] == 2])

            GDM_cluster_df.iloc[c] = [round(GDM_plus_counter/num_of_samples, 5), round(GDM_minus_counter/num_of_samples, 5)]


            file_52_counter = len([0 for z in sub_group if SampleID_to_file_map[z] == "52"])
            file_58_counter = len([0 for z in sub_group if SampleID_to_file_map[z] == "58"])
            file_59_counter = len([0 for z in sub_group if SampleID_to_file_map[z] == "59"])
            file_73_counter = len([0 for z in sub_group if SampleID_to_file_map[z] == "73"])
            file_86_counter = len([0 for z in sub_group if SampleID_to_file_map[z] == "86"])
            file_89_counter = len([0 for z in sub_group if SampleID_to_file_map[z] == "89"])
            file_90_counter = len([0 for z in sub_group if SampleID_to_file_map[z] == "90"])
            file_102_counter = len([0 for z in sub_group if SampleID_to_file_map[z] == "102"])
            file_104_counter = len([0 for z in sub_group if SampleID_to_file_map[z] == "104"])

            file_cluster_df.iloc[c] = [round(file_52_counter / num_of_samples, 5),
                                       round(file_58_counter / num_of_samples, 5),
                                       round(file_59_counter / num_of_samples, 5),
                                       round(file_73_counter / num_of_samples, 5),
                                       round(file_86_counter / num_of_samples, 5),
                                       round(file_89_counter / num_of_samples, 5),
                                       round(file_90_counter / num_of_samples, 5),
                                       round(file_102_counter / num_of_samples, 5),
                                       round(file_104_counter / num_of_samples, 5)]

            """
            print("group-" + str(c) + ": size- " + str(sub_g_size) + "\nsaliva=" + str(round(saliva_counter/sub_g_size*100, 2)) +
                  "% stool=" + str(round(stool_counter/sub_g_size*100, 2)) + "% meconium=" + str(round(meconium_counter/sub_g_size*100, 2)) +
                  "%\n T1=" + str(round(t1_counter/sub_g_size*100, 2))
                  + "% T2=" + str(round(t2_counter/sub_g_size*100, 2)) + "% T3=" + str(round(t3_counter/sub_g_size*100, 2))
                  + "% T4=" + str(round(t4_counter/sub_g_size*100, 2)) + "%")
        print("\n-----------------------------------\n")
            """
    body_part_cluster_df.to_csv(os.path.join(folder, "body_part_cluster_df.csv"))
    for df, title in zip([body_part_cluster_df, GDM_cluster_df, trimester_cluster_df, file_cluster_df],
                         ["Body Part", "GDM", "Trimester", "File"]):
        df_array = [list(row) for row in df.as_matrix()]
        fig, ax = plt.subplots()
        ax = sns.heatmap(df_array, xticklabels=list(df.columns), yticklabels=list(df.index),
                         annot=True,  cmap='Blues', ax=ax)

        # calculate mutal information
        p_df_rows = sum([x * math.log10(x) for x in df.sum(axis=0) if x > 0])
        p_df_columns = sum([y * math.log10(y) for y in df.sum(axis=1) if y > 0])
        p_df_cells = 0
        for row in df_array:
            for item in row:
                if item > 0:
                    p_df_cells += item * math.log10(item)
        mi = p_df_rows + p_df_columns - p_df_cells

        plt.title("Cluster Distribution Of " + title + " - MI=" + str(round(mi, 4)))
        plt.savefig(os.path.join(folder, title.replace(" ", "_") + "_cluster_distribution.png"))




    GDM_cluster_df.to_csv(os.path.join(folder, "GDM_cluster_df.csv"))
    trimester_cluster_df.to_csv(os.path.join(folder, "trimester_cluster_df.csv"))
    file_cluster_df.to_csv(os.path.join(folder, "file_cluster_df.csv"))

    labels_true = y
    clusters_pred = [c+1 for c in clust.labels_]
    metrics.adjusted_mutual_info_score(labels_true, clusters_pred)
    cluster_1 = [ids[i] for i, c in enumerate(clusters_pred) if c == 1]
    cluster_2 = [ids[i] for i, c in enumerate(clusters_pred) if c == 2]
    cluster_3 = [ids[i] for i, c in enumerate(clusters_pred) if c == 3]
    cluster_4 = [ids[i] for i, c in enumerate(clusters_pred) if c == 4]





if __name__ == "__main__":  # microbial
    microbial_file_path = 'merged_GDM_tables_w_tax.csv'
    mapping_file_path = 'ok.csv'
    ddl = DiabetesDataLoader(microbial_file_path, mapping_file_path, to_csv=False, merge_files=False, tax_level=5,
                             add_tax_row=False, draw_self_corr_mat=False, from_pkl=True)
    id_to_features_map = ddl.id_to_features_map
    margin = 1.75
    val_to_comp = 'rho_score'  # 'rho' or 'rho_score'

    if False:
        calc_saliva_stool_correlation(ddl)
    if False:
        # in stead of running task on all samples, divide to states.
        ids_sub_sets, ids_states = ddl.get_ids_sub_sets_and_states
        for sub_set_of_ids, state_of_ids in zip(ids_sub_sets, ids_states):
            try:
                calc_diabetic_healty_correlation(ddl, sub_set_of_ids, state_of_ids)
            except ValueError:
                print(state_of_ids + " has only " + str(len(sub_set_of_ids)) + " samples")

        for sub_set_of_ids, state_of_ids in zip(ids_sub_sets, ids_states):
            print(state_of_ids.replace("_", " ") + " - " + str(len(sub_set_of_ids)) + " samples.")

    if False:
        run_diabetic_healthy_task(ddl, TUNED_PAREMETERS=True)

    if False:
        # calc_dynamics(ddl, plt=False)
        get_extreme_bact_in_corr(margin, val_to_comp)
    if False:
        visualize_extreme_bact_in_corr(margin, val_to_comp)
    if False:
        predict_GMM(ddl)
    if False:
        rare_bact = ddl.find_rare_bacteria(ddl.OtuMf)
    if False:
        # 'B001VVC-STOOL-T1-rep-1', 'B002LA-Meconium-T4-rep-1', 'B002LA-STOOL-T1-rep-1', 'B002LA-STOOL-T1-rep-2', 'B003SC-STOOL-T1-rep-1'
        # clac_distance_between_samples(id_to_features_map, 'B001VVC-STOOL-T1-rep-1', 'B002LA-Meconium-T4-rep-1')
        clac_distance_between_all_samples_and_plot(ddl)
    if True:


        calc_clusters(ddl)

