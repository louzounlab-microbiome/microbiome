import pickle
import sys
import os
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from infra_functions.preprocess_grid import preprocess_data
from Plot import draw_X_y_rhos_calculation_figure, PCA_t_test, plot_data_3d, plot_data_2d



class CreateOtuAndMappingFiles(object):
    # get two relative path of csv files
    def __init__(self, otu_file_path, tags_file_path):
        self.otu_path = otu_file_path
        self.tags_path = tags_file_path
        self.tag_name = tags_file_path.split(".")[0]
        print('read tag file...')
        self.extra_features_df = pd.read_csv(self.tags_path).drop(['Tag'], axis=1)
        self.tags_df = pd.read_csv(self.tags_path)[['ID', 'Tag']]
        # set index as ID
        self.tags_df = self.tags_df.set_index('ID')
        self.extra_features_df = self.extra_features_df.set_index('ID')
        # subset of ids according to the tags data frame
        self.ids = self.tags_df.index.tolist()
        self.ids.append('taxonomy')
        print('read otu file...')
        self.otu_features_df = pd.read_csv(self.otu_path)
        self.otu_features_df = self.otu_features_df.set_index('ID')
        self.pca_ocj = None

    def csv_to_learn(self, task_name, folder, tax, pca_n):
        if not os.path.exists(folder):
            os.mkdir(folder)
        tag_ids = list(self.tags_df.index)
        otu_ids = list(self.otu_features_df_b_pca.index)
        mutual_ids = [id for id in tag_ids if id in otu_ids]
        # ids_to_drop = list(set(self.otu_features_df.index.values) - set(self.tags_df.index.values))
        # use only the samples the correspond to the tag file
        self.otu_features_df = self.otu_features_df.loc[mutual_ids]
        self.tags_df = self.tags_df.loc[mutual_ids]
        # concat with extra features by index
        df = self.otu_features_df.join(self.extra_features_df)
        # create a new csv file
        otu_path = os.path.join(folder, 'OTU_merged_' + str(task_name) + "_tax_level_" + str(tax) + '_pca_' + str(pca_n) + '.csv')
        df.to_csv(otu_path)
        tag_path = os.path.join(folder, 'Tag_file_' + str(task_name) + '.csv')
        self.tags_df.to_csv(tag_path)
        if self.pca_ocj:
            pca_path = os.path.join(folder, "Pca_obj_" + str(task_name) + '_pca_' + str(pca_n) + '.pkl')
            pickle.dump(self.pca_ocj, open(pca_path, "wb"))
        else:
            pca_path = "No pca created"
        with open(os.path.join(folder, "bacteria_" + str(task_name) + "_tax_level_" + str(tax) + ".txt"), "w") as bact_file:
            for col in self.bacteria:
                bact_file.write(col + "\n")
        return otu_path, tag_path, pca_path

    def preprocess(self, preprocess_params, visualize):
        print('preprocess...')
        self.otu_features_df, self.otu_features_df_b_pca,  self.pca_ocj, self.bacteria = \
            preprocess_data(self.otu_features_df, preprocess_params, self.tags_df, self.tag_name, visualize_data=visualize)

    def rhos_and_pca_calculation(self, task, tax, pca, rhos_folder, pca_folder):
        # -------- rhos calculation --------
        tag_ids = list(self.tags_df.index)
        otu_ids = list(self.otu_features_df_b_pca.index)
        mutual_ids = [id for id in tag_ids if id in otu_ids]
        X = self.otu_features_df_b_pca.loc[mutual_ids]
        y = np.array(list(self.tags_df.loc[mutual_ids]["Tag"])).astype(int)

        if not os.path.exists(rhos_folder):
            os.makedirs(rhos_folder)
        draw_X_y_rhos_calculation_figure(X, y, task, tax, save_folder=rhos_folder)

        # -------- PCA visualizations --------
        if not os.path.exists(pca_folder):
            os.makedirs(pca_folder)
        PCA_t_test(group_1=[x for x, y in zip(X.values, y) if y == 0], group_2=[x for x, y in zip(X.values, y) if y == 1],
                   title="T test for PCA dimentions on " + task, save=True, folder=pca_folder)
        if pca >= 2:
            plot_data_2d(X.values, y, data_name=task.capitalize(), save=True, folder=pca_folder)
            if pca >= 3:
               plot_data_3d(X.values, y, data_name=task.capitalize(), save=True, folder=pca_folder)


if __name__ == "__main__":
    task = 'prognostic_PTSD_task'
    bactria_as_feature_file = '../sderot_anxiety/PTSD_data.csv'
    samples_data_file = '../sderot_anxiety/PTSD_tag.csv'
    rhos_folder = os.path.join('..', 'sderot_anxiety', 'rhos')
    pca_folder = os.path.join('..', 'sderot_anxiety', 'pca')

    # parameters for preprocess
    tax = 5
    preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 0.1, 'normalization': 'log',
                       'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': 2}

    mapping_file = CreateOtuAndMappingFiles(bactria_as_feature_file, samples_data_file)
    mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=True)
    mapping_file.rhos_and_pca_calculation(task, preprocess_prms['taxonomy_level'], preprocess_prms['pca'],
                                          rhos_folder, pca_folder)

    otu_path, tag_path, pca_path = mapping_file.csv_to_learn('PTST_task', os.path.join('..', 'sderot_anxiety'), tax=tax)

    print(otu_path)
    print(tag_path)
    print(pca_path)


