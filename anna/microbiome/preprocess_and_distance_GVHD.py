from load_merge_otu_mf import OtuMfHandler
from preprocess import preprocess_data
from pca import *
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

def gvhd(perform_distance=True,level =3):
    otu = 'C:/Users/Anna/Documents/otu_saliva_GVHD.csv'
    mapping = 'C:/Users/Anna/Documents/mapping_saliva_GVHD.csv'
    OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
    preproccessed_data = preprocess_data(OtuMf.otu_file, taxnomy_level=7)
    mapping_file = OtuMf.mapping_file
    mapping_file['DATE'] = pd.to_datetime(OtuMf.mapping_file['DATE'])
    mapping_file['Date_Of_Transplantation'] = pd.to_datetime(OtuMf.mapping_file['Date_Of_Transplantation'])
    mapping_file['Date_of_engraftmen'] = pd.to_datetime(OtuMf.mapping_file['Date_of_engraftmen'])
    mapping_file['aGVHD1_Stat'] = pd.to_datetime(OtuMf.mapping_file['aGVHD1_Stat'])
    mapping_file['cGVHD_start'] = pd.to_datetime(OtuMf.mapping_file['cGVHD_start'])
    end = pd.to_datetime('2020-01-01')
    mapping_file['aGVHD1_Stat'] = mapping_file['aGVHD1_Stat'].fillna(end)
    mapping_file['cGVHD_start'] = mapping_file['cGVHD_start'].fillna(end)
    mapping_file = mapping_file[(mapping_file['DATE'] > mapping_file['Date_Of_Transplantation']) & (
                mapping_file['DATE'] < mapping_file['aGVHD1_Stat']) & (mapping_file['DATE'] < mapping_file['cGVHD_start'])].sort_values(['Personal_ID', 'DATE'])

    mapping_file = mapping_file.reset_index()
    mapping_file = mapping_file.sort_values("DATE").groupby("Personal_ID", as_index=False).last().set_index('#SampleID')

    mapping_file = mapping_file[['MTX', 'aGVHD1 ', 'cGVHD ']]
    mapping_file = mapping_file.fillna('No')
    # preproccessed_data = preproccessed_data.join(mapping_file[['MTX', 'aGVHD1 ', 'cGVHD ']], how='inner')
    # preproccessed_data = preproccessed_data.fillna('No')

    mapping_yes_no = {'Yes': 1, 'No': 0}
    mapping_file['aGVHD1 '] = mapping_file['aGVHD1 '].map(mapping_yes_no)
    mapping_file['cGVHD '] = mapping_file['cGVHD '].map(mapping_yes_no)
    mapping_file['MTX'] = mapping_file['MTX'].map(mapping_yes_no)
    mapping_file["disease"] = mapping_file["aGVHD1 "].map(str) + '_' + mapping_file["cGVHD "].map(str)
    mapping_diseases = {'0_0': 1, '1_0': 0, '0_1': 0, '1_1': 0}
    mapping_file["disease"] = mapping_file["disease"].map(mapping_diseases)
    mapping_file = mapping_file.drop(['aGVHD1 ', 'cGVHD '], axis=1)
    preproccessed_data = preproccessed_data.join(mapping_file, how='inner')
    preproccessed_data = preproccessed_data.drop(['MTX','disease'], axis =1)
    if perform_distance:
        cols = [col for col in preproccessed_data.columns if len(preproccessed_data[col].unique()) != 1]
        dict_bact = {'else': []}
        for col in preproccessed_data[cols]:
            col_name = preproccessed_data[col].name.split(';')
            bact_level = level - 1
            if len(col_name) > bact_level:
                if col_name[bact_level] in dict_bact:
                    dict_bact[col_name[bact_level]].append(preproccessed_data[col].name)
                else:
                    dict_bact[col_name[bact_level]] = [preproccessed_data[col].name]
            else:
                dict_bact['else'].append(preproccessed_data[col].name)
            print(col_name[-1])

        new_df = pd.DataFrame(index=preproccessed_data.index)
        col = 0
        new_dict = {}
        for key, values in dict_bact.items():
            new_dict[key] = []
            new_data = preproccessed_data[values]
            pca = PCA(n_components=round(new_data.shape[1] / 2) + 1)
            pca.fit(new_data)
            sum = 0
            num_comp = 0
            for (i, component) in enumerate(pca.explained_variance_ratio_):
                if sum <= 0.5:
                    sum += component
                else:
                    num_comp = i
                    break
            if num_comp == 0:
                num_comp += 1
            otu_after_pca_new, pca_components = apply_pca(new_data, n_components=num_comp)
            for j in range(otu_after_pca_new.shape[1]):
                new_df[col + j] = otu_after_pca_new[j]
                new_dict[key].append(col + j)
            col += num_comp
        return new_df, mapping_file,  new_dict, OtuMf.otu_file.T['taxonomy'].values
    else:
        return preproccessed_data, mapping_file, {}, []
