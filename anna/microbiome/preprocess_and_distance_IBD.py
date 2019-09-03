from load_merge_otu_mf import OtuMfHandler
from preprocess import preprocess_data
from pca import *
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

def ibd(perform_distance=True,level =3):
    otu = 'C:/Users/Anna/Documents/otu_IBD3.csv'
    mapping = 'C:/Users/Anna/Documents/mapping_IBD3.csv'
    OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
    preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=7)
    preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['CD_or_UC', 'preg_trimester', 'P-ID']],
                                                 how='inner')
    preproccessed_data = preproccessed_data.loc[(preproccessed_data['CD_or_UC'] != 'control')]
    preproccessed_data = preproccessed_data.groupby(['CD_or_UC', 'preg_trimester', 'P-ID'], as_index=False).mean()
    new_set2 = preproccessed_data.groupby(['preg_trimester']).mean()
    for i in range(0, len(preproccessed_data)):
        month = preproccessed_data['preg_trimester'][i]
        preproccessed_data.iloc[i:i + 1, 3:preproccessed_data.shape[1]] = (
                    preproccessed_data.iloc[i:i + 1, 3:preproccessed_data.shape[1]].values - new_set2.loc[month:month,
                                                                                             :].values)

    preproccessed_data = preproccessed_data.drop(['preg_trimester', 'P-ID','CD_or_UC'], axis=1)
    mapping_file = OtuMf.mapping_file.loc[(OtuMf.mapping_file['CD_or_UC'] != 'control')]
    mapping_disease = {'CD': 1, 'UC': 0}
    mapping_file['CD_or_UC'] = mapping_file['CD_or_UC'].map(mapping_disease)
    mapping_file = mapping_file['CD_or_UC']
    mapping_file = mapping_file.reset_index()
    if perform_distance:
        cols = [col for col in preproccessed_data.columns if len(preproccessed_data[col].unique()) != 1]
        dict_bact = {'else': []}
        for col in preproccessed_data[cols]:
            col_name = preproccessed_data[col].name.split(';')
            bact_level = level - 1
            if len(col_name) > bact_level:
                #if ",".join(col_name[0:bact_level+1]) in dict_bact:
                #    dict_bact[",".join(col_name[0:bact_level+1])].append(preproccessed_data[col].name)
                #else:
                #    dict_bact[",".join(col_name[0:bact_level+1])] = [preproccessed_data[col].name]
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
        return new_df, mapping_file, new_dict, OtuMf.otu_file.T['taxonomy'].values
    else:
        return preproccessed_data, mapping_file, {}

