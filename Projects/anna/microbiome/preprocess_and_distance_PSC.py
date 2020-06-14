from load_merge_otu_mf import OtuMfHandler
from Preprocess import preprocess_data
from pca import *
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut, KFold
from sklearn import metrics
from xgboost import XGBClassifier


def psc(perform_distance=True,level =3):
    otu = 'C:/Users/Anna/Desktop/docs/otu_psc2.csv'
    mapping = 'C:/Users/Anna/Desktop/docs/mapping_psc.csv'
    OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
    print('using padp')
    preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=7)
    mapping_file = OtuMf.mapping_file

    mapping_disease = {'Control':0,'Cirrhosis ':1, 'HCC':1, 'PSC+IBD':2,'PSC':2}
    mapping_file['DiagnosisGroup'] = mapping_file['DiagnosisGroup'].map(mapping_disease)
    mappin_boolean = {'yes' :1, 'no': 0, 'Control': 0, '0':0, '1':1}
    mapping_file['FattyLiver'] = mapping_file['FattyLiver'].map(mappin_boolean)
    mapping_file['RegularExercise'] = mapping_file['RegularExercise'].map(mappin_boolean)
    mapping_file['Smoking'] = mapping_file['Smoking'].map(mappin_boolean)
    mapping_file = mapping_file[['Age', 'BMI', 'FattyLiver','RegularExercise', 'Smoking', 'DiagnosisGroup']]

    if perform_distance:
        cols = [col for col in preproccessed_data.columns if len(preproccessed_data[col].unique()) !=1]
        dict_bact ={'else':[]}
        for col in preproccessed_data[cols]:
            col_name = preproccessed_data[col].name.split(';')
            bact_level = level-1
            if len(col_name)>bact_level:
                if col_name[bact_level] in dict_bact:
                    dict_bact[col_name[bact_level]].append(preproccessed_data[col].name)
                else:
                    dict_bact[col_name[bact_level]] = [preproccessed_data[col].name]
            else:
                dict_bact['else'].append(preproccessed_data[col].name)
            print(col_name[-1])

        new_df = pd.DataFrame(index = preproccessed_data.index)
        col = 0
        for key, values in dict_bact.items():
            new_data = preproccessed_data[values]
            pca = PCA(n_components=round(new_data.shape[1] / 2)+1)
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
                new_df[col+j] = otu_after_pca_new[j]
            col += num_comp
        return new_df, mapping_file
    else:
        return preproccessed_data, mapping_file
