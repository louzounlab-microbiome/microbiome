from load_merge_otu_mf import OtuMfHandler
from preprocess import preprocess_data
from pca import *
import pandas as pd

otu = 'allergy_otu.csv'
mapping = 'allergy_mf.csv'
## preprocessing
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=7)
preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['AllergyType', 'SuccessDescription']],
                                             how='inner')
preproccessed_data = preproccessed_data.loc[(preproccessed_data['AllergyType'] == 'Milk') | ((preproccessed_data['AllergyType'] == 'Peanut'))]
preproccessed_data = preproccessed_data.drop(['AllergyType', 'SuccessDescription'], axis =1)
mapping_file = OtuMf.mapping_file.loc[(OtuMf.mapping_file['AllergyType']  == 'Milk') | (OtuMf.mapping_file['AllergyType']  == 'Peanut')]
mapping_disease = {'Milk': 1, 'Peanut': 0}
mapping_file['AllergyType'] = mapping_file['AllergyType'].map(mapping_disease)
mapping_file = mapping_file['AllergyType']

## distance learning
def distance_learning(perform_distance=False,level =3,preproccessed_data= preproccessed_data, mappping_file = mapping_file):
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
        for key, values in dict_bact.items():
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
            col += num_comp
        return new_df, mapping_file
    else:
        return preproccessed_data, mapping_file
    #print('done')
