from sklearn.decomposition import PCA
import pandas as pd
from Preprocess.general import apply_pca


## distance learning
def distance_learning(perform_distance, level, preproccessed_data, mapping_file):
    if perform_distance:
        cols = [col for col in preproccessed_data.columns if preproccessed_data[col].nunique() != 1]
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
            #print(dict_bact)

        new_df = pd.DataFrame(index=preproccessed_data.index)
        col = 0
        for key, values in dict_bact.items():
            if values:
                new_data = preproccessed_data[values]
                pca = PCA(n_components=min(round(new_data.shape[1] / 2) + 1, new_data.shape[0]))
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
                # new
                otu_after_pca_new, pca_obj, pca_str = apply_pca(new_data, n_components=num_comp)
                # old
                # otu_after_pca_new, pca_components = apply_pca(new_data, n_components=num_comp)
                for j in range(otu_after_pca_new.shape[1]):
                    if key == 'else':
                        new_df['else;'] = otu_after_pca_new[j]
                    else:
                        new_df[str(values[0][0:values[0].find(key)+len(key)])+'_'+str(j)] = otu_after_pca_new[j]
                col += num_comp
        return new_df, mapping_file
    else:
        return preproccessed_data, mapping_file
    #print('done')