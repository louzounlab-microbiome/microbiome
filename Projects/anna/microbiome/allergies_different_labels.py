from load_merge_otu_mf import OtuMfHandler
from Preprocess import preprocess_data
from pca import *
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut
import pandas as pd
import re
import matplotlib.pyplot as plt
import csv
import numpy as np
from xgboost import XGBClassifier
from sklearn import metrics

def allergies(perform_distance=False,level =3):
    otu = 'allergy_otu.csv'
    mapping = 'allergy_mf.csv'
    OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
    preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=7)
    preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['AllergyType', 'SuccessDescription']],
                                                 how='inner')
    #preproccessed_data = preproccessed_data.loc[(preproccessed_data['AllergyType'] == 'Milk') | ((preproccessed_data['AllergyType'] == 'Peanut'))]
    preproccessed_data = preproccessed_data.drop(['AllergyType', 'SuccessDescription'], axis =1)
    #mapping_file = OtuMf.mapping_file.loc[(OtuMf.mapping_file['AllergyType']  == 'Milk') | (OtuMf.mapping_file['AllergyType']  == 'Peanut')]
    mapping_file = OtuMf.mapping_file
    mapping_disease = {'Milk': 1, 'Peanut': 0}
    mapping_health = {'Con': 1}
    mapping_success = {'A1': 1}
    mapping_file['Health'] = mapping_file['AllergyType'].map(mapping_health)
    mapping_file['AllergyType'] = mapping_file['AllergyType'].map(mapping_disease)
    mapping_file['SuccessDescription'] = mapping_file['SuccessDescription'].map(mapping_success)
    mapping_file[['Health', 'SuccessDescription']] = mapping_file[['Health', 'SuccessDescription']].fillna(value=0)

    mapping_file = mapping_file[['AllergyType', 'SuccessDescription']]

    # if perform_distance:
    #     cols = [col for col in preproccessed_data.columns if len(preproccessed_data[col].unique()) != 1]
    #     dict_bact = {'else': []}
    #     for col in preproccessed_data[cols]:
    #         col_name = preproccessed_data[col].name.split(';')
    #         bact_level = level - 1
    #         if len(col_name) > bact_level:
    #             if col_name[bact_level] in dict_bact:
    #                 dict_bact[col_name[bact_level]].append(preproccessed_data[col].name)
    #             else:
    #                 dict_bact[col_name[bact_level]] = [preproccessed_data[col].name]
    #         else:
    #             dict_bact['else'].append(preproccessed_data[col].name)
    #         print(col_name[-1])
    #
    #     new_df = pd.DataFrame(index=preproccessed_data.index)
    #     col = 0
    #     for key, values in dict_bact.items():
    #         new_data = preproccessed_data[values]
    #         pca = PCA(n_components=round(new_data.shape[1] / 2) + 1)
    #         pca.fit(new_data)
    #         sum = 0
    #         num_comp = 0
    #         for (i, component) in enumerate(pca.explained_variance_ratio_):
    #             if sum <= 0.5:
    #                 sum += component
    #             else:
    #                 num_comp = i
    #                 break
    #         if num_comp == 0:
    #             num_comp += 1
    #         otu_after_pca_new, pca_components = apply_pca(new_data, n_components=num_comp)
    #         for j in range(otu_after_pca_new.shape[1]):
    #             new_df[col + j] = otu_after_pca_new[j]
    #         col += num_comp
    #     return new_df, mapping_file
    # else:
    return preproccessed_data, mapping_file
    #print('done')
allergies()
df, mapping_file = allergies(perform_distance=False,level =3)
df = df.join(mapping_file, how='inner')

df = df.loc[(df['AllergyType'] == 1) | (df['AllergyType'] == 0)]
df = df.drop(['AllergyType', 'SuccessDescription'], axis =1)
mapping_file = mapping_file.loc[(mapping_file['AllergyType']  == 1) | (mapping_file['AllergyType']  == 0)]

#visualize_pca(df)
otu_after_pca, _ = apply_pca(df, n_components=21)
merged_data = otu_after_pca.join(mapping_file)

X = merged_data.drop(['AllergyType', 'SuccessDescription'], axis=1)
y = merged_data['SuccessDescription']

loo = LeaveOneOut()

# csvfile = 'C:/Users/Anna/Documents/allergies_xgboost.csv'
# headers = ['ms', 'ne', 'learning rate', 'regularization', 'auc test', 'auc train']
# with open(csvfile, "w") as output:
#     writer = csv.writer(output, delimiter=',', lineterminator='\n')
#     writer.writerow(headers)
for md in range(3,5):
      for ne in range (100,350,50):
           for lr in range (15, 20, 5):
               for rg in range(250,800,50):
                    accuracy = []
                    y_pred_list = []
                    auc = []
                    auc_train = []
                    for train_index, test_index in loo.split(X):
                        train_index=list(train_index)
                        #print("%s %s" % (train_index, test_index))
                        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
                        y_train, y_test = y[train_index], y[test_index]
                        model = XGBClassifier(max_depth=md,n_estimators = ne ,learning_rate = lr/100,  #objective='multi:softmax' )
                                              objective= 'binary:logistic',
                                              reg_lambda = rg)
                        model.fit(X_train, y_train)
                        #y_pred = model.predict(X_test)
                        pred_train = model.predict_proba(X_train)[:, 1]
                        auc_train.append(metrics.roc_auc_score(y_train, pred_train))
                        y_pred = model.predict_proba(X_test)[:,1]
                        y_pred_list.append(y_pred)
                    try:
                        auc = metrics.roc_auc_score(y, y_pred_list)
                    except:
                        pass
                    #with open(csvfile, "a") as output:
                    #     writer = csv.writer(output, delimiter=',', lineterminator='\n')
                    #     writer.writerow([md, ne, lr, rg, round(auc, 2), round(sum(auc_train) / len(auc_train), 2)])
                    print(md, ne, lr, rg, round(auc, 2), round(sum(auc_train) / len(auc_train), 2))
