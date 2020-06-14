from load_merge_otu_mf import OtuMfHandler
from Preprocess import preprocess_data
from pca import *
import scipy
from plot_confusion_matrix import *
import pandas as pd
import math
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import math
import seaborn as sns; sns.set(color_codes=True)
import operator
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

otu = 'C:/Users/Anna/Desktop/docs/otu_psc2.csv'
mapping = 'C:/Users/Anna/Desktop/docs/mapping_psc.csv'
max_num_of_pcas = 35
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=7)
mapping_file = OtuMf.mapping_file

mapping_disease = {'Control':0,'Cirrhosis ':1, 'HCC':1, 'PSC+IBD':2,'PSC':2}
mapping_file['DiagnosisGroup'] = mapping_file['DiagnosisGroup'].map(mapping_disease)
mappin_boolean = {'yes' :1, 'no': 0, 'Control': 0, '0':0, '1':1}
mapping_file['FattyLiver'] = mapping_file['FattyLiver'].map(mappin_boolean)
mapping_file['RegularExercise'] = mapping_file['RegularExercise'].map(mappin_boolean)
mapping_file['Smoking'] = mapping_file['Smoking'].map(mappin_boolean)

cols = [col for col in preproccessed_data.columns if len(preproccessed_data[col].unique()) !=1]
dict_bact ={'else':[]}
for col in preproccessed_data[cols]:
    col_name = preproccessed_data[col].name.split(';')
    # if 'c__' in col_name[-1]:
    #     if  col_name[-1] in dict_bact:
    #         dict_bact[col_name[-1]].append(preproccessed_data[col].name)
    #     else:
    #         dict_bact[col_name[-1]] = [preproccessed_data[col].name]
    # else:
    #     dict_bact['else'].append(preproccessed_data[col].name)
    if len(col_name)>2:
        if col_name[2] in dict_bact:
            dict_bact[col_name[2]].append(preproccessed_data[col].name)
        else:
            dict_bact[col_name[2]] = [preproccessed_data[col].name]
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


#visualize_pca(new_df)
pcas =[]
train_accuracy = []
test_accuracy = []
for n_comp in range(1, max_num_of_pcas):
    pcas.append(n_comp)
    otu_ap, _ = apply_pca(new_df, n_components=n_comp)
    new_df2 = otu_ap.join(mapping_file[['Age', 'BMI', 'FattyLiver','RegularExercise', 'Smoking', 'DiagnosisGroup']], how='inner')
    new_df2 = new_df2.fillna(0)

    X= new_df2.drop(['DiagnosisGroup'], axis =1)
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns.values]

    y = new_df2['DiagnosisGroup']

    loo = LeaveOneOut()
    y_pred_list = []
    auc = []
    auc_train = []
    for train_index, test_index in loo.split(X):
        train_index = list(train_index)
        # print("%s %s" % (train_index, test_index))
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model = XGBClassifier(max_depth=4, n_estimators=150, learning_rate=15 / 100,
                              objective='multi:softmax',  reg_lambda=150
                              #objective='binary:logistic',
                              #scale_pos_weight=(np.sum(y_train == -1) / np.sum(y_train == 1)),
                              )
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        auc_train.append(metrics.accuracy_score(y_train, pred_train))
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred[0])

    auc = metrics.accuracy_score(y, y_pred_list)
    scores = round(auc, 2)
    scores_train = round(np.array(auc_train).mean(), 2)
    train_accuracy.append(scores_train)
    test_accuracy.append(round(scores.mean(), 2))

train_accuracy_all = []
test_accuracy_all = []
def pca_graph(max_num_of_pcas = max_num_of_pcas):
    for i in range (1,max_num_of_pcas):
        otu_after_pca, _ = apply_pca(preproccessed_data, n_components=i)
        merged_data = otu_after_pca.join(mapping_file[['Age', 'BMI', 'FattyLiver','RegularExercise', 'Smoking','DiagnosisGroup']])

        merged_data.fillna(0)

        # mapping_disease = {'Control':0,'Cirrhosis ':1, 'HCC':1, 'PSC+IBD':2,'PSC':2}
        # merged_data['DiagnosisGroup'] = merged_data['DiagnosisGroup'].map(mapping_disease)
        # merged_data = merged_data.join(OtuMf.mapping_file[['Age', 'BMI', 'FattyLiver','RegularExercise', 'Smoking']])
        # mappin_boolean = {'yes' :1, 'no': 0, 'Control': 0, '0':0, '1':1}
        # merged_data['FattyLiver'] = merged_data['FattyLiver'].map(mappin_boolean)
        # merged_data['RegularExercise'] = merged_data['RegularExercise'].map(mappin_boolean)
        # merged_data['Smoking'] = merged_data['Smoking'].map(mappin_boolean)

        X = merged_data.loc[:, merged_data.columns != 'DiagnosisGroup']
        y = merged_data['DiagnosisGroup']

        loo = LeaveOneOut()
        y_pred_list = []
        x_indx = []
        y_pred_train =[]
        for train_index, test_index in loo.split(X):
            train_index = list(train_index)
            # print("%s %s" % (train_index, test_index))
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            model = XGBClassifier(max_depth=4, n_estimators=150, learning_rate=15/ 100,
                                  objective='multi:softmax',   reg_lambda=150)#,  reg_lambda=550)
            # # #                                           #objective= 'binary:logistic')
            model.fit(X_train, y_train)
            x_indx.append(X_test.index[0])
            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred[0])
            y_pre_tr = model.predict(X_train)
            accuracy_train = metrics.accuracy_score(y_train,y_pre_tr)
            y_pred_train.append(accuracy_train)
        scores = np.array(metrics.accuracy_score(y, y_pred_list))
        scores_train = round(np.array(y_pred_train).mean(), 2)
        train_accuracy_all.append(scores_train)
        test_accuracy_all.append(round(scores.mean(), 2))
pca_graph(max_num_of_pcas = max_num_of_pcas)
def plot_graph(test_accuracy, train_accuracy, train_accuracy_all,  test_accuracy_all, pcas):
    plt.plot(pcas,test_accuracy, color ='red', label ='test_fs')
    plt.plot(pcas,train_accuracy, color='blue', label ='train_fs')
    plt.plot(pcas, test_accuracy_all, color='orange', label='test')
    plt.plot(pcas, train_accuracy_all, color='black', label='train')
    plt.legend( loc=1,ncol=1)
    plt.show()
plot_graph(test_accuracy,train_accuracy,train_accuracy_all,  test_accuracy_all, pcas)

print('done')
