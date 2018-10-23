from load_merge_otu_mf import OtuMfHandler
from preprocess import preprocess_data
from pca import *
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

otu = 'C:/Users/Anna/Documents/otu_psc.csv'
mapping = 'C:/Users/Anna/Documents/mapping_psc.csv'
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
visualize_pca(preproccessed_data)

otu_after_pca = apply_pca(preproccessed_data, n_components=30)
merged_data = otu_after_pca.join(OtuMf.mapping_file['Diagnosis'])
mapping_disease = {'Cirrhosis ':1, 'HCC':2, 'CirrhosisHCC':2,'PSC':3, 'Control':4}
merged_data['Diagnosis'] = merged_data['Diagnosis'].map(mapping_disease)
merged_data.fillna(0)
print(merged_data.head())


for j in range(1, 20):
    for k in range(2, 9):
        for l in range(5, 500,5):
            auc = []
            auc_train = []
            for i in range(0,20):
                X_train, X_test, y_train, y_test = train_test_split(
                    merged_data.loc[:, merged_data.columns != 'Diagnosis'], merged_data['Diagnosis'],
                    test_size=0.2)
                clf = RandomForestClassifier(max_depth=k, min_samples_leaf=j, n_estimators=l)
                clf.fit(X_train, y_train)
                y_train = np.array(y_train)
                pred_train = np.array(clf.predict(X_train))
                y_train = y_train.astype(int)
                pred_train = pred_train.astype(int)
                fpr1, tpr1, tresholds1 = metrics.roc_curve(y_train, pred_train, pos_label=4)
                auc_train.append(metrics.auc(fpr1, tpr1))
                y = np.array(y_test)
                pred = np.array(clf.predict(X_test))
                y = y.astype(int)
                pred = pred.astype(int)
                fpr, tpr, tresholds = metrics.roc_curve(y, pred, pos_label=4)
                auc.append(metrics.auc(fpr, tpr))
            if (sum(auc) / len(auc) > 0.7):
                print(k, j, l, sum(auc) / len(auc), sum(auc_train) / len(auc_train))
