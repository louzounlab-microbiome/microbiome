from load_merge_otu_mf import OtuMfHandler
from preprocess import preprocess_data
from pca import *
from plot_confusion_matrix import *
from plot_roc_curve import *
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBClassifier

otu = 'C:/Users/Anna/Documents/otu_IBD2.csv'
mapping = 'C:/Users/Anna/Documents/mapping_IBD2.csv'
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
#print(OtuMf.otu_file.shape)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
#print(preproccessed_data.shape)
#visualize_pca(preproccessed_data)
preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['CD_or_UC', 'preg_trimester', 'P-ID']], how ='inner')
preproccessed_data = preproccessed_data.groupby(['CD_or_UC', 'preg_trimester', 'P-ID'], as_index=False).mean()
#preproccessed_data = preproccessed_data.drop(['CD_or_UC', 'preg_trimester', 'P-ID'], axis=1)
#visualize_pca(preproccessed_data)
new_set2=preproccessed_data.groupby(['preg_trimester']).mean()
for i in range(0,len(preproccessed_data)):
    month = preproccessed_data['preg_trimester'][i]
    preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]] =  (preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]].values - new_set2.loc[month:month,:].values)
otu_after_pca = apply_pca(preproccessed_data.drop(['CD_or_UC', 'preg_trimester', 'P-ID'], axis=1), n_components=40)
merged_data = otu_after_pca.join(preproccessed_data[['CD_or_UC', 'preg_trimester']], how ='inner')
merged_data.fillna(0)
print(merged_data.head())

#merged_data = merged_data.loc[merged_data['preg_trimester'] == 't_3']
# font = {'size': 22}
# max_pca = 5
# k =1
# for i in range(max_pca):
#      for j in range(max_pca):
#         if j>i:
#             plt.subplot(2, 5, k)
#             plt.scatter(x=merged_data[i][(merged_data['CD_or_UC']=='control')],y=merged_data[j][(merged_data['CD_or_UC']=='control')],
#                    marker='.', color='darkturquoise', label = 'control', lw=0,s=10**2)
#             plt.scatter(x=merged_data[i][(merged_data['CD_or_UC']=='CD')],y=merged_data[j][(merged_data['CD_or_UC']=='CD')],
#                    marker='.', color='red', label = 'CD', lw=0,s=10**2)
#             plt.scatter(x=merged_data[i][(merged_data['CD_or_UC'] == 'UC')],y=merged_data[j][(merged_data['CD_or_UC'] == 'UC')],
#                      marker='.', color='orange', label='UC', lw=0,s=10**2)
#             plt.ylabel('PCA %s' %j)
#             plt.xlabel('PCA %s' %i)
#             plt.grid(True)
#             k+=1
# plt.title('mean reduced')
# plt.legend( loc=1,ncol=1)
# plt.show()

mapping_disease_for_labels = {'Control':0,'CD':1,'UC':2}
mapping_disease = {'control':0,'CD':1,'UC':2}
merged_data['CD_or_UC'] = merged_data['CD_or_UC'].map(mapping_disease)

X = merged_data.drop(['CD_or_UC', 'preg_trimester'], axis=1)

y = merged_data['CD_or_UC']

loo = LeaveOneOut()

for md in range(3,6):
      for ne in range (100,350,50):
           for lr in range (5, 20, 5):
               accuracy = []
               y_pred_list = []
               for train_index, test_index in loo.split(X):
                    train_index=list(train_index)
                    #print("%s %s" % (train_index, test_index))
                    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
                    y_train, y_test = y[train_index], y[test_index]
                    model = XGBClassifier(max_depth=md,n_estimators = ne ,learning_rate = lr/100,  objective='multi:softmax' )
# # #                                           #objective= 'binary:logistic')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
# # #                     #pred_train = model.predict_proba(X_train)[:, 1]
# # #                     #auc_train.append(metrics.roc_auc_score(y_train, pred_train))
# # #                     #y_pred = model.predict_proba(X_test)[:,1]
# # #                     #try:
# # #                     #    auc.append(metrics.roc_auc_score(y_test, y_pred))
# # #                     #except:
# # #                     #    continue
                    y_pred_list.append(y_pred)
               cnf_matrix = metrics.confusion_matrix(y, y_pred_list)
               class_names = mapping_disease_for_labels.keys()
               # # Plot non-normalized confusion matrix
               plt.figure()
               plot_confusion_matrix(cnf_matrix, classes=class_names,
                                     title='Confusion matrix, without normalization')

               # # Plot normalized confusion matrix
               plt.figure()
               plot_confusion_matrix(cnf_matrix, classes=list(class_names), normalize=True,
                                     title='Normalized confusion matrix')

               plt.show()
               scores = np.array(metrics.accuracy_score(y, y_pred_list))
               print(md, ne, lr, round(scores.mean(), 2), round(scores.std(), 2) * 2)
