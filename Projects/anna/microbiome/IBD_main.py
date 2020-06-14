from load_merge_otu_mf import OtuMfHandler
from Preprocess import preprocess_data
from pca import *
from plot_confusion_matrix import *
from plot_roc_curve import *
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from plot_clustergram import *

otu = 'C:/Users/Anna/Documents/otu_IBD3.csv'
mapping = 'C:/Users/Anna/Documents/mapping_IBD3.csv'
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
#print(OtuMf.otu_file.shape)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['CD_or_UC', 'preg_trimester', 'P-ID']], how ='inner')
preproccessed_data = preproccessed_data.loc[(preproccessed_data['CD_or_UC'] != 'control')]
preproccessed_data = preproccessed_data.groupby(['CD_or_UC', 'preg_trimester', 'P-ID'], as_index=False).mean()
#preproccessed_data = preproccessed_data.drop(['CD_or_UC', 'preg_trimester', 'P-ID'], axis=1)
#visualize_pca(preproccessed_data)
new_set2=preproccessed_data.groupby(['preg_trimester']).mean()
for i in range(0,len(preproccessed_data)):
    month = preproccessed_data['preg_trimester'][i]
    preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]] =  (preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]].values - new_set2.loc[month:month,:].values)
otu_after_pca, pca_components = apply_pca(preproccessed_data.drop(['CD_or_UC', 'preg_trimester', 'P-ID'], axis=1), n_components=19)
merged_data = otu_after_pca.join(preproccessed_data[['CD_or_UC', 'preg_trimester']], how ='inner')
#merged_data = preproccessed_data.drop(['P-ID'], axis=1)
merged_data = merged_data.fillna(0)
#print(merged_data.head())
#merged_data = merged_data.loc[(merged_data['CD_or_UC'] != 'control')]
# merged_data= merged_data.reset_index()
# try:
#     merged_data=merged_data.drop('index',axis=1)
# except:
#     pass
#merged_data = merged_data.loc[merged_data['preg_trimester'] == 't_3']

#plot_clustergram(merged_data, ['CD_or_UC'])
#plot mean and std per time point
mapping_disease_for_labels = {'CD':-1,'UC':1}
#mapping_disease = {'control':1,'CD':-1,'UC':1}
mapping_disease = {'CD':-1,'UC':1}

merged_data['CD_or_UC'] = merged_data['CD_or_UC'].map(mapping_disease)

# new_test=merged_data.groupby(['CD_or_UC', 'preg_trimester']).agg([ np.mean, np.std, 'count' ]).reset_index('preg_trimester')
#
# health = plt.scatter(x=new_test['preg_trimester'][(new_test.index.values==0)], y=new_test[0]['mean'][(new_test.index.values==0)], marker='_', color='palevioletred')
# cd = plt.scatter(x=new_test['preg_trimester'][(new_test.index.values==1)], y=new_test[0]['mean'][(new_test.index.values==1)], marker='_', color='darkturquoise')
# uc = plt.scatter(x=new_test['preg_trimester'][(new_test.index.values==2)], y=new_test[0]['mean'][(new_test.index.values==2)], marker='_', color='red')
# plt.errorbar(new_test['preg_trimester'][(new_test.index.values==0)],new_test[0]['mean'][(new_test.index.values==0)], new_test[0]['std'][(new_test.index.values==0)]/np.sqrt(new_test[0]['count'][(new_test.index.values==0)]), c='palevioletred', linestyle= '-.')
# plt.errorbar(new_test['preg_trimester'][(new_test.index.values==1)],new_test[0]['mean'][(new_test.index.values==1)], new_test[0]['std'][(new_test.index.values==1)]/np.sqrt(new_test[0]['count'][(new_test.index.values==1)]), c='darkturquoise', linestyle= '-.')
# plt.errorbar(new_test['preg_trimester'][(new_test.index.values==2)],new_test[0]['mean'][(new_test.index.values==2)], new_test[0]['std'][(new_test.index.values==2)]/np.sqrt(new_test[0]['count'][(new_test.index.values==2)]), c='red', linestyle= '-.')
# #plt.ylim(-4,4)
# #plt.xlim(0,25)
# plt.legend((health,cd,uc),
#            ('health', 'cd','uc'),
#            scatterpoints=1,
#            loc='upper right',
#            fontsize=9)
# plt.xticks(np.arange(0, 3, 1))
# plt.ylabel('PCA 1')
# plt.xlabel('Time')
# #plt.title('Vaginal mode of birth ')
# plt.show()

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
# #plt.title('mean reduced')
# plt.legend( loc=1,ncol=1)
# plt.show()
merged_data= merged_data.reset_index()
try:
    merged_data=merged_data.drop('index',axis=1)
except:
    pass

X = merged_data.drop(['CD_or_UC', 'preg_trimester'], axis=1)

y = merged_data['CD_or_UC']

loo = LeaveOneOut()

for md in range(3,7):
      for ne in range (100,300,50):
           for lr in range (5, 20, 5):
               accuracy = []
               y_pred_list = []
               auc = []
               auc_train = []
               for train_index, test_index in loo.split(X):
                   train_index = list(train_index)
                   # print("%s %s" % (train_index, test_index))
                   X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                   y_train, y_test = y[train_index], y[test_index]
                   model = XGBClassifier(max_depth=md, n_estimators=ne, learning_rate=lr / 100,
                                         # objective='multi:softmax' )
                                         objective='binary:logistic',
                                         scale_pos_weight=(np.sum(y_train == -1) / np.sum(y_train == 1)),
                                         reg_lambda=250)
                   model.fit(X_train, y_train)
                   pred_train = model.predict_proba(X_train)[:, 1]
                   auc_train.append(metrics.roc_auc_score(y_train, pred_train))
                   y_pred = model.predict_proba(X_test)[:, 1]
                   y_pred_list.append(y_pred[0])
               try:
                   auc = metrics.roc_auc_score(y, y_pred_list)
               except:
                   pass
               print('PCA components' + str(50), round(auc, 2))
               scores = round(auc, 2)
               scores_train = round(np.array(auc_train).mean(), 2)
               # cnf_matrix = metrics.confusion_matrix(y, y_pred_list)
               # class_names = mapping_disease_for_labels.keys()
               # # # Plot non-normalized confusion matrix
               # plt.figure()
               # plot_confusion_matrix(cnf_matrix, classes=class_names,
               #                       title='Confusion matrix, without normalization')
               #
               # # # Plot normalized confusion matrix
               # plt.figure()
               # plot_confusion_matrix(cnf_matrix, classes=list(class_names), normalize=True,
               #                       title='Normalized confusion matrix')
               #
               # plt.show()
               #print('PCA components' + str(50), round(auc, 2))
               print(md, ne, lr, scores, scores_train)
               #fpr, tpr, threshold = metrics.roc_curve(y, y_pred_list)
               #plot_roc_curve(fpr,tpr)
               #print(md, ne, lr, round(scores.mean(), 2), round(scores.std(), 2) * 2)


#SVM

#clf = svm.SVC(kernel='linear', probability = True)
# clf =  LogisticRegression(penalty='l1', solver='liblinear', C= 0.55 , class_weight = {-1:.25, 1:.75})
# auc = []
# y_pred1 =[]
# y_test1 = []
# y_pred_train =[]
# for train_index, test_index in loo.split(X):
#     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
#     y_train, y_test = y[train_index], y[test_index]
#     clf.fit(X_train, y_train)
#     y_pred_train = clf.predict_proba(X_train)[:,1]
#     y_pred = clf.predict_proba(X_test)[:,1]
#     y_pred1.append(y_pred)
#     y_test1.append(y_test.values[0])
#     print(metrics.roc_auc_score(y_train,y_pred_train))
# auc = metrics.roc_auc_score(y_test1, y_pred1)
# print(auc)
# W = clf.coef_[0]
# print(W)
# try:
#     df = pd.DataFrame(
#         {'Taxonomy': preproccessed_data.columns[3:],
#         'Coefficients': np.dot(clf.coef_[0],pca_components)
#         })
# except:
#     df = pd.DataFrame(
#         {'Taxonomy': preproccessed_data.columns[3:],
#          'Coefficients':clf.coef_[0]
#          })
# df['Coefficients'] = df['Coefficients'] /np.linalg.norm(df['Coefficients'].values)
# df1 = df.round({'Coefficients': 4})
# df1.to_csv('C:/Users/Anna/Documents/bacteria_results_sick_t3.csv')
#     #
#     #decision_boundary =  sum (W[i]*x[i])+ intercept_[0] = 0
#
# fpr, tpr, threshold = metrics.roc_curve(y_test1, y_pred1)
# plot_roc_curve(fpr,tpr)
