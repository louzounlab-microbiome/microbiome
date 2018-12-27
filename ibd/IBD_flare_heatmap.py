from load_merge_otu_mf import OtuMfHandler
from preprocess import preprocess_data
import scipy
from mne.stats import bonferroni_correction, fdr_correction
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
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=5)
preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['CD_or_UC', 'preg_trimester', 'P-ID']], how ='inner')
preproccessed_data = preproccessed_data.groupby(['CD_or_UC', 'preg_trimester', 'P-ID'], as_index=False).mean()
preproccessed_data = preproccessed_data.fillna(0)
#preproccessed_data = preproccessed_data.drop(['P-ID'], axis=1)
#visualize_pca(preproccessed_data)
new_set2=preproccessed_data.groupby(['preg_trimester']).mean()
for i in range(0,len(preproccessed_data)):
    month = preproccessed_data['preg_trimester'][i]
    preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]] =  (preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]].values - new_set2.loc[month:month,:].values)
#otu_after_pca, pca_components = apply_pca(preproccessed_data.drop(['CD_or_UC', 'preg_trimester', 'P-ID'], axis=1), n_components=50)
#merged_data = preproccessed_data.join(OtuMf.mapping_file[['CD_or_UC', 'preg_trimester']], how ='inner')

merged_data = preproccessed_data.join(OtuMf.mapping_file.loc[:,'IL-4_(pg/mL)':], how ='inner')
#merged_data = preproccessed_data.drop(['P-ID'], axis=1)
#print(merged_data.head())
merged_data = merged_data.loc[(merged_data['CD_or_UC'] != 'control')]
merged_data = merged_data.drop(['preg_trimester', 'P-ID'], axis=1)
df_rho = pd.DataFrame(columns = merged_data.columns[-12:], index = merged_data.columns[1:merged_data.shape[1]-12])
df_p = pd.DataFrame(columns = merged_data.columns[-12:], index = merged_data.columns[1:merged_data.shape[1]-12])
for i in range(1,merged_data.shape[1]-12):
    for j in range(merged_data.shape[1]-12,merged_data.shape[1]):
        df_rho.set_value(merged_data.columns[i],merged_data.columns[j], scipy.stats.spearmanr(np.array(merged_data.iloc[:,i][merged_data.iloc[:,j] != 'na']).astype(np.float),np.array(merged_data.iloc[:,j][merged_data.iloc[:,j] != 'na']).astype(np.float))[0])
        df_p.set_value(merged_data.columns[i], merged_data.columns[j],  scipy.stats.spearmanr(np.array(merged_data.iloc[:,i][merged_data.iloc[:,j] != 'na']).astype(np.float),np.array(merged_data.iloc[:,j][merged_data.iloc[:,j] != 'na']).astype(np.float))[1])
df_p = df_p.drop(['k__Bacteria; p__Actinobacteria; c__Actinobacteria; o__Actinomycetales; f__Microbacteriaceae'],axis=0)
df_rho = df_rho.drop(['k__Bacteria; p__Actinobacteria; c__Actinobacteria; o__Actinomycetales; f__Microbacteriaceae'],axis=0)
for i in range(df_p.shape[1]):
    reject_fdr, pval_fdr = fdr_correction(df_p.iloc[:, i].values, alpha=0.05, method='indep')
    for j in range(len(reject_fdr)):
        if  reject_fdr[j] == True:
            print (df_p.columns[i],df_p.index[j], df_p.iloc[j,i], df_rho.iloc[j,i])
print('done')

