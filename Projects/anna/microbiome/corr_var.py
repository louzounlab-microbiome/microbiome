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
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# otu = 'C:/Users/Anna/Desktop/docs/otu_psc2.csv'
# mapping = 'C:/Users/Anna/Desktop/docs/mapping_psc.csv'

# otu = 'C:/Users/Anna/Documents/otu_IBD3.csv'
# mapping = 'C:/Users/Anna/Documents/mapping_IBD3.csv'

otu = 'C:/Users/Anna/Documents/otu_saliva_GVHD.csv'
mapping = 'C:/Users/Anna/Documents/mapping_saliva_GVHD.csv'

OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=7, preform_z_scoring=False, preform_log=True)


mapping_file = OtuMf.mapping_file
mapping_file['DATE'] = pd.to_datetime(OtuMf.mapping_file['DATE'])
mapping_file['Date_Of_Transplantation'] = pd.to_datetime(OtuMf.mapping_file['Date_Of_Transplantation'])
mapping_file['Date_of_engraftmen'] = pd.to_datetime(OtuMf.mapping_file['Date_of_engraftmen'])
mapping_file['aGVHD1_Stat'] = pd.to_datetime(OtuMf.mapping_file['aGVHD1_Stat'])
mapping_file['cGVHD_start'] = pd.to_datetime(OtuMf.mapping_file['cGVHD_start'])
end = pd.to_datetime('2020-01-01')
mapping_file['aGVHD1_Stat'] = mapping_file['aGVHD1_Stat'].fillna(end)
mapping_file['cGVHD_start'] = mapping_file['cGVHD_start'].fillna(end)
mapping_file = mapping_file[(mapping_file['DATE']>mapping_file['Date_Of_Transplantation']) & (mapping_file['DATE']<mapping_file['aGVHD1_Stat']) & (mapping_file['DATE']<mapping_file['cGVHD_start'])].sort_values(['Personal_ID', 'DATE'])

mapping_file = mapping_file.reset_index()
mapping_file = mapping_file.sort_values("DATE").groupby("Personal_ID", as_index=False).last().set_index('#SampleID')
preproccessed_data = preproccessed_data.join(mapping_file[['aGVHD1 ', 'cGVHD ']], how ='inner')
preproccessed_data = preproccessed_data.fillna('No')

mapping_yes_no = {'Yes':1,'No':0}
preproccessed_data['aGVHD1 '] = preproccessed_data['aGVHD1 '].map(mapping_yes_no)
preproccessed_data['cGVHD '] = preproccessed_data['cGVHD '].map(mapping_yes_no)
preproccessed_data["disease"] = preproccessed_data["aGVHD1 "].map(str) + '_' +preproccessed_data["cGVHD "].map(str)
mapping_diseases = {'0_0':1,'1_0':-1,'0_1':-1,'1_1':-1}
preproccessed_data["disease"] = preproccessed_data["disease"].map(mapping_diseases)

X = preproccessed_data.drop(['aGVHD1 ', 'cGVHD ', 'disease'], axis=1)
y =  preproccessed_data['disease']
# preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['CD_or_UC', 'preg_trimester', 'P-ID']], how ='inner')
# preproccessed_data = preproccessed_data.loc[(preproccessed_data['CD_or_UC'] != 'control')]
# preproccessed_data = preproccessed_data.groupby(['CD_or_UC', 'preg_trimester', 'P-ID'], as_index=False).mean()
#
# merged_data = preproccessed_data.fillna(0)
# mapping_disease = {'CD': 1, 'UC': -1}
# merged_data['CD_or_UC'] = merged_data['CD_or_UC'].map(mapping_disease)
# X = merged_data.drop(['CD_or_UC', 'preg_trimester', 'P-ID'], axis=1)
# y = merged_data['CD_or_UC']

# merged_data = preproccessed_data.join(OtuMf.mapping_file['DiagnosisGroup'])
# merged_data.fillna(0)
# mapping_disease_for_labels = {'Control':0,'Cirrhosis/HCC':1, 'PSC/PSC+IBD':2}
# mapping_disease = {'Control':0,'Cirrhosis ':1, 'HCC':1, 'PSC+IBD':2,'PSC':2}
# merged_data['DiagnosisGroup'] = merged_data['DiagnosisGroup'].map(mapping_disease)
# X = merged_data.loc[:, merged_data.columns != 'DiagnosisGroup']
# y = merged_data['DiagnosisGroup']

corr_list =[]
var_list =[]

for i in range(X.shape[1]):
    #if scipy.stats.spearmanr(predicted_data.iloc[:, i], predicted_data['pred'])[1]<0.05:  #/predicted_data.shape[1]:
    corr = scipy.stats.spearmanr(X.iloc[:, i], y)[0]
    var = np.var(X.iloc[:, i], ddof=1)
    corr_list.append(abs(corr))
    var_list.append(var)

def plot_graph(corr_list, var_list):
    plt.scatter(var_list,corr_list, color ='blue')
    plt.xlabel('Varinace')
    plt.ylabel('Abs Corr')
    plt.show()
plot_graph(corr_list, var_list)