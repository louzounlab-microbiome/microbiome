from load_merge_otu_mf import OtuMfHandler
from preprocess import preprocess_data
from pca import *
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

otu = 'C:/Users/Anna/Documents/otu_psc.csv'
mapping = 'C:/Users/Anna/Documents/mapping_psc.csv'
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
#visualize_pca(preproccessed_data)

otu_after_pca = apply_pca(preproccessed_data, n_components=30)
merged_data = otu_after_pca.join(OtuMf.mapping_file['Diagnosis'])
print(merged_data.tail())
X_train, X_test, y_train, y_test = train_test_split( merged_data.loc[:, merged_data.columns != 'Diagnosis'], merged_data['Diagnosis'],
                                                     test_size=0.2)

