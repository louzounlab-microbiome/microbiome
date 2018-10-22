from load_merge_otu_mf import OtuMfHandler
from preprocess import preprocess_data
from pca import visualize_pca
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

otu = '/otu_psc.csv'
mapping = '/mapping_psc.csv'
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
#print (OtuMf.otu_file)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
visualize_pca(preproccessed_data)
