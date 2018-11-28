from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
from infra_functions.generate_N_colors import getDistinctColors, rgb2hex
from infra_functions.general import apply_pca, use_spearmanr, use_pearsonr
from infra_functions.fit import fit_SVR, fit_random_forest
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from scipy.stats import pearsonr
import numpy as np
import pickle
from sklearn import svm
# from sklearn.svm import SV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import datetime
from gvhd.show_data import calc_results
from gvhd.calculate_distances import calculate_distance
from gvhd.cluster_time_events import cluster_based_on_time





n_components = 20
OtuMf = OtuMfHandler('otu_IBD_table.csv', 'metadata_ok94_ok59.csv', from_QIIME=True)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=True, taxnomy_level=5, preform_taxnomy_group=False)
otu_after_pca_wo_taxonomy, _ = apply_pca(preproccessed_data, n_components=n_components, visualize=False)
# otu_after_pca = OtuMf.add_taxonomy_col_to_new_otu_data(otu_after_pca_wo_taxonomy)
# merged_data_after_pca = OtuMf.merge_mf_with_new_otu_data(otu_after_pca_wo_taxonomy)
# merged_data_with_age = otu_after_pca_wo_taxonomy.join(OtuMf.mapping_file['age_in_days'])
# merged_data_with_age = merged_data_with_age[merged_data_with_age.age_in_days.notnull()] # remove NaN days
# merged_data_with_age_group = otu_after_pca_wo_taxonomy.join(OtuMf.mapping_file[['age_group', 'age_in_days','MouseNumber']])
# merged_data_with_age_group = merged_data_with_age_group[merged_data_with_age_group.age_group.notnull()] # remove NaN days