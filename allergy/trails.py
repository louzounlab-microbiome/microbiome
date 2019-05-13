from sklearn.model_selection import GridSearchCV, train_test_split
from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
from infra_functions.general import apply_pca, use_spearmanr, use_pearsonr  # sigmoid

import os
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
from sklearn import xgb
import math
from infra_functions.general import apply_pca, use_spearmanr, use_pearsonr, roc_auc, convert_pca_back_orig, draw_horizontal_bar_chart  # sigmoid
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
n_components = 20

# Define a function for a plot with two y axes
def lineplot2y(x_data, x_label, y1_data, y1_color, y1_label, y2_data, y2_color, y2_label, title):
    # Each variable will actually have its own plot object but they
    # will be displayed in just one plot
    # Create the first plot object and draw the line
    _, ax1 = plt.subplots()
    ax1.plot(x_data, y1_data, color=y1_color)
    # Label axes
    ax1.set_ylabel(y1_label, color=y1_color)
    ax1.set_xlabel(x_label)
    ax1.set_title(title)

    # Create the second plot object, telling matplotlib that the two
    # objects have the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(x_data, y2_data, color=y2_color)
    ax2.set_ylabel(y2_label, color=y2_color)
    # Show right frame line
    ax2.spines['right'].set_visible(True)
    _.show()
    _.savefig(title)

# Pre process
PRINT = False
bactria_as_feature_file = 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'
features = pd.read_csv(bactria_as_feature_file, header=1)
cols = list(features.columns)
# remove non-numeric values
cols.remove('Feature ID')
features_transformed = features[cols].T
features_transformed.columns = features_transformed.iloc[0]
cols = list(features_transformed.columns)
# cols.remove('Taxonomy')

means = [features_transformed[cols][col].mean() for col in cols]
error = [features_transformed[cols][col].std(ddof=0) for col in cols]

features_no_id = features_transformed[cols].apply(zscore)
normal_means = [features_no_id[col].mean() for col in cols]
normal_error = [features_no_id[col].std(ddof=0) for col in cols]

if PRINT:
    for col in cols:
        print(col + " mean=" + str(features_no_id[col].mean()) + " std=" + str(features_no_id[col].std(ddof=0)))
    print("total z_score mean=" + str(features_no_id.mean()) + " std=" + str(features_no_id.std(ddof=0)))
    # print(features)
# bacterias = features['Taxonomy']
#x_pos = np.arange(len(bacterias))



# Call the function to create plot
lineplot2y(x_data=cols # bacterias
           , x_label='Bacteria'
           , y1_data=means
           , y1_color='#539caf'
           , y1_label='Means'
           , y2_data=normal_means
           , y2_color='#7663b0'
           , y2_label='Normalized means'
           , title='before_and_after_z_score_per_person_bar_plot')

#


samples_data_file = 'mf_merge_ok84_ok93_ok66_69_merged_by_RestoredSampleCode_as_ID_290119.csv'
OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, bactria_as_feature_file),
                     os.path.join(SCRIPT_DIR, samples_data_file),
                     from_QIIME=True, id_col='Feature ID', taxonomy_col='Taxonomy')

preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=5, taxonomy_col='Taxonomy',
                                     preform_taxnomy_group=True)
otu_after_pca_wo_taxonomy, _, _ = apply_pca(preproccessed_data, n_components=n_components, visualize=False)


# Remove control group == 'Con'
tag_column = 'SuccessDescription'
X = otu_after_pca_wo_taxonomy.index.tolist()
y = []
for sample in X:
    if not sample.startswith('Con'):
        y.append(OtuMf.mapping_file.loc[sample, tag_column])

id_to_tag_map = {}
index_to_id_map = {}
id_to_features_map = {}
features = []
for i, row in enumerate(otu_after_pca_wo_taxonomy.values):
    if not otu_after_pca_wo_taxonomy.index[i].startswith('Con'):
        features.append(row)
        id_to_features_map[otu_after_pca_wo_taxonomy.index[i]] = row

# change tags from k-classes to success(A1)->1 and failure(the rest)->0
id_to_binary_tag_map = {}
for i, sample in enumerate(y):
    id_to_tag_map[X[i]] = sample
    if sample == 'A1':
        id_to_binary_tag_map[X[i]] = 1
    else:
        id_to_binary_tag_map[X[i]] = 0

ids = list(id_to_features_map.keys())  # id for each sample
X = list(id_to_features_map.values())  # features for each sample
y = list(id_to_binary_tag_map.get(id) for id in id_to_features_map.keys())  # binary tag for each sample

# remove 'None' tags samples
none_idx = []
none_idx = [i for i, tag in enumerate(y) if tag is None]
for i in reversed(none_idx):
    ids.pop(i)
    X.pop(i)
    y.pop(i)


# Learning
# Split the data set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# xgb
# {'C': 1000, 'gamma': 0.01, 'kernel': 'sigmoid}
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                     'gamma': [1e-3, 1e-4, 0.001, 0.01, 0.1],
                     'C': [0.01, 0.1, 1, 10, 100, 1000]}]
# tuned_parameters = [{'kernel': ['linear'], 'C': [10]}]

xgb_clf = GridSearchCV(xgb.SVC(class_weight='balanced'), tuned_parameters, cv=5,
                       scoring='roc_auc', return_train_score=True)

xgb_clf.fit(X, y)
print(xgb_clf.best_params_)
print(xgb_clf.best_score_)

means_test = xgb_clf.cv_results_['mean_test_score']
stds_test = xgb_clf.cv_results_['std_test_score']
means_train = xgb_clf.cv_results_['mean_train_score']
stds_train = xgb_clf.cv_results_['std_train_score']

"""
xgb_conf_stats = ''
for train_mean, train_std, test_mean, test_std, params in zip(means_train, stds_train, means_test, stds_test,
                                                              xgb_clf.cv_results_['params']):
    xgb_conf_stats += ("Train: %0.3f (+/-%0.03f) , Test: %0.3f (+/-%0.03f) for %r \n" % (
    train_mean, train_std * 2, test_mean, test_std * 2, params))

entire_W = xgb_clf.best_estimator_.coef_[0]
W_pca = entire_W[starting_col:starting_col + n_components]
bacteria_coeff = convert_pca_back_orig(pca_obj.components_, W_pca, original_names=preproccessed_data.columns[:],
                                       visualize=True)
draw_horizontal_bar_chart(entire_W[0:starting_col], interesting_cols, title='Feature Coeff', ylabel='Feature',
                          xlabel='Coeff Value', left_padding=0.3)
# y_true, y_pred = y_test, xgb_clf.predict(X_test)
# # xgb_class_report = classification_report(y_true, y_pred)
# _, _, _, xgb_roc_auc = roc_auc(y_true, y_pred, verbose=True, visualize=False,
#         graph_title='xgb\n' + permutation_str)
"""


# remove 'None' tags samples
#none_idx = []
#none_idx = [i for i, tag in enumerate(y) if tag is None]
#for i in reversed(none_idx):
    #x.pop(i)
    #index_list.pop(i)
    #y.pop(i)
    
