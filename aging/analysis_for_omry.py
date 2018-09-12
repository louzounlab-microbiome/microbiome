from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
from infra_functions.general import apply_pca, use_spearmanr
from infra_functions.fit import fit_SVR, fit_random_forest
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

import numpy as np

from sklearn import svm
# from sklearn.svm import SV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

OtuMf = OtuMfHandler('aging_otu_table.csv', 'mf.csv', from_QIIME=True)
# preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=5)
# otu_after_pca_wo_taxonomy, _ = apply_pca(preproccessed_data, n_components=60)
# # otu_after_pca = OtuMf.add_taxonomy_col_to_new_otu_data(otu_after_pca_wo_taxonomy)
# # merged_data_after_pca = OtuMf.merge_mf_with_new_otu_data(otu_after_pca_wo_taxonomy)
# # merged_data_with_age = otu_after_pca_wo_taxonomy.join(OtuMf.mapping_file['age_in_days'])
# # merged_data_with_age = merged_data_with_age[merged_data_with_age.age_in_days.notnull()] # remove NaN days
# merged_data_with_age_group = otu_after_pca_wo_taxonomy.join(OtuMf.mapping_file['age_group'])
# merged_data_with_age_group = merged_data_with_age_group[merged_data_with_age_group.age_group.notnull()] # remove NaN days

physoligcal_data = OtuMf.mapping_file[['lean_mass', 'fat_mass', 'weight', 'percent_lean', 'percent_fat', 'percent_both', 'high_lean', 'high_fat', 'hi_weight', 'IL1b', 'IL6', 'IL10', 'IL17', 'TNFa', 'INFg', 'insulin', 'leptin']]
physoligcal_data = physoligcal_data.apply(lambda x: x.fillna(x.mean()),axis=0)
physoligcal_data_with_age_group = physoligcal_data.join(OtuMf.mapping_file['age_group'])

physoligcal_data_with_age_group = physoligcal_data_with_age_group.loc[physoligcal_data_with_age_group['age_group'].isin(['young', 'old'])]


# create train set and test set

stats = {'svm': {'test': {'wrong': [], 'size': [], 'score': [], 'roc_auc': []}, 'train': {'wrong': [], 'size': [], 'score': [], 'roc_auc': []}}
        ,'lda': {'test': {'wrong': [], 'size': [], 'score': [], 'roc_auc': []}, 'train': {'wrong': [], 'size': [], 'score': [], 'roc_auc': []}}}
# coefficent_df = {'svm': {'coef': [], 'class': []}, 'lda': {'coef': [], 'class': []}}

for i in range(3):
    print('\nIteration number: ', str(i+1))
    physoligcal_data_with_age_group = physoligcal_data_with_age_group.sample(frac=1)
    train_size = math.ceil(physoligcal_data_with_age_group.shape[0] * 0.8)
    train_set = physoligcal_data_with_age_group.iloc[0:train_size]
    test_set = physoligcal_data_with_age_group.iloc[train_size+1:]

    train_x_data = train_set.loc[:, train_set.columns != 'age_group']
    train_y_values = train_set['age_group']
    train_y_values = train_y_values.str.replace('old', '0')
    train_y_values = train_y_values.str.replace('young', '1')
    train_y_values = pd.to_numeric(train_y_values)

    test_x_data = test_set.loc[:, test_set.columns != 'age_group']
    test_y_values = test_set['age_group']
    test_y_values = test_y_values.str.replace('old', '0')
    test_y_values = test_y_values.str.replace('young', '1')
    test_y_values = pd.to_numeric(test_y_values)

    # clf = svm.SVC(kernel='linear')
    # print('SVM fit')
    # clf.fit(train_x_data, train_y_values)
    # print('SVM prediction')
    #
    # # test accuracy on the test set
    # train_set_df = pd.DataFrame(train_y_values)
    # train_set_df['predicted'] = clf.predict(train_x_data.values)
    # train_set_df['wrong'] = train_set_df['predicted'] != train_set_df['age_group']
    # stats['svm']['train']['wrong'].append(train_set_df['wrong'].sum())
    # stats['svm']['train']['size'].append(train_set_df['wrong'].size)
    # stats['svm']['train']['score'].append(clf.score(train_x_data, train_y_values))
    # # coefficent_df['svm']['coef'].append(clf.coef_)
    # # coefficent_df['svm']['class'].append(clf.classes_)
    #
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y_values.values,
    #                                                                 clf.decision_function(train_x_data))
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # stats['svm']['train']['roc_auc'].append(roc_auc)
    # plt.subplot(2,1,1)
    # plt.plot(false_positive_rate, true_positive_rate, 'b',label='Train AUC = %0.2f' % roc_auc)
    # plt.title('Train set')
    # print('SVM-Train - Total wrong predictions : {}, out of: {}, accuracy: {}, auc: {}'.format(train_set_df['wrong'].sum(),
    #                                                                                            train_set_df['wrong'].size,
    #                                                                                            clf.score(train_x_data, train_y_values),
    #                                                                                            roc_auc))
    # # test accuracy on the test set
    # test_set_df = pd.DataFrame(test_y_values)
    # test_set_df['predicted'] = clf.predict(test_x_data.values)
    # test_set_df['wrong'] = test_set_df['predicted'] != test_set_df['age_group']
    # stats['svm']['test']['wrong'].append(test_set_df['wrong'].sum())
    # stats['svm']['test']['size'].append(test_set_df['wrong'].size)
    # stats['svm']['test']['score'].append(clf.score(test_x_data, test_y_values))
    # # coefficent_df['svm']['coef'].append(clf.coef_)
    # # coefficent_df['svm']['class'].append(clf.classes_)
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y_values.values,
    #                                                                 clf.decision_function(test_x_data))
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # stats['svm']['test']['roc_auc'].append(roc_auc)
    # plt.subplot(2, 1, 2)
    # plt.plot(false_positive_rate, true_positive_rate, 'b', label='Test AUC = %0.2f' % roc_auc)
    # plt.title('Test set')
    # print('SVM-Test - Total wrong predictions : {}, out of: {}, accuracy: {}, auc: {}'.format(test_set_df['wrong'].sum(),
    #                                                                                           test_set_df['wrong'].size,
    #                                                                                           clf.score(test_x_data,
    #                                                                                                     test_y_values),
    #                                                                                           roc_auc))
    clf = LinearDiscriminantAnalysis()
    print('LDA fit')
    clf.fit(train_x_data, train_y_values)
    print('LDA prediction')

    # test accuracy on the test set
    train_set_df = pd.DataFrame(train_y_values)
    train_set_df['predicted'] = clf.predict(train_x_data.values)
    train_set_df['wrong'] = train_set_df['predicted'] != train_set_df['age_group']
    stats['lda']['train']['wrong'].append(train_set_df['wrong'].sum())
    stats['lda']['train']['size'].append(train_set_df['wrong'].size)
    stats['lda']['train']['score'].append(clf.score(train_x_data, train_y_values))
    # coefficent_df['svm']['coef'].append(clf.coef_)
    # coefficent_df['svm']['class'].append(clf.classes_)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y_values.values,
                                                                    clf.decision_function(train_x_data))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    stats['lda']['train']['roc_auc'].append(roc_auc)
    plt.subplot(2, 1, 1)
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='Train AUC = %0.2f' % roc_auc)
    plt.title('Train set')
    print('LDA-Train - Total wrong predictions : {}, out of: {}, accuracy: {}, auc: {}'.format(
        train_set_df['wrong'].sum(),
        train_set_df['wrong'].size,
        clf.score(train_x_data, train_y_values),
        roc_auc))
    # test accuracy on the test set
    test_set_df = pd.DataFrame(test_y_values)
    test_set_df['predicted'] = clf.predict(test_x_data.values)
    test_set_df['wrong'] = test_set_df['predicted'] != test_set_df['age_group']
    stats['lda']['test']['wrong'].append(test_set_df['wrong'].sum())
    stats['lda']['test']['size'].append(test_set_df['wrong'].size)
    stats['lda']['test']['score'].append(clf.score(test_x_data, test_y_values))
    # coefficent_df['svm']['coef'].append(clf.coef_)
    # coefficent_df['svm']['class'].append(clf.classes_)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y_values.values,
                                                                    clf.decision_function(test_x_data))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    stats['lda']['test']['roc_auc'].append(roc_auc)
    plt.subplot(2, 1, 2)
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='Test AUC = %0.2f' % roc_auc)
    plt.title('Test set')
    print(
        'LDA-Test - Total wrong predictions : {}, out of: {}, accuracy: {}, auc: {}'.format(test_set_df['wrong'].sum(),
                                                                                            test_set_df['wrong'].size,
                                                                                            clf.score(test_x_data,
                                                                                                      test_y_values),
                                                                                            roc_auc))

    if i == 0:
        plt.show()

# print('\nSVM - Train - Total wrong predictions : {}, out of: {}, accuracy: {}, AUC: {}'.format(np.sum(stats['svm']['train']['wrong']), np.sum(stats['svm']['train']['size']),
#                                                                             np.mean(stats['svm']['train']['score']), np.mean(stats['svm']['train']['roc_auc'])))
#
# print('SVM - Test - Total wrong predictions : {}, out of: {}, accuracy: {}, AUC: {}'.format(np.sum(stats['svm']['test']['wrong']), np.sum(stats['svm']['test']['size']),
#                                                                             np.mean(stats['svm']['test']['score']), np.mean(stats['svm']['test']['roc_auc'])))

print('\nLDA - Train - Total wrong predictions : {}, out of: {}, accuracy: {}, AUC: {}'.format(np.sum(stats['lda']['train']['wrong']), np.sum(stats['lda']['train']['size']),
                                                                            np.mean(stats['lda']['train']['score']), np.mean(stats['lda']['train']['roc_auc'])))

print('LDA - Test - Total wrong predictions : {}, out of: {}, accuracy: {}, AUC: {}'.format(np.sum(stats['lda']['test']['wrong']), np.sum(stats['lda']['test']['size']),
                                                                            np.mean(stats['lda']['test']['score']), np.mean(stats['lda']['test']['roc_auc'])))

# print(coefficent_df)
