from scipy import interp
from pathlib import Path
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import sys
import itertools
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
    
def XGB_create_classifiers(params):  # suited for xgb only
        optional_classifiers = []
        # create all possible classifiers
        for max_depth in params['max_depth']:
            for learning_rate in params['learning_rate']:
                for n_estimators in params['n_estimators']:
                    for objective in params['objective']:
                        for gamma in params['gamma']:
                            for min_child_weight in params['min_child_weight']:
                                clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate,
                                                    n_estimators=n_estimators, objective=objective,
                                                    gamma=gamma, min_child_weight=min_child_weight,
                                                    booster='gblinear')
                                optional_classifiers.append(clf)
        return optional_classifiers

def SVM_create_classifiers(params, weights):  # suited for svm only
        optional_classifiers = []
        # create all possible classifiers
        for kernel in params['kernel']:
            for gamma in params['gamma']:
                for C in params['C']:
                    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, class_weight=weights)  # class_weight='balanced')
                    optional_classifiers.append(clf)
        return optional_classifiers

def read_otu_and_mapping_files(otu_path, mapping_path):
    otu_file = pd.read_csv(otu_path)
    mapping_file = pd.read_csv(mapping_path)
    X = otu_file.set_index("ID").values
    y = mapping_file["Tag"]
    return np.array(X), np.array(y)


def get_weights(y):
    classes_sum = [np.sum(np.array(y) == unique_class) for unique_class in
                   np.unique(np.array(y))]
    classes_ratio = [1 - (a / sum(classes_sum)) for a in classes_sum]
    weights_map = {a: classes_ratio[a] for a in set(y)}
    return weights_map


def run(X, y, k_cross_val, trimester, site, tax, classifier, model_flag):
    cv = StratifiedKFold(n_splits=k_cross_val)
    
    tprs = []
    aucs = []
    train_aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):

        classifier.fit(X[train], y[train])

        viz_train = metrics.plot_roc_curve(classifier, X[train], y[train],
                                     name='ROC fold {}'.format(i),
                                    alpha=0.3, lw=1, ax=ax)

        #viz = metrics.plot_roc_curve(classifier, X[test], y[test],
        #                     name='ROC fold {}'.format(i),
        #                     alpha=0.3, lw=1, ax=ax)
        #interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        #interp_tpr[0] = 0.0
        #tprs.append(interp_tpr)
        #aucs.append(viz.roc_auc)
        train_aucs.append(viz_train.roc_auc)

    print('tax level_'+str(tax)+' '+str(site)+ str(trimester) + ' train auc: ' + str(np.mean(train_aucs)) )
    
    '''
    directory  = 'GDM_'+ str(trimester) + '_' + str(site) +'_taxonomy_'+str(tax)
    svm_file = 'all_svm_results.csv'
    xgb_file = 'all_xgboost_results.csv'
    if not os.path.isdir('./' + directory):
        os.mkdir(directory)
        os.chdir(directory)
        os.mkdir('SVM')
        os.chdir('SVM')
        all_svm_results_SVM = pd.DataFrame(columns=['KERNEL', 'GAMMA', 'C', 'TEST-AUC'])
        all_svm_results_SVM.to_csv(svm_file, index=False)
        os.chdir('..')
        os.mkdir('XGBOOST')
        os.chdir('XGBOOST')
        all_svm_results_XGB = pd.DataFrame(columns=['LR', 'MAX-DEPTH', 'N-ESTIMATORS', 'OBJECTIVE', 'GAMMA', 'MIN-CHILD-WEIGHT', 'BOOSTER','TEST-AUC'])
        all_svm_results_XGB.to_csv(xgb_file, index=False)
        os.chdir('../..')
    os.chdir(directory)
    save_result(tprs, aucs, mean_fpr, trimester, site, tax, classifier, model_flag, xgb_file, svm_file,fig, ax)
    os.chdir('..')
    '''

    
      
def main_preprocess(tax, site, trimester, preprocess_prms):
    main_task = 'GDM_taxonomy_level_' + str(tax)+'_'+site+'_trimester_'+trimester
    bactria_as_feature_file = 'GDM_OTU_rmv_dup.csv'
    
    samples_data_file = 'GDM_tag_rmv_dup_' +trimester + '_' + site +'.csv'
    rhos_folder = os.path.join('pregnancy_diabetes_'+trimester+'_'+site, 'rhos')
    pca_folder = os.path.join('pregnancy_diabetes_'+trimester+'_'+site, 'pca')

    
    mapping_file = CreateOtuAndMappingFiles(bactria_as_feature_file, samples_data_file)
    mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=False)
    mapping_file.rhos_and_pca_calculation(main_task, preprocess_prms['taxonomy_level'], preprocess_prms['pca'],
                                         rhos_folder, pca_folder)
    otu_path, mapping_path, pca_path = mapping_file.csv_to_learn(main_task, os.path.join(os.getcwd(), 'pregnancy_diabetes_'+trimester+'_'+site), tax)
    return otu_path, mapping_path, pca_path


def save_result(tprs, aucs, mean_fpr, trimester, site, tax, classifier, model_flag, xgb_file, svm_file,fig, ax):
    os.chdir(model_flag)
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    #calc and plot mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    test_mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (test_mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic " + model_flag +" " + str(trimester) + '_' + str(site) +'_'+str(tax))
    ax.legend(loc="lower right")
    
    if model_flag == 'SVM':
        fig_name = 'C=' + str(classifier.C)+'_'+ 'G=' + str(classifier.gamma) + '_ROC_NO_PCA'+ str(trimester) + '_' + str(site) +'_'+str(tax)
        all_svm_results = pd.read_csv(svm_file)
        all_svm_results.loc[len(all_svm_results)] = [classifier.kernel, classifier.C, classifier.gamma, test_mean_auc]
        all_svm_results.to_csv(svm_file, index=False)
    elif model_flag == 'XGBOOST':
        fig_name = 'lr=' + str(classifier.learning_rate)+'_'+ 'd=' + str(classifier.max_depth)+'_'+ 'c=' + str(classifier.min_child_weight)+ '_'+ 'G=' + str(classifier.gamma) + '_ROC_NO_PCA'+ str(trimester) + '_' + str(site) +'_'+str(tax)
        all_xgb_results = pd.read_csv(xgb_file)
        all_xgb_results.loc[len(all_xgb_results)] = [classifier.learning_rate, classifier.max_depth, classifier.n_estimators,
                                                         classifier.objective, classifier.gamma, classifier.min_child_weight, classifier.booster, test_mean_auc]
        all_xgb_results.to_csv(xgb_file, index=False)
    plt.savefig(fig_name+'.png')
    plt.clf()
    plt.close()
    
    os.chdir('..')


def parallel_pipeline(t):
    trimester = t[0]
    site = t[1]
    tax = int(t[2])
    
    SVM_params = {'kernel': ['linear'],
                  'gamma': ['auto'],
                  'C': [0.01, 0.1, 1, 10, 100, 1000]}
                  
    XGB_params = {'learning_rate': [0.3],
                           'objective': ['binary:logistic'],
                           'n_estimators': [1000],
                           'max_depth': [5],  
                           'min_child_weight': [1,3,5,9],
                           'gamma': [0]}
    
    # parameters for Preprocess
    preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 0.1,
                     'normalization': 'log', 'z_scoring': 'row', 'norm_after_rel': '',
                     'std_to_delete': 0.25, 'pca': 5}
    otu_path, mapping_path, pca_path = main_preprocess(tax, site, trimester, preprocess_prms)
    
    X, y = read_otu_and_mapping_files(otu_path, mapping_path)
    #SVM
    #weights = get_weights(y)
    #clf_list = SVM_create_classifiers(SVM_params, weights)
    #for classifier in clf_list:
    #   run(X, y, 7, trimester, site, tax, classifier, 'SVM')
    
    #XGBOOST
    clf_list = XGB_create_classifiers(XGB_params)
    for classifier in clf_list:
        run(X, y, 7, trimester, site, tax, classifier,'XGBOOST')
        

if __name__ == '__main__':
    s = [['T1', 'T2_T3'], ['SALIVA', 'STOOL'], [5,6]]
    arg_list=list(itertools.product(*s))
    for arg in arg_list:
        parallel_pipeline(arg)
    #Parallel(n_jobs=8)(delayed(parallel_pipeline)(arg) for arg in arg_list)