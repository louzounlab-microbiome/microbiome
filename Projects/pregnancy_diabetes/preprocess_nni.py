import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import svm
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import interp
import statsmodels.api as sm
from sklearn.metrics import plot_roc_curve
import itertools
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Plot import calc_auc_on_flat_results
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from LearningMethods.nn_learning_model import nn_learn
from LearningMethods.nn_models import *
from LearningMethods.deep_learning_model import data_set


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

models_nn = {'relu_b':nn_2hl_relu_b_model, 'tanh_b':nn_2hl_tanh_b_model,
             'leaky_b':nn_2hl_leaky_b_model, 'sigmoid_b':nn_2hl_sigmoid_b_model,
             'relu_mul':nn_2hl_relu_mul_model, 'tanh_mul':nn_2hl_tanh_mul_model,
             'leaky_mul':nn_2hl_leaky_mul_model, 'sigmoid_mul':nn_2hl_sigmoid_mul_model,
             'relu1_b':nn_1hl_relu_b_model, 'tanh1_b':nn_1hl_tanh_b_model, 'leaky1_b':nn_1hl_leaky_b_model}
             
def split(X, y, test_size,k_fold):
    X_trains, X_tests, y_trains, y_tests= [], [], [], []
    for i in range(k_fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y, random_state=42+2*i)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    return  X_trains, X_tests, y_trains, y_tests

def read_otu_and_mapping_files(otu_path, mapping_path):
    otu_file = pd.read_csv(otu_path)
    mapping_file = pd.read_csv(mapping_path)
    X = otu_file.set_index("ID")
    X = X.rename(columns={x: y for x, y in zip(X.columns, range(0, len(X.columns)))})
    y = mapping_file.set_index("ID")
    return X, y

def print_auc_for_iter(y_test, y_score):
    fpr, tpr, thresholds = roc_curve(np.array(y_test), np.array(y_score))
    roc_auc = auc(fpr, tpr)
    print('ROC AUC = ' + str(round(roc_auc, 4)))
        
def generator(X_arr, y_arr, batch_size):
    net_params = {'batch_size': batch_size,
                  'shuffle': 1,
                  'num_workers': 4}
    set = data_set(X_arr, y_arr)
    generator = DataLoader(set, **net_params)
    return generator
    
def nn_model(X_train, X_test, y_train, y_test, Net):
    net = Net(list(X_train.shape)[1], 100, 10, 1).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr= 0.005)
    
    training_len = len(y_train) 
    validation_len = len(y_test)
    
    X_train = torch.from_numpy(np.array(X_train))
    X_test = torch.from_numpy(np.array(X_test))
    y_train = torch.from_numpy(np.array(y_train)).long()
    y_test = torch.from_numpy(np.array(y_test)).long()

    training_generator = generator(X_train, y_train, 16)
    validation_generator = generator(X_test, y_test, 16)
    
    params = {}
    params["epochs"] = 13
    
    test_auc, train_auc, net = nn_learn(net, training_generator, validation_generator, criterion, optimizer, plot = True, params = params, training_len = training_len, validation_len = validation_len, title = None)
    
    return train_auc, test_auc

def learn(X_trains, X_tests,y_trains ,y_tests, k_fold, task):
    all_y_train = []
    for i in range(k_fold):
        all_y_train.append(y_trains[i]['Tag'].values)
    all_y_train = np.array(all_y_train).flatten()

    all_y_test = []
    for i in range(k_fold):
        all_y_test.append(y_tests[i]['Tag'].values)
    all_y_test = np.array(all_y_test).flatten()

    #SVM
    clf = svm.SVC(kernel='linear', C=0.1, gamma='scale', class_weight='balanced')
    
    y_test_scores, y_train_scores = [], []
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i in range(k_fold):
        print('------------------------------\niteration number ' + str(i))
        X_train, X_test, y_train, y_test = X_trains[i], X_tests[i], y_trains[i], y_tests[i], 
        # FIT
        clf.fit(X_train, y_train)
        # GET RESULTS
        y_score = clf.decision_function(X_test)
        train_score = clf.decision_function(X_train)
        y_train_scores.append(train_score)
        y_test_scores.append(y_score)
        '''
        viz = plot_roc_curve(clf, X_test, y_test,
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        '''
        print_auc_for_iter(np.array(y_tests[i]['Tag'].values), np.array(y_score).T)
    '''
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)/np.sqrt(k_fold)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic " + task)
    ax.legend(loc="lower right")
    plt.savefig(task + ".svg")
    '''
    y_train_scores = np.array(y_train_scores).flatten()
    y_test_scores = np.array(y_test_scores).flatten()

    SVM_train_auc, SVM_test_auc, _, _ = calc_auc_on_flat_results(all_y_train, y_train_scores, all_y_test, y_test_scores)
    
    
    #XGBOOST
    clf = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, objective='binary:logistic',gamma=0.5, min_child_weight=3, booster='gbtree')
               
    y_test_scores, y_train_scores = [], []
    for i in range(k_fold):
        print('------------------------------\niteration number ' + str(i))
        X_train, X_test, y_train, y_test = X_trains[i], X_tests[i], y_trains[i], y_tests[i], 
        # FIT
        clf.fit(X_train, y_train)
        # GET RESULTS
        y_score = clf.predict_proba(X_test)
        train_score = clf.predict_proba(X_train)
        y_train_scores.append(train_score[:, 1])
        y_test_scores.append(y_score[:, 1])
        
        print_auc_for_iter(np.array(y_tests[i]['Tag'].values), np.array(y_score).T[1])
    
    y_train_scores = np.array(y_train_scores).flatten()
    y_test_scores = np.array(y_test_scores).flatten()

    XGB_train_auc, XGB_test_auc, _, _ = calc_auc_on_flat_results(all_y_train, y_train_scores, all_y_test, y_test_scores)
    
    #NN
    NN_test_auc = 0
    NN_train_auc = 0
    for i in range(k_fold):
        Net = models_nn['relu_b']
        print('------------------------------\niteration number ' + str(i))
        X_train, X_test, y_train, y_test = X_trains[i], X_tests[i], y_trains[i], y_tests[i]
        train_auc, test_auc = nn_model(X_train, X_test, y_train, y_test, Net)
        NN_train_auc += train_auc
        NN_test_auc += test_auc  
    
    NN_train_auc /= k_fold
    NN_test_auc /= k_fold

    
    return SVM_train_auc, SVM_test_auc, XGB_train_auc, XGB_test_auc, NN_train_auc, NN_test_auc
    
def parallel_pipeline(arg, task):    
    tax_dict = ['four', 'five', 'six']
    k_fold = 10
    # parameters for Preprocess
    preprocess_prms = {'taxonomy_level': int(arg[0]), 'taxnomy_group': arg[1], 'epsilon': 0.1,
                     'normalization': arg[2][0], 'z_scoring': arg[2][1], 'norm_after_rel': arg[2][1],
                     'std_to_delete': 0, 'pca': arg[3]}
    
    main_task = task +'_taxonomy_level_' + str(arg[0])
    bactria_as_feature_file = task +'_OTU.csv'
    samples_data_file = task +'_Tag.csv'
    
    rhos_folder = os.path.join(task, 'rhos')
    pca_folder = os.path.join(task , 'pca')
     
    mapping_file = CreateOtuAndMappingFiles(bactria_as_feature_file, samples_data_file)
    mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=False)
    #mapping_file.rhos_and_pca_calculation(main_task, preprocess_prms['taxonomy_level'], preprocess_prms['pca'],
    #                                     rhos_folder, pca_folder)
    otu_path, mapping_path, pca_path = mapping_file.csv_to_learn(main_task, os.path.join(os.getcwd(), task), arg[0])
   
    #create X,Y
    X, y = read_otu_and_mapping_files(otu_path, mapping_path)
    #split data
    X_trains, X_tests, y_trains, y_tests = split(X, y, 0.2, k_fold)
    #learn
    SVM_train_auc, SVM_test_auc, XGB_train_auc, XGB_test_auc, NN_train_auc, NN_test_auc = learn(X_trains, X_tests,y_trains ,y_tests, k_fold, arg[1])
    print('taxonomy ' + str(arg[0]) +' '+ str(arg[1])+ ' '+  str(arg[2][0])+ ' '+str(arg[2][1]) + ' PCA ' + str(mapping_file.pca_comp)+':')
    print("SVM Train AUC: " + str(SVM_train_auc))
    print("SVM Test AUC: " + str(SVM_test_auc))
    print("XGBOOST Train AUC: " + str(XGB_train_auc))
    print("XGBOOST Test AUC: " + str(XGB_test_auc))
    print("NN Train AUC: " + str(NN_train_auc))
    print("NN Test AUC: " + str(NN_test_auc))
    
    svm_results_file = Path(task + "/all_svm_results_ica.csv")
    if not svm_results_file.exists():
        all_svm_results = pd.DataFrame(columns=['Taxonomy level', 'taxonomy group', 'normalization 1', 'normalization 2','Dim Red', 'TRAIN-AUC', 'TEST-AUC'])
        all_svm_results.to_csv(svm_results_file, index=False)
    all_svm_results = pd.read_csv(svm_results_file)
    all_svm_results.loc[len(all_svm_results)] = [tax_dict[arg[0]-4], arg[1], arg[2][0], arg[2][1], arg[3][1], SVM_train_auc, SVM_test_auc]
    all_svm_results.to_csv(svm_results_file, index=False)
    
    xgboost_results_file = Path(task + "/all_xgboost_results_ica.csv")
    if not xgboost_results_file.exists():
        all_xgboost_results = pd.DataFrame(columns=['Taxonomy level', 'taxonomy group', 'normalization 1', 'normalization 2','Dim Red', 'TRAIN-AUC', 'TEST-AUC'])
        all_xgboost_results.to_csv(xgboost_results_file, index=False)
    all_xgboost_results = pd.read_csv(xgboost_results_file)
    all_xgboost_results.loc[len(all_xgboost_results)] = [tax_dict[arg[0]-4], arg[1], arg[2][0], arg[2][1], arg[3][1], XGB_train_auc, XGB_test_auc]
    all_xgboost_results.to_csv(xgboost_results_file, index=False)
    
    nn_results_file = Path(task + "/all_nn_results_ica.csv")
    if not nn_results_file.exists():
        all_nn_results = pd.DataFrame(columns=['Taxonomy level', 'taxonomy group', 'normalization 1', 'normalization 2','Dim Red', 'TRAIN-AUC', 'TEST-AUC'])
        all_nn_results.to_csv(nn_results_file, index=False)
    all_nn_results = pd.read_csv(nn_results_file)
    all_nn_results.loc[len(all_nn_results)] = [tax_dict[arg[0]-4], arg[1], arg[2][0], arg[2][1], arg[3][1], NN_train_auc, NN_test_auc]
    all_nn_results.to_csv(nn_results_file, index=False)
    
def calc_coeff_linear_model(task_list):
    beta_matrix_svm = []
    beta_matrix_xgboost = []
    beta_matrix_nn = []
    clf_list = ['svm', 'xgboost', 'nn']
    for task in task_list:
        for clf in clf_list:
            f = Path(task + "/all_"+clf+"_results_ica.csv")
            df = pd.read_csv(f)
            df = df.loc[:, df.columns != 'TRAIN-AUC']
            df["normalization"] = df["normalization 1"] + df["normalization 2"]
            df = df.drop(columns = ["normalization 1", "normalization 2"])
            X = df.loc[:, df.columns != 'TEST-AUC']
            X = pd.get_dummies(X)
            y = df['TEST-AUC']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            
            y_pred = regressor.predict(X_test)
            #df_test = pd.DataFrame({'Actual AUC': y_test, 'Predicted AUC': y_pred})
            #df_test.plot(kind='bar', figsize=(10, 8))
            #plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
            #plt.savefig(task + '_' + clf + '.svg')
            
            train_l, train_o = y_train, regressor.predict(X_train)
            test_l, test_o = y_test, y_pred
            #train_l, train_o = train_l.flatten().detach().numpy(), train_o.flatten().detach().numpy()
            #test_l, test_o = test_l.flatten().detach().numpy(), test_o.flatten().detach().numpy()
            min_val = min(min(np.min(train_l), np.min(train_o)), min(np.min(test_l), np.min(test_o)))
            max_val = max(max(np.max(train_l), np.max(train_o)), max(np.max(test_l), np.max(test_o)))
            fig, ax = plt.subplots()
            plt.ylim((min_val, max_val))
            plt.xlim((min_val, max_val))
            #plt.scatter(train_l, train_o, label='Train', color='red', s=3, alpha=0.3)
            #plt.scatter(test_l, test_o, label='Test', color='blue', s=3, alpha=0.3)
            plt.title('Regression task ' + task + ' ' + clf, fontsize=10)
            sns.regplot(x=y_pred, y=y_test)
            plt.xlabel('Real values')
            plt.ylabel('Predicted Values')
            plt.legend(loc='upper left')
            plt.savefig(task + '_' + clf + '.svg')

            
            beta = regressor.coef_
            #substract mean from coefficient 
            beta[0:3] -= np.mean(beta[0:3])
            beta[3:6] -= np.mean(beta[3:6])
            beta[6:9] -= np.mean(beta[6:9])
            beta[9:15] -= np.mean(beta[9:15])
            #add beta to matrix
            if clf == 'svm':
                beta_matrix_svm.append(beta)
            elif clf == 'xgboost':
                beta_matrix_xgboost.append(beta)
            else:
                beta_matrix_nn.append(beta)
             
    beta_matrix_svm = pd.DataFrame(data = beta_matrix_svm, columns = X.columns)
    beta_matrix_xgboost = pd.DataFrame(data = beta_matrix_xgboost, columns = X.columns)
    beta_matrix_nn = pd.DataFrame(data = beta_matrix_nn, columns = X.columns)
    #plot coeff
    plt.figure(figsize=(14, 12))    
    ax_svm = sns.stripplot(data=beta_matrix_svm)
    ax_svm.set_facecolor("grey")
    plt.title('SVM coefficient plot')
    plt.xticks(rotation=30, horizontalalignment='right', fontsize=8)
    plt.savefig("SVM_coeff.svg")
    plt.clf()
    ax_xgboost = sns.stripplot(data=beta_matrix_xgboost)
    ax_xgboost.set_facecolor("grey")
    plt.title('XGB coefficient plot')
    plt.xticks(rotation=30, horizontalalignment='right', fontsize=8)
    plt.savefig("XGB_coeff.svg")
    plt.clf()
    ax_nn = sns.stripplot(data=beta_matrix_nn)
    ax_nn.set_facecolor("grey")
    plt.title('NN coefficient plot')
    plt.xticks(rotation=30, horizontalalignment='right', fontsize=8)
    plt.savefig("NN_coeff.svg")
    
    
if __name__ == '__main__':
    task_list = ["ABX", "Allergy", "Mucositis", "Il1a", "Pregnancy", "Progesterone"]
    #task_list = ["ibd", "crc1", "crc2", "crc1+2"]
    #task_list = ["ibd_MetAML", "crc_MetAML"]
    
    for task in task_list:
        s = [[4,5,6], ['sum', 'mean', 'sub PCA'], [("log", "No"), ("log", "col"), ("log", "row"), ("log", "both"), ("relative", "No"), ("relative", "z_after_relative")], [(-1, 'ICA'), (-1, 'PCA'), (0, 'None')]]
        arg_list=list(itertools.product(*s))
        for arg in arg_list:
            parallel_pipeline(arg, task)
    
    
    #calc coefficient of linear regression model and plot
    calc_coeff_linear_model(task_list)
    
    