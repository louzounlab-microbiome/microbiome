import sys
import os
import nni
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Plot import calc_auc_on_flat_results
from LearningMethods.nn_learning_model import nn_main
from LearningMethods.nn_models import *

models_nn = {'relu_b':nn_2hl_relu_b_model, 'tanh_b':nn_2hl_tanh_b_model,
             'leaky_b':nn_2hl_leaky_b_model, 'sigmoid_b':nn_2hl_sigmoid_b_model,
             'relu_mul':nn_2hl_relu_mul_model, 'tanh_mul':nn_2hl_tanh_mul_model,
             'leaky_mul':nn_2hl_leaky_mul_model, 'sigmoid_mul':nn_2hl_sigmoid_mul_model}

if __name__ == '__main__':
    
    params = {
        "site": 'SALIVA',
        "taxonomy": 5,
        "model": 'sigmoid_b',
        "hid_dim_0": 10,
        "hid_dim_1": 5,
        "reg": 0.,
        "dims": [20, 40, 60, 2],
        "lr": 0.01,
        "test_size": 0.2,
        "batch_size": 8,
        "shuffle": 1,
        "num_workers": 4,
        "epochs": 100,
        "optimizer": 'Adam',
        "loss": 'MSE'
    }
    
    #params = nni.get_next_parameter()
    Net = models_nn[params["model"]]
    #init classifier
    #clf_params = nni.get_next_parameter()
    clf_score = XGBClassifier(learning_rate=0.45342, n_estimators=int(220),
                            objective='binary:logistic', gamma=4.690, 
                            reg_lambda=4.9986, booster='gblinear', alpha=3.6)
                        
    #read data
    print('load data...')
    min_max_scaler = preprocessing.MinMaxScaler()
    df_concat = pd.read_csv('DB/metadata.csv')
    df_concat = df_concat.set_index('ID')
    df_concat.iloc[:, :-1] = min_max_scaler.fit_transform(df_concat.iloc[:, :-1])
    X = df_concat.iloc[:, :-1]
    y = df_concat.iloc[:, -1]
    k_fold = 7
    
    #split data
    print('split data...')
    X_trains, X_tests, y_trains, y_tests= [], [], [], []
    for i in range(k_fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    
    #learn
    print('train data...')
    y_train_scores, y_test_scores  = [], []
    for i in range(k_fold):
        print('------------------------------\niteration number ' + str(i))
        X_train, X_test, y_train, y_test = X_trains[i], X_tests[i], y_trains[i], y_tests[i]
        clf_score.fit(X_train, y_train)
        clf_score.predict_proba(X_test)
        y_score = clf_score.predict_proba(X_test)
        y_test_scores.append(y_score[:, 1])
        train_score = clf_score.predict_proba(X_train)
        y_train_scores.append(train_score[:, 1])
        #calc AUC per iteration
        fpr, tpr, thresholds = roc_curve(np.array(y_test), np.array(np.array(y_score).T[1]))
        roc_auc = auc(fpr, tpr)
        print('ROC AUC = ' + str(round(roc_auc, 4)))
    
    #calc AUC on all iterations
    all_y_train = []
    for i in range(k_fold):
        all_y_train.append(y_trains[i].values)
    all_y_train = np.array(all_y_train).flatten()

    all_y_test = []
    for i in range(k_fold):
        all_y_test.append(y_tests[i].values)
    all_y_test = np.array(all_y_test).flatten()

    y_train_scores = np.array(y_train_scores).flatten()
    y_test_scores = np.array(y_test_scores).flatten()
    
    train_auc, test_auc, train_rho, test_rho = calc_auc_on_flat_results(all_y_train, y_train_scores, all_y_test, y_test_scores)
    
    '''  
    test_auc, acc = nn_main(X, y, params, 'GDM_extra_features', Net, plot=True, k_fold=5)
    print('Final auc: ' +str(test_auc))
    nni.report_final_result(test_auc)
    '''