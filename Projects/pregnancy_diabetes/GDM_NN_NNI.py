import nni
import sys
import os
import pandas as pd
import numpy as np
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from LearningMethods.nn_models import *
from LearningMethods.nn_learning_model import nn_main, nn_learn, create_optimizer_loss, generator
from ensmble_model import create_concate_df_to_learn

warnings.filterwarnings("ignore")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

models_nn = {'relu_b':nn_2hl_relu_b_model, 'tanh_b':nn_2hl_tanh_b_model,
             'leaky_b':nn_2hl_leaky_b_model, 'sigmoid_b':nn_2hl_sigmoid_b_model,
             'relu_mul':nn_2hl_relu_mul_model, 'tanh_mul':nn_2hl_tanh_mul_model,
             'leaky_mul':nn_2hl_leaky_mul_model, 'sigmoid_mul':nn_2hl_sigmoid_mul_model,
             'relu1_b':nn_1hl_relu_b_model, 'tanh1_b':nn_1hl_tanh_b_model, 'leaky1_b':nn_1hl_leaky_b_model}


def read_otu_and_mapping_files(otu_path, mapping_path):
    otu_file = pd.read_csv(otu_path)
    mapping_file = pd.read_csv(mapping_path)
    X = otu_file.set_index("ID")
    X = X.rename(columns={x: y for x, y in zip(X.columns, range(0, len(X.columns)))})
    y = mapping_file.set_index("ID")
    return X, y


def split(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y, random_state=42)
    training_len = len(X_train)
    validation_len = len(X_test)

    ids_list_train = [id for id in X_train.index.values]
    ids_list_test = [id for id in X_test.index.values]

    X_train = torch.from_numpy(np.array(X_train))
    X_test = torch.from_numpy(np.array(X_test))
    y_train = torch.from_numpy(np.array(y_train)).long()
    y_test = torch.from_numpy(np.array(y_test)).long()

    return  X_train, X_test, y_train, y_test, training_len, validation_len, ids_list_train, ids_list_test

def main_preprocess(tax, site, trimester, preprocess_prms):
    main_task = 'GDM_taxonomy_level_' + str(tax)+'_'+site+'_trimester_'+trimester
    bactria_as_feature_file = 'GDM_OTU_rmv_dup_arrange.csv'
    
    samples_data_file = 'GDM_tag_rmv_dup_' +trimester + '_' + site +'.csv'
    rhos_folder = os.path.join('pregnancy_diabetes_'+trimester+'_'+site, 'rhos')
    pca_folder = os.path.join('pregnancy_diabetes_'+trimester+'_'+site, 'pca')

    mapping_file = CreateOtuAndMappingFiles(bactria_as_feature_file, samples_data_file)
    mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=True)
    mapping_file.rhos_and_pca_calculation(main_task, preprocess_prms['taxonomy_level'], preprocess_prms['pca'],
                                         rhos_folder, pca_folder)
    otu_path, mapping_path, pca_path = mapping_file.csv_to_learn(main_task, os.path.join(os.getcwd(), 'pregnancy_diabetes_'+trimester+'_'+site), tax)
    return otu_path, mapping_path, pca_path


def learn(X, y, title, first_Net, second_Net, params_first, params_second,df_concat,  plot=False):
    #first network
    net = first_Net(list(X.shape)[1], int(params_first["hid_dim_0"]), int(params_first["hid_dim_1"]), 1).to(device)
    criterion, optimizer = create_optimizer_loss(net, params_first, device)
  
    X_train, X_test, y_train, y_test, training_len, validation_len, ids_list_train, ids_list_test = split(X, y, params_first['test_size'])

    training_generator = generator(X_train, y_train, params_first)
    validation_generator = generator(X_test, y_test, params_first)

    epoch_auc, test_running_acc, net = nn_learn(net, training_generator, validation_generator, criterion, optimizer, plot, params_first, training_len, validation_len, title)

    #get last hidden layer for all train samples
    net.eval()
    X_train = torch.tensor(X_train, device=device, dtype=torch.float)
    _, last_layer_train_list = net(X_train)
    
    X_test = torch.tensor(X_test, device=device, dtype=torch.float)
    _, last_layer_validation_list = net(X_test)

    
    #building new data frame for learning all model
    fv_train_df = build_features_vector_df(ids_list_train, last_layer_train_list)
    X_train_ens, y_train_ens = create_concate_df_to_learn(fv_train_df, df_concat, 1)
    fv_test_df = build_features_vector_df(ids_list_test,last_layer_validation_list)
    X_test_ens, y_test_ens = create_concate_df_to_learn(fv_test_df, df_concat, 2)
    
    auc_mean = 0
    for i in range(5):
        #second network  
        net_second = second_Net(list(X_train_ens.shape)[1], int(params_second["hid_dim_0"]), int(params_second["hid_dim_1"]), 1).to(device)
        criterion_second_model, optimizer_second_model = create_optimizer_loss(net_second, params_second, device)
      
        X_train_ens = torch.from_numpy(np.array(X_train_ens))
        X_test_ens = torch.from_numpy(np.array(X_test_ens))
        y_train_ens = torch.from_numpy(np.array(y_train_ens)).long()
        y_test_ens = torch.from_numpy(np.array(y_test_ens)).long()
      
        training_generator_ens = generator(X_train_ens, y_train_ens, params_second)
        validation_generator_ens = generator(X_test_ens, y_test_ens, params_second)
      
        epoch_auc, test_running_acc, net = nn_learn(net_second, training_generator_ens, validation_generator_ens, criterion_second_model, optimizer_second_model, plot, params_second, training_len, validation_len, title)
        auc_mean += epoch_auc
        
    return auc_mean/5, test_running_acc


def build_features_vector_df(idx_list, features_vec):
    columns_list = [str(i) for i in range(len(features_vec[0]))]
    df = pd.DataFrame(np.array(features_vec.data.cpu().numpy()),columns=columns_list)
    df['ID'] = idx_list
    df =df.set_index('ID')
    return df


def two_model_main(result_type):
    #params_second = nni.get_next_parameter()
    
    params_second = {
        "model": 'sigmoid_b',
        "hid_dim_0": 120,
        "hid_dim_1": 40,
        "reg": 0.13,
        "dims": [20, 40, 60, 2],
        "lr": 0.1,
        "test_size": 0.2,
        "batch_size": 8,
        "shuffle": 0,
        "num_workers": 4,
        "epochs": 250,
        "optimizer": 'SGD',
        "loss": 'BCE'
    }
    
    params_first = {
        "site": 'SALIVA',
        "taxonomy": 5,
        "model": 'sigmoid_b',
        "hid_dim_0": 50,
        "hid_dim_1": 20,
        "reg": 0.,
        "dims": [20, 40, 60, 2],
        "lr": 0.0012,
        "test_size": 0.5,
        "batch_size": 16,
        "shuffle": 1,
        "num_workers": 4,
        "epochs": 50,
        "optimizer": 'Adam',
        "loss": 'BCE'
    }

    trimester = 'T1'
    site = str(params_first['site'])
    tax = int(params_first['taxonomy'])
    
    preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 0.1,
                     'normalization': 'log', 'z_scoring': 'row', 'norm_after_rel': '',
                     'std_to_delete': 0.2, 'pca': 0}
    #create data with Preprocess
    #otu_path, mapping_path, pca_path = main_preprocess(tax, site, trimester, preprocess_prms)
    #read directly
    otu_path = "pregnancy_diabetes_T1_SALIVA/OTU_merged_GDM_taxonomy_level_5_SALIVA_trimester_T1.csv"
    mapping_path =  "pregnancy_diabetes_T1_SALIVA/Tag_file_GDM_taxonomy_level_5_SALIVA_trimester_T1.csv" 
    #read data from CSV
    X, y = read_otu_and_mapping_files(otu_path, mapping_path)
    # read extra features data frame
    df_concat = pd.read_csv('DB/metadata.csv')
    df_concat = df_concat.set_index('ID')
    #normalize column features
    min_max_scaler = preprocessing.MinMaxScaler()
    df_concat.iloc[:, :-1] = min_max_scaler.fit_transform(df_concat.iloc[:, :-1])
    #creat nn model
    first_Net = models_nn[params_first["model"]]
    #create second model
    second_Net = models_nn['sigmoid_b']
    #fit ant evaluate model
    auc, acc = learn(X, y, 'GDM_tax_level_'+ str(tax)+'from_'+ site+'trimester_'+trimester, first_Net, second_Net, params_first, params_second, df_concat, plot=True)
    
    print('AUC: ' + str(auc))
    
    if result_type == "acc":
        nni.report_final_result(acc)
    if result_type == "auc":
        nni.report_final_result(auc)
    else:
        raise Exception 


def main_nni(result_type):
    #params = nni.get_next_parameter()
    
    params = {
        "site": 'STOOL',
        "taxonomy": 5,
        "model": 'relu_b',
        "hid_dim_0": 100,
        "hid_dim_1": 20,
        "reg": 0.1,
        "dims": [20, 40, 60, 2],
        "lr": 0.001,
        "test_size": 0.3,
        "shuffle": 1,
        "num_workers": 4,
        "epochs": 200,
        "optimizer": 'Adam',
        "loss": 'MSE'
    }
    
    trimester = 'T1'
    site = 'STOOL'
    tax = 5
    
    preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 0.1,
                     'normalization': 'log', 'z_scoring': 'row', 'norm_after_rel': '',
                     'std_to_delete': 0.05, 'pca': 0}
    #create data with Preprocess
    otu_path, mapping_path, pca_path = main_preprocess(tax, site, trimester, preprocess_prms)
    #read directly
    #otu_path = "pregnancy_diabetes_T1_SALIVA/OTU_merged_GDM_taxonomy_level_5_SALIVA_trimester_T1.csv"
    #mapping_path =  "pregnancy_diabetes_T1_SALIVA/Tag_file_GDM_taxonomy_level_5_SALIVA_trimester_T1.csv" 
    #read data from CSV
    X, y = read_otu_and_mapping_files(otu_path, mapping_path)
    #creant nn model
    Net = models_nn[params['model']]
    #fit ant evaluate model
    #nni_trial_id = nni.get_trial_id(), nni_exp_id = nni.get_experiment_id()
    auc, acc = nn_main(X, y, params, 'GDM_tax_level_'+ str(tax)+'from_'+ site+'trimester_'+trimester, Net, plot=True, k_fold=3)
    
    print('Final AUC: ' + str(auc))
    
    if result_type == "acc":
        nni.report_final_result(acc)
    if result_type == "auc":
        nni.report_final_result(auc)
    else:
        raise Exception
    

if __name__ == "__main__":
    main_nni(result_type="auc")
    #two_model_main(result_type="auc")

