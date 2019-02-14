import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
def print_recursive(object_to_print, count=0):
    spaces = count * ' '
    if type(object_to_print) is dict:
        count += 1
        for key, val in object_to_print.items():
            if key in ['spearman', 'pearson', 'mse']:
                print(f'{spaces}{key}: {val}')
            else:
                print(f'{spaces}{key}')
                print_recursive(val, count=count)

    else:
        # spaces = (count-1) * '\t'+ ' '
        print(f'{spaces}{object_to_print}')
# grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_my_loss_corrected\grid_search_results.txt'
grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_mse_loss\grid_search_results.txt'
grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_mse_loss\grid_search_results.txt'
grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_my_loss_factor_layers_neurons\grid_search_results.txt'
grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_my_loss_fine\grid_search_results.txt'
grid_search_file = r'C:\Thesis\subject_test_100_processes_true\grid_search_results.txt'
grid_search_file = r'C:\Thesis\subject_test_100_processes_mse_false\grid_search_results.txt'
grid_search_file = r'C:\Thesis\subject_test_100_processes_myloss_USE_SUBJECT_ONLY_TEST_TRAIN=True\grid_search_results.txt'
grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_combine_algo_new_loss_epochs_dropout_l2_numberlayers_numberneurons\grid_search_results.txt'
grid_search_file = r'C:\Thesis\GridSearch_combine_algo_new_loss_50_times\grid_search_results.txt'

with open(grid_search_file,'r') as f:
    grid_search_data = f.readlines()


configuration_stats ={}
best_mse=10000
best_config=''
best_mse_config=''
best_rho=-100
hist_mse_best = None
hist_rho_best = None
best_rho_config = ''
best_mse_config= ''
total_test_rho=[]
total_test_mse=[]
count=0
for idx in range(0, int(len(grid_search_data)/6)):
    configuration_set = grid_search_data[idx*6:(idx+1)*6]
    current_config=''
    try:
        current_config = {configuration_set[2][:-1]: eval(configuration_set[3][1:-1])}
        current_config.update({configuration_set[4][:-1]: eval(configuration_set[5][1:-1])})
        test_mse = current_config['Test']['mse']
        train_mse = current_config['Train']['mse']
        test_rho = current_config['Test']['spearman']['rho']
        train_rho = current_config['Train']['spearman']['rho']

        if test_rho > best_rho:
            best_rho=test_rho
            best_rho_config = {configuration_set[1][:-1]: current_config}
            hist_rho_best = os.path.join(os.path.split(grid_search_file)[0], configuration_set[1][:-1], 'hist_res.p')
        if test_mse+train_mse < best_mse and train_rho>0 and test_rho>0:
            best_mse = test_mse+train_mse
            best_mse_config = {configuration_set[1][:-1]: current_config}
            hist_mse_best = os.path.join(os.path.split(grid_search_file)[0],configuration_set[1][:-1], 'hist_res.p')

        total_test_rho.append(test_rho)
        total_test_mse.append(test_mse)
        count+=1
    except:
        pass
    configuration_stats[configuration_set[1][:-1]]=current_config
    idx+=6

total_test_rho = np.array(total_test_rho)
total_test_mse = np.array(total_test_mse)
print(f'number of records = {total_test_rho.shape[0]}')
print(f'{total_test_rho.mean().round(3)} +- {total_test_rho.std().round(3)}')
print(f'{total_test_mse.mean().round(3)} +- {total_test_mse.std().round(3)}')
str = 'Best Spearman rho'
print(str)
print_recursive(best_rho_config)
with open(hist_rho_best, 'rb') as f:
    a=pickle.load(f)
plt.figure(str + '\n' + list(best_rho_config.keys())[0])
plt.plot(a['loss'])
plt.title('Train loss vs epoch of the best Spearman rho on test\n' + list(best_rho_config.keys())[0])

str = 'Best Test+Train MSE'
print('\n'+str)
print_recursive(best_mse_config)
with open(hist_mse_best, 'rb') as f:
    a=pickle.load(f)
plt.figure(str + '\n' + list(best_rho_config.keys())[0])
plt.plot(a['loss'])
plt.title('Train loss vs epoch of the best MSE on test\n' + list(best_mse_config.keys())[0])
#
# plt.show()