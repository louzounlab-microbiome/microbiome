import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.lines import Line2D
from shutil import copyfile, rmtree
use_hist = False

# # grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_my_loss_corrected\grid_search_results.txt'
# grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_mse_loss\grid_search_results.txt'
# grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_mse_loss\grid_search_results.txt'
# grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_my_loss_factor_layers_neurons\grid_search_results.txt'
# grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_my_loss_fine\grid_search_results.txt'
# grid_search_file = r'C:\Thesis\subject_test_100_processes_true\grid_search_results.txt'
# grid_search_file = r'C:\Thesis\subject_test_100_processes_mse_false\grid_search_results.txt'
# grid_search_file = r'C:\Thesis\subject_test_100_processes_myloss_USE_SUBJECT_ONLY_TEST_TRAIN=True\grid_search_results.txt'
# grid_search_file = r'C:\Thesis\GVHD_BAR\GridSearch_combine_algo_new_loss_epochs_dropout_l2_numberlayers_numberneurons\grid_search_results.txt'
# grid_search_file = r'C:\Thesis\GridSearch_combine_algo_new_loss_50_times\grid_search_results.txt'
# grid_search_file = r'C:\Thesis\allergy\GridSearch_epochs_factors_dropout_l2_layers_neurons\grid_search_results.txt'
# grid_search_file = r'C:\Thesis\allergy\grid_search_no_censored\grid_search_results.txt'
#
# grid_search_folder = r'C:\Thesis\gvhd_multi_grid_tf_without_censored'
# grid_search_file = os.path.join(grid_search_folder, 'grid_search_results.txt')
#
#
# grid_search_folder = r'C:\Thesis\multi_process_analysis\allergy_multi_grid_xgboost_with_similiarity'
# grid_search_folder = r'C:\Thesis\multi_process_analysis\allergy_multi_grid_xgboost_without_similiarity'
# grid_search_folder = r'C:\Thesis\multi_process_analysis\gvhd_multi_grid_xgboost_with_similiarity'
grid_search_folder = r'C:\Thesis\multi_process_analysis\gvhd_multi_grid_xgboost_without_similiarity'


# grid_search_folder = r'C:\Thesis\gvhd_multi_grid_xgboost_with_similiarity'
# grid_search_folder = r'C:\Thesis\allergy_multi_grid_xgboost_with_similiarity'
# grid_search_folder = r'C:\Thesis\allergy_multi_grid_xgboost_without_similiarity'
# grid_search_folder = r'C:\Thesis\gvhd_multi_grid_xgboost_without_similiarity'

def print_recursive(object_to_print, count=0):
    spaces = count * ' '
    if type(object_to_print) is dict:
        count += 1
        for key, val in object_to_print.items():
            if key in ['raw_stats', 'hist']:
                continue
            if key in ['rho', 'pval', 'mse']:
                print(f'{spaces}{key}: {val}')
            else:
                if 'best_' in key:
                    print(f'\n{spaces}{key}')
                else:
                    print(f'{spaces}{key}')

                print_recursive(val, count=count)

    else:
        # spaces = (count-1) * '\t'+ ' '
        print(f'{spaces}{object_to_print}')


def print_recursive2(object_to_print, count=0):
    spaces = count * ' '
    str_to_return =''
    if type(object_to_print) is dict:
        count += 1
        for key, val in object_to_print.items():
            if key in ['raw_stats', 'hist']:
                continue
            if key in ['rho', 'pval', 'mse']:
                str_to_return += f'{spaces}{key}: {val}\n'
            else:
                if 'best_' in key:
                    str_to_return += f'\n{spaces}{key}\n'
                else:
                    str_to_return += f'{spaces}{key}\n'

                try:
                    str_to_return += print_recursive2(val, count=count)
                except:
                    pass
    else:
        str_to_return += f'{spaces}{object_to_print}\n'
    return str_to_return

def get_stats_for_array(array):
    return {'mean': array.mean(), 'std': array.std(), 'max': array.max(), 'min': array.min(),
            'median': np.median(array)}

def filter_configs(grid_search_folder, filter_dict=None):
    different_configs = [os.path.join(grid_search_folder, o) for o in os.listdir(grid_search_folder)
                         if os.path.isdir(os.path.join(grid_search_folder, o))]
    if filter_dict is None:
        return different_configs

    filtered_configs = []
    for config in different_configs:
        config_name = os.path.basename(config)
        parameters = config_name.split('^')
        add_flag = True
        for key, values in filter_dict.items():
            cur_filter_flag = False
            for val in values:
                cur_filter = f'{key}={val}'
                if cur_filter in parameters:
                    cur_filter_flag = True
            if not cur_filter_flag:
                add_flag = False

        if add_flag:
            filtered_configs.append(config)
    return filtered_configs


def calc_result_per_param(different_configs):

    configuration_results = {}
    print(f'Total different configs: {len(different_configs)}')
    for config in different_configs:
        grid_search_file = os.path.join(config, 'grid_search_results.txt')
        config_name = os.path.basename(config)
        parameters = config_name.split('^')

        with open(grid_search_file, 'r') as f:
            grid_search_data = f.readlines()

        total_test_pearson_rho = []
        total_test_mse = []
        total_test_rho = []

        total_train_pearson_rho = []
        total_train_mse = []
        total_train_rho = []

        count = 0
        for idx in range(0, int(len(grid_search_data) / 4)):
            configuration_set = grid_search_data[idx * 4:(idx + 1) * 4]
            try:
                current_config = {configuration_set[0][:-1]: eval(configuration_set[1][1:-1])}
                current_config.update({configuration_set[2][:-1]: eval(configuration_set[3][1:-1])})
                test_mse = current_config['Test']['mse']
                train_mse = current_config['Train']['mse']
                test_rho = current_config['Test']['spearman']
                train_rho = current_config['Train']['spearman']
                test_pearson_rho = current_config['Test']['pearson']
                train_pearson_rho = current_config['Train']['pearson']

                total_test_rho.append(test_rho)
                total_test_mse.append(test_mse)
                total_test_pearson_rho.append(test_pearson_rho)

                total_train_rho.append(train_rho)
                total_train_mse.append(train_mse)
                total_train_pearson_rho.append(train_pearson_rho)

                count += 1
            except:
                pass
            idx += 4

        for param in parameters:
            param_name_val = param.split('=')
            param_name = param_name_val[0]
            param_val = param_name_val[1]
            if param_name not in configuration_results.keys():
                configuration_results[param_name] = {}
            if param_val not in configuration_results[param_name].keys():
                configuration_results[param_name][param_val] = {}
                configuration_results[param_name][param_val]['test_rho'] = []
                configuration_results[param_name][param_val]['test_mse'] = []
                configuration_results[param_name][param_val]['test_pearson_rho'] = []
                configuration_results[param_name][param_val]['train'] = []
                configuration_results[param_name][param_val]['train_mse'] = []
                configuration_results[param_name][param_val]['train_pearson_rho'] = []

            configuration_results[param_name][param_val]['test_rho']+= total_test_rho
            configuration_results[param_name][param_val]['test_mse']+= total_test_mse
            configuration_results[param_name][param_val]['test_pearson_rho'] += total_test_pearson_rho
            configuration_results[param_name][param_val]['train'] += total_train_rho
            configuration_results[param_name][param_val]['train_mse'] += total_train_mse
            configuration_results[param_name][param_val]['train_pearson_rho'] += total_train_pearson_rho
        pass
    return configuration_results

def plot_per_param\
                (configuration_results, save=False, plot=True, path='', show_one_only=False):
    for param_key, param_values in configuration_results.items():
        if not show_one_only and len(param_values) == 1:
            continue
        fig, axis = plt.subplots(2, len(list(param_values.values())[0].keys())//2)
        fig.suptitle(f'{param_key}', fontsize=16)

        tmp = {}

        for param_value, result_for_param_value in param_values.items():
            for metric, metric_values in result_for_param_value.items():
                values_of_param = []
                for metric_value in metric_values:
                    if isinstance(metric_value, dict): # rho
                        value_to_add = metric_value['rho']
                    else: # mse
                        value_to_add = metric_value
                    values_of_param.append(value_to_add)
                if metric not in tmp.keys():
                    tmp[metric] = [values_of_param]
                else:
                    tmp[metric].append(values_of_param)
            pass
        for ax, (key, val) in zip (axis.flatten(), tmp.items()):
            ax.boxplot(val)
            ax.set_title(key)
            ax.set_xticklabels(list(param_values.keys()))

        if save:
            folder_to_save = os.path.join('C:\\','ParameterChoosing')
            if not os.path.exists(folder_to_save):
                os.mkdir(folder_to_save)

            folder_to_save = os.path.join(folder_to_save,os.path.basename(path) )
            if not os.path.exists(folder_to_save):
                os.mkdir(folder_to_save)


            # fig.savefig(os.path.join(folder_to_save,f'{param_key}.png'))
            # plt.close(fig)

    if plot:
        plt.show()
        plt.tight_layout()

def calc_stats_for_config_new(different_configs):

    configuration_stats = {}
    print(f'Total different configs: {len(different_configs)}')
    all_train_rho = []
    all_test_rho = []
    for config in different_configs:
        grid_search_file = os.path.join(config, 'grid_search_results.txt')
        hist_file = os.path.join(config, 'hist_res.p')
        config_name = os.path.basename(config)
        with open(grid_search_file, 'r') as f:
            grid_search_data = f.readlines()

        total_test_pearson_rho = []
        total_test_mse = []
        total_test_rho = []

        total_train_pearson_rho = []
        total_train_mse = []
        total_train_rho = []

        count = 0
        number_of_runs = int(len(grid_search_data) / 4)
        if number_of_runs >= 1:
            for idx in range(0, number_of_runs):
                configuration_set = grid_search_data[idx * 4:(idx + 1) * 4]
                try:
                    current_config = {configuration_set[0][:-1]: eval(configuration_set[1][1:-1])}
                    current_config.update({configuration_set[2][:-1]: eval(configuration_set[3][1:-1])})
                    test_mse = current_config['Test']['mse']
                    train_mse = current_config['Train']['mse']
                    test_rho = current_config['Test']['spearman']
                    train_rho = current_config['Train']['spearman']
                    test_pearson_rho = current_config['Test']['pearson']
                    train_pearson_rho = current_config['Train']['pearson']

                    total_test_rho.append(test_rho)
                    total_test_mse.append(test_mse)
                    total_test_pearson_rho.append(test_pearson_rho)

                    total_train_rho.append(train_rho)
                    total_train_mse.append(train_mse)
                    total_train_pearson_rho.append(train_pearson_rho)

                    count += 1

                    all_train_rho.append(train_rho)
                    all_test_rho.append(test_rho)

                except:
                    pass
                idx += 4


            # get_stats_for_array([x['rho'] for x in total_train_rho])
            configuration_stats[config_name] = {
                'raw_stats': {'total_train_rho': total_train_rho, 'total_train_mse': total_train_mse,
                              'total_test_rho': total_test_rho, 'total_test_mse': total_test_mse},
                'test_rho': {'rho': get_stats_for_array(
                    np.array([x['rho'] for x in total_test_rho])),
                    'pval': get_stats_for_array(
                        np.array([x['pvalue'] for x in total_test_rho]))},
                'test_mse': {'mse': get_stats_for_array(np.array(total_test_mse))},
                'test_pearson_rho': {'rho': get_stats_for_array(
                    np.array([x['rho'] for x in total_test_pearson_rho])),
                    'pval': get_stats_for_array(np.array(
                        [x['pvalue'] for x in total_test_pearson_rho]))},

                'train_rho': {'rho': get_stats_for_array(np.array([x['rho'] for x in total_train_rho])),
                              'pval': get_stats_for_array(np.array([x['pvalue'] for x in total_train_rho]))},
                'train_mse': {'mse': get_stats_for_array(np.array(total_train_mse))},
                'train_pearson_rho': {'rho': get_stats_for_array(np.array([x['rho'] for x in total_train_pearson_rho])),
                                      'pval': get_stats_for_array(
                                          np.array([x['pvalue'] for x in total_train_pearson_rho]))},
                'hist': hist_file,
                'Total number of records in config': len(total_test_rho)
                }

    return configuration_stats, different_configs ,all_train_rho, all_test_rho

def calc_stats_for_config(grid_search_folder):
    different_configs = [os.path.join(grid_search_folder, o) for o in os.listdir(grid_search_folder)
                         if os.path.isdir(os.path.join(grid_search_folder, o))]

    return calc_stats_for_config_new(different_configs)


def find_best_config(configuration_stats):
    best_test_rho = 0
    best_combined_mse = 99999999999
    best_test_mse = 99999999999
    all_train_rho = []
    all_test_rho = []
    for key, val in configuration_stats.items():
        train_rho = val['train_rho']['rho']['mean']
        train_mse = val['train_mse']['mse']['mean']

        test_rho = val['test_rho']['rho']['mean']
        test_mse = val['test_mse']['mse']['mean']



        if test_rho > best_test_rho:
            best_test_rho = test_rho
            best_test_rho_config = key

        if test_mse < best_test_mse and train_rho > 0 and test_rho > 0:
            best_test_mse = test_mse
            best_test_mse_config = key

        if test_mse + train_mse < best_combined_mse and train_rho > 0 and test_rho > 0:
            best_combined_mse = test_mse + train_mse
            best_combined_mse_config = key

    def get_stats_for_best_config(config):
        test_mse = configuration_stats[config]['test_mse']
        test_rho = configuration_stats[config]['test_rho']
        train_mse = configuration_stats[config]['test_mse']
        train_rho = configuration_stats[config]['rain_rho']
        return {'train_rho': train_rho, 'train_mse': train_mse, 'test_rho': test_rho, 'test_mse': test_mse}

    best_configs = {
        'best_test_rho_config': {'name': best_test_rho_config, 'stats': configuration_stats[best_test_rho_config]},
        'best_test_mse_config': {'name': best_test_mse_config, 'stats': configuration_stats[best_test_mse_config]},
        'best_combined_mse_config': {'name': best_combined_mse_config,
                                     'stats': configuration_stats[best_combined_mse_config]}}

    return best_configs#, all_train_rho, all_test_rho


def main(grid_search_folder):
    print(f'Analyzing {grid_search_folder}')
    configuration_stats, different_configs, all_train_rho, all_test_rho = calc_stats_for_config(grid_search_folder)
    best_configs = find_best_config(configuration_stats)

    ## use history... currently not using it...

    with open(best_configs['best_test_rho_config']['stats']['hist'], 'rb') as f:
        a = pickle.load(f)

    a=print_recursive2(best_configs)
    print(a)
    b= f'Number of configurations: {len(different_configs)}'
    b+='\n'+a


    if True:
        legend={'blue': [False,20], 'red': [False, 1000],  'green': [False, 100], 'orange': [False, 80],'black':[False, 300] ,'purple': [False, '']}
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()

        # ax[0].hist([x['rho'] for x in all_train_rho], bins=100)
        # ax[1].hist([x['rho'] for x in all_test_rho], bins=100)

        all_train_rho_centered = []
        all_test_rho_centered = []
        for key, value in configuration_stats.items():
            if 'epochs=20^' in key:
                color_to_use='blue'

            elif 'epochs=100^' in key:
                color_to_use='green'
            elif 'epochs=1000^' in key:
                color_to_use='red'
            elif 'epochs=80^' in key:
                color_to_use = 'orange'
            elif 'epochs=300^' in key:
                color_to_use = 'black'
            else:
                color_to_use='purple'
                # print(key)
            legend[color_to_use][0]=True

            mean_train_value = value['train_rho']['rho']['mean']
            mean_test_value = value['test_rho']['rho']['mean']
            std_test_value = value['test_rho']['rho']['std']
            mse_mean_test_value = value['test_mse']['mse']['mean']
            mse_std_test_value = value['test_mse']['mse']['std']
            ax3.scatter(mse_mean_test_value, mean_test_value,color=color_to_use)
            ax3.set_xlabel('mean mse')
            ax3.set_ylabel('mean rho')
            ax3.set_title('Test - mean rho vs mean mse')



            ax4.scatter(mse_mean_test_value, mse_std_test_value,color=color_to_use)
            ax4.set_xlabel('mean')
            ax4.set_ylabel('std')
            ax4.set_title('Test - MSE std vs mean MSE')

            ax.scatter(mean_test_value, std_test_value,color=color_to_use)

            mean_test_value = value['train_rho']['rho']['mean']
            std_test_value = value['train_rho']['rho']['std']
            ax2.scatter(mean_test_value, std_test_value, color=color_to_use)

            test_rho_values = [x['rho']for x in value['raw_stats']['total_test_rho']]
            #
            # all_test_rho_centered = [x-mean_test_value for x in test_rho_values]
            #
            # mean_train_value = value['train_rho']['rho']['mean']
            train_rho_values = [x['rho'] for x in value['raw_stats']['total_train_rho']]

            ax5.scatter(test_rho_values, train_rho_values, color=color_to_use)
            ax5.set_xlabel('test rho')
            ax5.set_ylabel('train rho')
            ax5.set_title('test rho vs train rho')

            # all_train_rho_centered = [x-mean_train_value for x in train_rho_values]
            #
            # fig, ax = plt.subplots(2)
            # fig.suptitle('RHO')
            # ax[0].hist(train_rho_values, bins=100)
            # ax[1].hist(test_rho_values, bins=100)
            #
            # fig, ax = plt.subplots(2)
            # fig.suptitle('RHO-<RHO>')
            # ax[0].hist(all_train_rho_centered, bins=100)
            # ax[1].hist(all_test_rho_centered, bins=100)

            mean_test_value_pearson = value['test_pearson_rho']['rho']['mean']
            ax6.scatter(mse_mean_test_value, mean_test_value_pearson,color=color_to_use)
            ax6.set_xlabel('mean mse')
            ax6.set_ylabel('mean pearson rho')
            ax6.set_title('Test - mean pearson rho vs mean mse')

        legend_elements = []

        def add_legend(legend_elements, color):
            if legend[color][0]:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend[color][1],
                                              markerfacecolor=color, markersize=7))
            return legend_elements

        legend_elements=add_legend(legend_elements,'blue')
        legend_elements=add_legend(legend_elements,'green')
        legend_elements=add_legend(legend_elements,'red')
        legend_elements=add_legend(legend_elements,'orange')
        legend_elements=add_legend(legend_elements,'black')
        legend_elements=add_legend(legend_elements,'purple')

        ax.set_xlabel('mean')
        ax.set_ylabel('std')
        ax.set_title('Test - std rho vs mean rho')
        ax.legend(handles=legend_elements)
        ax2.set_xlabel('mean')
        ax2.set_ylabel('std')
        ax2.set_title('Train - std rho vs mean rho')
        ax2.legend(handles=legend_elements)
        ax3.legend(handles=legend_elements)
        ax4.legend(handles=legend_elements)

    plt.show()
    return b


def old_analyze():
    with open(grid_search_file, 'r') as f:
        grid_search_data = f.readlines()

    configuration_stats = {}
    best_mse = 10000
    best_config = ''
    best_mse_config = ''
    best_rho = -100
    hist_mse_best = None
    hist_rho_best = None
    best_rho_config = ''
    best_mse_config = ''
    total_test_rho = []
    total_test_mse = []
    count = 0
    for idx in range(0, int(len(grid_search_data) / 6)):
        configuration_set = grid_search_data[idx * 6:(idx + 1) * 6]
        current_config = ''
        try:
            current_config = {configuration_set[2][:-1]: eval(configuration_set[3][1:-1])}
            current_config.update({configuration_set[4][:-1]: eval(configuration_set[5][1:-1])})
            test_mse = current_config['Test']['mse']
            train_mse = current_config['Train']['mse']
            test_rho = current_config['Test']['spearman']['rho']
            train_rho = current_config['Train']['spearman']['rho']

            if test_rho > best_rho:
                best_rho = test_rho
                best_rho_config = {configuration_set[1][:-1]: current_config}
                hist_rho_best = os.path.join(os.path.split(grid_search_file)[0], configuration_set[1][:-1],
                                             'hist_res.p')
            if test_mse + train_mse < best_mse and train_rho > 0 and test_rho > 0:
                best_mse = test_mse + train_mse
                best_mse_config = {configuration_set[1][:-1]: current_config}
                hist_mse_best = os.path.join(os.path.split(grid_search_file)[0], configuration_set[1][:-1],
                                             'hist_res.p')

            total_test_rho.append(test_rho)
            total_test_mse.append(test_mse)
            count += 1
        except:
            pass
        configuration_stats[configuration_set[1][:-1]] = current_config
        idx += 6

    total_test_rho = np.array(total_test_rho)
    total_test_mse = np.array(total_test_mse)
    print(f'number of records = {total_test_rho.shape[0]}')
    print(f'{total_test_rho.mean().round(3)} +- {total_test_rho.std().round(3)}')
    print(f'{total_test_mse.mean().round(3)} +- {total_test_mse.std().round(3)}')
    str = 'Best Spearman rho'
    print(str)
    print_recursive(best_rho_config)

    if use_hist:
        with open(hist_rho_best, 'rb') as f:
            a = pickle.load(f)
        plt.figure(str + '\n' + list(best_rho_config.keys())[0])
        plt.plot(a['loss'])
        plt.title('Train loss vs epoch of the best Spearman rho on test\n' + list(best_rho_config.keys())[0])

    str = 'Best Test+Train MSE'
    print('\n' + str)
    print_recursive(best_mse_config)

    if use_hist:
        with open(hist_mse_best, 'rb') as f:
            a = pickle.load(f)
        plt.figure(str + '\n' + list(best_rho_config.keys())[0])
        plt.plot(a['loss'])
        plt.title('Train loss vs epoch of the best MSE on test\n' + list(best_mse_config.keys())[0])
        #
        # plt.show()

def plot_and_save(path, save=True, plot=False):
    different_configs = filter_configs(path)
    configuration_results = calc_result_per_param(different_configs)
    plot_per_param(configuration_results, save, plot, path)


if __name__ == '__main__':
    grid_search_folder=[]


###########FNN#######
    grid_search_folder.append(r'C:\Users\Bar\Desktop\testing\GVHD_FNN_AGAIN')
    # grid_search_folder.append(r'C:\Users\Bar\Desktop\testing\alergy_FNN_Again')

    # # # # naive#
    # grid_search_folder.append(r'C:\Users\Bar\Desktop\reports\gvhd_multi_grid_xgboost_without_similiarity')
    # # # similiarity#
    # grid_search_folder.append(r'C:\Users\Bar\Desktop\reports\gvhd_multi_grid_xgboost_with_similiarity')

    # grid_search_folder.append(r'C:\Users\Bar\Desktop\testing\gvhd_lstm_naive_iter_0')


    ### combine all the runs into one run for calculations ###
    combined_folder = os.path.join(grid_search_folder[0],'combined_folder')
    if os.path.exists(combined_folder):
        rmtree(combined_folder)
        import time
        time.sleep(1)
    os.mkdir(combined_folder)
    for folder in grid_search_folder:
        run_dirs = os.listdir(folder)
        run_dirs = [x for x in run_dirs if x != 'combined_folder' and '^' not in x]
        for run_dir in run_dirs:
            configs = (filter_configs(os.path.join(folder,run_dir)))
            for config in configs:
                folder_to_save = os.path.join(combined_folder, os.path.basename(config))
                if not os.path.exists(folder_to_save):
                    os.mkdir(folder_to_save)
                    copyfile(os.path.join(config,'grid_search_results.txt'), os.path.join(folder_to_save,'grid_search_results.txt'))
                else:
                    with open(os.path.join(config,'grid_search_results.txt'), 'r') as f:
                        current_run_config_result = f.readlines()
                    with open(os.path.join(folder_to_save,'grid_search_results.txt'),'a') as f:
                        for line in current_run_config_result:
                            f.writelines(line)


    # GVHD FNN #
    different_configs = filter_configs(combined_folder,filter_dict={'l2': [1], 'dropout': [0.2], 'number_layers': [2],  'neurons_per_layer': [50]})

    # Allergy FNN #
    # different_configs = filter_configs(combined_folder,filter_dict={'l2': [20], 'dropout': [0.6], 'number_layers': [1], 'neurons_per_layer': [20]})

    configuration_results = calc_result_per_param(different_configs)

    # save = True
    # for folder in grid_search_folder:
    #     tmp = main(folder)
    #     if save:
    #         folder_to_save = os.path.join('C:\\','ParameterChoosing')
    #         if not os.path.exists(folder_to_save):
    #             os.mkdir(folder_to_save)
    #
    #         folder_to_save = os.path.join(folder_to_save,os.path.basename(folder) )
    #         if not os.path.exists(folder_to_save):
    #             os.mkdir(folder_to_save)
    #
    #         with open(os.path.join(folder_to_save,'Best_config.txt'),'w') as f:
    #             f.writelines(tmp)
    #     # plot_and_save(folder, plot=False)
    # # different_configs = filter_configs(folder, filter_dict={'l2': [1], 'number_layers':[1], 'dropout': [0.2]})
    #
    # ########## FNN + Early Stop ##############
    #     # GVHD #
    #         # FNN #
    # # different_configs = filter_configs(folder, filter_dict={'l2': [1], 'dropout': [0.2], 'neurons_per_layer': [50], 'number_layers': [2]})
    #         # FNN + TS #
    # different_configs = filter_configs(combined_folder, filter_dict={'l2': [1], 'factor': [100], 'neurons_per_layer': [50], 'dropout': [0.6], 'number_layers': [3]})
    #
    # #C:\Users\Bar\Desktop\testing\gvhd_FNN_again_iter_0\l2=1^dropout=0.2^factor=1^epochs=1000^number_iterations=5^number_layers=2^neurons_per_layer=50#
    # # , filter_dict = {'dropout': [0.6], 'factor': [10], 'neurons_per_layer': [50], 'l2': [100], 'number_layers': [2]
    # ########## END FNN + Early Stop ##############
    #
    #
    #
    #
    #
    # # #  GVHD + RNN
    # # # different_configs = filter_configs(folder, filter_dict={'neurons_per_layer': [10],   'dropout': [0.2], 'l2': [0.1]})
    # #
    # # #  GVHD + RNN + TS
    # # # different_configs = filter_configs(folder, filter_dict={'factor': [100], 'neurons_per_layer': [10], 'dropout': [0.2], 'l2': [1]})
    # #
    # # #  GVHD + RNN + TS + SIM
    # # different_configs = filter_configs(folder,filter_dict={'factor': [100], 'l2': [1], 'neurons_per_layer': [10], 'dropout': [0.3], 'beta_for_similarity': [10]})
    # #
    # # #  Allergy + RNN
    # # # different_configs = filter_configs(folder, filter_dict={'l2': [0.1], 'neurons_per_layer': [30], 'dropout': [0.3], 'number_layers': [3]})
    # #
    # # #  Allergy + RNN + TS
    # # # different_configs = filter_configs(folder, filter_dict={'factor': [100], 'dropout': [0.2],  'neurons_per_layer': [30],  'number_layers': [2], 'l2': [0.01]})
    # #
    # # #  Allergy + RNN + TS + SIM
    # # # different_configs = filter_configs(folder)
    #
    # configuration_results = calc_result_per_param(different_configs)
    #
    # configuration_results = calc_result_per_param(different_configs)
    plot_per_param(configuration_results, show_one_only=True)
    # #
    plt.show()
    config_stats = calc_stats_for_config_new(different_configs)
    print_recursive(config_stats[0])