from Projects.GVHD_BAR.analyze_grid_search import get_stats_for_array
import os
import numpy as np
import matplotlib.pyplot as plt
from Projects.GVHD_BAR.show_data import calc_results_and_plot
from Projects.GVHD_BAR.analyze_grid_search import get_stats_for_array
from matplotlib.lines import Line2D
# remove_outliers

def main(folder_to_use, cv_title=None, epoch_to_use=None, verbose_per_cv=False):
    configuration_stats, different_configs, all_train_rho, all_test_rho = calc_stats_for_config(folder_to_use)

    for idx, (test_iter_cv_iter, value) in enumerate(configuration_stats.items()):
        if idx % len(configuration_stats) == 0:
            fig, ax = plt.subplots(2, 2)
        train_rho_values = [x['rho'] for x in value['raw_stats']['total_train_rho']]
        test_rho_values = [x['rho'] for x in value['raw_stats']['total_test_rho']]
        train_mse_values = [x for x in value['raw_stats']['total_train_mse']]
        test_mse_values = [x for x in value['raw_stats']['total_test_mse']]
        epochs_list = value['epoch_list']
        ax[0][0].scatter(epochs_list, train_rho_values, label=f'Train_cv_{idx % 10}', s=2)
        ax[0][0].set_title('Train Spearman RHO')
        ax[0][0].legend()

        ax[1][0].scatter(epochs_list, test_rho_values, label=f'Test_cv_{idx % 10}', s=2)
        ax[1][0].set_title('Test Spearman RHO')
        ax[1][0].legend()

        ax[0][1].scatter(epochs_list, train_mse_values, label=f'Train_cv_{idx % 10}', s=2)
        ax[0][1].set_title('Train MSE')
        ax[0][1].legend()

        ax[1][1].scatter(epochs_list, test_mse_values, label=f'Test_cv_{idx % 10}', s=2)
        ax[1][1].set_title('Test MSE')
        ax[1][1].legend()

    calc_full_rho(configuration_stats, epoch_to_use, verbose_per_cv)

    vis_predict(configuration_stats, cv_title, epoch_to_use)

def calc_stats_for_config(grid_search_folder):
    different_configs = [os.path.join(grid_search_folder, o) for o in os.listdir(grid_search_folder)
                         if os.path.isdir(os.path.join(grid_search_folder, o))]

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
        epoch_list = []
        count = 0
        number_of_runs = int(len(grid_search_data) / 5)
        if number_of_runs >= 1:
            for idx in range(0, number_of_runs):
                configuration_set = grid_search_data[idx * 5:(idx + 1) * 5]
                try:
                    epoch = configuration_set[0][:-1]
                    epoch_list.append(int(epoch.replace('Epoch','')))
                    current_config = {epoch: {configuration_set[1][1:-1]: eval(configuration_set[2][1:-1])}}
                    current_config[epoch].update({configuration_set[3][:-1]: eval(configuration_set[4][1:-1])})
                    test_mse = current_config[epoch]['Test']['mse']
                    train_mse = current_config[epoch]['Train']['mse']
                    test_rho = current_config[epoch]['Test']['spearman']
                    train_rho = current_config[epoch]['Train']['spearman']
                    test_pearson_rho = current_config[epoch]['Test']['pearson']
                    train_pearson_rho = current_config[epoch]['Train']['pearson']

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
                idx += 5

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
                'Total number of records in config': len(total_test_rho),
                'epoch_list': epoch_list
            }

    return configuration_stats, different_configs, all_train_rho, all_test_rho






# plt.show()


def get_data_from_last_epoch(test_iter_cv_iter, epochs_list, epoch=None):
    actual_path = os.path.join(folder, test_iter_cv_iter)
    # epochs_list = value['epoch_list']
    last_epoch = int(np.ceil((epochs_list[-1]-10)/5)*5) if epoch is None else epoch
    print(f'best epoch - {epochs_list[-1]-10} Using Epoch {last_epoch}')
    test_predicted = np.load(os.path.join(actual_path, f'y_test_predicted_values_epoch_{last_epoch}.npy'))
    test_real = np.load(os.path.join(actual_path, f'y_test_values_epoch_{last_epoch}.npy'))
    train_predicted = np.load(os.path.join(actual_path, f'y_train_predicted_values_epoch_{last_epoch}.npy'))
    train_real = np.load(os.path.join(actual_path, f'y_train_values_epoch_{last_epoch}.npy'))
    return test_real, test_predicted, train_real, train_predicted


def calc_full_rho(configuration_stats, epoch_to_use=None, verbose_per_cv=False):
    total_test_values = []
    total_test_predicted_values =[]
    total_train_values = []
    total_train_predicted_values =[]

    total_train_rho = []
    total_train_mse = []
    totral_train_pearson_rho = []
    total_test_rho = []
    total_test_mse = []
    total_test_pearson_rho = []



    for idx, (test_iter_cv_iter, value) in enumerate(configuration_stats.items()):
        print(test_iter_cv_iter)
        test_real, test_predicted, train_real, train_predicted = get_data_from_last_epoch(test_iter_cv_iter, value['epoch_list'], epoch_to_use)
        total_test_values+= test_real.tolist()
        total_test_predicted_values+= test_predicted.tolist()
        total_train_values += train_real.tolist()
        total_train_predicted_values+= train_predicted.tolist()

        current_train, current_test = calc_results_and_plot(train_real,
                                                                    train_predicted,
                                                                    test_real,
                                                                    test_predicted,
                                                                    algo_name='XGBoost',
                                                                    visualize=False,
                                                                    title='',
                                                                    show=False)
        total_train_rho.append(value['train_rho']['rho']['mean'])
        total_train_mse.append(value['train_mse']['mse']['mean'])
        totral_train_pearson_rho.append(value['train_pearson_rho']['rho']['mean'])
        total_test_rho.append(value['test_rho']['rho']['mean'])
        total_test_mse.append(value['test_mse']['mse']['mean'])
        total_test_pearson_rho.append(value['test_pearson_rho']['rho']['mean'])

        if verbose_per_cv:
            print(f'\tTrain\n\t\t{current_train}')
            print(f'\tTest\n\t\t{current_test}')

        if (idx+1)%len(configuration_stats)==0:
            current_train_res, current_test_res = calc_results_and_plot(total_train_values,
                                                                        total_train_predicted_values,
                                                                        total_test_values,
                                                                        total_test_predicted_values,
                                                                        algo_name='XGBoost',
                                                                        visualize=False,
                                                                        title='',
                                                                                show=False)
            print(f'Total RHO')
            print(f'\tTrain\n\t\t{current_train_res}')
            print(f'\tTest\n\t\t{current_test_res}')
            total_test_values = []
            total_test_predicted_values = []
            total_train_values = []
            total_train_predicted_values = []
            if verbose_per_cv:
                print(f'\n\tTrain\n\t\t')
                print(f'rho: {get_stats_for_array(np.array(total_train_rho))}')
                print(f'mse: {get_stats_for_array(np.array(total_train_mse))}')
                print(f'pearson: {get_stats_for_array(np.array(totral_train_pearson_rho))}')

                print(f'\tTest\n\t\t')
                print(f'rho: {get_stats_for_array(np.array(total_test_rho))}')
                print(f'mse: {get_stats_for_array(np.array(total_test_mse))}')
                print(f'pearson: {get_stats_for_array(np.array(total_test_pearson_rho))}')
                print(f'\n')

def vis_predict(configuration_stats, title=None , epoch_to_use=None):
    iter=0
    fig_total, ax_total = plt.subplots(1)
    fig_total.suptitle(f'{title}')
    total_train_real = []
    total_train_predicted = []
    total_test_real = []
    total_test_predicted = []
    for idx, (test_iter_cv_iter, value) in enumerate(configuration_stats.items()):

        test_real, test_predicted, train_real, train_predicted = get_data_from_last_epoch(test_iter_cv_iter, value['epoch_list'], epoch_to_use)
        if idx % len(configuration_stats) == 0:
            fig, ax = plt.subplots(2, len(configuration_stats))
            fig.suptitle(f'{title}')
        ax[0][idx % len(configuration_stats)].scatter(train_real, train_predicted, s=6)
        ax[0][idx % len(configuration_stats)].set_title('Train - Predicted vs Real', fontsize=6)
        ax[0][idx % len(configuration_stats)].set_xlabel('Real', fontsize=6)
        ax[0][idx % len(configuration_stats)].set_ylabel('Predicted', fontsize=6)
        ax[0][idx % len(configuration_stats)].yaxis.set_label_coords(0, 0.5)
        ax[1][idx % len(configuration_stats)].scatter(test_real, test_predicted, s=10)
        ax[1][idx % len(configuration_stats)].set_title('Test - Predicted vs Real', fontsize=6)
        ax[1][idx % len(configuration_stats)].set_xlabel('Real', fontsize=8)
        ax[1][idx % len(configuration_stats)].set_ylabel('Predicted', fontsize=8)
        ax[1][idx % len(configuration_stats)].yaxis.set_label_coords(0, 0.5)
        total_train_real.append(train_real)
        total_train_predicted.append(train_predicted)
        total_test_real.append(test_real)
        total_test_predicted.append(test_predicted)



    custom_lines = [Line2D([0], [0], marker='o', color='w', label='Train',
                          markerfacecolor='r'),
                    Line2D([0], [0], marker='o', color='w', label='Test',
                          markerfacecolor='b')]

    ax_total.legend(handles=custom_lines)
    ax_total.set_title('Test - Predicted vs Real')
    ax_total.set_xlabel('Real')
    ax_total.set_ylabel('Predicted')

    train_y = np.array(total_train_real).flatten()
    train_y_predicted = np.array(total_train_predicted).flatten()
    test_y = np.array(total_test_real).flatten()
    test_y_predicted = np.array(total_test_predicted).flatten()

    if train_y[0].shape is not ():
        total=[]
        for a in train_y:
            total += a.tolist()
        train_y = np.array(total)

        total=[]
        for a in train_y_predicted:
            total += a.tolist()
        train_y_predicted = np.array(total)

    if test_y[0].shape is not ():
        total = []
        for a in test_y:
            total += a.tolist()
            test_y = np.array(total)

        total = []
        for a in test_y_predicted:
            total += a.tolist()
        test_y_predicted = np.array(total)

    mask = remove_outliers(train_y)
    train_y = train_y[mask]
    train_y_predicted = train_y_predicted[mask]

    mask = remove_outliers(train_y_predicted)
    train_y = train_y[mask]
    train_y_predicted = train_y_predicted[mask]

    mask = remove_outliers(test_y)
    test_y = test_y[mask]
    test_y_predicted = test_y_predicted[mask]

    mask = remove_outliers(test_y_predicted)
    test_y = test_y[mask]
    test_y_predicted = test_y_predicted[mask]

    ax_total.scatter(train_y, train_y_predicted, s=10, color='red')
    ax_total.scatter(test_y, test_y_predicted, s=10, color='blue')

    x_regressor, y_regressor = get_regressor_input(train_y, train_y_predicted, precenteage_for_regressor=20)
    x_pred, y_pred, lower, upper, prediction_lower, prediction_upper = fit_line_with_confidence(x_regressor,
                                                                                                y_regressor,
                                                                                                intercept=True)
    ax_total.plot(x_pred, y_pred, '-', color='darkred', linewidth=2)
    ax_total.fill_between(x_pred, lower, upper, color='lightcoral', alpha=0.6)
    ax_total.fill_between(x_pred, prediction_lower, prediction_upper, color='lightcoral', alpha=0.2)

    x_regressor, y_regressor = get_regressor_input(test_y, test_y_predicted, precenteage_for_regressor=20)
    x_pred, y_pred, lower, upper, prediction_lower, prediction_upper = fit_line_with_confidence(x_regressor,
                                                                                                y_regressor,
                                                                                                intercept=True)
    ax_total.plot(x_pred, y_pred, '-', color='navy', linewidth=2)
    ax_total.fill_between(x_pred, lower, upper, color='cornflowerblue', alpha=0.6)
    ax_total.fill_between(x_pred, prediction_lower, prediction_upper, color='cornflowerblue', alpha=0.2)


    plt.tight_layout()


# folder = r'C:\Users\Bar\Desktop\testing\gvhd_FNN_best_config_iter_0\l2=1^dropout=0.2^factor=1^epochs=1000^number_iterations=10^number_layers=2^neurons_per_layer=50'
# main(folder, cv_title='GVHD FNN ', epoch_to_use=None, verbose_per_cv=True)
#

# folder = r'C:\Users\Bar\Desktop\testing\allergy_FNN_best_config_iter_0\l2=20^dropout=0.6^factor=1^epochs=1000^number_iterations=10^number_layers=1^neurons_per_layer=20'
# main(folder, cv_title='Allergy FNN ', epoch_to_use=None, verbose_per_cv=True)

# folder = r'C:\Users\Bar\Desktop\testing\gvhd_FNN_TS_best_config_iter_0\CVS'
# main(folder, cv_title='GVHD FNN TS', epoch_to_use=None, verbose_per_cv=True)

# folder = r'C:\Users\Bar\Desktop\testing\allergy_FNN_TS_again_best_config_iter_0\l2=100^dropout=0.6^factor=1000^epochs=1000^number_iterations=10^number_layers=2^neurons_per_layer=20'
# main(folder, cv_title='Allergy FNN TS', epoch_to_use=None, verbose_per_cv=True)

# folder = r'Z:\allergy_FNN_TS_SIM_best_config_iter_0\CVS'
# main(folder, cv_title='Allergy FNN TS SIM', epoch_to_use=None, verbose_per_cv=True)

# folder = r'C:\Users\Bar\Desktop\testing\gvhd_FNN_TS_SIM_best_config_iter_0\CVS'
# main(folder, cv_title='GVHD FNN TS SIM', epoch_to_use=None, verbose_per_cv=True)

folder = r'C:\Users\Bar\Desktop\testing\gvhd_lstm_best\l2=1^dropout=0^factor=1^epochs=1000^number_iterations=20^number_layers=1^neurons_per_layer=10'
# main(folder, cv_title='GVHD LSTM', epoch_to_use=None, verbose_per_cv=True)
#

# folder = r'Z:\allergy_best_lstm\l2=10^dropout=0^factor=1^epochs=1000^number_iterations=5^number_layers=2^neurons_per_layer=30'
# main(folder, cv_title='Allergy LSTM', epoch_to_use=None, verbose_per_cv=True)


folder = r'Z:\gvhd_lstm_TS_best_config_iter_1\l2=100^dropout=0^factor=100^epochs=1000^number_iterations=40^number_layers=3^neurons_per_layer=30'
# main(folder, cv_title='GVHD LSTM TS', epoch_to_use=None, verbose_per_cv=True)

folder = r'z:\gvhd_lstm_TS_sim_best_config_iter_0\l2=100^dropout=0^factor=100^epochs=1000^number_iterations=30^number_layers=1^neurons_per_layer=30^censored_mse_factor=50.0^beta_for_similarity=5'
# main(folder, cv_title='GVHD LSTM TS SIM', epoch_to_use=None, verbose_per_cv=True)

folder = r'Z:\allergy_lstm_TS_best_config_right_iter_0\l2=100^dropout=0^factor=100^epochs=1000^number_iterations=30^number_layers=3^neurons_per_layer=30\CVS'
folder = r'C:\Users\Bar\Desktop\testing\allergy_lstm_TS_best_config_right_pls_iter_0\l2=100^dropout=0^factor=100^epochs=1000^number_iterations=30^number_layers=1^neurons_per_layer=30'
# main(folder, cv_title='Allergy LSTM TS', epoch_to_use=None, verbose_per_cv=True)

folder =r'z:\allergy_lstm_TS_sim_best_conf_pls_iter_0\CVS'
# main(folder, cv_title='Allergy LSTM TS SIM', epoch_to_use=None, verbose_per_cv=True)


plt.show()

