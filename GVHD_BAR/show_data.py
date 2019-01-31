from infra_functions.general import use_spearmanr, use_pearsonr
import pickle
import os
import matplotlib.pyplot as plt
import math
import sys
from sklearn.metrics import mean_squared_error

def calc_results(y_train_dict, y_test_dict, algo_name, n_rows=3, n_cols=3, visualize=False):
    train_res = {}
    test_res = {}
    for sample_num, ((key, train), (_, test)) in enumerate(zip(y_train_dict.items(), y_test_dict.items())):
        if visualize:
            if len(y_train_dict.items()) == 1:
                n_rows=n_cols=1
                bbox_to_anchor = None
            else:
                bbox_to_anchor = (0.5, -0.65)
            subplot_idx = sample_num % (n_rows * n_cols)
            if subplot_idx == 0 and sample_num != 0:
                plt.subplots_adjust(hspace=1.1, wspace=0.8)
                plt.show()
        train_res[key] = {'spearman': use_spearmanr(train['y_train_values'], train['y_train_predicted_values']), 'pearson': use_pearsonr(train['y_train_values'], train['y_train_predicted_values'])}
        test_res[key] = {'spearman': use_spearmanr(test['y_test_values'], test['y_test_predicted_values']), 'pearson': use_pearsonr(test['y_test_values'], test['y_test_predicted_values'])}
        print('\n**********')
        print('Sample number: ', str(sample_num))
        print(key)
        print('train ' + str(train_res[key]) + ', test' + str(test_res[key]))
        spearman_train_data = train_res[key]['spearman']
        pearson_train_data = train_res[key]['pearson']
        spearman_test_data = test_res[key]['spearman']
        pearson_test_data = test_res[key]['pearson']
        train_label = 'Train\nSpearman - rho: ' + str(round(spearman_train_data['rho'], 2)) + ' pval: ' + str(
            round(spearman_train_data['pvalue'], 10)) + '\nPearson  - rho: ' + str(
            round(pearson_train_data['rho'], 2)) + ' pval: ' + str(
            round(pearson_train_data['pvalue'], 10))
        test_label = 'Test\nSpearman - rho: ' + str(round(spearman_test_data['rho'], 2)) + ' pval: ' + str(
            round(spearman_test_data['pvalue'], 10)) + '\nPearson  - rho: ' + str(
            round(pearson_test_data['rho'], 2)) + ' pval: ' + str(
            round(pearson_test_data['pvalue'], 10))

        try:
            file_name = 'results_file_'+sys.argv[1]+'.txt'
        except:
            file_name = 'results_file.txt'

        with open(file_name, 'a') as file_to_write:
            file_to_write.writelines([train_label, '\n', test_label, 2 * '\n'])
        if visualize:

            plt.subplot(n_rows, n_cols, subplot_idx + 1)

            plt.scatter(train['y_train_values'], train['y_train_predicted_values'],
                        label=train_label)



            plt.scatter(test['y_test_values'], test['y_test_predicted_values'],
                        label=test_label)
            plt.xlabel('Real Values')
            plt.ylabel('Predicted Values')
            key_as_list = key.split('|')
            title = [key_as_list[idx * 3] + ', ' + key_as_list[(idx + 1) * 3 - 1] + ', ' + key_as_list[(idx + 1) * 3 - 2] for idx in
                     range(math.floor(len(key_as_list) / 3))]
            if len(key_as_list)%3 != 0:
                title.append(key_as_list[-2]+', '+key_as_list[-1])
            plt.title(algo_name + '\n' + '\n'.join(title))
            plt.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor)



    if visualize:
        plt.figure()
        train_data = [x['spearman']['rho'] for x in train_res.values()]
        plt.scatter(list(range(len(train_data))), train_data, label='Train', linewidth=0.3)
        test_data = [x['spearman']['rho'] for x in test_res.values()]
        plt.scatter(list(range(len(test_data))), test_data, label='Test', linewidth=0.3)
        plt.legend()
        plt.title(algo_name + ' Spearman' + r'$\rho$ vs params')
        plt.xlabel('sample #')
        plt.ylabel(r'$\rho$ value')
        plt.show()

    return train_res, test_res


def calc_results_and_plot(y_train_values, y_train_predicted_values, y_test_values, y_test_predicted_values , algo_name, visualize=False, title=None, show=True):

    train_res = {'spearman': use_spearmanr(y_train_values, y_train_predicted_values), 'pearson': use_pearsonr(y_train_values, y_train_predicted_values), 'mse': mean_squared_error(y_train_values, y_train_predicted_values) }
    test_res = {'spearman': use_spearmanr(y_test_values, y_test_predicted_values), 'pearson': use_pearsonr(y_test_values, y_test_predicted_values), 'mse': mean_squared_error(y_test_values, y_test_predicted_values) }
    spearman_train_data = train_res['spearman']
    pearson_train_data = train_res['pearson']
    mse_train_data = train_res['mse']

    spearman_test_data = test_res['spearman']
    pearson_test_data = test_res['pearson']
    mse_test_data = test_res['mse']

    train_label = 'Train\nSpearman - rho: ' + str(round(spearman_train_data['rho'], 2)) + ' pval: ' + str(
        round(spearman_train_data['pvalue'], 10)) + '\nPearson  - rho: ' + str(
        round(pearson_train_data['rho'], 2)) + ' pval: ' + str(
        round(pearson_train_data['pvalue'], 10)) + '\nmse: ' + str(round(mse_train_data,2))
    test_label = 'Test\nSpearman - rho: ' + str(round(spearman_test_data['rho'], 2)) + ' pval: ' + str(
        round(spearman_test_data['pvalue'], 10)) + '\nPearson  - rho: ' + str(
        round(pearson_test_data['rho'], 2)) + ' pval: ' + str(
        round(pearson_test_data['pvalue'], 10)) + '\nmse: ' + str(round(mse_test_data,2))

    if visualize:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.scatter(y_train_values, y_train_predicted_values, label=train_label)

        ax.scatter(y_test_values,y_test_predicted_values, label=test_label)
        plt.xlabel('Real Values')
        plt.ylabel('Predicted Values')
        title = '' if title is None else title
        plt.title(algo_name + '\n' + title)
        # ax.axis('equal')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)


        if show:
            plt.show()

    return train_res, test_res




if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    algo_name = 'xgboost'
    y_train_dict = pickle.load(open(os.path.join(SCRIPT_DIR, algo_name + "_train_data.p"), "rb"))
    y_test_dict = pickle.load(open(os.path.join(SCRIPT_DIR, algo_name + "_test_data.p"), "rb"))
    calc_results(y_train_dict, y_test_dict, algo_name)
