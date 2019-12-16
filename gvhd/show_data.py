from infra_functions.general import use_spearmanr, use_pearsonr
import pickle
import os
import matplotlib.pyplot as plt
import math

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
        if visualize:
            spearman_train_data = train_res[key]['spearman']
            pearson_train_data = train_res[key]['pearson']

            plt.subplot(n_rows, n_cols, subplot_idx + 1)
            plt.scatter(train['y_train_values'], train['y_train_predicted_values'],
                        label='Train\nSpearman - rho: ' + str(round(spearman_train_data['rho'], 2)) + ' pval: ' + str(
                            round(spearman_train_data['pvalue'], 10)) + '\nPearson  - rho: ' + str(round(pearson_train_data['rho'], 2)) + ' pval: ' + str(
                            round(pearson_train_data['pvalue'], 10)))

            spearman_test_data = test_res[key]['spearman']
            pearson_test_data = test_res[key]['pearson']
            plt.scatter(test['y_test_values'], test['y_test_predicted_values'],
                        label='Test\nSpearman - rho: ' + str(round(spearman_test_data['rho'], 2)) + ' pval: ' + str(
                            round(spearman_test_data['pvalue'], 10)) + '\nPearson  - rho: ' + str(round(pearson_test_data['rho'], 2)) + ' pval: ' + str(
                            round(pearson_test_data['pvalue'], 10))
                        )
            plt.xlabel('Real Values')
            plt.ylabel('Predicted Values')
            key_as_list = key.split('|')
            title = [key_as_list[idx * 3] + ', ' + key_as_list[(idx + 1) * 3 - 1] + ', ' + key_as_list[(idx + 1) * 3 - 2] for idx in
                     range(math.floor(len(key_as_list) / 3))]
            if len(key_as_list)%3 != 0:
                title.append(key_as_list[-2]+', '+key_as_list[-1])
            plt.title(algo_name + '\n' + '\n'.join(title))
            plt.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor)

    plt.figure()
    train_data = [x['spearman']['rho'] for x in train_res.values()]
    plt.scatter(list(range(len(train_data))), train_data, label='Train', linewidth=0.3)
    test_data = [x['spearman']['rho'] for x in test_res.values()]
    plt.scatter(list(range(len(test_data))), test_data, label='Test', linewidth=0.3)
    plt.legend()
    plt.title(algo_name + ' Spearman' + r'$\rho$ vs params.json')
    plt.xlabel('sample #')
    plt.ylabel(r'$\rho$ value')
    plt.show()
    return train_res, test_res

if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    algo_name = 'xgboost'
    y_train_dict = pickle.load(open(os.path.join(SCRIPT_DIR, algo_name + "_train_data.p"), "rb"))
    y_test_dict = pickle.load(open(os.path.join(SCRIPT_DIR, algo_name + "_test_data.p"), "rb"))
    calc_results(y_train_dict, y_test_dict, algo_name)
