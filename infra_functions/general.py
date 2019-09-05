import random

import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.manifold import TSNE

# from sc


def apply_tsne(data, n_components=15, visualize=False):
    tsne = TSNE(n_components=n_components, method='exact')
    tsne.fit(data)
    data_components = tsne.fit_transform(data)
    str_to_print =''
    # str_to_print = str("Explained variance per component: \n" +
    #       '\n'.join(['Component ' + str(i) + ': ' +
    #                  str(component) + ', Accumalative variance: ' + str(accu_var) for accu_var, (i, component) in zip(pca.explained_variance_ratio_.cumsum(), enumerate(tsne.explained_variance_ratio_))]))
    #
    # str_to_print += str("\nTotal explained variance: " + str(tsne.explained_variance_ratio_.sum()))
    #
    # print(str_to_print)
    # if visualize:
    #     plt.figure()
    #     plt.plot(tsne.explained_variance_ratio_.cumsum())
    #     plt.bar(np.arange(0,n_components), height=tsne.explained_variance_ratio_)
    #     plt.title(f'PCA - Explained variance using {n_components} components: {tsne.explained_variance_ratio_.sum()}')
    #     plt.xlabel('PCA #')
    #     plt.xticks(list(range(0,n_components)), list(range(1,n_components+1)))
    #
    #     plt.ylabel('Explained Variance')
    #     plt.show()
    return pd.DataFrame(data_components).set_index(data.index), tsne, str_to_print


def apply_pca(data, n_components=15, visualize=False):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_components = pca.fit_transform(data)

    str_to_print = str("Explained variance per component: \n" +
          '\n'.join(['Component ' + str(i) + ': ' +
                     str(component) + ', Accumalative variance: ' + str(accu_var) for accu_var, (i, component) in zip(pca.explained_variance_ratio_.cumsum(), enumerate(pca.explained_variance_ratio_))]))

    str_to_print += str("\nTotal explained variance: " + str(pca.explained_variance_ratio_.sum()))

    print(str_to_print)
    if visualize:
        plt.figure()
        plt.plot(pca.explained_variance_ratio_.cumsum())
        plt.bar(np.arange(0,n_components), height=pca.explained_variance_ratio_)
        plt.title(f'PCA - Explained variance using {n_components} components: {pca.explained_variance_ratio_.sum()}')
        plt.xlabel('PCA #')
        plt.xticks(list(range(0,n_components)), list(range(1,n_components+1)))

        plt.ylabel('Explained Variance')
        plt.show()
    return pd.DataFrame(data_components).set_index(data.index), pca, str_to_print

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def draw_horizontal_bar_chart(data, names=None, title=None, ylabel=None, xlabel=None, use_pos_neg_colors=True, left_padding=0.4):
    fig, ax = plt.subplots()
    y_pos = np.arange(len(data))
    if use_pos_neg_colors:
        coeff_color = ['blue' if x else 'red' for x in data >= 0]
    else:
        coeff_color = ['blue' for x in data >= 0]
    ax.barh(y_pos, data, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    plt.title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.subplots_adjust(left=left_padding)
    # set_size(5, 5, ax)
    plt.show()

def convert_pca_back_orig(pca_components, w, original_names=None, visualize=False, title='Bacteria Coeff', ylabel='Bacteria', xlabel='Coeff Value'):
    coeff = np.dot(w, pca_components)
    if original_names is None:
        object_to_return = pd.DataFrame({'Coefficients': coeff})
    else:
        object_to_return = pd.DataFrame(
        {'Taxonome': original_names,
         'Coefficients': coeff
         })

    if visualize:
        draw_horizontal_bar_chart(coeff, original_names, title, ylabel, xlabel)
    return object_to_return

def use_spearmanr(x,y, axis=None):
    if x is None or y is None:
        print('Got None')
    rho, pvalue = spearmanr(x, y ,axis)
    return {'rho': rho, 'pvalue': pvalue}

def use_pearsonr(x,y):
    rho, pvalue = pearsonr(x, y)
    return {'rho': rho, 'pvalue': pvalue}


def roc_auc(y_test, y_score, verbose=False, visualize=False, graph_title='ROC Curve'):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC = {roc_auc}')
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr)
        plt.title(f'{graph_title}\nroc={roc_auc}')
        plt.xlabel('fpr')
        plt.ylabel('tpr')
    return fpr, tpr, thresholds, roc_auc


def draw_rhos_calculation_figure(id_to_binary_tag_map, preproccessed_data, title, num_of_mixtures=10, ids_list=None, save_folder=None, percentile=1):
    # calc ro for x=all samples values for each bacteria and y=all samples tags
    bacterias = []
    features_by_bacteria = []
    if ids_list:
        y = [id_to_binary_tag_map[id] for id in ids_list]
        X = preproccessed_data.loc[ids_list]

    else:
        y = list(id_to_binary_tag_map.values())
        X = preproccessed_data.loc[list(id_to_binary_tag_map.keys())]

    real_rhos = []
    real_pvalues = []
    mixed_y_list = []
    mixed_y_sum = [0 for i in y]

    for num in range(num_of_mixtures):  # run a couple time to avoid accidental results
        mixed_y = y.copy()
        random.shuffle(mixed_y)
        mixed_y_list.append(mixed_y)
        #for i, item in enumerate(mixed_y):
         #   mixed_y_sum[i] += item


    mixed_rhos = []
    mixed_pvalues = []
    for i, item in enumerate(X.iteritems()):
        bacterias.append(item[0])
        f = item[1]
        features_by_bacteria.append(f)

        rho, pvalue = spearmanr(f, y, axis=None)
        real_rhos.append(rho)
        real_pvalues.append(pvalue)

        for mix_y in mixed_y_list:
            rho_, pvalue_ = spearmanr(f, mix_y, axis=None)
            mixed_rhos.append(rho_)
            mixed_pvalues.append(pvalue_)



    # we want to take those who are located on the sides of most (center 98%) of the mixed tags entries
    # there for the bound isn't fixed, and is dependent on the distribution of the mixed tags
    real_min_rho = min(real_rhos)
    real_max_rho = max(real_rhos)
    mix_min_rho = min(mixed_rhos)
    mix_max_rho = max(mixed_rhos)

    real_rho_range = real_max_rho - real_min_rho
    mix_rho_range = mix_max_rho - mix_min_rho

    # old fixed bound approach - 20% range for each side from which we take the real bacterias (highest and lowest)
    # lower_bound = real_min_rho + (real_rho_range * 0.2)
    # upper_bound = real_max_rho - (real_rho_range * 0.2)

    # new method - all the items out of the mix range + 1% from the edge of the mix
    # lower_bound = mix_min_rho + (0.01 * mix_rho_range)
    # upper_bound = mix_max_rho - (0.01 * mix_rho_range)
    upper_bound = np.percentile(mixed_rhos, 100-percentile)
    lower_bound = np.percentile(mixed_rhos, percentile)

    significant_bacteria_and_rhos = []
    for i, bact in enumerate(bacterias):
        if real_rhos[i] < lower_bound or real_rhos[i] > upper_bound:  # significant
            significant_bacteria_and_rhos.append([bact, real_rhos[i]])

    significant_bacteria_and_rhos.sort(key=lambda s: s[1])
    if save_folder:
        with open(save_folder + "/significant_bacteria_" + title + ".csv", "w") as file:
            for s in significant_bacteria_and_rhos:
                file.write(str(s[1]) + "," + str(s[0]) + "\n")

    # draw the distribution of real rhos vs. mixed rhos
    # old plots
    [count, bins] = np.histogram(mixed_rhos, 50)
    plt.bar(bins[:-1], count/10, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="mixed tags", color="#d95f0e")
    [count, bins2] = np.histogram(real_rhos, 50)
    plt.bar(bins2[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.8, label="real tags", color="#43a2ca")
    # plt.hist(real_rhos, rwidth=0.6, bins=50, label="real tags", color="#43a2ca" )
    # plt.hist(mixed_rhos, rwidth=0.9, bins=50, alpha=0.5, label="mixed tags", color="#d95f0e")
    plt.title("Real tags vs. Mixed tags at " + title.replace("_", " "))
    plt.xlabel('Rho value')
    plt.ylabel('Number of bacteria')
    plt.legend()
    # print("Real tags_vs_Mixed_tags_at_" + title + "_combined.png")
    plt.show()
    if save_folder:
        plt.savefig(save_folder + "/Real tags_vs_Mixed_tags_at_" + title + "_combined.png")
    # plt.close()

    """
    combined_mixed_rhos = np.array(mixed_rhos)
    combined_mixed_rhos = combined_mixed_rhos.reshape(len(real_rhos), num_of_mixtures)
    combined_mixed_rhos = [sum(l) for l in combined_mixed_rhos]
    combined_rhos = np.array([[real_rhos[i], combined_mixed_rhos[i]] for i in range(len(real_rhos))])
    plt.hist(combined_rhos, bins=50, rwidth=0.8, histtype='bar', label=["real tags", "mixed tags"], color=["#43a2ca", "#d95f0e"])
    plt.title("Real tags vs. Mixed tags at " + title.replace("_", " "))
    plt.xlabel('Rho value')
    plt.ylabel('Number of bacteria')
    plt.legend()
    # print("Real_tags_vs_Mixed_tags_at_" + title + ".png")
    # plt.show()
    if save_folder:
        plt.savefig(save_folder + "/Real_tags_vs_Mixed_tags_at_" + title + ".png")
    plt.close()
    """

    # positive negative figures
    bacterias = [s[0] for s in significant_bacteria_and_rhos]
    real_rhos = [s[1] for s in significant_bacteria_and_rhos]
    # extract the last meaningful name - long multi level names to the lowest level definition
    short_bacterias_names = []
    for f in bacterias:
        i = 1
        while len(f.split(";")[-i]) < 5:  # meaningless name
            i += 1
        short_bacterias_names.append(f.split(";")[-i])

    left_padding = 0.4
    fig, ax = plt.subplots()
    y_pos = np.arange(len(short_bacterias_names))
    #coeff_color = ['green' if x else 'red' for x in real_rhos >= 0]
    coeff_color = []
    for x in real_rhos:
        if x >= 0:
            coeff_color.append('green')
        else:
            coeff_color.append('red')
    # coeff_color = ['blue' for x in data >= 0]
    ax.barh(y_pos, real_rhos, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_bacterias_names)
    plt.yticks(fontsize=10)
    plt.title(title)
    #ax.set_ylabel(ylabel)
    ax.set_xlabel("Coeff value")
    fig.subplots_adjust(left=left_padding)
    # set_size(5, 5, ax)
    # print("pos_neg_correlation_at_" + title + ".png")
    plt.show()
    if save_folder:
        plt.savefig(save_folder + "/pos_neg_correlation_at_" + title + ".png")
    # plt.close()

