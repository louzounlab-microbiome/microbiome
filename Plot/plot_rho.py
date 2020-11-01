import pickle
import random
from os.path import join
from scipy.stats import spearmanr
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pop_idx(idx, objects_to_remove_idx_from):
    idx.reverse()
    for obj in objects_to_remove_idx_from:
        for i in idx:
            obj.pop(i)
    return objects_to_remove_idx_from


def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df


def draw_rhos_calculation_figure(id_to_binary_tag_map, preproccessed_data, title, taxnomy_level, num_of_mixtures=10,
                                 ids_list=None, save_folder=None):
    import matplotlib.pyplot as plt

    # calc ro for x=all samples values for each bacteria and y=all samples tags
    features_by_bacteria = []
    if ids_list:
        ids_list = [i for i in ids_list if i in preproccessed_data.index]
        X = preproccessed_data.loc[ids_list]
        y = [id_to_binary_tag_map[id] for id in ids_list]

    else:
        x_y = [[preproccessed_data.loc[key], val] for key, val in id_to_binary_tag_map.items()]
        X = pd.DataFrame([tag[0] for tag in x_y])
        y = [tag[1] for tag in x_y]

    # remove samples with nan as their tag
    not_nan_idxs = [i for i, y_ in enumerate(y) if str(y_) != "nan"]
    y = [y_ for i, y_ in enumerate(y) if i in not_nan_idxs]
    X = X.iloc[not_nan_idxs]



    mixed_y_list = []
    for num in range(num_of_mixtures):  # run a couple time to avoid accidental results
        mixed_y = y.copy()
        random.shuffle(mixed_y)
        mixed_y_list.append(mixed_y)

    bacterias = X.columns
    real_rhos = []
    real_pvalues = []
    used_bacterias = []
    mixed_rhos = []
    mixed_pvalues = []

    bacterias_to_dump = []
    for i, bact in enumerate(bacterias):
        f = X[bact]
        num_of_different_values = set(f)
        if len(num_of_different_values) < 2:
            bacterias_to_dump.append(bact)
        else:
            features_by_bacteria.append(f)
            used_bacterias.append(bact)

            rho, pvalue = spearmanr(f, y, axis=None)
            if str(rho) == "nan":
                print(bact)
            real_rhos.append(rho)
            real_pvalues.append(pvalue)

            for mix_y in mixed_y_list:
                rho_, pvalue_ = spearmanr(f, mix_y, axis=None)
                mixed_rhos.append(rho_)
                mixed_pvalues.append(pvalue_)

    print("number of bacterias to dump: " + str(len(bacterias_to_dump)))
    print("percent of bacterias to dump: " + str(len(bacterias_to_dump)/len(bacterias) * 100) + "%")

    # we want to take those who are located on the sides of most (center 98%) of the mixed tags entries
    # there for the bound isn't fixed, and is dependent on the distribution of the mixed tags
    real_min_rho = min(real_rhos)
    real_max_rho = max(real_rhos)
    mix_min_rho = min(mixed_rhos)
    mix_max_rho = max(mixed_rhos)

    real_rho_range = real_max_rho - real_min_rho
    mix_rho_range = mix_max_rho - mix_min_rho

    # new method - all the items out of the mix range + 1% from the edge of the mix
    upper_bound = np.percentile(mixed_rhos, 99)
    lower_bound = np.percentile(mixed_rhos, 1)

    significant_bacteria_and_rhos = []
    for i, bact in enumerate(used_bacterias):
        if real_rhos[i] < lower_bound or real_rhos[i] > upper_bound:  # significant
            significant_bacteria_and_rhos.append([bact, real_rhos[i]])

    significant_bacteria_and_rhos.sort(key=lambda s: s[1])
    if save_folder:
        with open(join(save_folder, "significant_bacteria_" + title + "_taxnomy_level_" + str(taxnomy_level)
                                    + "_.csv"), "w") as file:
            file.write("rho,bact\n")
            for s in significant_bacteria_and_rhos:
                file.write(str(s[1]) + "," + str(s[0]) + "\n")
                # דפנההה מה קורה?? איך עובר היום?
    # draw the distribution of real rhos vs. mixed rhos
    # old plots
    [count, bins] = np.histogram(mixed_rhos, 50)
    # divide by 'num_of_mixtures' fo avoid high number of occurrences due to multiple runs for each mixture
    plt.bar(bins[:-1], count/num_of_mixtures, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="mixed tags",
            color="#d95f0e")
    [count, bins2] = np.histogram(real_rhos, 50)
    plt.bar(bins2[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.8, label="real tags", color="#43a2ca")
    # plt.hist(real_rhos, rwidth=0.6, bins=50, label="real tags", color="#43a2ca" )
    # plt.hist(mixed_rhos, rwidth=0.9, bins=50, alpha=0.5, label="mixed tags", color="#d95f0e")
    plt.title("Real tags vs. Mixed tags at " + title.replace("_", " "))
    plt.xlabel('Rho value')
    plt.ylabel('Number of bacteria')
    plt.legend()
    # print("Real tags_vs_Mixed_tags_at_" + title + "_combined.png")
    # plt.show()
    if save_folder:
        plt.savefig(join(save_folder, "Real_tags_vs_Mixed_tags_at_" + title.replace(" ", "_")
                         + "_taxnomy_level_" + str(taxnomy_level) + ".svg"), bbox_inches='tight', format='svg')
    plt.close()

    # positive negative figures
    bacterias = [s[0] for s in significant_bacteria_and_rhos]
    real_rhos = [s[1] for s in significant_bacteria_and_rhos]
    # extract the last meaningful name - long multi level names to the lowest level definition

    short_bacterias_names = []
    for f in bacterias:
        i = 1
        while len(f.split(";")[-i]) < 5 or f.split(";")[-i] == 'Unassigned':  # meaningless name
            i += 1
            if i > len(f.split(";")):
                i -= 1
                break
        short_bacterias_names.append(f.split(";")[-i])
    # remove "k_bacteria" and "Unassigned" samples - irrelevant
    k_bact_idx = []
    for i, bact in enumerate(short_bacterias_names):
        if bact == 'k__Bacteria' or bact == 'Unassigned':
            k_bact_idx.append(i)

    if k_bact_idx:
        [short_bacterias_names, real_rhos, bacterias] = pop_idx(k_bact_idx, [short_bacterias_names, real_rhos, bacterias])

    left_padding = 0.4
    fig, ax = plt.subplots()
    y_pos = np.arange(len(bacterias))
    coeff_color = []
    for x in real_rhos:
        if x >= 0:
            coeff_color.append('green')
        else:
            coeff_color.append('red')
    ax.barh(y_pos, real_rhos, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_bacterias_names)
    plt.yticks(fontsize=10)
    plt.title(title.replace("_", " "))
    ax.set_xlabel("Coeff value")
    fig.subplots_adjust(left=left_padding)
    if save_folder:
        plt.savefig(join(save_folder, "pos_neg_correlation_at_" + title.replace(" ", "_")
                         + "_taxnomy_level_" + str(taxnomy_level) + ".svg"), bbox_inches='tight', format='svg')
    plt.close()


def draw_X_y_rhos_calculation_figure(X, y, title, taxnomy_level,
                                            num_of_mixtures=10, save_folder="rhos"):
    features_by_bacteria, mixed_y_list = [], []
    # run a couple of times to avoid accidental results
    for num in range(num_of_mixtures):
        mixed_y = y.copy()
        random.shuffle(mixed_y)
        mixed_y_list.append(mixed_y)

    bacterias = X.columns
    real_rhos, real_pvalues, used_bacterias, mixed_rhos, mixed_pvalues, bacterias_to_dump = [], [], [], [], [], []

    for i, bact in enumerate(bacterias):
        f = X[bact]
        num_of_different_values = set(f)
        if len(num_of_different_values) < 2:
            bacterias_to_dump.append(bact)
        else:
            features_by_bacteria.append(f)
            used_bacterias.append(bact)

            rho, pvalue = spearmanr(f, y, axis=None)
            if str(rho) == "nan":
                print(bact)
            real_rhos.append(rho)
            real_pvalues.append(pvalue)

            for mix_y in mixed_y_list:
                rho_, pvalue_ = spearmanr(f, mix_y, axis=None)
                mixed_rhos.append(rho_)
                mixed_pvalues.append(pvalue_)

    print("number of bacterias to dump: " + str(len(bacterias_to_dump)))
    print("percent of bacterias to dump: " + str(len(bacterias_to_dump)/len(bacterias) * 100) + "%")

    # we want to take those who are located on the sides of most (center 98%) of the mixed tags entries
    # there for the bound isn't fixed, and is dependent on the distribution of the mixed tags
    upper_bound = np.percentile(mixed_rhos, 99)
    lower_bound = np.percentile(mixed_rhos, 1)

    significant_bacteria_and_rhos = []
    for i, bact in enumerate(used_bacterias):
        if real_rhos[i] < lower_bound or real_rhos[i] > upper_bound:  # significant
            significant_bacteria_and_rhos.append([bact, real_rhos[i]])

    significant_bacteria_and_rhos.sort(key=lambda s: s[1])
    if save_folder:
        with open(join(save_folder, "significant_bacteria_" + title + "_taxnomy_level_" + str(taxnomy_level)
                                    + "_.csv"), "w") as file:
            for s in significant_bacteria_and_rhos:
                file.write(str(s[1]) + "," + str(s[0]) + "\n")

    # draw the distribution of real rhos vs. mixed rhos
    import matplotlib.pyplot as plt
    plt.figure()
    [count, bins] = np.histogram(mixed_rhos, 50)
    # divide by 'num_of_mixtures' fo avoid high number of occurrences due to multiple runs for each mixture
    plt.bar(bins[:-1], count/num_of_mixtures, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="mixed tags",
            color="#d95f0e")
    [count, bins2] = np.histogram(real_rhos, 50)
    plt.bar(bins2[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.8, label="real tags", color="#43a2ca")
    plt.title("Real tags vs. Mixed tags at " + title.replace("_", " "))
    plt.xlabel('Rho value')
    plt.ylabel('Number of bacteria')
    plt.legend()
    if save_folder:
        plt.savefig(join(save_folder, "Real_tags_vs_Mixed_tags_at_" + title.replace(" ", "_")
                         + "_taxnomy_level_" + str(taxnomy_level) + ".svg"), bbox_inches='tight', format='svg')
    plt.close()

    # positive negative figures
    bacterias = [s[0] for s in significant_bacteria_and_rhos]
    real_rhos = [s[1] for s in significant_bacteria_and_rhos]
    # extract the last meaningful name - long multi level names to the lowest level definition

    short_bacterias_names = []
    for f in bacterias:
        i = 1
        while len(f.split(";")[-i]) < 5 or f.split(";")[-i] == 'Unassigned':  # meaningless name
            i += 1
            if i > len(f.split(";")):
                i -= 1
                break
        short_bacterias_names.append(f.split(";")[-i])
    # remove "k_bacteria" and "Unassigned" samples - irrelevant
    k_bact_idx = []
    for i, bact in enumerate(short_bacterias_names):
        if bact == 'k__Bacteria' or bact == 'Unassigned':
            k_bact_idx.append(i)

    if k_bact_idx:
        [short_bacterias_names, real_rhos, bacterias] = pop_idx(k_bact_idx, [short_bacterias_names, real_rhos, bacterias])

    left_padding = 0.4
    fig, ax = plt.subplots()
    y_pos = np.arange(len(bacterias))
    coeff_color = []
    for x in real_rhos:
        if x >= 0:
            coeff_color.append('green')
        else:
            coeff_color.append('red')
    ax.barh(y_pos, real_rhos, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_bacterias_names)
    plt.yticks(fontsize=10)
    plt.title(title.replace("_", " "))
    ax.set_xlabel("Coeff value")
    fig.subplots_adjust(left=left_padding)
    if save_folder:
        plt.savefig(join(save_folder, "pos_neg_correlation_at_" + title.replace(" ", "_")
                         + "_taxnomy_level_" + str(taxnomy_level) + ".svg"), bbox_inches='tight', format='svg')
    plt.close()




def draw_dynamics_rhos_calculation_figure(preproccessed_data, title, tri_to_tri, num_of_mixtures=10, save_folder=None,
                                          up_per=99, low_per=1, type="", health=""):
    # calc ro for x=all samples values for each bacteria and y=all samples tags
    samples = preproccessed_data.index
    bacterial = preproccessed_data.columns

    real_X = preproccessed_data
    df_len = real_X.shape[1]

    # calculate correlation from real X to real X nad from real X to mix X

    mixed_X_list = []
    mixed_rhos = []
    mixed_pvalues = []

    for num in range(num_of_mixtures):  # run a couple time to avoid accidental results
        mixed_X = shuffle(preproccessed_data)
        mixed_X_list.append(mixed_X)

    rho, pvalue = spearmanr(real_X.T, real_X.T, axis=1)
    real_rho = rho[:df_len, df_len:]
    real_pvalue = pvalue[:df_len, df_len:]

    for mix_X in mixed_X_list:
        rho_, pvalue_ = spearmanr(real_X.T, mix_X.T, axis=1)
        mixed_rhos.append(rho_[:df_len, df_len:])
        mixed_pvalues.append(pvalue_[:df_len, df_len:])

    flat_real_rhos = real_rho.flatten()
    real_rho[[np.arange(real_rho.shape[0])] * 2] = 0

    flat_mixed_rhos = []
    for m in mixed_rhos:
        flat_mixed_rhos.append(m.flatten())

    # we want to take those who are located on the sides of most (center 98%) of the mixed tags entries
    # there for the bound isn't fixed, and is dependent on the distribution of the mixed tags
    real_min_rho = min(flat_real_rhos)
    real_max_rho = max(flat_real_rhos)

    mix_min_rho = min([min(f) for f in flat_mixed_rhos])
    mix_max_rho = max([max(f) for f in flat_mixed_rhos])

    # new method - all the items out of the mix range + 1% from the edge of the mix
    mixed_rhos = np.hstack([f for f in flat_mixed_rhos])
    upper_bound = np.percentile(mixed_rhos, up_per)
    lower_bound = np.percentile(mixed_rhos, low_per)

    upper_range = real_max_rho - upper_bound
    lower_range = lower_bound - real_min_rho

    out_of_bound_idx_and_corr_score = []
    for i in range(len(real_rho)):
        for j in range(len(real_rho)):
            rho = real_rho[i][j]
            if rho >= upper_bound:
                rho_score = (rho - upper_bound) / upper_range
                out_of_bound_idx_and_corr_score.append([bacterial[i], bacterial[j], rho_score, rho])

            elif real_rho[i][j] <= lower_bound:
                rho_score = (rho - lower_bound) / lower_range
                out_of_bound_idx_and_corr_score.append([bacterial[i], bacterial[j], rho_score, rho])
    out_of_bound_idx_and_corr_score.sort(key=lambda s: s[3])
    pickle.dump(out_of_bound_idx_and_corr_score, open(os.path.join(save_folder, "out_of_bound_corr_idx_" + tri_to_tri +
                                                                   '_' + type + '_' + health + ".pkl"), "wb"))
    with open(join(save_folder, "out_of_bound_corr_idx_" + tri_to_tri + '_' + type + '_' + health + ".csv"), "w") as file:
        file.write("bacteria_1,bacteria_2,rho_score,rho\n")
        for entry in out_of_bound_idx_and_corr_score:
           file.write(entry[0] + "," + entry[1] + "," + str(round(entry[2], 3)) + "," + str(round(entry[3], 3)) + "\n")

    # positive negative figures
    bacterias_0 = [s[0] for s in out_of_bound_idx_and_corr_score]
    bacterias_1 = [s[1] for s in out_of_bound_idx_and_corr_score]

    real_rhos = [s[3] for s in out_of_bound_idx_and_corr_score]
    # extract the last meaningful name - long multi level names to the lowest level definition
    short_bacterias_0_names = []
    for f in bacterias_0:
        i = 1
        while len(f.split(";")[-i]) < 5:  # meaningless name
            i += 1
        short_bacterias_0_names.append(f.split(";")[-i])

    short_bacterias_1_names = []
    for f in bacterias_1:
        i = 1
        while len(f.split(";")[-i]) < 5:  # meaningless name
            i += 1
        short_bacterias_1_names.append(f.split(";")[-i])

    short_bacterias_names = [short_bacterias_0_names[i].strip() + "-" + short_bacterias_1_names[i].strip()
                             for i in range(len(short_bacterias_0_names))]
    if False:
        left_padding = 0.4
        fig, ax = plt.subplots()
        y_pos = np.arange(len(short_bacterias_names))
        coeff_color = []
        for x in real_rhos:
            if x >= 0:
                coeff_color.append('green')
            else:
                coeff_color.append('red')
        ax.barh(y_pos, real_rhos, color=coeff_color)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_bacterias_names)
        plt.yticks(fontsize=10)
        plt.title(title.replace("_", " ") + " positive and negetive correlation")
        ax.set_xlabel("Coeff value")
        fig.subplots_adjust(left=left_padding)
        if save_folder:
            plt.savefig(join(save_folder, "pos_neg_correlation_at_" + title + ".png"))
        plt.close()


def draw_component_rhos_calculation_figure(bact_df, tag_df, task_name="prognosis", save_folder=False):
    bact_df = bact_df.drop("taxonomy")

    otu_ids = bact_df.index
    tag_ids = tag_df.index
    mutual_ids = [id for id in otu_ids if id in tag_ids]
    bact_df = bact_df.loc[mutual_ids]
    tag_df = tag_df.loc[mutual_ids]
    y = tag_df["Tag"]


    # calc pca
    n = min(100, min(len(bact_df.index), len(bact_df.columns)))
    pca = PCA(n_components=n)
    pca.fit(bact_df)
    data_components = pca.fit_transform(bact_df)
    components_df = pd.DataFrame(data_components)

    real_rhos = []
    for com in components_df.columns:
        f = components_df[com]
        rho, pvalue = spearmanr(f, y, axis=None)
        real_rhos.append(rho)

    # draw rhos
    left_padding = 0.4
    fig, ax = plt.subplots()
    x_pos = np.arange(len(components_df.columns))
    coeff_color = []
    for x in real_rhos:
        if x >= 0:
            coeff_color.append('green')
        else:
            coeff_color.append('red')
    ax.bar(x_pos, real_rhos, color=coeff_color)
    ax.set_ylim(-1, 1)
    ax.set_xticks(x_pos)
    plt.title("Correlation between each component and the label\n" + task_name + " task")
    ax.set_xlabel("spearman correlation")
    # plt.show()
    fig.subplots_adjust(left=left_padding)
    if save_folder:
        title = "Correlation between each component and the label" + task_name + "task"
        plt.savefig(join(save_folder, title.replace(" ", "_")
                          + ".svg"), bbox_inches='tight', format='svg')
    plt.close()


if __name__ == "__main__":
    components_df = pd.read_csv("Allergy_OTU.csv")
    components_df = components_df.set_index("ID")
    tag_df = pd.read_csv("Allergy_Tag.csv")
    tag_df = tag_df.set_index("ID")
    task_name = "healthy vs. sick"
    save_folder = "corr"
    draw_component_rhos_calculation_figure(components_df, tag_df, task_name=task_name, save_folder=save_folder)